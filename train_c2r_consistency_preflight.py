#!/usr/bin/env python3
"""Run a parameter-matched, label-free C2R mechanism preflight.

This is a causal mechanism screen, not the final parameterized architecture.
Every trained variant is a standard ReLU SAE with exactly the same parameter
count and starts from the same checkpoint. The only treatment is a
cross-sample consistency loss based on decoder-neighbor geometry. The wrong
control preserves the activation-norm multiset but misaligns it with decoder
pairs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn


SOURCE_PARAMETER_COUNT = 402_721_792
EXPOSED_FEATURE_COUNT = 65_536


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_activation(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(
        dim=-1, keepdim=True
    ).clamp_min(eps)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"b_pre", "encoder.weight", "encoder.bias", "decoder.weight"}
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"Checkpoint {path} is missing {missing}")
    return {key: raw[key].float() for key in required}


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def data_manifest(paths: list[Path]) -> dict[str, object]:
    digest = hashlib.sha256()
    entries = []
    for path in paths:
        info = path.stat()
        entry = {
            "name": path.name,
            "size": int(info.st_size),
            "mtime_ns": int(info.st_mtime_ns),
            "mode": stat.S_IMODE(info.st_mode),
        }
        entries.append(entry)
        digest.update(
            f"{entry['name']}\0{entry['size']}\0{entry['mtime_ns']}\0"
            f"{entry['mode']}\n".encode()
        )
    return {
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "metadata_sha256": digest.hexdigest(),
        "entries": entries,
    }


def git_metadata() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    try:
        return {
            "branch": subprocess.check_output(
                ["git", "-C", str(root), "branch", "--show-current"], text=True
            ).strip(),
            "commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"branch": "unknown", "commit": "unknown"}


def parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def gradient_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def iter_batches(
    paths: list[Path], batch_tokens: int, seed: int
) -> Iterator[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    file_rng = random.Random(seed)
    order = list(range(len(paths)))
    while True:
        file_rng.shuffle(order)
        for file_index in order:
            values = torch.load(paths[file_index], map_location="cpu", weights_only=True)
            if not torch.is_tensor(values):
                raise TypeError(f"Expected tensor in {paths[file_index]}")
            values = values.reshape(-1, values.shape[-1])
            permutation = torch.randperm(values.shape[0], generator=generator)
            for start in range(0, values.shape[0], batch_tokens):
                indices = permutation[start : start + batch_tokens]
                if indices.numel() == batch_tokens:
                    yield values[indices]


class ReLUSAE(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.b_pre = nn.Parameter(state["b_pre"].clone())
        self.encoder_weight = nn.Parameter(state["encoder.weight"].clone())
        self.encoder_bias = nn.Parameter(state["encoder.bias"].clone())
        self.decoder_weight = nn.Parameter(state["decoder.weight"].clone())
        self.normalize_decoder()

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder_weight.div_(
                self.decoder_weight.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
            )

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.linear(
            x_norm - self.b_pre,
            self.encoder_weight,
            self.encoder_bias,
        )
        features = F.relu(hidden)
        reconstruction = F.linear(features, self.decoder_weight) + self.b_pre
        return features, reconstruction

    def encoder_parameters(self) -> list[nn.Parameter]:
        return [self.b_pre, self.encoder_weight, self.encoder_bias]

    def decoder_parameters(self) -> list[nn.Parameter]:
        return [self.decoder_weight]

    def export_state(self) -> dict[str, torch.Tensor]:
        return {
            "b_pre": self.b_pre.detach().cpu().float(),
            "encoder.weight": self.encoder_weight.detach().cpu().float(),
            "encoder.bias": self.encoder_bias.detach().cpu().float(),
            "decoder.weight": self.decoder_weight.detach().cpu().float(),
        }


@dataclass(frozen=True)
class C2RDiagnostics:
    selected_count: int
    neighbor_cosine_mean: float
    neighbor_cosine_max: float
    activation_norm_mean: float
    wrong_shift: int
    wrong_fixed_pair_count: int
    raw_loss: float


def compute_c2r_loss(
    features: torch.Tensor,
    decoder_weight: torch.Tensor,
    subset_size: int,
    wrong_alignment: bool,
    wrong_shift: int,
) -> tuple[torch.Tensor, C2RDiagnostics]:
    """Compute active-subset C2R with an exact activation-marginal control."""
    feature_norms = features.float().norm(p=2, dim=0) / math.sqrt(
        max(int(features.shape[0]), 1)
    )
    selected_count = min(int(subset_size), int(feature_norms.numel()))
    if selected_count < 2:
        raise ValueError("C2R requires at least two selected features")
    selected = torch.topk(feature_norms.detach(), k=selected_count).indices
    selected_decoder = decoder_weight[:, selected].T.float()
    normalized_decoder = F.normalize(selected_decoder, p=2, dim=1)
    with torch.no_grad():
        similarities = normalized_decoder.detach() @ normalized_decoder.detach().T
        similarities.fill_diagonal_(-2.0)
        neighbor = similarities.argmax(dim=1)

    neighbor_cosine = (
        normalized_decoder * normalized_decoder[neighbor]
    ).sum(dim=1).clamp_min(0.0)
    selected_norms = feature_norms[selected]
    fixed_pair_count = 0
    applied_shift = 0
    if wrong_alignment:
        applied_shift = int(wrong_shift) % selected_count
        if applied_shift == 0:
            applied_shift = 1
        mapping = torch.roll(
            torch.arange(selected_count, device=selected.device),
            shifts=applied_shift,
        )
        fixed_pair_count = int((mapping == torch.arange(
            selected_count, device=selected.device
        )).sum().item())
        selected_norms = selected_norms[mapping]

    pair_norm = selected_norms + selected_norms[neighbor]
    raw_loss = (neighbor_cosine.square() * pair_norm).mean()
    diagnostics = C2RDiagnostics(
        selected_count=selected_count,
        neighbor_cosine_mean=float(neighbor_cosine.detach().mean().item()),
        neighbor_cosine_max=float(neighbor_cosine.detach().max().item()),
        activation_norm_mean=float(selected_norms.detach().mean().item()),
        wrong_shift=applied_shift,
        wrong_fixed_pair_count=fixed_pair_count,
        raw_loss=float(raw_loss.detach().item()),
    )
    return raw_loss, diagnostics


def validation_metrics(
    model: ReLUSAE,
    paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    batches = iter_batches(paths, args.batch_tokens, args.seed + 1_000_003)
    squared_error = 0.0
    total_variance = 0.0
    active = 0.0
    token_count = 0
    model.eval()
    with torch.inference_mode():
        for _ in range(args.validation_batches):
            x = normalize_activation(next(batches), args.normalize_eps).to(device)
            features, reconstruction = model(x)
            squared_error += float((reconstruction - x).float().square().sum().item())
            centered = x.float() - x.float().mean(dim=0, keepdim=True)
            total_variance += float(centered.square().sum().item())
            active += float((features > 0).sum().item())
            token_count += int(x.shape[0])
    model.train()
    elements = token_count * int(model.b_pre.numel())
    mse = squared_error / max(elements, 1)
    return {
        "validation_mse": mse,
        "validation_explained_variance": 1.0
        - squared_error / max(total_variance, 1.0e-12),
        "validation_active_per_token": active / max(token_count, 1),
        "validation_batches": float(args.validation_batches),
    }


def parameter_max_delta(
    model: ReLUSAE, initial: dict[str, torch.Tensor]
) -> dict[str, float]:
    pairs = {
        "b_pre": model.b_pre,
        "encoder.weight": model.encoder_weight,
        "encoder.bias": model.encoder_bias,
        "decoder.weight": model.decoder_weight,
    }
    return {
        key: float(
            (parameter.detach().cpu().float() - initial[key]).abs().max().item()
        )
        for key, parameter in pairs.items()
    }


def train_variant(
    variant_key: str,
    state: dict[str, torch.Tensor],
    train_paths: list[Path],
    validation_paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    set_seed(args.seed)
    model = ReLUSAE(state).to(device)
    count = parameter_count(model)
    if count != SOURCE_PARAMETER_COUNT:
        raise AssertionError(f"Unexpected parameter count: {count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.2
    )
    batches = iter_batches(train_paths, args.batch_tokens, args.seed)
    observed = torch.zeros(EXPOSED_FEATURE_COUNT, dtype=torch.bool, device=device)
    history: list[dict[str, float]] = []
    any_encoder_c2r_gradient = False
    any_decoder_c2r_gradient = False
    all_wrong_fixed_pairs = 0
    started = time.time()

    for step in range(1, args.steps + 1):
        x = normalize_activation(next(batches), args.normalize_eps).to(device)
        features, reconstruction = model(x)
        reconstruction_loss = F.mse_loss(reconstruction.float(), x.float())
        l1 = features.float().mean()
        sae_loss = reconstruction_loss + args.l1_coeff * l1
        c2r_raw = torch.zeros((), device=device)
        c2r_scaled = torch.zeros((), device=device)
        diagnostics = C2RDiagnostics(0, 0.0, 0.0, 0.0, 0, 0, 0.0)
        use_c2r = variant_key != "relu_finetune" and step % args.c2r_interval == 0
        if use_c2r:
            wrong = variant_key == "wrong_alignment_c2r"
            shift = 1 + (step // args.c2r_interval) % max(args.c2r_subset - 1, 1)
            c2r_raw, diagnostics = compute_c2r_loss(
                features,
                model.decoder_weight,
                args.c2r_subset,
                wrong,
                shift,
            )
            if not torch.isfinite(c2r_raw) or float(c2r_raw.detach().item()) <= 0:
                raise FloatingPointError(
                    f"{variant_key} invalid C2R loss at step {step}: {c2r_raw}"
                )
            scale = (
                args.c2r_fraction
                * args.c2r_interval
                * sae_loss.detach()
                / c2r_raw.detach().clamp_min(1.0e-12)
            )
            c2r_scaled = scale * c2r_raw
            all_wrong_fixed_pairs += diagnostics.wrong_fixed_pair_count

        loss = sae_loss + c2r_scaled
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{variant_key} non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        encoder_grad = gradient_norm(model.encoder_parameters())
        decoder_grad = gradient_norm(model.decoder_parameters())
        if use_c2r:
            any_encoder_c2r_gradient |= encoder_grad > 0
            any_decoder_c2r_gradient |= decoder_grad > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.normalize_decoder()
        scheduler.step()
        observed |= (features.detach() > 0).any(dim=0)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            variance = x.float().var(unbiased=False).clamp_min(1.0e-12)
            explained_variance = 1.0 - reconstruction_loss.detach() / variance
            row = {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "reconstruction_loss": float(reconstruction_loss.detach().item()),
                "l1_mean": float(l1.detach().item()),
                "explained_variance": float(explained_variance.item()),
                "active_per_token": float((features.detach() > 0).sum(dim=1).float().mean().item()),
                "dead_ratio_so_far": float((~observed).float().mean().item()),
                "encoder_grad_norm": encoder_grad,
                "decoder_grad_norm": decoder_grad,
                "c2r_raw": float(c2r_raw.detach().item()),
                "c2r_scaled": float(c2r_scaled.detach().item()),
                "c2r_selected_count": float(diagnostics.selected_count),
                "neighbor_cosine_mean": diagnostics.neighbor_cosine_mean,
                "neighbor_cosine_max": diagnostics.neighbor_cosine_max,
                "wrong_shift": float(diagnostics.wrong_shift),
                "wrong_fixed_pair_count": float(diagnostics.wrong_fixed_pair_count),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps({"variant": variant_key, **row}), flush=True)

    validation = validation_metrics(model, validation_paths, args, device)
    deltas = parameter_max_delta(model, state)
    checkpoint = args.output_dir / f"trained_sae-{variant_key}.pt"
    checkpoint_hash = None
    if not args.no_save_checkpoints:
        torch.save(model.export_state(), checkpoint)
        checkpoint_hash = file_sha256(checkpoint)
    result = {
        "variant_key": variant_key,
        "parameter_count": count,
        "exposed_feature_count": EXPOSED_FEATURE_COUNT,
        "steps": args.steps,
        "history": history,
        "validation": validation,
        "parameter_max_delta": deltas,
        "any_encoder_c2r_gradient": any_encoder_c2r_gradient,
        "any_decoder_c2r_gradient": any_decoder_c2r_gradient,
        "wrong_fixed_pair_count_total": all_wrong_fixed_pairs,
        "final_dead_ratio": float((~observed).float().mean().item()),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint) if not args.no_save_checkpoints else None,
        "checkpoint_sha256": checkpoint_hash,
    }
    del optimizer, scheduler, model
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("/root/saint/trained_sae-relu-l22.pt"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/root/autodl-tmp/activation_outputs_batched"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-coeff", type=float, default=1.0e-6)
    parser.add_argument("--c2r-interval", type=int, default=5)
    parser.add_argument("--c2r-subset", type=int, default=2048)
    parser.add_argument("--c2r-fraction", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.steps < 1 or args.batch_tokens < 2:
        raise ValueError("steps and batch_tokens must be positive")
    if not (0 < args.validation_fraction < 0.5):
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    if not (0 < args.c2r_fraction <= 0.1):
        raise ValueError("c2r_fraction must lie in (0, 0.1]")
    if args.c2r_interval < 1:
        raise ValueError("c2r_interval must be positive")

    torch.set_float32_matmul_precision("high")
    paths = sorted(args.data_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No activation files under {args.data_dir}")
    before = data_manifest(paths)
    split = max(1, int(len(paths) * (1.0 - args.validation_fraction)))
    train_paths = paths[:split]
    validation_paths = paths[split:]
    if not validation_paths:
        raise ValueError("No validation files after split")
    state = load_state(args.base_checkpoint)
    base_hash = file_sha256(args.base_checkpoint)
    base_count = sum(int(tensor.numel()) for tensor in state.values())
    if base_count != SOURCE_PARAMETER_COUNT:
        raise AssertionError(f"Base parameter count {base_count} is unexpected")

    device = torch.device(args.device)
    variants = {}
    for variant_key in (
        "relu_finetune",
        "wrong_alignment_c2r",
        "true_c2r",
    ):
        variants[variant_key] = train_variant(
            variant_key,
            state,
            train_paths,
            validation_paths,
            args,
            device,
        )

    after = data_manifest(paths)
    data_unchanged = before == after
    if not data_unchanged:
        raise RuntimeError("Activation cache metadata changed during training")
    targets = [
        {
            "label": "Frozen ReLU reference",
            "kind": "relu",
            "layer": 22,
            "checkpoint": str(args.base_checkpoint),
        },
        {
            "label": "Matched ReLU finetune",
            "kind": "relu",
            "layer": 22,
            "checkpoint": variants["relu_finetune"]["checkpoint"],
        },
        {
            "label": "Wrong-alignment C2R control",
            "kind": "relu",
            "layer": 22,
            "checkpoint": variants["wrong_alignment_c2r"]["checkpoint"],
        },
        {
            "label": "True C2R preflight",
            "kind": "relu",
            "layer": 22,
            "checkpoint": variants["true_c2r"]["checkpoint"],
        },
    ]
    targets_path = args.output_dir / "targets-c2r-preflight.json"
    targets_path.write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")
    summary = {
        "experiment": "parameter-matched fixed C2R causal preflight",
        "status": "mechanism-preflight-not-final-architecture",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git": git_metadata(),
        "source_paper": {
            "arxiv": "2606.30609v1",
            "official_code_commit": "21aa150cee335ddef072b5389c73d746a9a4d504",
        },
        "base_checkpoint_sha256": base_hash,
        "source_parameter_count": SOURCE_PARAMETER_COUNT,
        "parameter_count_each": SOURCE_PARAMETER_COUNT,
        "exposed_feature_count_each": EXPOSED_FEATURE_COUNT,
        "data_manifest_before": before,
        "data_manifest_after": after,
        "data_unchanged": data_unchanged,
        "train_file_count": len(train_paths),
        "validation_file_count": len(validation_paths),
        "targets_json": str(targets_path),
        "variants": variants,
        "fairness": {
            "same_initial_checkpoint": True,
            "same_data_files": True,
            "same_data_order": True,
            "same_optimizer": True,
            "same_training_steps": True,
            "same_parameter_count": True,
            "same_exposed_feature_count": True,
            "wrong_control_preserves_activation_norm_multiset": True,
            "uses_saebench_labels_for_training": False,
            "uses_eval_split_for_training": False,
            "uses_one_vs_rest_targets_for_training": False,
            "uses_mean_diff_selection_for_training": False,
            "uses_test_feedback_for_training": False,
        },
    }
    summary_path = args.output_dir / "train-summary-c2r-preflight.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "training_complete",
        "summary": str(summary_path),
        "targets": str(targets_path),
        "data_unchanged": data_unchanged,
    }), flush=True)


if __name__ == "__main__":
    main()
