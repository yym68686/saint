#!/usr/bin/env python3
"""Train an exact-parameter V396 control and joint Cascaded Concept SAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import stat
import subprocess
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn


V396_KEYS = (
    "b_pre",
    "encoder.weight",
    "encoder.bias",
    "decoder.weight",
    "v396.raw_beta",
    "v396.log_gain",
    "v396.max_beta",
    "v396.max_log_gain",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_activation(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=-1, keepdim=True)) / (
        x.std(dim=-1, keepdim=True) + eps
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_list_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        metadata = path.stat()
        digest.update(f"{path.name}\0{metadata.st_size}\0".encode())
    return digest.hexdigest()


def git_metadata() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "branch": subprocess.check_output(
            ["git", "-C", str(root), "branch", "--show-current"], text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def gradient_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def iter_batches(
    paths: list[Path],
    batch_tokens: int,
    seed: int,
) -> Iterator[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rng = random.Random(seed)
    order = list(range(len(paths)))
    while True:
        rng.shuffle(order)
        for file_index in order:
            values = torch.load(
                paths[file_index], map_location="cpu", weights_only=True
            )
            values = values.reshape(-1, values.shape[-1])
            permutation = torch.randperm(len(values), generator=generator)
            for start in range(0, len(values) - batch_tokens + 1, batch_tokens):
                yield values.index_select(
                    0, permutation[start : start + batch_tokens]
                )


def load_v396(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    missing = sorted(set(V396_KEYS) - set(raw))
    if missing:
        raise KeyError(f"V396 checkpoint missing {missing}")
    state = {key: raw[key].float() for key in V396_KEYS}
    n_latents, d_model = state["encoder.weight"].shape
    if state["decoder.weight"].shape != (d_model, n_latents):
        raise ValueError("Unexpected V396 encoder/decoder shapes")
    if state["v396.raw_beta"].shape != (n_latents,):
        raise ValueError("V396 raw_beta must be feature-wise")
    if state["v396.log_gain"].shape != (n_latents,):
        raise ValueError("V396 log_gain must be feature-wise")
    return state


def compand(
    preactivation: torch.Tensor,
    raw_beta: torch.Tensor,
    log_gain: torch.Tensor,
    max_beta: float,
    max_log_gain: float,
) -> torch.Tensor:
    positive = torch.relu(preactivation).float()
    beta = F.softplus(raw_beta.float()).clamp(1.0e-4, max_beta)
    gain = log_gain.float().clamp(-max_log_gain, max_log_gain).exp()
    return (
        torch.log1p(positive * beta.unsqueeze(0))
        / torch.log1p(beta).unsqueeze(0)
        * gain.unsqueeze(0)
    )


class PartitionedV396(nn.Module):
    """V396 finetune control split into optimizer-matched parameter groups."""

    def __init__(self, state: dict[str, torch.Tensor], high_features: int) -> None:
        super().__init__()
        n_total, d_model = state["encoder.weight"].shape
        if high_features != d_model:
            raise ValueError("Exact mapping requires high_features == d_model")
        self.n_total = int(n_total)
        self.n_high = int(high_features)
        self.n_low = self.n_total - self.n_high
        self.max_beta = float(state["v396.max_beta"].item())
        self.max_log_gain = float(state["v396.max_log_gain"].item())
        split = self.n_low
        self.b_pre = nn.Parameter(state["b_pre"].clone())
        self.low_encoder = nn.Parameter(state["encoder.weight"][:split].clone())
        self.low_bias = nn.Parameter(state["encoder.bias"][:split].clone())
        self.low_decoder = nn.Parameter(state["decoder.weight"][:, :split].clone())
        self.low_beta = nn.Parameter(state["v396.raw_beta"][:split].clone())
        self.low_gain = nn.Parameter(state["v396.log_gain"][:split].clone())
        self.high_encoder = nn.Parameter(state["encoder.weight"][split:].clone())
        self.high_bias = nn.Parameter(state["encoder.bias"][split:].clone())
        self.high_decoder = nn.Parameter(state["decoder.weight"][:, split:].clone())
        self.high_beta = nn.Parameter(state["v396.raw_beta"][split:].clone())
        self.high_gain = nn.Parameter(state["v396.log_gain"][split:].clone())
        self.register_buffer("initial_beta", state["v396.raw_beta"].clone())
        self.register_buffer("initial_gain", state["v396.log_gain"].clone())
        self.normalize_decoder()

    def low_parameters(self) -> list[nn.Parameter]:
        return [
            self.b_pre,
            self.low_encoder,
            self.low_bias,
            self.low_decoder,
            self.low_beta,
            self.low_gain,
        ]

    def high_parameters(self) -> list[nn.Parameter]:
        return [
            self.high_encoder,
            self.high_bias,
            self.high_decoder,
            self.high_beta,
            self.high_gain,
        ]

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.low_decoder.div_(
                self.low_decoder.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
            )
            self.high_decoder.div_(
                self.high_decoder.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
            )

    def project_decoder_gradients(self) -> None:
        for decoder in (self.low_decoder, self.high_decoder):
            if decoder.grad is None:
                continue
            with torch.no_grad():
                projection = (decoder * decoder.grad).sum(dim=0, keepdim=True)
                decoder.grad.sub_(projection * decoder)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        centered = x - self.b_pre
        low_h = F.linear(centered, self.low_encoder, self.low_bias)
        high_h = F.linear(centered, self.high_encoder, self.high_bias)
        low_z = compand(
            low_h,
            self.low_beta,
            self.low_gain,
            self.max_beta,
            self.max_log_gain,
        )
        high_z = compand(
            high_h,
            self.high_beta,
            self.high_gain,
            self.max_beta,
            self.max_log_gain,
        )
        reconstruction = (
            F.linear(low_z, self.low_decoder)
            + F.linear(high_z, self.high_decoder)
            + self.b_pre
        )
        return {
            "reconstruction": reconstruction,
            "exported": torch.cat([low_z, high_z], dim=1),
            "low_z": low_z,
            "high_z": high_z,
            "hierarchy_reconstruction": x.new_zeros((0, x.shape[1])),
            "hierarchy_target": x.new_zeros((0, x.shape[1])),
            "hierarchy_code": x.new_zeros((0, self.n_high)),
            "active_atom_count": x.new_zeros(()),
        }

    def regularization(self, args: argparse.Namespace) -> torch.Tensor:
        current_beta = torch.cat([self.low_beta, self.high_beta])
        current_gain = torch.cat([self.low_gain, self.high_gain])
        return (
            args.beta_anchor_coeff
            * (current_beta - self.initial_beta).square().mean()
            + args.gain_anchor_coeff
            * (current_gain - self.initial_gain).square().mean()
        )

    def export_state(self) -> dict[str, torch.Tensor]:
        return {
            "b_pre": self.b_pre.detach().cpu(),
            "encoder.weight": torch.cat(
                [self.low_encoder, self.high_encoder], dim=0
            ).detach().cpu(),
            "encoder.bias": torch.cat(
                [self.low_bias, self.high_bias], dim=0
            ).detach().cpu(),
            "decoder.weight": torch.cat(
                [self.low_decoder, self.high_decoder], dim=1
            ).detach().cpu(),
            "v396.raw_beta": torch.cat(
                [self.low_beta, self.high_beta]
            ).detach().cpu(),
            "v396.log_gain": torch.cat(
                [self.low_gain, self.high_gain]
            ).detach().cpu(),
            "v396.max_beta": torch.tensor(self.max_beta),
            "v396.max_log_gain": torch.tensor(self.max_log_gain),
        }


def deranged_parent_assignment(parent: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(parent, stable=True)
    sorted_parent = parent.index_select(0, order)
    counts = torch.bincount(parent)
    shift = int(counts.max().item())
    if 2 * shift > len(parent):
        raise ValueError("Cannot construct count-preserving parent derangement")
    rotated = torch.roll(sorted_parent, shifts=shift)
    wrong = torch.empty_like(parent)
    wrong[order] = rotated
    if torch.any(wrong == parent):
        raise RuntimeError("Parent derangement retained fixed assignments")
    if not torch.equal(
        torch.bincount(wrong, minlength=len(counts)),
        counts,
    ):
        raise RuntimeError("Parent derangement changed cluster sizes")
    return wrong


class CascadedConceptSAE(PartitionedV396):
    """Joint Level-2 SAE over Level-1 decoder atoms with exact parameter parity."""

    def __init__(
        self,
        state: dict[str, torch.Tensor],
        high_features: int,
        active_atom_cap: int,
    ) -> None:
        super().__init__(state, high_features)
        self.active_atom_cap = int(active_atom_cap)
        self.module_probe_width = min(256, self.n_high)
        self.module_initial = [
            self.high_encoder[: self.module_probe_width].detach().cpu().clone(),
            self.high_bias.detach().cpu().clone(),
            self.high_decoder[:, : self.module_probe_width].detach().cpu().clone(),
        ]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        centered = x - self.b_pre
        low_h = F.linear(centered, self.low_encoder, self.low_bias)
        low_z = compand(
            low_h,
            self.low_beta,
            self.low_gain,
            self.max_beta,
            self.max_log_gain,
        )
        reconstruction = F.linear(low_z, self.low_decoder) + self.b_pre

        activation_mass = low_z.detach().sum(dim=0)
        active = torch.nonzero(activation_mass > 0, as_tuple=False).flatten()
        if len(active) > self.active_atom_cap:
            active = torch.topk(
                activation_mass, k=self.active_atom_cap, sorted=True
            ).indices
        atoms = self.low_decoder.index_select(1, active).T
        hierarchy_h = F.linear(atoms - self.high_bias, self.high_encoder)
        hierarchy_z = compand(
            hierarchy_h,
            self.high_beta,
            self.high_gain,
            self.max_beta,
            self.max_log_gain,
        )
        hierarchy_reconstruction = (
            F.linear(hierarchy_z, self.high_decoder) + self.high_bias
        )
        return {
            "reconstruction": reconstruction,
            "exported": low_z,
            "low_z": low_z,
            "high_z": hierarchy_z,
            "hierarchy_reconstruction": hierarchy_reconstruction,
            "hierarchy_target": atoms,
            "hierarchy_code": hierarchy_z,
            "active_atom_count": torch.tensor(
                float(len(active)), device=x.device
            ),
        }

    def module_gradient_norm(self) -> float:
        return gradient_norm(self.high_parameters())

    def module_parameter_delta(self) -> float:
        width = self.module_probe_width
        current = [
            self.high_encoder[:width].detach().cpu(),
            self.high_bias.detach().cpu(),
            self.high_decoder[:, :width].detach().cpu(),
        ]
        return max(
            float((value - initial).abs().max().item())
            for value, initial in zip(current, self.module_initial, strict=True)
        )

    @torch.inference_mode()
    def build_hierarchy(self, chunk_size: int) -> dict[str, torch.Tensor]:
        parents = []
        strengths = []
        for start in range(0, self.n_low, chunk_size):
            end = min(self.n_low, start + chunk_size)
            atoms = self.low_decoder[:, start:end].T
            hidden = F.linear(atoms - self.high_bias, self.high_encoder)
            code = compand(
                hidden,
                self.high_beta,
                self.high_gain,
                self.max_beta,
                self.max_log_gain,
            )
            value, parent = code.max(dim=1)
            zero = value <= 0
            if zero.any():
                raw_value, raw_parent = hidden[zero].max(dim=1)
                value[zero] = raw_value
                parent[zero] = raw_parent
            parents.append(parent.cpu())
            strengths.append(value.cpu())
        parent = torch.cat(parents)
        strength = torch.cat(strengths)
        counts = torch.bincount(parent, minlength=self.n_high)
        wrong = deranged_parent_assignment(parent)
        return {
            "parent": parent,
            "wrong_parent": wrong,
            "strength": strength,
            "cluster_count": counts,
            "cluster_scale": counts.float().clamp_min(1).rsqrt(),
        }

    def export_state(self, hierarchy_chunk_size: int = 128) -> dict[str, torch.Tensor]:
        hierarchy = self.build_hierarchy(hierarchy_chunk_size)
        return {
            "cascaded.kind": torch.tensor(1),
            "cascaded.n_total": torch.tensor(self.n_total),
            "cascaded.n_low": torch.tensor(self.n_low),
            "cascaded.n_high": torch.tensor(self.n_high),
            "cascaded.parent": hierarchy["parent"],
            "cascaded.wrong_parent": hierarchy["wrong_parent"],
            "cascaded.parent_strength": hierarchy["strength"],
            "cascaded.cluster_count": hierarchy["cluster_count"],
            "cascaded.cluster_scale": hierarchy["cluster_scale"],
            "b_pre": self.b_pre.detach().cpu(),
            "level1.encoder.weight": self.low_encoder.detach().cpu(),
            "level1.encoder.bias": self.low_bias.detach().cpu(),
            "level1.decoder.weight": self.low_decoder.detach().cpu(),
            "level1.raw_beta": self.low_beta.detach().cpu(),
            "level1.log_gain": self.low_gain.detach().cpu(),
            "level2.center": self.high_bias.detach().cpu(),
            "level2.encoder.weight": self.high_encoder.detach().cpu(),
            "level2.decoder.weight": self.high_decoder.detach().cpu(),
            "level2.raw_beta": self.high_beta.detach().cpu(),
            "level2.log_gain": self.high_gain.detach().cpu(),
            "v396.max_beta": torch.tensor(self.max_beta),
            "v396.max_log_gain": torch.tensor(self.max_log_gain),
        }


def train_variant(
    label: str,
    model: PartitionedV396,
    paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.low_parameters(), "lr": args.low_lr},
            {"params": model.high_parameters(), "lr": args.high_lr},
        ],
        weight_decay=args.weight_decay,
    )
    batches = iter_batches(paths, args.batch_tokens, args.seed)
    logs = []
    started = time.time()
    for step in range(1, args.steps + 1):
        x = normalize_activation(next(batches), args.normalize_eps).to(device)
        output = model(x)
        reconstruction = F.mse_loss(output["reconstruction"].float(), x.float())
        low_l1 = output["low_z"].float().mean()
        hierarchy_reconstruction = reconstruction.new_zeros(())
        hierarchy_l1 = reconstruction.new_zeros(())
        if output["hierarchy_code"].numel():
            hierarchy_reconstruction = F.mse_loss(
                output["hierarchy_reconstruction"].float(),
                output["hierarchy_target"].float(),
            )
            hierarchy_l1 = output["hierarchy_code"].float().mean()
        anchor = model.regularization(args)
        loss = (
            reconstruction
            + args.l1_coeff * low_l1
            + args.hierarchy_weight
            * (
                hierarchy_reconstruction
                + args.hierarchy_l1_coeff * hierarchy_l1
            )
            + anchor
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{label} produced non-finite loss at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.project_decoder_gradients()
        low_grad = gradient_norm(model.low_parameters())
        high_grad = gradient_norm(model.high_parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.normalize_decoder()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "label": label,
                "step": step,
                "loss": float(loss.detach().item()),
                "reconstruction": float(reconstruction.detach().item()),
                "low_l1": float(low_l1.detach().item()),
                "hierarchy_reconstruction": float(
                    hierarchy_reconstruction.detach().item()
                ),
                "hierarchy_l1": float(hierarchy_l1.detach().item()),
                "anchor": float(anchor.detach().item()),
                "low_gradient_norm": low_grad,
                "high_gradient_norm": high_grad,
                "active_atom_count": float(
                    output["active_atom_count"].detach().item()
                ),
                "elapsed_seconds": time.time() - started,
            }
            print(json.dumps(row), flush=True)
            logs.append(row)
        del output, x, reconstruction, low_l1, hierarchy_reconstruction
        del hierarchy_l1, anchor, loss
    result: dict[str, object] = {
        "label": label,
        "parameter_count": parameter_count(model),
        "elapsed_seconds": time.time() - started,
        "logs": logs,
        "final_low_gradient_norm": logs[-1]["low_gradient_norm"],
        "final_high_gradient_norm": logs[-1]["high_gradient_norm"],
    }
    if isinstance(model, CascadedConceptSAE):
        result["module_parameter_delta"] = model.module_parameter_delta()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v396-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-tokens", type=int, default=64)
    parser.add_argument("--high-features", type=int, default=3072)
    parser.add_argument("--active-atom-cap", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-lr", type=float, default=1.0e-6)
    parser.add_argument("--high-lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-coeff", type=float, default=1.0e-6)
    parser.add_argument("--hierarchy-weight", type=float, default=1.0)
    parser.add_argument("--hierarchy-l1-coeff", type=float, default=1.0e-6)
    parser.add_argument("--beta-anchor-coeff", type=float, default=1.0e-3)
    parser.add_argument("--gain-anchor-coeff", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.v396_checkpoint = args.v396_checkpoint.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.data_dir.stat().st_mode & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    ):
        raise PermissionError(f"Activation cache must be read-only: {args.data_dir}")
    paths = sorted(args.data_dir.glob("*.pt"))
    if not paths or any(path.stat().st_mode & 0o222 for path in paths):
        raise PermissionError("Activation files are missing or writable")
    state = load_v396(args.v396_checkpoint)
    n_total, d_model = state["encoder.weight"].shape
    if args.high_features != d_model:
        raise ValueError(
            f"Exact parity fixes high_features to d_model={d_model}, got "
            f"{args.high_features}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preregistration = {
        "experiment": "exact-parameter joint Cascaded Concept SAE v1",
        "status": "registered-before-training-and-saebench-evaluation",
        "git": git_metadata(),
        "architecture_source": "Cascaded Sparse Autoencoders, arXiv:2606.16193v1",
        "n_total": n_total,
        "n_level1": n_total - args.high_features,
        "n_level2": args.high_features,
        "parameter_mapping": (
            "the final d_model Level-1 slots map one-for-one to the bias-free "
            "Level-2 encoder, decoder, and shared center"
        ),
        "steps": args.steps,
        "batch_tokens": args.batch_tokens,
        "same_stream_seed": args.seed,
        "active_atom_cap": args.active_atom_cap,
        "initial3": [
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "fancyzhx/ag_news",
        ],
        "gate": {
            "candidate_minus_v396_finetune": ">= 0.005",
            "candidate_minus_best_hierarchy_control": ">= 0.002",
            "minimum_dataset_delta_vs_v396": ">= -0.01",
            "minimum_same_dataset_v396_reference": "within 0.01 of 0.837543",
        },
        "controls": [
            "same-parameter V396 finetune",
            "same-checkpoint Level-1 only",
            "count-preserving zero-fixed-point wrong hierarchy",
        ],
        "uses_saebench_labels_for_training": False,
        "uses_class_names_for_training": False,
        "uses_eval_split_for_training": False,
        "uses_mean_diff_for_training": False,
        "uses_test_feedback_for_training": False,
    }
    preregistration_path = args.output_dir / "preregistration.json"
    if preregistration_path.exists():
        if json.loads(preregistration_path.read_text()) != preregistration:
            raise RuntimeError("Refusing to overwrite a different preregistration")
    else:
        preregistration_path.write_text(
            json.dumps(preregistration, indent=2) + "\n", encoding="utf-8"
        )

    device = torch.device(args.device)
    results: dict[str, object] = {}
    targets = []
    for key, model in (
        ("v396_finetune", PartitionedV396(state, args.high_features)),
        (
            "cascaded_concept",
            CascadedConceptSAE(
                state, args.high_features, args.active_atom_cap
            ),
        ),
    ):
        set_seed(args.seed)
        result = train_variant(key, model, paths, args, device)
        checkpoint = args.output_dir / f"trained_sae-{key}.pt"
        if isinstance(model, CascadedConceptSAE):
            exported = model.export_state()
            kind = "cascaded_concept"
        else:
            exported = model.export_state()
            kind = "v396_finetune"
        torch.save(exported, checkpoint)
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_sha256"] = sha256_file(checkpoint)
        results[key] = result
        target = {
            "label": key,
            "variant_key": key,
            "kind": kind,
            "layer": 22,
            "checkpoint": str(checkpoint),
            "trainable_parameters": result["parameter_count"],
        }
        targets.append(target)
        if isinstance(model, CascadedConceptSAE):
            for readout in ("wrong_hierarchy", "level1_only"):
                targets.append(
                    {
                        **target,
                        "label": f"{key}_{readout}",
                        "variant_key": f"{key}_{readout}",
                        "readout": readout,
                    }
                )
        model.to("cpu")
        del model
        torch.cuda.empty_cache()

    base_count = int(results["v396_finetune"]["parameter_count"])
    candidate_count = int(results["cascaded_concept"]["parameter_count"])
    if base_count != candidate_count:
        raise RuntimeError(
            f"Parameter mismatch: control={base_count}, candidate={candidate_count}"
        )
    if float(results["cascaded_concept"]["final_high_gradient_norm"]) <= 0:
        raise RuntimeError("Cascaded module received no gradient")
    targets_path = args.output_dir / "targets-cascaded-concept.json"
    targets_path.write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")
    summary = {
        "experiment": preregistration["experiment"],
        "git": git_metadata(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "v396_checkpoint_sha256": sha256_file(args.v396_checkpoint),
        "data_file_count": len(paths),
        "data_file_list_sha256": file_list_fingerprint(paths),
        "parameter_count_control": base_count,
        "parameter_count_candidate": candidate_count,
        "parameter_matched": True,
        "exported_feature_count_control": n_total,
        "exported_feature_count_candidate": n_total,
        "results": results,
        "targets": str(targets_path),
        "fairness": {
            "activation_cache_read_only": True,
            "same_initial_parameter_tensor_mapping": True,
            "same_data_order": True,
            "same_steps": True,
            "same_optimizer_and_group_learning_rates": True,
            "same_total_parameters": True,
            "same_exposed_feature_count": True,
            "joint_level2_gradient_to_level1_decoder": True,
            "uses_labels_or_test_feedback": False,
        },
    }
    summary_path = args.output_dir / "train-summary-cascaded-concept.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "training_complete", **summary}, indent=2))


if __name__ == "__main__":
    main()
