#!/usr/bin/env python3
"""Warm-start matched V396 SAEs with a next-token prediction head."""

from __future__ import annotations

import argparse
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


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def max_parameter_delta(
    parameters: list[nn.Parameter],
    initial: list[torch.Tensor],
) -> float:
    if len(parameters) != len(initial):
        raise ValueError("Parameter snapshot length mismatch")
    return max(
        (
            float((parameter.detach().cpu() - reference).abs().max().item())
            for parameter, reference in zip(parameters, initial, strict=True)
        ),
        default=0.0,
    )


@dataclass(frozen=True)
class TransitionBatch:
    source_activations: torch.Tensor
    next_token_ids: torch.Tensor
    sample_ids: torch.Tensor


class StructuredTransitionCache:
    def __init__(
        self,
        cache_dir: Path,
        layer: int,
        train_fraction: float,
        seed: int,
    ) -> None:
        self.cache_dir = cache_dir.resolve()
        self.layer = int(layer)
        self.train_fraction = float(train_fraction)
        self.seed = int(seed)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest["status"] != "complete":
            raise ValueError(f"Incomplete cache: {self.manifest['status']}")
        if self.layer not in map(int, self.manifest["configuration"]["layers"]):
            raise ValueError(f"Layer {self.layer} is absent from the cache")
        if self.cache_dir.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise PermissionError(f"Cache directory must be read-only: {self.cache_dir}")
        self.sample_count = int(self.manifest["summary"]["sample_count"])
        self.train_cutoff = int(self.sample_count * self.train_fraction)
        self.shards = list(self.manifest["shards"])

    def fingerprint(self) -> str:
        return sha256_file(self.manifest_path)

    def iter_batches(
        self,
        batch_tokens: int,
        max_transitions_per_sample: int,
    ) -> Iterator[TransitionBatch]:
        if batch_tokens <= 1:
            raise ValueError("batch_tokens must exceed one")
        if not 0 < max_transitions_per_sample <= batch_tokens // 2:
            raise ValueError(
                "max_transitions_per_sample must be positive and at most half "
                "the transition batch size"
            )
        epoch = 0
        while True:
            shard_order = list(range(len(self.shards)))
            py_rng = random.Random(self.seed + epoch * 1_000_003)
            py_rng.shuffle(shard_order)
            carry_x: list[torch.Tensor] = []
            carry_tokens: list[torch.Tensor] = []
            carry_samples: list[torch.Tensor] = []
            carry_count = 0
            for shard_position in shard_order:
                entry = self.shards[shard_position]
                meta = torch.load(
                    self.cache_dir / entry["meta"]["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                activations = torch.load(
                    self.cache_dir / entry["layers"][str(self.layer)]["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                sample_ids = meta["sample_ids"].to(torch.int64)
                selected = torch.nonzero(
                    sample_ids < self.train_cutoff,
                    as_tuple=False,
                ).flatten()
                generator = torch.Generator(device="cpu").manual_seed(
                    self.seed + epoch * 1_000_003 + shard_position
                )
                selected = selected[
                    torch.randperm(selected.numel(), generator=generator)
                ]
                offsets = meta["offsets"].to(torch.int64)
                lengths = meta["lengths"].to(torch.int64)
                token_ids = meta["token_ids"].to(torch.int64)
                for index_tensor in selected:
                    index = int(index_tensor.item())
                    length = int(lengths[index].item())
                    if length < 2:
                        continue
                    start = int(offsets[index].item())
                    positions = torch.arange(length - 1)
                    if positions.numel() > max_transitions_per_sample:
                        position_generator = torch.Generator(device="cpu").manual_seed(
                            self.seed
                            + epoch * 1_000_003
                            + shard_position * 10_007
                            + int(sample_ids[index].item())
                        )
                        positions = positions[
                            torch.randperm(
                                positions.numel(),
                                generator=position_generator,
                            )[:max_transitions_per_sample]
                        ].sort().values
                    sources = activations[start + positions]
                    targets = token_ids[index, positions + 1]
                    samples = torch.full(
                        (positions.numel(),),
                        int(sample_ids[index].item()),
                        dtype=torch.int64,
                    )
                    cursor = 0
                    while cursor < positions.numel():
                        take = min(
                            batch_tokens - carry_count,
                            positions.numel() - cursor,
                        )
                        carry_x.append(sources[cursor : cursor + take])
                        carry_tokens.append(targets[cursor : cursor + take])
                        carry_samples.append(samples[cursor : cursor + take])
                        carry_count += take
                        cursor += take
                        if carry_count == batch_tokens:
                            yield TransitionBatch(
                                source_activations=torch.cat(carry_x, dim=0),
                                next_token_ids=torch.cat(carry_tokens, dim=0),
                                sample_ids=torch.cat(carry_samples, dim=0),
                            )
                            carry_x = []
                            carry_tokens = []
                            carry_samples = []
                            carry_count = 0
            epoch += 1


def sample_separating_permutation(
    sample_ids: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    count = int(sample_ids.numel())
    if count < 2 or int(torch.unique(sample_ids).numel()) < 2:
        raise ValueError("Wrong-alignment control requires multiple samples")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shifts = torch.randperm(count - 1, generator=generator) + 1
    base = torch.arange(count)
    for shift_tensor in shifts:
        permutation = torch.roll(base, shifts=int(shift_tensor.item()))
        if not bool((sample_ids[permutation] == sample_ids).any()):
            return permutation
    for _ in range(1024):
        permutation = torch.randperm(count, generator=generator)
        if not bool((sample_ids[permutation] == sample_ids).any()):
            return permutation
    raise RuntimeError("Could not construct a sample-separating target permutation")


def load_v396_state(path: Path) -> dict[str, torch.Tensor | float]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "b_pre",
        "encoder.weight",
        "encoder.bias",
        "decoder.weight",
        "v396.raw_beta",
        "v396.log_gain",
        "v396.init_beta",
        "v396.max_beta",
        "v396.max_log_gain",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"Checkpoint {path} missing keys: {missing}")
    return {
        "b_pre": raw["b_pre"].float(),
        "encoder.weight": raw["encoder.weight"].float(),
        "encoder.bias": raw["encoder.bias"].float(),
        "decoder.weight": raw["decoder.weight"].float(),
        "raw_beta": raw["v396.raw_beta"].float(),
        "log_gain": raw["v396.log_gain"].float(),
        "init_beta": float(raw["v396.init_beta"].item()),
        "max_beta": float(raw["v396.max_beta"].item()),
        "max_log_gain": float(raw["v396.max_log_gain"].item()),
    }


def load_output_weight(path: Path) -> torch.Tensor:
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "output.weight" not in state:
        raise KeyError(f"{path} does not contain output.weight")
    output = state["output.weight"]
    if output.ndim != 2:
        raise ValueError(f"Unexpected output.weight shape: {tuple(output.shape)}")
    del state
    return output


class NextTokenPredictiveV396SAE(nn.Module):
    def __init__(
        self,
        base_state: dict[str, torch.Tensor | float],
        predictor_rank: int,
        predictor_seed: int,
    ) -> None:
        super().__init__()
        self.b_pre = nn.Parameter(base_state["b_pre"].clone())
        self.encoder_weight = nn.Parameter(base_state["encoder.weight"].clone())
        self.encoder_bias = nn.Parameter(base_state["encoder.bias"].clone())
        self.decoder_weight = nn.Parameter(base_state["decoder.weight"].clone())
        self.raw_beta = nn.Parameter(base_state["raw_beta"].clone())
        self.log_gain = nn.Parameter(base_state["log_gain"].clone())
        self.init_beta = float(base_state["init_beta"])
        self.max_beta = float(base_state["max_beta"])
        self.max_log_gain = float(base_state["max_log_gain"])
        self.predictor_rank = int(predictor_rank)
        d_model = int(self.b_pre.numel())
        generator = torch.Generator(device="cpu").manual_seed(predictor_seed)
        down = torch.empty((self.predictor_rank, d_model), dtype=torch.float32)
        up = torch.empty((d_model, self.predictor_rank), dtype=torch.float32)
        nn.init.xavier_uniform_(down, generator=generator)
        nn.init.xavier_uniform_(up, generator=generator)
        self.predictor_down_weight = nn.Parameter(down)
        self.predictor_down_bias = nn.Parameter(
            torch.zeros(self.predictor_rank, dtype=torch.float32)
        )
        self.predictor_up_weight = nn.Parameter(up)
        self.predictor_up_bias = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
        self.normalize_decoder()

    def trunk_parameters(self) -> list[nn.Parameter]:
        return [
            self.b_pre,
            self.encoder_weight,
            self.encoder_bias,
            self.decoder_weight,
            self.raw_beta,
            self.log_gain,
        ]

    def predictor_parameters(self) -> list[nn.Parameter]:
        return [
            self.predictor_down_weight,
            self.predictor_down_bias,
            self.predictor_up_weight,
            self.predictor_up_bias,
        ]

    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta.float()).clamp(1.0e-4, self.max_beta)

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder_weight.div_(
                self.decoder_weight.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
            )

    def forward(
        self,
        x_norm: torch.Tensor,
        detach_predictor_input: bool,
    ) -> dict[str, torch.Tensor]:
        centered = x_norm - self.b_pre
        h = F.linear(centered, self.encoder_weight, self.encoder_bias)
        u = torch.relu(h).float()
        beta = self.beta().to(u.device)
        z = torch.log1p(beta.unsqueeze(0) * u) / torch.log1p(beta).unsqueeze(0)
        gain = self.log_gain.clamp(
            -self.max_log_gain,
            self.max_log_gain,
        ).exp().to(z.device)
        z = z * gain.unsqueeze(0)
        recon = F.linear(z.to(self.decoder_weight.dtype), self.decoder_weight) + self.b_pre
        predictor_input = F.layer_norm(
            (recon - self.b_pre).float(),
            (recon.shape[-1],),
        )
        if detach_predictor_input:
            predictor_input = predictor_input.detach()
        hidden = F.silu(
            F.linear(
                predictor_input,
                self.predictor_down_weight,
                self.predictor_down_bias,
            )
        )
        prediction = F.linear(
            hidden,
            self.predictor_up_weight,
            self.predictor_up_bias,
        )
        prediction = F.normalize(prediction.float(), dim=-1)
        return {
            "h": h,
            "z": z,
            "recon": recon,
            "prediction": prediction,
        }

    def export_state(self) -> dict[str, torch.Tensor]:
        return {
            "b_pre": self.b_pre.detach().cpu(),
            "encoder.weight": self.encoder_weight.detach().cpu(),
            "encoder.bias": self.encoder_bias.detach().cpu(),
            "decoder.weight": self.decoder_weight.detach().cpu(),
            "causal.raw_beta": self.raw_beta.detach().cpu(),
            "causal.log_gain": self.log_gain.detach().cpu(),
            "causal.init_beta": torch.tensor(self.init_beta),
            "causal.max_beta": torch.tensor(self.max_beta),
            "causal.max_log_gain": torch.tensor(self.max_log_gain),
            "nexttoken.predictor_down_weight": self.predictor_down_weight.detach().cpu(),
            "nexttoken.predictor_down_bias": self.predictor_down_bias.detach().cpu(),
            "nexttoken.predictor_up_weight": self.predictor_up_weight.detach().cpu(),
            "nexttoken.predictor_up_bias": self.predictor_up_bias.detach().cpu(),
        }


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    detach_predictor_input: bool
    wrong_alignment: bool


VARIANTS = [
    Variant(
        key="detach_control",
        label="same-parameter detached predictive-head control",
        detach_predictor_input=True,
        wrong_alignment=False,
    ),
    Variant(
        key="wrong_alignment",
        label="same-parameter wrong-next-token predictive control",
        detach_predictor_input=False,
        wrong_alignment=True,
    ),
    Variant(
        key="candidate",
        label="next-token predictive V396 SAE",
        detach_predictor_input=False,
        wrong_alignment=False,
    ),
]


def train_variant(
    variant: Variant,
    model: NextTokenPredictiveV396SAE,
    cache: StructuredTransitionCache,
    output_weight: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.trunk_parameters(), "lr": args.trunk_lr},
            {"params": model.predictor_parameters(), "lr": args.predictor_lr},
        ],
        weight_decay=args.weight_decay,
    )
    trunk_initial = [p.detach().cpu().clone() for p in model.trunk_parameters()]
    predictor_initial = [p.detach().cpu().clone() for p in model.predictor_parameters()]
    seen_features = torch.zeros(
        model.encoder_weight.shape[0],
        dtype=torch.bool,
        device=device,
    )
    batches = cache.iter_batches(
        args.batch_tokens,
        args.max_transitions_per_sample,
    )
    logs: list[dict[str, float | str]] = []
    started = time.time()
    wrong_fixed_total = 0
    wrong_same_sample_total = 0
    for step in range(1, args.steps + 1):
        batch = next(batches)
        x = normalize_activation(
            batch.source_activations,
            args.normalize_eps,
        ).to(device)
        target = F.normalize(
            output_weight[batch.next_token_ids].float(),
            dim=-1,
        )
        if variant.wrong_alignment:
            permutation = sample_separating_permutation(
                batch.sample_ids,
                args.seed * 10_000_019 + step,
            )
            wrong_fixed_total += int(
                (permutation == torch.arange(permutation.numel())).sum().item()
            )
            wrong_same_sample_total += int(
                (batch.sample_ids[permutation] == batch.sample_ids).sum().item()
            )
            target = target[permutation]
        target = target.to(device, non_blocking=True)
        out = model(x, variant.detach_predictor_input)
        rec_loss = F.mse_loss(out["recon"].float(), x.float())
        prediction_loss = 1.0 - (
            out["prediction"] * target
        ).sum(dim=-1).mean()
        l1 = out["z"].float().mean()
        beta_anchor = (
            torch.log(model.beta()) - math.log(model.init_beta)
        ).square().mean()
        gain_l2 = model.log_gain.float().square().mean()
        loss = (
            rec_loss
            + args.prediction_loss_weight * prediction_loss
            + args.l1_coeff * l1
            + args.beta_anchor_coeff * beta_anchor
            + args.gain_l2_coeff * gain_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"{variant.key} produced NaN/Inf at step {step}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trunk_grad = grad_norm(model.trunk_parameters())
        predictor_grad = grad_norm(model.predictor_parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.normalize_decoder()
        with torch.no_grad():
            seen_features |= (out["z"] > 0).any(dim=0)
            residual = out["recon"].float() - x.float()
            ev = 1.0 - residual.square().sum() / x.float().square().sum()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row: dict[str, float | str] = {
                "variant": variant.key,
                "step": float(step),
                "loss": float(loss.detach().item()),
                "reconstruction_loss": float(rec_loss.detach().item()),
                "prediction_loss": float(prediction_loss.detach().item()),
                "l1": float(l1.detach().item()),
                "explained_variance": float(ev.item()),
                "active_features_per_token": float(
                    (out["z"].detach() > 0).float().sum(dim=1).mean().item()
                ),
                "dead_feature_ratio_seen": float((~seen_features).float().mean().item()),
                "trunk_grad_norm": trunk_grad,
                "predictor_grad_norm": predictor_grad,
                "trunk_lr": optimizer.param_groups[0]["lr"],
                "predictor_lr": optimizer.param_groups[1]["lr"],
                "elapsed_seconds": time.time() - started,
            }
            print(json.dumps(row), flush=True)
            logs.append(row)
        del x, target, out, rec_loss, prediction_loss, l1, beta_anchor, gain_l2, loss
    return {
        "logs": logs,
        "elapsed_seconds": time.time() - started,
        "parameter_count": parameter_count(model),
        "trunk_parameter_max_delta": max_parameter_delta(
            model.trunk_parameters(), trunk_initial
        ),
        "predictor_parameter_max_delta": max_parameter_delta(
            model.predictor_parameters(), predictor_initial
        ),
        "wrong_fixed_pair_count": wrong_fixed_total,
        "wrong_same_sample_pair_count": wrong_same_sample_total,
        "final_dead_feature_ratio_seen": float((~seen_features).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--model-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--predictor-rank", type=int, default=64)
    parser.add_argument("--predictor-seed", type=int, default=490_396)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--max-transitions-per-sample", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--trunk-lr", type=float, default=1.0e-6)
    parser.add_argument("--predictor-lr", type=float, default=1.0e-5)
    parser.add_argument("--prediction-loss-weight", type=float, default=0.1)
    parser.add_argument("--l1-coeff", type=float, default=1.0e-6)
    parser.add_argument("--beta-anchor-coeff", type=float, default=1.0e-3)
    parser.add_argument("--gain-l2-coeff", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    cache = StructuredTransitionCache(
        args.cache_dir,
        args.layer,
        args.train_fraction,
        args.seed,
    )
    base_state = load_v396_state(args.base_checkpoint)
    output_weight = load_output_weight(args.model_weights)
    vocab_size = int(output_weight.shape[0])
    d_model = int(output_weight.shape[1])
    if d_model != int(base_state["b_pre"].numel()):
        raise ValueError("Output directions and SAE dimensions differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets: list[dict[str, object]] = []
    results: dict[str, object] = {}
    parameter_counts: set[int] = set()

    for variant in VARIANTS:
        set_seed(args.seed)
        model = NextTokenPredictiveV396SAE(
            base_state,
            args.predictor_rank,
            args.predictor_seed,
        )
        result = train_variant(
            variant,
            model,
            cache,
            output_weight,
            args,
            device,
        )
        checkpoint = args.output_dir / f"trained_sae-{variant.key}-seed{args.seed}.pt"
        if args.save_checkpoints:
            torch.save(model.export_state(), checkpoint)
        count = int(result["parameter_count"])
        parameter_counts.add(count)
        if args.save_checkpoints:
            targets.append(
                {
                    "label": variant.label,
                    "kind": "v396_causal_learned_beta",
                    "layer": args.layer,
                    "checkpoint": str(checkpoint),
                    "variant_key": variant.key,
                    "seed": args.seed,
                    "trainable_parameters": count,
                }
            )
        results[variant.key] = {
            "label": variant.label,
            "checkpoint": str(checkpoint) if args.save_checkpoints else None,
            "checkpoint_sha256": (
                sha256_file(checkpoint) if args.save_checkpoints else None
            ),
            "detach_predictor_input": variant.detach_predictor_input,
            "wrong_alignment": variant.wrong_alignment,
            **result,
        }
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(parameter_counts) != 1:
        raise RuntimeError(f"Variant parameter mismatch: {sorted(parameter_counts)}")

    targets_path = args.output_dir / "targets-nexttoken-predictive.json"
    if args.save_checkpoints:
        targets_path.write_text(
            json.dumps(targets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    base_parameter_count = sum(
        int(value.numel())
        for value in base_state.values()
        if isinstance(value, torch.Tensor)
    )
    summary = {
        "experiment": "warm-start end-to-end next-token predictive V396 SAE v1",
        "git": {
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cache_manifest_sha256": cache.fingerprint(),
        "cache_read_only": True,
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "model_weights_size": args.model_weights.stat().st_size,
        "base_parameter_count": base_parameter_count,
        "trainable_parameter_count_each": next(iter(parameter_counts)),
        "new_predictor_parameter_count": (
            next(iter(parameter_counts)) - base_parameter_count
        ),
        "exposed_feature_count_each": int(base_state["encoder.weight"].shape[0]),
        "vocab_size": vocab_size,
        "targets_json": str(targets_path) if args.save_checkpoints else None,
        "variants": results,
        "fairness": {
            "same_base_checkpoint": True,
            "same_initial_trunk_tensors": True,
            "same_initial_predictor_tensors": True,
            "same_trainable_parameter_count": True,
            "same_exposed_feature_count": True,
            "same_cache": True,
            "same_data_order": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_batch_tokens": True,
            "same_target_marginal_candidate_and_wrong_control": True,
            "sae_trunk_unfrozen_in_all_variants": True,
            "predictor_trainable_in_all_variants": True,
            "uses_saebench_labels_for_training": False,
            "uses_eval_split_for_training": False,
            "uses_one_vs_rest_targets_for_training": False,
            "uses_mean_diff_selection_for_training": False,
            "uses_test_feedback_for_training": False,
        },
    }
    summary_path = args.output_dir / "train-summary-nexttoken-predictive.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "training_complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
