#!/usr/bin/env python3
"""Warm-start matched shared SAEs with true or deranged cross-layer concordance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import stat
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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def normalize_activation(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=-1, keepdim=True)) / (
        x.std(dim=-1, keepdim=True) + eps
    )


@dataclass(frozen=True)
class MultiLayerPackedBatch:
    activations: dict[int, torch.Tensor]
    sample_index: torch.Tensor
    lengths: torch.Tensor


class MultiLayerStructuredActivationCache:
    def __init__(
        self,
        cache_dir: Path,
        layers: tuple[int, ...],
        batch_samples: int,
        train_fraction: float,
        seed: int,
    ) -> None:
        self.cache_dir = cache_dir.resolve()
        self.layers = tuple(map(int, layers))
        self.batch_samples = int(batch_samples)
        self.train_fraction = float(train_fraction)
        self.seed = int(seed)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise ValueError("Structured cache is incomplete")
        if not self.manifest.get("summary", {}).get("read_only_finalized"):
            raise ValueError("Structured cache was not finalized read-only")
        if self.cache_dir.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise PermissionError(f"Cache directory must be read-only: {self.cache_dir}")
        available = set(map(int, self.manifest["configuration"]["layers"]))
        missing = sorted(set(self.layers) - available)
        if missing:
            raise ValueError(f"Layers absent from structured cache: {missing}")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Layers must be unique")
        if self.batch_samples < 2:
            raise ValueError("batch_samples must be at least two")
        self.sample_count = int(self.manifest["summary"]["sample_count"])
        self.train_cutoff = int(self.sample_count * self.train_fraction)
        self.shards = list(self.manifest["shards"])
        first_mean = self.cache_dir / self.manifest["layer_means"][str(self.layers[0])]["path"]
        self.d_model = int(
            torch.load(first_mean, map_location="cpu", weights_only=True).numel()
        )

    def fingerprint(self) -> str:
        return sha256_file(self.manifest_path)

    def iter_batches(
        self,
        epoch: int,
        split: str,
    ) -> Iterator[MultiLayerPackedBatch]:
        if split not in {"train", "validation"}:
            raise ValueError(split)
        order = list(range(len(self.shards)))
        if split == "train":
            random.Random(self.seed + epoch * 1_000_003).shuffle(order)
        for shard_position in order:
            entry = self.shards[shard_position]
            meta = torch.load(
                self.cache_dir / entry["meta"]["path"],
                map_location="cpu",
                weights_only=True,
            )
            layer_tensors = {
                layer: torch.load(
                    self.cache_dir / entry["layers"][str(layer)]["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                for layer in self.layers
            }
            row_counts = {int(tensor.shape[0]) for tensor in layer_tensors.values()}
            if len(row_counts) != 1:
                raise ValueError(f"Unaligned layer rows in shard {shard_position}")
            sample_ids = meta["sample_ids"].to(torch.int64)
            if split == "train":
                selected = torch.nonzero(
                    sample_ids < self.train_cutoff,
                    as_tuple=False,
                ).flatten()
            else:
                selected = torch.nonzero(
                    sample_ids >= self.train_cutoff,
                    as_tuple=False,
                ).flatten()
            if selected.numel() == 0:
                continue
            if split == "train":
                generator = torch.Generator(device="cpu").manual_seed(
                    self.seed + epoch * 1_000_003 + shard_position
                )
                selected = selected[
                    torch.randperm(selected.numel(), generator=generator)
                ]
            offsets = meta["offsets"].to(torch.int64)
            lengths_all = meta["lengths"].to(torch.int64)
            for start in range(0, int(selected.numel()), self.batch_samples):
                indices = selected[start : start + self.batch_samples]
                if split == "train" and int(indices.numel()) < self.batch_samples:
                    continue
                lengths = lengths_all[indices]
                if int(lengths.max().item()) * 2 > int(lengths.sum().item()):
                    # A bijective circular token derangement cannot exist for this batch.
                    if split == "train":
                        continue
                packed = {
                    layer: torch.cat(
                        [
                            tensor[int(offsets[index]) : int(offsets[index + 1])]
                            for index in indices.tolist()
                        ],
                        dim=0,
                    ).contiguous()
                    for layer, tensor in layer_tensors.items()
                }
                packed_counts = {int(tensor.shape[0]) for tensor in packed.values()}
                if len(packed_counts) != 1:
                    raise ValueError("Packed cross-layer token rows are not aligned")
                sample_index = torch.repeat_interleave(
                    torch.arange(indices.numel(), dtype=torch.int64),
                    lengths,
                )
                yield MultiLayerPackedBatch(
                    activations=packed,
                    sample_index=sample_index,
                    lengths=lengths,
                )


def wrong_token_permutation(sample_index: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Return a bijection that never pairs tokens from the same sample."""
    token_count = int(sample_index.numel())
    shift = int(lengths.max().item())
    if shift <= 0 or shift * 2 > token_count:
        raise ValueError("Batch does not admit the preregistered circular derangement")
    permutation = torch.arange(token_count, device=sample_index.device).roll(shift)
    if torch.unique(permutation).numel() != token_count:
        raise AssertionError("Wrong-alignment permutation is not bijective")
    if bool((sample_index == sample_index[permutation]).any().item()):
        raise AssertionError("Wrong-alignment permutation retained a same-sample pair")
    return permutation


def load_relu_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    keys = {"b_pre", "encoder.weight", "encoder.bias", "decoder.weight"}
    missing = sorted(keys - set(raw))
    if missing:
        raise KeyError(f"Checkpoint {path} is missing keys: {missing}")
    state = {key: raw[key].float().contiguous() for key in keys}
    if state["encoder.weight"].shape[::-1] != state["decoder.weight"].shape:
        raise ValueError("Encoder and decoder shapes are incompatible")
    return state


class CrossLayerSharedReLUSAE(nn.Module):
    """One shared dictionary with a parameter-neutral layer calibration module."""

    def __init__(
        self,
        base_state: dict[str, torch.Tensor],
        layers: tuple[int, ...],
        reference_layer: int,
        calibration_groups: int,
        max_log_scale: float,
    ) -> None:
        super().__init__()
        self.layers = tuple(map(int, layers))
        self.reference_layer = int(reference_layer)
        self.calibration_groups = int(calibration_groups)
        self.max_log_scale = float(max_log_scale)
        if self.reference_layer not in self.layers:
            raise ValueError("reference_layer must be one of layers")
        d_model = int(base_state["b_pre"].numel())
        n_latents = int(base_state["encoder.weight"].shape[0])
        calibration_parameter_count = len(self.layers) * self.calibration_groups
        if d_model % self.calibration_groups:
            raise ValueError("d_model must be divisible by calibration_groups")
        if calibration_parameter_count >= n_latents:
            raise ValueError("Calibration reserve exceeds the encoder bias")
        self.bias_core_count = n_latents - calibration_parameter_count
        self.b_pre = nn.Parameter(base_state["b_pre"].clone())
        self.encoder_weight = nn.Parameter(base_state["encoder.weight"].clone())
        self.encoder_bias_core = nn.Parameter(
            base_state["encoder.bias"][: self.bias_core_count].clone()
        )
        self.decoder_weight = nn.Parameter(base_state["decoder.weight"].clone())
        self.layer_group_log_scale = nn.Parameter(
            torch.zeros(
                (len(self.layers), self.calibration_groups),
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            "zero_bias_tail",
            torch.zeros(calibration_parameter_count, dtype=torch.float32),
            persistent=False,
        )
        self.normalize_decoder()

    @property
    def n_latents(self) -> int:
        return int(self.encoder_weight.shape[0])

    def dictionary_parameters(self) -> list[nn.Parameter]:
        return [
            self.b_pre,
            self.encoder_weight,
            self.encoder_bias_core,
            self.decoder_weight,
        ]

    def calibration_parameters(self) -> list[nn.Parameter]:
        return [self.layer_group_log_scale]

    def full_bias(self) -> torch.Tensor:
        return torch.cat(
            [
                self.encoder_bias_core,
                self.zero_bias_tail.to(self.encoder_bias_core.device),
            ]
        )

    def layer_scale(self, layer: int) -> torch.Tensor:
        layer_index = self.layers.index(int(layer))
        group_scale = self.layer_group_log_scale[layer_index].clamp(
            -self.max_log_scale,
            self.max_log_scale,
        ).exp()
        repeats = self.b_pre.numel() // self.calibration_groups
        return group_scale.repeat_interleave(repeats)

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder_weight.div_(
                self.decoder_weight.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
            )

    def project_decoder_grads(self) -> None:
        if self.decoder_weight.grad is None:
            return
        with torch.no_grad():
            projection = (
                self.decoder_weight * self.decoder_weight.grad
            ).sum(dim=1, keepdim=True)
            self.decoder_weight.grad.sub_(projection * self.decoder_weight)

    def forward(self, x_norm: torch.Tensor, layer: int) -> dict[str, torch.Tensor]:
        centered = (x_norm - self.b_pre) * self.layer_scale(layer)
        z = torch.relu(F.linear(centered, self.encoder_weight, self.full_bias()))
        recon = F.linear(z, self.decoder_weight) + self.b_pre
        return {"z": z, "recon": recon}

    def export_state(self) -> dict[str, torch.Tensor]:
        reference_scale = self.layer_scale(self.reference_layer).detach().cpu()
        effective_encoder = self.encoder_weight.detach().cpu() * reference_scale.unsqueeze(0)
        state = {
            "b_pre": self.b_pre.detach().cpu(),
            "encoder.weight": effective_encoder,
            "encoder.bias": self.full_bias().detach().cpu(),
            "decoder.weight": self.decoder_weight.detach().cpu(),
            "crosslayer.encoder_bias_core": self.encoder_bias_core.detach().cpu(),
            "crosslayer.layer_group_log_scale": self.layer_group_log_scale.detach().cpu(),
            "crosslayer.layers": torch.tensor(self.layers, dtype=torch.int64),
            "crosslayer.reference_layer": torch.tensor(self.reference_layer),
            "crosslayer.calibration_groups": torch.tensor(self.calibration_groups),
            "crosslayer.max_log_scale": torch.tensor(self.max_log_scale),
        }
        return state


def cosine_concordance_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left.float() * right.float()).sum(dim=-1)
    denominator = (
        left.float().square().sum(dim=-1).sqrt()
        * right.float().square().sum(dim=-1).sqrt()
    ).clamp_min(1.0e-8)
    return (1.0 - numerator / denominator).mean()


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    concordance_mode: str


VARIANTS = (
    Variant(
        key="reconstruction_control",
        label="matched four-layer reconstruction control",
        concordance_mode="none",
    ),
    Variant(
        key="wrong_alignment",
        label="matched wrong-alignment concordance control",
        concordance_mode="wrong",
    ),
    Variant(
        key="candidate",
        label="cross-layer concordance shared SAE",
        concordance_mode="true",
    ),
)


def validation_metrics(
    model: CrossLayerSharedReLUSAE,
    cache: MultiLayerStructuredActivationCache,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rec_sum = 0.0
    squared_sum = 0.0
    token_count = 0
    batches = 0
    with torch.inference_mode():
        for batch in cache.iter_batches(0, "validation"):
            x = normalize_activation(
                batch.activations[args.reference_layer],
                args.normalize_eps,
            ).to(device)
            out = model(x, args.reference_layer)
            residual = out["recon"].float() - x.float()
            rec_sum += float(residual.square().sum().item())
            squared_sum += float(x.float().square().sum().item())
            token_count += int(x.shape[0])
            batches += 1
            if batches >= args.validation_batches:
                break
    model.train()
    return {
        "reference_layer_validation_mse": rec_sum / max(token_count * cache.d_model, 1),
        "reference_layer_validation_explained_variance": 1.0
        - rec_sum / max(squared_sum, 1.0e-12),
        "validation_tokens": float(token_count),
        "validation_batches": float(batches),
    }


def backbone_max_delta(
    model: CrossLayerSharedReLUSAE,
    base_state: dict[str, torch.Tensor],
) -> float:
    deltas = [
        float((model.b_pre.detach().cpu() - base_state["b_pre"]).abs().max().item()),
        float(
            (
                model.encoder_weight.detach().cpu()
                - base_state["encoder.weight"]
            )
            .abs()
            .max()
            .item()
        ),
        float(
            (
                model.encoder_bias_core.detach().cpu()
                - base_state["encoder.bias"][: model.bias_core_count]
            )
            .abs()
            .max()
            .item()
        ),
        float(
            (
                model.decoder_weight.detach().cpu()
                - base_state["decoder.weight"]
            )
            .abs()
            .max()
            .item()
        ),
    ]
    return max(deltas)


def train_variant(
    variant: Variant,
    base_state: dict[str, torch.Tensor],
    cache: MultiLayerStructuredActivationCache,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CrossLayerSharedReLUSAE, dict[str, object]]:
    set_seed(args.seed)
    model = CrossLayerSharedReLUSAE(
        base_state=base_state,
        layers=tuple(args.layers),
        reference_layer=args.reference_layer,
        calibration_groups=args.calibration_groups,
        max_log_scale=args.max_log_scale,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.optimizer_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.lr / 5,
    )
    started = time.time()
    history: list[dict[str, float]] = []
    ever_active = torch.zeros(model.n_latents, dtype=torch.bool, device=device)
    epoch = 0
    step = 0
    wrong_fixed_pairs = 0
    wrong_same_sample_pairs = 0
    while step < args.steps:
        for batch in cache.iter_batches(epoch, "train"):
            normalized = {
                layer: normalize_activation(tensor, args.normalize_eps).to(device)
                for layer, tensor in batch.activations.items()
            }
            sample_index = batch.sample_index.to(device)
            lengths = batch.lengths.to(device)
            outputs = {
                layer: model(normalized[layer], layer) for layer in args.layers
            }
            rec_loss = torch.stack(
                [
                    F.mse_loss(outputs[layer]["recon"].float(), normalized[layer].float())
                    for layer in args.layers
                ]
            ).mean()
            l1 = torch.stack(
                [outputs[layer]["z"].float().sum(dim=1).mean() for layer in args.layers]
            ).mean()
            concordance = rec_loss.new_zeros(())
            if variant.concordance_mode != "none":
                reference = outputs[args.reference_layer]["z"]
                permutation = None
                if variant.concordance_mode == "wrong":
                    permutation = wrong_token_permutation(sample_index, lengths)
                    wrong_fixed_pairs += int(
                        (permutation == torch.arange(permutation.numel(), device=device))
                        .sum()
                        .item()
                    )
                    wrong_same_sample_pairs += int(
                        (sample_index == sample_index[permutation]).sum().item()
                    )
                terms = []
                for layer in args.layers:
                    if layer == args.reference_layer:
                        continue
                    other = outputs[layer]["z"]
                    if permutation is not None:
                        other = other[permutation]
                    terms.append(cosine_concordance_loss(reference, other))
                concordance = torch.stack(terms).mean()
            normalized_concordance = rec_loss.new_zeros(())
            loss = rec_loss + args.l1_coeff * l1
            if variant.concordance_mode != "none":
                # Equalize true/wrong objective scale without changing either
                # pairing's gradient direction. This prevents the naturally
                # larger wrong-pair cosine distance from becoming a stronger
                # regularizer than the candidate receives.
                normalized_concordance = concordance / concordance.detach().clamp_min(
                    1.0e-8
                )
                loss = loss + args.concordance_weight * normalized_concordance
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{variant.key} produced NaN/Inf at step {step + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            dictionary_grad = grad_norm(model.dictionary_parameters())
            calibration_grad = grad_norm(model.calibration_parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            model.project_decoder_grads()
            optimizer.step()
            model.normalize_decoder()
            scheduler.step()
            step += 1
            with torch.no_grad():
                active = torch.stack(
                    [(outputs[layer]["z"] > 0).any(dim=0) for layer in args.layers]
                ).any(dim=0)
                ever_active |= active
                reference_residual = (
                    outputs[args.reference_layer]["recon"].float()
                    - normalized[args.reference_layer].float()
                )
                reference_ev = 1.0 - reference_residual.square().sum() / normalized[
                    args.reference_layer
                ].float().square().sum()
                active_per_token = torch.stack(
                    [
                        (outputs[layer]["z"] > 0).float().sum(dim=1).mean()
                        for layer in args.layers
                    ]
                ).mean()
            row = {
                "step": float(step),
                "epoch": float(epoch),
                "loss": float(loss.detach().item()),
                "reconstruction_loss": float(rec_loss.detach().item()),
                "l1": float(l1.detach().item()),
                "concordance_loss": float(concordance.detach().item()),
                "normalized_concordance_objective": float(
                    normalized_concordance.detach().item()
                ),
                "weighted_concordance_objective": float(
                    (args.concordance_weight * normalized_concordance.detach()).item()
                ),
                "reference_layer_ev": float(reference_ev.item()),
                "active_per_token": float(active_per_token.item()),
                "dead_ratio_so_far": float((~ever_active).float().mean().item()),
                "dictionary_grad_norm": dictionary_grad,
                "calibration_grad_norm": calibration_grad,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - started,
            }
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                history.append(row)
                print(
                    json.dumps(
                        {"variant": variant.key, **row},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            del normalized, sample_index, lengths, outputs, rec_loss, l1
            del concordance, normalized_concordance, loss
            del reference_residual, reference_ev, active_per_token
            if step >= args.steps:
                break
        epoch += 1
    validation = validation_metrics(model, cache, args, device)
    result: dict[str, object] = {
        "variant_key": variant.key,
        "label": variant.label,
        "concordance_mode": variant.concordance_mode,
        "parameter_count": parameter_count(model),
        "global_steps": step,
        "epochs_touched": epoch,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "validation": validation,
        "backbone_parameter_max_delta": backbone_max_delta(model, base_state),
        "calibration_parameter_max_delta": float(
            model.layer_group_log_scale.detach().abs().max().item()
        ),
        "wrong_fixed_pair_count": wrong_fixed_pairs,
        "wrong_same_sample_pair_count": wrong_same_sample_pairs,
        "final_dead_ratio": float((~ever_active).float().mean().item()),
    }
    return model, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[20, 21, 22, 23])
    parser.add_argument("--reference-layer", type=int, default=22)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-samples", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--beta1", type=float, default=0.85)
    parser.add_argument("--beta2", type=float, default=0.9999)
    parser.add_argument("--optimizer-eps", type=float, default=6.25e-10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-coeff", type=float, default=1.0e-4)
    parser.add_argument("--concordance-weight", type=float, default=0.005)
    parser.add_argument("--calibration-groups", type=int, default=16)
    parser.add_argument("--max-log-scale", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.layers = list(dict.fromkeys(map(int, args.layers)))
    if args.reference_layer not in args.layers:
        raise ValueError("reference-layer must be present in layers")
    set_seed(args.seed)
    device = torch.device(args.device)
    cache = MultiLayerStructuredActivationCache(
        cache_dir=args.cache_dir,
        layers=tuple(args.layers),
        batch_samples=args.batch_samples,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    base_state = load_relu_state(args.base_checkpoint)
    if int(base_state["b_pre"].numel()) != cache.d_model:
        raise ValueError("Base checkpoint and cache d_model differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    targets: list[dict[str, object]] = []
    counts: set[int] = set()
    for variant in VARIANTS:
        model, result = train_variant(variant, base_state, cache, args, device)
        checkpoint = args.output_dir / f"trained_sae-{variant.key}.pt"
        if args.save_checkpoints:
            torch.save(model.export_state(), checkpoint)
            result["checkpoint"] = str(checkpoint)
            result["checkpoint_sha256"] = sha256_file(checkpoint)
        else:
            result["checkpoint"] = None
            result["checkpoint_sha256"] = None
        results[variant.key] = result
        counts.add(int(result["parameter_count"]))
        if args.save_checkpoints:
            targets.append(
                {
                    "label": variant.label,
                    "kind": "relu",
                    "layer": args.reference_layer,
                    "checkpoint": str(checkpoint),
                    "variant_key": variant.key,
                }
            )
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(counts) != 1:
        raise RuntimeError(f"Variant parameter counts differ: {sorted(counts)}")
    expected = sum(int(tensor.numel()) for tensor in base_state.values())
    if counts != {expected}:
        raise RuntimeError(
            f"Parameter count differs from source ReLU: variants={counts}, source={expected}"
        )
    targets_path = args.output_dir / "targets-crosslayer-concordance-v1.json"
    if args.save_checkpoints:
        targets_path.write_text(
            json.dumps(targets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "experiment": "cross-layer concordance shared SAE v1 short warm-start screen",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cache_manifest_sha256": cache.fingerprint(),
        "cache_read_only": True,
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "source_parameter_count": expected,
        "parameter_count_each": next(iter(counts)),
        "exposed_feature_count_each": int(base_state["encoder.weight"].shape[0]),
        "same_initial_tensors": True,
        "targets_json": str(targets_path) if args.save_checkpoints else None,
        "variants": results,
        "fairness": {
            "same_structured_cache": True,
            "same_layers": True,
            "same_reference_layer": True,
            "same_data_order": True,
            "same_optimizer": True,
            "same_training_steps": True,
            "same_parameter_count": True,
            "same_exposed_feature_count": True,
            "sae_dictionary_unfrozen_in_all_variants": True,
            "calibration_module_trainable_in_all_variants": True,
            "only_causal_difference_is_concordance_pairing": True,
            "uses_saebench_labels_for_training": False,
            "uses_eval_split_for_training": False,
            "uses_one_vs_rest_targets_for_training": False,
            "uses_mean_diff_selection_for_training": False,
            "uses_test_feedback_for_training": False,
        },
    }
    summary_path = args.output_dir / "train-summary-crosslayer-concordance-v1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "training_complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
