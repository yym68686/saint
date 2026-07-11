#!/usr/bin/env python3
"""Train an exact-parameter CP-factorized multi-view SAE and causal controls."""

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


def load_relu_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    keys = {"b_pre", "encoder.weight", "encoder.bias", "decoder.weight"}
    missing = sorted(keys - set(raw))
    if missing:
        raise KeyError(f"Checkpoint {path} missing keys: {missing}")
    state = {key: raw[key].float().contiguous() for key in keys}
    if state["encoder.weight"].shape[::-1] != state["decoder.weight"].shape:
        raise ValueError("Encoder and decoder shapes are incompatible")
    return state


@dataclass(frozen=True)
class PackedViewBatch:
    activations: torch.Tensor
    sample_index: torch.Tensor
    view_index: torch.Tensor
    lengths: torch.Tensor


class StructuredViewCache:
    def __init__(
        self,
        cache_dir: Path,
        layer: int,
        n_views: int,
        batch_samples: int,
        train_fraction: float,
        seed: int,
    ) -> None:
        self.cache_dir = cache_dir.resolve()
        self.layer = int(layer)
        self.n_views = int(n_views)
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
            raise PermissionError(f"Cache must be read-only: {self.cache_dir}")
        layers = set(map(int, self.manifest["configuration"]["layers"]))
        if self.layer not in layers:
            raise ValueError(f"Layer {self.layer} is absent from the cache")
        if self.n_views < 2:
            raise ValueError("n_views must be at least two")
        if self.batch_samples <= self.n_views:
            raise ValueError("batch_samples must exceed n_views for wrong-view controls")
        self.sample_count = int(self.manifest["summary"]["sample_count"])
        self.train_cutoff = int(self.sample_count * self.train_fraction)
        self.shards = list(self.manifest["shards"])
        mean_path = self.cache_dir / self.manifest["layer_means"][str(self.layer)]["path"]
        self.d_model = int(
            torch.load(mean_path, map_location="cpu", weights_only=True).numel()
        )

    def fingerprint(self) -> str:
        return sha256_file(self.manifest_path)

    def iter_batches(self, epoch: int, split: str) -> Iterator[PackedViewBatch]:
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
            activations = torch.load(
                self.cache_dir / entry["layers"][str(self.layer)]["path"],
                map_location="cpu",
                weights_only=True,
            )
            sample_ids = meta["sample_ids"].to(torch.int64)
            lengths_all = meta["lengths"].to(torch.int64)
            eligible = lengths_all >= self.n_views
            if split == "train":
                selected = torch.nonzero(
                    (sample_ids < self.train_cutoff) & eligible,
                    as_tuple=False,
                ).flatten()
            else:
                selected = torch.nonzero(
                    (sample_ids >= self.train_cutoff) & eligible,
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
            for start in range(0, int(selected.numel()), self.batch_samples):
                indices = selected[start : start + self.batch_samples]
                if split == "train" and int(indices.numel()) < self.batch_samples:
                    continue
                pieces = [
                    activations[int(offsets[index]) : int(offsets[index + 1])]
                    for index in indices.tolist()
                ]
                lengths = lengths_all[indices]
                packed = torch.cat(pieces, dim=0).contiguous()
                sample_index = torch.repeat_interleave(
                    torch.arange(indices.numel(), dtype=torch.int64),
                    lengths,
                )
                view_index = torch.cat(
                    [
                        torch.arange(int(length.item()), dtype=torch.int64)
                        % self.n_views
                        for length in lengths
                    ]
                )
                yield PackedViewBatch(
                    activations=packed,
                    sample_index=sample_index,
                    view_index=view_index,
                    lengths=lengths,
                )


def pool_interleaved_views(
    x_norm: torch.Tensor,
    sample_index: torch.Tensor,
    view_index: torch.Tensor,
    sample_count: int,
    n_views: int,
) -> torch.Tensor:
    combined = sample_index * n_views + view_index
    sums = torch.zeros(
        (sample_count * n_views, x_norm.shape[-1]),
        device=x_norm.device,
        dtype=x_norm.dtype,
    )
    sums.index_add_(0, combined, x_norm)
    counts = torch.zeros(
        sample_count * n_views,
        device=x_norm.device,
        dtype=x_norm.dtype,
    )
    counts.index_add_(0, combined, torch.ones_like(combined, dtype=x_norm.dtype))
    if bool((counts == 0).any().item()):
        raise AssertionError("Training cache emitted an empty interleaved view")
    return (sums / counts.unsqueeze(1)).reshape(sample_count, n_views, -1)


def derange_views(views: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Permute each view by a distinct nonzero sample shift."""
    batch_size, n_views, _ = views.shape
    if batch_size <= n_views:
        raise ValueError("Batch is too small for distinct view derangements")
    rows = []
    collisions = 0
    base = torch.arange(batch_size, device=views.device)
    for view in range(n_views):
        permutation = base.roll(view + 1)
        collisions += int((permutation == base).sum().item())
        rows.append(views[permutation, view])
    return torch.stack(rows, dim=1), collisions


def interpolated_gain(knots: torch.Tensor, n_latents: int, max_log_gain: float) -> torch.Tensor:
    values = F.interpolate(
        knots.view(1, 1, -1),
        size=n_latents,
        mode="linear",
        align_corners=True,
    ).view(-1)
    return values.clamp(-max_log_gain, max_log_gain).exp()


def best_rank_projection(
    matrix: torch.Tensor,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return P and matrix@P for the optimal right rank-r projection."""
    if matrix.ndim != 2 or matrix.shape[1] < rank:
        raise ValueError("Invalid matrix/rank for projection")
    matrix_device = matrix.to(device)
    gram = matrix_device.T @ matrix_device
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    basis = eigenvectors[:, -rank:].contiguous()
    feature_factor = (matrix_device @ basis).contiguous()
    discarded = eigenvalues[:-rank].clamp_min(0).sum()
    total = eigenvalues.clamp_min(0).sum().clamp_min(1.0e-12)
    relative_error = float((discarded / total).sqrt().item())
    basis_cpu = basis.cpu()
    feature_cpu = feature_factor.cpu()
    del matrix_device, gram, eigenvalues, eigenvectors, basis, feature_factor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return basis_cpu, feature_cpu, relative_error


def make_cp_initial_state(
    base_state: dict[str, torch.Tensor],
    n_views: int,
    rank: int,
    target_parameter_count: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    n_latents, d_model = map(int, base_state["encoder.weight"].shape)
    encoder_basis, encoder_feature, encoder_error = best_rank_projection(
        base_state["encoder.weight"], rank, device
    )
    decoder_basis, decoder_feature, decoder_error = best_rank_projection(
        base_state["decoder.weight"].T.contiguous(), rank, device
    )
    projected_center_shift = encoder_feature @ (
        encoder_basis.T @ base_state["b_pre"]
    )
    fixed_count = (
        2 * rank * (d_model + n_latents + n_views)
        + n_latents
        + n_views * d_model
    )
    gain_count = target_parameter_count - fixed_count
    if gain_count < 2:
        raise ValueError(
            f"CP rank {rank} exceeds parameter budget: remaining={gain_count}"
        )
    state = {
        "encoder_input_basis": encoder_basis,
        "encoder_feature_factor": encoder_feature,
        "encoder_view_factor": torch.full(
            (n_views, rank), 1.0 / n_views, dtype=torch.float32
        ),
        "encoder_bias": base_state["encoder.bias"].clone()
        - projected_center_shift,
        "decoder_input_basis": decoder_basis,
        "decoder_feature_factor": decoder_feature,
        "decoder_view_factor": torch.ones(n_views, rank, dtype=torch.float32),
        "decoder_bias": base_state["b_pre"].repeat(n_views, 1),
        "feature_gain_knots": torch.zeros(gain_count, dtype=torch.float32),
    }
    metadata: dict[str, float | int] = {
        "rank": rank,
        "gain_parameter_count": gain_count,
        "fixed_factor_parameter_count": fixed_count,
        "target_parameter_count": target_parameter_count,
        "encoder_projection_relative_frobenius_error": encoder_error,
        "decoder_projection_relative_frobenius_error": decoder_error,
    }
    return state, metadata


class MeanPooledReLUSAE(nn.Module):
    def __init__(self, base_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.b_pre = nn.Parameter(base_state["b_pre"].clone())
        self.encoder_weight = nn.Parameter(base_state["encoder.weight"].clone())
        self.encoder_bias = nn.Parameter(base_state["encoder.bias"].clone())
        self.decoder_weight = nn.Parameter(base_state["decoder.weight"].clone())

    @property
    def n_latents(self) -> int:
        return int(self.encoder_weight.shape[0])

    def core_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def module_parameters(self) -> list[nn.Parameter]:
        return []

    def forward(self, views: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = views.mean(dim=1)
        z = torch.relu(F.linear(pooled - self.b_pre, self.encoder_weight, self.encoder_bias))
        one = F.linear(z, self.decoder_weight) + self.b_pre
        recon = one.unsqueeze(1).expand(-1, views.shape[1], -1)
        return {"z": z, "recon": recon}

    def export_state(self) -> dict[str, torch.Tensor]:
        return {
            "b_pre": self.b_pre.detach().cpu(),
            "encoder.weight": self.encoder_weight.detach().cpu(),
            "encoder.bias": self.encoder_bias.detach().cpu(),
            "decoder.weight": self.decoder_weight.detach().cpu(),
            "multiview.kind": torch.tensor(0),
        }


class CPMultiViewSAE(nn.Module):
    def __init__(
        self,
        initial_state: dict[str, torch.Tensor],
        max_log_gain: float,
    ) -> None:
        super().__init__()
        self.encoder_input_basis = nn.Parameter(initial_state["encoder_input_basis"].clone())
        self.encoder_feature_factor = nn.Parameter(
            initial_state["encoder_feature_factor"].clone()
        )
        self.encoder_view_factor = nn.Parameter(initial_state["encoder_view_factor"].clone())
        self.encoder_bias = nn.Parameter(initial_state["encoder_bias"].clone())
        self.decoder_input_basis = nn.Parameter(initial_state["decoder_input_basis"].clone())
        self.decoder_feature_factor = nn.Parameter(
            initial_state["decoder_feature_factor"].clone()
        )
        self.decoder_view_factor = nn.Parameter(initial_state["decoder_view_factor"].clone())
        self.decoder_bias = nn.Parameter(initial_state["decoder_bias"].clone())
        self.feature_gain_knots = nn.Parameter(initial_state["feature_gain_knots"].clone())
        self.max_log_gain = float(max_log_gain)

    @property
    def n_latents(self) -> int:
        return int(self.encoder_feature_factor.shape[0])

    @property
    def n_views(self) -> int:
        return int(self.encoder_view_factor.shape[0])

    @property
    def rank(self) -> int:
        return int(self.encoder_input_basis.shape[1])

    def core_parameters(self) -> list[nn.Parameter]:
        return [
            self.encoder_input_basis,
            self.encoder_feature_factor,
            self.encoder_view_factor,
            self.encoder_bias,
            self.decoder_input_basis,
            self.decoder_feature_factor,
            self.decoder_view_factor,
            self.decoder_bias,
        ]

    def module_parameters(self) -> list[nn.Parameter]:
        return [
            self.encoder_view_factor,
            self.decoder_view_factor,
            self.feature_gain_knots,
        ]

    def feature_gain(self) -> torch.Tensor:
        return interpolated_gain(
            self.feature_gain_knots,
            self.n_latents,
            self.max_log_gain,
        )

    def forward(self, views: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = torch.einsum("bvd,dr->bvr", views, self.encoder_input_basis)
        mixed = (projected * self.encoder_view_factor.unsqueeze(0)).sum(dim=1)
        pre = F.linear(mixed, self.encoder_feature_factor, self.encoder_bias)
        z = torch.relu(pre) * self.feature_gain().unsqueeze(0)
        decoder_code = z @ self.decoder_feature_factor
        view_codes = decoder_code.unsqueeze(1) * self.decoder_view_factor.unsqueeze(0)
        recon = torch.einsum("bvr,dr->bvd", view_codes, self.decoder_input_basis)
        recon = recon + self.decoder_bias.unsqueeze(0)
        return {"z": z, "recon": recon}

    def export_state(self) -> dict[str, torch.Tensor]:
        return {
            "multiview.kind": torch.tensor(1),
            "multiview.n_views": torch.tensor(self.n_views),
            "multiview.rank": torch.tensor(self.rank),
            "multiview.max_log_gain": torch.tensor(self.max_log_gain),
            "encoder_input_basis": self.encoder_input_basis.detach().cpu(),
            "encoder_feature_factor": self.encoder_feature_factor.detach().cpu(),
            "encoder_view_factor": self.encoder_view_factor.detach().cpu(),
            "encoder_bias": self.encoder_bias.detach().cpu(),
            "decoder_input_basis": self.decoder_input_basis.detach().cpu(),
            "decoder_feature_factor": self.decoder_feature_factor.detach().cpu(),
            "decoder_view_factor": self.decoder_view_factor.detach().cpu(),
            "decoder_bias": self.decoder_bias.detach().cpu(),
            "feature_gain_knots": self.feature_gain_knots.detach().cpu(),
        }


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    kind: str
    wrong_views: bool


VARIANTS = (
    Variant(
        key="mean_pooled_control",
        label="matched mean-pooled ReLU SAE control",
        kind="mean_relu",
        wrong_views=False,
    ),
    Variant(
        key="wrong_alignment",
        label="CP multi-view wrong-sample control",
        kind="cp_multiview",
        wrong_views=True,
    ),
    Variant(
        key="candidate",
        label="CP factorized multi-view SAE",
        kind="cp_multiview",
        wrong_views=False,
    ),
)


def validation_metrics(
    model: MeanPooledReLUSAE | CPMultiViewSAE,
    cache: StructuredViewCache,
    args: argparse.Namespace,
    device: torch.device,
    wrong_views: bool,
) -> dict[str, float]:
    model.eval()
    rec_sum = 0.0
    squared_sum = 0.0
    values = 0
    batches = 0
    with torch.inference_mode():
        for batch in cache.iter_batches(0, "validation"):
            x = normalize_activation(batch.activations, args.normalize_eps).to(device)
            views = pool_interleaved_views(
                x,
                batch.sample_index.to(device),
                batch.view_index.to(device),
                int(batch.lengths.numel()),
                args.n_views,
            )
            if wrong_views:
                views, _ = derange_views(views)
            out = model(views)
            residual = out["recon"].float() - views.float()
            rec_sum += float(residual.square().sum().item())
            squared_sum += float(views.float().square().sum().item())
            values += int(views.numel())
            batches += 1
            if batches >= args.validation_batches:
                break
    model.train()
    return {
        "validation_mse": rec_sum / max(values, 1),
        "validation_explained_variance": 1.0 - rec_sum / max(squared_sum, 1.0e-12),
        "validation_batches": float(batches),
    }


def max_delta(
    model: MeanPooledReLUSAE | CPMultiViewSAE,
    initial: dict[str, torch.Tensor],
) -> tuple[float, float]:
    if isinstance(model, MeanPooledReLUSAE):
        pairs = [
            (model.b_pre, initial["b_pre"]),
            (model.encoder_weight, initial["encoder.weight"]),
            (model.encoder_bias, initial["encoder.bias"]),
            (model.decoder_weight, initial["decoder.weight"]),
        ]
        core = max(
            float((parameter.detach().cpu() - reference).abs().max().item())
            for parameter, reference in pairs
        )
        return core, 0.0
    names = {
        "encoder_input_basis": model.encoder_input_basis,
        "encoder_feature_factor": model.encoder_feature_factor,
        "encoder_view_factor": model.encoder_view_factor,
        "encoder_bias": model.encoder_bias,
        "decoder_input_basis": model.decoder_input_basis,
        "decoder_feature_factor": model.decoder_feature_factor,
        "decoder_view_factor": model.decoder_view_factor,
        "decoder_bias": model.decoder_bias,
        "feature_gain_knots": model.feature_gain_knots,
    }
    deltas = {
        name: float((parameter.detach().cpu() - initial[name]).abs().max().item())
        for name, parameter in names.items()
    }
    module_names = {
        "encoder_view_factor",
        "decoder_view_factor",
        "feature_gain_knots",
    }
    return (
        max(value for name, value in deltas.items() if name not in module_names),
        max(deltas[name] for name in module_names),
    )


def train_variant(
    variant: Variant,
    base_state: dict[str, torch.Tensor],
    cp_state: dict[str, torch.Tensor],
    cache: StructuredViewCache,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MeanPooledReLUSAE | CPMultiViewSAE, dict[str, object]]:
    set_seed(args.seed)
    if variant.kind == "mean_relu":
        model: MeanPooledReLUSAE | CPMultiViewSAE = MeanPooledReLUSAE(base_state)
        initial = base_state
    else:
        model = CPMultiViewSAE(cp_state, args.max_log_gain)
        initial = cp_state
    model.to(device)
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
    collisions = 0
    step = 0
    epoch = 0
    while step < args.steps:
        for batch in cache.iter_batches(epoch, "train"):
            x = normalize_activation(batch.activations, args.normalize_eps).to(device)
            views = pool_interleaved_views(
                x,
                batch.sample_index.to(device),
                batch.view_index.to(device),
                int(batch.lengths.numel()),
                args.n_views,
            )
            if variant.wrong_views:
                views, local_collisions = derange_views(views)
                collisions += local_collisions
            out = model(views)
            rec_loss = F.mse_loss(out["recon"].float(), views.float())
            l1 = out["z"].float().sum(dim=1).mean()
            loss = rec_loss + args.l1_coeff * l1
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{variant.key} produced NaN/Inf at step {step + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            core_grad = grad_norm(model.core_parameters())
            module_grad = grad_norm(model.module_parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            with torch.no_grad():
                ever_active |= (out["z"] > 0).any(dim=0)
                residual = out["recon"].float() - views.float()
                ev = 1.0 - residual.square().sum() / views.float().square().sum()
                active_per_sample = (out["z"] > 0).float().sum(dim=1).mean()
            row = {
                "step": float(step),
                "epoch": float(epoch),
                "loss": float(loss.detach().item()),
                "reconstruction_loss": float(rec_loss.detach().item()),
                "l1": float(l1.detach().item()),
                "explained_variance": float(ev.item()),
                "active_per_sample": float(active_per_sample.item()),
                "dead_ratio_so_far": float((~ever_active).float().mean().item()),
                "core_grad_norm": core_grad,
                "module_grad_norm": module_grad,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - started,
            }
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                history.append(row)
                print(json.dumps({"variant": variant.key, **row}), flush=True)
            del x, views, out, rec_loss, l1, loss, residual, ev, active_per_sample
            if step >= args.steps:
                break
        epoch += 1
    validation = validation_metrics(
        model,
        cache,
        args,
        device,
        variant.wrong_views,
    )
    core_delta, module_delta = max_delta(model, initial)
    return model, {
        "variant_key": variant.key,
        "label": variant.label,
        "kind": variant.kind,
        "wrong_views": variant.wrong_views,
        "parameter_count": parameter_count(model),
        "global_steps": step,
        "epochs_touched": epoch,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "validation": validation,
        "core_parameter_max_delta": core_delta,
        "module_parameter_max_delta": module_delta,
        "wrong_view_fixed_pair_count": collisions,
        "final_dead_ratio": float((~ever_active).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--cp-rank", type=int, default=2_934)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-samples", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--beta1", type=float, default=0.85)
    parser.add_argument("--beta2", type=float, default=0.9999)
    parser.add_argument("--optimizer-eps", type=float, default=6.25e-10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-coeff", type=float, default=1.0e-4)
    parser.add_argument("--max-log-gain", type=float, default=0.25)
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

    set_seed(args.seed)
    device = torch.device(args.device)
    cache = StructuredViewCache(
        cache_dir=args.cache_dir,
        layer=args.layer,
        n_views=args.n_views,
        batch_samples=args.batch_samples,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    base_state = load_relu_state(args.base_checkpoint)
    if int(base_state["b_pre"].numel()) != cache.d_model:
        raise ValueError("Base checkpoint and cache d_model differ")
    target_parameter_count = sum(tensor.numel() for tensor in base_state.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "cp_initialization_started"}), flush=True)
    cp_state, cp_metadata = make_cp_initial_state(
        base_state=base_state,
        n_views=args.n_views,
        rank=args.cp_rank,
        target_parameter_count=target_parameter_count,
        device=device,
    )
    print(
        json.dumps({"event": "cp_initialization_complete", **cp_metadata}),
        flush=True,
    )

    results: dict[str, object] = {}
    targets: list[dict[str, object]] = [
        {
            "label": "frozen token-mean ReLU reference",
            "kind": "token_relu",
            "layer": args.layer,
            "checkpoint": str(args.base_checkpoint),
            "variant_key": "frozen_relu",
        }
    ]
    counts: set[int] = set()
    for variant in VARIANTS:
        model, result = train_variant(
            variant,
            base_state,
            cp_state,
            cache,
            args,
            device,
        )
        checkpoint = args.output_dir / f"trained_sae-{variant.key}.pt"
        if args.save_checkpoints:
            torch.save(model.export_state(), checkpoint)
            result["checkpoint"] = str(checkpoint)
            result["checkpoint_sha256"] = sha256_file(checkpoint)
            targets.append(
                {
                    "label": variant.label,
                    "kind": variant.kind,
                    "layer": args.layer,
                    "checkpoint": str(checkpoint),
                    "variant_key": variant.key,
                }
            )
        else:
            result["checkpoint"] = None
            result["checkpoint_sha256"] = None
        results[variant.key] = result
        counts.add(int(result["parameter_count"]))
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if counts != {target_parameter_count}:
        raise RuntimeError(
            f"Parameter mismatch: variants={sorted(counts)} source={target_parameter_count}"
        )
    targets_path = args.output_dir / "targets-cp-multiview-v1.json"
    if args.save_checkpoints:
        targets_path.write_text(
            json.dumps(targets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "experiment": "exact-parameter CP factorized multi-view SAE v1 short screen",
        "paper_inspiration": "arXiv:2605.09438 fmxcoders",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cache_manifest_sha256": cache.fingerprint(),
        "cache_read_only": True,
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "source_parameter_count": target_parameter_count,
        "parameter_count_each": next(iter(counts)),
        "exposed_feature_count_each": int(base_state["encoder.weight"].shape[0]),
        "cp_initialization": cp_metadata,
        "targets_json": str(targets_path) if args.save_checkpoints else None,
        "variants": results,
        "fairness": {
            "same_l22_activation_cache": True,
            "same_real_sample_boundaries": True,
            "same_interleaved_view_definition": True,
            "same_data_order": True,
            "same_optimizer": True,
            "same_training_steps": True,
            "same_parameter_count": True,
            "same_exposed_feature_count": True,
            "all_models_end_to_end_trainable": True,
            "wrong_control_preserves_each_view_marginal": True,
            "uses_saebench_labels_for_training": False,
            "uses_eval_split_for_training": False,
            "uses_one_vs_rest_targets_for_training": False,
            "uses_mean_diff_selection_for_training": False,
            "uses_test_feedback_for_training": False,
        },
    }
    summary_path = args.output_dir / "train-summary-cp-multiview-v1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "training_complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
