#!/usr/bin/env python3
"""Train matched BatchTopK and sample-nested BatchTopK SAEs."""

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


def grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


@dataclass(frozen=True)
class PackedBatch:
    activations: torch.Tensor
    sample_index: torch.Tensor
    lengths: torch.Tensor


class StructuredActivationCache:
    def __init__(
        self,
        cache_dir: Path,
        layer: int,
        batch_samples: int,
        train_fraction: float,
        seed: int,
    ) -> None:
        self.cache_dir = cache_dir.resolve()
        self.layer = int(layer)
        self.batch_samples = int(batch_samples)
        self.train_fraction = float(train_fraction)
        self.seed = int(seed)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest["status"] != "complete":
            raise ValueError(f"Cache is incomplete: {self.manifest['status']}")
        if self.layer not in map(int, self.manifest["configuration"]["layers"]):
            raise ValueError(f"Layer {self.layer} is absent from the cache")
        if self.cache_dir.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise PermissionError(f"Cache directory must be read-only: {self.cache_dir}")
        self.sample_count = int(self.manifest["summary"]["sample_count"])
        self.train_cutoff = int(self.sample_count * self.train_fraction)
        self.shards = list(self.manifest["shards"])
        self.mean_path = self.cache_dir / self.manifest["layer_means"][str(self.layer)]["path"]
        self.d_model = int(
            torch.load(
                self.mean_path,
                map_location="cpu",
                weights_only=True,
            ).numel()
        )

    def fingerprint(self) -> str:
        return sha256_file(self.manifest_path)

    def mean(self) -> torch.Tensor:
        return torch.load(self.mean_path, map_location="cpu", weights_only=True).float()

    def batch_count(self, split: str) -> int:
        if split not in {"train", "validation"}:
            raise ValueError(split)
        count = 0
        for entry in self.shards:
            meta = torch.load(
                self.cache_dir / entry["meta"]["path"],
                map_location="cpu",
                weights_only=True,
            )
            sample_ids = meta["sample_ids"].to(torch.int64)
            if split == "train":
                selected = int((sample_ids < self.train_cutoff).sum().item())
                count += selected // self.batch_samples
            else:
                selected = int((sample_ids >= self.train_cutoff).sum().item())
                count += math.ceil(selected / self.batch_samples)
        return count

    def iter_batches(self, epoch: int, split: str) -> Iterator[PackedBatch]:
        if split not in {"train", "validation"}:
            raise ValueError(split)
        order = list(range(len(self.shards)))
        py_rng = random.Random(self.seed + epoch * 1_000_003)
        if split == "train":
            py_rng.shuffle(order)
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
                yield PackedBatch(
                    activations=packed,
                    sample_index=sample_index,
                    lengths=lengths,
                )


def make_initial_tensors(
    n_latents: int,
    d_model: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    encoder = torch.empty(
        (n_latents, d_model),
        dtype=torch.float32,
    )
    nn.init.orthogonal_(encoder, generator=generator)
    encoder_bias = torch.zeros(n_latents, dtype=torch.float32)
    decoder = encoder.T.contiguous()
    # Match the existing baseline implementation exactly: the stored decoder
    # matrix is normalized and projected row-wise.
    decoder.div_(decoder.norm(dim=1, keepdim=True).clamp_min(1.0e-6))
    return encoder, encoder_bias, decoder


class BatchTopKFromScratchSAE(nn.Module):
    def __init__(
        self,
        b_pre: torch.Tensor,
        n_latents: int,
        top_k: int,
        seed: int,
    ) -> None:
        super().__init__()
        if top_k <= 0 or top_k > n_latents:
            raise ValueError("top_k must lie in [1, n_latents]")
        d_model = int(b_pre.numel())
        encoder, encoder_bias, decoder = make_initial_tensors(
            n_latents,
            d_model,
            seed,
        )
        self.b_pre = nn.Parameter(b_pre.clone())
        self.encoder_weight = nn.Parameter(encoder)
        self.encoder_bias = nn.Parameter(encoder_bias)
        self.decoder_weight = nn.Parameter(decoder)
        self.top_k = int(top_k)

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

    def base_parameters(self) -> list[nn.Parameter]:
        return [
            self.b_pre,
            self.encoder_weight,
            self.encoder_bias,
            self.decoder_weight,
        ]

    @property
    def n_total(self) -> int:
        return int(self.encoder_weight.shape[0])

    def semantic_parameters(self) -> list[nn.Parameter]:
        return []

    def module_snapshot(self) -> list[torch.Tensor]:
        return []

    def module_grad_norm(self) -> float:
        return 0.0

    def module_parameter_delta(
        self,
        initial: list[torch.Tensor],
    ) -> float:
        if initial:
            raise ValueError("BatchTopK control has no nested-module snapshot")
        return 0.0

    @staticmethod
    def batchtopk_sparse(h: torch.Tensor, top_k: int) -> torch.Tensor:
        h_relu = torch.relu(h)
        keep = min(top_k * h_relu.shape[0], h_relu.numel())
        values, indices = torch.topk(h_relu.reshape(-1), k=keep, dim=0)
        sparse = torch.zeros_like(h_relu).reshape(-1)
        sparse.scatter_(0, indices, values)
        return sparse.reshape_as(h_relu)

    @staticmethod
    def l1_per_token(
        out: dict[str, torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        del lengths
        return out["z_token"].sum(dim=1).mean()

    @staticmethod
    def active_features(out: dict[str, torch.Tensor]) -> torch.Tensor:
        return (out["z_token"] > 0).any(dim=0)

    @staticmethod
    def reconstruction_losses(
        out: dict[str, torch.Tensor],
        x_norm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del lengths
        total = F.mse_loss(out["recon"].float(), x_norm.float())
        return {
            "total": total,
            "token": total,
            "semantic": total.new_zeros(()),
        }

    def auxiliary_reconstruction(
        self,
        out: dict[str, torch.Tensor],
        dead_mask: torch.Tensor,
        k_aux: int,
        sample_index: torch.Tensor,
    ) -> torch.Tensor:
        del sample_index
        h_masked = out["h_token"] * dead_mask.to(out["h_token"].dtype)
        values, indices = torch.topk(
            torch.relu(h_masked),
            k=min(k_aux, self.n_total),
            dim=1,
        )
        sparse = torch.zeros_like(h_masked).scatter_(1, indices, values)
        return F.linear(sparse, self.decoder_weight)

    def forward(
        self,
        x_norm: torch.Tensor,
        sample_index: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del sample_index, lengths
        centered = x_norm - self.b_pre
        h = F.linear(centered, self.encoder_weight, self.encoder_bias)
        z = self.batchtopk_sparse(h, self.top_k)
        recon = F.linear(z, self.decoder_weight) + self.b_pre
        return {
            "recon": recon,
            "h_token": h,
            "h_semantic": h.new_zeros((0, 0)),
            "z_token": z,
            "z_semantic": z.new_zeros((0, 0)),
        }

    def export_state(self, threshold: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "b_pre": self.b_pre.detach().cpu(),
            "encoder.weight": self.encoder_weight.detach().cpu(),
            "encoder.bias": self.encoder_bias.detach().cpu(),
            "decoder.weight": self.decoder_weight.detach().cpu(),
            "threshold": threshold.detach().float().cpu(),
        }


class SampleNestedBatchTopKSAE(BatchTopKFromScratchSAE):
    """Shared dictionary with a learnable inner sample-mean partition."""

    def __init__(
        self,
        b_pre: torch.Tensor,
        n_latents: int,
        inner_features: int,
        top_k: int,
        seed: int,
        nested_loss_weight: float,
    ) -> None:
        if inner_features <= 0 or inner_features >= n_latents:
            raise ValueError("inner_features must lie strictly inside the dictionary")
        if nested_loss_weight <= 0:
            raise ValueError("nested_loss_weight must be positive")
        super().__init__(
            b_pre=b_pre,
            n_latents=n_latents,
            top_k=top_k,
            seed=seed,
        )
        self.inner_features = int(inner_features)
        self.nested_loss_weight = float(nested_loss_weight)
        self.module_probe_width = min(256, self.inner_features)

    def module_snapshot(self) -> list[torch.Tensor]:
        width = self.module_probe_width
        return [
            self.encoder_weight[:width].detach().cpu().clone(),
            self.encoder_bias[: self.inner_features].detach().cpu().clone(),
            self.decoder_weight[:, :width].detach().cpu().clone(),
        ]

    def module_grad_norm(self) -> float:
        width = self.module_probe_width
        tensors = [
            (
                None
                if self.encoder_weight.grad is None
                else self.encoder_weight.grad[:width]
            ),
            (
                None
                if self.encoder_bias.grad is None
                else self.encoder_bias.grad[: self.inner_features]
            ),
            (
                None
                if self.decoder_weight.grad is None
                else self.decoder_weight.grad[:, :width]
            ),
        ]
        total = sum(
            float(tensor.detach().float().square().sum().item())
            for tensor in tensors
            if tensor is not None
        )
        return math.sqrt(total)

    def module_parameter_delta(
        self,
        initial: list[torch.Tensor],
    ) -> float:
        if len(initial) != 3:
            raise ValueError("Nested module snapshot must contain three tensors")
        width = self.module_probe_width
        current = [
            self.encoder_weight[:width].detach().cpu(),
            self.encoder_bias[: self.inner_features].detach().cpu(),
            self.decoder_weight[:, :width].detach().cpu(),
        ]
        return max(
            float((value - reference).abs().max().item())
            for value, reference in zip(current, initial, strict=True)
        )

    def l1_per_token(
        self,
        out: dict[str, torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        token_l1 = out["z_token"].sum() / lengths.sum()
        inner_l1 = (
            out["z_semantic"].sum(dim=1)
            * lengths.to(out["z_semantic"].dtype)
        ).sum() / lengths.sum()
        return token_l1 + inner_l1

    def active_features(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        active = (out["z_token"] > 0).any(dim=0)
        active[: self.inner_features] |= (
            out["z_semantic"] > 0
        ).any(dim=0)
        return active

    def reconstruction_losses(
        self,
        out: dict[str, torch.Tensor],
        x_norm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token = F.mse_loss(out["recon"].float(), x_norm.float())
        per_sample = (
            out["semantic_recon_component"].float()
            - out["semantic_target"].float()
        ).square().mean(dim=1)
        inner = (per_sample * lengths.float()).sum() / lengths.sum().float()
        return {
            "total": token + self.nested_loss_weight * inner,
            "token": token,
            "semantic": inner,
        }

    def forward(
        self,
        x_norm: torch.Tensor,
        sample_index: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        centered = x_norm - self.b_pre
        h = F.linear(centered, self.encoder_weight, self.encoder_bias)
        z = self.batchtopk_sparse(h, self.top_k)
        recon = F.linear(z, self.decoder_weight) + self.b_pre

        sample_count = int(lengths.numel())
        pooled = torch.zeros(
            (sample_count, centered.shape[1]),
            device=centered.device,
            dtype=centered.dtype,
        )
        pooled.index_add_(0, sample_index, centered)
        pooled = pooled / lengths.to(centered.dtype).unsqueeze(1)
        inner_h = F.linear(
            pooled,
            self.encoder_weight[: self.inner_features],
            self.encoder_bias[: self.inner_features],
        )
        inner_z = self.batchtopk_sparse(inner_h, self.top_k)
        inner_recon = F.linear(
            inner_z,
            self.decoder_weight[:, : self.inner_features],
        )
        return {
            "recon": recon,
            "h_token": h,
            "h_semantic": inner_h,
            "z_token": z,
            "z_semantic": inner_z,
            "semantic_target": pooled,
            "semantic_recon_component": inner_recon,
        }


def evaluate_validation(
    model: BatchTopKFromScratchSAE | SampleNestedBatchTopKSAE,
    cache: StructuredActivationCache,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rec_sum = 0.0
    squared_sum = 0.0
    token_count = 0
    inner_squared_sum = 0.0
    inner_element_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(
            cache.iter_batches(0, "validation")
        ):
            if (
                args.max_validation_batches > 0
                and batch_index >= args.max_validation_batches
            ):
                break
            x = normalize_activation(
                batch.activations,
                args.normalize_eps,
            ).to(device)
            sample_index = batch.sample_index.to(device)
            lengths = batch.lengths.to(device)
            out = model(x, sample_index, lengths)
            residual = out["recon"].float() - x.float()
            rec_sum += float(residual.square().sum().item())
            squared_sum += float(x.float().square().sum().item())
            token_count += int(x.shape[0])
            if out["z_semantic"].numel():
                inner_residual = (
                    out["semantic_recon_component"].float()
                    - out["semantic_target"].float()
                )
                inner_squared_sum += float(
                    inner_residual.square().sum().item()
                )
                inner_element_count += int(inner_residual.numel())
    model.train()
    result = {
        "validation_mse": rec_sum / (token_count * cache.d_model),
        "validation_explained_variance": 1.0 - rec_sum / squared_sum,
        "validation_tokens": float(token_count),
    }
    if inner_element_count:
        result["validation_inner_mean_mse"] = (
            inner_squared_sum / inner_element_count
        )
        result["validation_inner_mean_samples"] = float(
            inner_element_count // cache.d_model
        )
    return result


def calibrate_threshold(
    model: BatchTopKFromScratchSAE | SampleNestedBatchTopKSAE,
    cache: StructuredActivationCache,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    thresholds: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(cache.iter_batches(0, "train")):
            if batch_index >= args.threshold_calibration_batches:
                break
            x = normalize_activation(
                batch.activations,
                args.normalize_eps,
            ).to(device)
            h = torch.relu(
                F.linear(
                    x - model.b_pre,
                    model.encoder_weight,
                    model.encoder_bias,
                )
            )
            keep = min(model.top_k * h.shape[0], h.numel())
            threshold = torch.topk(h.reshape(-1), k=keep, dim=0).values[-1]
            thresholds.append(threshold.detach().float().cpu())
    model.train()
    if not thresholds:
        raise RuntimeError("Threshold calibration produced no batches")
    return torch.stack(thresholds).median()


def train_one(
    label: str,
    model: BatchTopKFromScratchSAE | SampleNestedBatchTopKSAE,
    cache: StructuredActivationCache,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
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
        T_max=args.epochs,
        eta_min=args.lr / 5,
    )
    module_initial = model.module_snapshot()
    history: list[dict[str, float]] = []
    global_step = 0
    started = time.time()
    dead_steps_threshold = (
        args.dead_steps_threshold
        if args.dead_steps_threshold > 0
        else cache.batch_count("train") + 1
    )
    latent_last_nonzero = torch.zeros(
        model.n_total,
        dtype=torch.long,
        device=device,
    )

    for epoch in range(args.epochs):
        accum = {
            "loss": 0.0,
            "recon": 0.0,
            "token_recon": 0.0,
            "inner_mean_recon": 0.0,
            "aux": 0.0,
            "l1": 0.0,
            "ev": 0.0,
            "active_token": 0.0,
            "active_inner_mean": 0.0,
            "dead_ratio": 0.0,
            "base_grad": 0.0,
            "nested_module_grad": 0.0,
            "tokens": 0.0,
        }
        batch_count = 0
        for batch_index, batch in enumerate(cache.iter_batches(epoch, "train")):
            if (
                args.max_train_batches > 0
                and batch_index >= args.max_train_batches
            ):
                break
            x = normalize_activation(
                batch.activations,
                args.normalize_eps,
            ).to(device)
            sample_index = batch.sample_index.to(device)
            lengths = batch.lengths.to(device)
            out = model(x, sample_index, lengths)
            reconstruction = model.reconstruction_losses(out, x, lengths)
            rec_loss = reconstruction["total"]
            token_rec_loss = reconstruction["token"]
            inner_mean_rec_loss = reconstruction["semantic"]
            l1 = model.l1_per_token(out, lengths).float()
            dead_mask = latent_last_nonzero > dead_steps_threshold
            dead_count = int(dead_mask.sum().item())
            if dead_count >= args.k_aux:
                residual_target = x.float() - out["recon"].detach().float()
                auxiliary_recon = model.auxiliary_reconstruction(
                    out,
                    dead_mask,
                    args.k_aux,
                    sample_index,
                )
                aux_loss = F.mse_loss(
                    auxiliary_recon.float(),
                    residual_target,
                )
            else:
                aux_loss = rec_loss.new_zeros(())
            loss = (
                rec_loss
                + args.aux_loss_coeff * aux_loss
                + args.l1_coeff * l1
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{label} produced NaN/Inf at epoch={epoch + 1} step={global_step + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            base_grad = grad_norm(model.base_parameters())
            nested_module_grad = model.module_grad_norm()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            model.project_decoder_grads()
            optimizer.step()
            model.normalize_decoder()
            active_features = model.active_features(out).detach()
            latent_last_nonzero.mul_((~active_features).to(torch.long))
            latent_last_nonzero.add_(1)

            residual = out["recon"].detach().float() - x.float()
            ev = 1.0 - residual.square().sum() / x.float().square().sum()
            accum["loss"] += float(loss.detach().item())
            accum["recon"] += float(rec_loss.detach().item())
            accum["token_recon"] += float(token_rec_loss.detach().item())
            accum["inner_mean_recon"] += float(
                inner_mean_rec_loss.detach().item()
            )
            accum["aux"] += float(aux_loss.detach().item())
            accum["l1"] += float(l1.detach().item())
            accum["ev"] += float(ev.item())
            accum["active_token"] += float(
                (out["z_token"].detach() > 0).float().sum(dim=1).mean().item()
            )
            if out["z_semantic"].numel():
                accum["active_inner_mean"] += float(
                    (out["z_semantic"].detach() > 0)
                    .float()
                    .sum(dim=1)
                    .mean()
                    .item()
                )
            accum["dead_ratio"] += dead_count / model.n_total
            accum["base_grad"] += base_grad
            accum["nested_module_grad"] += nested_module_grad
            accum["tokens"] += float(x.shape[0])
            batch_count += 1
            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "label": label,
                            "epoch": epoch + 1,
                            "step": global_step,
                            "recon": float(rec_loss.detach().item()),
                            "token_recon": float(
                                token_rec_loss.detach().item()
                            ),
                            "inner_mean_recon": float(
                                inner_mean_rec_loss.detach().item()
                            ),
                            "aux": float(aux_loss.detach().item()),
                            "l1": float(l1.detach().item()),
                            "ev": float(ev.item()),
                            "dead_ratio": dead_count / model.n_total,
                            "dead_steps_threshold": dead_steps_threshold,
                            "base_grad_norm": base_grad,
                            "nested_module_grad_norm": nested_module_grad,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "elapsed_seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
            del (
                x,
                sample_index,
                lengths,
                out,
                reconstruction,
                rec_loss,
                token_rec_loss,
                inner_mean_rec_loss,
                aux_loss,
                l1,
                loss,
                residual,
                ev,
                active_features,
            )
        scheduler.step()
        row = {
            "epoch": float(epoch + 1),
            "steps": float(batch_count),
            **{
                key: value / max(batch_count, 1)
                for key, value in accum.items()
                if key != "tokens"
            },
            "tokens": accum["tokens"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps({"event": "epoch_complete", "label": label, **row}), flush=True)

    validation = evaluate_validation(model, cache, args, device)
    return {
        "history": history,
        "validation": validation,
        "global_steps": global_step,
        "elapsed_seconds": time.time() - started,
        "dead_steps_threshold": dead_steps_threshold,
        "nested_module_parameter_probe_max_delta": (
            model.module_parameter_delta(module_initial)
        ),
        "parameter_count": parameter_count(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--n-total", type=int, default=65_536)
    parser.add_argument("--inner-features", type=int, default=32_768)
    parser.add_argument(
        "--readout-mode",
        choices=["all", "outer"],
        default="all",
    )
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--nested-loss-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-samples", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument("--threshold-calibration-batches", type=int, default=128)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=420_396)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--beta1", type=float, default=0.85)
    parser.add_argument("--beta2", type=float, default=0.9999)
    parser.add_argument("--optimizer-eps", type=float, default=6.25e-10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-coeff", type=float, default=0.0)
    parser.add_argument("--k-aux", type=int, default=2_048)
    parser.add_argument("--aux-loss-coeff", type=float, default=1.0 / 32.0)
    parser.add_argument(
        "--dead-steps-threshold",
        type=int,
        default=0,
        help="Use <=0 to match one full training epoch plus one step.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    cache = StructuredActivationCache(
        cache_dir=args.cache_dir,
        layer=args.layer,
        batch_samples=args.batch_samples,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b_pre = cache.mean()

    base = BatchTopKFromScratchSAE(
        b_pre=b_pre,
        n_latents=args.n_total,
        top_k=args.top_k,
        seed=args.initialization_seed,
    )
    base_result = train_one(
        "structured BatchTopK base",
        base,
        cache,
        args,
        device,
    )
    base_threshold = calibrate_threshold(base, cache, args, device)
    base_result["threshold"] = float(base_threshold.item())
    base_path = args.output_dir / "trained_sae-structured-batchtopk-base.pt"
    torch.save(base.export_state(base_threshold), base_path)
    base_params = int(base_result["parameter_count"])
    base.to("cpu")
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(args.seed)
    candidate = SampleNestedBatchTopKSAE(
        b_pre=b_pre,
        n_latents=args.n_total,
        inner_features=args.inner_features,
        top_k=args.top_k,
        seed=args.initialization_seed,
        nested_loss_weight=args.nested_loss_weight,
    )
    candidate_result = train_one(
        "structured sample-nested BatchTopK candidate",
        candidate,
        cache,
        args,
        device,
    )
    candidate_threshold = calibrate_threshold(candidate, cache, args, device)
    candidate_result["threshold"] = float(candidate_threshold.item())
    candidate_path = (
        args.output_dir
        / "trained_sae-structured-sample-nested-batchtopk.pt"
    )
    torch.save(candidate.export_state(candidate_threshold), candidate_path)

    candidate_params = int(candidate_result["parameter_count"])
    if base_params != candidate_params:
        raise RuntimeError(
            f"Parameter mismatch: base={base_params}, candidate={candidate_params}"
        )
    feature_start = 0
    feature_end = args.n_total
    if args.readout_mode == "outer":
        feature_start = args.inner_features
    exposed_feature_count = feature_end - feature_start
    targets = [
        {
            "label": (
                "structured-cache BatchTopK outer-readout control"
                if args.readout_mode == "outer"
                else "structured-cache BatchTopK base-only"
            ),
            "kind": "batchtopk",
            "layer": args.layer,
            "checkpoint": str(base_path),
            "variant_key": "base",
            "top_k": args.top_k,
            "feature_start": feature_start,
            "feature_end": feature_end,
        },
        {
            "label": (
                "structured-cache auxiliary-absorber BatchTopK SAE"
                if args.readout_mode == "outer"
                else "structured-cache sample-nested BatchTopK SAE"
            ),
            "kind": "batchtopk",
            "layer": args.layer,
            "checkpoint": str(candidate_path),
            "variant_key": "candidate",
            "top_k": args.top_k,
            "feature_start": feature_start,
            "feature_end": feature_end,
        },
    ]
    targets_path = (
        args.output_dir / "targets-structured-sample-nested-batchtopk.json"
    )
    targets_path.write_text(
        json.dumps(targets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment": (
            "parameter-matched structured-cache auxiliary-absorber BatchTopK SAE"
            if args.readout_mode == "outer"
            else "parameter-matched structured-cache nested BatchTopK SAE"
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cache_manifest_sha256": cache.fingerprint(),
        "cache_read_only": True,
        "base_parameter_count": base_params,
        "candidate_parameter_count": candidate_params,
        "parameter_matched": base_params == candidate_params,
        "exposed_feature_count": exposed_feature_count,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "candidate_checkpoint": str(candidate_path),
        "targets_json": str(targets_path),
        "base_result": base_result,
        "candidate_result": candidate_result,
        "fairness": {
            "same_initial_parameter_tensors": True,
            "same_cache": True,
            "same_data_order": True,
            "same_optimizer": True,
            "same_epochs": True,
            "same_exposed_feature_count": True,
            "readout_mode": args.readout_mode,
            "readout_feature_start": feature_start,
            "readout_feature_end": feature_end,
            "inner_partition_hidden_from_sparse_readout": (
                args.readout_mode == "outer"
            ),
            "matches_batchtopk_orthogonal_initialization": True,
            "matches_batchtopk_joint_decoder_constraint": True,
            "matches_batchtopk_dead_feature_auxiliary_objective": True,
            "same_batchtopk_k": True,
            "top_k": args.top_k,
            "same_threshold_calibration_protocol": True,
            "threshold_calibration_batches": args.threshold_calibration_batches,
            "shared_encoder_decoder_dictionary": True,
            "full_dictionary_token_reconstruction": True,
            "inner_partition_sample_mean_reconstruction": True,
            "inner_features": args.inner_features,
            "nested_loss_weight": args.nested_loss_weight,
            "l1_training_weight": args.l1_coeff,
            "uses_saebench_labels_for_training": False,
            "uses_eval_split_for_training": False,
            "uses_mean_diff_selection_for_training": False,
            "uses_test_feedback_for_training": False,
            "base_and_candidate_trained_from_scratch": True,
        },
    }
    summary_path = (
        args.output_dir / "train-summary-structured-sample-nested-batchtopk.json"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "training_complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
