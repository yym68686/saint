#!/usr/bin/env python3
"""Train an exactly parameter-matched ReLU and dual-granularity SAE pair."""

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
    # Match the existing ReLU SAE implementation exactly: the stored decoder
    # matrix is normalized and projected row-wise.
    decoder.div_(decoder.norm(dim=1, keepdim=True).clamp_min(1.0e-6))
    return encoder, encoder_bias, decoder


class ReLUFromScratchSAE(nn.Module):
    def __init__(
        self,
        b_pre: torch.Tensor,
        n_latents: int,
        seed: int,
    ) -> None:
        super().__init__()
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
        return F.linear(sparse, self.decoder_weight) + self.b_pre

    def forward(
        self,
        x_norm: torch.Tensor,
        sample_index: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del sample_index, lengths
        centered = x_norm - self.b_pre
        h = F.linear(centered, self.encoder_weight, self.encoder_bias)
        z = torch.relu(h)
        recon = F.linear(z, self.decoder_weight) + self.b_pre
        return {
            "recon": recon,
            "h_token": h,
            "h_semantic": h.new_zeros((0, 0)),
            "z_token": z,
            "z_semantic": z.new_zeros((0, 0)),
        }

    def export_state(self) -> dict[str, torch.Tensor]:
        return {
            "b_pre": self.b_pre.detach().cpu(),
            "encoder.weight": self.encoder_weight.detach().cpu(),
            "encoder.bias": self.encoder_bias.detach().cpu(),
            "decoder.weight": self.decoder_weight.detach().cpu(),
        }


class DualGranularitySAE(nn.Module):
    def __init__(
        self,
        b_pre: torch.Tensor,
        n_total: int,
        n_semantic: int,
        seed: int,
    ) -> None:
        super().__init__()
        if n_semantic <= 0 or n_semantic >= n_total:
            raise ValueError("n_semantic must lie strictly between 0 and n_total")
        d_model = int(b_pre.numel())
        encoder, encoder_bias, decoder = make_initial_tensors(
            n_total,
            d_model,
            seed,
        )
        self.n_total = int(n_total)
        self.n_semantic = int(n_semantic)
        self.n_token = self.n_total - self.n_semantic
        self.b_pre = nn.Parameter(b_pre.clone())
        self.token_encoder_weight = nn.Parameter(encoder[: self.n_token].clone())
        self.token_encoder_bias = nn.Parameter(
            encoder_bias[: self.n_token].clone()
        )
        self.token_decoder_weight = nn.Parameter(
            decoder[:, : self.n_token].clone()
        )
        self.semantic_encoder_weight = nn.Parameter(
            encoder[self.n_token :].clone()
        )
        self.semantic_encoder_bias = nn.Parameter(
            encoder_bias[self.n_token :].clone()
        )
        self.semantic_decoder_weight = nn.Parameter(
            decoder[:, self.n_token :].clone()
        )

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            row_norm = (
                self.token_decoder_weight.square().sum(dim=1, keepdim=True)
                + self.semantic_decoder_weight.square().sum(dim=1, keepdim=True)
            ).sqrt().clamp_min(1.0e-6)
            self.token_decoder_weight.div_(row_norm)
            self.semantic_decoder_weight.div_(row_norm)

    def project_decoder_grads(self) -> None:
        if (
            self.token_decoder_weight.grad is None
            or self.semantic_decoder_weight.grad is None
        ):
            return
        with torch.no_grad():
            projection = (
                (
                    self.token_decoder_weight
                    * self.token_decoder_weight.grad
                ).sum(dim=1, keepdim=True)
                + (
                    self.semantic_decoder_weight
                    * self.semantic_decoder_weight.grad
                ).sum(dim=1, keepdim=True)
            )
            self.token_decoder_weight.grad.sub_(
                projection * self.token_decoder_weight
            )
            self.semantic_decoder_weight.grad.sub_(
                projection * self.semantic_decoder_weight
            )

    def base_parameters(self) -> list[nn.Parameter]:
        return [
            self.b_pre,
            self.token_encoder_weight,
            self.token_encoder_bias,
            self.token_decoder_weight,
        ]

    def semantic_parameters(self) -> list[nn.Parameter]:
        return [
            self.semantic_encoder_weight,
            self.semantic_encoder_bias,
            self.semantic_decoder_weight,
        ]

    @staticmethod
    def l1_per_token(
        out: dict[str, torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        token_sum = out["z_token"].sum()
        semantic_sum = (
            out["z_semantic"].sum(dim=1)
            * lengths.to(out["z_semantic"].dtype)
        ).sum()
        return (token_sum + semantic_sum) / lengths.sum()

    @staticmethod
    def active_features(out: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                (out["z_token"] > 0).any(dim=0),
                (out["z_semantic"] > 0).any(dim=0),
            ]
        )

    def auxiliary_reconstruction(
        self,
        out: dict[str, torch.Tensor],
        dead_mask: torch.Tensor,
        k_aux: int,
        sample_index: torch.Tensor,
    ) -> torch.Tensor:
        semantic_per_token = out["h_semantic"][sample_index]
        h_all = torch.cat([out["h_token"], semantic_per_token], dim=1)
        h_masked = h_all * dead_mask.to(h_all.dtype)
        values, indices = torch.topk(
            torch.relu(h_masked),
            k=min(k_aux, self.n_total),
            dim=1,
        )
        sparse = torch.zeros_like(h_masked).scatter_(1, indices, values)
        token_sparse = sparse[:, : self.n_token]
        semantic_sparse = sparse[:, self.n_token :]
        return (
            F.linear(token_sparse, self.token_decoder_weight)
            + F.linear(semantic_sparse, self.semantic_decoder_weight)
            + self.b_pre
        )

    def forward(
        self,
        x_norm: torch.Tensor,
        sample_index: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        centered = x_norm - self.b_pre
        token_h = F.linear(
            centered,
            self.token_encoder_weight,
            self.token_encoder_bias,
        )
        z_token = torch.relu(token_h)
        sample_count = int(lengths.numel())
        pooled = torch.zeros(
            (sample_count, centered.shape[1]),
            device=centered.device,
            dtype=centered.dtype,
        )
        pooled.index_add_(0, sample_index, centered)
        pooled = pooled / lengths.to(centered.dtype).unsqueeze(1)
        semantic_h = F.linear(
            pooled,
            self.semantic_encoder_weight,
            self.semantic_encoder_bias,
        )
        z_semantic = torch.relu(semantic_h)
        token_recon = F.linear(z_token, self.token_decoder_weight)
        semantic_recon = F.linear(
            z_semantic,
            self.semantic_decoder_weight,
        )[sample_index]
        recon = token_recon + semantic_recon + self.b_pre
        return {
            "recon": recon,
            "h_token": token_h,
            "h_semantic": semantic_h,
            "z_token": z_token,
            "z_semantic": z_semantic,
        }

    def export_state(self) -> dict[str, torch.Tensor]:
        return {
            "structured.kind": torch.tensor(1),
            "structured.n_total": torch.tensor(self.n_total),
            "structured.n_token": torch.tensor(self.n_token),
            "structured.n_semantic": torch.tensor(self.n_semantic),
            "b_pre": self.b_pre.detach().cpu(),
            "token_encoder.weight": self.token_encoder_weight.detach().cpu(),
            "token_encoder.bias": self.token_encoder_bias.detach().cpu(),
            "token_decoder.weight": self.token_decoder_weight.detach().cpu(),
            "semantic_encoder.weight": self.semantic_encoder_weight.detach().cpu(),
            "semantic_encoder.bias": self.semantic_encoder_bias.detach().cpu(),
            "semantic_decoder.weight": self.semantic_decoder_weight.detach().cpu(),
        }


def evaluate_validation(
    model: ReLUFromScratchSAE | DualGranularitySAE,
    cache: StructuredActivationCache,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rec_sum = 0.0
    squared_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in cache.iter_batches(0, "validation"):
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
    model.train()
    return {
        "validation_mse": rec_sum / (token_count * cache.d_model),
        "validation_explained_variance": 1.0 - rec_sum / squared_sum,
        "validation_tokens": float(token_count),
    }


def train_one(
    label: str,
    model: ReLUFromScratchSAE | DualGranularitySAE,
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
    semantic_initial = [
        parameter.detach().cpu().clone()
        for parameter in model.semantic_parameters()
    ]
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
            "aux": 0.0,
            "l1": 0.0,
            "ev": 0.0,
            "active_token": 0.0,
            "active_semantic": 0.0,
            "dead_ratio": 0.0,
            "base_grad": 0.0,
            "semantic_grad": 0.0,
            "tokens": 0.0,
        }
        batch_count = 0
        for batch in cache.iter_batches(epoch, "train"):
            x = normalize_activation(
                batch.activations,
                args.normalize_eps,
            ).to(device)
            sample_index = batch.sample_index.to(device)
            lengths = batch.lengths.to(device)
            out = model(x, sample_index, lengths)
            rec_loss = F.mse_loss(out["recon"].float(), x.float())
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
            semantic_grad = grad_norm(model.semantic_parameters())
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
            accum["aux"] += float(aux_loss.detach().item())
            accum["l1"] += float(l1.detach().item())
            accum["ev"] += float(ev.item())
            accum["active_token"] += float(
                (out["z_token"].detach() > 0).float().sum(dim=1).mean().item()
            )
            if out["z_semantic"].numel():
                accum["active_semantic"] += float(
                    (out["z_semantic"].detach() > 0)
                    .float()
                    .sum(dim=1)
                    .mean()
                    .item()
                )
            accum["dead_ratio"] += dead_count / model.n_total
            accum["base_grad"] += base_grad
            accum["semantic_grad"] += semantic_grad
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
                            "aux": float(aux_loss.detach().item()),
                            "l1": float(l1.detach().item()),
                            "ev": float(ev.item()),
                            "dead_ratio": dead_count / model.n_total,
                            "dead_steps_threshold": dead_steps_threshold,
                            "base_grad_norm": base_grad,
                            "semantic_grad_norm": semantic_grad,
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
                rec_loss,
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
    semantic_deltas = [
        float(
            (parameter.detach().cpu() - initial)
            .abs()
            .max()
            .item()
        )
        for parameter, initial in zip(
            model.semantic_parameters(),
            semantic_initial,
            strict=True,
        )
    ]
    return {
        "history": history,
        "validation": validation,
        "global_steps": global_step,
        "elapsed_seconds": time.time() - started,
        "dead_steps_threshold": dead_steps_threshold,
        "semantic_parameter_max_delta": max(semantic_deltas, default=0.0),
        "parameter_count": parameter_count(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--n-total", type=int, default=65_536)
    parser.add_argument("--n-semantic", type=int, default=4_096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-samples", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=420_396)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--beta1", type=float, default=0.85)
    parser.add_argument("--beta2", type=float, default=0.9999)
    parser.add_argument("--optimizer-eps", type=float, default=6.25e-10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-coeff", type=float, default=1.0e-4)
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

    base = ReLUFromScratchSAE(
        b_pre=b_pre,
        n_latents=args.n_total,
        seed=args.initialization_seed,
    )
    base_result = train_one("structured ReLU base", base, cache, args, device)
    base_path = args.output_dir / "trained_sae-structured-relu-base.pt"
    torch.save(base.export_state(), base_path)
    base.to("cpu")
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(args.seed)
    candidate = DualGranularitySAE(
        b_pre=b_pre,
        n_total=args.n_total,
        n_semantic=args.n_semantic,
        seed=args.initialization_seed,
    )
    candidate_result = train_one(
        "structured dual-granularity candidate",
        candidate,
        cache,
        args,
        device,
    )
    candidate_path = args.output_dir / "trained_sae-structured-dual-granularity.pt"
    torch.save(candidate.export_state(), candidate_path)

    base_params = int(base_result["parameter_count"])
    candidate_params = int(candidate_result["parameter_count"])
    if base_params != candidate_params:
        raise RuntimeError(
            f"Parameter mismatch: base={base_params}, candidate={candidate_params}"
        )
    targets = [
        {
            "label": "structured-cache ReLU base-only",
            "kind": "relu",
            "layer": args.layer,
            "checkpoint": str(base_path),
            "variant_key": "base",
        },
        {
            "label": "structured-cache dual-granularity SAE",
            "kind": "structured_dual_granularity",
            "layer": args.layer,
            "checkpoint": str(candidate_path),
            "variant_key": "candidate",
        },
    ]
    targets_path = args.output_dir / "targets-structured-dual-granularity.json"
    targets_path.write_text(
        json.dumps(targets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment": "parameter-matched structured-cache dual-granularity SAE",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cache_manifest_sha256": cache.fingerprint(),
        "cache_read_only": True,
        "base_parameter_count": base_params,
        "candidate_parameter_count": candidate_params,
        "parameter_matched": base_params == candidate_params,
        "exposed_feature_count": args.n_total,
        "base_checkpoint": str(base_path),
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
            "matches_relu_orthogonal_initialization": True,
            "matches_relu_joint_decoder_constraint": True,
            "matches_relu_dead_feature_auxiliary_objective": True,
            "uses_saebench_labels_for_training": False,
            "uses_eval_split_for_training": False,
            "uses_mean_diff_selection_for_training": False,
            "uses_test_feedback_for_training": False,
        },
    }
    summary_path = args.output_dir / "train-summary-structured-dual-granularity.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "training_complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
