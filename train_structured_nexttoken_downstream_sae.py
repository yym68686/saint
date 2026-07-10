#!/usr/bin/env python3
"""Train matched V396 SAEs with a true downstream next-token objective."""

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
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from capture_activations import load_model
from llama_3.args import ModelArgs


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


def max_parameter_delta(
    parameters: list[nn.Parameter],
    initial: list[torch.Tensor],
) -> float:
    return max(
        (
            float((parameter.detach().cpu() - reference).abs().max().item())
            for parameter, reference in zip(parameters, initial, strict=True)
        ),
        default=0.0,
    )


def normalize_with_stats(
    x: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    scale = x.std(dim=-1, keepdim=True) + eps
    return (x - mean) / scale, mean, scale


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


@dataclass(frozen=True)
class SequenceBatch:
    token_ids: torch.Tensor
    cached_source: torch.Tensor
    sample_ids: torch.Tensor


class StructuredSequenceCache:
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
        if self.manifest.get("status") != "complete":
            raise ValueError("Structured cache is incomplete")
        if not self.manifest.get("summary", {}).get("read_only_finalized"):
            raise ValueError("Structured cache was not finalized read-only")
        if self.cache_dir.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise PermissionError(f"Cache directory must be read-only: {self.cache_dir}")
        layers = set(map(int, self.manifest["configuration"]["layers"]))
        if self.layer not in layers:
            raise ValueError(f"Layer {self.layer} is absent from the cache")
        self.sample_count = int(self.manifest["summary"]["sample_count"])
        self.train_cutoff = int(self.sample_count * self.train_fraction)
        self.shards = list(self.manifest["shards"])

    def fingerprint(self) -> str:
        return sha256_file(self.manifest_path)

    def iter_batches(
        self,
        batch_sequences: int,
        sequence_length: int,
    ) -> Iterator[SequenceBatch]:
        if batch_sequences < 2:
            raise ValueError("batch_sequences must be at least two for derangement")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        epoch = 0
        while True:
            shard_order = list(range(len(self.shards)))
            random.Random(self.seed + epoch * 1_000_003).shuffle(shard_order)
            pending_tokens: list[torch.Tensor] = []
            pending_sources: list[torch.Tensor] = []
            pending_samples: list[int] = []
            for shard_position in shard_order:
                entry = self.shards[shard_position]
                meta = torch.load(
                    self.cache_dir / entry["meta"]["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                sources = torch.load(
                    self.cache_dir / entry["layers"][str(self.layer)]["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                lengths = meta["lengths"].to(torch.int64)
                sample_ids = meta["sample_ids"].to(torch.int64)
                eligible = torch.nonzero(
                    (sample_ids < self.train_cutoff) & (lengths >= sequence_length),
                    as_tuple=False,
                ).flatten()
                generator = torch.Generator(device="cpu").manual_seed(
                    self.seed + epoch * 1_000_003 + shard_position * 10_007
                )
                eligible = eligible[
                    torch.randperm(eligible.numel(), generator=generator)
                ]
                offsets = meta["offsets"].to(torch.int64)
                tokens = meta["token_ids"].to(torch.int64)
                for index_tensor in eligible:
                    index = int(index_tensor.item())
                    sample_id = int(sample_ids[index].item())
                    # Cached activations include the complete causal prefix. Starting at
                    # zero lets the frozen model reproduce them exactly without adding
                    # uncached prefix state to the training protocol.
                    local_start = 0
                    packed_start = int(offsets[index].item()) + local_start
                    pending_tokens.append(
                        tokens[index, local_start : local_start + sequence_length].clone()
                    )
                    pending_sources.append(
                        sources[packed_start : packed_start + sequence_length].clone()
                    )
                    pending_samples.append(sample_id)
                    if len(pending_tokens) == batch_sequences:
                        yield SequenceBatch(
                            token_ids=torch.stack(pending_tokens),
                            cached_source=torch.stack(pending_sources),
                            sample_ids=torch.tensor(pending_samples, dtype=torch.int64),
                        )
                        pending_tokens = []
                        pending_sources = []
                        pending_samples = []
            epoch += 1


class DownstreamNextTokenV396SAE(nn.Module):
    def __init__(
        self,
        base_state: dict[str, torch.Tensor | float],
        context_rank: int,
        context_seed: int,
        max_context_log_gain: float,
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
        self.context_rank = int(context_rank)
        self.max_context_log_gain = float(max_context_log_gain)
        d_model = int(self.b_pre.numel())
        n_latents = int(self.encoder_weight.shape[0])
        generator = torch.Generator(device="cpu").manual_seed(context_seed)
        down = torch.empty((self.context_rank, d_model), dtype=torch.float32)
        nn.init.xavier_uniform_(down, generator=generator)
        self.context_down_weight = nn.Parameter(down)
        self.context_down_bias = nn.Parameter(
            torch.zeros(self.context_rank, dtype=torch.float32)
        )
        # A zero output projection makes every variant exactly equal to V396 at step zero.
        self.context_up_weight = nn.Parameter(
            torch.zeros((n_latents, self.context_rank), dtype=torch.float32)
        )
        self.context_up_bias = nn.Parameter(torch.zeros(n_latents, dtype=torch.float32))
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

    def context_parameters(self) -> list[nn.Parameter]:
        return [
            self.context_down_weight,
            self.context_down_bias,
            self.context_up_weight,
            self.context_up_bias,
        ]

    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta.float()).clamp(1.0e-4, self.max_beta)

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder_weight.div_(
                self.decoder_weight.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
            )

    def encode(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = x_norm - self.b_pre
        h = F.linear(centered, self.encoder_weight, self.encoder_bias)
        u = torch.relu(h).float()
        beta = self.beta().to(u.device)
        z = torch.log1p(beta.unsqueeze(0) * u) / torch.log1p(beta).unsqueeze(0)
        base_gain = self.log_gain.clamp(
            -self.max_log_gain,
            self.max_log_gain,
        ).exp().to(z.device)
        z = z * base_gain.unsqueeze(0)
        context_input = F.layer_norm(x_norm.float(), (x_norm.shape[-1],))
        context_hidden = F.silu(
            F.linear(
                context_input,
                self.context_down_weight,
                self.context_down_bias,
            )
        )
        context_gain = self.max_context_log_gain * torch.tanh(
            F.linear(
                context_hidden,
                self.context_up_weight,
                self.context_up_bias,
            )
            / math.sqrt(self.context_rank)
        )
        z = z * context_gain.exp()
        return z, context_gain

    def forward(self, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        original_shape = x_norm.shape
        flat = x_norm.reshape(-1, original_shape[-1])
        z, context_gain = self.encode(flat)
        recon = F.linear(z, self.decoder_weight) + self.b_pre
        return {
            "z": z.reshape(*original_shape[:-1], -1),
            "recon": recon.reshape(original_shape),
            "context_gain": context_gain.reshape(*original_shape[:-1], -1),
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
            "downstream.context_down_weight": self.context_down_weight.detach().cpu(),
            "downstream.context_down_bias": self.context_down_bias.detach().cpu(),
            "downstream.context_up_weight": self.context_up_weight.detach().cpu(),
            "downstream.context_up_bias": self.context_up_bias.detach().cpu(),
            "downstream.context_rank": torch.tensor(self.context_rank),
            "downstream.max_context_log_gain": torch.tensor(
                self.max_context_log_gain
            ),
        }


def causal_mask(
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if sequence_length <= 1:
        return None
    mask = torch.full(
        (sequence_length, sequence_length),
        float("-inf"),
        device=device,
    )
    return torch.triu(mask, diagonal=1).to(dtype=dtype)


def detach_attention_caches(model: nn.Module, start_layer: int = 0) -> None:
    for layer_index in range(start_layer, len(model.layers)):
        attention = model.layers[layer_index].attention
        attention.cache_k = attention.cache_k.detach()
        attention.cache_v = attention.cache_v.detach()


def forward_to_source(
    model: nn.Module,
    tokens: torch.Tensor,
    source_layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    sequence_length = int(tokens.shape[1])
    hidden = model.tok_embeddings(tokens)
    model.freqs_cis = model.freqs_cis.to(hidden.device)
    frequencies = model.freqs_cis[:sequence_length]
    mask = causal_mask(sequence_length, hidden.device, hidden.dtype)
    detach_attention_caches(model, 0)
    with torch.no_grad():
        for layer_index in range(source_layer):
            hidden = model.layers[layer_index](hidden, 0, frequencies, mask)
    return hidden.detach(), frequencies, mask


def forward_from_reconstructed_source(
    model: nn.Module,
    hidden_at_source: torch.Tensor,
    reconstructed_normalized_source: torch.Tensor,
    frequencies: torch.Tensor,
    mask: torch.Tensor | None,
    source_layer: int,
) -> torch.Tensor:
    detach_attention_caches(model, source_layer)
    source = model.layers[source_layer]
    hidden = hidden_at_source + source.attention(
        reconstructed_normalized_source,
        0,
        frequencies,
        mask,
    )
    hidden = hidden + source.feed_forward(source.ffn_norm(hidden))
    for layer_index in range(source_layer + 1, len(model.layers)):
        hidden = model.layers[layer_index](hidden, 0, frequencies, mask)
    return model.output(model.norm(hidden)).float()


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    downstream_loss: bool
    wrong_alignment: bool


VARIANTS = [
    Variant(
        key="reconstruction_control",
        label="same-parameter reconstruction-only contextual control",
        downstream_loss=False,
        wrong_alignment=False,
    ),
    Variant(
        key="wrong_alignment",
        label="same-parameter wrong-next-token downstream control",
        downstream_loss=True,
        wrong_alignment=True,
    ),
    Variant(
        key="candidate",
        label="true downstream next-token contextual V396 SAE",
        downstream_loss=True,
        wrong_alignment=False,
    ),
]


def train_variant(
    variant: Variant,
    model: DownstreamNextTokenV396SAE,
    frozen_lm: nn.Module,
    cache: StructuredSequenceCache,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.trunk_parameters(), "lr": args.trunk_lr},
            {"params": model.context_parameters(), "lr": args.context_lr},
        ],
        weight_decay=args.weight_decay,
    )
    trunk_initial = [p.detach().cpu().clone() for p in model.trunk_parameters()]
    context_initial = [p.detach().cpu().clone() for p in model.context_parameters()]
    batches = cache.iter_batches(args.batch_sequences, args.sequence_length)
    seen_features = torch.zeros(
        model.encoder_weight.shape[0],
        dtype=torch.bool,
        device=device,
    )
    logs: list[dict[str, float | str]] = []
    source_max_abs = 0.0
    wrong_fixed_pairs = 0
    wrong_same_sample_pairs = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        batch = next(batches)
        tokens = batch.token_ids.to(device, non_blocking=True)
        cached_source = batch.cached_source.to(device, non_blocking=True)
        hidden_at_source, frequencies, mask = forward_to_source(
            frozen_lm,
            tokens,
            args.source_layer,
        )
        with torch.no_grad():
            recomputed_source = frozen_lm.layers[args.source_layer].attention_norm(
                hidden_at_source
            )
            source_error = (
                recomputed_source.float() - cached_source.float()
            ).abs().max()
            source_max_abs = max(source_max_abs, float(source_error.item()))
            if float(source_error.item()) > args.source_reproduction_tolerance:
                raise RuntimeError(
                    "Frozen prefix does not reproduce the cached L22 source: "
                    f"max_abs={float(source_error.item()):.8f}, "
                    f"tolerance={args.source_reproduction_tolerance:.8f}"
                )
        x_norm, source_mean, source_scale = normalize_with_stats(
            cached_source,
            args.normalize_eps,
        )
        out = model(x_norm)
        reconstructed_source = (
            out["recon"].float() * source_scale + source_mean
        ).to(hidden_at_source.dtype)
        logits = forward_from_reconstructed_source(
            frozen_lm,
            hidden_at_source,
            reconstructed_source,
            frequencies,
            mask,
            args.source_layer,
        )
        true_targets = tokens[:, 1:].contiguous()
        optimization_targets = true_targets
        if variant.wrong_alignment:
            permutation = torch.roll(
                torch.arange(tokens.shape[0], device=device),
                shifts=1,
            )
            wrong_fixed_pairs += int(
                (permutation == torch.arange(tokens.shape[0], device=device)).sum().item()
            )
            wrong_same_sample_pairs += int(
                (
                    batch.sample_ids[permutation.cpu()]
                    == batch.sample_ids
                ).sum().item()
            )
            optimization_targets = true_targets[permutation]
        true_ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            true_targets.reshape(-1),
        )
        optimization_ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            optimization_targets.reshape(-1),
        )
        rec_loss = F.mse_loss(out["recon"].float(), x_norm.float())
        l1 = out["z"].float().mean()
        beta_anchor = (
            torch.log(model.beta()) - math.log(model.init_beta)
        ).square().mean()
        gain_l2 = model.log_gain.float().square().mean()
        context_l2 = out["context_gain"].float().square().mean()
        loss = (
            rec_loss
            + args.l1_coeff * l1
            + args.beta_anchor_coeff * beta_anchor
            + args.gain_l2_coeff * gain_l2
            + args.context_l2_coeff * context_l2
        )
        if variant.downstream_loss:
            loss = loss + args.downstream_loss_weight * optimization_ce
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{variant.key} produced NaN/Inf at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trunk_grad = grad_norm(model.trunk_parameters())
        context_grad = grad_norm(model.context_parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.normalize_decoder()
        with torch.no_grad():
            seen_features |= (out["z"].reshape(-1, out["z"].shape[-1]) > 0).any(dim=0)
            residual = out["recon"].float() - x_norm.float()
            ev = 1.0 - residual.square().sum() / x_norm.float().square().sum()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row: dict[str, float | str] = {
                "variant": variant.key,
                "step": float(step),
                "loss": float(loss.detach().item()),
                "reconstruction_loss": float(rec_loss.detach().item()),
                "true_next_token_ce": float(true_ce.detach().item()),
                "optimization_ce": float(optimization_ce.detach().item()),
                "l1": float(l1.detach().item()),
                "context_gain_rms": float(context_l2.detach().sqrt().item()),
                "explained_variance": float(ev.item()),
                "active_features_per_token": float(
                    (out["z"].detach() > 0).float().sum(dim=-1).mean().item()
                ),
                "dead_feature_ratio_seen": float((~seen_features).float().mean().item()),
                "trunk_grad_norm": trunk_grad,
                "context_grad_norm": context_grad,
                "source_reproduction_max_abs": source_max_abs,
                "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_bytes": float(torch.cuda.max_memory_allocated(device)),
            }
            print(json.dumps(row), flush=True)
            logs.append(row)
        detach_attention_caches(frozen_lm, 0)
        del (
            tokens,
            cached_source,
            hidden_at_source,
            recomputed_source,
            x_norm,
            source_mean,
            source_scale,
            out,
            reconstructed_source,
            logits,
            true_targets,
            optimization_targets,
            true_ce,
            optimization_ce,
            rec_loss,
            l1,
            beta_anchor,
            gain_l2,
            context_l2,
            loss,
        )
    return {
        "logs": logs,
        "elapsed_seconds": time.time() - started,
        "parameter_count": parameter_count(model),
        "trunk_parameter_max_delta": max_parameter_delta(
            model.trunk_parameters(), trunk_initial
        ),
        "context_parameter_max_delta": max_parameter_delta(
            model.context_parameters(), context_initial
        ),
        "source_reproduction_max_abs": source_max_abs,
        "wrong_fixed_pair_count": wrong_fixed_pairs,
        "wrong_same_sample_pair_count": wrong_same_sample_pairs,
        "final_dead_feature_ratio_seen": float((~seen_features).float().mean().item()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-layer", type=int, default=22)
    parser.add_argument("--context-rank", type=int, default=32)
    parser.add_argument("--context-seed", type=int, default=500_396)
    parser.add_argument("--max-context-log-gain", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-sequences", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--trunk-lr", type=float, default=1.0e-6)
    parser.add_argument("--context-lr", type=float, default=2.0e-5)
    parser.add_argument("--downstream-loss-weight", type=float, default=0.01)
    parser.add_argument("--l1-coeff", type=float, default=1.0e-6)
    parser.add_argument("--beta-anchor-coeff", type=float, default=1.0e-3)
    parser.add_argument("--gain-l2-coeff", type=float, default=1.0e-4)
    parser.add_argument("--context-l2-coeff", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--source-reproduction-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    cache = StructuredSequenceCache(
        args.cache_dir,
        args.source_layer,
        args.train_fraction,
        args.seed,
    )
    base_state = load_v396_state(args.base_checkpoint)
    params = ModelArgs(
        **json.loads((args.model_dir / "params.json").read_text(encoding="utf-8"))
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    frozen_lm = load_model(
        model_path=args.model_dir / "consolidated.00.pth",
        model_args=params,
        store_layer_activ=[],
        device=device,
        dtype=dtype,
    )
    frozen_lm.requires_grad_(False)
    frozen_lm.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets: list[dict[str, Any]] = [
        {
            "label": "frozen V396 warm-start reference",
            "kind": "v396_causal_learned_beta",
            "layer": args.source_layer,
            "checkpoint": str(args.base_checkpoint),
            "trainable_parameters": sum(
                int(value.numel())
                for value in base_state.values()
                if isinstance(value, torch.Tensor)
            ),
        }
    ]
    results: dict[str, Any] = {}
    parameter_counts: set[int] = set()
    initial_states: dict[str, dict[str, torch.Tensor]] = {}
    for variant in VARIANTS:
        set_seed(args.seed)
        model = DownstreamNextTokenV396SAE(
            base_state,
            args.context_rank,
            args.context_seed,
            args.max_context_log_gain,
        )
        initial_states[variant.key] = {
            key: value.clone() for key, value in model.state_dict().items()
        }
        result = train_variant(
            variant,
            model,
            frozen_lm,
            cache,
            args,
            device,
        )
        checkpoint = args.output_dir / f"trained_sae-{variant.key}-seed{args.seed}.pt"
        if args.save_checkpoints:
            torch.save(model.export_state(), checkpoint)
            targets.append(
                {
                    "label": variant.label,
                    "kind": "structured_nexttoken_downstream_v2",
                    "layer": args.source_layer,
                    "checkpoint": str(checkpoint),
                    "variant_key": variant.key,
                    "seed": args.seed,
                    "trainable_parameters": int(result["parameter_count"]),
                }
            )
        count = int(result["parameter_count"])
        parameter_counts.add(count)
        results[variant.key] = {
            "label": variant.label,
            "checkpoint": str(checkpoint) if args.save_checkpoints else None,
            "checkpoint_sha256": (
                sha256_file(checkpoint) if args.save_checkpoints else None
            ),
            "downstream_loss": variant.downstream_loss,
            "wrong_alignment": variant.wrong_alignment,
            **result,
        }
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
    if len(parameter_counts) != 1:
        raise RuntimeError(f"Variant parameter mismatch: {sorted(parameter_counts)}")
    reference_initial = initial_states[VARIANTS[0].key]
    for variant in VARIANTS[1:]:
        other = initial_states[variant.key]
        if reference_initial.keys() != other.keys() or any(
            not torch.equal(reference_initial[key], other[key])
            for key in reference_initial
        ):
            raise RuntimeError(f"Initial tensor mismatch for {variant.key}")
    targets_path = args.output_dir / "targets-nexttoken-downstream-v2.json"
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
        "experiment": "structured true downstream next-token contextual V396 SAE v2",
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
        "model_weight_sha256": sha256_file(
            args.model_dir / "consolidated.00.pth"
        ),
        "base_parameter_count": base_parameter_count,
        "trainable_parameter_count_each": next(iter(parameter_counts)),
        "new_context_parameter_count": (
            next(iter(parameter_counts)) - base_parameter_count
        ),
        "exposed_feature_count_each": int(base_state["encoder.weight"].shape[0]),
        "same_initial_tensors": True,
        "targets_json": str(targets_path) if args.save_checkpoints else None,
        "variants": results,
        "fairness": {
            "same_base_checkpoint": True,
            "same_initial_tensors": True,
            "same_trainable_parameter_count": True,
            "same_exposed_feature_count": True,
            "same_cache": True,
            "same_data_order": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_batch_sequences": True,
            "same_sequence_length": True,
            "same_target_marginal_candidate_and_wrong_control": True,
            "sae_trunk_unfrozen_in_all_variants": True,
            "context_module_trainable_in_all_variants": True,
            "downstream_lm_frozen": True,
            "uses_saebench_labels_for_training": False,
            "uses_eval_split_for_training": False,
            "uses_one_vs_rest_targets_for_training": False,
            "uses_mean_diff_selection_for_training": False,
            "uses_test_feedback_for_training": False,
        },
    }
    summary_path = args.output_dir / "train-summary-nexttoken-downstream-v2.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "training_complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
