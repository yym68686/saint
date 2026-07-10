#!/usr/bin/env python3
"""Measure label-free health and parameter drift of nested SAE partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from train_structured_nested_turn_sae import (
    StructuredActivationCache,
    normalize_activation,
)


def quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    levels = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    result = torch.quantile(values, levels)
    return {
        name: float(value.item())
        for name, value in zip(
            ("min", "p10", "median", "p90", "max"),
            result,
            strict=True,
        )
    }


def group_static_stats(
    state: dict[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, Any]:
    encoder = state["encoder.weight"][start:end].float()
    decoder = state["decoder.weight"][:, start:end].float()
    bias = state["encoder.bias"][start:end].float()
    return {
        "width": end - start,
        "encoder_row_norm": quantiles(encoder.norm(dim=1)),
        "decoder_column_norm": quantiles(decoder.norm(dim=0)),
        "encoder_bias": quantiles(bias),
        "encoder_bias_mean": float(bias.mean().item()),
    }


def activation_stats(
    checkpoint: Path,
    cache: StructuredActivationCache,
    inner_features: int,
    max_tokens: int,
    normalize_eps: float,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    width = int(state["encoder.weight"].shape[0])
    groups = {
        "prefix": (0, inner_features),
        "suffix": (inner_features, width),
    }
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "parameter_groups": {
            name: group_static_stats(state, start, end)
            for name, (start, end) in groups.items()
        },
    }
    b_pre = state["b_pre"].to(device=device, dtype=torch.bfloat16)
    encoder = state["encoder.weight"].to(
        device=device,
        dtype=torch.bfloat16,
    )
    encoder_bias = state["encoder.bias"].to(
        device=device,
        dtype=torch.bfloat16,
    )
    del state

    active_counts = torch.zeros(width, dtype=torch.int64)
    activation_sums = torch.zeros(width, dtype=torch.float64)
    tokens = 0
    with torch.inference_mode():
        for batch in cache.iter_batches(0, "validation"):
            remaining = max_tokens - tokens
            if remaining <= 0:
                break
            x = normalize_activation(
                batch.activations[:remaining],
                normalize_eps,
            ).to(device=device, dtype=torch.bfloat16)
            z = torch.relu(F.linear(x - b_pre, encoder, encoder_bias))
            active_counts += (z > 0).sum(dim=0).cpu()
            activation_sums += z.float().sum(dim=0).double().cpu()
            tokens += int(z.shape[0])
            del x, z

    group_results = {}
    for name, (start, end) in groups.items():
        local_active = active_counts[start:end]
        local_sums = activation_sums[start:end]
        active_rate = local_active.float() / tokens
        nonzero_count = int(local_active.sum().item())
        group_results[name] = {
            "sampled_tokens": tokens,
            "mean_l0_per_token": nonzero_count / tokens,
            "sampled_dead_fraction": float((local_active == 0).float().mean()),
            "feature_active_rate": quantiles(active_rate),
            "mean_activation_per_feature_per_token": float(
                (local_sums / tokens).mean().item()
            ),
            "mean_value_conditioned_on_active": (
                float(local_sums.sum().item() / nonzero_count)
                if nonzero_count
                else 0.0
            ),
        }
    result["activation_groups"] = group_results
    return result


def direction_drift(
    base_checkpoint: Path,
    candidate_checkpoint: Path,
    inner_features: int,
) -> dict[str, Any]:
    base = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    candidate = torch.load(
        candidate_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    width = int(base["encoder.weight"].shape[0])
    groups = {
        "prefix_or_inner": (0, inner_features),
        "suffix_or_outer": (inner_features, width),
    }
    result = {}
    for name, (start, end) in groups.items():
        base_encoder = base["encoder.weight"][start:end].float()
        candidate_encoder = candidate["encoder.weight"][start:end].float()
        encoder_cosine = F.cosine_similarity(
            base_encoder,
            candidate_encoder,
            dim=1,
        )
        base_decoder = base["decoder.weight"][:, start:end].float()
        candidate_decoder = candidate["decoder.weight"][:, start:end].float()
        decoder_cosine = F.cosine_similarity(
            base_decoder,
            candidate_decoder,
            dim=0,
        )
        bias_delta = (
            candidate["encoder.bias"][start:end].float()
            - base["encoder.bias"][start:end].float()
        )
        result[name] = {
            "encoder_direction_cosine": quantiles(encoder_cosine),
            "decoder_direction_cosine": quantiles(decoder_cosine),
            "encoder_bias_delta": quantiles(bias_delta),
            "encoder_bias_delta_abs_mean": float(bias_delta.abs().mean()),
        }
    result["b_pre_delta_l2"] = float(
        (
            candidate["b_pre"].float() - base["b_pre"].float()
        ).norm().item()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--inner-features", type=int, default=32_768)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--batch-samples", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cache = StructuredActivationCache(
        cache_dir=args.cache_dir,
        layer=args.layer,
        batch_samples=args.batch_samples,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    device = torch.device(args.device)
    report = {
        "label_free": True,
        "split": "structured OWT validation",
        "max_tokens": args.max_tokens,
        "base": activation_stats(
            args.base_checkpoint,
            cache,
            args.inner_features,
            args.max_tokens,
            args.normalize_eps,
            device,
        ),
        "candidate": activation_stats(
            args.candidate_checkpoint,
            cache,
            args.inner_features,
            args.max_tokens,
            args.normalize_eps,
            device,
        ),
        "candidate_vs_base_direction_drift": direction_drift(
            args.base_checkpoint,
            args.candidate_checkpoint,
            args.inner_features,
        ),
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
