#!/usr/bin/env python3
"""Measure whether a trained structured semantic branch remains active."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_structured_dual_granularity_sae import (
    StructuredActivationCache,
    normalize_activation,
)


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.float()
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--batch-samples", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=32)
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
    state = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    device = torch.device(args.device)
    b_pre = state["b_pre"].to(device)
    encoder_weight = state["semantic_encoder.weight"].to(device)
    encoder_bias = state["semantic_encoder.bias"].to(device)
    decoder_weight = state["semantic_decoder.weight"].to(device)

    preactivations = []
    active_feature_mask = torch.zeros(
        encoder_weight.shape[0],
        dtype=torch.bool,
        device=device,
    )
    active_per_sample_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(cache.iter_batches(0, "validation")):
            x = normalize_activation(
                batch.activations,
                args.normalize_eps,
            ).to(device)
            sample_index = batch.sample_index.to(device)
            lengths = batch.lengths.to(device)
            centered = x - b_pre
            pooled = torch.zeros(
                (int(lengths.numel()), centered.shape[1]),
                device=device,
                dtype=centered.dtype,
            )
            pooled.index_add_(0, sample_index, centered)
            pooled = pooled / lengths.to(centered.dtype).unsqueeze(1)
            hidden = F.linear(pooled, encoder_weight, encoder_bias)
            active = hidden > 0
            preactivations.append(hidden.float().cpu())
            active_feature_mask |= active.any(dim=0)
            active_per_sample_sum += float(active.sum().item())
            sample_count += int(active.shape[0])
            if batch_index + 1 >= args.max_batches:
                break

    hidden_all = torch.cat(preactivations, dim=0)
    positive = hidden_all > 0
    report = {
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_manifest_sha256": cache.fingerprint(),
        "checkpoint": str(args.checkpoint.resolve()),
        "layer": args.layer,
        "validation_samples": sample_count,
        "semantic_features": int(encoder_weight.shape[0]),
        "semantic_bias": {
            **tensor_stats(encoder_bias),
            "positive_fraction": float(
                (encoder_bias > 0).float().mean().item()
            ),
        },
        "semantic_encoder_row_norm": tensor_stats(
            encoder_weight.float().norm(dim=1)
        ),
        "semantic_decoder_column_norm": tensor_stats(
            decoder_weight.float().norm(dim=0)
        ),
        "semantic_preactivation": {
            **tensor_stats(hidden_all),
            "p99": float(torch.quantile(hidden_all, 0.99).item()),
            "positive_fraction": float(positive.float().mean().item()),
            "active_feature_count": int(active_feature_mask.sum().item()),
            "active_per_sample": active_per_sample_sum / max(sample_count, 1),
        },
    }
    report["branch_dead"] = (
        report["semantic_preactivation"]["positive_fraction"] < 1.0e-4
        and report["semantic_preactivation"]["active_per_sample"] < 1.0
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
