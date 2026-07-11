#!/usr/bin/env python3
"""Measure label-free geometry of averaged Jacobian transforms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def load_matrix(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    matrix = payload["jacobian"].float()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square Jacobian, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise FloatingPointError(f"Non-finite Jacobian: {path}")
    return matrix


def load_activations(cache_dir: Path, token_count: int) -> torch.Tensor:
    chunks = []
    remaining = token_count
    for path in sorted(cache_dir.glob("*.pt")):
        values = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise ValueError(f"Unexpected activation payload: {path}")
        take = min(remaining, int(values.shape[0]))
        chunks.append(values[:take].float())
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"Requested {token_count} tokens, missing {remaining}")
    return torch.cat(chunks, dim=0)


def quantiles(values: torch.Tensor) -> dict[str, float]:
    points = torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], device=values.device)
    result = torch.quantile(values.float(), points).cpu().tolist()
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p01", "p10", "p50", "p90", "p99", "max"),
            result,
            strict=True,
        )
    }


def rank_summary(singular_values: torch.Tensor) -> dict[str, float | dict[str, float]]:
    singular_values = singular_values.double().clamp_min(0)
    squared = singular_values.square()
    spectral = singular_values.max()
    frobenius = squared.sum().sqrt()
    probability = singular_values / singular_values.sum().clamp_min(1.0e-30)
    entropy = -(probability * probability.clamp_min(1.0e-30).log()).sum()
    return {
        "spectral_norm": float(spectral.item()),
        "frobenius_norm": float(frobenius.item()),
        "stable_rank": float((squared.sum() / spectral.square().clamp_min(1.0e-30)).item()),
        "entropy_effective_rank": float(entropy.exp().item()),
        "participation_rank": float(
            (squared.sum().square() / squared.square().sum().clamp_min(1.0e-30)).item()
        ),
        "top_10_energy_fraction": float(
            (squared[:10].sum() / squared.sum().clamp_min(1.0e-30)).item()
        ),
        "top_100_energy_fraction": float(
            (squared[:100].sum() / squared.sum().clamp_min(1.0e-30)).item()
        ),
        "singular_value_quantiles": quantiles(singular_values.float()),
    }


def activation_summary(
    source: torch.Tensor,
    transformed: torch.Tensor,
) -> dict[str, float | dict[str, float]]:
    source = source.float()
    transformed = transformed.float()
    centered = transformed - transformed.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    row_norms = transformed.norm(dim=1)
    unit_transformed = transformed / row_norms.unsqueeze(1).clamp_min(1.0e-30)
    unit_centered = unit_transformed - unit_transformed.mean(dim=0, keepdim=True)
    unit_singular_values = torch.linalg.svdvals(unit_centered)
    coordinate_std = centered.square().mean(dim=0).sqrt()
    cosine = F.cosine_similarity(source, transformed, dim=1)
    return {
        "row_norm_mean": float(row_norms.mean().item()),
        "row_norm_std": float(row_norms.std(unbiased=False).item()),
        "source_transform_cosine_mean": float(cosine.mean().item()),
        "source_transform_cosine_std": float(cosine.std(unbiased=False).item()),
        "coordinate_std_mean": float(coordinate_std.mean().item()),
        "coordinate_std_cv": float(
            (coordinate_std.std(unbiased=False) / coordinate_std.mean().clamp_min(1.0e-30)).item()
        ),
        "covariance_rank": rank_summary(singular_values),
        "row_unit_covariance_rank": rank_summary(unit_singular_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-jacobian", type=Path, required=True)
    parser.add_argument("--v2-jacobian", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-count", type=int, default=4096)
    parser.add_argument("--random-seed", type=int, default=42026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    v1_cpu = load_matrix(args.v1_jacobian)
    v2_cpu = load_matrix(args.v2_jacobian)
    if v1_cpu.shape != v2_cpu.shape:
        raise ValueError("Jacobian shapes differ")
    hidden_size = int(v1_cpu.shape[0])
    identity_cpu = torch.eye(hidden_size, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(args.random_seed)
    permutation = torch.randperm(hidden_size, generator=generator)
    signs = (torch.randint(0, 2, (hidden_size,), generator=generator) * 2 - 1).float()
    random_scale = float(v2_cpu.norm().item() / math.sqrt(hidden_size))

    transforms = {
        "identity": identity_cpu,
        "penultimate_jacobian_v1": v1_cpu,
        "final_jacobian_v2": v2_cpu,
    }
    source = load_activations(args.cache_dir, args.token_count).to(device)
    report: dict[str, object] = {
        "token_count": args.token_count,
        "hidden_size": hidden_size,
        "random_seed": args.random_seed,
        "random_scale": random_scale,
        "uses_labels": False,
        "uses_eval_split": False,
        "matrices": {},
        "activations": {},
    }
    for name, matrix_cpu in transforms.items():
        matrix = matrix_cpu.to(device)
        report["matrices"][name] = rank_summary(torch.linalg.svdvals(matrix))
        transformed = F.linear(source, matrix)
        report["activations"][name] = activation_summary(source, transformed)
        del matrix, transformed
        torch.cuda.empty_cache()

    random_transformed = source[:, permutation.to(device)] * signs.to(device) * random_scale
    report["activations"]["random_orthogonal_control"] = activation_summary(
        source,
        random_transformed,
    )
    matrix_cosine = F.cosine_similarity(v1_cpu.flatten(), v2_cpu.flatten(), dim=0)
    report["v1_v2"] = {
        "matrix_cosine": float(matrix_cosine.item()),
        "relative_delta_vs_v1": float((v2_cpu - v1_cpu).norm().item() / v1_cpu.norm().item()),
        "relative_delta_vs_v2": float((v2_cpu - v1_cpu).norm().item() / v2_cpu.norm().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
