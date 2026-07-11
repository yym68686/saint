#!/usr/bin/env python3
"""Fit the label-free contribution-mode split specification on OWT activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import stat
import time
from pathlib import Path

import torch

from contribution_mode_split import (
    allocation_from_centers,
    circularly_misalign_allocations,
    contribution_cluster_score,
    normalize_activation,
    spherical_two_cluster,
    v396_from_centered,
)


REQUIRED_KEYS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-files", type=int, default=64)
    parser.add_argument("--top-instances", type=int, default=64)
    parser.add_argument("--scan-batch-tokens", type=int, default=256)
    parser.add_argument("--feature-chunk", type=int, default=128)
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--minimum-cluster-size", type=int, default=16)
    parser.add_argument("--split-pairs", type=int, default=4096)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_manifest(paths: list[Path]) -> dict[str, object]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        info = path.stat()
        mode = stat.S_IMODE(info.st_mode)
        total_bytes += info.st_size
        digest.update(
            f"{path.name}\0{info.st_size}\0{info.st_mtime_ns}\0{mode}\n".encode()
        )
    return {
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "metadata_sha256": digest.hexdigest(),
        "all_files_have_no_write_bits": all(
            stat.S_IMODE(path.stat().st_mode) & 0o222 == 0 for path in paths
        ),
        "directory_mode": oct(stat.S_IMODE(paths[0].parent.stat().st_mode)),
    }


def load_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    missing = sorted(REQUIRED_KEYS - set(raw))
    if missing:
        raise KeyError(f"Checkpoint is missing {missing}")
    return {key: raw[key].detach().cpu() for key in REQUIRED_KEYS}


def select_sample_paths(
    all_paths: list[Path], sample_files: int, seed: int
) -> list[Path]:
    if sample_files > len(all_paths):
        raise ValueError("sample_files exceeds available files")
    paths = list(all_paths)
    random.Random(seed).shuffle(paths)
    return paths[:sample_files]


def load_sample_tokens(
    paths: list[Path], eps: float
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    chunks = []
    records = []
    offset = 0
    for path in paths:
        values = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise TypeError(f"Unexpected cache payload in {path}")
        normalized = normalize_activation(values, eps).to(torch.float16)
        chunks.append(normalized)
        records.append(
            {
                "name": path.name,
                "tokens": int(values.shape[0]),
                "width": int(values.shape[1]),
                "global_start": offset,
                "global_end": offset + int(values.shape[0]),
            }
        )
        offset += int(values.shape[0])
    return torch.cat(chunks, dim=0), records


def state_for_scan(
    state: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "b_pre": state["b_pre"].to(device=device, dtype=torch.bfloat16),
        "encoder.weight": state["encoder.weight"].to(
            device=device, dtype=torch.bfloat16
        ),
        "encoder.bias": state["encoder.bias"].to(
            device=device, dtype=torch.bfloat16
        ),
        "v396.raw_beta": state["v396.raw_beta"].to(device),
        "v396.log_gain": state["v396.log_gain"].to(device),
        "v396.max_beta": state["v396.max_beta"].to(device),
        "v396.max_log_gain": state["v396.max_log_gain"].to(device),
    }


def scan_top_instances(
    x_cpu: torch.Tensor,
    state: dict[str, torch.Tensor],
    top_instances: int,
    batch_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scan_state = state_for_scan(state, device)
    feature_count = int(scan_state["encoder.weight"].shape[0])
    top_values = torch.full(
        (feature_count, top_instances),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    top_indices = torch.full(
        (feature_count, top_instances),
        -1,
        dtype=torch.int64,
        device=device,
    )
    activation_sum = torch.zeros(feature_count, dtype=torch.float64, device=device)
    activation_count = torch.zeros(feature_count, dtype=torch.int64, device=device)
    for start in range(0, x_cpu.shape[0], batch_tokens):
        end = min(x_cpu.shape[0], start + batch_tokens)
        x = x_cpu[start:end].to(device=device, dtype=torch.bfloat16)
        centered = x - scan_state["b_pre"]
        _, z = v396_from_centered(centered, scan_state)
        activation_sum += z.double().sum(dim=0)
        activation_count += (z > 0).sum(dim=0)
        candidates = torch.cat([top_values, z.T], dim=1)
        new_indices = torch.arange(start, end, device=device).expand(
            feature_count, -1
        )
        candidate_indices = torch.cat([top_indices, new_indices], dim=1)
        top_values, positions = torch.topk(
            candidates, k=top_instances, dim=1, largest=True, sorted=True
        )
        top_indices = candidate_indices.gather(1, positions)
        del x, centered, z, candidates, new_indices, candidate_indices, positions
    if (top_indices < 0).any():
        raise AssertionError("Top-instance scan left invalid indices")
    return (
        top_values.cpu(),
        top_indices.cpu(),
        activation_sum.cpu(),
        activation_count.cpu(),
    )


def contribution_patterns(
    x_cpu: torch.Tensor,
    top_indices: torch.Tensor,
    feature_indices: torch.Tensor,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    selected = top_indices.index_select(0, feature_indices.cpu())
    instances = x_cpu.index_select(0, selected.reshape(-1)).reshape(
        selected.shape[0], selected.shape[1], x_cpu.shape[1]
    )
    centered = instances.to(device=device, dtype=torch.float32)
    centered -= state["b_pre"].to(device=device, dtype=torch.float32)
    weight = state["encoder.weight"].index_select(
        0, feature_indices.cpu()
    ).to(device=device, dtype=torch.float32)
    return centered * weight.unsqueeze(1)


def score_all_features(
    x_cpu: torch.Tensor,
    top_indices: torch.Tensor,
    state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    feature_count = int(state["encoder.weight"].shape[0])
    scores = torch.empty(feature_count, dtype=torch.float32)
    for start in range(0, feature_count, args.feature_chunk):
        end = min(feature_count, start + args.feature_chunk)
        feature_indices = torch.arange(start, end, dtype=torch.int64)
        patterns = contribution_patterns(
            x_cpu, top_indices, feature_indices, state, device
        )
        assignments, centers, counts = spherical_two_cluster(
            patterns, args.kmeans_iterations
        )
        score = contribution_cluster_score(
            patterns,
            assignments,
            centers,
            counts,
            args.minimum_cluster_size,
        )
        scores[start:end] = score.cpu()
        del patterns, assignments, centers, counts, score
    return scores


def fit_selected_allocations(
    x_cpu: torch.Tensor,
    top_indices: torch.Tensor,
    parent_indices: torch.Tensor,
    state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    allocations = []
    cluster_counts = []
    center_cosines = []
    for start in range(0, len(parent_indices), args.feature_chunk):
        ids = parent_indices[start : start + args.feature_chunk]
        patterns = contribution_patterns(x_cpu, top_indices, ids, state, device)
        _, centers, counts = spherical_two_cluster(
            patterns, args.kmeans_iterations
        )
        allocation = allocation_from_centers(centers, args.rho)
        allocations.append(allocation.to(torch.float16).cpu())
        cluster_counts.append(counts.cpu())
        center_cosines.append((centers[:, 0] * centers[:, 1]).sum(dim=1).cpu())
        del patterns, centers, counts, allocation
    allocation = torch.cat(allocations, dim=0)
    counts = torch.cat(cluster_counts, dim=0)
    cosines = torch.cat(center_cosines, dim=0)
    diagnostics = {
        "cluster_min_count_min": int(counts.min(dim=1).values.min().item()),
        "cluster_min_count_mean": float(counts.min(dim=1).values.float().mean().item()),
        "center_cosine_mean": float(cosines.mean().item()),
        "center_cosine_min": float(cosines.min().item()),
        "exclusive_fraction_mean": float((allocation != 0.5).float().mean().item()),
    }
    return allocation, diagnostics


def choose_disjoint_recipients(
    mean_activation: torch.Tensor,
    parents: torch.Tensor,
    count: int,
) -> torch.Tensor:
    parent_mask = torch.zeros_like(mean_activation, dtype=torch.bool)
    parent_mask[parents] = True
    order = torch.argsort(mean_activation, descending=False)
    recipients = order[~parent_mask[order]][:count]
    if len(recipients) != count:
        raise AssertionError("Not enough recipient features")
    return recipients


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()

    all_paths = sorted(args.data_dir.glob("*.pt"))
    if not all_paths:
        raise FileNotFoundError(args.data_dir)
    manifest_before = metadata_manifest(all_paths)
    if not manifest_before["all_files_have_no_write_bits"]:
        raise PermissionError("Activation cache must be read-only")
    sample_paths = select_sample_paths(all_paths, args.sample_files, args.seed)
    state = load_state(args.checkpoint)
    parameter_count = int(
        sum(
            state[key].numel()
            for key in (
                "b_pre",
                "encoder.weight",
                "encoder.bias",
                "decoder.weight",
                "v396.raw_beta",
                "v396.log_gain",
            )
        )
    )
    feature_count = int(state["encoder.weight"].shape[0])
    if args.split_pairs * 2 > feature_count:
        raise ValueError("Too many split pairs")
    x_cpu, sample_records = load_sample_tokens(sample_paths, args.normalize_eps)
    if int(x_cpu.shape[1]) != int(state["b_pre"].numel()):
        raise ValueError("Activation width and checkpoint width differ")

    top_values, top_indices, activation_sum, activation_count = scan_top_instances(
        x_cpu,
        state,
        args.top_instances,
        args.scan_batch_tokens,
        device,
    )
    scores = score_all_features(x_cpu, top_indices, state, args, device)
    finite_count = int(torch.isfinite(scores).sum().item())
    if finite_count < args.split_pairs:
        raise RuntimeError(
            f"Only {finite_count} features meet the cluster-size requirement"
        )
    parent_indices = torch.topk(
        scores, k=args.split_pairs, largest=True, sorted=True
    ).indices.to(torch.int64)
    allocation, cluster_diagnostics = fit_selected_allocations(
        x_cpu,
        top_indices,
        parent_indices,
        state,
        args,
        device,
    )
    wrong_allocation, wrong_shifts = circularly_misalign_allocations(
        allocation.float(), parent_indices
    )
    wrong_allocation = wrong_allocation.to(torch.float16)
    unchanged_wrong = int(
        (wrong_allocation == allocation).all(dim=1).sum().item()
    )
    mean_activation = activation_sum / int(x_cpu.shape[0])
    recipient_indices = choose_disjoint_recipients(
        mean_activation, parent_indices, args.split_pairs
    )
    if torch.isin(parent_indices, recipient_indices).any():
        raise AssertionError("Parent/recipient overlap")

    spec = {
        "experiment": "ELUDe-inspired contribution-mode split frozen causal gate",
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": file_sha256(args.checkpoint),
        "base_parameter_count": parameter_count,
        "exposed_feature_count": feature_count,
        "parent_indices": parent_indices,
        "recipient_indices": recipient_indices,
        "candidate_allocation": allocation,
        "wrong_allocation": wrong_allocation,
        "wrong_shifts": wrong_shifts.cpu(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    spec_path = args.output_dir / "contribution-mode-split-spec.pt"
    torch.save(spec, spec_path)

    labels = [
        ("Frozen V396 reference", "v396_reference"),
        ("Mass-fold-only control", "cms_mass_fold"),
        ("Coordinate-misaligned split control", "cms_wrong"),
        ("True contribution-mode split candidate", "cms_candidate"),
    ]
    targets = [
        {
            "label": label,
            "kind": kind,
            "layer": 22,
            "checkpoint": str(args.checkpoint),
            "split_spec": str(spec_path),
        }
        for label, kind in labels
    ]
    targets_path = args.output_dir / "targets-contribution-mode-split.json"
    targets_path.write_text(json.dumps(targets, indent=2) + "\n")

    manifest_after = metadata_manifest(all_paths)
    selected_scores = scores[parent_indices]
    recipient_mean = mean_activation[recipient_indices]
    parent_mean = mean_activation[parent_indices]
    summary = {
        "experiment": spec["experiment"],
        "status": "mechanism-preflight-not-final-architecture",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "base_checkpoint_sha256": spec["base_checkpoint_sha256"],
        "base_parameter_count": parameter_count,
        "exposed_feature_count_each": feature_count,
        "sample_token_count": int(x_cpu.shape[0]),
        "sample_files": sample_records,
        "data_manifest_before": manifest_before,
        "data_manifest_after": manifest_after,
        "data_unchanged": manifest_before == manifest_after,
        "finite_cluster_score_count": finite_count,
        "selected_parent_score": {
            "minimum": float(selected_scores.min().item()),
            "mean": float(selected_scores.mean().item()),
            "maximum": float(selected_scores.max().item()),
        },
        "parent_mean_activation": {
            "minimum": float(parent_mean.min().item()),
            "mean": float(parent_mean.mean().item()),
            "maximum": float(parent_mean.max().item()),
        },
        "recipient_mean_activation": {
            "minimum": float(recipient_mean.min().item()),
            "mean": float(recipient_mean.mean().item()),
            "maximum": float(recipient_mean.max().item()),
        },
        "cluster_diagnostics": cluster_diagnostics,
        "wrong_control": {
            "zero_shifts": int((wrong_shifts == 0).sum().item()),
            "unchanged_allocation_rows": unchanged_wrong,
            "allocation_multisets_preserved": bool(
                torch.equal(
                    allocation.sort(dim=1).values,
                    wrong_allocation.sort(dim=1).values,
                )
            ),
        },
        "integrity": {
            "parent_recipient_disjoint": True,
            "same_exposed_feature_count": True,
            "pairwise_activation_mass_preserved_by_definition": True,
            "uses_saebench_labels_for_fitting": False,
            "uses_eval_split_for_fitting": False,
            "uses_one_vs_rest_targets_for_fitting": False,
            "uses_mean_diff_selection_for_fitting": False,
            "uses_test_feedback_for_fitting": False,
        },
        "spec_path": str(spec_path),
        "spec_sha256": file_sha256(spec_path),
        "targets_path": str(targets_path),
        "elapsed_seconds": time.time() - started,
    }
    if not summary["data_unchanged"]:
        raise AssertionError("Activation cache metadata changed")
    if unchanged_wrong or summary["wrong_control"]["zero_shifts"]:
        raise AssertionError("Wrong control contains an identity mapping")
    if not summary["wrong_control"]["allocation_multisets_preserved"]:
        raise AssertionError("Wrong control changed an allocation multiset")
    summary_path = args.output_dir / "fit-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
