#!/usr/bin/env python3
"""Shared math for the contribution-mode split causal gate."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


SplitMode = Literal["fold", "candidate", "wrong"]


def normalize_activation(
    x: torch.Tensor, eps: float = 1.0e-6
) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(
        dim=-1, keepdim=True
    ).clamp_min(eps)


def v396_from_centered(
    x_centered: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return V396 preactivations and nonnegative log-companded responses."""
    h = F.linear(
        x_centered,
        state["encoder.weight"],
        state.get("encoder.bias"),
    )
    u = torch.relu(h).float()
    max_beta = float(state.get("v396.max_beta", torch.tensor(4.0)).item())
    max_log_gain = float(
        state.get("v396.max_log_gain", torch.tensor(2.0)).item()
    )
    beta = F.softplus(state["v396.raw_beta"].float()).clamp(
        1.0e-4, max_beta
    )
    z = torch.log1p(beta.unsqueeze(0) * u) / torch.log1p(beta).unsqueeze(0)
    gain = state["v396.log_gain"].float().clamp(
        -max_log_gain, max_log_gain
    ).exp()
    return h, z * gain.unsqueeze(0)


def split_route_probability(
    x_centered: torch.Tensor,
    parent_h: torch.Tensor,
    parent_weight: torch.Tensor,
    parent_bias: torch.Tensor,
    allocation: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Route parent mass using an additive partition of its affine score."""
    first_weight = parent_weight * allocation
    bias_fraction = allocation.float().mean(dim=1).to(parent_bias.dtype)
    first_bias = parent_bias * bias_fraction
    first = F.linear(x_centered, first_weight, first_bias)
    second = parent_h - first
    first_pos = torch.relu(first.float())
    second_pos = torch.relu(second.float())
    denominator = first_pos + second_pos
    return torch.where(
        denominator > eps,
        first_pos / denominator.clamp_min(eps),
        torch.full_like(denominator, 0.5),
    )


def apply_pair_transform(
    base_features: torch.Tensor,
    parent_indices: torch.Tensor,
    recipient_indices: torch.Tensor,
    mode: SplitMode,
    first_probability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a dimension-preserving, pairwise mass-preserving split."""
    if parent_indices.numel() != recipient_indices.numel():
        raise ValueError("Parent and recipient counts must match")
    if torch.isin(parent_indices, recipient_indices).any():
        raise ValueError("Parent and recipient sets must be disjoint")
    parent = base_features.index_select(1, parent_indices)
    recipient = base_features.index_select(1, recipient_indices)
    if mode == "fold":
        first = parent + recipient
        second = torch.zeros_like(parent)
    else:
        if first_probability is None:
            raise ValueError(f"{mode} requires routing probabilities")
        if first_probability.shape != parent.shape:
            raise ValueError("Routing probability shape mismatch")
        first = parent * first_probability + recipient
        second = parent * (1.0 - first_probability)
    output = base_features.clone()
    output[:, parent_indices] = first
    output[:, recipient_indices] = second
    return output


def spherical_two_cluster(
    patterns: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic spherical k-means with a farthest-point second seed."""
    if patterns.ndim != 3:
        raise ValueError("patterns must have shape [features, instances, dims]")
    normalized = F.normalize(patterns.float(), p=2, dim=-1)
    center0 = normalized[:, 0]
    similarity0 = torch.einsum("fkd,fd->fk", normalized, center0)
    second_index = similarity0.argmin(dim=1)
    center1 = normalized[
        torch.arange(normalized.shape[0], device=normalized.device),
        second_index,
    ]
    assignments = torch.zeros(
        normalized.shape[:2], dtype=torch.bool, device=normalized.device
    )
    for _ in range(iterations):
        sim0 = torch.einsum("fkd,fd->fk", normalized, center0)
        sim1 = torch.einsum("fkd,fd->fk", normalized, center1)
        assignments = sim1 > sim0
        mask1 = assignments.float()
        mask0 = 1.0 - mask1
        count0 = mask0.sum(dim=1, keepdim=True).clamp_min(1.0)
        count1 = mask1.sum(dim=1, keepdim=True).clamp_min(1.0)
        center0 = F.normalize(
            torch.einsum("fk,fkd->fd", mask0, normalized) / count0,
            p=2,
            dim=-1,
        )
        center1 = F.normalize(
            torch.einsum("fk,fkd->fd", mask1, normalized) / count1,
            p=2,
            dim=-1,
        )
    centers = torch.stack([center0, center1], dim=1)
    counts = torch.stack(
        [(~assignments).sum(dim=1), assignments.sum(dim=1)], dim=1
    )
    return assignments, centers, counts


def contribution_cluster_score(
    patterns: torch.Tensor,
    assignments: torch.Tensor,
    centers: torch.Tensor,
    counts: torch.Tensor,
    minimum_cluster_size: int,
) -> torch.Tensor:
    normalized = F.normalize(patterns.float(), p=2, dim=-1)
    similarities = torch.einsum("fkd,fcd->fkc", normalized, centers)
    within = similarities.max(dim=-1).values.mean(dim=-1).clamp_min(0.0)
    separation = (1.0 - (centers[:, 0] * centers[:, 1]).sum(dim=-1)).clamp_min(0.0)
    balance = counts.min(dim=1).values.float() / patterns.shape[1]
    score = separation * balance * within
    invalid = counts.min(dim=1).values < int(minimum_cluster_size)
    return score.masked_fill(invalid, -torch.inf)


def allocation_from_centers(
    centers: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Construct the two-way additive ELUDe-style input allocation."""
    if centers.shape[1] != 2:
        raise ValueError("Exactly two centers are required")
    first = centers[:, 0].abs()
    second = centers[:, 1].abs()
    first_wins = rho * first > second
    second_wins = rho * second > first
    allocation = torch.full_like(first, 0.5)
    allocation[first_wins] = 1.0
    allocation[second_wins] = 0.0
    return allocation


def circularly_misalign_allocations(
    allocations: torch.Tensor,
    parent_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Break coordinate alignment while preserving every allocation multiset."""
    width = allocations.shape[1]
    rank = torch.arange(
        allocations.shape[0], device=allocations.device, dtype=torch.int64
    )
    shifts = (parent_indices.to(torch.int64) * 17 + rank * 31) % (width - 1) + 1
    coordinates = torch.arange(width, device=allocations.device)
    gather = (coordinates.unsqueeze(0) - shifts.unsqueeze(1)) % width
    wrong = allocations.gather(1, gather)
    for _ in range(width - 2):
        unchanged = (wrong == allocations).all(dim=1)
        if not unchanged.any():
            break
        shifts[unchanged] = shifts[unchanged] % (width - 1) + 1
        gather = (coordinates.unsqueeze(0) - shifts.unsqueeze(1)) % width
        wrong = allocations.gather(1, gather)
    if (wrong == allocations).all(dim=1).any():
        raise ValueError("At least one allocation is invariant to every circular shift")
    return wrong, shifts
