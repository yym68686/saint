#!/usr/bin/env python3
"""Mechanism tests for stable moment routing."""

from __future__ import annotations

import torch

from diagnose_structured_stable_moment_routing_v3 import (
    apply_moment_route,
    correlation_from_sums,
    derangement,
    pack_samples,
    pool_sample_statistics,
    select_effect_size,
)


def test_wrong_sample_derangement() -> None:
    permutation = derangement(64, 44027)
    assert torch.equal(torch.sort(permutation).values, torch.arange(64))
    assert not torch.any(permutation == torch.arange(64))


def test_pack_samples() -> None:
    packed = pack_samples([torch.zeros(4, 2), torch.ones(6, 2)])
    assert packed["lengths"].tolist() == [4, 6]
    assert packed["sample_index"].tolist() == [0] * 4 + [1] * 6
    assert packed["view_index"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


def test_effect_size_scale_invariance() -> None:
    train_x = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.2],
            [1.0, 0.0, 0.0],
            [1.2, 1.0, 0.0],
        ]
    )
    train_y = torch.tensor([0, 0, 1, 1])
    scale = torch.tensor([100.0, 0.1, 7.0])
    selected = select_effect_size(train_x, train_y, 2)
    scaled_selected = select_effect_size(train_x * scale, train_y, 2)
    assert torch.equal(selected, scaled_selected)


def test_mean_dispersion_formula() -> None:
    token_features = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 2.0],
            [5.0, 4.0],
            [5.0, 4.0],
        ]
    )
    sample_ids = torch.tensor([0, 0, 1, 1])
    pooled = pool_sample_statistics(token_features, sample_ids, 2)
    expected_mean = torch.tensor([[2.0, 2.0], [5.0, 4.0]])
    expected_std = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(pooled["mean_pool"], expected_mean)
    assert torch.allclose(pooled["std_pool"], expected_std)
    assert torch.allclose(
        pooled["mean_std_pool"], expected_mean + expected_std
    )
    assert torch.all(pooled["mean_std_pool"] >= pooled["mean_pool"])
    assert torch.equal(pooled["mean_std_pool"][1], expected_mean[1])


def test_rms_is_distinct_from_mean_plus_std() -> None:
    token_features = torch.tensor([[0.0], [2.0]])
    sample_ids = torch.tensor([0, 0])
    pooled = pool_sample_statistics(token_features, sample_ids, 1)
    assert torch.allclose(pooled["mean_pool"], torch.tensor([[1.0]]))
    assert torch.allclose(pooled["std_pool"], torch.tensor([[1.0]]))
    assert torch.allclose(pooled["mean_std_pool"], torch.tensor([[2.0]]))
    assert torch.allclose(pooled["rms_pool"], torch.tensor([[2.0**0.5]]))


def test_correlation_from_sums() -> None:
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    b = torch.tensor([[2.0, 3.0], [4.0, 2.0], [6.0, 1.0]])
    correlation = correlation_from_sums(
        a.sum(0),
        b.sum(0),
        a.square().sum(0),
        b.square().sum(0),
        (a * b).sum(0),
        3,
    )
    assert torch.allclose(correlation, torch.tensor([1.0, 0.0]))


def test_apply_moment_route() -> None:
    means = {"c": torch.tensor([[1.0, 2.0, 3.0]])}
    stds = {"c": torch.tensor([[4.0, 5.0, 6.0]])}
    routed = apply_moment_route(
        means, stds, torch.tensor([False, True, False])
    )
    assert torch.equal(routed["c"], torch.tensor([[1.0, 5.0, 3.0]]))


if __name__ == "__main__":
    test_wrong_sample_derangement()
    test_pack_samples()
    test_effect_size_scale_invariance()
    test_mean_dispersion_formula()
    test_rms_is_distinct_from_mean_plus_std()
    test_correlation_from_sums()
    test_apply_moment_route()
    print("structured stable-moment routing v3 mechanism tests passed")
