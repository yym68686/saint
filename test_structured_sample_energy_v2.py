#!/usr/bin/env python3
"""Mechanism tests for the mean-preserving sample-dispersion gate."""

from __future__ import annotations

import torch

from diagnose_structured_sample_energy_v2 import (
    class_balanced_wrong_sample,
    pool_sample_statistics,
    select_effect_size,
)


def test_wrong_sample_control() -> None:
    features = {
        "a": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        "b": torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]),
        "c": torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    }
    wrong, stats = class_balanced_wrong_sample(features)
    assert stats["same_class_pairs"] == 0
    assert stats["full_row_multiset_equal_by_bijective_source_counts"]
    assert all(count == 3 for count in stats["source_counts"].values())
    assert set(wrong) == set(features)


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


if __name__ == "__main__":
    test_wrong_sample_control()
    test_effect_size_scale_invariance()
    test_mean_dispersion_formula()
    test_rms_is_distinct_from_mean_plus_std()
    print("structured sample-energy v2 mechanism tests passed")
