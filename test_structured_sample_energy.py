#!/usr/bin/env python3
"""Mechanism tests for the structured sample-energy gate."""

from __future__ import annotations

import torch

from diagnose_structured_sample_energy import (
    class_balanced_wrong_sample,
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


if __name__ == "__main__":
    test_wrong_sample_control()
    test_effect_size_scale_invariance()
    print("structured sample-energy mechanism tests passed")
