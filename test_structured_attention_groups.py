#!/usr/bin/env python3
"""Small mechanism tests for the structured attention-group frozen gate."""

from __future__ import annotations

import numpy as np
import torch

from diagnose_structured_attention_groups import (
    attention_group_labels,
    contiguous_group_labels,
    maximally_shifted_labels,
    strongest_group_rms,
)


def test_grouping_and_controls() -> None:
    sequence_length = 40
    block = torch.zeros(sequence_length, sequence_length)
    for start in range(0, sequence_length, 4):
        block[start : start + 4, start : start + 4] = 1.0
    block += torch.eye(sequence_length)
    labels, stats = attention_group_labels([block] * 4, 10, 0.02)
    assert labels.shape == (sequence_length,)
    assert torch.unique(labels).numel() == 10
    assert stats["within_minus_between"] > 0.0

    contiguous = contiguous_group_labels(sequence_length, 10)
    assert torch.equal(torch.bincount(contiguous), torch.full((10,), 4))

    shifted, shift, agreement = maximally_shifted_labels(labels)
    assert 1 <= shift < sequence_length
    assert agreement < 1.0
    assert torch.equal(
        torch.sort(torch.bincount(labels)).values,
        torch.sort(torch.bincount(shifted)).values,
    )


def test_group_rms() -> None:
    token_features = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    output = strongest_group_rms(token_features, labels)
    np.testing.assert_allclose(output.numpy(), np.array([1.0, 2.0]), atol=1.0e-6)


if __name__ == "__main__":
    test_grouping_and_controls()
    test_group_rms()
    print("structured attention-group mechanism tests passed")
