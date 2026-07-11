#!/usr/bin/env python3
"""Numerical tests for the contribution-mode split preflight."""

from __future__ import annotations

import unittest

import torch

from contribution_mode_split import (
    allocation_from_centers,
    apply_pair_transform,
    circularly_misalign_allocations,
    contribution_cluster_score,
    spherical_two_cluster,
    split_route_probability,
)


class ContributionModeSplitTest(unittest.TestCase):
    def test_spherical_two_cluster_recovers_separated_modes(self) -> None:
        generator = torch.Generator().manual_seed(42)
        first = torch.tensor([1.0, 0.0, 0.0, 0.0])
        second = torch.tensor([0.0, 1.0, 0.0, 0.0])
        p0 = first + 0.02 * torch.randn(1, 16, 4, generator=generator)
        p1 = second + 0.02 * torch.randn(1, 16, 4, generator=generator)
        patterns = torch.cat([p0, p1], dim=1)
        assignments, centers, counts = spherical_two_cluster(patterns, 8)
        score = contribution_cluster_score(
            patterns, assignments, centers, counts, 8
        )
        self.assertEqual(sorted(counts[0].tolist()), [16, 16])
        self.assertGreater(float(score.item()), 0.2)

    def test_allocation_is_additive(self) -> None:
        centers = torch.tensor(
            [[[2.0, 0.1, 1.0, 0.4], [0.1, 2.0, 0.9, 0.5]]]
        )
        allocation = allocation_from_centers(centers, rho=0.5)
        self.assertTrue(torch.equal(allocation, torch.tensor([[1.0, 0.0, 0.5, 0.5]])))
        weight = torch.randn(1, 4)
        self.assertTrue(
            torch.allclose(weight * allocation + weight * (1.0 - allocation), weight)
        )

    def test_wrong_control_preserves_multisets_without_identity_shift(self) -> None:
        allocations = torch.tensor(
            [[0.0, 0.5, 1.0, 0.5], [1.0, 0.0, 0.5, 0.0]]
        )
        wrong, shifts = circularly_misalign_allocations(
            allocations, torch.tensor([3, 7])
        )
        self.assertTrue((shifts > 0).all())
        self.assertTrue((shifts < allocations.shape[1]).all())
        self.assertTrue(
            torch.equal(
                allocations.sort(dim=1).values,
                wrong.sort(dim=1).values,
            )
        )

    def test_pair_transform_preserves_dimension_and_mass(self) -> None:
        features = torch.rand(5, 8)
        parents = torch.tensor([1, 5])
        recipients = torch.tensor([2, 7])
        probability = torch.rand(5, 2)
        for mode in ("candidate", "wrong"):
            output = apply_pair_transform(
                features, parents, recipients, mode, probability
            )
            self.assertEqual(output.shape, features.shape)
            before = features[:, parents] + features[:, recipients]
            after = output[:, parents] + output[:, recipients]
            self.assertTrue(torch.allclose(before, after, atol=1.0e-6))
        folded = apply_pair_transform(features, parents, recipients, "fold")
        self.assertTrue(
            torch.allclose(
                features[:, parents] + features[:, recipients],
                folded[:, parents] + folded[:, recipients],
            )
        )

    def test_split_route_probability_is_bounded(self) -> None:
        x = torch.randn(6, 4)
        weight = torch.randn(2, 4)
        bias = torch.randn(2)
        allocation = torch.tensor(
            [[1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, 0.5]]
        )
        parent_h = torch.nn.functional.linear(x, weight, bias)
        probability = split_route_probability(
            x, parent_h, weight, bias, allocation
        )
        self.assertEqual(probability.shape, (6, 2))
        self.assertTrue((probability >= 0).all())
        self.assertTrue((probability <= 1).all())

    def test_split_route_probability_preserves_bfloat16_linear_dtype(self) -> None:
        x = torch.randn(6, 4).to(torch.bfloat16)
        weight = torch.randn(2, 4).to(torch.bfloat16)
        bias = torch.randn(2).to(torch.bfloat16)
        allocation = torch.tensor(
            [[1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, 0.5]],
            dtype=torch.bfloat16,
        )
        parent_h = torch.nn.functional.linear(x, weight, bias)
        probability = split_route_probability(
            x, parent_h, weight, bias, allocation
        )
        self.assertEqual(probability.dtype, torch.float32)
        self.assertTrue(torch.isfinite(probability).all())


if __name__ == "__main__":
    unittest.main()
