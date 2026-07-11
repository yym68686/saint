#!/usr/bin/env python3
"""Tests for the exact-parameter CP multi-view SAE."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from train_structured_cp_multiview_sae import (
    CPMultiViewSAE,
    MeanPooledReLUSAE,
    derange_views,
    make_cp_initial_state,
    parameter_count,
    pool_interleaved_views,
)


def fake_state(n_latents: int = 32, d_model: int = 8) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(19)
    return {
        "b_pre": torch.randn(d_model, generator=generator),
        "encoder.weight": torch.randn(n_latents, d_model, generator=generator),
        "encoder.bias": torch.randn(n_latents, generator=generator),
        "decoder.weight": torch.randn(d_model, n_latents, generator=generator),
    }


class CPMultiViewTests(unittest.TestCase):
    def test_interleaved_pooling(self) -> None:
        x = torch.arange(7, dtype=torch.float32).unsqueeze(1)
        sample = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        view = torch.tensor([0, 1, 0, 1, 0, 1, 0])
        pooled = pool_interleaved_views(x, sample, view, 2, 2).squeeze(-1)
        expected = torch.tensor([[1.0, 2.0], [5.0, 5.0]])
        torch.testing.assert_close(pooled, expected)

    def test_derangement_preserves_view_marginals(self) -> None:
        views = torch.arange(8 * 4 * 3, dtype=torch.float32).reshape(8, 4, 3)
        wrong, collisions = derange_views(views)
        self.assertEqual(collisions, 0)
        for view in range(4):
            self.assertEqual(
                sorted(map(tuple, wrong[:, view].tolist())),
                sorted(map(tuple, views[:, view].tolist())),
            )

    def test_parameter_count_matches_standard_relu(self) -> None:
        state = fake_state()
        standard = MeanPooledReLUSAE(state)
        target = parameter_count(standard)
        cp_state, metadata = make_cp_initial_state(
            state,
            n_views=4,
            rank=3,
            target_parameter_count=target,
            device=torch.device("cpu"),
        )
        model = CPMultiViewSAE(cp_state, max_log_gain=0.25)
        self.assertEqual(parameter_count(model), target)
        self.assertEqual(metadata["gain_parameter_count"], 224)

    def test_full_rank_initialization_reproduces_mean_relu(self) -> None:
        state = fake_state()
        rank = state["b_pre"].numel()
        fixed = (
            2 * rank * (8 + 32 + 4)
            + 32
            + 4 * 8
        )
        cp_state, _ = make_cp_initial_state(
            state,
            n_views=4,
            rank=rank,
            target_parameter_count=fixed + 2,
            device=torch.device("cpu"),
        )
        cp = CPMultiViewSAE(cp_state, max_log_gain=0.25)
        standard = MeanPooledReLUSAE(state)
        views = torch.randn(5, 4, 8)
        cp_out = cp(views)
        standard_out = standard(views)
        torch.testing.assert_close(cp_out["z"], standard_out["z"], atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(
            cp_out["recon"],
            standard_out["recon"],
            atol=5e-4,
            rtol=5e-4,
        )

    def test_all_cp_modules_receive_gradient(self) -> None:
        state = fake_state()
        standard = MeanPooledReLUSAE(state)
        cp_state, _ = make_cp_initial_state(
            state,
            n_views=4,
            rank=3,
            target_parameter_count=parameter_count(standard),
            device=torch.device("cpu"),
        )
        model = CPMultiViewSAE(cp_state, max_log_gain=0.25)
        views = torch.randn(7, 4, 8)
        out = model(views)
        loss = F.mse_loss(out["recon"], views) + 1e-3 * out["z"].sum(dim=1).mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().max()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
