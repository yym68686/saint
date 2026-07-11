#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from train_c2r_consistency_preflight import (
    ReLUSAE,
    compute_c2r_loss,
    parameter_count,
)


class C2RConsistencyPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.d_model = 8
        self.n_latents = 16
        self.state = {
            "b_pre": torch.randn(self.d_model),
            "encoder.weight": torch.randn(self.n_latents, self.d_model) * 0.1,
            "encoder.bias": torch.randn(self.n_latents) * 0.1,
            "decoder.weight": torch.randn(self.d_model, self.n_latents),
        }

    def test_relu_model_parameter_count_matches_state(self) -> None:
        model = ReLUSAE(self.state)
        expected = sum(tensor.numel() for tensor in self.state.values())
        self.assertEqual(parameter_count(model), expected)

    def test_wrong_alignment_has_no_fixed_mapping(self) -> None:
        model = ReLUSAE(self.state)
        features, _ = model(torch.randn(12, self.d_model))
        _, diagnostics = compute_c2r_loss(
            features,
            model.decoder_weight,
            subset_size=10,
            wrong_alignment=True,
            wrong_shift=3,
        )
        self.assertEqual(diagnostics.selected_count, 10)
        self.assertEqual(diagnostics.wrong_shift, 3)
        self.assertEqual(diagnostics.wrong_fixed_pair_count, 0)

    def test_c2r_gradient_reaches_encoder_and_decoder(self) -> None:
        model = ReLUSAE(self.state)
        features, _ = model(torch.randn(12, self.d_model))
        loss, diagnostics = compute_c2r_loss(
            features,
            model.decoder_weight,
            subset_size=10,
            wrong_alignment=False,
            wrong_shift=0,
        )
        self.assertGreater(diagnostics.raw_loss, 0.0)
        loss.backward()
        self.assertIsNotNone(model.encoder_weight.grad)
        self.assertIsNotNone(model.decoder_weight.grad)
        self.assertGreater(float(model.encoder_weight.grad.norm()), 0.0)
        self.assertGreater(float(model.decoder_weight.grad.norm()), 0.0)

    def test_true_and_wrong_losses_are_finite(self) -> None:
        model = ReLUSAE(self.state)
        features, _ = model(torch.randn(12, self.d_model))
        true_loss, _ = compute_c2r_loss(
            features,
            model.decoder_weight,
            subset_size=10,
            wrong_alignment=False,
            wrong_shift=0,
        )
        wrong_loss, _ = compute_c2r_loss(
            features,
            model.decoder_weight,
            subset_size=10,
            wrong_alignment=True,
            wrong_shift=4,
        )
        self.assertTrue(bool(torch.isfinite(true_loss)))
        self.assertTrue(bool(torch.isfinite(wrong_loss)))


if __name__ == "__main__":
    unittest.main()
