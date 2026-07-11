#!/usr/bin/env python3
"""Unit tests for the parameter-neutral cross-layer concordance SAE."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from train_structured_crosslayer_concordance_sae import (
    CrossLayerSharedReLUSAE,
    cosine_concordance_loss,
    parameter_count,
    wrong_token_permutation,
)


def fake_state(n_latents: int = 32, d_model: int = 8) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(123)
    encoder = torch.randn(n_latents, d_model, generator=generator)
    decoder = torch.randn(d_model, n_latents, generator=generator)
    decoder = F.normalize(decoder, dim=1)
    return {
        "b_pre": torch.randn(d_model, generator=generator),
        "encoder.weight": encoder,
        "encoder.bias": torch.randn(n_latents, generator=generator),
        "decoder.weight": decoder,
    }


class CrossLayerConcordanceTests(unittest.TestCase):
    def test_wrong_permutation_is_bijective_without_same_sample_pairs(self) -> None:
        for lengths in (
            torch.tensor([4, 4]),
            torch.tensor([3, 4, 3]),
            torch.tensor([2, 3, 4, 2]),
        ):
            sample_index = torch.repeat_interleave(
                torch.arange(lengths.numel()),
                lengths,
            )
            permutation = wrong_token_permutation(sample_index, lengths)
            self.assertEqual(torch.unique(permutation).numel(), permutation.numel())
            self.assertFalse(bool((sample_index == sample_index[permutation]).any()))

    def test_parameter_count_exactly_matches_source_relu(self) -> None:
        state = fake_state()
        model = CrossLayerSharedReLUSAE(
            state,
            layers=(20, 21, 22, 23),
            reference_layer=22,
            calibration_groups=2,
            max_log_scale=0.25,
        )
        expected = sum(tensor.numel() for tensor in state.values())
        self.assertEqual(parameter_count(model), expected)
        self.assertEqual(model.export_state()["encoder.weight"].shape, state["encoder.weight"].shape)

    def test_true_identity_has_zero_concordance(self) -> None:
        z = torch.relu(torch.randn(7, 13)) + 0.1
        self.assertLess(float(cosine_concordance_loss(z, z).abs()), 1.0e-6)

    def test_concordance_reaches_dictionary_and_calibration(self) -> None:
        state = fake_state()
        model = CrossLayerSharedReLUSAE(
            state,
            layers=(20, 21, 22, 23),
            reference_layer=22,
            calibration_groups=2,
            max_log_scale=0.25,
        )
        x20 = torch.randn(9, 8)
        x22 = torch.randn(9, 8)
        out20 = model(x20, 20)
        out22 = model(x22, 22)
        loss = cosine_concordance_loss(out20["z"], out22["z"])
        loss.backward()
        self.assertGreater(float(model.encoder_weight.grad.abs().max()), 0.0)
        self.assertGreater(float(model.layer_group_log_scale.grad.abs().max()), 0.0)

    def test_total_loss_reaches_decoder(self) -> None:
        state = fake_state()
        model = CrossLayerSharedReLUSAE(
            state,
            layers=(20, 21, 22, 23),
            reference_layer=22,
            calibration_groups=2,
            max_log_scale=0.25,
        )
        x = torch.randn(9, 8)
        out = model(x, 22)
        F.mse_loss(out["recon"], x).backward()
        self.assertGreater(float(model.decoder_weight.grad.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
