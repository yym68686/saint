#!/usr/bin/env python3
"""Numerical invariants for the structured next-token downstream SAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from train_structured_nexttoken_downstream_sae import (
    DownstreamNextTokenV396SAE,
    normalize_with_stats,
)


def make_base_state(d_model: int = 8, n_latents: int = 16) -> dict[str, object]:
    generator = torch.Generator().manual_seed(123)
    decoder = torch.randn(d_model, n_latents, generator=generator)
    decoder = decoder / decoder.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
    return {
        "b_pre": torch.randn(d_model, generator=generator) * 0.01,
        "encoder.weight": torch.randn(n_latents, d_model, generator=generator) * 0.1,
        "encoder.bias": torch.randn(n_latents, generator=generator) * 0.01,
        "decoder.weight": decoder,
        "raw_beta": torch.full((n_latents,), -1.0),
        "log_gain": torch.zeros(n_latents),
        "init_beta": 0.25,
        "max_beta": 4.0,
        "max_log_gain": 2.0,
    }


def assert_initial_noop_and_roundtrip() -> None:
    base = make_base_state()
    model = DownstreamNextTokenV396SAE(base, 4, 99, 0.5)
    raw = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(321))
    normalized, mean, scale = normalize_with_stats(raw, 1.0e-6)
    restored = normalized * scale + mean
    if not torch.allclose(restored, raw.float(), atol=2.0e-6, rtol=2.0e-6):
        raise AssertionError("normalize_with_stats does not round-trip")
    out = model(normalized)
    if not torch.equal(out["context_gain"], torch.zeros_like(out["context_gain"])):
        raise AssertionError("Context path must be an exact no-op at initialization")


def assert_downstream_gradient_reaches_trunk_and_context() -> None:
    base = make_base_state()
    model = DownstreamNextTokenV396SAE(base, 4, 99, 0.5)
    raw = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(456))
    normalized, mean, scale = normalize_with_stats(raw, 1.0e-6)
    out = model(normalized)
    reconstructed = out["recon"] * scale + mean
    frozen_output = torch.randn(
        19,
        8,
        generator=torch.Generator().manual_seed(789),
    )
    logits = F.linear(reconstructed[:, :-1], frozen_output)
    targets = torch.randint(
        0,
        19,
        (2, 4),
        generator=torch.Generator().manual_seed(987),
    )
    loss = F.cross_entropy(logits.reshape(-1, 19), targets.reshape(-1))
    loss.backward()
    trunk_grad = sum(
        float(parameter.grad.square().sum().item())
        for parameter in model.trunk_parameters()
        if parameter.grad is not None
    )
    context_grad = sum(
        float(parameter.grad.square().sum().item())
        for parameter in model.context_parameters()
        if parameter.grad is not None
    )
    if trunk_grad <= 0.0:
        raise AssertionError("Downstream objective did not reach the SAE trunk")
    if context_grad <= 0.0:
        raise AssertionError("Downstream objective did not reach the context module")


def assert_matched_initialization() -> None:
    base = make_base_state()
    first = DownstreamNextTokenV396SAE(base, 4, 99, 0.5)
    second = DownstreamNextTokenV396SAE(base, 4, 99, 0.5)
    if first.state_dict().keys() != second.state_dict().keys():
        raise AssertionError("Variant state keys differ")
    for key, value in first.state_dict().items():
        if not torch.equal(value, second.state_dict()[key]):
            raise AssertionError(f"Variant initialization differs at {key}")


def assert_batch_derangement() -> None:
    sample_ids = torch.tensor([11, 22, 33, 44])
    permutation = torch.roll(torch.arange(sample_ids.numel()), shifts=1)
    if bool((permutation == torch.arange(sample_ids.numel())).any()):
        raise AssertionError("Wrong-target permutation has fixed points")
    if bool((sample_ids[permutation] == sample_ids).any()):
        raise AssertionError("Wrong-target permutation retains a sample")
    values = torch.tensor([5, 7, 7, 9])
    if not torch.equal(values.sort().values, values[permutation].sort().values):
        raise AssertionError("Wrong-target permutation changed the marginal")


def main() -> None:
    assert_initial_noop_and_roundtrip()
    assert_downstream_gradient_reaches_trunk_and_context()
    assert_matched_initialization()
    assert_batch_derangement()
    print("structured next-token downstream SAE invariants: PASS")


if __name__ == "__main__":
    main()
