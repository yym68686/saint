from __future__ import annotations

import torch
import torch.nn.functional as F

from train_cascaded_concept_sae import (
    CascadedConceptSAE,
    PartitionedV396,
    permuted_parent_assignment,
    parameter_count,
)


def make_state(d_model: int = 4, n_latents: int = 10) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    decoder = torch.randn(d_model, n_latents, generator=generator)
    decoder = decoder / decoder.norm(dim=0, keepdim=True)
    return {
        "b_pre": torch.randn(d_model, generator=generator),
        "encoder.weight": torch.randn(
            n_latents, d_model, generator=generator
        ),
        "encoder.bias": torch.randn(n_latents, generator=generator),
        "decoder.weight": decoder,
        "v396.raw_beta": torch.full((n_latents,), -1.25),
        "v396.log_gain": torch.zeros(n_latents),
        "v396.max_beta": torch.tensor(4.0),
        "v396.max_log_gain": torch.tensor(2.0),
    }


def test_exact_parameter_mapping() -> None:
    state = make_state()
    control = PartitionedV396(state, high_features=4)
    candidate = CascadedConceptSAE(
        state, high_features=4, active_atom_cap=6
    )
    assert parameter_count(control) == parameter_count(candidate)
    assert parameter_count(control) == sum(
        value.numel()
        for key, value in state.items()
        if key not in {"v396.max_beta", "v396.max_log_gain"}
    )


def test_control_initial_state_round_trip() -> None:
    state = make_state()
    control = PartitionedV396(state, high_features=4)
    exported = control.export_state()
    for key in (
        "b_pre",
        "encoder.weight",
        "encoder.bias",
        "decoder.weight",
        "v396.raw_beta",
        "v396.log_gain",
    ):
        assert torch.allclose(exported[key], state[key])


def test_level2_loss_updates_both_levels() -> None:
    state = make_state()
    candidate = CascadedConceptSAE(
        state, high_features=4, active_atom_cap=6
    )
    x = torch.randn(5, 4)
    output = candidate(x)
    loss = F.mse_loss(
        output["hierarchy_reconstruction"], output["hierarchy_target"]
    ) + 1.0e-3 * output["hierarchy_code"].mean()
    loss.backward()
    assert candidate.high_encoder.grad is not None
    assert candidate.high_decoder.grad is not None
    assert candidate.high_bias.grad is not None
    assert candidate.low_decoder.grad is not None
    assert candidate.high_encoder.grad.norm() > 0
    assert candidate.high_decoder.grad.norm() > 0
    assert candidate.low_decoder.grad.norm() > 0


def test_parent_permutation_preserves_cluster_size_multiset() -> None:
    parent = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2])
    wrong = permuted_parent_assignment(parent)
    assert not torch.any(parent == wrong)
    assert torch.equal(
        torch.sort(torch.bincount(parent)[torch.unique(parent)]).values,
        torch.sort(torch.bincount(wrong)[torch.unique(wrong)]).values,
    )
