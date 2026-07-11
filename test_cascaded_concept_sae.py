from __future__ import annotations

import torch
import torch.nn.functional as F

from train_cascaded_concept_sae import (
    CascadedConceptSAE,
    PartitionedV396,
    hierarchy_information_loss,
    maximally_deranged_parent_assignment,
    parameter_count,
    rank_low_activity_slots,
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
    kept = torch.tensor([0, 2, 4, 5, 6, 9])
    reallocated = torch.tensor([1, 3, 7, 8])
    control = PartitionedV396(state, kept, reallocated)
    candidate = CascadedConceptSAE(
        state, kept, reallocated, active_atom_cap=6, balance_temperature=0.1
    )
    assert parameter_count(control) == parameter_count(candidate)
    assert parameter_count(control) == sum(
        value.numel()
        for key, value in state.items()
        if key not in {"v396.max_beta", "v396.max_log_gain"}
    )


def test_control_initial_state_round_trip() -> None:
    state = make_state()
    kept = torch.tensor([0, 2, 4, 5, 6, 9])
    reallocated = torch.tensor([1, 3, 7, 8])
    order = torch.cat([kept, reallocated])
    control = PartitionedV396(state, kept, reallocated)
    exported = control.export_state()
    assert torch.allclose(exported["b_pre"], state["b_pre"])
    assert torch.allclose(
        exported["encoder.weight"], state["encoder.weight"].index_select(0, order)
    )
    assert torch.allclose(
        exported["encoder.bias"], state["encoder.bias"].index_select(0, order)
    )
    assert torch.allclose(
        exported["decoder.weight"], state["decoder.weight"].index_select(1, order)
    )


def test_level2_loss_updates_both_levels() -> None:
    state = make_state()
    kept = torch.tensor([0, 2, 4, 5, 6, 9])
    reallocated = torch.tensor([1, 3, 7, 8])
    candidate = CascadedConceptSAE(
        state, kept, reallocated, active_atom_cap=6, balance_temperature=0.1
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


def test_parent_derangement_is_maximal_and_count_preserving() -> None:
    parent = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2])
    wrong = maximally_deranged_parent_assignment(parent)
    assert int((parent == wrong).sum()) == 2
    assert torch.equal(
        torch.bincount(parent),
        torch.bincount(wrong),
    )


def test_information_loss_prefers_balanced_confident_assignments() -> None:
    balanced = torch.eye(4) * 20
    collapsed = torch.zeros(4, 4)
    collapsed[:, 0] = 20
    balanced_loss, _, _ = hierarchy_information_loss(balanced, 0.1)
    collapsed_loss, _, _ = hierarchy_information_loss(collapsed, 0.1)
    assert balanced_loss < collapsed_loss - 0.9


def test_low_activity_slot_ranking_is_deterministic() -> None:
    counts = torch.tensor([5, 1, 1, 8, 2])
    mass = torch.tensor([1.0, 0.9, 0.2, 0.1, 0.1])
    kept, reallocated = rank_low_activity_slots(counts, mass, 2)
    assert torch.equal(reallocated, torch.tensor([2, 1]))
    assert torch.equal(kept, torch.tensor([0, 3, 4]))
