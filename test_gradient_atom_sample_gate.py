from __future__ import annotations

import torch

from evaluate_gradient_atom_sample_heads import sparse_head
from extract_document_activation_gradients import causal_mask, normalize_activation
from fit_gradient_atom_sample_heads import ridge_student, row_normalize


def test_causal_mask() -> None:
    mask = causal_mask(4, torch.device("cpu"), torch.float32)
    assert mask is not None
    assert torch.isneginf(mask[0, 1])
    assert mask[3, 0] == 0


def test_normalized_gradient_chain_rule() -> None:
    source = torch.randn(3, 5)
    normalized, scale = normalize_activation(source, 1.0e-6)
    normalized = normalized.detach().requires_grad_(True)
    reconstructed = normalized * scale + source.mean(dim=-1, keepdim=True)
    coefficient = torch.randn_like(source)
    gradient = torch.autograd.grad((reconstructed * coefficient).sum(), normalized)[0]
    assert torch.allclose(gradient, coefficient * scale)


def test_ridge_student_recovers_linear_map() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(32, 6, generator=generator)
    weight = torch.randn(6, 4, generator=generator)
    bias = torch.randn(4, generator=generator)
    y = x @ weight + bias
    fitted_weight, fitted_bias, ridge = ridge_student(x, y, 1.0e-7)
    prediction = x @ fitted_weight + fitted_bias
    assert ridge > 0
    assert torch.mean((prediction - y).square()) < 1.0e-5


def test_row_normalize() -> None:
    values = torch.randn(8, 5)
    normalized = row_normalize(values)
    assert torch.allclose(normalized.norm(dim=1), torch.ones(8), atol=1.0e-6)


def test_sparse_head_keeps_registered_top_k() -> None:
    artifact = {
        "activation_mean": torch.zeros(3),
        "activation_std": torch.ones(3),
        "heads.test.weight": torch.eye(3),
        "heads.test.bias": torch.zeros(3),
        "heads.test.mean": torch.zeros(3),
        "heads.test.std": torch.ones(3),
    }
    values = torch.tensor([[1.0, -3.0, 2.0]])
    output = sparse_head(values, artifact, "test", top_k=2)
    assert output.tolist() == [[0.0, -3.0, 2.0]]
