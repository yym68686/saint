#!/usr/bin/env python3
"""Fit gradient atoms and activation-only students on unlabeled OWT documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import MiniBatchDictionaryLearning


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def row_normalize(values: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return values / values.norm(dim=1, keepdim=True).clamp_min(eps)


def ridge_student(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    ridge_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    input_mean = train_x.mean(dim=0)
    target_mean = train_y.mean(dim=0)
    centered_input = train_x - input_mean
    centered_target = train_y - target_mean
    kernel = centered_input @ centered_input.T
    ridge = ridge_scale * float(kernel.diagonal().mean().item())
    dual = torch.linalg.solve(
        kernel + ridge * torch.eye(kernel.shape[0], dtype=kernel.dtype),
        centered_target,
    )
    weight = centered_input.T @ dual
    bias = target_mean - input_mean @ weight
    return weight, bias, ridge


def prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    top_k: int,
) -> dict[str, float]:
    prediction_centered = prediction - prediction.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    cosine = F.cosine_similarity(prediction_centered, target_centered, dim=1)
    residual = (prediction - target).square().sum()
    total = (target - target.mean(dim=0, keepdim=True)).square().sum()
    pred_top = torch.topk(prediction.abs(), k=top_k, dim=1).indices
    target_top = torch.topk(target.abs(), k=top_k, dim=1).indices
    overlaps = []
    for left, right in zip(pred_top, target_top, strict=True):
        overlaps.append(len(set(left.tolist()) & set(right.tolist())) / top_k)
    return {
        "centered_cosine_mean": float(cosine.mean().item()),
        "centered_cosine_std": float(cosine.std(unbiased=False).item()),
        "r2": float((1.0 - residual / total.clamp_min(1.0e-30)).item()),
        "topk_overlap_mean": float(sum(overlaps) / len(overlaps)),
    }


def fit_output_calibration(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return values.mean(dim=0), values.std(dim=0).clamp_min(1.0e-6)


def v396_features(
    values: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    normalized = (
        values.float() - values.float().mean(dim=-1, keepdim=True)
    ) / (values.float().std(dim=-1, keepdim=True) + 1.0e-6)
    centered = normalized - state["b_pre"]
    preactivation = F.linear(
        centered,
        state["encoder.weight"],
        state["encoder.bias"],
    )
    positive = torch.relu(preactivation)
    beta = F.softplus(state["v396.raw_beta"].float()).clamp(
        1.0e-4,
        float(state["v396.max_beta"].item()),
    )
    features = torch.log1p(positive * beta) / torch.log1p(beta)
    gain = state["v396.log_gain"].float().clamp(
        -float(state["v396.max_log_gain"].item()),
        float(state["v396.max_log_gain"].item()),
    ).exp()
    return features * gain


def load_v396_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    keys = (
        "b_pre",
        "encoder.weight",
        "encoder.bias",
        "v396.raw_beta",
        "v396.log_gain",
        "v396.max_beta",
        "v396.max_log_gain",
    )
    missing = sorted(set(keys) - set(raw))
    if missing:
        raise KeyError(f"V396 checkpoint missing {missing}")
    return {key: raw[key].float() for key in keys}


def collect_slot_statistics(
    cache_dir: Path,
    state: dict[str, torch.Tensor],
    slot_indices: torch.Tensor,
    token_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_device = {key: value.to(device) for key, value in state.items()}
    sums = torch.zeros(len(slot_indices), dtype=torch.float64)
    squared = torch.zeros_like(sums)
    seen = 0
    for path in sorted(cache_dir.glob("*.pt")):
        values = torch.load(path, map_location="cpu", weights_only=True)
        for start in range(0, int(values.shape[0]), batch_size):
            if seen >= token_count:
                break
            take = min(batch_size, token_count - seen, int(values.shape[0]) - start)
            chunk = values[start : start + take].to(device)
            with torch.inference_mode():
                selected = v396_features(chunk, state_device).index_select(
                    1, slot_indices.to(device)
                ).double().cpu()
            sums += selected.sum(dim=0)
            squared += selected.square().sum(dim=0)
            seen += take
        if seen >= token_count:
            break
    if seen != token_count:
        raise ValueError(f"Collected {seen} tokens, expected {token_count}")
    mean = sums / seen
    variance = (squared / seen - mean.square()).clamp_min(1.0e-12)
    return mean.float(), variance.sqrt().float()


def atom_coherence(
    coefficients: torch.Tensor,
    raw_gradients: torch.Tensor,
    top_documents: int,
) -> dict[str, float]:
    normalized_gradients = row_normalize(raw_gradients)
    scores = []
    random_scores = []
    generator = torch.Generator().manual_seed(42027)
    for atom_index in range(coefficients.shape[1]):
        count = min(top_documents, coefficients.shape[0])
        indices = torch.topk(coefficients[:, atom_index].abs(), k=count).indices
        selected = normalized_gradients.index_select(0, indices)
        cosine = selected @ selected.T
        mask = ~torch.eye(count, dtype=torch.bool)
        scores.append(float(cosine[mask].mean().item()))
        random_indices = torch.randperm(
            coefficients.shape[0], generator=generator
        )[:count]
        random_selected = normalized_gradients.index_select(0, random_indices)
        random_cosine = random_selected @ random_selected.T
        random_scores.append(float(random_cosine[mask].mean().item()))
    return {
        "atom_mean": float(np.mean(scores)),
        "atom_median": float(np.median(scores)),
        "random_mean": float(np.mean(random_scores)),
        "random_median": float(np.median(random_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--v396-checkpoint", type=Path, required=True)
    parser.add_argument("--flat-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=768)
    parser.add_argument("--atom-count", type=int, default=256)
    parser.add_argument("--dict-alpha", type=float, default=0.1)
    parser.add_argument("--dict-max-iter", type=int, default=100)
    parser.add_argument("--ridge-scale", type=float, default=0.01)
    parser.add_argument("--head-top-k", type=int, default=4)
    parser.add_argument("--slot-seed", type=int, default=42026)
    parser.add_argument("--slot-stat-tokens", type=int, default=32768)
    parser.add_argument("--slot-stat-batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    extraction = torch.load(args.extraction, map_location="cpu", weights_only=True)
    gradients = extraction["gradients"].float()
    activations = extraction["activation_means"].float()
    sample_count, hidden_size = gradients.shape
    if activations.shape != gradients.shape:
        raise ValueError("Gradient and activation shapes differ")
    if not (args.atom_count < args.train_count < sample_count):
        raise ValueError("Need atom_count < train_count < sample_count")
    device = torch.device(args.device)
    train_gradients = gradients[: args.train_count]
    train_activations = activations[: args.train_count]
    holdout_gradients = gradients[args.train_count :]
    holdout_activations = activations[args.train_count :]
    activation_mean = train_activations.mean(dim=0)
    activation_std = train_activations.std(dim=0).clamp_min(1.0e-6)
    train_x = (train_activations - activation_mean) / activation_std
    holdout_x = (holdout_activations - activation_mean) / activation_std

    gradient_unit = row_normalize(gradients)
    train_gradient_unit = gradient_unit[: args.train_count].to(device)
    _, singular_values, right_vectors = torch.linalg.svd(
        train_gradient_unit,
        full_matrices=False,
    )
    right_vectors = right_vectors[: args.atom_count]
    singular_values = singular_values[: args.atom_count]
    eigenvalues = singular_values.square() / args.train_count
    projected = (
        gradient_unit.to(device) @ right_vectors.T
    ) / eigenvalues.sqrt().clamp_min(1.0e-6)
    projected = row_normalize(projected).cpu()
    del train_gradient_unit
    torch.cuda.empty_cache()

    dictionary = MiniBatchDictionaryLearning(
        n_components=args.atom_count,
        alpha=args.dict_alpha,
        batch_size=128,
        max_iter=args.dict_max_iter,
        transform_algorithm="lasso_lars",
        random_state=args.random_seed,
        n_jobs=-1,
    )
    projected_numpy = projected.numpy()
    dictionary.fit(projected_numpy[: args.train_count])
    coefficients = torch.from_numpy(dictionary.transform(projected_numpy)).float()
    components = torch.from_numpy(dictionary.components_).float()
    gradient_atom_directions = (
        components.to(device) * eigenvalues.sqrt().unsqueeze(0)
    ) @ right_vectors
    gradient_atom_directions = row_normalize(gradient_atom_directions).cpu()

    true_weight, true_bias, true_ridge = ridge_student(
        train_x,
        coefficients[: args.train_count],
        args.ridge_scale,
    )
    shift = max(1, args.train_count // 3)
    wrong_target = torch.roll(coefficients[: args.train_count], shifts=shift, dims=0)
    wrong_weight, wrong_bias, wrong_ridge = ridge_student(
        train_x,
        wrong_target,
        args.ridge_scale,
    )
    pca_target = projected
    pca_weight, pca_bias, pca_ridge = ridge_student(
        train_x,
        pca_target[: args.train_count],
        args.ridge_scale,
    )
    _, _, activation_vectors = torch.linalg.svd(
        train_x.to(device),
        full_matrices=False,
    )
    activation_weight = activation_vectors[: args.atom_count].T.cpu()
    activation_bias = torch.zeros(args.atom_count)
    generator = torch.Generator().manual_seed(args.random_seed + 1)
    random_matrix = torch.randn(
        hidden_size,
        args.atom_count,
        generator=generator,
    )
    random_weight = torch.linalg.qr(random_matrix, mode="reduced").Q
    random_bias = torch.zeros(args.atom_count)

    heads = {
        "gradient_atom_student": (true_weight, true_bias),
        "wrong_alignment_student": (wrong_weight, wrong_bias),
        "gradient_pca_student": (pca_weight, pca_bias),
        "activation_pca_control": (activation_weight, activation_bias),
        "random_orthogonal_control": (random_weight, random_bias),
    }
    calibrations: dict[str, dict[str, torch.Tensor]] = {}
    heldout_predictions: dict[str, torch.Tensor] = {}
    for name, (weight, bias) in heads.items():
        train_prediction = train_x @ weight + bias
        output_mean, output_std = fit_output_calibration(train_prediction)
        calibrations[name] = {"mean": output_mean, "std": output_std}
        heldout_predictions[name] = holdout_x @ weight + bias

    true_holdout_target = coefficients[args.train_count :]
    prediction_report = {
        name: prediction_metrics(
            prediction,
            true_holdout_target,
            args.head_top_k,
        )
        for name, prediction in heldout_predictions.items()
    }
    coefficient_active = coefficients.abs() > 1.0e-8
    coherence = atom_coherence(
        coefficients[: args.train_count],
        train_gradients,
        top_documents=10,
    )

    v396_state = load_v396_state(args.v396_checkpoint)
    latent_count = int(v396_state["encoder.weight"].shape[0])
    slot_generator = torch.Generator().manual_seed(args.slot_seed)
    slot_indices = torch.randperm(latent_count, generator=slot_generator)[
        : args.atom_count
    ].sort().values
    slot_mean, slot_std = collect_slot_statistics(
        args.flat_cache_dir,
        v396_state,
        slot_indices,
        args.slot_stat_tokens,
        args.slot_stat_batch_size,
        device,
    )

    artifact: dict[str, object] = {
        "activation_mean": activation_mean,
        "activation_std": activation_std,
        "gradient_atom_directions": gradient_atom_directions,
        "gradient_pca_vectors": right_vectors.cpu(),
        "gradient_pca_eigenvalues": eigenvalues.cpu(),
        "slot_indices": slot_indices,
        "slot_mean": slot_mean,
        "slot_std": slot_std,
        "head_top_k": torch.tensor(args.head_top_k),
        "train_count": torch.tensor(args.train_count),
        "atom_count": torch.tensor(args.atom_count),
    }
    for name, (weight, bias) in heads.items():
        artifact[f"heads.{name}.weight"] = weight
        artifact[f"heads.{name}.bias"] = bias
        artifact[f"heads.{name}.mean"] = calibrations[name]["mean"]
        artifact[f"heads.{name}.std"] = calibrations[name]["std"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    report = {
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "extraction": str(args.extraction),
        "extraction_sha256": sha256_file(args.extraction),
        "v396_checkpoint": str(args.v396_checkpoint),
        "v396_checkpoint_sha256": sha256_file(args.v396_checkpoint),
        "sample_count": sample_count,
        "train_count": args.train_count,
        "holdout_count": sample_count - args.train_count,
        "hidden_size": hidden_size,
        "atom_count": args.atom_count,
        "dict_alpha": args.dict_alpha,
        "dict_max_iter": args.dict_max_iter,
        "ridge_scale": args.ridge_scale,
        "true_ridge": true_ridge,
        "wrong_ridge": wrong_ridge,
        "pca_ridge": pca_ridge,
        "wrong_alignment_shift": shift,
        "wrong_alignment_fixed_points": 0,
        "coefficient_l0_mean": float(coefficient_active.float().sum(dim=1).mean().item()),
        "coefficient_zero_row_fraction": float(
            (~coefficient_active.any(dim=1)).float().mean().item()
        ),
        "atom_coherence": coherence,
        "holdout_prediction": prediction_report,
        "slot_seed": args.slot_seed,
        "slot_stat_tokens": args.slot_stat_tokens,
        "slot_std_mean": float(slot_std.mean().item()),
        "slot_std_min": float(slot_std.min().item()),
        "slot_std_max": float(slot_std.max().item()),
        "uses_saebench_labels": False,
        "uses_saebench_class_names": False,
        "uses_eval_split": False,
        "uses_test_feedback": False,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
