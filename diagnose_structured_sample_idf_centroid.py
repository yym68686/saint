#!/usr/bin/env python3
"""Gate IDF-weighted sample lexical predictability before SAE training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


INITIAL3 = [
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "fancyzhx/ag_news",
]


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("idf_centroid_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_weights(scores: torch.Tensor, tie_seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(tie_seed)
    tie_break = torch.rand(scores.shape, generator=generator) * 1.0e-7
    order = torch.argsort(scores.float() + tie_break)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
    if order.numel() == 1:
        return torch.ones(1, dtype=torch.float32)
    return 0.5 + ranks / (order.numel() - 1)


def distribution(
    values: torch.Tensor, max_quantile_values: int = 1_000_000
) -> dict[str, float | int]:
    values = values.detach().float().cpu().reshape(-1)
    value_count = int(values.numel())
    if value_count == 0:
        raise ValueError("Cannot summarize an empty tensor")
    quantile_stride = max(
        1,
        (value_count + max_quantile_values - 1) // max_quantile_values,
    )
    quantile_values = values[::quantile_stride][:max_quantile_values].contiguous()
    quantiles = torch.quantile(
        quantile_values, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
    )
    return {
        "value_count": value_count,
        "quantile_sample_count": int(quantile_values.numel()),
        "quantile_stride": quantile_stride,
        "mean": float(values.mean().item()),
        "std": float(values.std().item()),
        "min": float(values.min().item()),
        "p01": float(quantiles[0].item()),
        "p10": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "p90": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
        "max": float(values.max().item()),
    }


def sample_separating_cyclic_permutation(
    sample_ids: torch.Tensor, seed: int
) -> tuple[torch.Tensor, int]:
    length = int(sample_ids.numel())
    if length < 2:
        raise ValueError("Wrong-sample control requires at least two samples")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = torch.arange(length)
    for _ in range(1000):
        shift = int(torch.randint(1, length, (1,), generator=generator).item())
        permutation = torch.roll(identity, shifts=shift)
        if not torch.any(sample_ids[permutation] == sample_ids):
            return permutation, shift
    for shift in range(1, length):
        permutation = torch.roll(identity, shifts=shift)
        if not torch.any(sample_ids[permutation] == sample_ids):
            return permutation, shift
    raise RuntimeError("Could not construct a sample-separating permutation")


def score_from_cross_statistics(
    cross: torch.Tensor,
    sum_z: torch.Tensor,
    sum_target: torch.Tensor,
    var_z: torch.Tensor,
    var_target: torch.Tensor,
    count: int,
    block_features: int,
) -> torch.Tensor:
    feature_count = int(cross.shape[0])
    scores = torch.zeros(feature_count, dtype=torch.float32)
    for start in range(0, feature_count, block_features):
        end = min(feature_count, start + block_features)
        covariance = cross[start:end] - (
            sum_z[start:end, None] * sum_target[None, :] / count
        )
        denominator = (
            var_z[start:end, None] * var_target[None, :]
        ).clamp_min(0).sqrt()
        correlation = torch.where(
            denominator > 1.0e-12,
            covariance / denominator.clamp_min(1.0e-12),
            torch.zeros_like(covariance),
        ).clamp(-1, 1)
        scores[start:end] = correlation.square().mean(dim=1).sqrt().cpu()
    return scores


def mean_relu_features(
    module: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            sample_count = int(acts_cpu.shape[0])
            feature_count = int(state["encoder.weight"].shape[0])
            output = torch.zeros((sample_count, feature_count), dtype=torch.float32)
            for start in range(0, sample_count, config.sae_seq_batch_size):
                end = min(sample_count, start + config.sae_seq_batch_size)
                acts = acts_cpu[start:end].to(config.device, non_blocking=True)
                mask = mask_cpu[start:end].to(config.device, non_blocking=True)
                local_count = end - start
                local_ids = (
                    torch.arange(local_count, device=config.device)
                    .unsqueeze(1)
                    .expand_as(mask)
                    .reshape(-1)
                )
                flat_mask = mask.reshape(-1)
                x = acts.reshape(-1, acts.shape[-1])[flat_mask]
                sample_index = local_ids[flat_mask]
                lengths = mask.sum(dim=1).clamp_min(1)
                x = module.normalize_activation(
                    x, config.dtype, config.normalize_eps
                )
                z = torch.relu(
                    F.linear(
                        x - state["b_pre"],
                        state["encoder.weight"],
                        state["encoder.bias"],
                    )
                )
                sums = torch.zeros(
                    (local_count, feature_count),
                    device=config.device,
                    dtype=torch.float32,
                )
                sums.index_add_(0, sample_index, z.float())
                output[start:end] = (
                    sums / lengths.float().unsqueeze(1)
                ).cpu()
            result[class_name] = output
    return result


def scale_features(
    features: dict[str, torch.Tensor], weights: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {name: values * weights for name, values in features.items()}


def summarize(
    rows: list[dict[str, Any]], k_values: list[int]
) -> list[dict[str, Any]]:
    summary = []
    for row in rows:
        aggregate = {}
        for k in k_values:
            for metric in ("test_accuracy", "test_auc"):
                key = f"sae_top_{k}_{metric}"
                aggregate[key] = sum(
                    dataset[key] for dataset in row["dataset_results"].values()
                ) / len(row["dataset_results"])
        summary.append(
            {
                "label": row["label"],
                "variant_key": row["variant_key"],
                "mean_acc": sum(
                    aggregate[f"sae_top_{k}_test_accuracy"] for k in k_values
                )
                / len(k_values),
                "mean_auc": sum(
                    aggregate[f"sae_top_{k}_test_auc"] for k in k_values
                )
                / len(k_values),
                **{
                    f"top_{k}_acc": aggregate[f"sae_top_{k}_test_accuracy"]
                    for k in k_values
                },
            }
        )
    return sorted(summary, key=lambda row: row["mean_acc"], reverse=True)


def dataset_mean(row: dict[str, Any], dataset: str, k_values: list[int]) -> float:
    metrics = row["dataset_results"][dataset]
    return sum(
        metrics[f"sae_top_{k}_test_accuracy"] for k in k_values
    ) / len(k_values)


def load_output_weight(model_weights: Path) -> torch.Tensor:
    try:
        state = torch.load(
            model_weights,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state = torch.load(
            model_weights,
            map_location="cpu",
            weights_only=True,
        )
    output_weight = state["output.weight"]
    if output_weight.ndim != 2:
        raise ValueError(f"Unexpected output.weight shape: {tuple(output_weight.shape)}")
    del state
    return output_weight


def estimate_document_frequency(
    cache_dir: Path,
    manifest: dict[str, Any],
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    document_frequency = torch.zeros(vocab_size, dtype=torch.int64)
    token_frequency = torch.zeros(vocab_size, dtype=torch.int64)
    document_count = 0
    token_count = 0
    for shard in manifest["shards"]:
        meta = torch.load(
            cache_dir / shard["meta"]["path"],
            map_location="cpu",
            weights_only=True,
        )
        for token_ids, length_tensor in zip(meta["token_ids"], meta["lengths"]):
            length = int(length_tensor.item())
            ids = token_ids[:length].to(torch.int64)
            if int(ids.min().item()) < 0 or int(ids.max().item()) >= vocab_size:
                raise ValueError("Token ID is outside the output vocabulary")
            token_frequency += torch.bincount(ids, minlength=vocab_size)
            document_frequency[ids.unique()] += 1
            document_count += 1
            token_count += length
    expected_documents = int(manifest["summary"]["sample_count"])
    expected_tokens = int(manifest["summary"]["token_count"])
    if document_count != expected_documents or token_count != expected_tokens:
        raise RuntimeError(
            "Document-frequency pass disagrees with the cache manifest: "
            f"documents={document_count}/{expected_documents}, "
            f"tokens={token_count}/{expected_tokens}"
        )
    smooth_idf = (
        torch.log(
            torch.tensor(float(document_count + 1))
            / (document_frequency.float() + 1.0)
        )
        + 1.0
    )
    observed = document_frequency > 0
    document_fraction = document_frequency.float() / document_count
    weighted_mass = token_frequency.float() * smooth_idf
    bins = []
    edges = [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 0.05, 0.1, 0.5, 1.01]
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = observed & (document_fraction >= lower) & (document_fraction < upper)
        bins.append(
            {
                "document_fraction_lower": lower,
                "document_fraction_upper": upper,
                "token_type_count": int(mask.sum().item()),
                "token_mass_fraction": float(
                    token_frequency[mask].sum().item() / token_count
                ),
                "idf_weighted_mass_fraction": float(
                    weighted_mass[mask].sum().item()
                    / weighted_mass.sum().item()
                ),
            }
        )
    report = {
        "document_count": document_count,
        "token_count": token_count,
        "observed_token_types": int(observed.sum().item()),
        "smooth_idf_formula": "log((N+1)/(df+1))+1",
        "smooth_idf_observed": distribution(smooth_idf[observed]),
        "document_frequency_bins": bins,
    }
    return document_frequency, smooth_idf, report


def permute_observed_idf(
    smooth_idf: torch.Tensor,
    document_frequency: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, int, int]:
    observed = torch.nonzero(document_frequency > 0, as_tuple=False).flatten()
    if observed.numel() < 2:
        raise ValueError("IDF permutation requires at least two observed tokens")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shift = int(
        torch.randint(1, int(observed.numel()), (1,), generator=generator).item()
    )
    source = torch.roll(observed, shifts=shift)
    permuted = smooth_idf.clone()
    permuted[observed] = smooth_idf[source]
    fixed_token_ids = int((observed == source).sum().item())
    return permuted, shift, fixed_token_ids


def collect_sample_summaries(
    module: Any,
    cache_dir: Path,
    manifest: dict[str, Any],
    state: dict[str, torch.Tensor],
    output_weight: torch.Tensor,
    smooth_idf: torch.Tensor,
    permuted_idf: torch.Tensor,
    sample_count: int,
    min_sample_tokens: int,
    sample_batch_size: int,
    token_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[str, Any],
]:
    feature_count = int(state["encoder.weight"].shape[0])
    d_model = int(state["encoder.weight"].shape[1])
    vocab_size = int(output_weight.shape[0])
    feature_means = torch.empty((sample_count, feature_count), dtype=torch.float32)
    centroids = {
        "uniform_centroid": torch.empty(
            (sample_count, d_model), dtype=torch.float32
        ),
        "idf_centroid": torch.empty((sample_count, d_model), dtype=torch.float32),
        "token_idf_permuted": torch.empty(
            (sample_count, d_model), dtype=torch.float32
        ),
    }
    collected_sample_ids = torch.empty(sample_count, dtype=torch.int64)
    token_histogram = torch.zeros(vocab_size, dtype=torch.int64)
    sample_lengths: list[int] = []
    emitted = 0
    with torch.inference_mode():
        for shard in manifest["shards"]:
            meta = torch.load(
                cache_dir / shard["meta"]["path"],
                map_location="cpu",
                weights_only=True,
            )
            activations = torch.load(
                cache_dir / shard["layers"]["22"]["path"],
                map_location="cpu",
                weights_only=True,
            )
            offsets = meta["offsets"].to(torch.int64)
            lengths = meta["lengths"].to(torch.int64)
            sample_ids = meta["sample_ids"].to(torch.int64)
            token_ids = meta["token_ids"].to(torch.int64)
            attention_mask = meta["attention_mask"].to(torch.bool)
            eligible = torch.nonzero(
                lengths >= min_sample_tokens,
                as_tuple=False,
            ).flatten()
            for batch_start in range(0, int(eligible.numel()), sample_batch_size):
                selected = eligible[batch_start : batch_start + sample_batch_size]
                remaining = sample_count - emitted
                if remaining <= 0:
                    break
                selected = selected[:remaining]
                if selected.numel() == 0:
                    continue
                selected_lengths = lengths[selected]
                token_activations = []
                selected_tokens = []
                owners = []
                for owner, index_tensor in enumerate(selected):
                    index = int(index_tensor.item())
                    length = int(lengths[index].item())
                    if not bool(attention_mask[index, :length].all().item()):
                        raise RuntimeError(
                            "Stored attention mask disagrees with sample length"
                        )
                    packed_start = int(offsets[index].item())
                    token_activations.append(
                        activations[packed_start : packed_start + length]
                    )
                    ids = token_ids[index, :length]
                    if int(ids.min().item()) < 0 or int(ids.max().item()) >= vocab_size:
                        raise ValueError("Token ID is outside the output vocabulary")
                    selected_tokens.append(ids)
                    owners.append(torch.full((length,), owner, dtype=torch.int64))
                    sample_lengths.append(length)
                flat_activations = torch.cat(token_activations, dim=0)
                flat_tokens = torch.cat(selected_tokens, dim=0)
                flat_owners = torch.cat(owners, dim=0)
                token_histogram += torch.bincount(
                    flat_tokens,
                    minlength=vocab_size,
                )
                feature_sums = torch.zeros(
                    (selected.numel(), feature_count),
                    device=device,
                    dtype=torch.float32,
                )
                target_sums = torch.zeros(
                    (selected.numel(), d_model),
                    device=device,
                    dtype=torch.float32,
                )
                idf_target_sums = torch.zeros_like(target_sums)
                permuted_target_sums = torch.zeros_like(target_sums)
                idf_denominators = torch.zeros(
                    selected.numel(), device=device, dtype=torch.float32
                )
                permuted_denominators = torch.zeros_like(idf_denominators)
                for token_start in range(0, flat_tokens.numel(), token_batch_size):
                    token_end = min(
                        int(flat_tokens.numel()), token_start + token_batch_size
                    )
                    token_slice = flat_tokens[token_start:token_end].to(torch.int64)
                    x = module.normalize_activation(
                        flat_activations[token_start:token_end].to(
                            device=device,
                            non_blocking=True,
                        ),
                        dtype,
                        eps,
                    )
                    z = torch.relu(
                        F.linear(
                            x - state["b_pre"],
                            state["encoder.weight"],
                            state["encoder.bias"],
                        )
                    ).float()
                    target = F.normalize(
                        output_weight[token_slice].float(),
                        dim=1,
                        eps=1.0e-12,
                    ).to(device=device, non_blocking=True)
                    idf_weight = smooth_idf[token_slice].to(
                        device=device, non_blocking=True
                    )
                    permuted_weight = permuted_idf[token_slice].to(
                        device=device, non_blocking=True
                    )
                    owner = flat_owners[token_start:token_end].to(
                        device=device,
                        non_blocking=True,
                    )
                    feature_sums.index_add_(0, owner, z)
                    target_sums.index_add_(0, owner, target)
                    idf_target_sums.index_add_(
                        0, owner, target * idf_weight.unsqueeze(1)
                    )
                    permuted_target_sums.index_add_(
                        0, owner, target * permuted_weight.unsqueeze(1)
                    )
                    idf_denominators.index_add_(0, owner, idf_weight)
                    permuted_denominators.index_add_(
                        0, owner, permuted_weight
                    )
                    del (
                        x,
                        z,
                        target,
                        idf_weight,
                        permuted_weight,
                        token_slice,
                        owner,
                    )
                denominators = selected_lengths.to(
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(1)
                batch_feature_means = feature_sums / denominators
                batch_centroids = {
                    "uniform_centroid": F.normalize(
                        target_sums / denominators,
                        dim=1,
                        eps=1.0e-12,
                    ),
                    "idf_centroid": F.normalize(
                        idf_target_sums
                        / idf_denominators.clamp_min(1.0e-12).unsqueeze(1),
                        dim=1,
                        eps=1.0e-12,
                    ),
                    "token_idf_permuted": F.normalize(
                        permuted_target_sums
                        / permuted_denominators.clamp_min(1.0e-12).unsqueeze(1),
                        dim=1,
                        eps=1.0e-12,
                    ),
                }
                batch_end = emitted + int(selected.numel())
                feature_means[emitted:batch_end] = batch_feature_means.cpu()
                for key, values in batch_centroids.items():
                    centroids[key][emitted:batch_end] = values.cpu()
                collected_sample_ids[emitted:batch_end] = sample_ids[selected]
                emitted = batch_end
                del (
                    flat_activations,
                    flat_tokens,
                    flat_owners,
                    feature_sums,
                    target_sums,
                    idf_target_sums,
                    permuted_target_sums,
                    idf_denominators,
                    permuted_denominators,
                    denominators,
                    batch_feature_means,
                )
                del batch_centroids
            del activations, meta
            if emitted >= sample_count:
                break
    if emitted != sample_count:
        raise RuntimeError(f"Expected {sample_count} samples, got {emitted}")
    probabilities = token_histogram[token_histogram > 0].float()
    probabilities /= probabilities.sum()
    token_entropy = float(-(probabilities * probabilities.log()).sum().item())
    metadata = {
        "sample_count": sample_count,
        "token_count": int(sum(sample_lengths)),
        "min_sample_tokens": min_sample_tokens,
        "sample_length_mean": float(torch.tensor(sample_lengths).float().mean().item()),
        "sample_length_min": int(min(sample_lengths)),
        "sample_length_max": int(max(sample_lengths)),
        "unique_token_count": int((token_histogram > 0).sum().item()),
        "token_entropy_nats": token_entropy,
        "token_perplexity": math.exp(token_entropy),
    }
    return feature_means, centroids, collected_sample_ids, metadata


def estimate_idf_centroid_weights(
    module: Any,
    checkpoint: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
    model_weights: Path,
    sample_count: int,
    min_sample_tokens: int,
    sample_batch_size: int,
    token_batch_size: int,
    cross_batch_samples: int,
    score_feature_block: int,
    permutation_seed: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw = module.load_state(checkpoint)
    state = module.move_keys(
        raw,
        ["b_pre", "encoder.weight", "encoder.bias"],
        device,
        dtype,
    )
    del raw
    output_weight = load_output_weight(model_weights)
    document_frequency, smooth_idf, document_frequency_report = (
        estimate_document_frequency(
            cache_dir,
            manifest,
            int(output_weight.shape[0]),
        )
    )
    permuted_idf, token_idf_shift, token_idf_fixed_count = permute_observed_idf(
        smooth_idf,
        document_frequency,
        permutation_seed + 11,
    )
    feature_means, centroids, sample_ids, sample_report = (
        collect_sample_summaries(
            module,
            cache_dir,
            manifest,
            state,
            output_weight,
            smooth_idf,
            permuted_idf,
            sample_count,
            min_sample_tokens,
            sample_batch_size,
            token_batch_size,
            device,
            dtype,
            eps,
        )
    )
    wrong_permutation, wrong_shift = sample_separating_cyclic_permutation(
        sample_ids, permutation_seed
    )
    feature_count = int(state["encoder.weight"].shape[0])
    d_model = int(state["encoder.weight"].shape[1])
    target_names = [
        "uniform_centroid",
        "idf_centroid",
        "token_idf_permuted",
        "wrong_alignment",
    ]
    target_blocks = [
        centroids["uniform_centroid"],
        centroids["idf_centroid"],
        centroids["token_idf_permuted"],
        centroids["idf_centroid"][wrong_permutation],
    ]
    all_targets = torch.cat(target_blocks, dim=1)
    sum_target = all_targets.sum(dim=0).to(device)
    sum_target2 = all_targets.square().sum(dim=0).to(device)
    var_target = (sum_target2 - sum_target.square() / sample_count).clamp_min(0)
    sum_z = torch.zeros(feature_count, device=device, dtype=torch.float32)
    sum_z2 = torch.zeros_like(sum_z)
    cross = torch.zeros(
        (feature_count, len(target_names) * d_model),
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        for start in range(0, sample_count, cross_batch_samples):
            end = min(sample_count, start + cross_batch_samples)
            z = feature_means[start:end].to(device=device, non_blocking=True)
            target = all_targets[start:end].to(
                device=device, non_blocking=True
            )
            sum_z += z.sum(dim=0)
            sum_z2 += z.square().sum(dim=0)
            cross.addmm_(z.T, target)
            del z, target
    var_z = (sum_z2 - sum_z.square() / sample_count).clamp_min(0)
    scores = {}
    target_variances = {}
    for index, name in enumerate(target_names):
        start = index * d_model
        end = start + d_model
        scores[name] = score_from_cross_statistics(
            cross[:, start:end],
            sum_z,
            sum_target[start:end],
            var_z,
            var_target[start:end],
            sample_count,
            score_feature_block,
        )
        target_variances[name] = distribution(
            var_target[start:end].cpu() / sample_count
        )
    del cross, state
    torch.cuda.empty_cache()

    score_weights = {
        name: rank_weights(score, permutation_seed + 101)
        for name, score in scores.items()
    }
    feature_generator = torch.Generator(device="cpu").manual_seed(
        permutation_seed + 201
    )
    feature_permutation = torch.randperm(
        feature_count, generator=feature_generator
    )
    weights = {
        "raw": torch.ones(feature_count, dtype=torch.float32),
        "uniform_centroid": score_weights["uniform_centroid"],
        "idf_centroid": score_weights["idf_centroid"],
        "token_idf_permuted": score_weights["token_idf_permuted"],
        "feature_permuted": score_weights["idf_centroid"][feature_permutation],
        "wrong_alignment": score_weights["wrong_alignment"],
    }
    report = {
        "signal_definition": (
            "sqrt(mean_d corr(mean_t z_s,t,j, "
            "l2_normalize(sum_t smooth_idf(token_s,t) * "
            "l2_normalize(W_out[token_s,t])))_d^2)"
        ),
        "smooth_idf_formula": "log((N+1)/(df+1))+1",
        **sample_report,
        "document_frequency": document_frequency_report,
        "feature_count": feature_count,
        "d_model": d_model,
        "vocab_size": int(output_weight.shape[0]),
        "permutation_seed": permutation_seed,
        "token_idf_permutation_seed": permutation_seed + 11,
        "token_idf_cyclic_shift": token_idf_shift,
        "token_idf_fixed_token_count": token_idf_fixed_count,
        "wrong_cyclic_shift": wrong_shift,
        "wrong_fixed_pair_count": int(
            (wrong_permutation == torch.arange(sample_count)).sum().item()
        ),
        "wrong_same_sample_pair_count": int(
            (sample_ids[wrong_permutation] == sample_ids).sum().item()
        ),
        "shared_rank_tie_seed": permutation_seed + 101,
        "feature_permutation_seed": permutation_seed + 201,
        "centroids": {
            name: distribution(values) for name, values in centroids.items()
        },
        "target_coordinate_variance": target_variances,
        "feature_variance": distribution(var_z.cpu() / sample_count),
        "scores": {name: distribution(score) for name, score in scores.items()},
        "weights": {
            name: distribution(weight)
            for name, weight in score_weights.items()
        },
        "score_correlation_names": target_names,
        "score_correlation_matrix": torch.corrcoef(
            torch.stack([scores[name] for name in target_names])
        ).tolist(),
    }
    return weights, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path(
            "/root/autodl-tmp/saebench_sparse_probing_all_architectures.py"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-weights", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--min-sample-tokens", type=int, default=4)
    parser.add_argument("--sample-batch-size", type=int, default=32)
    parser.add_argument("--token-batch-size", type=int, default=128)
    parser.add_argument("--cross-batch-samples", type=int, default=32)
    parser.add_argument("--score-feature-block", type=int, default=2048)
    parser.add_argument("--permutation-seed", type=int, default=47026)
    parser.add_argument("--datasets", nargs="+", default=INITIAL3)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--sae-seq-batch-size", type=int, default=2)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    module = load_eval_module(args.eval_script)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    started = time.time()
    weights, idf_report = estimate_idf_centroid_weights(
        module,
        args.checkpoint,
        args.cache_dir,
        manifest,
        args.model_weights,
        args.sample_count,
        args.min_sample_tokens,
        args.sample_batch_size,
        args.token_batch_size,
        args.cross_batch_samples,
        args.score_feature_block,
        args.permutation_seed,
        device,
        dtype,
        1.0e-6,
    )
    torch.save(weights, args.output_dir / "sample-idf-centroid-weights.pt")
    (args.output_dir / "sample-idf-centroid-statistics.json").write_text(
        json.dumps(idf_report, indent=2) + "\n", encoding="utf-8"
    )

    kwargs = {
        "model_dir": args.model_dir,
        "datasets": args.datasets,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "context_length": args.context_length,
        "llm_batch_size": args.llm_batch_size,
        "sae_seq_batch_size": args.sae_seq_batch_size,
        "k_values": args.k_values,
        "random_seed": args.random_seed,
        "dtype_name": args.dtype,
        "dtype": dtype,
        "device": device,
        "normalize_eps": 1.0e-6,
    }
    allowed = set(inspect.signature(module.EvalConfig).parameters)
    config = module.EvalConfig(
        **{key: value for key, value in kwargs.items() if key in allowed}
    )
    tokenizer = module.Tokenizer(str(args.model_dir / "tokenizer.model"))
    llm = module.load_model(args.model_dir, [22], config.device, config.dtype)
    raw = module.load_state(args.checkpoint)
    state = module.move_keys(
        raw,
        ["b_pre", "encoder.weight", "encoder.bias"],
        config.device,
        config.dtype,
    )
    del raw
    cached: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        print(f"== Cache dataset: {dataset}", flush=True)
        train_data, test_data = module.get_multi_label_train_test_data(
            dataset,
            args.train_size,
            args.test_size,
            args.random_seed,
        )
        train_layers, train_masks = module.collect_layer_activations(
            llm, tokenizer, train_data, config, [22]
        )
        test_layers, test_masks = module.collect_layer_activations(
            llm, tokenizer, test_data, config, [22]
        )
        cached[dataset] = {
            "train": mean_relu_features(
                module, train_layers[22], train_masks, state, config
            ),
            "test": mean_relu_features(
                module, test_layers[22], test_masks, state, config
            ),
        }
        del train_layers, test_layers, train_masks, test_masks
    del llm, state
    torch.cuda.empty_cache()
    gc.collect()

    variants = [
        ("raw", "L22 ReLU reference"),
        ("uniform_centroid", "uniform lexical-centroid control"),
        ("idf_centroid", "true-sample IDF lexical-centroid weighting"),
        ("token_idf_permuted", "wrong-token-IDF mapping control"),
        ("feature_permuted", "feature-permuted IDF-weight control"),
        ("wrong_alignment", "wrong-sample IDF-centroid control"),
    ]
    rows = []
    for variant_key, label in variants:
        print(f"== Variant: {label}", flush=True)
        row = {
            "variant_key": variant_key,
            "label": label,
            "dataset_results": {},
            "per_class": {},
        }
        weight = weights[variant_key]
        for dataset_index, (dataset, features) in enumerate(cached.items()):
            train_features = scale_features(features["train"], weight)
            test_features = scale_features(features["test"], weight)
            probe = module.probe_one_architecture_dataset(
                train_features,
                test_features,
                args.k_values,
                args.random_seed + 1009 * dataset_index,
            )
            row["dataset_results"][dataset] = probe["metrics"]
            row["per_class"][dataset] = probe["per_class"]
            print(
                f"   {dataset}: "
                + " ".join(
                    f"k{k}={probe['metrics'][f'sae_top_{k}_test_accuracy']:.4f}"
                    for k in args.k_values
                ),
                flush=True,
            )
        rows.append(row)

    summary = summarize(rows, args.k_values)
    summary_by_key = {row["variant_key"]: row for row in summary}
    rows_by_key = {row["variant_key"]: row for row in rows}
    candidate = summary_by_key["idf_centroid"]
    reference = summary_by_key["raw"]
    uniform = summary_by_key["uniform_centroid"]
    token_permuted = summary_by_key["token_idf_permuted"]
    feature_permuted = summary_by_key["feature_permuted"]
    wrong = summary_by_key["wrong_alignment"]
    dataset_deltas = {
        dataset: {
            "reference": dataset_mean(
                rows_by_key["raw"], dataset, args.k_values
            ),
            "uniform_centroid": dataset_mean(
                rows_by_key["uniform_centroid"], dataset, args.k_values
            ),
            "idf_centroid": dataset_mean(
                rows_by_key["idf_centroid"], dataset, args.k_values
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta_reference"] = values["idf_centroid"] - values["reference"]
        values["delta_uniform"] = (
            values["idf_centroid"] - values["uniform_centroid"]
        )
    gate = {
        "candidate_minus_reference_at_least_0p005": (
            candidate["mean_acc"] - reference["mean_acc"] >= 0.005
        ),
        "no_dataset_drop_below_minus_0p01": all(
            values["delta_reference"] >= -0.01
            for values in dataset_deltas.values()
        ),
        "candidate_minus_uniform_at_least_0p002": (
            candidate["mean_acc"] - uniform["mean_acc"] >= 0.002
        ),
        "candidate_minus_token_idf_permuted_at_least_0p002": (
            candidate["mean_acc"] - token_permuted["mean_acc"] >= 0.002
        ),
        "candidate_minus_feature_permuted_at_least_0p002": (
            candidate["mean_acc"] - feature_permuted["mean_acc"] >= 0.002
        ),
        "candidate_minus_wrong_alignment_at_least_0p002": (
            candidate["mean_acc"] - wrong["mean_acc"] >= 0.002
        ),
    }
    gate["pass"] = all(gate.values())
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "cache_dir": str(args.cache_dir),
            "cache_manifest_sha256": sha256(manifest_path),
            "model_weights": str(args.model_weights),
            "model_weights_size": args.model_weights.stat().st_size,
            "layer": 22,
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "sample_idf_centroid_signal": idf_report,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-structured-idf-centroid-sae-v2-screen"
            if gate["pass"]
            else "stop-before-structured-idf-centroid-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "sample-idf-centroid-gate.json"
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Structured sample IDF-centroid gate",
        "",
        "| Variant | Mean Acc | Mean AUC | Top-1 | Top-2 | Top-5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['label']} | {row['mean_acc']:.6f} | "
            f"{row['mean_auc']:.6f} | {row['top_1_acc']:.6f} | "
            f"{row['top_2_acc']:.6f} | {row['top_5_acc']:.6f} |"
        )
    lines.extend(["", f"Decision: `{payload['decision']}`", ""])
    (args.output_dir / "sample-idf-centroid-gate.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {"summary": summary, "gate": gate, "decision": payload["decision"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
