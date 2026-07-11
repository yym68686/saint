#!/usr/bin/env python3
"""Route each frozen SAE feature to its more reproducible sample moment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


INITIAL3 = [
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "fancyzhx/ag_news",
]
LOW_K = [1, 2, 5]
WIDE_K = [3, 10, 20]


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("stable_moment_routing_eval", path)
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


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    quantiles = torch.quantile(
        values, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
    )
    return {
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


def derangement(length: int, seed: int) -> torch.Tensor:
    if length < 2:
        raise ValueError("A wrong-sample control requires at least two samples")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = torch.arange(length)
    for _ in range(100):
        permutation = torch.randperm(length, generator=generator)
        if not torch.any(permutation == identity):
            return permutation
    shift = int(torch.randint(1, length, (1,), generator=generator).item())
    return torch.roll(identity, shifts=shift)


def pack_samples(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([sample.shape[0] for sample in samples], dtype=torch.int64)
    activations = torch.cat(samples, dim=0).contiguous()
    sample_index = torch.repeat_interleave(
        torch.arange(len(samples), dtype=torch.int64), lengths
    )
    view_index = torch.cat(
        [torch.arange(length, dtype=torch.int64) % 2 for length in lengths.tolist()]
    )
    return {
        "activations": activations,
        "sample_index": sample_index,
        "view_index": view_index,
        "lengths": lengths,
    }


def iter_true_sample_batches(
    cache_dir: Path,
    manifest: dict[str, Any],
    layer: int,
    max_samples: int,
    batch_samples: int,
    min_sample_tokens: int,
) -> Iterator[dict[str, torch.Tensor]]:
    pending: list[torch.Tensor] = []
    emitted = 0
    for shard in manifest["shards"]:
        meta = torch.load(
            cache_dir / shard["meta"]["path"],
            map_location="cpu",
            weights_only=True,
        )
        activations = torch.load(
            cache_dir / shard["layers"][str(layer)]["path"],
            map_location="cpu",
            weights_only=True,
        )
        offsets = meta["offsets"].to(torch.int64)
        lengths = meta["lengths"].to(torch.int64)
        for index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            if length < min_sample_tokens:
                continue
            start = int(offsets[index].item())
            end = int(offsets[index + 1].item())
            pending.append(activations[start:end].clone())
            if len(pending) == batch_samples:
                yield pack_samples(pending)
                emitted += len(pending)
                pending = []
                if emitted >= max_samples:
                    return
        del activations, meta
    if emitted != max_samples:
        raise RuntimeError(f"Expected {max_samples} complete samples, got {emitted}")


def correlation_from_sums(
    sum_a: torch.Tensor,
    sum_b: torch.Tensor,
    sum_a2: torch.Tensor,
    sum_b2: torch.Tensor,
    sum_ab: torch.Tensor,
    count: int,
) -> torch.Tensor:
    covariance = sum_ab - sum_a * sum_b / count
    variance_a = (sum_a2 - sum_a.square() / count).clamp_min(0.0)
    variance_b = (sum_b2 - sum_b.square() / count).clamp_min(0.0)
    denominator = torch.sqrt(variance_a * variance_b)
    correlation = torch.where(
        denominator > 1.0e-12,
        covariance / denominator.clamp_min(1.0e-12),
        torch.zeros_like(covariance),
    )
    return correlation.clamp(-1.0, 1.0)


def select_mean_diff(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    k: int,
) -> torch.Tensor:
    positive = train_y == 1
    negative = train_y == 0
    score = (
        train_x[positive].mean(dim=0) - train_x[negative].mean(dim=0)
    ).abs()
    return torch.argsort(score, descending=True)[:k]


def select_effect_size(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    k: int,
) -> torch.Tensor:
    positive = train_y == 1
    negative = train_y == 0
    positive_x = train_x[positive].float()
    negative_x = train_x[negative].float()
    difference = positive_x.mean(dim=0) - negative_x.mean(dim=0)
    pooled_variance = 0.5 * (
        positive_x.var(dim=0, unbiased=False) + negative_x.var(dim=0, unbiased=False)
    )
    score = difference.abs() / torch.sqrt(pooled_variance + 1.0e-8)
    return torch.argsort(score, descending=True)[:k]


def pool_sample_statistics(
    token_features: torch.Tensor,
    sample_ids: torch.Tensor,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    feature_count = int(token_features.shape[1])
    sums = torch.zeros(
        (sample_count, feature_count),
        device=token_features.device,
        dtype=torch.float32,
    )
    square_sums = torch.zeros_like(sums)
    counts = torch.bincount(sample_ids, minlength=sample_count).float().clamp_min(1.0)
    counts = counts.unsqueeze(1)
    values = token_features.float()
    sums.index_add_(0, sample_ids, values)
    square_sums.index_add_(0, sample_ids, values.square())
    mean = sums / counts
    mean_square = square_sums / counts
    variance = (mean_square - mean.square()).clamp_min(0.0)
    standard_deviation = torch.sqrt(variance)
    return {
        "mean_pool": mean,
        "std_pool": standard_deviation,
        "mean_std_pool": mean + standard_deviation,
        "rms_pool": torch.sqrt(mean_square.clamp_min(0.0)),
    }


def estimate_stable_moment_routes(
    module: Any,
    checkpoint: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
    sample_count: int,
    batch_samples: int,
    min_sample_tokens: int,
    permutation_seed: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if sample_count % batch_samples != 0:
        raise ValueError("sample_count must be divisible by batch_samples")
    raw = module.load_state(checkpoint)
    state = module.move_keys(
        raw,
        ["b_pre", "encoder.weight", "encoder.bias"],
        device,
        dtype,
    )
    del raw
    feature_count = int(state["encoder.weight"].shape[0])
    statistic_names = ("mean", "std")
    accumulator_names = (
        "sum_a",
        "sum_b",
        "sum_a2",
        "sum_b2",
        "sum_ab",
        "sum_ab_wrong",
    )
    accumulators = {
        statistic: {
            name: torch.zeros(feature_count, device=device, dtype=torch.float32)
            for name in accumulator_names
        }
        for statistic in statistic_names
    }
    samples_seen = 0
    tokens_seen = 0
    view_token_counts = [0, 0]
    with torch.inference_mode():
        for batch_index, batch in enumerate(
            iter_true_sample_batches(
                cache_dir,
                manifest,
                22,
                sample_count,
                batch_samples,
                min_sample_tokens,
            )
        ):
            x = module.normalize_activation(
                batch["activations"].to(device=device, non_blocking=True),
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
            sample_index = batch["sample_index"].to(device)
            view_index = batch["view_index"].to(device)
            local_samples = int(batch["lengths"].numel())
            views = []
            for view in (0, 1):
                mask = view_index == view
                pooled = pool_sample_statistics(
                    z[mask], sample_index[mask], local_samples
                )
                views.append(
                    {"mean": pooled["mean_pool"], "std": pooled["std_pool"]}
                )
                view_token_counts[view] += int(mask.sum().item())
            permutation = derangement(
                local_samples, permutation_seed + batch_index
            ).to(device)
            for statistic in statistic_names:
                a = views[0][statistic]
                b = views[1][statistic]
                stats = accumulators[statistic]
                stats["sum_a"] += a.sum(dim=0)
                stats["sum_b"] += b.sum(dim=0)
                stats["sum_a2"] += a.square().sum(dim=0)
                stats["sum_b2"] += b.square().sum(dim=0)
                stats["sum_ab"] += (a * b).sum(dim=0)
                stats["sum_ab_wrong"] += (a * b[permutation]).sum(dim=0)
            samples_seen += local_samples
            tokens_seen += int(batch["activations"].shape[0])
            del x, z, sample_index, view_index, views, permutation
    if samples_seen != sample_count:
        raise RuntimeError(f"Expected {sample_count} samples, got {samples_seen}")

    reliability: dict[str, torch.Tensor] = {}
    wrong_reliability: dict[str, torch.Tensor] = {}
    for statistic in statistic_names:
        stats = accumulators[statistic]
        reliability[statistic] = correlation_from_sums(
            stats["sum_a"],
            stats["sum_b"],
            stats["sum_a2"],
            stats["sum_b2"],
            stats["sum_ab"],
            samples_seen,
        ).cpu()
        wrong_reliability[statistic] = correlation_from_sums(
            stats["sum_a"],
            stats["sum_b"],
            stats["sum_a2"],
            stats["sum_b2"],
            stats["sum_ab_wrong"],
            samples_seen,
        ).cpu()

    true_route = reliability["std"] > reliability["mean"]
    wrong_route = wrong_reliability["std"] > wrong_reliability["mean"]
    generator = torch.Generator(device="cpu").manual_seed(permutation_seed + 100_003)
    feature_permutation = torch.randperm(feature_count, generator=generator)
    permuted_route = true_route[feature_permutation]
    routes = {
        "stable_moment_route": true_route,
        "permuted_route": permuted_route,
        "wrong_sample_route": wrong_route,
    }
    report = {
        "signal_definition": (
            "route each feature to std exactly when same-sample odd/even std "
            "correlation exceeds same-sample odd/even mean correlation"
        ),
        "sample_count": samples_seen,
        "tokens_seen": tokens_seen,
        "average_tokens_per_sample": tokens_seen / samples_seen,
        "view_a_tokens": view_token_counts[0],
        "view_b_tokens": view_token_counts[1],
        "batch_samples": batch_samples,
        "min_sample_tokens": min_sample_tokens,
        "feature_count": feature_count,
        "permutation_seed_base": permutation_seed,
        "feature_permutation_seed": permutation_seed + 100_003,
        "mean_reliability": distribution(reliability["mean"]),
        "std_reliability": distribution(reliability["std"]),
        "wrong_mean_reliability": distribution(wrong_reliability["mean"]),
        "wrong_std_reliability": distribution(wrong_reliability["std"]),
        "true_std_route_fraction": float(true_route.float().mean().item()),
        "permuted_std_route_fraction": float(permuted_route.float().mean().item()),
        "wrong_std_route_fraction": float(wrong_route.float().mean().item()),
        "true_permuted_route_agreement": float(
            (true_route == permuted_route).float().mean().item()
        ),
        "true_wrong_route_agreement": float(
            (true_route == wrong_route).float().mean().item()
        ),
        "mean_std_reliability_correlation": float(
            torch.corrcoef(
                torch.stack([reliability["mean"], reliability["std"]])
            )[0, 1].item()
        ),
    }
    del state, accumulators
    torch.cuda.empty_cache()
    return routes, report


def apply_moment_route(
    mean_features: dict[str, torch.Tensor],
    std_features: dict[str, torch.Tensor],
    route_to_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    route = route_to_std.to(dtype=torch.bool, device="cpu").unsqueeze(0)
    routed = {}
    for class_name in mean_features:
        if mean_features[class_name].shape != std_features[class_name].shape:
            raise ValueError(f"Moment shapes differ for {class_name}")
        if mean_features[class_name].shape[1] != route.shape[1]:
            raise ValueError("Route width differs from feature width")
        routed[class_name] = torch.where(
            route, std_features[class_name], mean_features[class_name]
        )
    return routed


def compute_mean_dispersion_variants(
    module: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    config: Any,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    feature_count = int(state["encoder.weight"].shape[0])
    variant_keys = ["mean_pool", "std_pool", "mean_std_pool", "rms_pool"]
    variants: dict[str, dict[str, torch.Tensor]] = {key: {} for key in variant_keys}
    sample_count_total = 0
    token_count_total = 0
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            sample_count = int(acts_cpu.shape[0])
            outputs = {
                key: torch.empty((sample_count, feature_count), dtype=torch.float32)
                for key in variant_keys
            }
            for start in range(0, sample_count, config.sae_seq_batch_size):
                end = min(sample_count, start + config.sae_seq_batch_size)
                acts = acts_cpu[start:end].to(config.device, non_blocking=True)
                mask = mask_cpu[start:end].to(config.device, non_blocking=True)
                local_ids = torch.arange(end - start, device=config.device).unsqueeze(1)
                local_ids = local_ids.expand_as(mask)
                sample_ids = local_ids[mask]
                x_flat = acts.reshape(-1, acts.shape[-1])[mask.reshape(-1)]
                x_norm = module.normalize_activation(
                    x_flat, config.dtype, config.normalize_eps
                )
                token_features = torch.relu(
                    F.linear(
                        x_norm - state["b_pre"],
                        state["encoder.weight"],
                        state["encoder.bias"],
                    )
                ).float()
                pooled = pool_sample_statistics(
                    token_features, sample_ids, end - start
                )
                for key in variant_keys:
                    outputs[key][start:end] = pooled[key].cpu()
                sample_count_total += end - start
                token_count_total += int(mask.sum().item())
                del (
                    acts,
                    mask,
                    local_ids,
                    sample_ids,
                    x_flat,
                    x_norm,
                    token_features,
                    pooled,
                )
            for key in variant_keys:
                variants[key][class_name] = outputs[key]
    statistics = {
        "sample_count": sample_count_total,
        "token_count": token_count_total,
        "feature_count": feature_count,
    }
    return variants, statistics


def probe_topk_dataset(
    module: Any,
    train_acts: dict[str, torch.Tensor],
    test_acts: dict[str, torch.Tensor],
    k_values: list[int],
    seed: int,
    selector: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    metrics: dict[str, float] = {}
    for class_index, class_name in enumerate(train_acts):
        train_x, train_y = module.prepare_probe_data(
            train_acts, class_name, seed + 17 * class_index
        )
        test_x, test_y = module.prepare_probe_data(
            test_acts, class_name, seed + 29 * class_index
        )
        class_metrics = {}
        for k in k_values:
            selected = selector(train_x, train_y, k)
            result = module.train_probe(
                train_x[:, selected],
                train_y,
                test_x[:, selected],
                test_y,
                seed + k + 101 * class_index,
            )
            result["selected_features"] = [int(index) for index in selected.tolist()]
            class_metrics[f"top_{k}"] = result
        per_class[class_name] = class_metrics
    for k in k_values:
        for metric in ("test_accuracy", "test_auc"):
            metrics[f"top_{k}_{metric}"] = float(
                np.mean(
                    [per_class[name][f"top_{k}"][metric] for name in per_class]
                )
            )
    return {"metrics": metrics, "per_class": per_class}


def probe_full_ridge_dataset(
    module: Any,
    train_acts: dict[str, torch.Tensor],
    test_acts: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    for class_index, class_name in enumerate(train_acts):
        train_x, train_y = module.prepare_probe_data(
            train_acts, class_name, seed + 17 * class_index
        )
        test_x, test_y = module.prepare_probe_data(
            test_acts, class_name, seed + 29 * class_index
        )
        mean = train_x.mean(dim=0)
        std = train_x.std(dim=0, unbiased=False)
        valid = std > 1.0e-6
        train_np = ((train_x[:, valid] - mean[valid]) / std[valid]).numpy()
        test_np = ((test_x[:, valid] - mean[valid]) / std[valid]).numpy()
        train_y_np = train_y.numpy()
        test_y_np = test_y.numpy()
        model = RidgeClassifier(
            alpha=1.0,
            solver="lsqr",
            tol=1.0e-3,
        )
        model.fit(train_np, train_y_np)
        predictions = model.predict(test_np)
        scores = model.decision_function(test_np)
        per_class[class_name] = {
            "test_accuracy": float(accuracy_score(test_y_np, predictions)),
            "test_auc": float(roc_auc_score(test_y_np, scores)),
            "valid_feature_count": int(valid.sum().item()),
        }
        del train_x, test_x, train_np, test_np, model
    return {
        "metrics": {
            "test_accuracy": float(
                np.mean([values["test_accuracy"] for values in per_class.values()])
            ),
            "test_auc": float(
                np.mean([values["test_auc"] for values in per_class.values()])
            ),
        },
        "per_class": per_class,
    }


def aggregate_topk(
    results: dict[str, dict[str, dict[str, Any]]],
    variants: list[str],
    mode: str,
    k_values: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        datasets = results[variant][mode]
        rows.append(
            {
                "variant": variant,
                "mode": mode,
                "mean_acc": float(
                    np.mean(
                        [
                            dataset["metrics"][f"top_{k}_test_accuracy"]
                            for dataset in datasets.values()
                            for k in k_values
                        ]
                    )
                ),
                "mean_auc": float(
                    np.mean(
                        [
                            dataset["metrics"][f"top_{k}_test_auc"]
                            for dataset in datasets.values()
                            for k in k_values
                        ]
                    )
                ),
                **{
                    f"top_{k}_acc": float(
                        np.mean(
                            [
                                dataset["metrics"][f"top_{k}_test_accuracy"]
                                for dataset in datasets.values()
                            ]
                        )
                    )
                    for k in k_values
                },
            }
        )
    return sorted(rows, key=lambda row: row["mean_acc"], reverse=True)


def aggregate_ridge(
    results: dict[str, dict[str, dict[str, Any]]],
    variants: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        datasets = results[variant]["full_ridge"]
        rows.append(
            {
                "variant": variant,
                "test_accuracy": float(
                    np.mean(
                        [
                            dataset["metrics"]["test_accuracy"]
                            for dataset in datasets.values()
                        ]
                    )
                ),
                "test_auc": float(
                    np.mean(
                        [
                            dataset["metrics"]["test_auc"]
                            for dataset in datasets.values()
                        ]
                    )
                ),
            }
        )
    return sorted(rows, key=lambda row: row["test_accuracy"], reverse=True)


def dataset_topk_mean(
    results: dict[str, dict[str, dict[str, Any]]],
    variant: str,
    mode: str,
    dataset: str,
    k_values: list[int],
) -> float:
    metrics = results[variant][mode][dataset]["metrics"]
    return float(np.mean([metrics[f"top_{k}_test_accuracy"] for k in k_values]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path("/root/autodl-tmp/saebench_sparse_probing_all_architectures.py"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-sample-count", type=int, default=8192)
    parser.add_argument("--route-batch-samples", type=int, default=64)
    parser.add_argument("--route-min-sample-tokens", type=int, default=4)
    parser.add_argument("--route-permutation-seed", type=int, default=44027)
    parser.add_argument("--datasets", nargs="+", default=INITIAL3)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--sae-seq-batch-size", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    module = load_eval_module(args.eval_script)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    routes, routing_report = estimate_stable_moment_routes(
        module,
        args.checkpoint,
        args.cache_dir,
        manifest,
        args.route_sample_count,
        args.route_batch_samples,
        args.route_min_sample_tokens,
        args.route_permutation_seed,
        device,
        dtype,
        1.0e-6,
    )
    torch.save(routes, args.output_dir / "stable-moment-routes.pt")
    (args.output_dir / "stable-moment-routing-statistics.json").write_text(
        json.dumps(routing_report, indent=2) + "\n", encoding="utf-8"
    )
    kwargs = {
        "model_dir": args.model_dir,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "context_length": args.context_length,
        "llm_batch_size": args.llm_batch_size,
        "sae_seq_batch_size": args.sae_seq_batch_size,
        "k_values": LOW_K,
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

    variants = [
        "mean_pool",
        "std_pool",
        "rms_pool",
        "mean_std_pool",
        "stable_moment_route",
        "permuted_route",
        "wrong_sample_route",
    ]
    results: dict[str, dict[str, dict[str, Any]]] = {
        variant: {
            "low_mean_diff": {},
            "low_effect_size": {},
            "wide_mean_diff": {},
            "full_ridge": {},
        }
        for variant in variants
    }
    split_statistics: dict[str, Any] = {}
    for dataset_index, dataset in enumerate(args.datasets):
        print(f"== Dataset: {dataset}", flush=True)
        train_data, test_data = module.get_multi_label_train_test_data(
            dataset, args.train_size, args.test_size, args.random_seed
        )
        train_layers, train_masks = module.collect_layer_activations(
            llm, tokenizer, train_data, config, [22]
        )
        test_layers, test_masks = module.collect_layer_activations(
            llm, tokenizer, test_data, config, [22]
        )
        train_features, train_stats = compute_mean_dispersion_variants(
            module, train_layers[22], train_masks, state, config
        )
        test_features, test_stats = compute_mean_dispersion_variants(
            module, test_layers[22], test_masks, state, config
        )
        for route_name, route in routes.items():
            train_features[route_name] = apply_moment_route(
                train_features["mean_pool"], train_features["std_pool"], route
            )
            test_features[route_name] = apply_moment_route(
                test_features["mean_pool"], test_features["std_pool"], route
            )
        split_statistics[dataset] = {
            "train": train_stats,
            "test": test_stats,
        }
        seed = args.random_seed + 1009 * dataset_index
        for variant in variants:
            print(f"   {variant}", flush=True)
            results[variant]["low_mean_diff"][dataset] = probe_topk_dataset(
                module,
                train_features[variant],
                test_features[variant],
                LOW_K,
                seed,
                select_mean_diff,
            )
            results[variant]["low_effect_size"][dataset] = probe_topk_dataset(
                module,
                train_features[variant],
                test_features[variant],
                LOW_K,
                seed,
                select_effect_size,
            )
            results[variant]["wide_mean_diff"][dataset] = probe_topk_dataset(
                module,
                train_features[variant],
                test_features[variant],
                WIDE_K,
                seed,
                select_mean_diff,
            )
            results[variant]["full_ridge"][dataset] = probe_full_ridge_dataset(
                module,
                train_features[variant],
                test_features[variant],
                seed,
            )
        del (
            train_layers,
            test_layers,
            train_masks,
            test_masks,
            train_features,
            test_features,
        )
        gc.collect()
    del llm, state
    torch.cuda.empty_cache()
    gc.collect()

    regular_variants = variants
    low_summary = aggregate_topk(results, variants, "low_mean_diff", LOW_K)
    effect_summary = aggregate_topk(
        results, regular_variants, "low_effect_size", LOW_K
    )
    wide_summary = aggregate_topk(
        results, regular_variants, "wide_mean_diff", WIDE_K
    )
    ridge_summary = aggregate_ridge(results, regular_variants)
    low_by_variant = {row["variant"]: row for row in low_summary}
    effect_by_variant = {row["variant"]: row for row in effect_summary}
    wide_by_variant = {row["variant"]: row for row in wide_summary}
    ridge_by_variant = {row["variant"]: row for row in ridge_summary}
    candidate = "stable_moment_route"
    dataset_deltas = {
        mode: {
            dataset: {
                "mean_pool": dataset_topk_mean(
                    results,
                    "mean_pool",
                    mode,
                    dataset,
                    LOW_K if mode != "wide_mean_diff" else WIDE_K,
                ),
                "rms_pool": dataset_topk_mean(
                    results,
                    "rms_pool",
                    mode,
                    dataset,
                    LOW_K if mode != "wide_mean_diff" else WIDE_K,
                ),
                "stable_moment_route": dataset_topk_mean(
                    results,
                    candidate,
                    mode,
                    dataset,
                    LOW_K if mode != "wide_mean_diff" else WIDE_K,
                ),
                "permuted_route": dataset_topk_mean(
                    results,
                    "permuted_route",
                    mode,
                    dataset,
                    LOW_K if mode != "wide_mean_diff" else WIDE_K,
                ),
                "wrong_sample_route": dataset_topk_mean(
                    results,
                    "wrong_sample_route",
                    mode,
                    dataset,
                    LOW_K if mode != "wide_mean_diff" else WIDE_K,
                ),
            }
            for dataset in args.datasets
        }
        for mode in ("low_mean_diff", "low_effect_size", "wide_mean_diff")
    }
    for mode_values in dataset_deltas.values():
        for values in mode_values.values():
            values["candidate_minus_mean"] = (
                values[candidate] - values["mean_pool"]
            )
            values["candidate_minus_rms"] = (
                values[candidate] - values["rms_pool"]
            )
            values["candidate_minus_permuted"] = (
                values[candidate] - values["permuted_route"]
            )
            values["candidate_minus_wrong"] = (
                values[candidate] - values["wrong_sample_route"]
            )
    ridge_dataset_deltas = {}
    for dataset in args.datasets:
        mean_metrics = results["mean_pool"]["full_ridge"][dataset]["metrics"]
        rms_metrics = results["rms_pool"]["full_ridge"][dataset]["metrics"]
        candidate_metrics = results[candidate]["full_ridge"][dataset]["metrics"]
        ridge_dataset_deltas[dataset] = {
            "mean_accuracy": mean_metrics["test_accuracy"],
            "rms_accuracy": rms_metrics["test_accuracy"],
            "candidate_accuracy": candidate_metrics["test_accuracy"],
            "candidate_minus_mean_accuracy": (
                candidate_metrics["test_accuracy"] - mean_metrics["test_accuracy"]
            ),
            "candidate_minus_rms_accuracy": (
                candidate_metrics["test_accuracy"] - rms_metrics["test_accuracy"]
            ),
            "mean_auc": mean_metrics["test_auc"],
            "rms_auc": rms_metrics["test_auc"],
            "candidate_auc": candidate_metrics["test_auc"],
            "candidate_minus_mean_auc": (
                candidate_metrics["test_auc"] - mean_metrics["test_auc"]
            ),
        }
    gate = {
        "official_candidate_minus_mean_at_least_0p005": (
            low_by_variant[candidate]["mean_acc"]
            - low_by_variant["mean_pool"]["mean_acc"]
            >= 0.005
        ),
        "official_no_dataset_drop_below_minus_0p01": all(
            values["candidate_minus_mean"] >= -0.01
            for values in dataset_deltas["low_mean_diff"].values()
        ),
        "official_candidate_minus_permuted_at_least_0p003": (
            low_by_variant[candidate]["mean_acc"]
            - low_by_variant["permuted_route"]["mean_acc"]
            >= 0.003
        ),
        "official_candidate_minus_wrong_at_least_0p003": (
            low_by_variant[candidate]["mean_acc"]
            - low_by_variant["wrong_sample_route"]["mean_acc"]
            >= 0.003
        ),
        "effect_size_candidate_minus_mean_at_least_0p003": (
            effect_by_variant[candidate]["mean_acc"]
            - effect_by_variant["mean_pool"]["mean_acc"]
            >= 0.003
        ),
        "effect_size_no_dataset_drop_below_minus_0p01": all(
            values["candidate_minus_mean"] >= -0.01
            for values in dataset_deltas["low_effect_size"].values()
        ),
        "wide_candidate_minus_mean_nonnegative": (
            wide_by_variant[candidate]["mean_acc"]
            >= wide_by_variant["mean_pool"]["mean_acc"]
        ),
        "full_ridge_candidate_minus_mean_accuracy_nonnegative": (
            ridge_by_variant[candidate]["test_accuracy"]
            >= ridge_by_variant["mean_pool"]["test_accuracy"]
        ),
        "full_ridge_candidate_minus_mean_auc_nonnegative": (
            ridge_by_variant[candidate]["test_auc"]
            >= ridge_by_variant["mean_pool"]["test_auc"]
        ),
        "full_ridge_no_dataset_accuracy_drop_below_minus_0p01": all(
            values["candidate_minus_mean_accuracy"] >= -0.01
            for values in ridge_dataset_deltas.values()
        ),
    }
    gate["pass"] = all(gate.values())
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "cache_dir": str(args.cache_dir),
            "cache_manifest_sha256": sha256(args.cache_dir / "manifest.json"),
            "layer": 22,
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "low_k": LOW_K,
            "wide_k": WIDE_K,
            "random_seed": args.random_seed,
            "route_sample_count": args.route_sample_count,
            "route_batch_samples": args.route_batch_samples,
            "route_min_sample_tokens": args.route_min_sample_tokens,
            "route_permutation_seed": args.route_permutation_seed,
        },
        "mechanism": {
            "mean_pool": "mean_t z_t",
            "std_pool": "sqrt(max(mean_t z_t^2 - mean_t(z_t)^2, 0))",
            "mean_std_pool": "mean_t z_t + std_t z_t with fixed unit coefficient",
            "rms_pool": "sqrt(mean_t z_t^2)",
            "candidate": "per feature, route to std iff OWT odd/even std reliability exceeds mean reliability",
            "permuted_control": "fixed feature permutation of the true route, preserving exact std-route count",
            "wrong_sample_control": "route selected from odd/even correlations after within-batch sample derangement",
            "same_checkpoint": True,
            "same_parameter_count": True,
            "same_exposed_feature_count": True,
            "transform_sweep": False,
        },
        "routing_signal": routing_report,
        "split_statistics": split_statistics,
        "readout_results": results,
        "summary": {
            "low_mean_diff": low_summary,
            "low_effect_size": effect_summary,
            "wide_mean_diff": wide_summary,
            "full_ridge": ridge_summary,
        },
        "dataset_deltas": dataset_deltas,
        "ridge_dataset_deltas": ridge_dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-end-to-end-stable-moment-routing-sae-v3-screen"
            if gate["pass"]
            else "close-sample-energy-family-after-v3"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "stable-moment-routing-v3-gate.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Structured stable-moment routing gate v3", ""]
    for title, rows, k_values in (
        ("Low-k mean-diff", low_summary, LOW_K),
        ("Low-k standardized effect-size", effect_summary, LOW_K),
        ("Wide-k mean-diff", wide_summary, WIDE_K),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Variant | Mean Acc | Mean AUC | "
                + " | ".join(f"Top-{k}" for k in k_values)
                + " |",
                "|---|---:|---:|" + "---:|" * len(k_values),
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['variant']} | {row['mean_acc']:.6f} | "
                f"{row['mean_auc']:.6f} | "
                + " | ".join(f"{row[f'top_{k}_acc']:.6f}" for k in k_values)
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Full-feature standardized ridge",
            "",
            "| Variant | Accuracy | AUC |",
            "|---|---:|---:|",
        ]
    )
    for row in ridge_summary:
        lines.append(
            f"| {row['variant']} | {row['test_accuracy']:.6f} | "
            f"{row['test_auc']:.6f} |"
        )
    lines.extend(["", f"Decision: `{payload['decision']}`", ""])
    (args.output_dir / "stable-moment-routing-v3-gate.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "dataset_deltas": dataset_deltas,
                "ridge_dataset_deltas": ridge_dataset_deltas,
                "gate": gate,
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
