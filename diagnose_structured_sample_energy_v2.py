#!/usr/bin/env python3
"""Test whether mean-preserving sample dispersion survives multiple readouts."""

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
from typing import Any, Callable

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
    spec = importlib.util.spec_from_file_location("sample_energy_v2_eval", path)
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


def class_balanced_wrong_sample(
    features: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    keys = list(features)
    if len(keys) < 2:
        raise ValueError("Wrong-sample control requires at least two classes")
    row_counts = {int(values.shape[0]) for values in features.values()}
    if len(row_counts) != 1:
        raise ValueError(f"Class row counts differ: {sorted(row_counts)}")
    row_count = row_counts.pop()
    wrong = {key: torch.empty_like(features[key]) for key in keys}
    same_class_pairs = 0
    source_counts = {key: 0 for key in keys}
    for target_index, target_key in enumerate(keys):
        for row in range(row_count):
            source_index = (
                target_index + 1 + row % (len(keys) - 1)
            ) % len(keys)
            source_key = keys[source_index]
            same_class_pairs += int(source_key == target_key)
            source_counts[source_key] += 1
            wrong[target_key][row] = features[source_key][row]
    multiset_equal = all(count == row_count for count in source_counts.values())
    if same_class_pairs != 0 or not multiset_equal:
        raise RuntimeError(
            "Wrong-sample control failed: "
            f"same_class_pairs={same_class_pairs}, multiset_equal={multiset_equal}"
        )
    return wrong, {
        "row_count_per_class": row_count,
        "same_class_pairs": same_class_pairs,
        "source_counts": source_counts,
        "full_row_multiset_equal_by_bijective_source_counts": multiset_equal,
    }


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

    module = load_eval_module(args.eval_script)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
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
        "wrong_sample_mean_std",
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
    started = time.time()
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
        train_wrong, train_wrong_stats = class_balanced_wrong_sample(
            train_features["mean_std_pool"]
        )
        test_wrong, test_wrong_stats = class_balanced_wrong_sample(
            test_features["mean_std_pool"]
        )
        train_features["wrong_sample_mean_std"] = train_wrong
        test_features["wrong_sample_mean_std"] = test_wrong
        split_statistics[dataset] = {
            "train": train_stats,
            "test": test_stats,
            "train_wrong_control": train_wrong_stats,
            "test_wrong_control": test_wrong_stats,
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
            if variant != "wrong_sample_mean_std":
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
            train_wrong,
            test_wrong,
        )
        gc.collect()
    del llm, state
    torch.cuda.empty_cache()
    gc.collect()

    regular_variants = ["mean_pool", "std_pool", "rms_pool", "mean_std_pool"]
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
    candidate = "mean_std_pool"
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
                "mean_std_pool": dataset_topk_mean(
                    results,
                    candidate,
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
        "official_candidate_within_0p003_of_rms": (
            low_by_variant[candidate]["mean_acc"]
            >= low_by_variant["rms_pool"]["mean_acc"] - 0.003
        ),
        "candidate_minus_wrong_sample_at_least_0p05": (
            low_by_variant[candidate]["mean_acc"]
            - low_by_variant["wrong_sample_mean_std"]["mean_acc"]
            >= 0.05
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
        },
        "mechanism": {
            "mean_pool": "mean_t z_t",
            "std_pool": "sqrt(max(mean_t z_t^2 - mean_t(z_t)^2, 0))",
            "mean_std_pool": "mean_t z_t + std_t z_t with fixed unit coefficient",
            "rms_pool": "sqrt(mean_t z_t^2)",
            "wrong_sample_control": "row-balanced class derangement preserving the complete mean-plus-std row multiset",
            "same_checkpoint": True,
            "same_parameter_count": True,
            "same_exposed_feature_count": True,
            "candidate_coefficient": 1.0,
            "transform_sweep": False,
        },
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
            "authorize-end-to-end-mean-dispersion-sae-v2-screen"
            if gate["pass"]
            else "stop-before-end-to-end-mean-dispersion-sae-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "sample-energy-v2-gate.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Structured mean-preserving sample-dispersion gate v2", ""]
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
    (args.output_dir / "sample-energy-v2-gate.md").write_text(
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
