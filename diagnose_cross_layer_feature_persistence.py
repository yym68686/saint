#!/usr/bin/env python3
"""Gate cross-layer feature persistence before training a new SAE family."""

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
from typing import Any, Iterator

import torch
import torch.nn.functional as F


INITIAL3 = [
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "fancyzhx/ag_news",
]
LAYERS = [20, 21, 22, 23]


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("persistence_eval", path)
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


def iter_aligned_tokens(
    cache_dir: Path,
    manifest: dict[str, Any],
    max_tokens: int,
    batch_tokens: int,
) -> Iterator[dict[int, torch.Tensor]]:
    emitted = 0
    for shard in manifest["shards"]:
        tensors = {
            layer: torch.load(
                cache_dir / shard["layers"][str(layer)]["path"],
                map_location="cpu",
                weights_only=True,
            )
            for layer in LAYERS
        }
        lengths = {int(value.shape[0]) for value in tensors.values()}
        if len(lengths) != 1:
            raise RuntimeError(f"Unaligned layer shard: {shard}")
        available = min(next(iter(lengths)), max_tokens - emitted)
        for start in range(0, available, batch_tokens):
            end = min(available, start + batch_tokens)
            yield {layer: value[start:end] for layer, value in tensors.items()}
            emitted += end - start
            if emitted >= max_tokens:
                return
        del tensors


def normalized_layer_centers(
    module: Any,
    cache_dir: Path,
    manifest: dict[str, Any],
    b_pre: torch.Tensor,
    max_tokens: int,
    batch_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> dict[int, torch.Tensor]:
    sums = {
        layer: torch.zeros(b_pre.numel(), device=device, dtype=torch.float32)
        for layer in LAYERS
    }
    count = 0
    with torch.inference_mode():
        for batch in iter_aligned_tokens(
            cache_dir, manifest, max_tokens, batch_tokens
        ):
            local = next(iter(batch.values())).shape[0]
            for layer, values in batch.items():
                x = values.to(device=device, dtype=dtype, non_blocking=True)
                x = module.normalize_activation(x, dtype, eps)
                sums[layer] += x.float().sum(dim=0)
            count += local
    if count != max_tokens:
        raise RuntimeError(f"Expected {max_tokens} center tokens, got {count}")
    centers = {layer: sums[layer] / count for layer in LAYERS}
    centers[22] = b_pre.float()
    return centers


def correlation_from_sums(
    sum_x: torch.Tensor,
    sum_y: torch.Tensor,
    sum_x2: torch.Tensor,
    sum_y2: torch.Tensor,
    sum_xy: torch.Tensor,
    count: int,
) -> torch.Tensor:
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = (sum_x2 - sum_x.square() / count).clamp_min(0)
    variance_y = (sum_y2 - sum_y.square() / count).clamp_min(0)
    denominator = (variance_x * variance_y).sqrt()
    return torch.where(
        denominator > 1.0e-12,
        covariance / denominator.clamp_min(1.0e-12),
        torch.zeros_like(covariance),
    ).clamp(-1, 1)


def rank_weights(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
    if order.numel() == 1:
        return torch.ones(1, dtype=torch.float32)
    return 0.5 + ranks / (order.numel() - 1)


def estimate_persistence(
    module: Any,
    checkpoint: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
    center_tokens: int,
    correlation_tokens: int,
    batch_tokens: int,
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
    b_pre = state["b_pre"]
    centers = normalized_layer_centers(
        module,
        cache_dir,
        manifest,
        b_pre,
        center_tokens,
        batch_tokens,
        device,
        dtype,
        eps,
    )
    feature_count = int(state["encoder.weight"].shape[0])
    sum_x = torch.zeros(feature_count, device=device, dtype=torch.float32)
    sum_x2 = torch.zeros_like(sum_x)
    pair_stats = {
        layer: {
            name: torch.zeros_like(sum_x)
            for name in ("sum_y", "sum_y2", "sum_xy", "sum_xy_wrong")
        }
        for layer in LAYERS
        if layer != 22
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(permutation_seed)
    count = 0
    with torch.inference_mode():
        for batch in iter_aligned_tokens(
            cache_dir, manifest, correlation_tokens, batch_tokens
        ):
            activations: dict[int, torch.Tensor] = {}
            for layer, values in batch.items():
                x = values.to(device=device, dtype=dtype, non_blocking=True)
                x = module.normalize_activation(x, dtype, eps)
                centered = x - centers[layer].to(device=device, dtype=dtype)
                activations[layer] = torch.relu(
                    F.linear(
                        centered,
                        state["encoder.weight"],
                        state["encoder.bias"],
                    )
                ).float()
            reference = activations[22]
            sum_x += reference.sum(dim=0)
            sum_x2 += reference.square().sum(dim=0)
            local = int(reference.shape[0])
            permutation = torch.randperm(local, generator=generator).to(device)
            for layer, stats in pair_stats.items():
                view = activations[layer]
                stats["sum_y"] += view.sum(dim=0)
                stats["sum_y2"] += view.square().sum(dim=0)
                stats["sum_xy"] += (reference * view).sum(dim=0)
                stats["sum_xy_wrong"] += (
                    reference * view[permutation]
                ).sum(dim=0)
            count += local
            del activations, reference
    if count != correlation_tokens:
        raise RuntimeError(
            f"Expected {correlation_tokens} correlation tokens, got {count}"
        )
    correlations = []
    wrong_correlations = []
    per_layer_summary = {}
    for layer, stats in pair_stats.items():
        corr = correlation_from_sums(
            sum_x,
            stats["sum_y"],
            sum_x2,
            stats["sum_y2"],
            stats["sum_xy"],
            count,
        )
        wrong = correlation_from_sums(
            sum_x,
            stats["sum_y"],
            sum_x2,
            stats["sum_y2"],
            stats["sum_xy_wrong"],
            count,
        )
        correlations.append(corr.cpu())
        wrong_correlations.append(wrong.cpu())
        per_layer_summary[str(layer)] = {
            "correlation_mean": float(corr.mean().item()),
            "correlation_median": float(corr.median().item()),
            "wrong_correlation_mean": float(wrong.mean().item()),
            "wrong_correlation_median": float(wrong.median().item()),
        }
    persistence_score = torch.stack(correlations).mean(dim=0)
    wrong_score = torch.stack(wrong_correlations).mean(dim=0)
    persistence_weight = rank_weights(persistence_score)
    wrong_weight = rank_weights(wrong_score)
    feature_generator = torch.Generator(device="cpu")
    feature_generator.manual_seed(permutation_seed)
    feature_permutation = torch.randperm(
        feature_count, generator=feature_generator
    )
    permuted_weight = persistence_weight[feature_permutation]
    weights = {
        "raw": torch.ones(feature_count, dtype=torch.float32),
        "persistence": persistence_weight,
        "permuted": permuted_weight,
        "wrong_alignment": wrong_weight,
    }

    def distribution(values: torch.Tensor) -> dict[str, float]:
        quantiles = torch.quantile(
            values.float(), torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
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

    report = {
        "center_tokens": center_tokens,
        "correlation_tokens": count,
        "feature_count": feature_count,
        "permutation_seed": permutation_seed,
        "per_layer": per_layer_summary,
        "persistence_score": distribution(persistence_score),
        "wrong_alignment_score": distribution(wrong_score),
        "persistence_weight": distribution(persistence_weight),
        "wrong_alignment_weight": distribution(wrong_weight),
        "center_l2": {
            str(layer): float(value.norm().item())
            for layer, value in centers.items()
        },
    }
    del state
    torch.cuda.empty_cache()
    return weights, report


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
                    f"top_{k}_acc": aggregate[
                        f"sae_top_{k}_test_accuracy"
                    ]
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-tokens", type=int, default=4096)
    parser.add_argument("--correlation-tokens", type=int, default=8192)
    parser.add_argument("--persistence-batch-tokens", type=int, default=128)
    parser.add_argument("--permutation-seed", type=int, default=42026)
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
    weights, persistence_report = estimate_persistence(
        module,
        args.checkpoint,
        args.cache_dir,
        manifest,
        args.center_tokens,
        args.correlation_tokens,
        args.persistence_batch_tokens,
        args.permutation_seed,
        device,
        dtype,
        1.0e-6,
    )
    weights_path = args.output_dir / "persistence-weights.pt"
    torch.save(weights, weights_path)
    (args.output_dir / "persistence-statistics.json").write_text(
        json.dumps(persistence_report, indent=2) + "\n", encoding="utf-8"
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
        ("persistence", "cross-layer persistence rank weighting"),
        ("permuted", "permuted persistence-weight control"),
        ("wrong_alignment", "wrong-token alignment control"),
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
            del train_features, test_features
        rows.append(row)

    summary = summarize(rows, args.k_values)
    summary_by_key = {row["variant_key"]: row for row in summary}
    rows_by_key = {row["variant_key"]: row for row in rows}
    candidate = summary_by_key["persistence"]
    reference = summary_by_key["raw"]
    permuted = summary_by_key["permuted"]
    wrong = summary_by_key["wrong_alignment"]
    dataset_deltas = {
        dataset: {
            "reference": dataset_mean(
                rows_by_key["raw"], dataset, args.k_values
            ),
            "persistence": dataset_mean(
                rows_by_key["persistence"], dataset, args.k_values
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta"] = values["persistence"] - values["reference"]
    gate = {
        "candidate_minus_reference_at_least_0p005": (
            candidate["mean_acc"] - reference["mean_acc"] >= 0.005
        ),
        "no_dataset_drop_below_minus_0p01": all(
            values["delta"] >= -0.01 for values in dataset_deltas.values()
        ),
        "candidate_minus_permuted_at_least_0p002": (
            candidate["mean_acc"] - permuted["mean_acc"] >= 0.002
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
            "layers": LAYERS,
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "persistence": persistence_report,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-cross-layer-persistence-sae-v1-screen"
            if gate["pass"]
            else "stop-before-cross-layer-persistence-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "cross-layer-persistence-gate.json"
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Cross-layer feature persistence gate",
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
    (args.output_dir / "cross-layer-persistence-gate.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "gate": gate, "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
