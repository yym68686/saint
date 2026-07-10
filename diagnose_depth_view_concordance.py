#!/usr/bin/env python3
"""Gate equal-weight depth/view concordance before joint SAE training."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import time
from pathlib import Path
from typing import Any

import torch

import diagnose_cross_layer_feature_persistence as common


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


def top_fraction_overlap(
    left: torch.Tensor, right: torch.Tensor, fraction: float
) -> dict[str, float]:
    count = max(1, int(left.numel() * fraction))
    left_set = set(torch.topk(left, count).indices.tolist())
    right_set = set(torch.topk(right, count).indices.tolist())
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return {
        "fraction": fraction,
        "count": count,
        "intersection_fraction": intersection / count,
        "jaccard": intersection / union,
    }


def construct_weights(
    persistence_path: Path,
    splitview_path: Path,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    persistence_bundle = torch.load(
        persistence_path, map_location="cpu", weights_only=True
    )
    splitview_bundle = torch.load(
        splitview_path, map_location="cpu", weights_only=True
    )
    persistence = persistence_bundle["persistence"].float()
    persistence_wrong = persistence_bundle["wrong_alignment"].float()
    splitview = splitview_bundle["splitview_reliability"].float()
    splitview_wrong = splitview_bundle["wrong_sample"].float()
    if not (
        persistence.shape
        == persistence_wrong.shape
        == splitview.shape
        == splitview_wrong.shape
    ):
        raise ValueError("Source signal weight shapes do not match")
    feature_count = int(persistence.numel())
    mismatch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    mismatched_splitview = splitview[
        torch.randperm(feature_count, generator=mismatch_generator)
    ]
    joint_score = 0.5 * (persistence + splitview)
    mismatched_score = 0.5 * (persistence + mismatched_splitview)
    wrong_score = 0.5 * (persistence_wrong + splitview_wrong)
    joint_weight = common.rank_weights(joint_score, seed + 2)
    mismatched_weight = common.rank_weights(mismatched_score, seed + 3)
    wrong_weight = common.rank_weights(wrong_score, seed + 4)
    final_generator = torch.Generator(device="cpu").manual_seed(seed + 5)
    permuted_weight = joint_weight[
        torch.randperm(feature_count, generator=final_generator)
    ]
    weights = {
        "raw": torch.ones(feature_count, dtype=torch.float32),
        "concordance": joint_weight,
        "permuted": permuted_weight,
        "mismatched": mismatched_weight,
        "wrong_joint": wrong_weight,
    }
    source_stack = torch.stack([persistence, splitview, persistence_wrong, splitview_wrong])
    report = {
        "definition": (
            "rank(0.5 * cross_layer_persistence_rank + "
            "0.5 * true_sample_splitview_reliability_rank)"
        ),
        "feature_count": feature_count,
        "seed": seed,
        "mismatched_splitview_permutation_seed": seed + 1,
        "joint_rank_tie_seed": seed + 2,
        "mismatched_rank_tie_seed": seed + 3,
        "wrong_joint_rank_tie_seed": seed + 4,
        "final_permutation_seed": seed + 5,
        "source_rank_correlation": torch.corrcoef(source_stack).tolist(),
        "source_top10_overlap": top_fraction_overlap(
            persistence, splitview, 0.10
        ),
        "joint_score": distribution(joint_score),
        "mismatched_score": distribution(mismatched_score),
        "wrong_joint_score": distribution(wrong_score),
        "concordance_weight": distribution(joint_weight),
        "mismatched_weight": distribution(mismatched_weight),
        "wrong_joint_weight": distribution(wrong_weight),
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
    parser.add_argument("--persistence-weights", type=Path, required=True)
    parser.add_argument("--splitview-weights", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concordance-seed", type=int, default=45026)
    parser.add_argument("--datasets", nargs="+", default=common.INITIAL3)
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

    module = common.load_eval_module(args.eval_script)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    started = time.time()
    weights, concordance_report = construct_weights(
        args.persistence_weights,
        args.splitview_weights,
        args.concordance_seed,
    )
    torch.save(weights, args.output_dir / "depth-view-concordance-weights.pt")
    (args.output_dir / "depth-view-concordance-statistics.json").write_text(
        json.dumps(concordance_report, indent=2) + "\n", encoding="utf-8"
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
            "train": common.mean_relu_features(
                module, train_layers[22], train_masks, state, config
            ),
            "test": common.mean_relu_features(
                module, test_layers[22], test_masks, state, config
            ),
        }
        del train_layers, test_layers, train_masks, test_masks
    del llm, state
    torch.cuda.empty_cache()
    gc.collect()

    variants = [
        ("raw", "L22 ReLU reference"),
        ("concordance", "depth-view concordance weighting"),
        ("permuted", "permuted concordance-weight control"),
        ("mismatched", "mismatched-signal concordance control"),
        ("wrong_joint", "dual-wrong-signal concordance control"),
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
            train_features = common.scale_features(features["train"], weight)
            test_features = common.scale_features(features["test"], weight)
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

    summary = common.summarize(rows, args.k_values)
    summary_by_key = {row["variant_key"]: row for row in summary}
    rows_by_key = {row["variant_key"]: row for row in rows}
    candidate = summary_by_key["concordance"]
    reference = summary_by_key["raw"]
    permuted = summary_by_key["permuted"]
    mismatched = summary_by_key["mismatched"]
    wrong = summary_by_key["wrong_joint"]
    dataset_deltas = {
        dataset: {
            "reference": common.dataset_mean(
                rows_by_key["raw"], dataset, args.k_values
            ),
            "concordance": common.dataset_mean(
                rows_by_key["concordance"], dataset, args.k_values
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta"] = values["concordance"] - values["reference"]
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
        "candidate_minus_mismatched_at_least_0p002": (
            candidate["mean_acc"] - mismatched["mean_acc"] >= 0.002
        ),
        "candidate_minus_wrong_joint_at_least_0p002": (
            candidate["mean_acc"] - wrong["mean_acc"] >= 0.002
        ),
    }
    gate["pass"] = all(gate.values())
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": common.sha256(args.checkpoint),
            "persistence_weights": str(args.persistence_weights),
            "persistence_weights_sha256": common.sha256(args.persistence_weights),
            "splitview_weights": str(args.splitview_weights),
            "splitview_weights_sha256": common.sha256(args.splitview_weights),
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "depth_view_concordance": concordance_report,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-depth-view-concordance-sae-v1-screen"
            if gate["pass"]
            else "stop-before-depth-view-concordance-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "depth-view-concordance-gate.json"
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Depth-view concordance gate",
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
    (args.output_dir / "depth-view-concordance-gate.md").write_text(
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
