#!/usr/bin/env python3
"""Gate true-sequence next-token innovation predictability before SAE training."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import diagnose_cross_layer_feature_persistence as common
from diagnose_cross_layer_delta_predictive import (
    distribution,
    score_from_cross_statistics,
)


def load_temporal_pairs(
    cache_dir: Path,
    manifest: dict[str, Any],
    layer: int,
    max_pairs: int,
    max_pairs_per_sample: int,
    position_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    pair_sample_ids: list[torch.Tensor] = []
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
        sample_ids = meta["sample_ids"].to(torch.int64)
        for index, length_tensor in enumerate(lengths):
            available_pairs = int(length_tensor.item()) - 1
            if available_pairs <= 0:
                continue
            pair_count = min(
                available_pairs,
                max_pairs_per_sample,
                max_pairs - emitted,
            )
            start = int(offsets[index].item())
            sample_id = int(sample_ids[index].item())
            generator = torch.Generator(device="cpu").manual_seed(
                position_seed + sample_id * 1_000_003
            )
            positions = torch.randperm(
                available_pairs, generator=generator
            )[:pair_count].sort().values
            sources.append(activations[start + positions].clone())
            targets.append(activations[start + positions + 1].clone())
            pair_sample_ids.append(
                torch.full(
                    (pair_count,),
                    sample_id,
                    dtype=torch.int64,
                )
            )
            emitted += pair_count
            if emitted >= max_pairs:
                return (
                    torch.cat(sources, dim=0),
                    torch.cat(targets, dim=0),
                    torch.cat(pair_sample_ids, dim=0),
                )
        del activations, meta
    raise RuntimeError(f"Expected {max_pairs} temporal pairs, got {emitted}")


def sample_separating_cyclic_permutation(
    sample_ids: torch.Tensor, seed: int
) -> tuple[torch.Tensor, int]:
    length = int(sample_ids.numel())
    if length < 2:
        raise ValueError("Temporal wrong control requires at least two pairs")
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
    raise RuntimeError("Could not construct a sample-separating cyclic permutation")


def normalized_innovations(
    module: Any,
    sources: torch.Tensor,
    targets: torch.Tensor,
    batch_pairs: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    result = torch.empty(
        (sources.shape[0], sources.shape[1]), dtype=torch.float32
    )
    with torch.inference_mode():
        for start in range(0, sources.shape[0], batch_pairs):
            end = min(sources.shape[0], start + batch_pairs)
            current = module.normalize_activation(
                sources[start:end].to(device=device, non_blocking=True),
                dtype,
                eps,
            )
            following = module.normalize_activation(
                targets[start:end].to(device=device, non_blocking=True),
                dtype,
                eps,
            )
            result[start:end] = (following.float() - current.float()).cpu()
    return result


def estimate_temporal_innovation_weights(
    module: Any,
    checkpoint: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
    pair_count: int,
    max_pairs_per_sample: int,
    position_seed: int,
    batch_pairs: int,
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
    sources, following, sample_ids = load_temporal_pairs(
        cache_dir,
        manifest,
        22,
        pair_count,
        max_pairs_per_sample,
        position_seed,
    )
    innovations = normalized_innovations(
        module,
        sources,
        following,
        batch_pairs,
        device,
        dtype,
        eps,
    )
    del following
    wrong_permutation, wrong_shift = sample_separating_cyclic_permutation(
        sample_ids, permutation_seed
    )
    feature_count = int(state["encoder.weight"].shape[0])
    d_model = int(state["encoder.weight"].shape[1])
    sum_target = innovations.sum(dim=0).to(device)
    sum_target2 = innovations.square().sum(dim=0).to(device)
    var_target = (
        sum_target2 - sum_target.square() / pair_count
    ).clamp_min(0)
    sum_z = torch.zeros(feature_count, device=device, dtype=torch.float32)
    sum_z2 = torch.zeros_like(sum_z)
    cross_true = torch.zeros(
        (feature_count, d_model), device=device, dtype=torch.float32
    )
    cross_wrong = torch.zeros_like(cross_true)
    with torch.inference_mode():
        for start in range(0, pair_count, batch_pairs):
            end = min(pair_count, start + batch_pairs)
            x = module.normalize_activation(
                sources[start:end].to(device=device, non_blocking=True),
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
            true_target = innovations[start:end].to(
                device=device, non_blocking=True
            )
            wrong_target = innovations[
                wrong_permutation[start:end]
            ].to(device=device, non_blocking=True)
            sum_z += z.sum(dim=0)
            sum_z2 += z.square().sum(dim=0)
            cross_true.addmm_(z.T, true_target)
            cross_wrong.addmm_(z.T, wrong_target)
            del x, z, true_target, wrong_target
    var_z = (sum_z2 - sum_z.square() / pair_count).clamp_min(0)
    true_score = score_from_cross_statistics(
        cross_true,
        sum_z,
        sum_target,
        var_z,
        var_target,
        pair_count,
        score_feature_block,
    )
    wrong_score = score_from_cross_statistics(
        cross_wrong,
        sum_z,
        sum_target,
        var_z,
        var_target,
        pair_count,
        score_feature_block,
    )
    del cross_true, cross_wrong, state
    torch.cuda.empty_cache()

    temporal_weight = common.rank_weights(true_score, permutation_seed + 1)
    wrong_weight = common.rank_weights(wrong_score, permutation_seed + 2)
    feature_generator = torch.Generator(device="cpu").manual_seed(
        permutation_seed + 3
    )
    feature_permutation = torch.randperm(
        feature_count, generator=feature_generator
    )
    weights = {
        "raw": torch.ones(feature_count, dtype=torch.float32),
        "temporal_innovation": temporal_weight,
        "permuted": temporal_weight[feature_permutation],
        "wrong_alignment": wrong_weight,
    }
    report = {
        "signal_definition": (
            "sqrt(mean_d corr(z_t,j, normalize(x_t+1)-normalize(x_t))_d^2)"
        ),
        "pair_count": pair_count,
        "sample_count": int(torch.unique(sample_ids).numel()),
        "max_pairs_per_sample": max_pairs_per_sample,
        "position_seed": position_seed,
        "feature_count": feature_count,
        "d_model": d_model,
        "permutation_seed": permutation_seed,
        "wrong_cyclic_shift": wrong_shift,
        "wrong_fixed_pair_count": int(
            (wrong_permutation == torch.arange(pair_count)).sum().item()
        ),
        "wrong_same_sample_pair_count": int(
            (sample_ids[wrong_permutation] == sample_ids).sum().item()
        ),
        "temporal_rank_tie_seed": permutation_seed + 1,
        "wrong_rank_tie_seed": permutation_seed + 2,
        "feature_permutation_seed": permutation_seed + 3,
        "innovation": distribution(innovations),
        "innovation_coordinate_variance": distribution(
            var_target.cpu() / pair_count
        ),
        "feature_variance": distribution(var_z.cpu() / pair_count),
        "temporal_score": distribution(true_score),
        "wrong_alignment_score": distribution(wrong_score),
        "temporal_weight": distribution(temporal_weight),
        "wrong_alignment_weight": distribution(wrong_weight),
        "score_correlation_true_wrong": float(
            torch.corrcoef(torch.stack([true_score, wrong_score]))[0, 1].item()
        ),
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=16384)
    parser.add_argument("--max-pairs-per-sample", type=int, default=2)
    parser.add_argument("--position-seed", type=int, default=46126)
    parser.add_argument("--batch-pairs", type=int, default=128)
    parser.add_argument("--score-feature-block", type=int, default=2048)
    parser.add_argument("--permutation-seed", type=int, default=46026)
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
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    started = time.time()
    weights, temporal_report = estimate_temporal_innovation_weights(
        module,
        args.checkpoint,
        args.cache_dir,
        manifest,
        args.pair_count,
        args.max_pairs_per_sample,
        args.position_seed,
        args.batch_pairs,
        args.score_feature_block,
        args.permutation_seed,
        device,
        dtype,
        1.0e-6,
    )
    torch.save(weights, args.output_dir / "temporal-innovation-weights.pt")
    (args.output_dir / "temporal-innovation-statistics.json").write_text(
        json.dumps(temporal_report, indent=2) + "\n", encoding="utf-8"
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
        ("temporal_innovation", "true-sequence temporal-innovation weighting"),
        ("permuted", "permuted temporal-weight control"),
        ("wrong_alignment", "wrong-sequence innovation control"),
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
    candidate = summary_by_key["temporal_innovation"]
    reference = summary_by_key["raw"]
    permuted = summary_by_key["permuted"]
    wrong = summary_by_key["wrong_alignment"]
    dataset_deltas = {
        dataset: {
            "reference": common.dataset_mean(
                rows_by_key["raw"], dataset, args.k_values
            ),
            "temporal_innovation": common.dataset_mean(
                rows_by_key["temporal_innovation"], dataset, args.k_values
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta"] = values["temporal_innovation"] - values["reference"]
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
            "checkpoint_sha256": common.sha256(args.checkpoint),
            "cache_dir": str(args.cache_dir),
            "cache_manifest_sha256": common.sha256(manifest_path),
            "layer": 22,
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "temporal_innovation_signal": temporal_report,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-structured-temporal-innovation-sae-v1-screen"
            if gate["pass"]
            else "stop-before-structured-temporal-innovation-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "structured-temporal-innovation-gate.json"
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Structured temporal-innovation gate",
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
    (args.output_dir / "structured-temporal-innovation-gate.md").write_text(
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
