#!/usr/bin/env python3
"""Gate true next-token output-direction predictability before SAE training."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
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
from diagnose_structured_temporal_innovation import (
    sample_separating_cyclic_permutation,
)


def load_next_token_pairs(
    cache_dir: Path,
    manifest: dict[str, Any],
    layer: int,
    max_pairs: int,
    max_pairs_per_sample: int,
    position_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sources: list[torch.Tensor] = []
    next_token_ids: list[torch.Tensor] = []
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
        token_ids = meta["token_ids"].to(torch.int64)
        attention_mask = meta["attention_mask"].to(torch.bool)
        for index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            available_pairs = length - 1
            if available_pairs <= 0:
                continue
            if not bool(attention_mask[index, :length].all().item()):
                raise RuntimeError("Stored attention mask disagrees with sample length")
            pair_count = min(
                available_pairs,
                max_pairs_per_sample,
                max_pairs - emitted,
            )
            sample_id = int(sample_ids[index].item())
            generator = torch.Generator(device="cpu").manual_seed(
                position_seed + sample_id * 1_000_003
            )
            positions = torch.randperm(
                available_pairs, generator=generator
            )[:pair_count].sort().values
            start = int(offsets[index].item())
            sources.append(activations[start + positions].clone())
            next_token_ids.append(token_ids[index, positions + 1].clone())
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
                    torch.cat(next_token_ids, dim=0),
                    torch.cat(pair_sample_ids, dim=0),
                )
        del activations, meta
    raise RuntimeError(f"Expected {max_pairs} next-token pairs, got {emitted}")


def load_output_directions(
    model_weights: Path,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, int]:
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
    vocab_size = int(output_weight.shape[0])
    if int(token_ids.min().item()) < 0 or int(token_ids.max().item()) >= vocab_size:
        raise ValueError("Next-token ID is outside the output vocabulary")
    directions = output_weight[token_ids].float()
    directions = F.normalize(directions, dim=1, eps=1.0e-12)
    del state, output_weight
    return directions, vocab_size


def estimate_nexttoken_output_weights(
    module: Any,
    checkpoint: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
    model_weights: Path,
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
    sources, next_token_ids, sample_ids = load_next_token_pairs(
        cache_dir,
        manifest,
        22,
        pair_count,
        max_pairs_per_sample,
        position_seed,
    )
    output_directions, vocab_size = load_output_directions(
        model_weights, next_token_ids
    )
    wrong_permutation, wrong_shift = sample_separating_cyclic_permutation(
        sample_ids, permutation_seed
    )
    feature_count = int(state["encoder.weight"].shape[0])
    d_model = int(state["encoder.weight"].shape[1])
    sum_target = output_directions.sum(dim=0).to(device)
    sum_target2 = output_directions.square().sum(dim=0).to(device)
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
            true_target = output_directions[start:end].to(
                device=device, non_blocking=True
            )
            wrong_target = output_directions[
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

    predictive_weight = common.rank_weights(true_score, permutation_seed + 1)
    wrong_weight = common.rank_weights(wrong_score, permutation_seed + 2)
    feature_generator = torch.Generator(device="cpu").manual_seed(
        permutation_seed + 3
    )
    feature_permutation = torch.randperm(
        feature_count, generator=feature_generator
    )
    weights = {
        "raw": torch.ones(feature_count, dtype=torch.float32),
        "nexttoken_output": predictive_weight,
        "permuted": predictive_weight[feature_permutation],
        "wrong_alignment": wrong_weight,
    }
    unique_tokens, token_counts = torch.unique(
        next_token_ids, return_counts=True
    )
    token_probabilities = token_counts.float() / pair_count
    token_entropy = float(
        -(token_probabilities * token_probabilities.log()).sum().item()
    )
    report = {
        "signal_definition": (
            "sqrt(mean_d corr(z_t,j, l2_normalize(W_out[next_token]))_d^2)"
        ),
        "pair_count": pair_count,
        "sample_count": int(torch.unique(sample_ids).numel()),
        "max_pairs_per_sample": max_pairs_per_sample,
        "position_seed": position_seed,
        "feature_count": feature_count,
        "d_model": d_model,
        "vocab_size": vocab_size,
        "unique_next_token_count": int(unique_tokens.numel()),
        "next_token_entropy_nats": token_entropy,
        "next_token_perplexity": math.exp(token_entropy),
        "permutation_seed": permutation_seed,
        "wrong_cyclic_shift": wrong_shift,
        "wrong_fixed_pair_count": int(
            (wrong_permutation == torch.arange(pair_count)).sum().item()
        ),
        "wrong_same_sample_pair_count": int(
            (sample_ids[wrong_permutation] == sample_ids).sum().item()
        ),
        "predictive_rank_tie_seed": permutation_seed + 1,
        "wrong_rank_tie_seed": permutation_seed + 2,
        "feature_permutation_seed": permutation_seed + 3,
        "output_direction": distribution(output_directions),
        "output_coordinate_variance": distribution(
            var_target.cpu() / pair_count
        ),
        "feature_variance": distribution(var_z.cpu() / pair_count),
        "predictive_score": distribution(true_score),
        "wrong_alignment_score": distribution(wrong_score),
        "predictive_weight": distribution(predictive_weight),
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
    parser.add_argument("--model-weights", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=16384)
    parser.add_argument("--max-pairs-per-sample", type=int, default=2)
    parser.add_argument("--position-seed", type=int, default=47126)
    parser.add_argument("--batch-pairs", type=int, default=128)
    parser.add_argument("--score-feature-block", type=int, default=2048)
    parser.add_argument("--permutation-seed", type=int, default=47026)
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
    weights, nexttoken_report = estimate_nexttoken_output_weights(
        module,
        args.checkpoint,
        args.cache_dir,
        manifest,
        args.model_weights,
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
    torch.save(weights, args.output_dir / "nexttoken-output-weights.pt")
    (args.output_dir / "nexttoken-output-statistics.json").write_text(
        json.dumps(nexttoken_report, indent=2) + "\n", encoding="utf-8"
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
        ("nexttoken_output", "true next-token output-direction weighting"),
        ("permuted", "permuted next-token-weight control"),
        ("wrong_alignment", "wrong next-token alignment control"),
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
    candidate = summary_by_key["nexttoken_output"]
    reference = summary_by_key["raw"]
    permuted = summary_by_key["permuted"]
    wrong = summary_by_key["wrong_alignment"]
    dataset_deltas = {
        dataset: {
            "reference": common.dataset_mean(
                rows_by_key["raw"], dataset, args.k_values
            ),
            "nexttoken_output": common.dataset_mean(
                rows_by_key["nexttoken_output"], dataset, args.k_values
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta"] = values["nexttoken_output"] - values["reference"]
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
            "model_weights": str(args.model_weights),
            "model_weights_size": args.model_weights.stat().st_size,
            "layer": 22,
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "nexttoken_output_signal": nexttoken_report,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-structured-nexttoken-output-sae-v1-screen"
            if gate["pass"]
            else "stop-before-structured-nexttoken-output-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "structured-nexttoken-output-gate.json"
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Structured next-token output-direction gate",
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
    (args.output_dir / "structured-nexttoken-output-gate.md").write_text(
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
