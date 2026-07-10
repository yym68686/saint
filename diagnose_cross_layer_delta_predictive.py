#!/usr/bin/env python3
"""Gate next-layer delta-predictive feature weighting before SAE training."""

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


def distribution(
    values: torch.Tensor, max_quantile_values: int = 1_000_000
) -> dict[str, float | int]:
    values = values.detach().float().cpu().reshape(-1)
    value_count = int(values.numel())
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


def derangement(length: int, seed: int) -> torch.Tensor:
    if length < 2:
        raise ValueError("A wrong-token control requires at least two tokens")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = torch.arange(length)
    for _ in range(100):
        permutation = torch.randperm(length, generator=generator)
        if not torch.any(permutation == identity):
            return permutation
    # A cyclic shift is an exact derangement and preserves the target marginal.
    shift = int(torch.randint(1, length, (1,), generator=generator).item())
    return torch.roll(identity, shifts=shift)


def load_delta_targets(
    module: Any,
    cache_dir: Path,
    manifest: dict[str, Any],
    max_tokens: int,
    batch_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in common.iter_aligned_tokens(
            cache_dir, manifest, max_tokens, batch_tokens
        ):
            x22 = module.normalize_activation(
                batch[22].to(device=device, non_blocking=True), dtype, eps
            )
            x23 = module.normalize_activation(
                batch[23].to(device=device, non_blocking=True), dtype, eps
            )
            targets.append((x23.float() - x22.float()).cpu())
    result = torch.cat(targets, dim=0)
    if result.shape[0] != max_tokens:
        raise RuntimeError(f"Expected {max_tokens} delta targets, got {result.shape[0]}")
    return result


def score_from_cross_statistics(
    cross: torch.Tensor,
    sum_z: torch.Tensor,
    sum_delta: torch.Tensor,
    var_z: torch.Tensor,
    var_delta: torch.Tensor,
    count: int,
    block_features: int,
) -> torch.Tensor:
    feature_count = int(cross.shape[0])
    scores = torch.zeros(feature_count, dtype=torch.float32)
    for start in range(0, feature_count, block_features):
        end = min(feature_count, start + block_features)
        covariance = cross[start:end] - (
            sum_z[start:end, None] * sum_delta[None, :] / count
        )
        denominator = (
            var_z[start:end, None] * var_delta[None, :]
        ).clamp_min(0).sqrt()
        correlation = torch.where(
            denominator > 1.0e-12,
            covariance / denominator.clamp_min(1.0e-12),
            torch.zeros_like(covariance),
        ).clamp(-1, 1)
        scores[start:end] = correlation.square().mean(dim=1).sqrt().cpu()
    return scores


def estimate_delta_predictive_weights(
    module: Any,
    checkpoint: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
    signal_tokens: int,
    batch_tokens: int,
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
    feature_count = int(state["encoder.weight"].shape[0])
    d_model = int(state["encoder.weight"].shape[1])
    targets = load_delta_targets(
        module,
        cache_dir,
        manifest,
        signal_tokens,
        batch_tokens,
        device,
        dtype,
        eps,
    )
    wrong_permutation = derangement(signal_tokens, permutation_seed)
    sum_delta = targets.sum(dim=0).to(device)
    sum_delta2 = targets.square().sum(dim=0).to(device)
    var_delta = (
        sum_delta2 - sum_delta.square() / signal_tokens
    ).clamp_min(0)
    sum_z = torch.zeros(feature_count, device=device, dtype=torch.float32)
    sum_z2 = torch.zeros_like(sum_z)
    cross_true = torch.zeros(
        (feature_count, d_model), device=device, dtype=torch.float32
    )
    cross_wrong = torch.zeros_like(cross_true)
    count = 0
    with torch.inference_mode():
        for batch in common.iter_aligned_tokens(
            cache_dir, manifest, signal_tokens, batch_tokens
        ):
            local = int(batch[22].shape[0])
            x22 = module.normalize_activation(
                batch[22].to(device=device, non_blocking=True), dtype, eps
            )
            z = torch.relu(
                F.linear(
                    x22 - state["b_pre"],
                    state["encoder.weight"],
                    state["encoder.bias"],
                )
            ).float()
            true_delta = targets[count : count + local].to(
                device=device, non_blocking=True
            )
            wrong_delta = targets[
                wrong_permutation[count : count + local]
            ].to(device=device, non_blocking=True)
            sum_z += z.sum(dim=0)
            sum_z2 += z.square().sum(dim=0)
            cross_true.addmm_(z.T, true_delta)
            cross_wrong.addmm_(z.T, wrong_delta)
            count += local
            del z, true_delta, wrong_delta, x22
    if count != signal_tokens:
        raise RuntimeError(f"Expected {signal_tokens} feature tokens, got {count}")
    var_z = (sum_z2 - sum_z.square() / count).clamp_min(0)
    true_score = score_from_cross_statistics(
        cross_true,
        sum_z,
        sum_delta,
        var_z,
        var_delta,
        count,
        score_feature_block,
    )
    wrong_score = score_from_cross_statistics(
        cross_wrong,
        sum_z,
        sum_delta,
        var_z,
        var_delta,
        count,
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
    permuted_weight = predictive_weight[feature_permutation]
    weights = {
        "raw": torch.ones(feature_count, dtype=torch.float32),
        "delta_predictive": predictive_weight,
        "permuted": permuted_weight,
        "wrong_alignment": wrong_weight,
    }
    report = {
        "signal_definition": (
            "sqrt(mean_d corr(z_j(L22), normalize(x23)-normalize(x22))_d^2)"
        ),
        "signal_tokens": count,
        "feature_count": feature_count,
        "d_model": d_model,
        "permutation_seed": permutation_seed,
        "wrong_alignment": (
            "global token derangement applied to the complete L23-L22 delta"
        ),
        "predictive_rank_tie_seed": permutation_seed + 1,
        "wrong_alignment_rank_tie_seed": permutation_seed + 2,
        "feature_permutation_seed": permutation_seed + 3,
        "delta": distribution(targets),
        "delta_coordinate_variance": distribution(var_delta.cpu() / count),
        "feature_variance": distribution(var_z.cpu() / count),
        "predictive_score": distribution(true_score),
        "wrong_alignment_score": distribution(wrong_score),
        "predictive_weight": distribution(predictive_weight),
        "wrong_alignment_weight": distribution(wrong_weight),
        "score_rank_correlation_true_wrong": float(
            torch.corrcoef(torch.stack([true_score, wrong_score]))[0, 1].item()
        ),
        "wrong_permutation_fixed_points": int(
            (wrong_permutation == torch.arange(signal_tokens)).sum().item()
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
    parser.add_argument("--signal-tokens", type=int, default=8192)
    parser.add_argument("--signal-batch-tokens", type=int, default=128)
    parser.add_argument("--score-feature-block", type=int, default=2048)
    parser.add_argument("--permutation-seed", type=int, default=43026)
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
    weights, signal_report = estimate_delta_predictive_weights(
        module,
        args.checkpoint,
        args.cache_dir,
        manifest,
        args.signal_tokens,
        args.signal_batch_tokens,
        args.score_feature_block,
        args.permutation_seed,
        device,
        dtype,
        1.0e-6,
    )
    torch.save(weights, args.output_dir / "delta-predictive-weights.pt")
    (args.output_dir / "delta-predictive-statistics.json").write_text(
        json.dumps(signal_report, indent=2) + "\n", encoding="utf-8"
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
        ("delta_predictive", "next-layer delta-predictive rank weighting"),
        ("permuted", "permuted predictive-weight control"),
        ("wrong_alignment", "wrong-token delta alignment control"),
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
    candidate = summary_by_key["delta_predictive"]
    reference = summary_by_key["raw"]
    permuted = summary_by_key["permuted"]
    wrong = summary_by_key["wrong_alignment"]
    dataset_deltas = {
        dataset: {
            "reference": common.dataset_mean(
                rows_by_key["raw"], dataset, args.k_values
            ),
            "delta_predictive": common.dataset_mean(
                rows_by_key["delta_predictive"], dataset, args.k_values
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta"] = values["delta_predictive"] - values["reference"]
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
            "layers": [22, 23],
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "delta_predictive_signal": signal_report,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-predictive-delta-sae-v1-screen"
            if gate["pass"]
            else "stop-before-predictive-delta-sae-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "cross-layer-delta-predictive-gate.json"
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Cross-layer next-delta predictive feature gate",
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
    (args.output_dir / "cross-layer-delta-predictive-gate.md").write_text(
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
