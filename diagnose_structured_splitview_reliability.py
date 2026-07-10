#!/usr/bin/env python3
"""Gate true-sample split-view feature reliability before SAE training."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

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


def derangement(length: int, seed: int) -> torch.Tensor:
    if length < 2:
        raise ValueError("A shuffled-sample control requires at least two samples")
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


def estimate_reliability_weights(
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
    stats = {
        name: torch.zeros(feature_count, device=device, dtype=torch.float32)
        for name in ("sum_a", "sum_b", "sum_a2", "sum_b2", "sum_ab", "sum_ab_wrong")
    }
    samples_seen = 0
    tokens_seen = 0
    view_a_tokens = 0
    view_b_tokens = 0
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
            pooled = []
            counts = []
            for view in (0, 1):
                mask = view_index == view
                sums = torch.zeros(
                    (local_samples, feature_count),
                    device=device,
                    dtype=torch.float32,
                )
                sums.index_add_(0, sample_index[mask], z[mask])
                count = torch.bincount(
                    sample_index[mask], minlength=local_samples
                ).clamp_min(1)
                pooled.append(sums / count.float().unsqueeze(1))
                counts.append(count)
            view_a, view_b = pooled
            permutation = derangement(
                local_samples, permutation_seed + batch_index
            ).to(device)
            stats["sum_a"] += view_a.sum(dim=0)
            stats["sum_b"] += view_b.sum(dim=0)
            stats["sum_a2"] += view_a.square().sum(dim=0)
            stats["sum_b2"] += view_b.square().sum(dim=0)
            stats["sum_ab"] += (view_a * view_b).sum(dim=0)
            stats["sum_ab_wrong"] += (view_a * view_b[permutation]).sum(dim=0)
            samples_seen += local_samples
            tokens_seen += int(batch["activations"].shape[0])
            view_a_tokens += int(counts[0].sum().item())
            view_b_tokens += int(counts[1].sum().item())
            del x, z, pooled, view_a, view_b, permutation
    if samples_seen != sample_count:
        raise RuntimeError(f"Expected {sample_count} samples, got {samples_seen}")
    reliability = common.correlation_from_sums(
        stats["sum_a"],
        stats["sum_b"],
        stats["sum_a2"],
        stats["sum_b2"],
        stats["sum_ab"],
        samples_seen,
    ).cpu()
    wrong = common.correlation_from_sums(
        stats["sum_a"],
        stats["sum_b"],
        stats["sum_a2"],
        stats["sum_b2"],
        stats["sum_ab_wrong"],
        samples_seen,
    ).cpu()
    del state, stats
    torch.cuda.empty_cache()

    reliability_weight = common.rank_weights(
        reliability, permutation_seed + 100_001
    )
    wrong_weight = common.rank_weights(wrong, permutation_seed + 100_002)
    feature_generator = torch.Generator(device="cpu").manual_seed(
        permutation_seed + 100_003
    )
    feature_permutation = torch.randperm(
        feature_count, generator=feature_generator
    )
    weights = {
        "raw": torch.ones(feature_count, dtype=torch.float32),
        "splitview_reliability": reliability_weight,
        "permuted": reliability_weight[feature_permutation],
        "wrong_sample": wrong_weight,
    }
    report = {
        "signal_definition": (
            "per-feature Pearson correlation between odd/even token mean "
            "responses from the same true OWT sample"
        ),
        "sample_count": samples_seen,
        "tokens_seen": tokens_seen,
        "average_tokens_per_sample": tokens_seen / samples_seen,
        "view_a_tokens": view_a_tokens,
        "view_b_tokens": view_b_tokens,
        "batch_samples": batch_samples,
        "min_sample_tokens": min_sample_tokens,
        "feature_count": feature_count,
        "permutation_seed_base": permutation_seed,
        "reliability_rank_tie_seed": permutation_seed + 100_001,
        "wrong_sample_rank_tie_seed": permutation_seed + 100_002,
        "feature_permutation_seed": permutation_seed + 100_003,
        "reliability_score": distribution(reliability),
        "wrong_sample_score": distribution(wrong),
        "reliability_weight": distribution(reliability_weight),
        "wrong_sample_weight": distribution(wrong_weight),
        "score_correlation_true_wrong": float(
            torch.corrcoef(torch.stack([reliability, wrong]))[0, 1].item()
        ),
        "positive_reliability_fraction": float((reliability > 0).float().mean()),
        "positive_wrong_fraction": float((wrong > 0).float().mean()),
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
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--batch-samples", type=int, default=16)
    parser.add_argument("--min-sample-tokens", type=int, default=2)
    parser.add_argument("--permutation-seed", type=int, default=44026)
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
    weights, reliability_report = estimate_reliability_weights(
        module,
        args.checkpoint,
        args.cache_dir,
        manifest,
        args.sample_count,
        args.batch_samples,
        args.min_sample_tokens,
        args.permutation_seed,
        device,
        dtype,
        1.0e-6,
    )
    torch.save(weights, args.output_dir / "splitview-reliability-weights.pt")
    (args.output_dir / "splitview-reliability-statistics.json").write_text(
        json.dumps(reliability_report, indent=2) + "\n", encoding="utf-8"
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
        ("splitview_reliability", "true-sample split-view reliability weighting"),
        ("permuted", "permuted reliability-weight control"),
        ("wrong_sample", "wrong-sample split-view control"),
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
    candidate = summary_by_key["splitview_reliability"]
    reference = summary_by_key["raw"]
    permuted = summary_by_key["permuted"]
    wrong = summary_by_key["wrong_sample"]
    dataset_deltas = {
        dataset: {
            "reference": common.dataset_mean(
                rows_by_key["raw"], dataset, args.k_values
            ),
            "splitview_reliability": common.dataset_mean(
                rows_by_key["splitview_reliability"], dataset, args.k_values
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta"] = values["splitview_reliability"] - values["reference"]
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
        "candidate_minus_wrong_sample_at_least_0p002": (
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
        "splitview_reliability_signal": reliability_report,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-structured-splitview-reliability-sae-v1-screen"
            if gate["pass"]
            else "stop-before-structured-splitview-reliability-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "structured-splitview-reliability-gate.json"
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Structured true-sample split-view reliability gate",
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
    (args.output_dir / "structured-splitview-reliability-gate.md").write_text(
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
