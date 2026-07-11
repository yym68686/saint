#!/usr/bin/env python3
"""Gate cross-layer Procrustes alignment before joint SAE training."""

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
from typing import Any

import torch
import torch.nn.functional as F


INITIAL3 = [
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "fancyzhx/ag_news",
]
LAYERS = [20, 21, 22, 23]
SOURCE_LAYERS = [20, 21, 23]


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("procrustes_layer_eval", path)
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


def sample_separating_cyclic_permutation(
    sample_ids: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    identity = torch.arange(sample_ids.numel())
    for shift in range(1, sample_ids.numel()):
        permutation = torch.roll(identity, shifts=shift)
        if not torch.any(sample_ids[permutation] == sample_ids):
            return permutation, shift
    raise RuntimeError("Could not construct a sample-separating permutation")


def collect_aligned_tokens(
    cache_dir: Path,
    manifest: dict[str, Any],
    token_count: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    d_model = int(torch.load(
        cache_dir / manifest["layer_means"]["22"]["path"],
        map_location="cpu",
        weights_only=True,
    ).numel())
    outputs = {
        layer: torch.empty((token_count, d_model), dtype=torch.bfloat16)
        for layer in LAYERS
    }
    output_sample_ids = torch.empty(token_count, dtype=torch.int64)
    emitted = 0
    for shard in manifest["shards"]:
        meta = torch.load(
            cache_dir / shard["meta"]["path"],
            map_location="cpu",
            weights_only=True,
        )
        lengths = meta["lengths"].to(torch.int64)
        token_sample_ids = torch.repeat_interleave(
            meta["sample_ids"].to(torch.int64), lengths
        )
        shard_layers = {
            layer: torch.load(
                cache_dir / shard["layers"][str(layer)]["path"],
                map_location="cpu",
                weights_only=True,
            )
            for layer in LAYERS
        }
        shard_tokens = int(token_sample_ids.numel())
        if any(int(values.shape[0]) != shard_tokens for values in shard_layers.values()):
            raise RuntimeError("Structured cache layers and sample metadata are misaligned")
        take = min(token_count - emitted, shard_tokens)
        if take <= 0:
            break
        for layer in LAYERS:
            outputs[layer][emitted : emitted + take] = shard_layers[layer][:take]
        output_sample_ids[emitted : emitted + take] = token_sample_ids[:take]
        emitted += take
        if emitted == token_count:
            break
    if emitted != token_count:
        raise RuntimeError(f"Requested {token_count} tokens, collected {emitted}")
    return outputs, output_sample_ids


def fit_orthogonal_procrustes(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cross = source.transpose(0, 1) @ target
    u, singular_values, vh = torch.linalg.svd(cross, full_matrices=False)
    transform = u @ vh
    return transform, singular_values


def alignment_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    transform: torch.Tensor,
) -> dict[str, float]:
    aligned = source @ transform
    cosine = F.cosine_similarity(aligned, target, dim=1).float()
    relative_error = float(
        (aligned - target).norm().div(target.norm().clamp_min(1.0e-12)).item()
    )
    identity = torch.eye(
        transform.shape[0], device=transform.device, dtype=transform.dtype
    )
    orthogonality_error = float(
        (transform.transpose(0, 1) @ transform - identity)
        .norm()
        .div(transform.shape[0])
        .item()
    )
    return {
        "mean_cosine": float(cosine.mean().item()),
        "std_cosine": float(cosine.std().item()),
        "p10_cosine": float(torch.quantile(cosine, 0.1).item()),
        "median_cosine": float(torch.quantile(cosine, 0.5).item()),
        "p90_cosine": float(torch.quantile(cosine, 0.9).item()),
        "relative_frobenius_error": relative_error,
        "orthogonality_error": orthogonality_error,
    }


def estimate_procrustes(
    cache_dir: Path,
    manifest: dict[str, Any],
    fit_tokens: int,
    holdout_tokens: int,
    device: torch.device,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[str, Any],
]:
    all_tokens, all_sample_ids = collect_aligned_tokens(
        cache_dir, manifest, fit_tokens + holdout_tokens
    )
    means = {
        layer: torch.load(
            cache_dir / manifest["layer_means"][str(layer)]["path"],
            map_location="cpu",
            weights_only=True,
        ).float()
        for layer in LAYERS
    }
    permutation, wrong_shift = sample_separating_cyclic_permutation(
        all_sample_ids[:fit_tokens]
    )
    target_fit = (
        all_tokens[22][:fit_tokens].float() - means[22]
    ).to(device)
    target_holdout = (
        all_tokens[22][fit_tokens:].float() - means[22]
    ).to(device)
    transforms: dict[int, torch.Tensor] = {
        22: torch.eye(target_fit.shape[1], dtype=torch.float32)
    }
    wrong_transforms: dict[int, torch.Tensor] = {
        22: torch.eye(target_fit.shape[1], dtype=torch.float32)
    }
    layer_results: dict[str, Any] = {}
    for layer in SOURCE_LAYERS:
        source_fit = (
            all_tokens[layer][:fit_tokens].float() - means[layer]
        ).to(device)
        source_holdout = (
            all_tokens[layer][fit_tokens:].float() - means[layer]
        ).to(device)
        transform, singular_values = fit_orthogonal_procrustes(
            source_fit, target_fit
        )
        wrong_transform, _ = fit_orthogonal_procrustes(
            source_fit, target_fit[permutation.to(device)]
        )
        identity = torch.eye(
            transform.shape[0], device=device, dtype=transform.dtype
        )
        true_metrics = alignment_metrics(
            source_holdout, target_holdout, transform
        )
        identity_metrics = alignment_metrics(
            source_holdout, target_holdout, identity
        )
        wrong_metrics = alignment_metrics(
            source_holdout, target_holdout, wrong_transform
        )
        layer_results[str(layer)] = {
            "true_alignment": true_metrics,
            "identity_alignment": identity_metrics,
            "wrong_pair_alignment": wrong_metrics,
            "true_minus_identity_cosine": (
                true_metrics["mean_cosine"] - identity_metrics["mean_cosine"]
            ),
            "true_minus_wrong_cosine": (
                true_metrics["mean_cosine"] - wrong_metrics["mean_cosine"]
            ),
            "singular_value_mean": float(singular_values.mean().item()),
            "singular_value_min": float(singular_values.min().item()),
            "singular_value_max": float(singular_values.max().item()),
        }
        transforms[layer] = transform.cpu()
        wrong_transforms[layer] = wrong_transform.cpu()
        del source_fit, source_holdout, transform, wrong_transform, singular_values
        torch.cuda.empty_cache()
    improvements = [
        values["true_minus_identity_cosine"] for values in layer_results.values()
    ]
    wrong_margins = [
        values["true_minus_wrong_cosine"] for values in layer_results.values()
    ]
    precheck = {
        "mean_true_minus_identity_at_least_0p01": (
            sum(improvements) / len(improvements) >= 0.01
        ),
        "minimum_true_minus_identity_at_least_0p005": min(improvements) >= 0.005,
        "mean_true_minus_wrong_at_least_0p01": (
            sum(wrong_margins) / len(wrong_margins) >= 0.01
        ),
    }
    precheck["pass"] = all(precheck.values())
    report = {
        "fit_tokens": fit_tokens,
        "holdout_tokens": holdout_tokens,
        "wrong_pair_shift": wrong_shift,
        "wrong_pair_fixed_points": int(
            (permutation == torch.arange(fit_tokens)).sum().item()
        ),
        "wrong_pair_same_sample_pairs": int(
            (
                all_sample_ids[:fit_tokens][permutation]
                == all_sample_ids[:fit_tokens]
            ).sum().item()
        ),
        "layer_results": layer_results,
        "precheck": precheck,
    }
    del all_tokens, all_sample_ids, target_fit, target_holdout
    torch.cuda.empty_cache()
    return transforms, wrong_transforms, means, report


def pooled_relu_variants(
    module: Any,
    layer_acts: dict[int, dict[str, torch.Tensor]],
    masks: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    transforms: dict[int, torch.Tensor],
    wrong_transforms: dict[int, torch.Tensor],
    means: dict[int, torch.Tensor],
    config: Any,
) -> dict[str, dict[str, torch.Tensor]]:
    feature_count = int(state["encoder.weight"].shape[0])
    d_model = int(state["encoder.weight"].shape[1])
    gpu_means = {
        layer: values.to(config.device, dtype=config.dtype)
        for layer, values in means.items()
    }
    gpu_transforms = {
        layer: values.to(config.device, dtype=config.dtype)
        for layer, values in transforms.items()
    }
    gpu_wrong_transforms = {
        layer: values.to(config.device, dtype=config.dtype)
        for layer, values in wrong_transforms.items()
    }
    variants: dict[str, dict[str, torch.Tensor]] = {
        "raw_l22": {},
        "identity_layer_average": {},
        "procrustes_layer_average": {},
        "wrong_procrustes_average": {},
    }

    def encode_and_pool(
        activations: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(activations.shape[0])
        flat_mask = mask.reshape(-1)
        sample_index = (
            torch.arange(batch_size, device=config.device)
            .unsqueeze(1)
            .expand_as(mask)
            .reshape(-1)[flat_mask]
        )
        x = activations.reshape(-1, d_model)[flat_mask]
        x = module.normalize_activation(x, config.dtype, config.normalize_eps)
        z = torch.relu(
            F.linear(
                x - state["b_pre"],
                state["encoder.weight"],
                state["encoder.bias"],
            )
        )
        sums = torch.zeros(
            (batch_size, feature_count),
            device=config.device,
            dtype=torch.float32,
        )
        sums.index_add_(0, sample_index, z.float())
        lengths = mask.sum(dim=1).clamp_min(1).float().unsqueeze(1)
        return sums / lengths

    with torch.inference_mode():
        for class_name, mask_cpu in masks.items():
            sample_count = int(mask_cpu.shape[0])
            outputs = {
                key: torch.empty((sample_count, feature_count), dtype=torch.float32)
                for key in variants
            }
            for start in range(0, sample_count, config.sae_seq_batch_size):
                end = min(sample_count, start + config.sae_seq_batch_size)
                mask = mask_cpu[start:end].to(config.device, non_blocking=True)
                identity_average = torch.zeros(
                    (end - start, mask.shape[1], d_model),
                    device=config.device,
                    dtype=config.dtype,
                )
                aligned_average = torch.zeros_like(identity_average)
                wrong_average = torch.zeros_like(identity_average)
                raw_l22 = None
                for layer in LAYERS:
                    values = layer_acts[layer][class_name][start:end].to(
                        config.device, non_blocking=True
                    )
                    identity_average.add_(values)
                    if layer == 22:
                        raw_l22 = values
                        aligned_average.add_(values)
                        wrong_average.add_(values)
                    else:
                        centered = values - gpu_means[layer]
                        aligned_average.add_(
                            torch.matmul(centered, gpu_transforms[layer])
                            + gpu_means[22]
                        )
                        wrong_average.add_(
                            torch.matmul(centered, gpu_wrong_transforms[layer])
                            + gpu_means[22]
                        )
                if raw_l22 is None:
                    raise RuntimeError("L22 activations were not collected")
                identity_average.div_(len(LAYERS))
                aligned_average.div_(len(LAYERS))
                wrong_average.div_(len(LAYERS))
                outputs["raw_l22"][start:end] = encode_and_pool(
                    raw_l22, mask
                ).cpu()
                outputs["identity_layer_average"][start:end] = encode_and_pool(
                    identity_average, mask
                ).cpu()
                outputs["procrustes_layer_average"][start:end] = encode_and_pool(
                    aligned_average, mask
                ).cpu()
                outputs["wrong_procrustes_average"][start:end] = encode_and_pool(
                    wrong_average, mask
                ).cpu()
                del (
                    mask,
                    raw_l22,
                    identity_average,
                    aligned_average,
                    wrong_average,
                )
            for key in variants:
                variants[key][class_name] = outputs[key]
    return variants


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
    parser.add_argument("--fit-tokens", type=int, default=8192)
    parser.add_argument("--holdout-tokens", type=int, default=8192)
    parser.add_argument("--datasets", nargs="+", default=INITIAL3)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--sae-seq-batch-size", type=int, default=1)
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

    started = time.time()
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    transforms, wrong_transforms, means, alignment_report = estimate_procrustes(
        args.cache_dir,
        manifest,
        args.fit_tokens,
        args.holdout_tokens,
        device,
    )
    if not alignment_report["precheck"]["pass"]:
        payload = {
            "config": vars(args) | {
                "checkpoint": str(args.checkpoint),
                "cache_dir": str(args.cache_dir),
                "model_dir": str(args.model_dir),
                "output_dir": str(args.output_dir),
                "eval_script": str(args.eval_script),
            },
            "alignment_report": alignment_report,
            "decision": "stop-before-procrustes-initial3",
            "benchmark_evaluation_ran": False,
            "training_ran": False,
            "elapsed_seconds": time.time() - started,
        }
        output_json = args.output_dir / "procrustes-layer-gate.json"
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    module = load_eval_module(args.eval_script)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
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
    llm = module.load_model(args.model_dir, LAYERS, config.device, config.dtype)
    raw_state = module.load_state(args.checkpoint)
    state = module.move_keys(
        raw_state,
        ["b_pre", "encoder.weight", "encoder.bias"],
        config.device,
        config.dtype,
    )
    del raw_state

    cached: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for dataset in args.datasets:
        print(f"== Cache dataset: {dataset}", flush=True)
        train_data, test_data = module.get_multi_label_train_test_data(
            dataset,
            args.train_size,
            args.test_size,
            args.random_seed,
        )
        train_layers, train_masks = module.collect_layer_activations(
            llm, tokenizer, train_data, config, LAYERS
        )
        test_layers, test_masks = module.collect_layer_activations(
            llm, tokenizer, test_data, config, LAYERS
        )
        cached[dataset] = {
            "train": pooled_relu_variants(
                module,
                train_layers,
                train_masks,
                state,
                transforms,
                wrong_transforms,
                means,
                config,
            ),
            "test": pooled_relu_variants(
                module,
                test_layers,
                test_masks,
                state,
                transforms,
                wrong_transforms,
                means,
                config,
            ),
        }
        del train_layers, test_layers, train_masks, test_masks
    del llm, state, transforms, wrong_transforms, means
    torch.cuda.empty_cache()
    gc.collect()

    variants = [
        ("raw_l22", "standard L22 ReLU"),
        ("identity_layer_average", "unaligned L20-L23 average"),
        ("procrustes_layer_average", "Procrustes-aligned L20-L23 average"),
        ("wrong_procrustes_average", "wrong-pair Procrustes average"),
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
        for dataset_index, (dataset, splits) in enumerate(cached.items()):
            probe = module.probe_one_architecture_dataset(
                splits["train"][variant_key],
                splits["test"][variant_key],
                args.k_values,
                args.random_seed + 1009 * dataset_index,
            )
            row["dataset_results"][dataset] = probe["metrics"]
            row["per_class"][dataset] = probe["per_class"]
        rows.append(row)

    summary = summarize(rows, args.k_values)
    summary_by_key = {row["variant_key"]: row for row in summary}
    rows_by_key = {row["variant_key"]: row for row in rows}
    candidate = summary_by_key["procrustes_layer_average"]
    reference = summary_by_key["raw_l22"]
    identity = summary_by_key["identity_layer_average"]
    wrong = summary_by_key["wrong_procrustes_average"]
    dataset_deltas = {
        dataset: {
            "reference": dataset_mean(
                rows_by_key["raw_l22"], dataset, args.k_values
            ),
            "candidate": dataset_mean(
                rows_by_key["procrustes_layer_average"],
                dataset,
                args.k_values,
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta"] = values["candidate"] - values["reference"]
    benchmark_gate = {
        "candidate_minus_reference_at_least_0p005": (
            candidate["mean_acc"] - reference["mean_acc"] >= 0.005
        ),
        "no_dataset_drop_below_minus_0p01": all(
            values["delta"] >= -0.01 for values in dataset_deltas.values()
        ),
        "candidate_minus_identity_at_least_0p002": (
            candidate["mean_acc"] - identity["mean_acc"] >= 0.002
        ),
        "candidate_minus_wrong_at_least_0p002": (
            candidate["mean_acc"] - wrong["mean_acc"] >= 0.002
        ),
    }
    benchmark_gate["pass"] = all(benchmark_gate.values())
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "cache_dir": str(args.cache_dir),
            "cache_manifest_sha256": sha256(manifest_path),
            "layers": LAYERS,
            "fit_tokens": args.fit_tokens,
            "holdout_tokens": args.holdout_tokens,
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "mechanism": {
            "fit_objective": "argmin_Q ||(H_l-mu_l)Q-(H_22-mu_22)||_F, Q^TQ=I",
            "candidate": "encode(mean_l((H_l-mu_l)Q_l+mu_22))",
            "identity_control": "encode(mean_l(H_l))",
            "wrong_control": "same Procrustes fit with sample-separating target permutation",
            "same_checkpoint": True,
            "same_parameter_count": True,
            "same_exposed_feature_count": True,
            "alignment_or_layer_weight_sweep": False,
        },
        "alignment_report": alignment_report,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "benchmark_gate": benchmark_gate,
        "decision": (
            "authorize-procrustes-joint-layer-sae-v1-screen"
            if benchmark_gate["pass"]
            else "stop-before-procrustes-joint-layer-sae-training"
        ),
        "benchmark_evaluation_ran": True,
        "training_ran": False,
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "procrustes-layer-gate.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Structured cross-layer Procrustes gate",
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
    (args.output_dir / "procrustes-layer-gate.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "alignment_precheck": alignment_report["precheck"],
                "summary": summary,
                "benchmark_gate": benchmark_gate,
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
