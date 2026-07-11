#!/usr/bin/env python3
"""Gate true sample bag-of-token signal before lexical SAE training."""

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


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("structured_bow_eval", path)
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


def build_observed_vocabulary(
    cache_dir: Path,
    manifest: dict[str, Any],
    maximum_features: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    observed: set[int] = set()
    token_count = 0
    document_count = 0
    for shard in manifest["shards"]:
        meta = torch.load(
            cache_dir / shard["meta"]["path"],
            map_location="cpu",
            weights_only=True,
        )
        tokens = meta["token_ids"][meta["attention_mask"]].to(torch.int64)
        observed.update(int(value) for value in torch.unique(tokens).tolist())
        token_count += int(tokens.numel())
        document_count += int(meta["sample_ids"].numel())
    token_ids = torch.tensor(sorted(observed), dtype=torch.int64)
    if token_ids.numel() > maximum_features:
        raise RuntimeError(
            f"Observed vocabulary has {token_ids.numel()} tokens, exceeding "
            f"the preregistered {maximum_features}-feature cap"
        )
    mapping = torch.full(
        (int(token_ids.max().item()) + 1,), -1, dtype=torch.int64
    )
    mapping[token_ids] = torch.arange(token_ids.numel(), dtype=torch.int64)
    return token_ids, mapping, {
        "document_count": document_count,
        "token_count": token_count,
        "observed_token_count": int(token_ids.numel()),
        "minimum_token_id": int(token_ids.min().item()),
        "maximum_token_id": int(token_ids.max().item()),
        "maximum_features": maximum_features,
    }


def binary_bow_features(
    module: Any,
    tokenizer: Any,
    data: dict[str, list[str]],
    token_mapping: torch.Tensor,
    feature_count: int,
    context_length: int,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    result: dict[str, torch.Tensor] = {}
    total_tokens = 0
    covered_tokens = 0
    total_unique = 0
    covered_unique = 0
    empty_samples = 0
    for class_name, texts in data.items():
        output = torch.zeros(
            (len(texts), feature_count), dtype=torch.float32
        )
        for start in range(0, len(texts), batch_size):
            end = min(len(texts), start + batch_size)
            tokens, mask = module.tokenize_texts(
                texts[start:end],
                tokenizer,
                context_length,
                torch.device("cpu"),
            )
            for local_index in range(end - start):
                token_row = tokens[local_index, mask[local_index]].to(torch.int64)
                total_tokens += int(token_row.numel())
                in_range = token_row < token_mapping.numel()
                feature_row = torch.full_like(token_row, -1)
                feature_row[in_range] = token_mapping[token_row[in_range]]
                valid_features = feature_row[feature_row >= 0]
                covered_tokens += int(valid_features.numel())
                unique_tokens = torch.unique(token_row)
                total_unique += int(unique_tokens.numel())
                unique_in_range = unique_tokens < token_mapping.numel()
                unique_features = torch.full_like(unique_tokens, -1)
                unique_features[unique_in_range] = token_mapping[
                    unique_tokens[unique_in_range]
                ]
                unique_features = torch.unique(
                    unique_features[unique_features >= 0]
                )
                covered_unique += int(unique_features.numel())
                if unique_features.numel() == 0:
                    empty_samples += 1
                else:
                    output[start + local_index, unique_features] = 1.0
        result[class_name] = output
    statistics = {
        "sample_count": sum(len(texts) for texts in data.values()),
        "total_tokens": total_tokens,
        "covered_tokens": covered_tokens,
        "token_coverage": covered_tokens / max(total_tokens, 1),
        "total_unique_token_events": total_unique,
        "covered_unique_token_events": covered_unique,
        "unique_token_coverage": covered_unique / max(total_unique, 1),
        "empty_samples": empty_samples,
    }
    return result, statistics


def balanced_wrong_class_control(
    features: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    class_names = list(features)
    if len(class_names) < 2:
        raise ValueError("Wrong-class control requires at least two classes")
    row_count = int(features[class_names[0]].shape[0])
    if any(int(values.shape[0]) != row_count for values in features.values()):
        raise RuntimeError("Wrong-class control requires equal class sizes")
    result = {
        class_name: torch.empty_like(features[class_name])
        for class_name in class_names
    }
    source_counts = {
        target: {source: 0 for source in class_names}
        for target in class_names
    }
    same_class_pairs = 0
    for row in range(row_count):
        offset = 1 + row % (len(class_names) - 1)
        for target_index, target_class in enumerate(class_names):
            source_class = class_names[(target_index + offset) % len(class_names)]
            result[target_class][row] = features[source_class][row]
            source_counts[target_class][source_class] += 1
            same_class_pairs += int(target_class == source_class)
    return result, {
        "rule": "source_class=(target_class+1+row%(C-1))%C",
        "class_count": len(class_names),
        "row_count_per_class": row_count,
        "same_class_pairs": same_class_pairs,
        "source_counts": source_counts,
    }


def mean_relu_features(
    module: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    feature_count = int(state["encoder.weight"].shape[0])
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            sample_count = int(acts_cpu.shape[0])
            output = torch.empty(
                (sample_count, feature_count), dtype=torch.float32
            )
            for start in range(0, sample_count, config.sae_seq_batch_size):
                end = min(sample_count, start + config.sae_seq_batch_size)
                acts = acts_cpu[start:end].to(config.device, non_blocking=True)
                mask = mask_cpu[start:end].to(config.device, non_blocking=True)
                batch_size = end - start
                flat_mask = mask.reshape(-1)
                sample_index = (
                    torch.arange(batch_size, device=config.device)
                    .unsqueeze(1)
                    .expand_as(mask)
                    .reshape(-1)[flat_mask]
                )
                x = acts.reshape(-1, acts.shape[-1])[flat_mask]
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
                    (batch_size, feature_count),
                    device=config.device,
                    dtype=torch.float32,
                )
                sums.index_add_(0, sample_index, z.float())
                lengths = mask.sum(dim=1).clamp_min(1).float().unsqueeze(1)
                output[start:end] = (sums / lengths).cpu()
            result[class_name] = output
    return result


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
                "feature_count": row["feature_count"],
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


def annotate_selected_tokens(
    row: dict[str, Any],
    token_ids: torch.Tensor,
    tokenizer: Any,
) -> None:
    for classes in row["per_class"].values():
        for metrics in classes.values():
            for result in metrics.values():
                feature_indices = result["selected_features"]
                selected_token_ids = [
                    int(token_ids[index].item()) for index in feature_indices
                ]
                result["selected_token_ids"] = selected_token_ids
                result["selected_token_text"] = [
                    tokenizer.decode([token_id]) for token_id in selected_token_ids
                ]


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
    parser.add_argument("--maximum-features", type=int, default=65536)
    parser.add_argument("--datasets", nargs="+", default=INITIAL3)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--sae-seq-batch-size", type=int, default=2)
    parser.add_argument("--bow-batch-size", type=int, default=64)
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

    module = load_eval_module(args.eval_script)
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_ids, token_mapping, vocabulary_statistics = build_observed_vocabulary(
        args.cache_dir, manifest, args.maximum_features
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
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
    raw_state = module.load_state(args.checkpoint)
    state = module.move_keys(
        raw_state,
        ["b_pre", "encoder.weight", "encoder.bias"],
        config.device,
        config.dtype,
    )
    reference_feature_count = int(state["encoder.weight"].shape[0])
    del raw_state

    variant_specs = [
        ("raw_l22_relu", "standard L22 ReLU", reference_feature_count),
        (
            "binary_bow",
            "true-sample binary bag of observed OWT tokens",
            int(token_ids.numel()),
        ),
        (
            "wrong_class_bow",
            "cyclic wrong-class binary bag control",
            int(token_ids.numel()),
        ),
    ]
    rows_by_key = {
        key: {
            "variant_key": key,
            "label": label,
            "feature_count": feature_count,
            "dataset_results": {},
            "per_class": {},
        }
        for key, label, feature_count in variant_specs
    }
    coverage_statistics: dict[str, Any] = {}
    wrong_class_mappings: dict[str, Any] = {}

    for dataset_index, dataset in enumerate(args.datasets):
        print(f"== Dataset: {dataset}", flush=True)
        train_data, test_data = module.get_multi_label_train_test_data(
            dataset,
            args.train_size,
            args.test_size,
            args.random_seed,
        )
        train_bow, train_coverage = binary_bow_features(
            module,
            tokenizer,
            train_data,
            token_mapping,
            int(token_ids.numel()),
            args.context_length,
            args.bow_batch_size,
        )
        test_bow, test_coverage = binary_bow_features(
            module,
            tokenizer,
            test_data,
            token_mapping,
            int(token_ids.numel()),
            args.context_length,
            args.bow_batch_size,
        )
        wrong_train, train_mapping = balanced_wrong_class_control(train_bow)
        wrong_test, test_mapping = balanced_wrong_class_control(test_bow)
        if train_mapping["same_class_pairs"] != 0:
            raise RuntimeError("Train wrong-class control contains same-class pairs")
        if test_mapping["same_class_pairs"] != 0:
            raise RuntimeError("Test wrong-class control contains same-class pairs")
        train_layers, train_masks = module.collect_layer_activations(
            llm, tokenizer, train_data, config, [22]
        )
        test_layers, test_masks = module.collect_layer_activations(
            llm, tokenizer, test_data, config, [22]
        )
        train_relu = mean_relu_features(
            module, train_layers[22], train_masks, state, config
        )
        test_relu = mean_relu_features(
            module, test_layers[22], test_masks, state, config
        )
        dataset_features = {
            "raw_l22_relu": (train_relu, test_relu),
            "binary_bow": (train_bow, test_bow),
            "wrong_class_bow": (wrong_train, wrong_test),
        }
        for variant_key, (train_features, test_features) in dataset_features.items():
            probe = module.probe_one_architecture_dataset(
                train_features,
                test_features,
                args.k_values,
                args.random_seed + 1009 * dataset_index,
            )
            rows_by_key[variant_key]["dataset_results"][dataset] = probe["metrics"]
            rows_by_key[variant_key]["per_class"][dataset] = probe["per_class"]
        coverage_statistics[dataset] = {
            "train": train_coverage,
            "test": test_coverage,
        }
        wrong_class_mappings[dataset] = {
            "train": train_mapping,
            "test": test_mapping,
        }
        del (
            train_layers,
            test_layers,
            train_masks,
            test_masks,
            train_relu,
            test_relu,
            train_bow,
            test_bow,
            wrong_train,
            wrong_test,
        )
        gc.collect()
    del llm, state
    torch.cuda.empty_cache()

    rows = [rows_by_key[key] for key, _, _ in variant_specs]
    annotate_selected_tokens(rows_by_key["binary_bow"], token_ids, tokenizer)
    annotate_selected_tokens(rows_by_key["wrong_class_bow"], token_ids, tokenizer)
    summary = summarize(rows, args.k_values)
    summary_by_key = {row["variant_key"]: row for row in summary}
    candidate = summary_by_key["binary_bow"]
    reference = summary_by_key["raw_l22_relu"]
    wrong = summary_by_key["wrong_class_bow"]
    dataset_deltas = {
        dataset: {
            "reference": dataset_mean(
                rows_by_key["raw_l22_relu"], dataset, args.k_values
            ),
            "candidate": dataset_mean(
                rows_by_key["binary_bow"], dataset, args.k_values
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta"] = values["candidate"] - values["reference"]
    gate = {
        "candidate_feature_count_at_most_65536": int(token_ids.numel()) <= 65536,
        "candidate_minus_reference_at_least_0p005": (
            candidate["mean_acc"] - reference["mean_acc"] >= 0.005
        ),
        "no_dataset_drop_below_minus_0p01": all(
            values["delta"] >= -0.01 for values in dataset_deltas.values()
        ),
        "candidate_minus_wrong_at_least_0p05": (
            candidate["mean_acc"] - wrong["mean_acc"] >= 0.05
        ),
    }
    gate["pass"] = all(gate.values())
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "cache_dir": str(args.cache_dir),
            "cache_manifest_sha256": sha256(manifest_path),
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "context_length": args.context_length,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "mechanism": {
            "candidate": "binary presence of OWT-observed token IDs in each true text sample",
            "control": "cyclic assignment of complete sample vectors to another class",
            "uses_idf_or_frequency_weighting": False,
            "uses_nmf_svd_hash_cluster_or_stem": False,
            "vocabulary_size_sweep": False,
            "same_or_lower_exposed_feature_count_than_reference": True,
            "diagnostic_only": True,
        },
        "vocabulary_statistics": vocabulary_statistics,
        "coverage_statistics": coverage_statistics,
        "wrong_class_mappings": wrong_class_mappings,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-sample-lexical-reconstruction-sae-v1-screen"
            if gate["pass"]
            else "stop-before-sample-lexical-reconstruction-training"
        ),
        "training_ran": False,
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "structured-bow-signal-gate.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Structured sample binary bag-of-token signal gate",
        "",
        "| Variant | Features | Mean Acc | Mean AUC | Top-1 | Top-2 | Top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['label']} | {row['feature_count']} | "
            f"{row['mean_acc']:.6f} | {row['mean_auc']:.6f} | "
            f"{row['top_1_acc']:.6f} | {row['top_2_acc']:.6f} | "
            f"{row['top_5_acc']:.6f} |"
        )
    lines.extend(["", f"Decision: `{payload['decision']}`", ""])
    (args.output_dir / "structured-bow-signal-gate.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "vocabulary_statistics": vocabulary_statistics,
                "summary": summary,
                "dataset_deltas": dataset_deltas,
                "gate": gate,
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
