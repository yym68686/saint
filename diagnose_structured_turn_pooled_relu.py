#!/usr/bin/env python3
"""Compare mean-of-token encoding with encoding the true sample mean."""

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
    spec = importlib.util.spec_from_file_location("turn_pooled_eval", path)
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


def compute_relu_pooling_orders(
    module: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, dict[str, torch.Tensor]]:
    feature_count = int(state["encoder.weight"].shape[0])
    standard: dict[str, torch.Tensor] = {}
    pooled_first: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            sample_count = int(acts_cpu.shape[0])
            standard_output = torch.empty(
                (sample_count, feature_count), dtype=torch.float32
            )
            pooled_output = torch.empty_like(standard_output)
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
                x_norm = module.normalize_activation(
                    x, config.dtype, config.normalize_eps
                )
                centered = x_norm - state["b_pre"]

                token_features = torch.relu(
                    F.linear(
                        centered,
                        state["encoder.weight"],
                        state["encoder.bias"],
                    )
                )
                token_sums = torch.zeros(
                    (local_count, feature_count),
                    device=config.device,
                    dtype=torch.float32,
                )
                token_sums.index_add_(0, sample_index, token_features.float())
                standard_batch = token_sums / lengths.float().unsqueeze(1)

                centered_sums = torch.zeros(
                    (local_count, centered.shape[1]),
                    device=config.device,
                    dtype=torch.float32,
                )
                centered_sums.index_add_(0, sample_index, centered.float())
                pooled_centered = (
                    centered_sums / lengths.float().unsqueeze(1)
                ).to(dtype=state["encoder.weight"].dtype)
                pooled_batch = torch.relu(
                    F.linear(
                        pooled_centered,
                        state["encoder.weight"],
                        state["encoder.bias"],
                    )
                ).float()

                standard_output[start:end] = standard_batch.cpu()
                pooled_output[start:end] = pooled_batch.cpu()
                del (
                    acts,
                    mask,
                    local_ids,
                    flat_mask,
                    x,
                    sample_index,
                    x_norm,
                    centered,
                    token_features,
                    token_sums,
                    standard_batch,
                    centered_sums,
                    pooled_centered,
                    pooled_batch,
                )
            standard[class_name] = standard_output
            pooled_first[class_name] = pooled_output
    return {
        "mean_after_encode": standard,
        "encode_after_mean": pooled_first,
    }


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

    started = time.time()
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
            llm, tokenizer, train_data, config, [22]
        )
        test_layers, test_masks = module.collect_layer_activations(
            llm, tokenizer, test_data, config, [22]
        )
        cached[dataset] = {
            "train": compute_relu_pooling_orders(
                module, train_layers[22], train_masks, state, config
            ),
            "test": compute_relu_pooling_orders(
                module, test_layers[22], test_masks, state, config
            ),
        }
        del train_layers, test_layers, train_masks, test_masks
    del llm, state
    torch.cuda.empty_cache()
    gc.collect()

    variants = [
        ("mean_after_encode", "standard mean-of-token ReLU"),
        ("encode_after_mean", "turn-pooled ReLU"),
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
            print(
                f"   {dataset}: "
                + " ".join(
                    f"k{k}={probe['metrics'][f'sae_top_{k}_test_accuracy']:.4f}"
                    for k in args.k_values
                ),
                flush=True,
            )
        rows.append(row)

    summary = summarize(rows, args.k_values)
    summary_by_key = {row["variant_key"]: row for row in summary}
    rows_by_key = {row["variant_key"]: row for row in rows}
    candidate = summary_by_key["encode_after_mean"]
    reference = summary_by_key["mean_after_encode"]
    dataset_deltas = {
        dataset: {
            "reference": dataset_mean(
                rows_by_key["mean_after_encode"], dataset, args.k_values
            ),
            "candidate": dataset_mean(
                rows_by_key["encode_after_mean"], dataset, args.k_values
            ),
        }
        for dataset in args.datasets
    }
    for values in dataset_deltas.values():
        values["delta"] = values["candidate"] - values["reference"]
    gate = {
        "candidate_minus_reference_at_least_0p005": (
            candidate["mean_acc"] - reference["mean_acc"] >= 0.005
        ),
        "no_dataset_drop_below_minus_0p01": all(
            values["delta"] >= -0.01 for values in dataset_deltas.values()
        ),
    }
    gate["pass"] = all(gate.values())
    manifest_path = args.cache_dir / "manifest.json"
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "cache_dir": str(args.cache_dir),
            "cache_manifest_sha256": sha256(manifest_path),
            "layer": 22,
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "mechanism": {
            "reference": "mean_t ReLU(W_enc(normalize(x_t)-b_pre)+b_enc)",
            "candidate": "ReLU(W_enc(mean_t(normalize(x_t)-b_pre))+b_enc)",
            "same_checkpoint": True,
            "same_parameter_count": True,
            "same_exposed_feature_count": True,
            "pooling_rule_sweep": False,
        },
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-turn-averaged-sae-v1-screen"
            if gate["pass"]
            else "stop-before-turn-averaged-sae-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "turn-pooled-relu-gate.json"
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Structured turn-pooled ReLU gate",
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
    (args.output_dir / "turn-pooled-relu-gate.md").write_text(
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
