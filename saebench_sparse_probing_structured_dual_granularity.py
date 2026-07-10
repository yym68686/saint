#!/usr/bin/env python3
"""Evaluate the structured ReLU control and dual-granularity SAE on Initial3."""

from __future__ import annotations

import argparse
import gc
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
    spec = importlib.util.spec_from_file_location("structured_sparse_probe_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_target_state(
    module: Any,
    target: dict[str, Any],
    config: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, int | float]]:
    raw = module.load_state(Path(target["checkpoint"]))
    if target["kind"] == "relu":
        keys = ["b_pre", "encoder.weight", "encoder.bias"]
        state = module.move_keys(raw, keys, config.device, config.dtype)
        extra = {
            "n_total": int(raw["encoder.weight"].shape[0]),
            "n_token": int(raw["encoder.weight"].shape[0]),
            "n_semantic": 0,
        }
    elif target["kind"] in {
        "structured_dual_granularity",
        "structured_dual_granularity_softplus",
        "structured_dual_granularity_responsibility_split",
    }:
        keys = [
            "b_pre",
            "token_encoder.weight",
            "token_encoder.bias",
            "semantic_encoder.weight",
            "semantic_encoder.bias",
        ]
        state = module.move_keys(raw, keys, config.device, config.dtype)
        extra = {
            "n_total": int(raw["structured.n_total"].item()),
            "n_token": int(raw["structured.n_token"].item()),
            "n_semantic": int(raw["structured.n_semantic"].item()),
            "semantic_temperature": float(
                raw.get(
                    "structured.semantic_temperature",
                    torch.tensor(0.0),
                ).item()
            ),
        }
    else:
        raise ValueError(target["kind"])
    del raw
    return state, extra


def normalize(module: Any, x: torch.Tensor, config: Any) -> torch.Tensor:
    return module.normalize_activation(x, config.dtype, config.normalize_eps)


def mean_features(
    module: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    target: dict[str, Any],
    state: dict[str, torch.Tensor],
    extra: dict[str, int | float],
    config: Any,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            sample_count = int(acts_cpu.shape[0])
            output = torch.zeros(
                (sample_count, extra["n_total"]),
                dtype=torch.float32,
            )
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
                x_norm = normalize(module, x, config)
                centered = x_norm - state["b_pre"]

                if target["kind"] == "relu":
                    token = torch.relu(
                        F.linear(
                            centered,
                            state["encoder.weight"],
                            state["encoder.bias"],
                        )
                    )
                    sums = torch.zeros(
                        (local_count, extra["n_total"]),
                        device=config.device,
                        dtype=torch.float32,
                    )
                    sums.index_add_(0, sample_index, token.float())
                    features = sums / lengths.float().unsqueeze(1)
                else:
                    pooled = torch.zeros(
                        (local_count, centered.shape[1]),
                        device=config.device,
                        dtype=centered.dtype,
                    )
                    pooled.index_add_(0, sample_index, centered)
                    pooled = pooled / lengths.to(centered.dtype).unsqueeze(1)
                    token_input = centered
                    if (
                        target["kind"]
                        == "structured_dual_granularity_responsibility_split"
                    ):
                        token_input = centered - pooled[sample_index]
                    token = torch.relu(
                        F.linear(
                            token_input,
                            state["token_encoder.weight"],
                            state["token_encoder.bias"],
                        )
                    )
                    token_sums = torch.zeros(
                        (local_count, extra["n_token"]),
                        device=config.device,
                        dtype=torch.float32,
                    )
                    token_sums.index_add_(0, sample_index, token.float())
                    token_mean = token_sums / lengths.float().unsqueeze(1)
                    semantic_hidden = F.linear(
                        pooled,
                        state["semantic_encoder.weight"],
                        state["semantic_encoder.bias"],
                    )
                    if target["kind"] in {
                        "structured_dual_granularity_softplus",
                        "structured_dual_granularity_responsibility_split",
                    }:
                        temperature = float(extra["semantic_temperature"])
                        if temperature <= 0:
                            raise ValueError(
                                "Softplus target has no positive semantic temperature"
                            )
                        semantic = (
                            F.softplus(semantic_hidden / temperature)
                            * temperature
                        ).float()
                    else:
                        semantic = torch.relu(semantic_hidden).float()
                    features = torch.cat([token_mean, semantic], dim=1)
                output[start:end] = features.cpu()
                del acts, mask, local_ids, flat_mask, x, sample_index
            result[class_name] = output
    return result


def summarize(rows: list[dict[str, Any]], k_values: list[int]) -> list[dict[str, Any]]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path("/root/autodl-tmp/saebench_sparse_probing_all_architectures.py"),
    )
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/root/saint/llama_3.2-3B_model/original"),
    )
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
    targets = json.loads(args.targets_json.read_text(encoding="utf-8"))
    layers = sorted({int(target["layer"]) for target in targets})
    tokenizer = module.Tokenizer(str(args.model_dir / "tokenizer.model"))
    llm = module.load_model(args.model_dir, layers, config.device, config.dtype)
    cache: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        print(f"== Cache dataset: {dataset}", flush=True)
        train_data, test_data = module.get_multi_label_train_test_data(
            dataset,
            args.train_size,
            args.test_size,
            args.random_seed,
        )
        train_layers, train_masks = module.collect_layer_activations(
            llm,
            tokenizer,
            train_data,
            config,
            layers,
        )
        test_layers, test_masks = module.collect_layer_activations(
            llm,
            tokenizer,
            test_data,
            config,
            layers,
        )
        cache[dataset] = {
            "train_layers": train_layers,
            "train_masks": train_masks,
            "test_layers": test_layers,
            "test_masks": test_masks,
        }
    del llm
    torch.cuda.empty_cache()
    gc.collect()

    rows = []
    for target in targets:
        started = time.time()
        print(f"== Target: {target['label']}", flush=True)
        state, extra = load_target_state(module, target, config)
        row = {
            **target,
            "dataset_results": {},
            "per_class": {},
        }
        for dataset_index, (dataset, cached) in enumerate(cache.items()):
            layer = int(target["layer"])
            train_features = mean_features(
                module,
                cached["train_layers"][layer],
                cached["train_masks"],
                target,
                state,
                extra,
                config,
            )
            test_features = mean_features(
                module,
                cached["test_layers"][layer],
                cached["test_masks"],
                target,
                state,
                extra,
                config,
            )
            probe_seed = args.random_seed + 1009 * dataset_index
            probe_result = module.probe_one_architecture_dataset(
                train_features,
                test_features,
                args.k_values,
                probe_seed,
            )
            dataset_result = probe_result["metrics"]
            per_class = probe_result["per_class"]
            row["dataset_results"][dataset] = dataset_result
            row["per_class"][dataset] = per_class
            print(
                f"   {dataset}: "
                + " ".join(
                    f"k{k}={dataset_result[f'sae_top_{k}_test_accuracy']:.4f}"
                    for k in args.k_values
                ),
                flush=True,
            )
        row["seconds"] = time.time() - started
        rows.append(row)
        del state
        torch.cuda.empty_cache()

    payload = {
        "config": {
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "context_length": args.context_length,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
        },
        "architecture_results": rows,
        "summary": summarize(rows, args.k_values),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Structured-cache dual-granularity Initial3",
        "",
        "| Variant | Mean Acc ↑ | Mean AUC ↑ | Top-1 ↑ | Top-2 ↑ | Top-5 ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['variant_key']} | {row['mean_acc']:.6f} | "
            f"{row['mean_auc']:.6f} | {row['top_1_acc']:.6f} | "
            f"{row['top_2_acc']:.6f} | {row['top_5_acc']:.6f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
