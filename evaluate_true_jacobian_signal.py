#!/usr/bin/env python3
"""Evaluate a frozen exact J-lens against logit, ReLU, and random controls."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("true_jacobian_sparse_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mean_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=torch.float32).unsqueeze(-1)
    return (values.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def load_average(per_prompt_dir: Path, count: int) -> torch.Tensor:
    total = None
    for index in range(count):
        payload = torch.load(
            per_prompt_dir / f"prompt-{index:02d}-jacobian.pt",
            map_location="cpu",
            weights_only=True,
        )
        jacobian = payload["jacobian"].float()
        total = jacobian.double() if total is None else total + jacobian.double()
    if total is None:
        raise ValueError("No prompt Jacobians loaded")
    return (total / count).float()


def compute_vocab_features(
    model: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    config: Any,
    seq_batch_size: int,
    transform_kind: str,
    transform: torch.Tensor | None,
    permutation: torch.Tensor | None,
    signs: torch.Tensor | None,
    random_scale: float,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            chunks = []
            for start in range(0, int(acts_cpu.shape[0]), seq_batch_size):
                end = min(int(acts_cpu.shape[0]), start + seq_batch_size)
                acts = acts_cpu[start:end].to(
                    device=config.device,
                    dtype=config.dtype,
                    non_blocking=True,
                )
                mask = mask_cpu[start:end].to(config.device, non_blocking=True)
                if transform_kind == "identity":
                    transformed = acts
                elif transform_kind == "matrix":
                    if transform is None:
                        raise ValueError("Matrix transform is missing")
                    transformed = torch.nn.functional.linear(acts, transform)
                elif transform_kind == "signed_permutation":
                    if permutation is None or signs is None:
                        raise ValueError("Signed-permutation control is missing")
                    transformed = acts[..., permutation] * signs * random_scale
                else:
                    raise ValueError(transform_kind)
                logits = model.output(model.norm(transformed)).float()
                chunks.append(mean_pool(logits, mask).cpu())
                del acts, mask, transformed, logits
            result[class_name] = torch.cat(chunks, dim=0)
    return result


def decode_ids(tokenizer: Any, ids: list[int]) -> list[str]:
    decoded = []
    for token_id in ids:
        try:
            decoded.append(tokenizer.decode([token_id]))
        except Exception:
            decoded.append(str(token_id))
    return decoded


def probe_dataset(
    module: Any,
    train_acts: dict[str, torch.Tensor],
    test_acts: dict[str, torch.Tensor],
    k_values: list[int],
    seed: int,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    per_class = {}
    for class_index, class_name in enumerate(train_acts):
        train_x, train_y = module.prepare_probe_data(
            train_acts,
            class_name,
            seed + 17 * class_index,
        )
        test_x, test_y = module.prepare_probe_data(
            test_acts,
            class_name,
            seed + 29 * class_index,
        )
        class_metrics = {}
        for k in k_values:
            selected = module.select_topk_mean_diff(train_x, train_y, k)
            metrics = module.train_probe(
                train_x[:, selected],
                train_y,
                test_x[:, selected],
                test_y,
                seed + k + 101 * class_index,
            )
            selected_ids = [int(index) for index in selected.tolist()]
            metrics["selected_features"] = selected_ids
            if tokenizer is not None:
                metrics["selected_tokens"] = decode_ids(tokenizer, selected_ids)
            class_metrics[f"top_{k}"] = metrics
        per_class[class_name] = class_metrics
    aggregate = {}
    for k in k_values:
        for metric in ("test_accuracy", "test_auc"):
            aggregate[f"top_{k}_{metric}"] = sum(
                per_class[name][f"top_{k}"][metric] for name in per_class
            ) / len(per_class)
    aggregate["mean_acc"] = sum(
        aggregate[f"top_{k}_test_accuracy"] for k in k_values
    ) / len(k_values)
    aggregate["mean_auc"] = sum(
        aggregate[f"top_{k}_test_auc"] for k in k_values
    ) / len(k_values)
    return {"aggregate": aggregate, "per_class": per_class}


def summarize(payload: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(next(iter(payload["datasets"].values()))["representations"])
    summary = []
    for name in names:
        rows = [
            dataset["representations"][name]["aggregate"]
            for dataset in payload["datasets"].values()
        ]
        summary.append(
            {
                "representation": name,
                "datasets": len(rows),
                "mean_acc": sum(row["mean_acc"] for row in rows) / len(rows),
                "mean_auc": sum(row["mean_auc"] for row in rows) / len(rows),
                **{
                    f"top_{k}_acc": sum(
                        row[f"top_{k}_test_accuracy"] for row in rows
                    )
                    / len(rows)
                    for k in (1, 2, 5)
                },
            }
        )
    return sorted(summary, key=lambda row: row["mean_acc"], reverse=True)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# True averaged Jacobian lens frozen signal gate",
        "",
        "| Representation | Mean Acc ↑ | Mean AUC ↑ | Top-1 ↑ | Top-2 ↑ | Top-5 ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['representation']} | {row['mean_acc']:.6f} | "
            f"{row['mean_auc']:.6f} | {row['top_1_acc']:.6f} | "
            f"{row['top_2_acc']:.6f} | {row['top_5_acc']:.6f} |"
        )
    for dataset_name, dataset in payload["datasets"].items():
        lines.extend(
            [
                "",
                f"## {dataset_name}",
                "",
                "| Representation | Mean Acc ↑ | Mean AUC ↑ |",
                "|---|---:|---:|",
            ]
        )
        for name, result in dataset["representations"].items():
            aggregate = result["aggregate"]
            lines.append(
                f"| {name} | {aggregate['mean_acc']:.6f} | "
                f"{aggregate['mean_auc']:.6f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--relu-checkpoint", type=Path, required=True)
    parser.add_argument("--jacobian-checkpoint", type=Path, required=True)
    parser.add_argument("--per-prompt-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--seq-batch-size", type=int, default=1)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-control-seed", type=int, default=42026)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    module = load_eval_module(args.eval_script)
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    config = module.EvalConfig(
        model_dir=args.model_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        context_length=args.context_length,
        llm_batch_size=args.llm_batch_size,
        sae_seq_batch_size=args.seq_batch_size,
        k_values=args.k_values,
        random_seed=args.random_seed,
        device=device,
        dtype=dtype,
        dtype_name=args.dtype,
        normalize_eps=1.0e-6,
    )
    started = time.time()
    model = module.load_model(args.model_dir, [22], device, dtype)
    tokenizer = module.Tokenizer(str(args.model_dir / "tokenizer.model"))
    n10_payload = torch.load(
        args.jacobian_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    j_n10_cpu = n10_payload["jacobian"].float()
    j_n5_cpu = load_average(args.per_prompt_dir, 5)
    hidden_size = int(j_n10_cpu.shape[0])
    if tuple(j_n10_cpu.shape) != (hidden_size, hidden_size):
        raise ValueError(f"Jacobian is not square: {tuple(j_n10_cpu.shape)}")
    j_n10 = j_n10_cpu.to(device=device, dtype=dtype)
    j_n5 = j_n5_cpu.to(device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.random_control_seed)
    permutation = torch.randperm(hidden_size, generator=generator).to(device)
    signs = (
        torch.randint(0, 2, (hidden_size,), generator=generator) * 2 - 1
    ).to(device=device, dtype=dtype)
    random_scale = float(j_n10_cpu.norm().item() / math.sqrt(hidden_size))

    representations = [
        ("logit_lens", "identity", None),
        ("true_jacobian_lens_n5", "matrix", j_n5),
        ("true_jacobian_lens_n10", "matrix", j_n10),
        ("random_orthogonal_control", "signed_permutation", None),
    ]
    relu_target = {
        "label": "structured-cache ReLU control",
        "kind": "relu",
        "layer": 22,
        "checkpoint": str(args.relu_checkpoint),
    }
    payload: dict[str, Any] = {
        "config": {
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "context_length": args.context_length,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
            "random_control_seed": args.random_control_seed,
            "random_control_scale": random_scale,
            "jacobian_checkpoint": str(args.jacobian_checkpoint),
            "relu_checkpoint": str(args.relu_checkpoint),
        },
        "datasets": {},
    }
    for dataset_index, dataset_name in enumerate(args.datasets):
        print(f"Dataset {dataset_name}", flush=True)
        train_data, test_data = module.get_multi_label_train_test_data(
            dataset_name,
            args.train_size,
            args.test_size,
            args.random_seed,
        )
        train_layers, train_masks = module.collect_layer_activations(
            model,
            tokenizer,
            train_data,
            config,
            [22],
        )
        test_layers, test_masks = module.collect_layer_activations(
            model,
            tokenizer,
            test_data,
            config,
            [22],
        )
        seed = args.random_seed + 1009 * dataset_index
        dataset_results = {}
        train_relu = module.compute_sae_mean_activations(
            train_layers[22],
            train_masks,
            relu_target,
            config,
        )
        test_relu = module.compute_sae_mean_activations(
            test_layers[22],
            test_masks,
            relu_target,
            config,
        )
        dataset_results["relu_control"] = probe_dataset(
            module,
            train_relu,
            test_relu,
            args.k_values,
            seed,
        )
        del train_relu, test_relu
        for name, kind, transform in representations:
            train_features = compute_vocab_features(
                model,
                train_layers[22],
                train_masks,
                config,
                args.seq_batch_size,
                kind,
                transform,
                permutation,
                signs,
                random_scale,
            )
            test_features = compute_vocab_features(
                model,
                test_layers[22],
                test_masks,
                config,
                args.seq_batch_size,
                kind,
                transform,
                permutation,
                signs,
                random_scale,
            )
            dataset_results[name] = probe_dataset(
                module,
                train_features,
                test_features,
                args.k_values,
                seed,
                tokenizer,
            )
            del train_features, test_features
            gc.collect()
            torch.cuda.empty_cache()
        payload["datasets"][dataset_name] = {"representations": dataset_results}
        del train_layers, test_layers, train_masks, test_masks
        gc.collect()
        torch.cuda.empty_cache()
    payload["summary"] = summarize(payload)
    payload["elapsed_seconds"] = time.time() - started
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_md, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
