#!/usr/bin/env python3
"""Evaluate activation-only students of unlabeled document-gradient atoms."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


HEAD_NAMES = (
    "gradient_atom_student",
    "wrong_alignment_student",
    "gradient_pca_student",
    "activation_pca_control",
    "random_orthogonal_control",
)


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("gradient_atom_sparse_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_v396(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    keys = (
        "b_pre",
        "encoder.weight",
        "encoder.bias",
        "v396.raw_beta",
        "v396.log_gain",
        "v396.max_beta",
        "v396.max_log_gain",
    )
    missing = sorted(set(keys) - set(raw))
    if missing:
        raise KeyError(f"V396 checkpoint missing {missing}")
    return {key: raw[key].float().to(device) for key in keys}


def v396_features(
    normalized: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    centered = normalized.float() - state["b_pre"]
    preactivation = F.linear(
        centered,
        state["encoder.weight"],
        state["encoder.bias"],
    )
    positive = torch.relu(preactivation)
    beta = F.softplus(state["v396.raw_beta"]).clamp(
        1.0e-4,
        float(state["v396.max_beta"].item()),
    )
    features = torch.log1p(positive * beta) / torch.log1p(beta)
    gain = state["v396.log_gain"].clamp(
        -float(state["v396.max_log_gain"].item()),
        float(state["v396.max_log_gain"].item()),
    ).exp()
    return features * gain


def sparse_head(
    sample_activation: torch.Tensor,
    artifact: dict[str, torch.Tensor],
    name: str,
    top_k: int,
) -> torch.Tensor:
    standardized = (
        sample_activation.float() - artifact["activation_mean"]
    ) / artifact["activation_std"].clamp_min(1.0e-6)
    prediction = F.linear(
        standardized,
        artifact[f"heads.{name}.weight"].T,
        artifact[f"heads.{name}.bias"],
    )
    prediction = (
        prediction - artifact[f"heads.{name}.mean"]
    ) / artifact[f"heads.{name}.std"].clamp_min(1.0e-6)
    indices = torch.topk(prediction.abs(), k=top_k, dim=1).indices
    mask = torch.zeros_like(prediction, dtype=torch.bool).scatter_(1, indices, True)
    return prediction * mask


def compute_representations(
    module: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    artifact: dict[str, torch.Tensor],
    config: Any,
    seq_batch_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    latent_count = int(state["encoder.weight"].shape[0])
    slot_indices = artifact["slot_indices"].long()
    slot_mean = artifact["slot_mean"].float()
    slot_std = artifact["slot_std"].float()
    top_k = int(artifact["head_top_k"].item())
    result: dict[str, dict[str, torch.Tensor]] = {
        "v396_reference": {},
        **{f"replace_{name}": {} for name in HEAD_NAMES},
        **{f"head_only_{name}": {} for name in HEAD_NAMES},
    }
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            representation_chunks: dict[str, list[torch.Tensor]] = {
                key: [] for key in result
            }
            for start in range(0, int(acts_cpu.shape[0]), seq_batch_size):
                end = min(int(acts_cpu.shape[0]), start + seq_batch_size)
                acts = acts_cpu[start:end].to(config.device, non_blocking=True)
                mask = mask_cpu[start:end].to(config.device, non_blocking=True)
                flat = acts.reshape(-1, acts.shape[-1])
                flat_mask = mask.reshape(-1)
                normalized_flat = module.normalize_activation(
                    flat[flat_mask],
                    torch.float32,
                    config.normalize_eps,
                ).float()
                sample_ids = (
                    torch.arange(end - start, device=config.device)
                    .unsqueeze(1)
                    .expand_as(mask)
                    .reshape(-1)[flat_mask]
                )
                counts = mask.sum(dim=1).float().clamp_min(1.0)
                base_sum = torch.zeros(
                    (end - start, latent_count),
                    device=config.device,
                    dtype=torch.float32,
                )
                activation_sum = torch.zeros(
                    (end - start, acts.shape[-1]),
                    device=config.device,
                    dtype=torch.float32,
                )
                base_sum.index_add_(
                    0,
                    sample_ids,
                    v396_features(normalized_flat, state),
                )
                activation_sum.index_add_(0, sample_ids, normalized_flat)
                base_mean = base_sum / counts.unsqueeze(1)
                activation_mean = activation_sum / counts.unsqueeze(1)
                representation_chunks["v396_reference"].append(base_mean.cpu())
                for name in HEAD_NAMES:
                    head = sparse_head(activation_mean, artifact, name, top_k)
                    calibrated = head * slot_std + slot_mean
                    replaced = base_mean.clone()
                    replaced[:, slot_indices] = calibrated
                    representation_chunks[f"replace_{name}"].append(replaced.cpu())
                    representation_chunks[f"head_only_{name}"].append(head.cpu())
                    del head, calibrated, replaced
                del (
                    acts,
                    mask,
                    flat,
                    flat_mask,
                    normalized_flat,
                    sample_ids,
                    counts,
                    base_sum,
                    activation_sum,
                    base_mean,
                    activation_mean,
                )
                torch.cuda.empty_cache()
            for key, chunks in representation_chunks.items():
                result[key][class_name] = torch.cat(chunks, dim=0)
    return result


def probe_dataset(
    module: Any,
    train_acts: dict[str, torch.Tensor],
    test_acts: dict[str, torch.Tensor],
    k_values: list[int],
    seed: int,
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
            metrics["selected_features"] = [int(index) for index in selected.tolist()]
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
    rows = []
    for name in names:
        aggregates = [
            dataset["representations"][name]["aggregate"]
            for dataset in payload["datasets"].values()
        ]
        rows.append(
            {
                "representation": name,
                "mean_acc": sum(row["mean_acc"] for row in aggregates) / len(aggregates),
                "mean_auc": sum(row["mean_auc"] for row in aggregates) / len(aggregates),
                **{
                    f"top_{k}_acc": sum(
                        row[f"top_{k}_test_accuracy"] for row in aggregates
                    )
                    / len(aggregates)
                    for k in (1, 2, 5)
                },
            }
        )
    return sorted(rows, key=lambda row: row["mean_acc"], reverse=True)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Gradient-atom activation-only frozen signal gate",
        "",
        "| Representation | Mean Acc | Mean AUC | Top-1 | Top-2 | Top-5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['representation']} | {row['mean_acc']:.6f} | "
            f"{row['mean_auc']:.6f} | {row['top_1_acc']:.6f} | "
            f"{row['top_2_acc']:.6f} | {row['top_5_acc']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--v396-checkpoint", type=Path, required=True)
    parser.add_argument("--head-artifact", type=Path, required=True)
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
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

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
    model = module.load_model(args.model_dir, [22], device, dtype)
    tokenizer = module.Tokenizer(str(args.model_dir / "tokenizer.model"))
    state = load_v396(args.v396_checkpoint, device)
    raw_artifact = torch.load(args.head_artifact, map_location="cpu", weights_only=True)
    artifact = {
        key: value.float().to(device) if value.dtype.is_floating_point else value.to(device)
        for key, value in raw_artifact.items()
    }
    started = time.time()
    payload: dict[str, Any] = {
        "config": {
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "context_length": args.context_length,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
            "v396_checkpoint": str(args.v396_checkpoint),
            "head_artifact": str(args.head_artifact),
            "evaluation_requires_gradient": False,
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
            model, tokenizer, train_data, config, [22]
        )
        test_layers, test_masks = module.collect_layer_activations(
            model, tokenizer, test_data, config, [22]
        )
        train_representations = compute_representations(
            module,
            train_layers[22],
            train_masks,
            state,
            artifact,
            config,
            args.seq_batch_size,
        )
        test_representations = compute_representations(
            module,
            test_layers[22],
            test_masks,
            state,
            artifact,
            config,
            args.seq_batch_size,
        )
        seed = args.random_seed + 1009 * dataset_index
        dataset_results = {}
        for name in train_representations:
            dataset_results[name] = probe_dataset(
                module,
                train_representations[name],
                test_representations[name],
                args.k_values,
                seed,
            )
        payload["datasets"][dataset_name] = {"representations": dataset_results}
        del (
            train_layers,
            test_layers,
            train_masks,
            test_masks,
            train_representations,
            test_representations,
        )
        gc.collect()
        torch.cuda.empty_cache()
    payload["summary"] = summarize(payload)
    payload["elapsed_seconds"] = time.time() - started
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
