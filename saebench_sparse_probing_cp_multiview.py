#!/usr/bin/env python3
"""Evaluate token ReLU, mean-pooled ReLU, and CP multi-view SAE features."""

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
    spec = importlib.util.spec_from_file_location("cp_multiview_sparse_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def move(raw: dict[str, torch.Tensor], keys: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: raw[key].float().to(device) for key in keys}


def load_target_state(
    module: Any,
    target: dict[str, Any],
    config: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw = module.load_state(Path(target["checkpoint"]))
    kind = target["kind"]
    if kind in {"token_relu", "mean_relu"}:
        state = move(
            raw,
            ["b_pre", "encoder.weight", "encoder.bias"],
            config.device,
        )
        extra = {
            "n_latents": int(raw["encoder.weight"].shape[0]),
            "n_views": 4,
            "max_log_gain": 0.0,
        }
    elif kind == "cp_multiview":
        keys = [
            "encoder_input_basis",
            "encoder_feature_factor",
            "encoder_view_factor",
            "encoder_bias",
            "feature_gain_knots",
        ]
        state = move(raw, keys, config.device)
        extra = {
            "n_latents": int(raw["encoder_feature_factor"].shape[0]),
            "n_views": int(raw["multiview.n_views"].item()),
            "max_log_gain": float(raw["multiview.max_log_gain"].item()),
        }
    else:
        raise ValueError(kind)
    del raw
    return state, extra


def normalize(module: Any, x: torch.Tensor, config: Any) -> torch.Tensor:
    return module.normalize_activation(x, config.dtype, config.normalize_eps).float()


def pool_views(
    x: torch.Tensor,
    sample_index: torch.Tensor,
    position_index: torch.Tensor,
    sample_count: int,
    n_views: int,
) -> torch.Tensor:
    view_index = position_index % n_views
    combined = sample_index * n_views + view_index
    sums = torch.zeros(
        (sample_count * n_views, x.shape[-1]),
        device=x.device,
        dtype=x.dtype,
    )
    counts = torch.zeros(sample_count * n_views, device=x.device, dtype=x.dtype)
    sums.index_add_(0, combined, x)
    counts.index_add_(0, combined, torch.ones_like(combined, dtype=x.dtype))
    global_sums = torch.zeros(
        (sample_count, x.shape[-1]),
        device=x.device,
        dtype=x.dtype,
    )
    global_counts = torch.zeros(sample_count, device=x.device, dtype=x.dtype)
    global_sums.index_add_(0, sample_index, x)
    global_counts.index_add_(0, sample_index, torch.ones_like(sample_index, dtype=x.dtype))
    global_means = global_sums / global_counts.clamp_min(1).unsqueeze(1)
    empty = counts == 0
    if bool(empty.any().item()):
        empty_rows = torch.nonzero(empty, as_tuple=False).flatten()
        empty_samples = torch.div(empty_rows, n_views, rounding_mode="floor")
        sums[empty_rows] = global_means[empty_samples]
        counts[empty_rows] = 1
    return (sums / counts.unsqueeze(1)).reshape(sample_count, n_views, -1)


def feature_gain(
    knots: torch.Tensor,
    n_latents: int,
    max_log_gain: float,
) -> torch.Tensor:
    values = F.interpolate(
        knots.view(1, 1, -1),
        size=n_latents,
        mode="linear",
        align_corners=True,
    ).view(-1)
    return values.clamp(-max_log_gain, max_log_gain).exp()


def sample_features(
    module: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    target: dict[str, Any],
    state: dict[str, torch.Tensor],
    extra: dict[str, Any],
    config: Any,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    kind = target["kind"]
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            sample_count = int(acts_cpu.shape[0])
            output = torch.zeros(
                (sample_count, extra["n_latents"]),
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
                positions = (
                    torch.arange(mask.shape[1], device=config.device)
                    .unsqueeze(0)
                    .expand_as(mask)
                    .reshape(-1)
                )
                flat_mask = mask.reshape(-1)
                x = acts.reshape(-1, acts.shape[-1])[flat_mask]
                sample_index = local_ids[flat_mask]
                position_index = positions[flat_mask]
                x_norm = normalize(module, x, config)
                if kind == "token_relu":
                    z = torch.relu(
                        F.linear(
                            x_norm - state["b_pre"],
                            state["encoder.weight"],
                            state["encoder.bias"],
                        )
                    )
                    sums = torch.zeros(
                        (local_count, extra["n_latents"]),
                        device=config.device,
                        dtype=torch.float32,
                    )
                    sums.index_add_(0, sample_index, z.float())
                    lengths = mask.sum(dim=1).clamp_min(1).float()
                    features = sums / lengths.unsqueeze(1)
                else:
                    views = pool_views(
                        x_norm,
                        sample_index,
                        position_index,
                        local_count,
                        extra["n_views"],
                    )
                    if kind == "mean_relu":
                        pooled = views.mean(dim=1)
                        features = torch.relu(
                            F.linear(
                                pooled - state["b_pre"],
                                state["encoder.weight"],
                                state["encoder.bias"],
                            )
                        ).float()
                    else:
                        projected = torch.einsum(
                            "bvd,dr->bvr",
                            views,
                            state["encoder_input_basis"],
                        )
                        mixed = (
                            projected
                            * state["encoder_view_factor"].unsqueeze(0)
                        ).sum(dim=1)
                        pre = F.linear(
                            mixed,
                            state["encoder_feature_factor"],
                            state["encoder_bias"],
                        )
                        gain = feature_gain(
                            state["feature_gain_knots"],
                            extra["n_latents"],
                            extra["max_log_gain"],
                        )
                        features = (torch.relu(pre) * gain.unsqueeze(0)).float()
                output[start:end] = features.cpu()
                del acts, mask, local_ids, positions, flat_mask
                del x, sample_index, position_index, x_norm, features
            result[class_name] = output
    return result


def summarize(rows: list[dict[str, Any]], k_values: list[int]) -> list[dict[str, Any]]:
    summary = []
    for row in rows:
        aggregate: dict[str, float] = {}
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
            llm, tokenizer, train_data, config, layers
        )
        test_layers, test_masks = module.collect_layer_activations(
            llm, tokenizer, test_data, config, layers
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
        row = {**target, "dataset_results": {}, "per_class": {}}
        for dataset_index, (dataset, cached) in enumerate(cache.items()):
            layer = int(target["layer"])
            train_features = sample_features(
                module,
                cached["train_layers"][layer],
                cached["train_masks"],
                target,
                state,
                extra,
                config,
            )
            test_features = sample_features(
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
            metrics = probe_result["metrics"]
            row["dataset_results"][dataset] = metrics
            row["per_class"][dataset] = probe_result["per_class"]
            print(
                f"   {dataset}: "
                + " ".join(
                    f"k{k}={metrics[f'sae_top_{k}_test_accuracy']:.4f}"
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
        "# CP multi-view sparse probing",
        "",
        "| Variant | Mean Acc | Mean AUC | Top-1 | Top-2 | Top-5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['variant_key']} | {row['mean_acc']:.6f} | "
            f"{row['mean_auc']:.6f} | {row['top_1_acc']:.6f} | "
            f"{row['top_2_acc']:.6f} | {row['top_5_acc']:.6f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
