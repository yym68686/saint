#!/usr/bin/env python3
"""Sparse-probing evaluator for the exact-parameter Cascaded Concept SAE.

The L22 activations for every dataset are collected once and retained in CPU
memory. Each checkpoint is then loaded once, evaluated across all datasets, and
released. This avoids repeating LLM inference and repeated multi-gigabyte
checkpoint reads for every dataset.
"""

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


CUSTOM_KINDS = {"v396_finetune", "cascaded_concept"}


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("saebench_sparse_probe_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def patch_custom_kinds(module: Any) -> None:
    original_load = module.load_sae_state
    original_encode = module.encode_features_for_tokens

    def load_sae_state(target: dict[str, Any], config: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        kind = str(target["kind"])
        if kind not in CUSTOM_KINDS:
            return original_load(target, config)
        checkpoint = Path(target["checkpoint"])
        raw = module.load_state(checkpoint)
        if kind == "v396_finetune":
            n_latents = int(raw["decoder.weight"].shape[1])
            keys = [
                "b_pre",
                "encoder.weight",
                "encoder.bias",
                "v396.raw_beta",
                "v396.log_gain",
            ]
        else:
            n_latents = int(raw["cascaded.n_total"].item())
            keys = [
                "b_pre",
                "level1.encoder.weight",
                "level1.encoder.bias",
                "level1.raw_beta",
                "level1.log_gain",
                "cascaded.cluster_scale",
                "cascaded.wrong_cluster_scale",
            ]
        state = module.move_keys(raw, keys, config.device, config.dtype)
        if kind == "cascaded_concept":
            state["cascaded.parent"] = raw["cascaded.parent"].long().to(config.device)
            state["cascaded.wrong_parent"] = raw[
                "cascaded.wrong_parent"
            ].long().to(config.device)
        extra = {
            "n_latents": n_latents,
            "parameter_count": int(target.get("trainable_parameters", sum(v.numel() for v in raw.values()))),
            "max_beta": float(raw["v396.max_beta"].item()),
            "max_log_gain": float(raw["v396.max_log_gain"].item()),
        }
        del raw
        return state, extra

    def encode_features_for_tokens(
        x_flat: torch.Tensor,
        target: dict[str, Any],
        state: dict[str, torch.Tensor],
        extra: dict[str, Any],
        config: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        kind = str(target["kind"])
        if kind not in CUSTOM_KINDS:
            return original_encode(x_flat, target, state, extra, config)
        x_norm = module.normalize_activation(
            x_flat, config.dtype, config.normalize_eps
        ).float()
        centered = x_norm - state["b_pre"].float()
        if kind == "v396_finetune":
            preactivation = F.linear(
                centered,
                state["encoder.weight"].float(),
                state["encoder.bias"].float(),
            )
            raw_beta = state["v396.raw_beta"].float()
            log_gain = state["v396.log_gain"].float()
        else:
            preactivation = F.linear(
                centered,
                state["level1.encoder.weight"].float(),
                state["level1.encoder.bias"].float(),
            )
            raw_beta = state["level1.raw_beta"].float()
            log_gain = state["level1.log_gain"].float()
        positive = torch.relu(preactivation)
        beta = F.softplus(raw_beta).clamp(
            1.0e-4,
            float(extra["max_beta"]),
        )
        gain = log_gain.clamp(
            -float(extra["max_log_gain"]),
            float(extra["max_log_gain"]),
        ).exp()
        low = (
            torch.log1p(beta.unsqueeze(0) * positive)
            / torch.log1p(beta).unsqueeze(0)
            * gain.unsqueeze(0)
        )
        if kind == "v396_finetune":
            return None, None, low

        readout = str(target.get("readout", "learned_hierarchy"))
        n_high = extra["n_latents"] - int(low.shape[1])
        high = torch.zeros(
            (len(low), n_high), device=low.device, dtype=low.dtype
        )
        if readout != "level1_only":
            parent_key = (
                "cascaded.wrong_parent"
                if readout == "wrong_hierarchy"
                else "cascaded.parent"
            )
            parent = state[parent_key]
            scale_key = (
                "cascaded.wrong_cluster_scale"
                if readout == "wrong_hierarchy"
                else "cascaded.cluster_scale"
            )
            scale = state[scale_key].float().index_select(
                0, parent
            )
            high.scatter_add_(
                1,
                parent.unsqueeze(0).expand(len(low), -1),
                low * scale.unsqueeze(0),
            )
        return None, None, torch.cat([low, high], dim=1)

    module.load_sae_state = load_sae_state
    module.encode_features_for_tokens = encode_features_for_tokens


def compute_mean_activations_with_state(
    module: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    target: dict[str, Any],
    state: dict[str, torch.Tensor],
    extra: dict[str, Any],
    config: Any,
) -> dict[str, torch.Tensor]:
    n_latents = int(extra["n_latents"])
    output: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            num_samples = acts_cpu.shape[0]
            sums_cpu = torch.zeros((num_samples, n_latents), dtype=torch.float32)
            counts_cpu = mask_cpu.sum(dim=1).clamp_min(1).to(torch.float32)
            for seq_start in range(0, num_samples, config.sae_seq_batch_size):
                seq_end = min(num_samples, seq_start + config.sae_seq_batch_size)
                act_chunk = acts_cpu[seq_start:seq_end].to(config.device, non_blocking=True)
                mask_chunk = mask_cpu[seq_start:seq_end].to(config.device, non_blocking=True)
                local_sample_ids = (
                    torch.arange(seq_end - seq_start, device=config.device)
                    .unsqueeze(1)
                    .expand_as(mask_chunk)
                    .reshape(-1)
                )
                flat_mask = mask_chunk.reshape(-1)
                if flat_mask.sum().item() == 0:
                    continue
                x_flat = act_chunk.reshape(-1, act_chunk.shape[-1])[flat_mask]
                sample_ids = local_sample_ids[flat_mask]
                token_values, token_indices, dense_features = module.encode_features_for_tokens(
                    x_flat,
                    target,
                    state,
                    extra,
                    config,
                )
                sums_gpu = torch.zeros(
                    (seq_end - seq_start, n_latents),
                    device=config.device,
                    dtype=torch.float32,
                )
                if dense_features is not None:
                    sums_gpu.index_add_(0, sample_ids, dense_features.float())
                else:
                    sample_ids_2d = sample_ids.unsqueeze(1).expand_as(token_indices)
                    sums_gpu.index_put_(
                        (sample_ids_2d.reshape(-1), token_indices.reshape(-1)),
                        token_values.reshape(-1),
                        accumulate=True,
                    )
                sums_cpu[seq_start:seq_end] = sums_gpu.cpu()
                del act_chunk, mask_chunk, local_sample_ids, flat_mask, x_flat, sample_ids, sums_gpu
                if config.device.type == "cuda":
                    torch.cuda.empty_cache()
            output[class_name] = sums_cpu / counts_cpu.unsqueeze(1)
    return output


def summarize(rows: list[dict[str, Any]], k_values: list[int]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] != "ok":
            continue
        aggregate: dict[str, float] = {}
        for k in k_values:
            for metric in ("test_accuracy", "test_auc"):
                key = f"sae_top_{k}_{metric}"
                aggregate[key] = sum(
                    dataset[key] for dataset in row["dataset_results"].values()
                ) / len(row["dataset_results"])
        row["aggregate"] = aggregate
        summary.append({
            "label": row["label"],
            "variant_key": row.get("variant_key"),
            "seed": row.get("seed"),
            "datasets": len(row["dataset_results"]),
            "mean_acc": sum(aggregate[f"sae_top_{k}_test_accuracy"] for k in k_values) / len(k_values),
            "mean_auc": sum(aggregate[f"sae_top_{k}_test_auc"] for k in k_values) / len(k_values),
            **{f"top_{k}_acc": aggregate[f"sae_top_{k}_test_accuracy"] for k in k_values},
            **{f"top_{k}_auc": aggregate[f"sae_top_{k}_test_auc"] for k in k_values},
        })
    summary.sort(key=lambda item: item["mean_acc"], reverse=True)
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    k_values = payload["config"]["k_values"]
    lines = [
        "# Exact-Parameter Cascaded Concept SAE Sparse Probing",
        "",
        "| Variant | Seed | Mean Acc ↑ | Mean AUC ↑ | "
        + " | ".join(f"Top-{k} Acc ↑" for k in k_values)
        + " |",
        "|---|---:|---:|---:|" + "|".join("---:" for _ in k_values) + "|",
    ]
    for row in payload["summary"]:
        topk = " | ".join(f"{row[f'top_{k}_acc']:.6f}" for k in k_values)
        lines.append(
            f"| {row['variant_key']} | {row['seed']} | {row['mean_acc']:.6f} | "
            f"{row['mean_auc']:.6f} | {topk} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--model-dir", type=Path, default=Path("/root/saint/llama_3.2-3B_model/original"))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--sae-seq-batch-size", type=int, default=2)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.time()
    module = load_eval_module(args.eval_script)
    patch_custom_kinds(module)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config_kwargs = {
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
    config = module.EvalConfig(**{key: value for key, value in config_kwargs.items() if key in allowed})
    targets = json.loads(args.targets_json.read_text(encoding="utf-8"))
    layers = sorted({int(target["layer"]) for target in targets})

    tokenizer = module.Tokenizer(str(args.model_dir / "tokenizer.model"))
    model = module.load_model(args.model_dir, layers, config.device, config.dtype)
    dataset_cache: dict[str, dict[str, Any]] = {}
    for dataset_index, dataset_name in enumerate(args.datasets):
        print(f"== Cache dataset {dataset_index + 1}/{len(args.datasets)}: {dataset_name}", flush=True)
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
            layers,
        )
        test_layers, test_masks = module.collect_layer_activations(
            model,
            tokenizer,
            test_data,
            config,
            layers,
        )
        dataset_cache[dataset_name] = {
            "train_layers": train_layers,
            "train_masks": train_masks,
            "test_layers": test_layers,
            "test_masks": test_masks,
            "dataset_index": dataset_index,
        }
    del model
    if config.device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    architecture_results: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        label = target["label"]
        print(f"== Target {target_index + 1}/{len(targets)}: {label}", flush=True)
        row = {
            **target,
            "dataset_results": {},
            "per_class": {},
            "status": "ok",
            "seconds": 0.0,
        }
        target_started = time.time()
        try:
            state, extra = module.load_sae_state(target, config)
            for dataset_name, cached in dataset_cache.items():
                layer = int(target["layer"])
                train_features = compute_mean_activations_with_state(
                    module,
                    cached["train_layers"][layer],
                    cached["train_masks"],
                    target,
                    state,
                    extra,
                    config,
                )
                test_features = compute_mean_activations_with_state(
                    module,
                    cached["test_layers"][layer],
                    cached["test_masks"],
                    target,
                    state,
                    extra,
                    config,
                )
                probe_seed = args.random_seed + 1009 * int(cached["dataset_index"])
                result = module.probe_one_architecture_dataset(
                    train_features,
                    test_features,
                    args.k_values,
                    probe_seed,
                )
                row["dataset_results"][dataset_name] = result["metrics"]
                row["per_class"][dataset_name] = result["per_class"]
                print(
                    f"   {dataset_name}: "
                    + " ".join(
                        f"k{k}={result['metrics'][f'sae_top_{k}_test_accuracy']:.4f}"
                        for k in args.k_values
                    ),
                    flush=True,
                )
                del train_features, test_features
            del state
        except Exception as error:  # noqa: BLE001
            row["status"] = f"failed: {type(error).__name__}: {error}"
            row["error"] = repr(error)
            print(f"FAILED {row['status']}", flush=True)
        row["seconds"] = time.time() - target_started
        architecture_results.append(row)
        if config.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config": {
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "context_length": args.context_length,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
            "dataset_activations_cached_once": True,
            "checkpoint_loaded_once": True,
        },
        "summary": summarize(architecture_results, args.k_values),
        "architecture_results": architecture_results,
        "elapsed_sec": time.time() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
