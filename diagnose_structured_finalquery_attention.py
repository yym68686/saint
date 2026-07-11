#!/usr/bin/env python3
"""Gate actual L22 final-query attention pooling before SAE training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from llama_3.model_text_only import apply_rotary_emb, repeat_kv


INITIAL3 = [
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "fancyzhx/ag_news",
]


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("finalquery_attention_eval", path)
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


def final_query_attention(
    attention: Any,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    return final_query_attention_per_head(attention, x, freqs_cis).mean(dim=1)


def final_query_attention_per_head(
    attention: Any,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, sequence, d_model], got {tuple(x.shape)}")
    batch_size, sequence_length, _ = x.shape
    xq = attention.wq(x).view(
        batch_size,
        sequence_length,
        attention.n_local_heads,
        attention.head_dim,
    )
    xk = attention.wk(x).view(
        batch_size,
        sequence_length,
        attention.n_local_kv_heads,
        attention.head_dim,
    )
    xq, xk = apply_rotary_emb(
        xq,
        xk,
        freqs_cis=freqs_cis[:sequence_length].to(x.device),
    )
    keys = repeat_kv(xk, attention.n_rep).transpose(1, 2)
    query = xq[:, -1]
    scores = torch.matmul(
        query.unsqueeze(2),
        keys.transpose(2, 3),
    ).squeeze(2) / math.sqrt(attention.head_dim)
    return F.softmax(scores.float(), dim=-1).type_as(xq)


def validate_final_query_attention(
    attention: Any,
    freqs_cis: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(seed)
    sequence_length = 7
    model_dim = int(attention.n_local_heads * attention.head_dim)
    x = torch.randn(
        (1, sequence_length, model_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    local_freqs = freqs_cis[:sequence_length].to(device)
    causal_mask = torch.triu(
        torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            device=device,
            dtype=dtype,
        ),
        diagonal=1,
    )
    native_last = attention(x, 0, local_freqs, causal_mask)[:, -1].float()
    per_head = final_query_attention_per_head(attention, x, freqs_cis)
    values = attention.wv(x).view(
        1,
        sequence_length,
        attention.n_local_kv_heads,
        attention.head_dim,
    )
    values = repeat_kv(values, attention.n_rep).transpose(1, 2)
    manual_heads = torch.matmul(per_head.unsqueeze(2), values).squeeze(2)
    manual_last = attention.wo(manual_heads.reshape(1, -1)).float()
    difference = native_last - manual_last
    max_abs_error = float(difference.abs().max().item())
    mean_abs_error = float(difference.abs().mean().item())
    cosine = float(
        F.cosine_similarity(native_last, manual_last, dim=1)[0].item()
    )
    if max_abs_error > 0.01 or cosine < 0.9999:
        raise RuntimeError(
            "Final-query attention reconstruction failed: "
            f"max_abs_error={max_abs_error:.6g}, cosine={cosine:.8f}"
        )
    return {
        "sequence_length": float(sequence_length),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "cosine_similarity": cosine,
    }


def compute_relu_attention_pooling(
    module: Any,
    layer_acts: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    attention: Any,
    freqs_cis: torch.Tensor,
    config: Any,
    control_seed: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    feature_count = int(state["encoder.weight"].shape[0])
    variants: dict[str, dict[str, torch.Tensor]] = {
        "mean_pool": {},
        "final_query_attention": {},
        "shifted_attention": {},
    }
    entropy_values = []
    max_values = []
    first_values = []
    last_values = []
    uniform_l1_values = []
    representation_cosines = []
    shifted_fixed_positions = 0
    token_count = 0
    sample_count_total = 0
    single_token_samples = 0
    with torch.inference_mode():
        for class_name, acts_cpu in layer_acts.items():
            mask_cpu = masks[class_name]
            sample_count = int(acts_cpu.shape[0])
            outputs = {
                key: torch.empty((sample_count, feature_count), dtype=torch.float32)
                for key in variants
            }
            for start in range(0, sample_count, config.sae_seq_batch_size):
                end = min(sample_count, start + config.sae_seq_batch_size)
                acts = acts_cpu[start:end].to(config.device, non_blocking=True)
                mask = mask_cpu[start:end].to(config.device, non_blocking=True)
                for local_index in range(end - start):
                    valid = mask[local_index]
                    x_attention = acts[local_index, valid]
                    length = int(x_attention.shape[0])
                    if length == 0:
                        raise ValueError("Encountered an empty evaluation sample")
                    x_sae = module.normalize_activation(
                        x_attention,
                        config.dtype,
                        config.normalize_eps,
                    )
                    token_features = torch.relu(
                        F.linear(
                            x_sae - state["b_pre"],
                            state["encoder.weight"],
                            state["encoder.bias"],
                        )
                    ).float()
                    mean_pool = token_features.mean(dim=0)
                    attention_weights = final_query_attention(
                        attention,
                        x_attention.unsqueeze(0),
                        freqs_cis,
                    )[0].float()
                    if length > 1:
                        shift = 1 + (
                            control_seed + sample_count_total
                        ) % (length - 1)
                        shifted_weights = torch.roll(attention_weights, shifts=shift)
                        shifted_fixed_positions += int(shift % length == 0) * length
                    else:
                        shifted_weights = attention_weights
                        single_token_samples += 1
                    attention_pool = torch.matmul(attention_weights, token_features)
                    shifted_pool = torch.matmul(shifted_weights, token_features)
                    row = start + local_index
                    outputs["mean_pool"][row] = mean_pool.cpu()
                    outputs["final_query_attention"][row] = attention_pool.cpu()
                    outputs["shifted_attention"][row] = shifted_pool.cpu()

                    entropy = -(
                        attention_weights
                        * attention_weights.clamp_min(1.0e-12).log()
                    ).sum()
                    entropy_values.append(
                        float(
                            (entropy / math.log(length)).item()
                            if length > 1
                            else 0.0
                        )
                    )
                    max_values.append(float(attention_weights.max().item()))
                    first_values.append(float(attention_weights[0].item()))
                    last_values.append(float(attention_weights[-1].item()))
                    uniform_l1_values.append(
                        float(
                            (
                                attention_weights
                                - torch.full_like(attention_weights, 1.0 / length)
                            )
                            .abs()
                            .sum()
                            .item()
                        )
                    )
                    representation_cosines.append(
                        float(
                            F.cosine_similarity(
                                mean_pool.unsqueeze(0),
                                attention_pool.unsqueeze(0),
                                dim=1,
                            )[0].item()
                        )
                    )
                    token_count += length
                    sample_count_total += 1
                    del (
                        x_attention,
                        x_sae,
                        token_features,
                        mean_pool,
                        attention_weights,
                        shifted_weights,
                        attention_pool,
                        shifted_pool,
                    )
                del acts, mask
            for key in variants:
                variants[key][class_name] = outputs[key]
    tensors = {
        "normalized_entropy": torch.tensor(entropy_values),
        "max_weight": torch.tensor(max_values),
        "first_token_weight": torch.tensor(first_values),
        "last_token_weight": torch.tensor(last_values),
        "uniform_l1_distance": torch.tensor(uniform_l1_values),
        "mean_attention_representation_cosine": torch.tensor(
            representation_cosines
        ),
    }
    statistics = {
        "sample_count": sample_count_total,
        "token_count": token_count,
        "single_token_samples": single_token_samples,
        "shifted_fixed_position_count": shifted_fixed_positions,
        **{
            key: {
                "mean": float(values.mean().item()),
                "std": float(values.std().item()),
                "p10": float(torch.quantile(values, 0.1).item()),
                "median": float(torch.quantile(values, 0.5).item()),
                "p90": float(torch.quantile(values, 0.9).item()),
            }
            for key, values in tensors.items()
        },
    }
    return variants, statistics


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
    attention = llm.layers[22].attention
    freqs_cis = llm.freqs_cis
    attention_validation = validate_final_query_attention(
        attention,
        freqs_cis,
        config.device,
        config.dtype,
        args.random_seed,
    )
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
    attention_statistics: dict[str, dict[str, Any]] = {}
    for dataset_index, dataset in enumerate(args.datasets):
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
        train_features, train_statistics = compute_relu_attention_pooling(
            module,
            train_layers[22],
            train_masks,
            state,
            attention,
            freqs_cis,
            config,
            args.random_seed + 1009 * dataset_index,
        )
        test_features, test_statistics = compute_relu_attention_pooling(
            module,
            test_layers[22],
            test_masks,
            state,
            attention,
            freqs_cis,
            config,
            args.random_seed + 1009 * dataset_index + 503,
        )
        cached[dataset] = {"train": train_features, "test": test_features}
        attention_statistics[dataset] = {
            "train": train_statistics,
            "test": test_statistics,
        }
        del train_layers, test_layers, train_masks, test_masks
    del llm, attention, freqs_cis, state
    torch.cuda.empty_cache()
    gc.collect()

    variants = [
        ("mean_pool", "standard mean-pooled ReLU"),
        ("final_query_attention", "L22 final-query-attention-pooled ReLU"),
        ("shifted_attention", "cyclically shifted attention control"),
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
    candidate = summary_by_key["final_query_attention"]
    reference = summary_by_key["mean_pool"]
    shifted = summary_by_key["shifted_attention"]
    dataset_deltas = {
        dataset: {
            "reference": dataset_mean(
                rows_by_key["mean_pool"], dataset, args.k_values
            ),
            "candidate": dataset_mean(
                rows_by_key["final_query_attention"], dataset, args.k_values
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
        "candidate_minus_shifted_at_least_0p002": (
            candidate["mean_acc"] - shifted["mean_acc"] >= 0.002
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
            "candidate": "sum_t mean_heads softmax(q_last k_t / sqrt(d_h)) * ReLU(W_enc(normalize(x_t)-b_pre)+b_enc)",
            "control": "cyclically shift each sample attention vector before pooling",
            "attention_layer": 22,
            "attention_query": "last valid token",
            "head_aggregation": "mean after per-head softmax",
            "uses_actual_qk_with_rope": True,
            "removes_self_attention": False,
            "attention_temperature_sweep": False,
            "same_checkpoint": True,
            "same_parameter_count": True,
            "same_exposed_feature_count": True,
            "pooling_rule_sweep": False,
        },
        "attention_statistics": attention_statistics,
        "attention_reconstruction_validation": attention_validation,
        "architecture_results": rows,
        "summary": summary,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-attention-structured-sae-v1-screen"
            if gate["pass"]
            else "stop-before-attention-structured-sae-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "finalquery-attention-gate.json"
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Structured final-query attention pooling gate",
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
    (args.output_dir / "finalquery-attention-gate.md").write_text(
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
