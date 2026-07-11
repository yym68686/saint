#!/usr/bin/env python3
"""Gate full-attention token groups before structured SAE training."""

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

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

from llama_3.model_text_only import apply_rotary_emb, repeat_kv


INITIAL3 = [
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "fancyzhx/ag_news",
]
LAYERS = [20, 21, 22, 23]


def load_eval_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("attention_group_eval", path)
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


def causal_attention_matrix(
    attention: Any,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Return head-averaged native causal attention for one valid sequence."""
    if x.ndim != 2:
        raise ValueError(f"Expected [sequence, d_model], got {tuple(x.shape)}")
    sequence_length, _ = x.shape
    x_batched = x.unsqueeze(0)
    xq = attention.wq(x_batched).view(
        1,
        sequence_length,
        attention.n_local_heads,
        attention.head_dim,
    )
    xk = attention.wk(x_batched).view(
        1,
        sequence_length,
        attention.n_local_kv_heads,
        attention.head_dim,
    )
    xq, xk = apply_rotary_emb(
        xq,
        xk,
        freqs_cis=freqs_cis[:sequence_length].to(x.device),
    )
    queries = xq.transpose(1, 2)
    keys = repeat_kv(xk, attention.n_rep).transpose(1, 2)
    scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(
        attention.head_dim
    )
    causal_mask = torch.triu(
        torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            device=x.device,
            dtype=scores.dtype,
        ),
        diagonal=1,
    )
    weights = F.softmax((scores + causal_mask).float(), dim=-1)
    return weights.mean(dim=1)[0]


def validate_attention_matrix(
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
    causal_mask = torch.triu(
        torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            device=device,
            dtype=dtype,
        ),
        diagonal=1,
    )
    native = attention(
        x,
        0,
        freqs_cis[:sequence_length].to(device),
        causal_mask,
    ).float()
    weights = causal_attention_matrix(attention, x[0], freqs_cis)
    values = attention.wv(x).view(
        1,
        sequence_length,
        attention.n_local_kv_heads,
        attention.head_dim,
    )
    values = repeat_kv(values, attention.n_rep).transpose(1, 2)

    # Recompute per head for the output check; the returned matrix above is
    # deliberately head-averaged only for clustering.
    xq = attention.wq(x).view(
        1,
        sequence_length,
        attention.n_local_heads,
        attention.head_dim,
    )
    xk = attention.wk(x).view(
        1,
        sequence_length,
        attention.n_local_kv_heads,
        attention.head_dim,
    )
    xq, xk = apply_rotary_emb(
        xq,
        xk,
        freqs_cis=freqs_cis[:sequence_length].to(device),
    )
    queries = xq.transpose(1, 2)
    keys = repeat_kv(xk, attention.n_rep).transpose(1, 2)
    scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(
        attention.head_dim
    )
    per_head = F.softmax((scores + causal_mask).float(), dim=-1).type_as(xq)
    manual_heads = torch.matmul(per_head, values)
    manual = attention.wo(
        manual_heads.transpose(1, 2).reshape(1, sequence_length, -1)
    ).float()
    difference = native - manual
    max_abs_error = float(difference.abs().max().item())
    mean_abs_error = float(difference.abs().mean().item())
    cosine = float(
        F.cosine_similarity(native.flatten(), manual.flatten(), dim=0).item()
    )
    if max_abs_error > 0.01 or cosine < 0.9999:
        raise RuntimeError(
            "Full attention reconstruction failed: "
            f"max_abs_error={max_abs_error:.6g}, cosine={cosine:.8f}"
        )
    return {
        "sequence_length": float(sequence_length),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "cosine_similarity": cosine,
        "head_averaged_row_sum_error": float(
            (weights.sum(dim=1) - 1.0).abs().max().item()
        ),
    }


def attention_group_labels(
    attentions: list[torch.Tensor],
    group_count: int,
    positional_exponent: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not attentions:
        raise ValueError("At least one attention matrix is required")
    sequence_length = int(attentions[0].shape[0])
    if any(
        tuple(matrix.shape) != (sequence_length, sequence_length)
        for matrix in attentions
    ):
        raise ValueError("Attention matrices have inconsistent shapes")
    group_count = min(group_count, sequence_length)
    average_attention = torch.stack(attentions, dim=0).mean(dim=0)
    symmetric = 0.5 * (average_attention + average_attention.T)
    distance = symmetric.max() - symmetric
    distance = distance - distance.min()
    distance = distance / distance.max().clamp_min(1.0e-12)
    if sequence_length > 1:
        positions = torch.arange(sequence_length, device=distance.device)
        position_distance = (positions[:, None] - positions[None, :]).abs().float()
        position_distance = position_distance / float(sequence_length - 1)
        distance = distance * position_distance.pow(positional_exponent)
    distance.fill_diagonal_(0.0)
    distance = 0.5 * (distance + distance.T)
    labels_np = AgglomerativeClustering(
        n_clusters=group_count,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance.float().cpu().numpy())
    labels = torch.from_numpy(labels_np.astype(np.int64, copy=False))
    within_mask = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(sequence_length, dtype=torch.bool)
    within_values = symmetric.cpu()[within_mask & off_diagonal]
    between_values = symmetric.cpu()[(~within_mask) & off_diagonal]
    sizes = torch.bincount(labels, minlength=group_count).float()
    return labels, {
        "within_attention": (
            float(within_values.mean().item()) if within_values.numel() else 0.0
        ),
        "between_attention": (
            float(between_values.mean().item()) if between_values.numel() else 0.0
        ),
        "within_minus_between": float(
            (within_values.mean() - between_values.mean()).item()
        )
        if within_values.numel() and between_values.numel()
        else 0.0,
        "minimum_group_size": float(sizes.min().item()),
        "maximum_group_size": float(sizes.max().item()),
        "mean_group_size": float(sizes.mean().item()),
    }


def contiguous_group_labels(sequence_length: int, group_count: int) -> torch.Tensor:
    group_count = min(group_count, sequence_length)
    positions = torch.arange(sequence_length, dtype=torch.long)
    return torch.div(positions * group_count, sequence_length, rounding_mode="floor")


def maximally_shifted_labels(labels: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    sequence_length = int(labels.numel())
    if sequence_length <= 1:
        return labels.clone(), 0, 1.0
    agreements = torch.tensor(
        [
            float((labels == torch.roll(labels, shifts=shift)).float().mean().item())
            for shift in range(1, sequence_length)
        ]
    )
    best_index = int(torch.argmin(agreements).item())
    shift = best_index + 1
    return (
        torch.roll(labels, shifts=shift),
        shift,
        float(agreements[best_index].item()),
    )


def strongest_group_rms(
    token_features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    labels = labels.to(token_features.device)
    profiles = []
    for group_id in torch.unique(labels, sorted=True):
        selected = token_features[labels == group_id]
        profiles.append(torch.sqrt(selected.square().mean(dim=0) + 1.0e-8))
    return torch.stack(profiles, dim=0).max(dim=0).values


def compute_group_variants(
    module: Any,
    layer_acts: dict[int, dict[str, torch.Tensor]],
    masks: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    attentions: dict[int, Any],
    freqs_cis: torch.Tensor,
    config: Any,
    group_count: int,
    positional_exponent: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    feature_count = int(state["encoder.weight"].shape[0])
    variant_keys = [
        "mean_pool",
        "global_rms",
        "contiguous_group_rms",
        "shifted_attention_group_rms",
        "attention_group_rms",
    ]
    variants: dict[str, dict[str, torch.Tensor]] = {key: {} for key in variant_keys}
    statistics: dict[str, list[float]] = {
        "within_attention": [],
        "between_attention": [],
        "within_minus_between": [],
        "minimum_group_size": [],
        "maximum_group_size": [],
        "mean_group_size": [],
        "shift": [],
        "shifted_assignment_agreement": [],
        "attention_vs_contiguous_assignment_agreement": [],
        "attention_representation_cosine_to_mean": [],
        "attention_representation_cosine_to_shifted": [],
    }
    sample_count_total = 0
    token_count_total = 0
    with torch.inference_mode():
        for class_name, acts_l22_cpu in layer_acts[22].items():
            mask_cpu = masks[class_name]
            sample_count = int(acts_l22_cpu.shape[0])
            outputs = {
                key: torch.empty((sample_count, feature_count), dtype=torch.float32)
                for key in variant_keys
            }
            for start in range(0, sample_count, config.sae_seq_batch_size):
                end = min(sample_count, start + config.sae_seq_batch_size)
                batch_by_layer = {
                    layer: layer_acts[layer][class_name][start:end].to(
                        config.device, non_blocking=True
                    )
                    for layer in LAYERS
                }
                mask_batch = mask_cpu[start:end].to(config.device, non_blocking=True)
                for local_index in range(end - start):
                    valid = mask_batch[local_index]
                    sequence_by_layer = {
                        layer: batch_by_layer[layer][local_index, valid]
                        for layer in LAYERS
                    }
                    sequence_length = int(sequence_by_layer[22].shape[0])
                    if sequence_length == 0:
                        raise ValueError("Encountered an empty evaluation sample")
                    x_sae = module.normalize_activation(
                        sequence_by_layer[22], config.dtype, config.normalize_eps
                    )
                    token_features = torch.relu(
                        F.linear(
                            x_sae - state["b_pre"],
                            state["encoder.weight"],
                            state["encoder.bias"],
                        )
                    ).float()
                    layer_attention = [
                        causal_attention_matrix(
                            attentions[layer], sequence_by_layer[layer], freqs_cis
                        )
                        for layer in LAYERS
                    ]
                    labels, group_stats = attention_group_labels(
                        layer_attention,
                        group_count,
                        positional_exponent,
                    )
                    contiguous = contiguous_group_labels(sequence_length, group_count)
                    shifted, shift, shifted_agreement = maximally_shifted_labels(labels)

                    mean_pool = token_features.mean(dim=0)
                    global_rms = torch.sqrt(
                        token_features.square().mean(dim=0) + 1.0e-8
                    )
                    contiguous_rms = strongest_group_rms(token_features, contiguous)
                    shifted_rms = strongest_group_rms(token_features, shifted)
                    attention_rms = strongest_group_rms(token_features, labels)
                    row = start + local_index
                    outputs["mean_pool"][row] = mean_pool.cpu()
                    outputs["global_rms"][row] = global_rms.cpu()
                    outputs["contiguous_group_rms"][row] = contiguous_rms.cpu()
                    outputs["shifted_attention_group_rms"][row] = shifted_rms.cpu()
                    outputs["attention_group_rms"][row] = attention_rms.cpu()

                    for key, value in group_stats.items():
                        statistics[key].append(float(value))
                    statistics["shift"].append(float(shift))
                    statistics["shifted_assignment_agreement"].append(
                        shifted_agreement
                    )
                    statistics["attention_vs_contiguous_assignment_agreement"].append(
                        float((labels == contiguous).float().mean().item())
                    )
                    statistics["attention_representation_cosine_to_mean"].append(
                        float(
                            F.cosine_similarity(
                                attention_rms.unsqueeze(0),
                                mean_pool.unsqueeze(0),
                                dim=1,
                            )[0].item()
                        )
                    )
                    statistics["attention_representation_cosine_to_shifted"].append(
                        float(
                            F.cosine_similarity(
                                attention_rms.unsqueeze(0),
                                shifted_rms.unsqueeze(0),
                                dim=1,
                            )[0].item()
                        )
                    )
                    sample_count_total += 1
                    token_count_total += sequence_length
                    del (
                        sequence_by_layer,
                        x_sae,
                        token_features,
                        layer_attention,
                        labels,
                        contiguous,
                        shifted,
                        mean_pool,
                        global_rms,
                        contiguous_rms,
                        shifted_rms,
                        attention_rms,
                    )
                del batch_by_layer, mask_batch
            for key in variant_keys:
                variants[key][class_name] = outputs[key]
    summary = {
        "sample_count": sample_count_total,
        "token_count": token_count_total,
        **{
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p10": float(np.quantile(values, 0.1)),
                "median": float(np.quantile(values, 0.5)),
                "p90": float(np.quantile(values, 0.9)),
            }
            for key, values in statistics.items()
            if values
        },
    }
    return variants, summary


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


def dataset_mean(row: dict[str, Any], dataset: str, k_values: list[int]) -> float:
    metrics = row["dataset_results"][dataset]
    return sum(metrics[f"sae_top_{k}_test_accuracy"] for k in k_values) / len(
        k_values
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path("/root/autodl-tmp/saebench_sparse_probing_all_architectures.py"),
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
    parser.add_argument("--sae-seq-batch-size", type=int, default=1)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--group-count", type=int, default=20)
    parser.add_argument("--positional-exponent", type=float, default=0.02)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.group_count != 20 or args.positional_exponent != 0.02:
        raise ValueError("The registered gate fixes group_count=20 and exponent=0.02")
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
    attentions = {layer: llm.layers[layer].attention for layer in LAYERS}
    freqs_cis = llm.freqs_cis
    attention_validation = {
        str(layer): validate_attention_matrix(
            attentions[layer],
            freqs_cis,
            config.device,
            config.dtype,
            args.random_seed + layer,
        )
        for layer in LAYERS
    }
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
    group_statistics: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        print(f"== Cache dataset: {dataset}", flush=True)
        train_data, test_data = module.get_multi_label_train_test_data(
            dataset, args.train_size, args.test_size, args.random_seed
        )
        train_layers, train_masks = module.collect_layer_activations(
            llm, tokenizer, train_data, config, LAYERS
        )
        test_layers, test_masks = module.collect_layer_activations(
            llm, tokenizer, test_data, config, LAYERS
        )
        train_features, train_statistics = compute_group_variants(
            module,
            train_layers,
            train_masks,
            state,
            attentions,
            freqs_cis,
            config,
            args.group_count,
            args.positional_exponent,
        )
        test_features, test_statistics = compute_group_variants(
            module,
            test_layers,
            test_masks,
            state,
            attentions,
            freqs_cis,
            config,
            args.group_count,
            args.positional_exponent,
        )
        cached[dataset] = {"train": train_features, "test": test_features}
        group_statistics[dataset] = {
            "train": train_statistics,
            "test": test_statistics,
        }
        del train_layers, test_layers, train_masks, test_masks
    del llm, attentions, freqs_cis, state
    torch.cuda.empty_cache()
    gc.collect()

    variants = [
        ("mean_pool", "standard mean-pooled ReLU"),
        ("global_rms", "global token RMS control"),
        ("contiguous_group_rms", "equal-width contiguous-group RMS control"),
        (
            "shifted_attention_group_rms",
            "size-preserving shifted-attention-group RMS control",
        ),
        ("attention_group_rms", "full-attention sequential-group RMS"),
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
    candidate = summary_by_key["attention_group_rms"]
    reference = summary_by_key["mean_pool"]
    control_keys = [
        "global_rms",
        "contiguous_group_rms",
        "shifted_attention_group_rms",
    ]
    best_control = max(
        (summary_by_key[key] for key in control_keys), key=lambda row: row["mean_acc"]
    )
    dataset_deltas = {
        dataset: {
            "reference": dataset_mean(
                rows_by_key["mean_pool"], dataset, args.k_values
            ),
            "candidate": dataset_mean(
                rows_by_key["attention_group_rms"], dataset, args.k_values
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
        "candidate_minus_best_matched_control_at_least_0p003": (
            candidate["mean_acc"] - best_control["mean_acc"] >= 0.003
        ),
    }
    gate["pass"] = all(gate.values())
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "cache_dir": str(args.cache_dir),
            "cache_manifest_sha256": sha256(args.cache_dir / "manifest.json"),
            "layers": LAYERS,
            "datasets": args.datasets,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "k_values": args.k_values,
            "random_seed": args.random_seed,
            "group_count": args.group_count,
            "positional_exponent": args.positional_exponent,
        },
        "mechanism": {
            "grouping": "agglomerative clustering of symmetrized L20-L23 mean native attention times normalized 1D distance^0.02",
            "group_profile": "sqrt(mean_tokens_in_group(z_token^2)+1e-8)",
            "sample_feature": "maximum group profile per SAE coordinate",
            "controls": control_keys,
            "same_checkpoint": True,
            "same_parameter_count": True,
            "same_exposed_feature_count": True,
            "hyperparameter_sweep": False,
        },
        "attention_reconstruction_validation": attention_validation,
        "group_statistics": group_statistics,
        "architecture_results": rows,
        "summary": summary,
        "best_matched_control": best_control,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": (
            "authorize-attention-group-structured-sae-v2-screen"
            if gate["pass"]
            else "stop-before-attention-group-structured-sae-training"
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_json = args.output_dir / "attention-group-gate.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Structured full-attention token-group gate",
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
    (args.output_dir / "attention-group-gate.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "best_matched_control": best_control,
                "gate": gate,
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
