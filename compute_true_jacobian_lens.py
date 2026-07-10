#!/usr/bin/env python3
"""Compute and average exact downstream Jacobians on unlabeled OWT prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from benchmark_true_jacobian_lens import (
    detach_attention_caches,
    forward_source_to_target,
    forward_to_source,
    jacobian_rows,
    load_manifest,
    load_prompt,
)
from capture_activations import load_model
from llama_3.args import ModelArgs


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def count_eligible_prompts(cache_dir: Path, manifest: dict, sequence_length: int) -> int:
    count = 0
    for shard in manifest["shards"]:
        meta = torch.load(
            cache_dir / shard["meta"]["path"],
            map_location="cpu",
            weights_only=True,
        )
        count += int((meta["lengths"] >= sequence_length).sum().item())
    return count


def matrix_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.double() * right.double()).sum()
    denominator = left.double().norm() * right.double().norm()
    return float((numerator / denominator.clamp_min(1.0e-30)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-layer", type=int, default=22)
    parser.add_argument("--target-layer", type=int, default=26)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--prompt-count", type=int, default=10)
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--row-batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.model_dir = args.model_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = args.output_dir / "per_prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.cache_dir)
    eligible_count = count_eligible_prompts(
        args.cache_dir,
        manifest,
        args.sequence_length,
    )
    if eligible_count < args.prompt_count:
        raise ValueError(
            f"Need {args.prompt_count} eligible prompts, found {eligible_count}"
        )
    prompt_ordinals = random.Random(args.prompt_seed).sample(
        range(eligible_count),
        args.prompt_count,
    )

    params = ModelArgs(
        **json.loads((args.model_dir / "params.json").read_text(encoding="utf-8"))
    )
    if args.target_layer >= params.n_layers - 1:
        raise ValueError("Target must omit the final transformer block")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    device = torch.device(args.device)
    started = time.time()
    model = load_model(
        model_path=args.model_dir / "consolidated.00.pth",
        model_args=params,
        store_layer_activ=[],
        device=device,
        dtype=dtype,
    )
    model.requires_grad_(False)

    prompt_jacobians: list[torch.Tensor] = []
    prompt_metadata: list[dict] = []
    for prompt_index, prompt_ordinal in enumerate(prompt_ordinals):
        checkpoint_path = prompt_dir / f"prompt-{prompt_index:02d}-jacobian.pt"
        metadata_path = prompt_dir / f"prompt-{prompt_index:02d}-metadata.json"
        if checkpoint_path.exists() and metadata_path.exists():
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            jacobian = checkpoint["jacobian"].float()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["prompt_ordinal"] != prompt_ordinal:
                raise ValueError(f"Prompt checkpoint mismatch: {metadata_path}")
            print(
                json.dumps(
                    {
                        "event": "reuse_prompt_jacobian",
                        "prompt_index": prompt_index,
                        "prompt_ordinal": prompt_ordinal,
                    }
                ),
                flush=True,
            )
        else:
            token_ids, _, sample_id, _, _ = load_prompt(
                args.cache_dir,
                manifest,
                args.source_layer,
                args.sequence_length,
                prompt_ordinal,
            )
            tokens = token_ids.unsqueeze(0).to(device)
            hidden, frequencies, mask = forward_to_source(
                model,
                tokens,
                args.source_layer,
            )
            perturbation = torch.zeros(
                (1, 1, params.dim),
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            torch.cuda.reset_peak_memory_stats()
            target, _ = forward_source_to_target(
                model,
                hidden,
                frequencies,
                mask,
                args.source_layer,
                args.target_layer,
                perturbation,
            )
            target_sum = target.sum(dim=1).squeeze(0)
            prompt_started = time.time()
            jacobian, timing = jacobian_rows(
                target_sum,
                perturbation,
                params.dim,
                args.row_batch_size,
                args.sequence_length,
            )
            elapsed = time.time() - prompt_started
            metadata = {
                "prompt_index": prompt_index,
                "prompt_ordinal": prompt_ordinal,
                "sample_id": sample_id,
                "sequence_length": args.sequence_length,
                "source_layer": args.source_layer,
                "target_layer": args.target_layer,
                "row_batch_size": args.row_batch_size,
                "seconds": elapsed,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "jacobian_frobenius_norm": float(jacobian.norm().item()),
                "jacobian_finite": bool(torch.isfinite(jacobian).all().item()),
                "timing_first_batch": timing[0],
                "timing_last_batch": timing[-1],
            }
            if not metadata["jacobian_finite"]:
                raise FloatingPointError(f"Prompt {prompt_index} Jacobian is not finite")
            torch.save(
                {
                    "jacobian": jacobian,
                    "prompt_ordinal": torch.tensor(prompt_ordinal),
                    "sample_id": torch.tensor(sample_id),
                },
                checkpoint_path,
            )
            metadata["checkpoint_sha256"] = sha256_file(checkpoint_path)
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            detach_attention_caches(model, 0, args.target_layer)
            del tokens, hidden, frequencies, mask, perturbation, target, target_sum
            torch.cuda.empty_cache()
            print(
                json.dumps({"event": "prompt_complete", **metadata}),
                flush=True,
            )
        prompt_jacobians.append(jacobian)
        prompt_metadata.append(metadata)

    del model
    torch.cuda.empty_cache()
    cumulative = torch.zeros_like(prompt_jacobians[0], dtype=torch.float64)
    averages: dict[int, torch.Tensor] = {}
    checkpoints = sorted(set([1, 2, 5, args.prompt_count]))
    for index, jacobian in enumerate(prompt_jacobians, start=1):
        cumulative.add_(jacobian.double())
        if index in checkpoints:
            averages[index] = (cumulative / index).float()
    final_average = averages[args.prompt_count]

    convergence = {}
    final_norm = final_average.double().norm()
    for count, average in averages.items():
        difference = average.double() - final_average.double()
        convergence[str(count)] = {
            "cosine_to_final": matrix_cosine(average, final_average),
            "relative_frobenius_error_to_final": float(
                (difference.norm() / final_norm.clamp_min(1.0e-30)).item()
            ),
        }
    identity = torch.eye(params.dim, dtype=torch.float32)
    average_path = args.output_dir / f"average-jacobian-n{args.prompt_count}.pt"
    torch.save(
        {
            "jacobian": final_average,
            "prompt_ordinals": torch.tensor(prompt_ordinals, dtype=torch.int64),
            "sample_ids": torch.tensor(
                [row["sample_id"] for row in prompt_metadata],
                dtype=torch.int64,
            ),
        },
        average_path,
    )
    report = {
        "method": "exact mean downstream Jacobian",
        "paper_formula": (
            "E_prompt mean_source_position d(sum_target_positions "
            "target_residual)/d(source_activation)"
        ),
        "source_representation": "attention-normalized residual stream at layer input",
        "target_representation": "residual stream after penultimate transformer block",
        "source_layer": args.source_layer,
        "target_layer": args.target_layer,
        "final_transformer_block_omitted": True,
        "attention_pattern_gradients": "enabled",
        "sequence_length": args.sequence_length,
        "eligible_prompt_count": eligible_count,
        "prompt_count": args.prompt_count,
        "prompt_seed": args.prompt_seed,
        "prompt_ordinals": prompt_ordinals,
        "sample_ids": [row["sample_id"] for row in prompt_metadata],
        "row_batch_size": args.row_batch_size,
        "hidden_size": params.dim,
        "jacobian_shape": list(final_average.shape),
        "average_jacobian_path": str(average_path),
        "average_jacobian_sha256": sha256_file(average_path),
        "average_frobenius_norm": float(final_average.norm().item()),
        "cosine_to_identity": matrix_cosine(final_average, identity),
        "relative_frobenius_distance_to_identity": float(
            ((final_average - identity).double().norm() / identity.double().norm()).item()
        ),
        "convergence": convergence,
        "prompt_metadata": prompt_metadata,
        "elapsed_seconds": time.time() - started,
        "uses_saebench_labels": False,
        "uses_saebench_class_names": False,
        "uses_eval_split": False,
        "uses_test_feedback": False,
        "cache_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
    }
    report_path = args.output_dir / "average-jacobian-metadata.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "average_complete", **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
