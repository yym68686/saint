#!/usr/bin/env python3
"""Extract unlabeled per-document next-token gradients at the SAE input site."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import stat
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from capture_activations import load_model
from llama_3.args import ModelArgs


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def causal_mask(
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if sequence_length <= 1:
        return None
    mask = torch.full(
        (sequence_length, sequence_length),
        float("-inf"),
        device=device,
    )
    return torch.triu(mask, diagonal=1).to(dtype=dtype)


def detach_attention_caches(model: torch.nn.Module, start_layer: int = 0) -> None:
    for layer_index in range(start_layer, len(model.layers)):
        attention = model.layers[layer_index].attention
        attention.cache_k = attention.cache_k.detach()
        attention.cache_v = attention.cache_v.detach()


def forward_to_source(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    source_layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = int(tokens.shape[1])
    hidden = model.tok_embeddings(tokens)
    model.freqs_cis = model.freqs_cis.to(hidden.device)
    frequencies = model.freqs_cis[:sequence_length]
    mask = causal_mask(sequence_length, hidden.device, hidden.dtype)
    detach_attention_caches(model, 0)
    with torch.no_grad():
        for layer_index in range(source_layer):
            hidden = model.layers[layer_index](hidden, 0, frequencies, mask)
    return hidden.detach(), frequencies


def forward_from_source(
    model: torch.nn.Module,
    hidden_at_source: torch.Tensor,
    normalized_source: torch.Tensor,
    frequencies: torch.Tensor,
    source_layer: int,
) -> torch.Tensor:
    mask = causal_mask(
        int(hidden_at_source.shape[1]),
        hidden_at_source.device,
        hidden_at_source.dtype,
    )
    detach_attention_caches(model, source_layer)
    source = model.layers[source_layer]
    hidden = hidden_at_source + source.attention(
        normalized_source,
        0,
        frequencies,
        mask,
    )
    hidden = hidden + source.feed_forward(source.ffn_norm(hidden))
    for layer_index in range(source_layer + 1, len(model.layers)):
        hidden = model.layers[layer_index](hidden, 0, frequencies, mask)
    return model.output(model.norm(hidden)).float()


def normalize_activation(
    values: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.float()
    scale = values.std(dim=-1, keepdim=True) + eps
    normalized = (values - values.mean(dim=-1, keepdim=True)) / scale
    return normalized, scale


@dataclass(frozen=True)
class SampleRecord:
    selection_rank: int
    shard_index: int
    local_index: int
    capture_start: int
    sample_id: int


def select_samples(
    cache_dir: Path,
    manifest: dict,
    sequence_length: int,
    sample_count: int,
    seed: int,
) -> list[SampleRecord]:
    capture_batch_size = int(manifest["configuration"]["batch_size"])
    eligible: list[tuple[int, int, int, int]] = []
    for shard_index, shard in enumerate(manifest["shards"]):
        meta = torch.load(
            cache_dir / shard["meta"]["path"],
            map_location="cpu",
            weights_only=True,
        )
        lengths = meta["lengths"].to(torch.int64)
        sample_ids = meta["sample_ids"].to(torch.int64)
        indices = torch.nonzero(lengths >= sequence_length, as_tuple=False).flatten()
        for local_index in indices.tolist():
            eligible.append(
                (
                    shard_index,
                    int(local_index),
                    (int(local_index) // capture_batch_size) * capture_batch_size,
                    int(sample_ids[local_index].item()),
                )
            )
    if len(eligible) < sample_count:
        raise ValueError(f"Requested {sample_count} samples from {len(eligible)} eligible")
    chosen = random.Random(seed).sample(eligible, sample_count)
    return [
        SampleRecord(rank, shard, local, capture, sample_id)
        for rank, (shard, local, capture, sample_id) in enumerate(chosen)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-layer", type=int, default=22)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--source-tolerance", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.cache_dir = args.cache_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output = args.output.resolve()
    if args.cache_dir.stat().st_mode & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    ):
        raise PermissionError(f"Cache must be read-only: {args.cache_dir}")
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Structured cache is incomplete")
    if not manifest.get("summary", {}).get("read_only_finalized"):
        raise ValueError("Structured cache is not finalized read-only")
    if args.source_layer not in map(int, manifest["configuration"]["layers"]):
        raise ValueError(f"Layer {args.source_layer} is missing from the cache")
    if args.microbatch_size < 1:
        raise ValueError("microbatch-size must be positive")

    records = select_samples(
        args.cache_dir,
        manifest,
        args.sequence_length,
        args.sample_count,
        args.sample_seed,
    )
    groups: dict[tuple[int, int], list[SampleRecord]] = defaultdict(list)
    for record in records:
        groups[(record.shard_index, record.capture_start)].append(record)

    params = ModelArgs(
        **json.loads((args.model_dir / "params.json").read_text(encoding="utf-8"))
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    device = torch.device(args.device)
    model = load_model(
        model_path=args.model_dir / "consolidated.00.pth",
        model_args=params,
        store_layer_activ=[],
        device=device,
        dtype=dtype,
    )
    model.requires_grad_(False)
    model.eval()

    gradients = torch.empty((args.sample_count, params.dim), dtype=torch.float32)
    activation_means = torch.empty_like(gradients)
    losses = torch.empty(args.sample_count, dtype=torch.float32)
    sample_ids = torch.empty(args.sample_count, dtype=torch.int64)
    source_max_abs = 0.0
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    processed = 0
    current_shard_index = -1
    current_meta = None
    current_sources = None
    capture_batch_size = int(manifest["configuration"]["batch_size"])

    for group_index, ((shard_index, capture_start), group) in enumerate(
        sorted(groups.items())
    ):
        if shard_index != current_shard_index:
            shard = manifest["shards"][shard_index]
            current_meta = torch.load(
                args.cache_dir / shard["meta"]["path"],
                map_location="cpu",
                weights_only=True,
            )
            current_sources = torch.load(
                args.cache_dir / shard["layers"][str(args.source_layer)]["path"],
                map_location="cpu",
                weights_only=True,
            )
            current_shard_index = shard_index
        assert current_meta is not None and current_sources is not None
        lengths = current_meta["lengths"].to(torch.int64)
        offsets = current_meta["offsets"].to(torch.int64)
        tokens_cpu = current_meta["token_ids"].to(torch.int64)
        capture_end = min(capture_start + capture_batch_size, int(lengths.numel()))
        capture_width = int(lengths[capture_start:capture_end].max().item())
        capture_tokens = tokens_cpu[
            capture_start:capture_end,
            :capture_width,
        ].to(device)
        capture_hidden, capture_frequencies = forward_to_source(
            model,
            capture_tokens,
            args.source_layer,
        )
        with torch.no_grad():
            capture_source = model.layers[args.source_layer].attention_norm(
                capture_hidden
            )

        for cursor in range(0, len(group), args.microbatch_size):
            chunk = group[cursor : cursor + args.microbatch_size]
            capture_rows = torch.tensor(
                [record.local_index - capture_start for record in chunk],
                device=device,
                dtype=torch.int64,
            )
            hidden = capture_hidden.index_select(0, capture_rows)[
                :, : args.sequence_length
            ]
            source = capture_source.index_select(0, capture_rows)[
                :, : args.sequence_length
            ]
            cached = torch.stack(
                [
                    current_sources[
                        int(offsets[record.local_index].item())
                        : int(offsets[record.local_index].item())
                        + args.sequence_length
                    ]
                    for record in chunk
                ]
            ).to(device)
            error = (source.float() - cached.float()).abs().max()
            source_max_abs = max(source_max_abs, float(error.item()))
            if float(error.item()) > args.source_tolerance:
                raise RuntimeError(
                    f"Source reproduction error {float(error.item()):.8f} exceeds "
                    f"{args.source_tolerance:.8f}"
                )
            source_variable = source.detach().clone().requires_grad_(True)
            frequencies = capture_frequencies[: args.sequence_length]
            logits = forward_from_source(
                model,
                hidden,
                source_variable,
                frequencies,
                args.source_layer,
            )
            selected_tokens = torch.stack(
                [tokens_cpu[record.local_index, : args.sequence_length] for record in chunk]
            ).to(device)
            per_token_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                selected_tokens[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(len(chunk), args.sequence_length - 1)
            per_sample_loss = per_token_loss.mean(dim=1)
            gradient = torch.autograd.grad(per_sample_loss.sum(), source_variable)[0]
            normalized, scale = normalize_activation(source_variable.detach(), args.normalize_eps)
            normalized_gradient = gradient.float() * scale
            gradient_vectors = normalized_gradient.mean(dim=1)
            mean_vectors = normalized.mean(dim=1)
            for local_row, record in enumerate(chunk):
                gradients[record.selection_rank] = gradient_vectors[local_row].cpu()
                activation_means[record.selection_rank] = mean_vectors[local_row].cpu()
                losses[record.selection_rank] = per_sample_loss[local_row].detach().cpu()
                sample_ids[record.selection_rank] = record.sample_id
            processed += len(chunk)
            detach_attention_caches(model, args.source_layer)
            del (
                capture_rows,
                hidden,
                source,
                cached,
                error,
                source_variable,
                frequencies,
                logits,
                selected_tokens,
                per_token_loss,
                per_sample_loss,
                gradient,
                normalized,
                scale,
                normalized_gradient,
                gradient_vectors,
                mean_vectors,
            )
            torch.cuda.empty_cache()
        del capture_tokens, capture_hidden, capture_frequencies, capture_source
        if group_index == 0 or (group_index + 1) % 25 == 0 or processed == args.sample_count:
            print(
                json.dumps(
                    {
                        "event": "gradient_extraction_progress",
                        "processed": processed,
                        "sample_count": args.sample_count,
                        "groups_processed": group_index + 1,
                        "group_count": len(groups),
                        "source_max_abs": source_max_abs,
                        "elapsed_seconds": time.time() - started,
                        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    }
                ),
                flush=True,
            )

    if processed != args.sample_count:
        raise RuntimeError(f"Extracted {processed}, expected {args.sample_count}")
    if not torch.isfinite(gradients).all() or not torch.isfinite(activation_means).all():
        raise FloatingPointError("Extraction produced NaN/Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gradients": gradients,
        "activation_means": activation_means,
        "losses": losses,
        "sample_ids": sample_ids,
        "selection_rank": torch.arange(args.sample_count, dtype=torch.int64),
    }
    torch.save(payload, args.output)
    report = {
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "sample_count": args.sample_count,
        "sequence_length": args.sequence_length,
        "sample_seed": args.sample_seed,
        "source_layer": args.source_layer,
        "source_representation": "attention-normalized L22 cache coordinate",
        "gradient_target": "mean next-token cross-entropy over positions 1..31",
        "gradient_coordinate": "per-token normalized SAE input coordinate",
        "gradient_norm_mean": float(gradients.norm(dim=1).mean().item()),
        "gradient_norm_min": float(gradients.norm(dim=1).min().item()),
        "activation_norm_mean": float(activation_means.norm(dim=1).mean().item()),
        "next_token_ce_mean": float(losses.mean().item()),
        "source_reproduction_max_abs": source_max_abs,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "uses_saebench_labels": False,
        "uses_saebench_class_names": False,
        "uses_eval_split": False,
        "uses_test_feedback": False,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "gradient_extraction_complete", **report}, indent=2))


if __name__ == "__main__":
    main()
