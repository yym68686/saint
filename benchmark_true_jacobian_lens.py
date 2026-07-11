#!/usr/bin/env python3
"""Benchmark an exact downstream Jacobian from a cached SAE activation site."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from capture_activations import load_model
from llama_3.args import ModelArgs


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Structured activation cache is incomplete")
    if not manifest.get("summary", {}).get("read_only_finalized"):
        raise ValueError("Structured activation cache is not finalized read-only")
    return manifest


def load_prompt(
    cache_dir: Path,
    manifest: dict[str, Any],
    source_layer: int,
    sequence_length: int,
    prompt_ordinal: int,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]:
    candidates_seen = 0
    for shard in manifest["shards"]:
        meta = torch.load(
            cache_dir / shard["meta"]["path"],
            map_location="cpu",
            weights_only=True,
        )
        lengths = meta["lengths"].to(torch.int64)
        eligible = torch.nonzero(lengths >= sequence_length, as_tuple=False).flatten()
        if prompt_ordinal >= candidates_seen + int(eligible.numel()):
            candidates_seen += int(eligible.numel())
            continue
        local_index = int(eligible[prompt_ordinal - candidates_seen].item())
        token_ids = meta["token_ids"][local_index, :sequence_length].to(torch.long)
        offsets = meta["offsets"].to(torch.int64)
        activation_file = cache_dir / shard["layers"][str(source_layer)]["path"]
        activations = torch.load(
            activation_file,
            map_location="cpu",
            weights_only=True,
        )
        start = int(offsets[local_index].item())
        cached = activations[start : start + sequence_length].float()
        sample_id = int(meta["sample_ids"][local_index].item())
        capture_batch_size = int(manifest["configuration"]["batch_size"])
        capture_start = (local_index // capture_batch_size) * capture_batch_size
        capture_end = min(capture_start + capture_batch_size, int(lengths.numel()))
        capture_width = int(lengths[capture_start:capture_end].max().item())
        capture_tokens = meta["token_ids"][
            capture_start:capture_end,
            :capture_width,
        ].to(torch.long)
        capture_row = local_index - capture_start
        return token_ids, cached, sample_id, capture_tokens, capture_row
    raise IndexError(
        f"Only {candidates_seen} prompts have at least {sequence_length} tokens; "
        f"cannot select ordinal {prompt_ordinal}"
    )


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


def forward_to_source(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    source_layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    sequence_length = int(tokens.shape[1])
    hidden = model.tok_embeddings(tokens)
    model.freqs_cis = model.freqs_cis.to(hidden.device)
    frequencies = model.freqs_cis[:sequence_length]
    mask = causal_mask(sequence_length, hidden.device, hidden.dtype)
    with torch.no_grad():
        for layer_index in range(source_layer):
            hidden = model.layers[layer_index](hidden, 0, frequencies, mask)
    return hidden.detach(), frequencies, mask


def forward_source_to_target(
    model: torch.nn.Module,
    hidden_at_source: torch.Tensor,
    frequencies: torch.Tensor,
    mask: torch.Tensor | None,
    source_layer: int,
    target_layer: int,
    perturbation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = model.layers[source_layer]
    normalized = source.attention_norm(hidden_at_source)
    perturbed = normalized + perturbation
    hidden = hidden_at_source + source.attention(perturbed, 0, frequencies, mask)
    hidden = hidden + source.feed_forward(source.ffn_norm(hidden))
    for layer_index in range(source_layer + 1, target_layer + 1):
        hidden = model.layers[layer_index](hidden, 0, frequencies, mask)
    return hidden, normalized


def detach_attention_caches(
    model: torch.nn.Module,
    source_layer: int,
    target_layer: int,
) -> None:
    for layer_index in range(source_layer, target_layer + 1):
        attention = model.layers[layer_index].attention
        attention.cache_k = attention.cache_k.detach()
        attention.cache_v = attention.cache_v.detach()


def jacobian_rows(
    target_sum: torch.Tensor,
    perturbation: torch.Tensor,
    row_count: int,
    row_batch_size: int,
    source_positions: int,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    rows = []
    timing = []
    hidden_size = int(target_sum.numel())
    for start in range(0, row_count, row_batch_size):
        end = min(row_count, start + row_batch_size)
        grad_outputs = torch.zeros(
            (end - start, hidden_size),
            device=target_sum.device,
            dtype=target_sum.dtype,
        )
        indices = torch.arange(start, end, device=target_sum.device)
        grad_outputs[torch.arange(end - start, device=target_sum.device), indices] = 1
        torch.cuda.synchronize()
        started = time.perf_counter()
        gradient = torch.autograd.grad(
            target_sum,
            perturbation,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=end < row_count,
        )[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        gradient = gradient.reshape(end - start, -1) / source_positions
        rows.append(gradient.float().cpu())
        timing.append(
            {
                "start_row": float(start),
                "end_row": float(end),
                "rows": float(end - start),
                "seconds": elapsed,
                "seconds_per_row": elapsed / (end - start),
            }
        )
    return torch.cat(rows, dim=0), timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-layer", type=int, default=22)
    parser.add_argument("--target-layer", type=int, default=26)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--prompt-ordinal", type=int, default=0)
    parser.add_argument("--jacobian-rows", type=int, default=8)
    parser.add_argument("--row-batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-position-average", action="store_true")
    args = parser.parse_args()

    if args.target_layer <= args.source_layer:
        raise ValueError("target layer must be later than source layer")
    args.model_dir = args.model_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    manifest = load_manifest(args.cache_dir)
    token_ids, cached_source, sample_id, capture_tokens, capture_row = load_prompt(
        args.cache_dir,
        manifest,
        args.source_layer,
        args.sequence_length,
        args.prompt_ordinal,
    )

    params = ModelArgs(
        **json.loads((args.model_dir / "params.json").read_text(encoding="utf-8"))
    )
    if args.target_layer >= params.n_layers:
        raise ValueError("target layer must be a valid transformer block index")
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
    capture_tokens = capture_tokens.to(device)
    capture_hidden, _, _ = forward_to_source(
        model,
        capture_tokens,
        args.source_layer,
    )
    with torch.no_grad():
        capture_recomputed = model.layers[args.source_layer].attention_norm(
            capture_hidden
        )[capture_row, : args.sequence_length]
    capture_difference = (
        capture_recomputed.float().cpu() - cached_source
    )
    del capture_tokens, capture_hidden, capture_recomputed

    tokens = token_ids.unsqueeze(0).to(device)
    hidden, frequencies, mask = forward_to_source(model, tokens, args.source_layer)

    perturbation = torch.zeros(
        (1, 1, params.dim),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    torch.cuda.reset_peak_memory_stats()
    target, recomputed_source = forward_source_to_target(
        model,
        hidden,
        frequencies,
        mask,
        args.source_layer,
        args.target_layer,
        perturbation,
    )
    target_sum = target.sum(dim=1).squeeze(0)
    rows, timing = jacobian_rows(
        target_sum,
        perturbation,
        args.jacobian_rows,
        args.row_batch_size,
        args.sequence_length,
    )
    source_difference = recomputed_source.detach().float().cpu().squeeze(0) - cached_source

    position_average_error = None
    if args.verify_position_average:
        del target, target_sum, perturbation
        detach_attention_caches(model, args.source_layer, args.target_layer)
        position_perturbation = torch.zeros(
            (1, args.sequence_length, params.dim),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        target_position, _ = forward_source_to_target(
            model,
            hidden,
            frequencies,
            mask,
            args.source_layer,
            args.target_layer,
            position_perturbation,
        )
        scalar = target_position[:, :, 0].sum()
        explicit = torch.autograd.grad(scalar, position_perturbation)[0]
        explicit_mean = explicit.mean(dim=1).squeeze(0).float().cpu()
        position_average_error = float((explicit_mean - rows[0]).abs().max().item())

    seconds_per_row = sum(row["seconds"] for row in timing) / args.jacobian_rows
    cached_rms = float(cached_source.square().mean().sqrt().item())

    def reproduction_stats(difference: torch.Tensor) -> dict[str, float]:
        return {
            "max_abs_error": float(difference.abs().max().item()),
            "mean_abs_error": float(difference.abs().mean().item()),
            "root_mean_squared_error": float(
                difference.square().mean().sqrt().item()
            ),
            "relative_root_mean_squared_error": float(
                difference.square().mean().sqrt().item() / cached_rms
            ),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    (cached_source + difference).reshape(1, -1),
                    cached_source.reshape(1, -1),
                ).item()
            ),
        }

    report = {
        "method": "exact batched reverse-mode averaged downstream Jacobian",
        "source_representation": "attention-normalized residual stream at layer input",
        "target_representation": "residual stream after target transformer block",
        "source_layer": args.source_layer,
        "target_layer": args.target_layer,
        "final_transformer_block_omitted": args.target_layer < params.n_layers - 1,
        "sample_id": sample_id,
        "prompt_ordinal": args.prompt_ordinal,
        "sequence_length": args.sequence_length,
        "hidden_size": params.dim,
        "jacobian_rows_measured": args.jacobian_rows,
        "row_batch_size": args.row_batch_size,
        "timing": timing,
        "seconds_per_row": seconds_per_row,
        "estimated_seconds_per_prompt_full_jacobian": seconds_per_row * params.dim,
        "estimated_hours_for_10_prompts": seconds_per_row * params.dim * 10 / 3600,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "cache_reproduction": {
            "single_prompt_batch": reproduction_stats(source_difference),
            "original_capture_batch": reproduction_stats(capture_difference),
            "cached_activation_rms": cached_rms,
        },
        "jacobian_rows": {
            "finite": bool(torch.isfinite(rows).all().item()),
            "mean_norm": float(rows.norm(dim=1).mean().item()),
            "min_norm": float(rows.norm(dim=1).min().item()),
            "max_norm": float(rows.norm(dim=1).max().item()),
            "mean_abs": float(rows.abs().mean().item()),
        },
        "shared_delta_vs_explicit_position_mean_max_abs_error": position_average_error,
        "paper_correspondence": {
            "target_positions": "sum over all present/future causal target positions",
            "source_positions": "mean over all source positions",
            "prompt_aggregation": "not performed in this single-prompt benchmark",
            "attention_pattern_gradients": "enabled",
        },
    }
    if not report["jacobian_rows"]["finite"]:
        raise FloatingPointError("Jacobian contains NaN or Inf")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
