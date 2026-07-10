#!/usr/bin/env python3
"""Capture a versioned activation cache without discarding sample boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from capture_activations import load_model
from llama_3.args import ModelArgs
from llama_3.tokenizer import Tokenizer
from openwebtext_sentences_dataset import OpenWebTextSentencesDataset
from utils.cuda_utils import set_up_cuda


SCHEMA_VERSION = "structured-activation-cache-v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def collate_chunks(
    chunks: list[dict[str, Any]],
    layers: list[int],
    shard_index: int,
) -> dict[str, Any]:
    sample_ids = torch.cat([chunk["sample_ids"] for chunk in chunks]).to(torch.int64)
    lengths = torch.cat([chunk["lengths"] for chunk in chunks]).to(torch.int32)
    if (lengths <= 0).any():
        raise ValueError("Structured cache does not permit empty samples")

    max_length = int(lengths.max().item())
    sample_count = int(lengths.numel())
    token_ids = torch.full((sample_count, max_length), -1, dtype=torch.int32)
    attention_mask = torch.zeros((sample_count, max_length), dtype=torch.bool)
    cursor = 0
    for chunk in chunks:
        count = int(chunk["lengths"].numel())
        width = int(chunk["token_ids"].shape[1])
        token_ids[cursor : cursor + count, :width] = chunk["token_ids"].to(torch.int32)
        attention_mask[cursor : cursor + count, :width] = chunk["attention_mask"]
        cursor += count

    offsets = torch.zeros(sample_count + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths.to(torch.int64), dim=0)
    activations = {
        str(layer): torch.cat(
            [chunk["activations"][layer] for chunk in chunks],
            dim=0,
        ).contiguous()
        for layer in layers
    }
    total_tokens = int(offsets[-1].item())
    for layer, tensor in activations.items():
        if tensor.ndim != 2 or int(tensor.shape[0]) != total_tokens:
            raise ValueError(
                f"Layer {layer} has shape {tuple(tensor.shape)}, "
                f"expected ({total_tokens}, d_model)"
            )
        if not torch.isfinite(tensor.float()).all():
            raise FloatingPointError(f"Layer {layer} contains NaN or Inf")

    if not torch.equal(attention_mask.sum(dim=1).to(torch.int32), lengths):
        raise ValueError("Attention mask and lengths disagree")
    if (token_ids[attention_mask] < 0).any():
        raise ValueError("Valid token positions contain the padding sentinel")

    return {
        "schema_version": SCHEMA_VERSION,
        "shard_index": shard_index,
        "sample_ids": sample_ids,
        "lengths": lengths,
        "offsets": offsets,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "activations": activations,
    }


def finalize_read_only(output_dir: Path) -> None:
    for path in output_dir.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    output_dir.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[20, 21, 22, 23])
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--max-token-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-model-weight-hash", action="store_true")
    parser.add_argument("--finalize-read-only", action="store_true")
    args = parser.parse_args()

    args.model_dir = args.model_dir.resolve()
    args.parquet_path = args.parquet_path.resolve()
    args.output_dir = args.output_dir.resolve()
    layers = sorted(set(args.layers))
    if len(layers) != len(args.layers):
        raise ValueError(f"Duplicate layers are not allowed: {args.layers}")
    if args.shard_samples % args.batch_size != 0:
        raise ValueError("--shard-samples must be divisible by --batch-size")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.model_dir / "tokenizer.model"
    params_path = args.model_dir / "params.json"
    model_path = args.model_dir / "consolidated.00.pth"
    for source in (args.parquet_path, tokenizer_path, params_path, model_path):
        if not source.exists():
            raise FileNotFoundError(source)

    source_fingerprints = {
        "parquet_sha256": sha256_file(args.parquet_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "params_sha256": sha256_file(params_path),
        "model_weight_sha256": (
            None if args.skip_model_weight_hash else sha256_file(model_path)
        ),
        "model_weight_size_bytes": model_path.stat().st_size,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "capturing",
        "created_unix": time.time(),
        "configuration": {
            "model_dir": str(args.model_dir),
            "parquet_path": str(args.parquet_path),
            "layers": layers,
            "num_samples": args.num_samples,
            "max_token_length": args.max_token_length,
            "batch_size": args.batch_size,
            "shard_samples": args.shard_samples,
            "seed": args.seed,
            "dataset_shuffle": True,
            "add_bos_token": False,
            "activation_dtype": args.dtype,
            "sample_id_definition": (
                "zero-based ordinal after deterministic dataset shuffle and truncation"
            ),
            "activation_position": "attention-normalized residual stream at layer input",
        },
        "sources": source_fingerprints,
        "shards": [],
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_json_dump(manifest, manifest_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        set_up_cuda()
    dtype = dtype_from_name(args.dtype)
    tokenizer = Tokenizer(str(tokenizer_path))
    dataset = OpenWebTextSentencesDataset(
        tokenizer=tokenizer,
        max_token_length=args.max_token_length,
        num_samples=args.num_samples,
        shuffle=True,
        add_bos_token=False,
        parquet_path=args.parquet_path,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_args = ModelArgs(**json.loads(params_path.read_text(encoding="utf-8")))
    model = load_model(
        model_path=model_path,
        model_args=model_args,
        store_layer_activ=layers,
        device=device,
        dtype=dtype,
    )

    pending: list[dict[str, Any]] = []
    captured_samples = 0
    captured_tokens = 0
    shard_index = 0
    started = time.time()
    layer_sums = {
        layer: torch.zeros(model_args.dim, dtype=torch.float64)
        for layer in layers
    }

    def flush_pending() -> None:
        nonlocal pending, captured_samples, captured_tokens, shard_index
        if not pending:
            return
        payload = collate_chunks(pending, layers, shard_index)
        meta_path = args.output_dir / f"shard-{shard_index:05d}.meta.pt"
        meta_payload = {
            key: value
            for key, value in payload.items()
            if key != "activations"
        }
        atomic_torch_save(meta_payload, meta_path)
        layer_entries: dict[str, Any] = {}
        for layer in layers:
            layer_path = args.output_dir / f"shard-{shard_index:05d}.layer-{layer}.pt"
            atomic_torch_save(payload["activations"][str(layer)], layer_path)
            layer_entries[str(layer)] = {
                "path": layer_path.name,
                "sha256": sha256_file(layer_path),
                "size_bytes": layer_path.stat().st_size,
            }
        sample_count = int(payload["sample_ids"].numel())
        token_count = int(payload["offsets"][-1].item())
        manifest["shards"].append(
            {
                "meta": {
                    "path": meta_path.name,
                    "sha256": sha256_file(meta_path),
                    "size_bytes": meta_path.stat().st_size,
                },
                "layers": layer_entries,
                "sample_count": sample_count,
                "token_count": token_count,
                "first_sample_id": int(payload["sample_ids"][0].item()),
                "last_sample_id": int(payload["sample_ids"][-1].item()),
            }
        )
        captured_samples += sample_count
        captured_tokens += token_count
        shard_index += 1
        pending = []
        manifest["progress"] = {
            "captured_samples": captured_samples,
            "captured_tokens": captured_tokens,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json_dump(manifest, manifest_path)
        print(
            json.dumps(
                {
                    "event": "shard_saved",
                    "shard": shard_index - 1,
                    "samples": captured_samples,
                    "tokens": captured_tokens,
                    "elapsed_seconds": time.time() - started,
                }
            ),
            flush=True,
        )

    for tokens, sample_ids, lengths in dataloader:
        if (lengths <= 0).any():
            raise ValueError("Encountered an empty tokenized sample")
        tokens_device = tokens.to(device, non_blocking=True)
        with torch.inference_mode():
            model(tokens_device, start_pos=0)
        layer_activations = model.get_layer_residual_activs()
        attention_mask = (
            torch.arange(tokens.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        )
        packed_activations = {
            layer: layer_activations[layer][attention_mask].to(dtype).contiguous()
            for layer in layers
        }
        for layer in layers:
            layer_sums[layer] += packed_activations[layer].double().sum(dim=0)
        pending.append(
            {
                "sample_ids": sample_ids.cpu(),
                "lengths": lengths.cpu(),
                "token_ids": tokens.cpu(),
                "attention_mask": attention_mask.cpu(),
                "activations": packed_activations,
            }
        )
        if sum(int(chunk["sample_ids"].numel()) for chunk in pending) >= args.shard_samples:
            flush_pending()
        del tokens_device, layer_activations, packed_activations

    flush_pending()
    if captured_samples != len(dataset):
        raise RuntimeError(
            f"Captured {captured_samples} samples, expected {len(dataset)}"
        )
    mean_entries: dict[str, Any] = {}
    for layer in layers:
        mean_path = args.output_dir / f"mean-layer-{layer}.pt"
        layer_mean = (layer_sums[layer] / captured_tokens).float()
        atomic_torch_save(layer_mean, mean_path)
        mean_entries[str(layer)] = {
            "path": mean_path.name,
            "sha256": sha256_file(mean_path),
            "size_bytes": mean_path.stat().st_size,
        }
    manifest["layer_means"] = mean_entries
    manifest["status"] = "complete"
    manifest["completed_unix"] = time.time()
    manifest["summary"] = {
        "sample_count": captured_samples,
        "token_count": captured_tokens,
        "average_tokens_per_sample": captured_tokens / captured_samples,
        "shard_count": shard_index,
        "elapsed_seconds": time.time() - started,
        "read_only_finalized": bool(args.finalize_read_only),
    }
    atomic_json_dump(manifest, manifest_path)
    if args.finalize_read_only:
        finalize_read_only(args.output_dir)
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
