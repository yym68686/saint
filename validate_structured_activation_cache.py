#!/usr/bin/env python3
"""Validate schema, checksums, masks, offsets, and tensor shapes of a cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "structured-activation-cache-v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shard(
    cache_dir: Path,
    expected: dict[str, Any],
    layers: list[int],
    verify_checksum: bool,
) -> dict[str, int]:
    meta_path = cache_dir / expected["meta"]["path"]
    if verify_checksum and sha256_file(meta_path) != expected["meta"]["sha256"]:
        raise ValueError(f"Checksum mismatch: {meta_path}")
    payload = torch.load(meta_path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Schema mismatch in {meta_path}")
    sample_ids = payload["sample_ids"]
    lengths = payload["lengths"]
    offsets = payload["offsets"]
    token_ids = payload["token_ids"]
    attention_mask = payload["attention_mask"]
    sample_count = int(sample_ids.numel())
    token_count = int(offsets[-1].item())
    if sample_count != int(expected["sample_count"]):
        raise ValueError(f"Sample count mismatch in {meta_path}")
    if token_count != int(expected["token_count"]):
        raise ValueError(f"Token count mismatch in {meta_path}")
    if offsets.shape != (sample_count + 1,):
        raise ValueError(f"Offset shape mismatch in {meta_path}")
    if not torch.equal(offsets[1:] - offsets[:-1], lengths.to(torch.int64)):
        raise ValueError(f"Offsets and lengths disagree in {meta_path}")
    if token_ids.shape != attention_mask.shape:
        raise ValueError(f"Token/mask shape mismatch in {meta_path}")
    if token_ids.shape[0] != sample_count:
        raise ValueError(f"Token/sample shape mismatch in {meta_path}")
    if not torch.equal(attention_mask.sum(dim=1).to(lengths.dtype), lengths):
        raise ValueError(f"Attention mask and lengths disagree in {meta_path}")
    if sorted(map(int, expected["layers"])) != layers:
        raise ValueError(f"Layer set mismatch in {meta_path}")
    d_models = set()
    for layer in layers:
        layer_entry = expected["layers"][str(layer)]
        layer_path = cache_dir / layer_entry["path"]
        if verify_checksum and sha256_file(layer_path) != layer_entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {layer_path}")
        tensor = torch.load(layer_path, map_location="cpu", weights_only=True)
        if tensor.ndim != 2 or int(tensor.shape[0]) != token_count:
            raise ValueError(
                f"Activation shape mismatch for layer {layer} in {layer_path}"
            )
        if not torch.isfinite(tensor.float()).all():
            raise FloatingPointError(f"NaN/Inf for layer {layer} in {layer_path}")
        d_models.add(int(tensor.shape[1]))
    if len(d_models) != 1:
        raise ValueError(f"Layer widths disagree in {meta_path}: {d_models}")
    return {
        "samples": sample_count,
        "tokens": token_count,
        "d_model": d_models.pop(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--require-read-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema: {manifest['schema_version']}")
    if manifest["status"] != "complete":
        raise ValueError(f"Cache is not complete: {manifest['status']}")
    layers = list(map(int, manifest["configuration"]["layers"]))

    total_samples = 0
    total_tokens = 0
    d_models = set()
    previous_last_id: int | None = None
    for expected in manifest["shards"]:
        result = validate_shard(
            cache_dir,
            expected,
            layers,
            verify_checksum=not args.skip_checksums,
        )
        if previous_last_id is not None and int(expected["first_sample_id"]) != previous_last_id + 1:
            raise ValueError(
                f"Non-contiguous sample IDs at {expected['meta']['path']}"
            )
        previous_last_id = int(expected["last_sample_id"])
        total_samples += result["samples"]
        total_tokens += result["tokens"]
        d_models.add(result["d_model"])
        if args.require_read_only:
            referenced_paths = [
                cache_dir / expected["meta"]["path"],
                *[
                    cache_dir / expected["layers"][str(layer)]["path"]
                    for layer in layers
                ],
            ]
            for referenced_path in referenced_paths:
                if referenced_path.stat().st_mode & (
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                ):
                    raise PermissionError(f"Writable cache file: {referenced_path}")

    for layer in layers:
        mean_entry = manifest["layer_means"][str(layer)]
        mean_path = cache_dir / mean_entry["path"]
        if not mean_path.exists():
            raise FileNotFoundError(mean_path)
        if not args.skip_checksums and sha256_file(mean_path) != mean_entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {mean_path}")
        layer_mean = torch.load(mean_path, map_location="cpu", weights_only=True)
        if layer_mean.shape != (next(iter(d_models)),):
            raise ValueError(f"Mean shape mismatch: {mean_path}")
        if not torch.isfinite(layer_mean).all():
            raise FloatingPointError(f"NaN/Inf in mean: {mean_path}")

    summary = manifest["summary"]
    if total_samples != int(summary["sample_count"]):
        raise ValueError("Manifest sample total mismatch")
    if total_tokens != int(summary["token_count"]):
        raise ValueError("Manifest token total mismatch")
    if len(d_models) != 1:
        raise ValueError(f"Inconsistent d_model values: {d_models}")
    if args.require_read_only and cache_dir.stat().st_mode & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    ):
        raise PermissionError(f"Writable cache directory: {cache_dir}")

    report = {
        "status": "valid",
        "schema_version": SCHEMA_VERSION,
        "cache_dir": str(cache_dir),
        "layers": layers,
        "sample_count": total_samples,
        "token_count": total_tokens,
        "d_model": d_models.pop(),
        "shard_count": len(manifest["shards"]),
        "checksums_verified": not args.skip_checksums,
        "read_only_required": args.require_read_only,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
