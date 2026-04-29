import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_token_ids_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> list[dict]:
    records = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return records


def token_sidecar_path(raw_token_ids_dir: Path, dataset_idx: int) -> Path:
    return raw_token_ids_dir / f"token_ids_idx{dataset_idx}.pt"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_arguments()
    args.raw_token_ids_dir = args.raw_token_ids_dir.resolve()
    args.manifest_path = args.manifest_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = load_manifest(args.manifest_path)
    written = 0
    skipped = 0

    for record in tqdm(manifest_records, desc="Building batched token sidecar"):
        output_path = args.output_dir / record["activation_batch_name"]
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        pieces = []
        for segment in record["segments"]:
            raw_path = token_sidecar_path(args.raw_token_ids_dir, segment["dataset_idx"])
            token_ids = torch.load(raw_path, weights_only=True)
            piece = token_ids[segment["raw_start"] : segment["raw_end"]].clone()
            expected_len = segment["batch_end"] - segment["batch_start"]
            if piece.shape[0] != expected_len:
                raise ValueError(
                    f"Segment length mismatch for {raw_path}: expected {expected_len}, got {piece.shape[0]}."
                )
            pieces.append(piece)

        batched_token_ids = torch.cat(pieces, dim=0)
        if batched_token_ids.ndim != 1:
            raise ValueError(f"Expected 1D token ids tensor, got shape {tuple(batched_token_ids.shape)}.")
        if batched_token_ids.shape[0] != record["total_tokens"]:
            raise ValueError(
                f"Batch length mismatch for {output_path.name}: expected {record['total_tokens']}, "
                f"got {batched_token_ids.shape[0]}."
            )

        torch.save(batched_token_ids.long(), output_path)
        written += 1

    logging.info("Finished. written=%d skipped=%d total=%d", written, skipped, len(manifest_records))


if __name__ == "__main__":
    main()
