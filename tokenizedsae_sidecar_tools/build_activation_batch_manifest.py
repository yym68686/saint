import argparse
import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm


IDX_PATTERN = re.compile(r"idx(\d+)\.pt$")
BATCH_PATTERN = re.compile(r"batch_(\d+)_(\d+)\.pt$")


@dataclass
class RawRecord:
    path: Path
    dataset_idx: int
    seq_len: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_activation_dir", type=Path, required=True)
    parser.add_argument("--batched_activation_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--order_mode", choices=("auto", "filesystem", "lexical", "numeric"), default="auto")
    parser.add_argument("--validation_batches_per_chunk", type=int, default=2)
    return parser.parse_args()


def parse_dataset_idx(filepath: Path) -> int:
    match = IDX_PATTERN.search(filepath.name)
    if match is None:
        raise ValueError(f"Could not parse dataset idx from {filepath}.")
    return int(match.group(1))


def parse_batch_filename(filepath: Path) -> tuple[str, int]:
    match = BATCH_PATTERN.fullmatch(filepath.name)
    if match is None:
        raise ValueError(f"Unexpected batch filename format: {filepath.name}.")
    pid, batch_idx = match.groups()
    return pid, int(batch_idx)


def tensor_sha1(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha1(contiguous).hexdigest()


def collect_raw_records(raw_dir: Path, order_mode: str) -> list[RawRecord]:
    raw_files = list(raw_dir.rglob("*.pt"))
    if order_mode == "lexical":
        raw_files = sorted(raw_files)
    elif order_mode == "numeric":
        raw_files = sorted(raw_files, key=parse_dataset_idx)

    records = []
    for path in tqdm(raw_files, desc=f"Collecting raw records [{order_mode}]"):
        tensor = torch.load(path, weights_only=True)
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D raw tensor at {path}, got shape {tuple(tensor.shape)}.")
        records.append(
            RawRecord(
                path=path,
                dataset_idx=parse_dataset_idx(path),
                seq_len=tensor.shape[0],
            )
        )
    return records


def group_actual_batches(batched_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for path in batched_dir.glob("*.pt"):
        pid, batch_idx = parse_batch_filename(path)
        grouped.setdefault(pid, []).append((batch_idx, path))

    if not grouped:
        raise ValueError(f"No batched activation files found in {batched_dir}.")

    return {
        pid: [path for _, path in sorted(entries, key=lambda item: item[0])]
        for pid, entries in grouped.items()
    }


def simulate_chunk_batches(
    records: list[RawRecord],
    num_processes: int,
    batch_size: int,
) -> list[list[dict]]:
    chunk_size = math.ceil(len(records) / num_processes)
    chunked_records = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]
    batches_per_chunk: list[list[dict]] = []

    for chunk_id, chunk in enumerate(chunked_records):
        chunk_batches = []
        current_segments = []
        current_fill = 0

        for record in chunk:
            raw_pos = 0
            while raw_pos < record.seq_len:
                take = min(batch_size - current_fill, record.seq_len - raw_pos)
                current_segments.append(
                    {
                        "dataset_idx": record.dataset_idx,
                        "raw_file": str(record.path),
                        "raw_start": raw_pos,
                        "raw_end": raw_pos + take,
                        "batch_start": current_fill,
                        "batch_end": current_fill + take,
                    }
                )
                raw_pos += take
                current_fill += take

                if current_fill == batch_size:
                    chunk_batches.append(
                        {
                            "chunk_id": chunk_id,
                            "local_batch_index": len(chunk_batches),
                            "segments": current_segments,
                            "total_tokens": batch_size,
                        }
                    )
                    current_segments = []
                    current_fill = 0

        batches_per_chunk.append(chunk_batches)

    return batches_per_chunk


def build_simulated_first_batch_tensor(batch_record: dict) -> torch.Tensor:
    pieces = []
    for segment in batch_record["segments"]:
        tensor = torch.load(segment["raw_file"], weights_only=True)
        pieces.append(tensor[segment["raw_start"] : segment["raw_end"]].clone())
    return torch.cat(pieces, dim=0)


def choose_order_mode(
    raw_dir: Path,
    actual_groups: dict[str, list[Path]],
    batch_size: int,
) -> tuple[str, int, int, list[RawRecord], list[list[dict]], dict[int, list[str]]]:
    num_actual_groups = len(actual_groups)
    actual_count_by_pid = {pid: len(files) for pid, files in actual_groups.items()}
    actual_count_counter = Counter(actual_count_by_pid.values())
    actual_first_hashes = {
        pid: tensor_sha1(torch.load(files[0], weights_only=True))
        for pid, files in actual_groups.items()
    }

    for candidate in ("filesystem", "lexical", "numeric"):
        records = collect_raw_records(raw_dir, order_mode=candidate)
        for processes_per_run in range(1, num_actual_groups + 1):
            if num_actual_groups % processes_per_run != 0:
                continue
            repeat_factor = num_actual_groups // processes_per_run
            simulated_batches = simulate_chunk_batches(
                records,
                num_processes=processes_per_run,
                batch_size=batch_size,
            )
            simulated_count_counter = Counter(len(chunk_batches) for chunk_batches in simulated_batches)
            scaled_simulated_count_counter = Counter(
                {count: freq * repeat_factor for count, freq in simulated_count_counter.items()}
            )
            if scaled_simulated_count_counter != actual_count_counter:
                continue

            simulated_first_hashes = {}
            for chunk_id, chunk_batches in enumerate(simulated_batches):
                simulated_first_hashes[chunk_id] = tensor_sha1(build_simulated_first_batch_tensor(chunk_batches[0]))

            mapping: dict[int, list[str]] = {}
            used_pids: set[str] = set()
            for chunk_id, chunk_batches in enumerate(simulated_batches):
                target_count = len(chunk_batches)
                target_hash = simulated_first_hashes[chunk_id]
                matches = sorted(
                    [
                        pid
                        for pid, files in actual_groups.items()
                        if len(files) == target_count
                        and actual_first_hashes[pid] == target_hash
                        and pid not in used_pids
                    ]
                )
                if len(matches) != repeat_factor:
                    break
                mapping[chunk_id] = matches
                used_pids.update(matches)

            if len(mapping) == len(simulated_batches):
                return candidate, processes_per_run, repeat_factor, records, simulated_batches, mapping

    raise RuntimeError("Failed to find a raw file order that matches the existing batched activations.")


def validate_mapping(
    simulated_batches: list[list[dict]],
    chunk_to_pids: dict[int, list[str]],
    actual_groups: dict[str, list[Path]],
    validation_batches_per_chunk: int,
) -> None:
    for chunk_id, chunk_batches in enumerate(simulated_batches):
        sample_indices = {0, len(chunk_batches) - 1}
        while len(sample_indices) < min(validation_batches_per_chunk, len(chunk_batches)):
            sample_indices.add(len(sample_indices))

        for pid in chunk_to_pids[chunk_id]:
            actual_files = actual_groups[pid]
            if len(actual_files) != len(chunk_batches):
                raise ValueError(f"Chunk {chunk_id} batch count mismatch with pid {pid}.")

            for batch_idx in sorted(sample_indices):
                simulated_tensor = build_simulated_first_batch_tensor(chunk_batches[batch_idx])
                actual_tensor = torch.load(actual_files[batch_idx], weights_only=True)
                if not torch.equal(simulated_tensor, actual_tensor):
                    raise ValueError(
                        f"Validation failed for pid={pid} batch_idx={batch_idx}. "
                        "Simulated batch does not equal actual activation batch."
                    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_arguments()
    args.raw_activation_dir = args.raw_activation_dir.resolve()
    args.batched_activation_dir = args.batched_activation_dir.resolve()
    args.manifest_path = args.manifest_path.resolve()
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    actual_groups = group_actual_batches(args.batched_activation_dir)
    num_processes = len(actual_groups)
    logging.info("Detected %d preprocessing process groups from existing batch filenames.", num_processes)

    if args.order_mode == "auto":
        (
            selected_order,
            processes_per_run,
            repeat_factor,
            records,
            simulated_batches,
            chunk_to_pids,
        ) = choose_order_mode(
            raw_dir=args.raw_activation_dir,
            actual_groups=actual_groups,
            batch_size=args.batch_size,
        )
    else:
        records = collect_raw_records(args.raw_activation_dir, order_mode=args.order_mode)
        simulated_batches = simulate_chunk_batches(records, num_processes=num_processes, batch_size=args.batch_size)
        actual_count_counter = Counter(len(files) for files in actual_groups.values())
        simulated_count_counter = Counter(len(chunk_batches) for chunk_batches in simulated_batches)
        if simulated_count_counter != actual_count_counter:
            raise ValueError(
                f"Count mismatch for order={args.order_mode}: simulated={simulated_count_counter} actual={actual_count_counter}"
            )
        selected_order = args.order_mode
        (
            _,
            processes_per_run,
            repeat_factor,
            _,
            _,
            chunk_to_pids,
        ) = choose_order_mode(
            raw_dir=args.raw_activation_dir,
            actual_groups=actual_groups,
            batch_size=args.batch_size,
        )

    logging.info("Selected raw file order: %s", selected_order)
    logging.info("Inferred preprocessing runs: processes_per_run=%d repeat_factor=%d", processes_per_run, repeat_factor)
    validate_mapping(
        simulated_batches=simulated_batches,
        chunk_to_pids=chunk_to_pids,
        actual_groups=actual_groups,
        validation_batches_per_chunk=args.validation_batches_per_chunk,
    )
    logging.info("Validation against existing activation_outputs_batched passed.")

    records_written = 0
    with args.manifest_path.open("w", encoding="utf-8") as f:
        for chunk_id, chunk_batches in enumerate(simulated_batches):
            for pid in chunk_to_pids[chunk_id]:
                for batch_record in chunk_batches:
                    output_record = {
                        "activation_batch_file": str(actual_groups[pid][batch_record["local_batch_index"]].resolve()),
                        "activation_batch_name": actual_groups[pid][batch_record["local_batch_index"]].name,
                        "chunk_id": chunk_id,
                        "process_pid": pid,
                        "local_batch_index": batch_record["local_batch_index"],
                        "total_tokens": batch_record["total_tokens"],
                        "segments": batch_record["segments"],
                    }
                    f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    records_written += 1

    logging.info("Manifest written to %s with %d batch records.", args.manifest_path, records_written)


if __name__ == "__main__":
    main()
