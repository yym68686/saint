import argparse
import logging
import math
import multiprocessing as mp
import os
import re
from functools import partial
from pathlib import Path

import torch


IDX_PATTERN = re.compile(r"idx(\d+)\.pt$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_activation_dir", type=Path, required=True)
    parser.add_argument("--raw_token_ids_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mean_filepath", type=Path, required=True)
    parser.add_argument("--num_processes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--order_mode", choices=("filesystem", "lexical", "numeric"), default="filesystem")
    return parser.parse_args()


def parse_dataset_idx(filepath: Path) -> int:
    match = IDX_PATTERN.search(filepath.name)
    if match is None:
        raise ValueError(f"Could not parse dataset idx from {filepath}.")
    return int(match.group(1))


def collect_input_files(raw_activation_dir: Path, order_mode: str) -> list[Path]:
    files = list(raw_activation_dir.rglob("*.pt"))
    if order_mode == "lexical":
        files = sorted(files)
    elif order_mode == "numeric":
        files = sorted(files, key=parse_dataset_idx)
    return files


class MeanAccumulator:
    def __init__(self, shape: int, dtype: torch.dtype = torch.float64):
        self.count = 0
        self.mean = torch.zeros(shape, dtype=dtype)

    def update(self, tensor: torch.Tensor) -> None:
        self.count += tensor.shape[0]
        tensor = tensor.to(self.mean.dtype)
        delta = tensor - self.mean
        self.mean += delta.sum(dim=0) / self.count

    def merge(self, other: "MeanAccumulator") -> None:
        total_count = self.count + other.count
        delta = other.mean - self.mean
        self.mean += delta * (other.count / total_count)
        self.count = total_count


def load_pair(raw_activation_path: Path, raw_token_ids_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_idx = parse_dataset_idx(raw_activation_path)
    activation = torch.load(raw_activation_path, weights_only=True)
    token_path = raw_token_ids_dir / f"token_ids_idx{dataset_idx}.pt"
    token_ids = torch.load(token_path, weights_only=True)
    if activation.shape[0] != token_ids.shape[0]:
        raise ValueError(
            f"Length mismatch for idx={dataset_idx}: activations={activation.shape[0]} tokens={token_ids.shape[0]}"
        )
    return activation, token_ids.long()


def process_chunk(
    input_files: list[Path],
    raw_token_ids_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> tuple[MeanAccumulator, int, int]:
    pid = os.getpid()
    logging.info("[PID %s] Processing %d files", pid, len(input_files))

    mean_acc = None
    carry_activation = torch.empty(0)
    carry_token_ids = torch.empty(0, dtype=torch.long)
    output_count = 0

    for raw_activation_path in input_files:
        activation, token_ids = load_pair(raw_activation_path, raw_token_ids_dir)
        if mean_acc is None:
            mean_acc = MeanAccumulator(shape=activation.shape[1])
        mean_acc.update(activation)

        if carry_activation.numel() > 0:
            activation = torch.cat([carry_activation, activation], dim=0)
            token_ids = torch.cat([carry_token_ids, token_ids], dim=0)

        if activation.shape[0] != token_ids.shape[0]:
            raise ValueError(f"Carry-over mismatch at {raw_activation_path}")

        num_full_batches = activation.shape[0] // batch_size
        for batch_idx in range(num_full_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch = {
                "activations": activation[start:end].clone(),
                "token_ids": token_ids[start:end].clone(),
            }
            torch.save(batch, output_dir / f"batch_{pid}_{output_count}.pt")
            output_count += 1

        carry_rows = activation.shape[0] % batch_size
        if carry_rows > 0:
            carry_activation = activation[-carry_rows:].clone()
            carry_token_ids = token_ids[-carry_rows:].clone()
        else:
            carry_activation = torch.empty(0)
            carry_token_ids = torch.empty(0, dtype=torch.long)

    discarded = carry_activation.shape[0]
    logging.info("[PID %s] Finished. batches=%d discarded_rows=%d", pid, output_count, discarded)
    return mean_acc, discarded, output_count


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_arguments()
    args.raw_activation_dir = args.raw_activation_dir.resolve()
    args.raw_token_ids_dir = args.raw_token_ids_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.mean_filepath = args.mean_filepath.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mean_filepath.parent.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files(args.raw_activation_dir, args.order_mode)
    if not input_files:
        raise ValueError(f"No raw activation files found in {args.raw_activation_dir}.")

    chunk_size = math.ceil(len(input_files) / args.num_processes)
    chunks = [input_files[i : i + chunk_size] for i in range(0, len(input_files), chunk_size)]
    worker = partial(
        process_chunk,
        raw_token_ids_dir=args.raw_token_ids_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    with mp.Pool(processes=args.num_processes) as pool:
        results = pool.map(worker, chunks)

    mean_acc = results[0][0]
    discarded_total = results[0][1]
    batch_total = results[0][2]
    for chunk_mean, discarded, batch_count in results[1:]:
        mean_acc.merge(chunk_mean)
        discarded_total += discarded
        batch_total += batch_count

    torch.save(mean_acc.mean, args.mean_filepath)
    logging.info("Saved paired mean tensor to %s", args.mean_filepath)
    logging.info(
        "Done. total_batches=%d discarded_rows=%d total_rows=%d",
        batch_total,
        discarded_total,
        mean_acc.count,
    )


if __name__ == "__main__":
    main()
