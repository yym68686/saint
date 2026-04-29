import argparse
import logging
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

IDX_PATTERN = re.compile(r"idx(\d+)\.pt$")


def add_project_root_to_syspath() -> None:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "openwebtext_sentences_dataset.py").exists() and (candidate / "llama_3").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return
    raise RuntimeError("Could not locate project root containing llama_3 and openwebtext_sentences_dataset.py.")


add_project_root_to_syspath()

from llama_3.tokenizer import Tokenizer
from openwebtext_sentences_dataset import OpenWebTextSentencesDataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_activation_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--parquet_path", type=Path, required=True)
    parser.add_argument("--tokenizer_path", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--max_token_length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no_shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--add_bos_token", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_dataset_idx(filepath: Path) -> int:
    match = IDX_PATTERN.search(filepath.name)
    if match is None:
        raise ValueError(f"Could not parse dataset idx from {filepath}.")
    return int(match.group(1))


def build_output_path(output_dir: Path, dataset_idx: int) -> Path:
    return output_dir / f"token_ids_idx{dataset_idx}.pt"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_arguments()
    args.raw_activation_dir = args.raw_activation_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.parquet_path = args.parquet_path.resolve()
    args.tokenizer_path = args.tokenizer_path.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = list(args.raw_activation_dir.glob("*.pt"))
    if not raw_files:
        raise ValueError(f"No raw activation files found in {args.raw_activation_dir}.")

    tokenizer = Tokenizer(str(args.tokenizer_path))
    dataset = OpenWebTextSentencesDataset(
        tokenizer=tokenizer,
        max_token_length=args.max_token_length,
        num_samples=args.num_samples,
        shuffle=args.shuffle,
        add_bos_token=args.add_bos_token,
        parquet_path=args.parquet_path,
        seed=args.seed,
    )

    processed = 0
    skipped = 0
    for filepath in tqdm(raw_files, desc="Building raw token sidecar"):
        dataset_idx = parse_dataset_idx(filepath)
        output_path = build_output_path(args.output_dir, dataset_idx)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        token_ids, rebuilt_idx, _ = dataset[dataset_idx]
        if rebuilt_idx != dataset_idx:
            raise ValueError(f"Dataset idx mismatch: expected {dataset_idx}, got {rebuilt_idx}.")

        activation = torch.load(filepath, weights_only=True)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        if token_tensor.shape[0] != activation.shape[0]:
            raise ValueError(
                "Token length mismatch for "
                f"{filepath.name}: tokens={token_tensor.shape[0]}, activation={activation.shape[0]}.",
            )

        torch.save(token_tensor, output_path)
        processed += 1

    logging.info("Finished. processed=%d skipped=%d total=%d", processed, skipped, len(raw_files))


if __name__ == "__main__":
    main()
