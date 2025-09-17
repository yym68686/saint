import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def create_ablation_datasets(
    dataset_path: Path,
    target_keywords: list[str],
    num_target_samples: int,
    num_control_samples: int,
    output_dir: Path,
):
    """
    Creates and saves target and control datasets for feature ablation experiments.

    Args:
        dataset_path: Path to the local Parquet dataset file.
        target_keywords: A list of keywords to identify sentences for the target dataset.
        num_target_samples: The number of samples to collect for the target dataset.
        num_control_samples: The number of samples to collect for the control dataset.
        output_dir: The directory to save the output JSONL files.
    """
    logging.info(f"Loading dataset from local path: {dataset_path}")
    dataset = load_dataset("parquet", data_files={"train": str(dataset_path)}, split="train")

    target_samples = []
    control_samples = []

    # Create a regex pattern for case-insensitive matching of any keyword
    keyword_pattern = "|".join(target_keywords)

    logging.info("Iterating through dataset to find samples...")
    pbar = tqdm(total=num_target_samples + num_control_samples, desc="Scanning dataset")

    for item in dataset:
        text = item["text"]
        # Case-insensitive search for keywords
        if pd.Series(text).str.contains(keyword_pattern, case=False, regex=True).any():
            if len(target_samples) < num_target_samples:
                target_samples.append({"text": text})
                pbar.update(1)
        else:
            if len(control_samples) < num_control_samples:
                control_samples.append({"text": text})
                pbar.update(1)

        if len(target_samples) >= num_target_samples and len(control_samples) >= num_control_samples:
            break

    pbar.close()

    if len(target_samples) < num_target_samples:
        logging.warning(f"Could only find {len(target_samples)} target samples, requested {num_target_samples}.")
    if len(control_samples) < num_control_samples:
        logging.warning(f"Could only find {len(control_samples)} control samples, requested {num_control_samples}.")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets to JSONL files
    target_path = output_dir / "target_dataset.jsonl"
    control_path = output_dir / "control_dataset.jsonl"

    pd.DataFrame(target_samples).to_json(target_path, orient="records", lines=True)
    logging.info(f"Saved {len(target_samples)} samples to {target_path}")

    pd.DataFrame(control_samples).to_json(control_path, orient="records", lines=True)
    logging.info(f"Saved {len(control_samples)} samples to {control_path}")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Create datasets for feature ablation experiments.")
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to the local Parquet dataset file.",
    )
    parser.add_argument(
        "--target_keywords",
        nargs="+",
        required=True,
        help="List of keywords to identify target sentences.",
    )
    parser.add_argument(
        "--num_target_samples",
        type=int,
        default=200,
        help="Number of samples for the target dataset.",
    )
    parser.add_argument(
        "--num_control_samples",
        type=int,
        default=200,
        help="Number of samples for the control dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./ablation_datasets"),
        help="Directory to save the output datasets.",
    )
    return parser.parse_args()


def main():
    """Main function to run the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_arguments()

    logging.info("Starting dataset creation...")
    logging.info(f"  Target Keywords: {args.target_keywords}")
    logging.info(f"  Num Target Samples: {args.num_target_samples}")
    logging.info(f"  Num Control Samples: {args.num_control_samples}")
    logging.info(f"  Output Directory: {args.output_dir.resolve()}")

    create_ablation_datasets(
        dataset_path=args.dataset_path,
        target_keywords=args.target_keywords,
        num_target_samples=args.num_target_samples,
        num_control_samples=args.num_control_samples,
        output_dir=args.output_dir,
    )
    logging.info("Dataset creation finished.")


if __name__ == "__main__":
    main()