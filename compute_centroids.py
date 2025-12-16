import argparse
import logging
from pathlib import Path

import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import numpy as np

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute centroids for Archetypal SAE.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing activation data.")
    parser.add_argument("--save_path", type=Path, required=True, help="Path to save the computed centroids.")
    parser.add_argument("--n_centroids", type=int, default=4096, help="Number of centroids to compute.")
    parser.add_argument("--sample_files", type=int, default=500, help="Number of files to sample for clustering.")
    return parser.parse_args()

def main():
    """Main function to compute and save centroids."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_arguments()

    logging.info(f"Loading data from {args.data_dir}...")
    data_files = list(args.data_dir.glob("*.pt"))
    if not data_files:
        logging.error(f"No data files found in {args.data_dir}.")
        return

    # Sample a subset of files to avoid memory issues
    sample_files = min(len(data_files), args.sample_files)
    sampled_files = np.random.choice(data_files, sample_files, replace=False)

    logging.info(f"Loading {len(sampled_files)} files for clustering...")
    all_data = []
    for file_path in tqdm(sampled_files, desc="Loading data files"):
        try:
            # .squeeze(0) to remove the batch dimension of 1
            data = torch.load(file_path, weights_only=True).squeeze(0)
            all_data.append(data.cpu().to(torch.float32).numpy())
        except Exception as e:
            logging.warning(f"Could not load file {file_path}: {e}")
            continue
    
    if not all_data:
        logging.error("No data could be loaded. Aborting.")
        return

    all_data = np.concatenate(all_data, axis=0)
    logging.info(f"Data concatenated. Shape: {all_data.shape}")

    logging.info(f"Starting MiniBatchKMeans clustering with {args.n_centroids} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_centroids,
        random_state=42,
        batch_size=2048,
        n_init="auto",
        verbose=1,
    )
    kmeans.fit(all_data)

    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    logging.info(f"Clustering complete. Centroids shape: {centroids.shape}")

    logging.info(f"Saving centroids to {args.save_path}...")
    torch.save(centroids, args.save_path)
    logging.info("Centroids saved successfully.")

if __name__ == "__main__":
    main()