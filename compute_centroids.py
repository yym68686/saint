import argparse
import logging
from pathlib import Path

import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser(description="Compute centroids for Archetypal SAE.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing activation data batches.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the computed centroids.")
    parser.add_argument("--n_centroids", type=int, default=4096, help="Number of centroids to compute.")
    parser.add_argument("--sample_fraction", type=float, default=0.2, help="Fraction of data files to sample for clustering.")
    return parser.parse_args()


def main():
    """"""
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

    logging.info(f"Starting centroid computation...")
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Output path: {args.output_path}")
    logging.info(f"Number of centroids: {args.n_centroids}")
    logging.info(f"Sample fraction: {args.sample_fraction}")

    data_files = sorted(list(args.data_dir.glob("*.pt")))
    num_samples = int(len(data_files) * args.sample_fraction)
    sampled_files = torch.randperm(len(data_files))[:num_samples].tolist()

    logging.info(f"Loading {len(sampled_files)} files for clustering...")
    all_activations = []
    for i in tqdm(sampled_files, desc="Loading activation samples"):
        file_path = data_files[i]
        activations = torch.load(file_path, weights_only=True)
        # activations shape is (1, batch_size, d_model), squeeze to (batch_size, d_model)
        all_activations.append(activations.squeeze(0))

    all_activations_tensor = torch.cat(all_activations, dim=0)
    logging.info(f"Total activations loaded for clustering: {all_activations_tensor.shape}")

    logging.info("Fitting MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_centroids,
        random_state=42,
        batch_size=2048,
        n_init="auto",
    )
    kmeans.fit(all_activations_tensor.to(torch.float32).numpy())

    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    logging.info(f"Computed centroids shape: {centroids.shape}")

    logging.info(f"Saving centroids to {args.output_path}...")
    torch.save(centroids, args.output_path)
    logging.info("Centroid computation complete.")


if __name__ == "__main__":
    main()