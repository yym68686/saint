import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def plot_activation_diff_scatterplot():
    """
    Parses feature activation summaries for dense and batchtopk models across multiple features,
    and generates a scatter plot to compare their max activation differences for each feature.
    """
    base_dir = Path("ablation_datasets")
    features = {
        "canadian_political": "Canadian Politics",
        "female_subjects": "Female Subjects",
        "football": "Football",
        "indian_politics": "Indian Politics",
        "photo_captions": "Photo Captions"
    }

    plot_data = []

    logging.info("Processing features for activation strength difference scatter plot...")

    for feature_key, feature_name in features.items():
        feature_dir = base_dir / feature_key

        try:
            dense_dir = next(feature_dir.glob("dense-l11-f*"))
            batchtopk_dir = next(feature_dir.glob("batchtopk-l11-f*"))

            dense_json_path = dense_dir / "feature_activation_summary.json"
            batchtopk_json_path = batchtopk_dir / "feature_activation_summary.json"

            with open(dense_json_path, 'r') as f:
                dense_diff = json.load(f)["diff"]["max"]

            with open(batchtopk_json_path, 'r') as f:
                batchtopk_diff = json.load(f)["diff"]["max"]

            plot_data.append({
                "name": feature_name,
                "batchtopk": batchtopk_diff,
                "dense": dense_diff
            })
            logging.info(f"  - Data for '{feature_name}': BatchTopK={batchtopk_diff:.2f}, Dense={dense_diff:.2f}")

        except (StopIteration, FileNotFoundError, KeyError) as e:
            logging.warning(f"Warning: Could not process feature '{feature_key}'. Skipping. Reason: {e}")
            continue

    if not plot_data:
        logging.error("Error: No valid data was collected. Cannot generate plot.")
        return

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    batchtopk_vals = [p['batchtopk'] for p in plot_data]
    dense_vals = [p['dense'] for p in plot_data]

    ax.scatter(batchtopk_vals, dense_vals, s=100, alpha=0.7, edgecolors="k", zorder=3,
               label="Feature Data Point")

    # Annotate points with feature names
    for point in plot_data:
        ax.text(point['batchtopk'], point['dense'] + 0.5, point['name'], ha='center', va='bottom', fontsize=9)

    # Draw y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=1, label='y = x (Equal Strength)')

    ax.set_xlabel("batchtopk (SOTA) Activation Strength Difference", fontsize=12)
    ax.set_ylabel("sae_exp11_dense (Ours) Activation Strength Difference", fontsize=12)
    ax.set_title("Activation Strength Consistency Across 5 Features", fontsize=16, pad=20)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # Save the plot
    output_dir = Path("pic")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "activation_diff_consistency_scatterplot.png"
    plt.savefig(output_path, dpi=300)
    logging.info(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    plot_activation_diff_scatterplot()
