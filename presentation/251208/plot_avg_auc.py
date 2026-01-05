import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_architecture_auc(base_dir, feature_key, arch_name):
    """
    Retrieves the max AUC-ROC for a specific architecture and feature.
    """
    feature_dir = base_dir / feature_key

    # Try to find the directory for the architecture
    try:
        arch_dir = next(feature_dir.glob(f"{arch_name}-l*"))
    except StopIteration:
        return None

    json_path = arch_dir / "feature_activation_summary.json"
    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data["auc_roc"]["max"]
    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"  Warning: Error reading JSON for {arch_name} in {feature_key}: {e}")
        return None

def plot_average_auc(architectures=None):
    """
    Parses feature activation summaries for multiple architectures across features,
    calculates the average AUC-ROC, and generates a bar chart comparing them.
    """

    # Configuration
    if architectures is None:
        architectures = ['topk', 'dense', 'sigreg', 'batchtopk', 'relu', 'jumprelu', 'gatedsae']

    # Handle directory search
    base_dir_candidates = [Path("ablation_datasets-dense"), Path("ablation_datasets")]
    base_dir = None
    for d in base_dir_candidates:
        if d.exists():
            base_dir = d
            break

    if base_dir is None:
        logging.error("Error: Could not find 'ablation_datasets-dense' or 'ablation_datasets' directory.")
        return

    logging.info(f"Using base directory: {base_dir}")
    logging.info(f"Comparing Architectures: {architectures}")

    features = [
        "canadian_political",
        "female_subjects",
        "football",
        "indian_politics",
        "photo_captions"
    ]

    # Collect data
    arch_data = {arch: [] for arch in architectures}

    logging.info("Processing features for AUC comparison...")

    for feature in features:
        logging.info(f"  Processing {feature}...")
        for arch in architectures:
            auc_val = get_architecture_auc(base_dir, feature, arch)
            if auc_val is not None:
                arch_data[arch].append(auc_val)
                logging.info(f"    - {arch}: {auc_val:.4f}")
            else:
                logging.debug(f"    - {arch}: missing")

    # Calculate averages and filter out architectures with no data
    valid_archs = []
    avg_aucs = []

    logging.info("\n--- Results ---")
    for arch in architectures:
        vals = arch_data[arch]
        if vals:
            avg = np.mean(vals)
            valid_archs.append(arch)
            avg_aucs.append(avg)
            logging.info(f"{arch}: Average AUC = {avg:.4f} (from {len(vals)} features)")
        else:
            logging.warning(f"{arch}: No valid data found. Skipping.")

    if not valid_archs:
        logging.error("Error: No valid data was collected for any architecture. Cannot generate plot.")
        return

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map
    colors = plt.cm.get_cmap('tab10', len(valid_archs))
    bar_colors = [colors(i) for i in range(len(valid_archs))]

    bars = ax.bar(valid_archs, avg_aucs, color=bar_colors, width=0.6)

    ax.set_ylabel("Average AUC-ROC", fontsize=14)
    ax.set_title(f"Average Feature Recognition Performance (Across available features)", fontsize=16, pad=20)

    # Zoom in on the top values to highlight differences if they are close
    if avg_aucs:
        min_val = min(avg_aucs)
        max_val = max(avg_aucs)
        # If values are very high (like AUC usually is), zoom in
        if min_val > 0.9:
            lower_bound = max(0, min_val - 0.005) # Zoom tight if all are high
            upper_bound = min(1.0, max_val + 0.002)
            ax.set_ylim(lower_bound, upper_bound)
        else:
            ax.set_ylim(0, 1.05)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        # Position text slightly above bar, dynamic offset based on scale
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f'{yval:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(fontsize=12, rotation=15 if len(valid_archs) > 4 else 0)
    plt.yticks(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()

    # Save the plot
    output_dir = Path("presentation/251208/pic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "average_auc_roc_comparison.png"
    plt.savefig(output_path, dpi=300)
    logging.info(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    # Default list excluding 'topk' initially to match current environment status
    # but keeping the full structure logic available
    # python3 presentation/251208/plot_avg_auc.py
    default_archs = ['dense', 'sigreg', 'leech', 'batchtopk', 'relu', 'jumprelu', 'gatedsae']

    parser = argparse.ArgumentParser(description="Plot average AUC for multiple architectures.")
    parser.add_argument('--archs', nargs='+', default=default_archs,
                        help="List of architectures to compare.")

    args = parser.parse_args()

    plot_average_auc(args.archs)