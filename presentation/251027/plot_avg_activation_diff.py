import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def plot_average_activation_diff():
    """
    Parses feature activation summaries for dense and batchtopk models across multiple features,
    calculates the average of the max activation difference, and generates a bar chart comparing them.
    """
    base_dir = Path("ablation_datasets")
    features = [
        "canadian_political",
        "female_subjects",
        "football",
        "indian_politics",
        "photo_captions"
    ]

    dense_diffs = []
    batchtopk_diffs = []

    logging.info("Processing features for activation strength difference...")

    for feature in features:
        feature_dir = base_dir / feature

        try:
            # Find the specific dense and batchtopk directories using glob
            dense_dir = next(feature_dir.glob("dense-l11-f*"))
            batchtopk_dir = next(feature_dir.glob("batchtopk-l11-f*"))

            dense_json_path = dense_dir / "feature_activation_summary.json"
            batchtopk_json_path = batchtopk_dir / "feature_activation_summary.json"

            # Load and parse dense model's summary
            with open(dense_json_path, 'r') as f:
                dense_data = json.load(f)
                diff_value = dense_data["diff"]["max"]
                dense_diffs.append(diff_value)
                logging.info(f"  - Found Dense Diff (max) for {feature}: {diff_value:.2f}")

            # Load and parse batchtopk model's summary
            with open(batchtopk_json_path, 'r') as f:
                batchtopk_data = json.load(f)
                diff_value = batchtopk_data["diff"]["max"]
                batchtopk_diffs.append(diff_value)
                logging.info(f"  - Found BatchTopK Diff (max) for {feature}: {diff_value:.2f}")

        except StopIteration:
            logging.warning(f"Warning: Could not find dense or batchtopk directories for feature '{feature}'. Skipping.")
            continue
        except FileNotFoundError as e:
            logging.warning(f"Warning: Could not find summary file for feature '{feature}'. Skipping. Details: {e}")
            continue
        except KeyError:
            logging.error(f"Error: 'diff' or 'max' key not found in JSON for {feature}. Skipping.")
            continue

    if not dense_diffs or not batchtopk_diffs:
        logging.error("Error: No valid data was collected. Cannot generate plot.")
        return

    # Calculate averages
    avg_dense_diff = np.mean(dense_diffs)
    avg_batchtopk_diff = np.mean(batchtopk_diffs)

    logging.info("\n--- Averages ---")
    logging.info(f"Average Dense Activation Diff (max): {avg_dense_diff:.2f}")
    logging.info(f"Average BatchTopK Activation Diff (max): {avg_batchtopk_diff:.2f}")

    # Plotting
    labels = ["sae_exp11_dense (Ours)", "batchtopk (SOTA)"]
    values = [avg_dense_diff, avg_batchtopk_diff]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(labels, values, color=["#4c72b0", "#55a868"], width=0.5)
    ax.set_ylabel("Average Activation Strength Difference\n(Target vs Control)", fontsize=14)
    ax.set_title("Average Feature Activation Strength (5 Features)", fontsize=16, pad=20)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.tight_layout()

    # Save the plot
    output_dir = Path("pic")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "average_activation_diff_comparison.png"
    plt.savefig(output_path, dpi=300)
    logging.info(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    plot_average_activation_diff()
