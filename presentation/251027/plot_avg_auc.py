import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def plot_average_auc():
    """
    Parses feature activation summaries for dense and batchtopk models across multiple features,
    calculates the average AUC-ROC, and generates a bar chart comparing them.
    """
    base_dir = Path("ablation_datasets")
    features = [
        "canadian_political",
        "female_subjects",
        "football",
        "indian_politics",
        "photo_captions"
    ]

    dense_aucs = []
    batchtopk_aucs = []

    logging.info("Processing features...")

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
                # Using 'max' provides a good measure of peak classification performance
                auc_value = dense_data["auc_roc"]["max"]
                dense_aucs.append(auc_value)
                logging.info(f"  - Found Dense AUC for {feature}: {auc_value:.4f}")

            # Load and parse batchtopk model's summary
            with open(batchtopk_json_path, 'r') as f:
                batchtopk_data = json.load(f)
                auc_value = batchtopk_data["auc_roc"]["max"]
                batchtopk_aucs.append(auc_value)
                logging.info(f"  - Found BatchTopK AUC for {feature}: {auc_value:.4f}")

        except StopIteration:
            logging.warning(f"Warning: Could not find dense or batchtopk directories for feature '{feature}'. Skipping.")
            continue
        except FileNotFoundError as e:
            logging.warning(f"Warning: Could not find summary file for feature '{feature}'. Skipping. Details: {e}")
            continue
        except KeyError:
            logging.error(f"Error: 'auc_roc' or 'max' key not found in JSON for {feature}. Skipping.")
            continue


    if not dense_aucs or not batchtopk_aucs:
        logging.error("Error: No valid data was collected. Cannot generate plot.")
        return

    # Calculate averages
    avg_dense_auc = np.mean(dense_aucs)
    avg_batchtopk_auc = np.mean(batchtopk_aucs)

    logging.info("\n--- Averages ---")
    logging.info(f"Average Dense AUC ('sae_exp11_dense'): {avg_dense_auc:.4f}")
    logging.info(f"Average BatchTopK AUC ('batchtopk'): {avg_batchtopk_auc:.4f}")

    # Plotting
    labels = ["sae_exp11_dense (Ours)", "batchtopk (SOTA)"]
    values = [avg_dense_auc, avg_batchtopk_auc]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(labels, values, color=["#4c72b0", "#55a868"], width=0.5)
    ax.set_ylabel("Average AUC-ROC", fontsize=14)
    ax.set_title("Average Feature Recognition Performance (5 Features)", fontsize=16, pad=20)

    # Zoom in on the top values to highlight small (or no) differences
    # Dynamically set y-axis limits based on data
    min_val = min(values)
    max_val = max(values)
    ax.set_ylim(max(0, min_val - 0.01), min(1.0, max_val + 0.01))

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0005, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.tight_layout()

    # Save the plot
    output_dir = Path("pic")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "average_auc_roc_comparison.png"
    plt.savefig(output_path, dpi=300)
    logging.info(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    plot_average_auc()
