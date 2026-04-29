import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from plot_working_point_benchmark import (
    DISPLAY_NAMES,
    GROUPS,
    METHOD_COLORS,
    build_per_concept_table,
    summarize_group,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = REPO_ROOT.parent / "Thesis"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "presentation/260415"
DEFAULT_SUMMARY_LONG = DEFAULT_OUTPUT_DIR / "tables/working_point_summary_long.csv"
DEFAULT_IDEA1_TABLE = THESIS_ROOT / "exp/idea1-dense-success/table.tex"
DEFAULT_IDEA5_TABLE = THESIS_ROOT / "exp/idea5-kernel/table.tex"

SBTR_GROUP = "key_mainstream"
SBTR_ALPHA = 2
HTCC_METHODS = ["kernel", "dense", "relu", "gatedsae", "batchtopk", "topk", "jumprelu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot two focused control metrics: SBTR@2% and HTCC@0.95."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-long", type=Path, default=DEFAULT_SUMMARY_LONG)
    parser.add_argument("--idea1-table", type=Path, default=DEFAULT_IDEA1_TABLE)
    parser.add_argument("--idea5-table", type=Path, default=DEFAULT_IDEA5_TABLE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def display_name_for_method(method: str) -> str:
    if method == "topk":
        return "TopK SAE"
    return DISPLAY_NAMES.get(method, method)


def load_or_build_summary_long(summary_long_path: Path, seed: int) -> pd.DataFrame:
    if summary_long_path.exists():
        return pd.read_csv(summary_long_path)

    logging.info("Missing %s, rebuilding working-point summary from raw data.", summary_long_path)
    per_concept_df = build_per_concept_table(base_dir=REPO_ROOT / "ablation_datasets-dense", seed=seed)
    summary_frames = [
        summarize_group(per_concept_df=per_concept_df, group_name=group_name, seed=seed)
        for group_name in GROUPS
    ]
    return pd.concat(summary_frames, ignore_index=True)


def clean_tex(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace(r"\\", "")
    while True:
        updated = re.sub(r"\\(?:textbf|underline)\{([^{}]+)\}", r"\1", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = cleaned.replace("{", "").replace("}", "")
    return cleaned.strip()


def parse_htcc_from_table(table_path: Path) -> pd.DataFrame:
    method_rows: list[dict] = []

    with table_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if "L22" not in line or "&" not in line:
                continue

            columns = [clean_tex(part) for part in line.split("&")]
            if len(columns) < 7:
                continue

            label = columns[0]
            ehfc_val = float(columns[6])
            method = None

            if label.startswith("Topk 2024"):
                method = "topk"
            elif label.startswith("BatchTopK 2024"):
                method = "batchtopk"
            elif label.startswith("ReluSAE 2023"):
                method = "relu"
            elif label.startswith("GatedSAE 2024"):
                method = "gatedsae"
            elif label.startswith("JumpReLU 2024"):
                method = "jumprelu"
            elif label.startswith("Dense (ours"):
                method = "dense"
            elif label.startswith("Kernel (ours"):
                method = "kernel"

            if method is None:
                continue

            method_rows.append(
                {
                    "method": method,
                    "display_name": display_name_for_method(method),
                    "htcc_095": ehfc_val,
                    "source_table": str(table_path),
                }
            )

    df = pd.DataFrame.from_records(method_rows)
    df = df.drop_duplicates(subset=["method"], keep="last")
    return df


def build_htcc_table(idea1_table: Path, idea5_table: Path) -> pd.DataFrame:
    combined = pd.concat(
        [
            parse_htcc_from_table(idea1_table),
            parse_htcc_from_table(idea5_table),
        ],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset=["method"], keep="last")
    combined = combined[combined["method"].isin(HTCC_METHODS)].copy()
    combined["sort_key"] = combined["method"].apply(lambda method: HTCC_METHODS.index(method))
    combined = combined.sort_values("htcc_095", ascending=False).reset_index(drop=True)
    return combined.drop(columns=["sort_key"])


def add_bar_labels(ax: plt.Axes, x_positions: np.ndarray, heights: np.ndarray, is_percent: bool) -> None:
    for x_pos, height in zip(x_positions, heights):
        label = f"{height:.1%}" if is_percent else f"{height:.1f}"
        ax.text(
            x_pos,
            height + (0.015 if is_percent else 0.8),
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
            fontweight="semibold",
        )


def plot_sbtr(summary_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    sbtr_df = summary_df[
        (summary_df["group"] == SBTR_GROUP) &
        (summary_df["alpha_percent"] == SBTR_ALPHA)
    ].copy()
    sbtr_df = sbtr_df.sort_values("mean_target_reject_rate", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    x = np.arange(len(sbtr_df))
    heights = sbtr_df["mean_target_reject_rate"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            heights - sbtr_df["target_reject_ci_low"].to_numpy(dtype=float),
            sbtr_df["target_reject_ci_high"].to_numpy(dtype=float) - heights,
        ]
    )
    colors = [
        METHOD_COLORS.get(method, "#777777") if method == "dense" else "#D7DCE2"
        for method in sbtr_df["method"]
    ]
    edges = [
        "#1f1f1f" if method == "dense" else "#7F8790"
        for method in sbtr_df["method"]
    ]

    ax.bar(
        x,
        heights,
        yerr=yerr,
        color=colors,
        edgecolor=edges,
        linewidth=1.2,
        width=0.68,
        error_kw={
            "elinewidth": 1.6,
            "ecolor": "#1f1f1f",
            "capsize": 4,
            "capthick": 1.6,
        },
    )

    add_bar_labels(ax, x, heights, is_percent=True)

    ax.set_xticks(x)
    ax.set_xticklabels(sbtr_df["display_name"], rotation=16, ha="right")
    ax.set_ylabel("Average target reject rate")
    ax.set_xlabel("Method")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    # Leave a small margin below 0 so intervals with a true lower bound of 0 remain visible.
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Strict-Budget Trigger Rate (SBTR@2%)\nShared concepts: football, indian_politics, photo_captions",
        fontsize=14,
        pad=14,
    )

    fig.text(
        0.5,
        0.02,
        "Control reject target = 2%. Shared-concept set size n = 3. Error bars show 95% bootstrap confidence intervals.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return sbtr_df[
        [
            "group",
            "display_name",
            "method",
            "alpha_percent",
            "mean_target_reject_rate",
            "target_reject_ci_low",
            "target_reject_ci_high",
            "mean_realized_control_reject_rate",
            "n_concepts",
        ]
    ].rename(
        columns={
            "mean_target_reject_rate": "sbtr_2",
            "target_reject_ci_low": "sbtr_2_ci_low",
            "target_reject_ci_high": "sbtr_2_ci_high",
            "mean_realized_control_reject_rate": "mean_realized_control_2",
        }
    )


def plot_htcc(htcc_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    x = np.arange(len(htcc_df))
    heights = htcc_df["htcc_095"].to_numpy(dtype=float)

    colors = []
    alphas = []
    edges = []
    for method in htcc_df["method"]:
        if method == "kernel":
            colors.append(METHOD_COLORS.get(method, "#777777"))
            alphas.append(0.95)
            edges.append("#1f1f1f")
        elif method == "dense":
            colors.append(METHOD_COLORS.get(method, "#777777"))
            alphas.append(0.92)
            edges.append("#1f1f1f")
        else:
            colors.append("#D7DCE2")
            alphas.append(0.9)
            edges.append("#7F8790")

    bars = ax.bar(
        x,
        heights,
        width=0.72,
        color=colors,
        edgecolor=edges,
        linewidth=1.2,
    )
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    add_bar_labels(ax, x, heights, is_percent=False)

    ax.set_xticks(x)
    ax.set_xticklabels(htcc_df["display_name"], rotation=16, ha="right")
    ax.set_ylabel("High-trust candidate coverage")
    ax.set_xlabel("Method")
    ax.set_ylim(0.0, max(heights) + 6.0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "High-Trust Trigger Candidate Coverage (HTCC@0.95)\nL22 feature statistics; higher means a larger high-confidence trigger pool",
        fontsize=14,
        pad=14,
    )

    fig.text(
        0.5,
        0.02,
        "This is EHFC@0.95 renamed for control-facing presentation. Values come from the chapter tables at layer L22.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return htcc_df


def build_report(sbtr_df: pd.DataFrame, htcc_df: pd.DataFrame, output_path: Path) -> None:
    sbtr_best = sbtr_df.iloc[0]
    htcc_best = htcc_df.iloc[0]

    lines = [
        "# Focused Control Metrics",
        "",
        "## Metric 1: SBTR@2%",
        "- Definition: average target reject rate at a calibrated control reject target of 2%.",
        "- Purpose: emphasize strict-budget trigger quality, which is the control-facing advantage expected from PLRDC SAE.",
        f"- Current best: {sbtr_best['display_name']} = {float(sbtr_best['sbtr_2']):.1%}.",
        "",
        "## Metric 2: HTCC@0.95",
        "- Definition: high-trust trigger candidate coverage, equal to EHFC@0.95 at layer L22.",
        "- Purpose: emphasize how many high-confidence trigger candidates a method can provide, which is the control-facing advantage expected from SUR SAE.",
        f"- Current best: {htcc_best['display_name']} = {float(htcc_best['htcc_095']):.1f}.",
        "",
        "## Output Files",
        f"- SBTR figure: `{output_path.parent / 'pic' / 'sbtr_at_2_plrdc.png'}`",
        f"- HTCC figure: `{output_path.parent / 'pic' / 'htcc_at_095_sur.png'}`",
        f"- SBTR table: `{output_path.parent / 'tables' / 'sbtr_at_2_focus.csv'}`",
        f"- HTCC table: `{output_path.parent / 'tables' / 'htcc_at_095_focus.csv'}`",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    output_dir = args.output_dir.resolve()
    pic_dir = output_dir / "pic"
    table_dir = output_dir / "tables"
    pic_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_or_build_summary_long(args.summary_long.resolve(), seed=args.seed)
    htcc_df = build_htcc_table(args.idea1_table.resolve(), args.idea5_table.resolve())

    sbtr_export = plot_sbtr(summary_df, pic_dir / "sbtr_at_2_plrdc.png")
    htcc_export = plot_htcc(htcc_df, pic_dir / "htcc_at_095_sur.png")

    sbtr_export.to_csv(table_dir / "sbtr_at_2_focus.csv", index=False)
    htcc_export.to_csv(table_dir / "htcc_at_095_focus.csv", index=False)

    build_report(sbtr_export, htcc_export, output_dir / "focused_control_metrics_report.md")
    logging.info("Generated focused control metric figures under %s", output_dir)


if __name__ == "__main__":
    main()
