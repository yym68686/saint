import argparse
import logging
import math
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


SCORE_AGG = "max"
ALPHAS = [1, 2, 5, 10, 20]
DEFAULT_BASE_DIR = Path("ablation_datasets-dense")
DEFAULT_OUTPUT_DIR = Path("presentation/260415")
BOOTSTRAP_SAMPLES = 10_000
CALIBRATION_FRACTION = 0.5

DISPLAY_NAMES = {
    "batchtopk": "BatchTopK SAE",
    "dense": "PLRDC SAE",
    "denseimq": "PLRDC + IMQ",
    "denserepreg": "PLRDC + RepReg",
    "gatedsae": "Gated SAE",
    "imq": "IMQ SAE",
    "jumprelu": "JumpReLU SAE",
    "kernel": "SUR SAE",
    "relu": "ReLU SAE",
    "repreg": "RepReg SAE",
    "sigreg": "SigReg SAE",
    "sigregrepreg": "SigReg + RepReg",
}

METHOD_COLORS = {
    "batchtopk": "#E69F00",
    "dense": "#0072B2",
    "denseimq": "#56B4E9",
    "denserepreg": "#009E73",
    "gatedsae": "#D55E00",
    "imq": "#CC79A7",
    "jumprelu": "#7F7F7F",
    "kernel": "#6A3D9A",
    "relu": "#009E73",
    "repreg": "#F28E2B",
    "sigreg": "#4E79A7",
    "sigregrepreg": "#76B7B2",
}

LEGEND_METHOD_ORDER = ["kernel", "dense", "batchtopk", "relu", "gatedsae"]

GROUPS = {
    "key_mainstream": {
        "title": "PLRDC vs Mainstream Methods",
        "subtitle": "Shared concepts: football, indian_politics, photo_captions",
        "concepts": ["football", "indian_politics", "photo_captions"],
        "methods": ["dense", "batchtopk", "relu", "gatedsae"],
    },
    "key_sur": {
        "title": "SUR vs Mainstream Methods",
        "subtitle": "Shared concepts: football, indian_politics",
        "concepts": ["football", "indian_politics"],
        "methods": ["kernel", "dense", "batchtopk", "relu", "gatedsae"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fixed-control working-point metrics and plots for ablation_datasets-dense."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_seed(name: str, seed: int) -> int:
    return (seed + zlib.crc32(name.encode("utf-8"))) % (2**32)


def calculate_auc(target_scores: np.ndarray, control_scores: np.ndarray) -> float:
    if target_scores.size == 0 or control_scores.size == 0:
        return 0.0

    all_scores = sorted(
        [(float(score), 1) for score in target_scores] + [(float(score), 0) for score in control_scores]
    )

    rank_sum = 0.0
    i = 0
    while i < len(all_scores):
        j = i
        while j < len(all_scores) and all_scores[j][0] == all_scores[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if all_scores[k][1] == 1:
                rank_sum += avg_rank
        i = j

    n_target = target_scores.size
    n_control = control_scores.size
    u_statistic = rank_sum - (n_target * (n_target + 1)) / 2.0
    return float(u_statistic / (n_target * n_control))


def midpoint_threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores.astype(float))
    unique_scores.sort()
    candidates = [math.inf]
    if unique_scores.size == 1:
        candidates.append(unique_scores[0] - 1e-9)
        return np.array(candidates, dtype=float)

    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates.extend(midpoints.tolist())
    candidates.append(unique_scores[0] - 1e-9)
    return np.array(candidates, dtype=float)


def choose_threshold(control_scores: np.ndarray, alpha_fraction: float) -> tuple[float, float]:
    best_tau = math.inf
    best_rate = 0.0
    best_key = None

    for tau in midpoint_threshold_candidates(control_scores):
        rate = float(np.mean(control_scores > tau))
        key = (abs(rate - alpha_fraction), rate > alpha_fraction, -tau)
        if best_key is None or key < best_key:
            best_tau = float(tau)
            best_rate = rate
            best_key = key

    return best_tau, best_rate


def split_control_indices(concept: str, n_control: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(concept, seed))
    perm = rng.permutation(n_control)
    calib_count = max(1, int(round(n_control * CALIBRATION_FRACTION)))
    calib_idx = np.sort(perm[:calib_count])
    eval_idx = np.sort(perm[calib_count:])
    if eval_idx.size == 0:
        raise ValueError(f"No evaluation controls remain for concept {concept}.")
    return calib_idx, eval_idx


def locate_feature_dir(base_dir: Path, concept: str, method: str) -> Path:
    matches = sorted((base_dir / concept).glob(f"{method}-l*/feature_activation_summary.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing feature_activation_summary.csv for concept={concept}, method={method}")
    return matches[0].parent


def load_method_concept_scores(base_dir: Path, concept: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    feature_dir = locate_feature_dir(base_dir, concept, method)
    df = pd.read_csv(feature_dir / "feature_activation_summary.csv")
    df = df[df["agg"] == SCORE_AGG].copy()
    df["index"] = df["index"].astype(int)
    df["val"] = df["val"].astype(float)

    target_scores = (
        df[df["group"] == "target"]
        .sort_values("index")["val"]
        .to_numpy(dtype=float)
    )
    control_scores = (
        df[df["group"] == "control"]
        .sort_values("index")["val"]
        .to_numpy(dtype=float)
    )
    return target_scores, control_scores


def build_per_concept_table(base_dir: Path, seed: int) -> pd.DataFrame:
    records: list[dict] = []

    for concept_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        concept = concept_dir.name
        method_dirs = sorted(
            path for path in concept_dir.iterdir()
            if path.is_dir() and (path / "feature_activation_summary.csv").exists()
        )
        if not method_dirs:
            continue

        control_sizes = {}
        split_cache = {}

        for method_dir in method_dirs:
            method = method_dir.name.split("-l", 1)[0]
            target_scores, control_scores = load_method_concept_scores(base_dir, concept, method)
            control_sizes[method] = control_scores.size

            if concept not in split_cache:
                split_cache[concept] = split_control_indices(concept, control_scores.size, seed)

            calib_idx, eval_idx = split_cache[concept]
            control_calib = control_scores[calib_idx]
            control_eval = control_scores[eval_idx]
            auc_roc = calculate_auc(target_scores, control_scores)

            for alpha_percent in ALPHAS:
                alpha_fraction = alpha_percent / 100.0
                tau, calib_rate = choose_threshold(control_calib, alpha_fraction)
                realized_control = float(np.mean(control_eval > tau))
                target_reject = float(np.mean(target_scores > tau))

                records.append(
                    {
                        "concept": concept,
                        "method": method,
                        "display_name": DISPLAY_NAMES.get(method, method),
                        "alpha_percent": alpha_percent,
                        "score_aggregation": SCORE_AGG,
                        "tau": tau,
                        "calibration_control_reject_rate": calib_rate,
                        "realized_control_reject_rate": realized_control,
                        "target_reject_rate": target_reject,
                        "auc_roc": auc_roc,
                        "n_control_calibration": int(control_calib.size),
                        "n_control_evaluation": int(control_eval.size),
                        "n_target_evaluation": int(target_scores.size),
                        "feature_dir": str(method_dir),
                        "split_seed": seed,
                    }
                )

        concept_control_sizes = set(control_sizes.values())
        if len(concept_control_sizes) != 1:
            raise ValueError(f"Inconsistent control counts within concept {concept}: {control_sizes}")

    return pd.DataFrame.from_records(records)


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, values.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_group(per_concept_df: pd.DataFrame, group_name: str, seed: int) -> pd.DataFrame:
    group = GROUPS[group_name]
    subset = per_concept_df[
        per_concept_df["concept"].isin(group["concepts"]) &
        per_concept_df["method"].isin(group["methods"])
    ].copy()

    expected_rows = len(group["concepts"]) * len(group["methods"]) * len(ALPHAS)
    if len(subset) != expected_rows:
        raise ValueError(
            f"Group {group_name} expected {expected_rows} rows, found {len(subset)}."
        )

    records: list[dict] = []

    for method in group["methods"]:
        method_df = subset[subset["method"] == method].copy()
        concept_auc_df = method_df[["concept", "auc_roc"]].drop_duplicates().sort_values("concept")
        auc_values = concept_auc_df["auc_roc"].to_numpy(dtype=float)
        avg_auc = float(np.mean(auc_values))
        auc_ci_low, auc_ci_high = bootstrap_mean_ci(
            auc_values,
            stable_seed(f"{group_name}:{method}:auc", seed),
        )

        for alpha_percent in ALPHAS:
            alpha_df = method_df[method_df["alpha_percent"] == alpha_percent].sort_values("concept")
            target_rates = alpha_df["target_reject_rate"].to_numpy(dtype=float)
            control_rates = alpha_df["realized_control_reject_rate"].to_numpy(dtype=float)
            tau_values = alpha_df["tau"].to_numpy(dtype=float)

            target_ci_low, target_ci_high = bootstrap_mean_ci(
                target_rates,
                stable_seed(f"{group_name}:{method}:target:{alpha_percent}", seed),
            )
            control_ci_low, control_ci_high = bootstrap_mean_ci(
                control_rates,
                stable_seed(f"{group_name}:{method}:control:{alpha_percent}", seed),
            )

            records.append(
                {
                    "group": group_name,
                    "group_title": group["title"],
                    "group_subtitle": group["subtitle"],
                    "concepts": ",".join(group["concepts"]),
                    "method": method,
                    "display_name": DISPLAY_NAMES.get(method, method),
                    "alpha_percent": alpha_percent,
                    "avg_auc_roc": avg_auc,
                    "avg_auc_roc_ci_low": auc_ci_low,
                    "avg_auc_roc_ci_high": auc_ci_high,
                    "mean_target_reject_rate": float(target_rates.mean()),
                    "target_reject_ci_low": target_ci_low,
                    "target_reject_ci_high": target_ci_high,
                    "mean_realized_control_reject_rate": float(control_rates.mean()),
                    "realized_control_ci_low": control_ci_low,
                    "realized_control_ci_high": control_ci_high,
                    "mean_tau": float(tau_values.mean()),
                    "n_concepts": int(len(group["concepts"])),
                }
            )

    summary_df = pd.DataFrame.from_records(records)
    return summary_df


def build_wide_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for group_name, group_df in summary_df.groupby("group", sort=False):
        for method, method_df in group_df.groupby("method", sort=False):
            row = {
                "group": group_name,
                "group_title": method_df["group_title"].iloc[0],
                "group_subtitle": method_df["group_subtitle"].iloc[0],
                "method": method,
                "display_name": method_df["display_name"].iloc[0],
                "avg_auc_roc": method_df["avg_auc_roc"].iloc[0],
                "avg_auc_roc_ci_low": method_df["avg_auc_roc_ci_low"].iloc[0],
                "avg_auc_roc_ci_high": method_df["avg_auc_roc_ci_high"].iloc[0],
                "n_concepts": int(method_df["n_concepts"].iloc[0]),
            }

            for alpha_percent in [1, 5, 10]:
                alpha_row = method_df[method_df["alpha_percent"] == alpha_percent].iloc[0]
                row[f"target_reject_at_control_{alpha_percent}"] = alpha_row["mean_target_reject_rate"]
                row[f"target_reject_at_control_{alpha_percent}_ci_low"] = alpha_row["target_reject_ci_low"]
                row[f"target_reject_at_control_{alpha_percent}_ci_high"] = alpha_row["target_reject_ci_high"]
                row[f"realized_control_reject_at_{alpha_percent}"] = alpha_row["mean_realized_control_reject_rate"]
                row[f"realized_control_reject_at_{alpha_percent}_ci_low"] = alpha_row["realized_control_ci_low"]
                row[f"realized_control_reject_at_{alpha_percent}_ci_high"] = alpha_row["realized_control_ci_high"]

            rows.append(row)

    wide_df = pd.DataFrame.from_records(rows)
    return wide_df


def add_value_labels(ax: plt.Axes, heights: np.ndarray) -> None:
    for index, height in enumerate(heights):
        ax.text(
            index,
            height + 0.012,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
            fontweight="semibold",
        )


def plot_main_panels(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), sharey=True)

    for ax, group_name in zip(axes, GROUPS):
        group_df = summary_df[
            (summary_df["group"] == group_name) &
            (summary_df["alpha_percent"] == 5)
        ].copy()
        group_df = group_df.sort_values("mean_target_reject_rate", ascending=False)

        heights = group_df["mean_target_reject_rate"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                heights - group_df["target_reject_ci_low"].to_numpy(dtype=float),
                group_df["target_reject_ci_high"].to_numpy(dtype=float) - heights,
            ]
        )

        x = np.arange(len(group_df))
        colors = [METHOD_COLORS[row["method"]] for _, row in group_df.iterrows()]
        ax.bar(
            x,
            heights,
            yerr=yerr,
            color=colors,
            width=0.68,
            error_kw={
                "elinewidth": 1.6,
                "ecolor": "#1f1f1f",
                "capsize": 4,
                "capthick": 1.6,
            },
        )
        add_value_labels(ax, heights)

        ax.set_xticks(x)
        ax.set_xticklabels(group_df["display_name"], rotation=18, ha="right")
        ax.set_title(
            f"{GROUPS[group_name]['title']}\n{GROUPS[group_name]['subtitle']}",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Method")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(0.0, 1.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Average target reject rate")
    fig.suptitle("Working-Point Comparison at Fixed Control Reject Rate", fontsize=16, y=1.02)
    fig.text(
        0.5,
        0.02,
        "Fixed control reject target = 5%. Error bars show 95% bootstrap confidence intervals across concepts.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_operating_curves(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.3), sharey=True)
    x_positions = np.arange(len(ALPHAS))
    alpha_to_position = {alpha: pos for pos, alpha in enumerate(ALPHAS)}
    legend_handles: dict[str, plt.Line2D] = {}

    for ax, group_name in zip(axes, GROUPS):
        group_df = summary_df[summary_df["group"] == group_name].copy()
        sort_order = (
            group_df[group_df["alpha_percent"] == 5]
            .sort_values("mean_target_reject_rate", ascending=False)["method"]
            .tolist()
        )
        method_offsets = {
            method: offset
            for method, offset in zip(sort_order, np.linspace(-0.15, 0.15, len(sort_order)))
        }

        for method in sort_order:
            method_df = group_df[group_df["method"] == method].sort_values("alpha_percent")
            x = np.array(
                [alpha_to_position[int(alpha)] + method_offsets[method] for alpha in method_df["alpha_percent"]],
                dtype=float,
            )
            y = method_df["mean_target_reject_rate"].to_numpy(dtype=float)
            y_low = method_df["target_reject_ci_low"].to_numpy(dtype=float)
            y_high = method_df["target_reject_ci_high"].to_numpy(dtype=float)
            color = METHOD_COLORS[method]
            yerr = np.vstack([y - y_low, y_high - y])

            line, = ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=7,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=DISPLAY_NAMES.get(method, method),
                zorder=3,
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=color,
                elinewidth=1.8,
                capsize=4.5,
                capthick=1.8,
                alpha=0.85,
                zorder=2,
            )
            legend_handles.setdefault(method, line)

        ax.set_title(
            f"{GROUPS[group_name]['title']}\n{GROUPS[group_name]['subtitle']}",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Calibrated control reject target")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{alpha}%" for alpha in ALPHAS])
        ax.set_xlim(x_positions[0] - 0.28, x_positions[-1] + 0.28)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.5, 1.02)
        ax.set_axisbelow(True)
        ax.grid(axis="both", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Average target reject rate")
    fig.suptitle("Operating Curves Across Deployment Constraints", fontsize=16, y=1.02)
    ordered_methods = [method for method in LEGEND_METHOD_ORDER if method in legend_handles]
    fig.legend(
        [legend_handles[method] for method in ordered_methods],
        [DISPLAY_NAMES[method] for method in ordered_methods],
        loc="lower center",
        ncol=len(ordered_methods),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_auc_vs_working_point(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.3), sharey=True)
    legend_handles: dict[str, plt.Collection] = {}

    for ax, group_name in zip(axes, GROUPS):
        group_df = summary_df[
            (summary_df["group"] == group_name) &
            (summary_df["alpha_percent"] == 5)
        ].copy()
        group_df = group_df.sort_values("mean_target_reject_rate", ascending=False)

        for _, row in group_df.iterrows():
            x = float(row["avg_auc_roc"])
            y = float(row["mean_target_reject_rate"])
            handle = ax.scatter(
                x,
                y,
                s=110,
                color=METHOD_COLORS[row["method"]],
                edgecolor="black",
                linewidth=0.7,
            )
            legend_handles.setdefault(row["method"], handle)

        x_min = max(0.0, group_df["avg_auc_roc"].min() - 0.02)
        x_max = min(1.005, group_df["avg_auc_roc"].max() + 0.006)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.6, 1.02)
        ax.set_title(
            f"{GROUPS[group_name]['title']}\n{GROUPS[group_name]['subtitle']}",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Average AUC-ROC")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_axisbelow(True)
        ax.grid(axis="both", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Average target reject rate at control = 5%")
    fig.suptitle("AUC-ROC vs Working-Point Performance", fontsize=16, y=1.02)
    ordered_methods = [method for method in LEGEND_METHOD_ORDER if method in legend_handles]
    fig.legend(
        [legend_handles[method] for method in ordered_methods],
        [DISPLAY_NAMES[method] for method in ordered_methods],
        loc="lower center",
        ncol=len(ordered_methods),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_tables(per_concept_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    per_concept_df.sort_values(["concept", "method", "alpha_percent"]).to_csv(
        table_dir / "working_point_per_concept.csv",
        index=False,
    )
    summary_df.sort_values(["group", "method", "alpha_percent"]).to_csv(
        table_dir / "working_point_summary_long.csv",
        index=False,
    )
    build_wide_summary(summary_df).sort_values(["group", "method"]).to_csv(
        table_dir / "working_point_summary_wide.csv",
        index=False,
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_report(
    per_concept_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    base_dir: Path,
    seed: int,
) -> None:
    wide_df = build_wide_summary(summary_df).sort_values(["group", "target_reject_at_control_5"], ascending=[True, False])
    report_path = output_dir / "workpoint_report.md"

    lines = [
        "# Working-Point Benchmark Report",
        "",
        "## Protocol",
        f"- Source data: `{base_dir}`",
        f"- Score definition: `{SCORE_AGG}` sequence aggregation, to match the existing AUC benchmark figures.",
        f"- Threshold calibration: deterministic 50/50 split on the control set only, with seed `{seed}` and the same split reused across methods within each concept.",
        "- Evaluation: realized control reject rate on the held-out control split; target reject rate on the full target set (target samples are never used for threshold selection).",
        "- Fixed operating points: `1%`, `2%`, `5%`, `10%`, `20%`.",
        "- Error bars / intervals: 95% bootstrap confidence intervals over concepts.",
        "",
        "## Data Limits",
        "- `JumpReLU SAE` is available only for `indian_politics`, so it is excluded from multi-concept comparison figures.",
        "- `SUR SAE` (`kernel`) does not cover `female_subjects` or `photo_captions`, so the SUR panel is restricted to `football` and `indian_politics`.",
        "- `ReLU SAE` and `Gated SAE` are unavailable for `canadian_political` and `female_subjects`, so the mainstream panel uses `football`, `indian_politics`, and `photo_captions`.",
        "- Several methods have highly zero-inflated control-score distributions, so with the strict `score > tau` decision rule some concepts cannot realize exactly 5% on held-out control data; this is why a few realized control rates sit below the nominal target.",
        "- The per-concept CSV covers every method found under `ablation_datasets-dense`, but the figures intentionally use only method subsets with shared concept coverage so that the comparisons remain valid under the thesis plotting requirement.",
        "",
        "## Output Files",
        f"- Main figure: `{output_dir / 'pic' / 'fixed_control_5_target_reject_rate_panels.png'}`",
        f"- Operating curves: `{output_dir / 'pic' / 'operating_curves_panels.png'}`",
        f"- AUC vs working point: `{output_dir / 'pic' / 'auc_vs_target_reject_at_5_panels.png'}`",
        f"- Per-concept table: `{output_dir / 'tables' / 'working_point_per_concept.csv'}`",
        f"- Summary tables: `{output_dir / 'tables' / 'working_point_summary_long.csv'}` and `{output_dir / 'tables' / 'working_point_summary_wide.csv'}`",
        "",
        "## Group Summaries",
        "",
    ]

    for group_name, group_df in wide_df.groupby("group", sort=False):
        title = group_df["group_title"].iloc[0]
        subtitle = group_df["group_subtitle"].iloc[0]
        lines.append(f"### {title}")
        lines.append(f"- {subtitle}")
        for _, row in group_df.sort_values("target_reject_at_control_5", ascending=False).iterrows():
            lines.append(
                "- "
                f"{row['display_name']}: "
                f"AUC={row['avg_auc_roc']:.4f}, "
                f"Target@1={format_pct(row['target_reject_at_control_1'])}, "
                f"Target@5={format_pct(row['target_reject_at_control_5'])}, "
                f"Target@10={format_pct(row['target_reject_at_control_10'])}, "
                f"RealizedControl@5={format_pct(row['realized_control_reject_at_5'])}, "
                f"n_concepts={int(row['n_concepts'])}"
            )
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    pic_dir = output_dir / "pic"
    pic_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Using base directory: %s", base_dir)
    logging.info("Using output directory: %s", output_dir)
    logging.info("Score aggregation: %s", SCORE_AGG)

    per_concept_df = build_per_concept_table(base_dir=base_dir, seed=args.seed)
    summary_frames = [
        summarize_group(per_concept_df=per_concept_df, group_name=group_name, seed=args.seed)
        for group_name in GROUPS
    ]
    summary_df = pd.concat(summary_frames, ignore_index=True)

    save_tables(per_concept_df=per_concept_df, summary_df=summary_df, output_dir=output_dir)

    plot_main_panels(
        summary_df=summary_df,
        output_path=pic_dir / "fixed_control_5_target_reject_rate_panels.png",
    )
    plot_operating_curves(
        summary_df=summary_df,
        output_path=pic_dir / "operating_curves_panels.png",
    )
    plot_auc_vs_working_point(
        summary_df=summary_df,
        output_path=pic_dir / "auc_vs_target_reject_at_5_panels.png",
    )

    build_report(
        per_concept_df=per_concept_df,
        summary_df=summary_df,
        output_dir=output_dir,
        base_dir=base_dir,
        seed=args.seed,
    )

    logging.info("Generated working-point figures and tables under %s", output_dir)


if __name__ == "__main__":
    main()
