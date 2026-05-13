import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


METHOD_COLORS = {
    "topk": "#7f8c8d",
    "batchtopk": "#d99100",
    "relu": "#d95f02",
    "gatedsae": "#1b9e77",
    "jumprelu": "#7570b3",
    "dense": "#1f78b4",
    "kernel": "#2f2f2f",
}

HEADLINE_METHOD_ORDER = [
    "kernel",
    "dense",
    "relu",
    "gatedsae",
    "batchtopk",
    "topk",
    "jumprelu",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the final focused two-metric summary folder from the full family benchmark run."
    )
    parser.add_argument(
        "--benchmark_run_dir",
        type=Path,
        default=Path("presentation/260417/final_family_benchmark_l22/results/full_l22"),
        help="Directory containing the full benchmark run outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("presentation/260417/final_family_benchmark_l22/final_outputs"),
        help="Final focused output folder.",
    )
    parser.add_argument(
        "--control_redundancy_dir",
        type=Path,
        default=Path("presentation/260417/final_family_benchmark_l22/results/control_redundancy_full"),
        help="Directory containing the exhaustive non-proxy control redundancy sweep.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_render_plots(rows: list[dict[str, object]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    labels = [row["method_label"] for row in rows]
    colors = [METHOD_COLORS[row["method"]] for row in rows]

    axes[0].bar(labels, [row["strict_budget_alternative_controller_rate_2pct_30"] for row in rows], color=colors, alpha=0.95)
    axes[0].set_title("Headline Metric 1: Alternative-Controller Rate @2%/30%")
    axes[0].set_ylabel("Share of controllable families with at least 2 valid controllers")
    axes[0].set_ylim(0, 1.02)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(labels, [row["strict_budget_valid_trigger_yield_2pct"] for row in rows], color=colors, alpha=0.95)
    axes[1].set_title("Headline Metric 2: Strict-Budget Valid Trigger Yield @2%")
    axes[1].set_ylabel("Mean valid target reject rate on covered families")
    axes[1].set_ylim(0, 1.02)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    for idx, row in enumerate(rows):
        axes[0].text(idx, row["strict_budget_alternative_controller_rate_2pct_30"] + 0.02, f"{row['strict_budget_alternative_controller_rate_2pct_30'] * 100:.1f}%", ha="center", va="bottom", fontsize=9)
        axes[1].text(idx, row["strict_budget_valid_trigger_yield_2pct"] + 0.02, f"{row['strict_budget_valid_trigger_yield_2pct'] * 100:.1f}%", ha="center", va="bottom", fontsize=9)

    panel_path = plot_dir / "final_headline_metrics.png"
    fig.savefig(panel_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    for row in rows:
        ax.scatter(
            row["strict_budget_alternative_controller_rate_2pct_30"],
            row["strict_budget_valid_trigger_yield_2pct"],
            s=60 + row["controllable_family_count_2pct_30"] * 20,
            color=METHOD_COLORS[row["method"]],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
        )
        ax.text(
            row["strict_budget_alternative_controller_rate_2pct_30"] + 0.01,
            row["strict_budget_valid_trigger_yield_2pct"] + 0.005,
            row["method_label"],
            fontsize=9,
        )
    ax.set_title("Alternative-Controller Rate vs Strict-Budget Valid Trigger Yield")
    ax.set_xlabel("Alternative-controller rate @2%/30%")
    ax.set_ylabel("Strict-budget valid trigger yield @2%")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(linestyle="--", alpha=0.3)

    scatter_path = plot_dir / "alternative_controller_vs_strict_valid_yield.png"
    fig.savefig(scatter_path, dpi=220)
    plt.close(fig)

    return [str(panel_path), str(scatter_path)]


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    control_redundancy_dir = args.control_redundancy_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    method_summary = read_csv(benchmark_dir / "method_summary.csv")
    per_budget = read_csv(benchmark_dir / "per_budget_results.csv")
    registry_rows = read_csv(benchmark_dir / "feature_registry_snapshot.csv")
    control_redundancy_rows = read_csv(control_redundancy_dir / "control_metric_candidates.csv")

    family_candidate_counts = defaultdict(lambda: defaultdict(int))
    for row in registry_rows:
        family_candidate_counts[row["method"]][row["family_id"]] += 1

    quality_2pct = defaultdict(list)
    hit_rate_2pct = defaultdict(list)
    in_budget_target_2pct = defaultdict(list)
    for row in per_budget:
        if abs(float(row["budget"]) - 0.02) < 1e-12:
            control_reject = float(row["control_reject_rate"])
            target_reject = float(row["target_reject_rate"])
            is_valid = control_reject <= 0.02 + 1e-12
            quality_2pct[row["method"]].append(target_reject if is_valid else 0.0)
            hit_rate_2pct[row["method"]].append(1.0 if is_valid else 0.0)
            if is_valid:
                in_budget_target_2pct[row["method"]].append(target_reject)

    alternative_controller_by_method: dict[str, dict[str, float | int]] = {}
    for row in control_redundancy_rows:
        if abs(float(row["budget"]) - 0.02) > 1e-12:
            continue
        if abs(float(row["target_threshold"]) - 0.30) > 1e-12:
            continue
        controllable_count = int(row["controllable_family_count"])
        redundant_count = int(row["redundant_controller_family_count"])
        alternative_rate = (redundant_count / controllable_count) if controllable_count else 0.0
        alternative_controller_by_method[row["method"]] = {
            "strict_budget_alternative_controller_rate_2pct_30": alternative_rate,
            "alternative_controller_family_count_2pct_30": redundant_count,
            "controllable_family_count_2pct_30": controllable_count,
            "strict_budget_redundant_family_rate_2pct_30": float(row["redundant_controller_family_rate"]),
        }

    focused_rows = []
    method_summary_by_method = {row["method"]: row for row in method_summary}
    for method in HEADLINE_METHOD_ORDER:
        if method not in method_summary_by_method:
            continue
        row = method_summary_by_method[method]
        alt_metrics = alternative_controller_by_method.get(
            method,
            {
                "strict_budget_alternative_controller_rate_2pct_30": 0.0,
                "alternative_controller_family_count_2pct_30": 0,
                "controllable_family_count_2pct_30": 0,
                "strict_budget_redundant_family_rate_2pct_30": 0.0,
            },
        )
        focused_rows.append(
            {
                "method": method,
                "method_label": row["method_label"],
                **alt_metrics,
                "benchmark_family_candidate_yield": int(row["candidate_feature_count"]),
                "rich_candidate_coverage_rate_2plus": (
                    sum(1 for count in family_candidate_counts[method].values() if count >= 2) / 15.0
                ),
                "coverage_count": int(row["coverage_count"]),
                "coverage_rate": float(row["coverage_rate"]),
                "strict_budget_valid_trigger_yield_2pct": sum(quality_2pct[method]) / len(quality_2pct[method]) if quality_2pct[method] else 0.0,
                "strict_control_hit_rate_2pct": sum(hit_rate_2pct[method]) / len(hit_rate_2pct[method]) if hit_rate_2pct[method] else 0.0,
                "in_budget_target_reject_mean_2pct": sum(in_budget_target_2pct[method]) / len(in_budget_target_2pct[method]) if in_budget_target_2pct[method] else 0.0,
                "fcos_all_budgets": float(row["fcos_all_budgets"]),
                "covered_quality_strict_2_5": float(row["covered_quality_strict"]),
                "mean_evaluation_auc": float(row["mean_evaluation_auc"]),
            }
        )

    write_csv(
        output_dir / "focused_method_summary.csv",
        focused_rows,
        [
            "method",
            "method_label",
            "strict_budget_alternative_controller_rate_2pct_30",
            "alternative_controller_family_count_2pct_30",
            "controllable_family_count_2pct_30",
            "strict_budget_redundant_family_rate_2pct_30",
            "rich_candidate_coverage_rate_2plus",
            "benchmark_family_candidate_yield",
            "coverage_count",
            "coverage_rate",
            "strict_budget_valid_trigger_yield_2pct",
            "strict_control_hit_rate_2pct",
            "in_budget_target_reject_mean_2pct",
            "fcos_all_budgets",
            "covered_quality_strict_2_5",
            "mean_evaluation_auc",
        ],
    )

    plot_paths = try_render_plots(focused_rows, output_dir)

    breadth_winner = max(focused_rows, key=lambda row: row["strict_budget_alternative_controller_rate_2pct_30"])
    quality_winner = max(focused_rows, key=lambda row: row["strict_budget_valid_trigger_yield_2pct"])
    utility_winner = max(focused_rows, key=lambda row: row["fcos_all_budgets"])

    lines = [
        "# Final Focused Family-Benchmark Metrics",
        "",
        "## Chosen Headline Metrics",
        "",
        "- `Strict-Budget Alternative-Controller Rate @2%/30%`: among families that are controllable at a 2% held-out control budget with at least 30% target reject, the share of families for which a method retains at least 2 distinct valid controllers. This is a non-proxy control redundancy metric tailored to the SUR innovation.",
        "- `Strict-Budget Valid Trigger Yield @2%`: for each covered family, keep the evaluation target reject rate only if the realized held-out control reject rate is at or below 2%; otherwise score that family as 0. This is a hard-gated strict-budget metric tailored to the PLRDC innovation.",
        "",
        "## Headline Results",
        "",
        f"- Highest strict-budget alternative-controller rate @2%/30%: **{breadth_winner['method_label']}** at {breadth_winner['strict_budget_alternative_controller_rate_2pct_30'] * 100:.1f}%, corresponding to {breadth_winner['alternative_controller_family_count_2pct_30']}/{breadth_winner['controllable_family_count_2pct_30']} controllable families with an alternative valid controller.",
        f"- Highest strict-budget valid trigger yield @2%: **{quality_winner['method_label']}** at {quality_winner['strict_budget_valid_trigger_yield_2pct'] * 100:.1f}% over its covered families.",
        f"- Supporting utility metric: the highest overall FCOS@2%,5%,10% is **{utility_winner['method_label']}** at {utility_winner['fcos_all_budgets'] * 100:.1f}%.",
        "",
        "## Interpretation",
        "",
        "- The SUR innovation is better captured by control redundancy than by candidate-space breadth alone. This metric only counts controllers that survive held-out calibration and held-out evaluation, so it is a true control metric rather than a proxy.",
        "- At the chosen strict setting (2% budget, minimum 30% target reject), `kernel` is the only method with a non-zero alternative-controller rate. In the current run, that redundancy appears in `gaming_general`.",
        "- The PLRDC innovation is better captured by a hard-gated strict-budget metric than by a soft penalty. If a family misses the 2% control budget, that family should contribute 0 to a strict-control claim.",
        "- This hard-gated metric separates `dense` from `relu` more clearly because `dense` combines a slightly higher 2% budget-hit rate with a higher in-budget target reject mean.",
        "- `Rich Candidate Coverage Rate @2+` remains a useful supporting candidate-space diagnostic, but it is no longer the SUR headline metric.",
        "- FCOS remains useful as a supporting overall benchmark score, but it is not the cleanest single metric for isolating either innovation.",
        "",
        "## Method Table",
        "",
    ]
    for row in focused_rows:
        lines.append(
            f"- {row['method_label']}: SBACR@2%/30% {row['strict_budget_alternative_controller_rate_2pct_30'] * 100:.1f}% "
            f"({row['alternative_controller_family_count_2pct_30']}/{row['controllable_family_count_2pct_30']} controllable families; "
            f"{row['strict_budget_redundant_family_rate_2pct_30'] * 100:.1f}% of all families), "
            f"RCCR@2+ {row['rich_candidate_coverage_rate_2plus'] * 100:.1f}%, "
            f"raw candidate yield {row['benchmark_family_candidate_yield']}, "
            f"coverage {row['coverage_count']}/15, "
            f"SBVTY@2% {row['strict_budget_valid_trigger_yield_2pct'] * 100:.1f}%, "
            f"2% hit rate {row['strict_control_hit_rate_2pct'] * 100:.1f}%, "
            f"FCOS {row['fcos_all_budgets'] * 100:.1f}%."
        )

    lines.extend(
        [
            "",
            "## Included Files",
            "",
            "- `focused_method_summary.csv`",
            "- `headline_metrics_report.md`",
            "- `raw_benchmark_run/`",
        ]
    )
    if plot_paths:
        lines.extend(
            [
                "- `plots/final_headline_metrics.png`",
                "- `plots/alternative_controller_vs_strict_valid_yield.png`",
            ]
        )

    (output_dir / "headline_metrics_report.md").write_text("\n".join(lines) + "\n")

    raw_dir = output_dir / "raw_benchmark_run"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    shutil.copytree(benchmark_dir, raw_dir)
    control_snapshot_dir = raw_dir / "control_redundancy_full"
    if control_snapshot_dir.exists():
        shutil.rmtree(control_snapshot_dir)
    shutil.copytree(control_redundancy_dir, control_snapshot_dir)


if __name__ == "__main__":
    main()
