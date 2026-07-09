#!/usr/bin/env python3
"""Aggregate five-seed V396 causal results and apply the preregistered gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


T_CRITICAL_95_DF4 = 2.7764451051977987
FIXED_PREFIX = "fixed_beta_"
LEARNED_VARIANTS = ("global_beta", "feature_beta", "full_beta_gain")
REQUIRED_VARIANTS = {
    "relu_finetune",
    "scaled_relu",
    "fixed_beta_0p10",
    "fixed_beta_0p15",
    "fixed_beta_0p20",
    "fixed_beta_0p25",
    *LEARNED_VARIANTS,
}


def mean_std_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = T_CRITICAL_95_DF4 * std / math.sqrt(len(values)) if len(values) == 5 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci_half_width": half_width,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def paired(values_a: dict[int, float], values_b: dict[int, float]) -> dict[str, Any]:
    seeds = sorted(set(values_a) & set(values_b))
    deltas = [values_a[seed] - values_b[seed] for seed in seeds]
    return {"seeds": seeds, "deltas": deltas, **mean_std_ci(deltas)}


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# V396 Causal Attribution Decision",
        "",
        "## Five-seed summary",
        "",
        "| Variant | Mean Acc ↑ | SD | 95% CI half-width | Top-5 Acc ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in sorted(
        report["variant_summary"].items(),
        key=lambda item: item[1]["mean_acc"]["mean"],
        reverse=True,
    ):
        lines.append(
            f"| {key} | {row['mean_acc']['mean']:.6f} | {row['mean_acc']['std']:.6f} | "
            f"{row['mean_acc']['ci_half_width']:.6f} | {row['top_5_acc']['mean']:.6f} |"
        )
    comparison = report["primary_comparison"]
    gate = report["gate"]
    lines.extend([
        "",
        "## Primary learned-vs-fixed comparison",
        "",
        f"- Best learned variant: `{comparison['learned_variant']}`",
        f"- Best fixed variant: `{comparison['fixed_variant']}`",
        f"- Paired Mean Acc delta: `{comparison['mean_acc_delta']['mean']:+.6f}`",
        f"- Paired 95% CI: `[{comparison['mean_acc_delta']['ci_low']:+.6f}, "
        f"{comparison['mean_acc_delta']['ci_high']:+.6f}]`",
        f"- Paired Top-5 delta vs best fixed: `{comparison['top_5_delta']['mean']:+.6f}`",
        f"- Paired Top-5 delta vs ReLU finetune: "
        f"`{comparison['top_5_vs_relu_finetune']['mean']:+.6f}`",
        f"- Paired Top-5 delta vs scaled-ReLU: "
        f"`{comparison['top_5_vs_scaled_relu']['mean']:+.6f}`",
        "",
        "## Gate",
        "",
        f"- Delta at least +0.003: `{gate['delta_at_least_0p003']}`",
        f"- Mean Acc CI excludes zero: `{gate['ci_excludes_zero']}`",
        f"- Top-5 does not regress vs best fixed: "
        f"`{gate['top5_not_regressed_vs_best_fixed']}`",
        f"- Top-5 does not regress vs same-parameter scaled-ReLU: "
        f"`{gate['top5_not_regressed_vs_scaled_relu']}`",
        f"- Long from-scratch training allowed: `{gate['long_training_allowed']}`",
        "",
        f"Decision: **{report['decision']}**",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    by_variant: dict[str, dict[int, dict[str, float]]] = {}
    for row in payload["summary"]:
        variant = str(row["variant_key"])
        seed = int(row["seed"])
        by_variant.setdefault(variant, {})[seed] = {
            "mean_acc": float(row["mean_acc"]),
            "mean_auc": float(row["mean_auc"]),
            "top_1_acc": float(row["top_1_acc"]),
            "top_2_acc": float(row["top_2_acc"]),
            "top_5_acc": float(row["top_5_acc"]),
        }

    missing_variants = sorted(REQUIRED_VARIANTS - set(by_variant))
    unexpected_variants = sorted(set(by_variant) - REQUIRED_VARIANTS)
    if missing_variants or unexpected_variants:
        raise RuntimeError(
            "Variant-set mismatch: "
            f"missing={missing_variants}, unexpected={unexpected_variants}"
        )

    expected_seeds = {42, 43, 44, 45, 46}
    incomplete = {
        variant: sorted(expected_seeds - set(seed_rows))
        for variant, seed_rows in by_variant.items()
        if set(seed_rows) != expected_seeds
    }
    if incomplete:
        raise RuntimeError(f"Incomplete five-seed results: {incomplete}")

    variant_summary: dict[str, Any] = {}
    for variant, seed_rows in by_variant.items():
        variant_summary[variant] = {
            metric: mean_std_ci([seed_rows[seed][metric] for seed in sorted(seed_rows)])
            for metric in ("mean_acc", "mean_auc", "top_1_acc", "top_2_acc", "top_5_acc")
        }

    fixed_variants = [variant for variant in by_variant if variant.startswith(FIXED_PREFIX)]
    learned_variants = [variant for variant in LEARNED_VARIANTS if variant in by_variant]
    best_fixed = max(fixed_variants, key=lambda variant: variant_summary[variant]["mean_acc"]["mean"])
    best_learned = max(learned_variants, key=lambda variant: variant_summary[variant]["mean_acc"]["mean"])

    mean_acc_delta = paired(
        {seed: row["mean_acc"] for seed, row in by_variant[best_learned].items()},
        {seed: row["mean_acc"] for seed, row in by_variant[best_fixed].items()},
    )
    top_5_delta = paired(
        {seed: row["top_5_acc"] for seed, row in by_variant[best_learned].items()},
        {seed: row["top_5_acc"] for seed, row in by_variant[best_fixed].items()},
    )
    top_5_vs_relu = paired(
        {seed: row["top_5_acc"] for seed, row in by_variant[best_learned].items()},
        {seed: row["top_5_acc"] for seed, row in by_variant["relu_finetune"].items()},
    )
    top_5_vs_scaled = paired(
        {seed: row["top_5_acc"] for seed, row in by_variant[best_learned].items()},
        {seed: row["top_5_acc"] for seed, row in by_variant["scaled_relu"].items()},
    )
    gate = {
        "delta_at_least_0p003": mean_acc_delta["mean"] >= 0.003,
        "ci_excludes_zero": mean_acc_delta["ci_low"] > 0.0,
        "top5_not_regressed_vs_best_fixed": top_5_delta["mean"] >= 0.0,
        "top5_not_regressed_vs_scaled_relu": top_5_vs_scaled["mean"] >= 0.0,
    }
    gate["long_training_allowed"] = all(gate.values())

    all_pairwise: dict[str, Any] = {}
    for learned in learned_variants:
        for fixed in fixed_variants:
            all_pairwise[f"{learned}_vs_{fixed}"] = {
                "mean_acc_delta": paired(
                    {seed: row["mean_acc"] for seed, row in by_variant[learned].items()},
                    {seed: row["mean_acc"] for seed, row in by_variant[fixed].items()},
                ),
                "top_5_delta": paired(
                    {seed: row["top_5_acc"] for seed, row in by_variant[learned].items()},
                    {seed: row["top_5_acc"] for seed, row in by_variant[fixed].items()},
                ),
            }

    report = {
        "eval_json": str(args.eval_json),
        "variant_summary": variant_summary,
        "primary_comparison": {
            "learned_variant": best_learned,
            "fixed_variant": best_fixed,
            "mean_acc_delta": mean_acc_delta,
            "top_5_delta": top_5_delta,
            "top_5_vs_relu_finetune": top_5_vs_relu,
            "top_5_vs_scaled_relu": top_5_vs_scaled,
        },
        "all_pairwise_comparisons": all_pairwise,
        "gate": gate,
        "decision": (
            "proceed-to-fair-from-scratch"
            if gate["long_training_allowed"]
            else "stop-old-cache-architecture-search"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.output_md, report)
    print(json.dumps(report["primary_comparison"] | {"gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
