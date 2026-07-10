#!/usr/bin/env python3
"""Diagnose how the nested partition changes sparse-probe feature selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def selection_stats(
    per_class: dict[str, Any],
    k: int,
    inner_features: int,
) -> dict[str, Any]:
    lists = [
        metrics[f"top_{k}"]["selected_features"]
        for metrics in per_class.values()
    ]
    flattened = [feature for selected in lists for feature in selected]
    backup = [feature for selected in lists for feature in selected[1:]]
    counts = Counter(flattened)
    rank_inner_fractions = []
    for rank in range(k):
        rank_features = [selected[rank] for selected in lists]
        rank_inner_fractions.append(
            sum(feature < inner_features for feature in rank_features)
            / len(rank_features)
        )
    return {
        "selection_count": len(flattened),
        "unique_feature_count": len(counts),
        "unique_fraction": len(counts) / len(flattened),
        "maximum_reuse_count": max(counts.values()),
        "maximum_reuse_fraction": max(counts.values()) / len(lists),
        "inner_fraction": (
            sum(feature < inner_features for feature in flattened)
            / len(flattened)
        ),
        "backup_inner_fraction": (
            sum(feature < inner_features for feature in backup) / len(backup)
            if backup
            else None
        ),
        "rank_inner_fractions": rank_inner_fractions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--inner-features", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    rows = {
        row["variant_key"]: row for row in payload["architecture_results"]
    }
    base = rows["base"]
    candidate = rows["candidate"]
    report: dict[str, Any] = {
        "inner_features": args.inner_features,
        "total_features": 65_536,
        "datasets": {},
    }
    for dataset, candidate_classes in candidate["per_class"].items():
        base_classes = base["per_class"][dataset]
        dataset_report: dict[str, Any] = {"k": {}, "per_class": {}}
        for k in (1, 2, 5):
            base_stats = selection_stats(
                base_classes,
                k,
                args.inner_features,
            )
            candidate_stats = selection_stats(
                candidate_classes,
                k,
                args.inner_features,
            )
            dataset_report["k"][str(k)] = {
                "base_prefix_control": base_stats,
                "candidate_inner_partition": candidate_stats,
                "inner_selection_fraction_delta": (
                    candidate_stats["inner_fraction"]
                    - base_stats["inner_fraction"]
                ),
                "accuracy_delta": (
                    candidate["dataset_results"][dataset][
                        f"sae_top_{k}_test_accuracy"
                    ]
                    - base["dataset_results"][dataset][
                        f"sae_top_{k}_test_accuracy"
                    ]
                ),
                "auc_delta": (
                    candidate["dataset_results"][dataset][
                        f"sae_top_{k}_test_auc"
                    ]
                    - base["dataset_results"][dataset][
                        f"sae_top_{k}_test_auc"
                    ]
                ),
            }
        for class_name, candidate_metrics in candidate_classes.items():
            base_metrics = base_classes[class_name]
            dataset_report["per_class"][class_name] = {
                f"top_{k}": {
                    "base_accuracy": base_metrics[f"top_{k}"]["test_accuracy"],
                    "candidate_accuracy": candidate_metrics[f"top_{k}"][
                        "test_accuracy"
                    ],
                    "accuracy_delta": (
                        candidate_metrics[f"top_{k}"]["test_accuracy"]
                        - base_metrics[f"top_{k}"]["test_accuracy"]
                    ),
                    "candidate_selected_features": candidate_metrics[
                        f"top_{k}"
                    ]["selected_features"],
                    "candidate_selected_partitions": [
                        "inner" if feature < args.inner_features else "outer"
                        for feature in candidate_metrics[f"top_{k}"][
                            "selected_features"
                        ]
                    ],
                }
                for k in (1, 2, 5)
            }
        report["datasets"][dataset] = dataset_report

    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Sample-nested feature-selection diagnostic",
        "",
        (
            "The base prefix is the same index range before the nested loss and "
            "serves as the 50% capacity control."
        ),
        "",
        "| Dataset | k | Base prefix share | Candidate inner share | "
        "Backup inner share | Acc delta | AUC delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, dataset_report in report["datasets"].items():
        for k in (1, 2, 5):
            row = dataset_report["k"][str(k)]
            base_stats = row["base_prefix_control"]
            candidate_stats = row["candidate_inner_partition"]
            backup = candidate_stats["backup_inner_fraction"]
            backup_text = "--" if backup is None else f"{backup:.3f}"
            lines.append(
                f"| {dataset} | {k} | {base_stats['inner_fraction']:.3f} | "
                f"{candidate_stats['inner_fraction']:.3f} | {backup_text} | "
                f"{row['accuracy_delta']:+.6f} | {row['auc_delta']:+.6f} |"
            )
    lines.extend(["", "## Per-class accuracy deltas", ""])
    for dataset, dataset_report in report["datasets"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Class | Top-1 delta | Top-2 delta | Top-5 delta | "
                "Candidate Top-5 partitions |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for class_name, class_report in dataset_report["per_class"].items():
            partitions = "/".join(
                class_report["top_5"]["candidate_selected_partitions"]
            )
            lines.append(
                f"| {class_name} | "
                f"{class_report['top_1']['accuracy_delta']:+.6f} | "
                f"{class_report['top_2']['accuracy_delta']:+.6f} | "
                f"{class_report['top_5']['accuracy_delta']:+.6f} | "
                f"{partitions} |"
            )
        lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
