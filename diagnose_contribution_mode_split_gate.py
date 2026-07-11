#!/usr/bin/env python3
"""Diagnose how contribution-mode splitting changes sparse-probe selections."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


LABELS = {
    "reference": "Frozen V396 reference",
    "fold": "Mass-fold-only control",
    "wrong": "Coordinate-misaligned split control",
    "candidate": "True contribution-mode split candidate",
}


def selection_map(row: dict[str, object]) -> dict[str, set[int]]:
    selections: dict[str, set[int]] = {}
    for dataset, classes in row["per_class"].items():
        for class_name, k_rows in classes.items():
            for k_name, values in k_rows.items():
                key = f"{dataset}|{class_name}|{k_name}"
                selections[key] = {int(index) for index in values["selected_features"]}
    return selections


def overlap_summary(
    candidate: dict[str, set[int]], control: dict[str, set[int]]
) -> dict[str, float | int]:
    if set(candidate) != set(control):
        raise ValueError("Selection keys differ between candidate and control")
    jaccards = []
    exact = 0
    for key in candidate:
        left = candidate[key]
        right = control[key]
        exact += left == right
        union = left | right
        jaccards.append(len(left & right) / len(union) if union else 1.0)
    return {
        "comparisons": len(jaccards),
        "exact_selection_sets": exact,
        "exact_fraction": exact / len(jaccards),
        "mean_jaccard": sum(jaccards) / len(jaccards),
        "minimum_jaccard": min(jaccards),
    }


def role_summary(
    selections: dict[str, set[int]], parents: set[int], recipients: set[int]
) -> dict[str, object]:
    occurrences = Counter(index for values in selections.values() for index in values)
    unique = set(occurrences)
    return {
        "selection_occurrences": sum(occurrences.values()),
        "unique_selected_features": len(unique),
        "parent_occurrences": sum(occurrences[index] for index in parents),
        "recipient_occurrences": sum(occurrences[index] for index in recipients),
        "unique_selected_parents": len(unique & parents),
        "unique_selected_recipients": len(unique & recipients),
        "unmodified_occurrences": sum(
            count
            for index, count in occurrences.items()
            if index not in parents and index not in recipients
        ),
    }


def k_metric_summary(
    row: dict[str, object], datasets: list[str], metric: str
) -> dict[str, float]:
    return {
        f"k{k}": sum(
            float(row["dataset_results"][dataset][f"sae_top_{k}_test_{metric}"])
            for dataset in datasets
        )
        / len(datasets)
        for k in (1, 2, 5)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text())
    rows = {row["label"]: row for row in payload["architecture_results"]}
    keyed = {key: rows[label] for key, label in LABELS.items()}
    spec = torch.load(args.spec, map_location="cpu", weights_only=True)
    parents = {int(value) for value in spec["parent_indices"].tolist()}
    recipients = {int(value) for value in spec["recipient_indices"].tolist()}
    if parents & recipients:
        raise ValueError("Parent and recipient feature sets overlap")

    selections = {key: selection_map(row) for key, row in keyed.items()}
    datasets = list(payload["config"]["dataset_names"])
    candidate_accuracy = k_metric_summary(keyed["candidate"], datasets, "accuracy")
    candidate_auc = k_metric_summary(keyed["candidate"], datasets, "auc")
    metric_summary = {}
    for key, row in keyed.items():
        accuracy = k_metric_summary(row, datasets, "accuracy")
        auc = k_metric_summary(row, datasets, "auc")
        metric_summary[key] = {
            "accuracy": accuracy,
            "auc": auc,
            "candidate_accuracy_delta": {
                k: candidate_accuracy[k] - accuracy[k] for k in candidate_accuracy
            },
            "candidate_auc_delta": {
                k: candidate_auc[k] - auc[k] for k in candidate_auc
            },
        }

    report = {
        "experiment": "Contribution-mode split post-gate diagnosis",
        "status": "posthoc-diagnosis-after-preregistered-family-closure",
        "selection_overlap": {
            key: overlap_summary(selections["candidate"], selections[key])
            for key in ("reference", "fold", "wrong")
        },
        "selected_feature_roles": {
            key: role_summary(values, parents, recipients)
            for key, values in selections.items()
        },
        "k_level_metrics": metric_summary,
        "interpretation_guardrail": (
            "This diagnosis was run only after the preregistered gate failed and "
            "cannot promote the closed family."
        ),
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Contribution-mode split post-gate diagnosis",
        "",
        "| Comparator | Exact sets | Mean Jaccard | Minimum Jaccard |",
        "|---|---:|---:|---:|",
    ]
    for key, values in report["selection_overlap"].items():
        lines.append(
            f"| {key} | {values['exact_selection_sets']}/{values['comparisons']} "
            f"| {values['mean_jaccard']:.6f} | {values['minimum_jaccard']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Variant | Parent hits | Recipient hits | Unmodified hits |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, values in report["selected_feature_roles"].items():
        lines.append(
            f"| {key} | {values['parent_occurrences']} | "
            f"{values['recipient_occurrences']} | {values['unmodified_occurrences']} |"
        )
    lines.extend(
        [
            "",
            report["interpretation_guardrail"],
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
