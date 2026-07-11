#!/usr/bin/env python3
"""Apply the preregistered Initial3 promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "reference": "Frozen V396 reference",
    "fold": "Mass-fold-only control",
    "wrong": "Coordinate-misaligned split control",
    "candidate": "True contribution-mode split candidate",
}


def dataset_mean(row: dict[str, object], dataset: str, metric: str) -> float:
    values = row["dataset_results"][dataset]
    return sum(
        float(values[f"sae_top_{k}_test_{metric}"]) for k in (1, 2, 5)
    ) / 3


def overall_mean(
    row: dict[str, object], datasets: list[str], metric: str
) -> float:
    return sum(dataset_mean(row, dataset, metric) for dataset in datasets) / len(
        datasets
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--fit-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--minimum-control-delta", type=float, default=0.005)
    parser.add_argument("--maximum-dataset-drop", type=float, default=0.01)
    parser.add_argument("--reference-initial3", type=float, default=0.837543)
    parser.add_argument("--maximum-reference-gap", type=float, default=0.01)
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text())
    fit = json.loads(args.fit_summary.read_text())
    rows = {row["label"]: row for row in payload["architecture_results"]}
    missing = sorted(set(LABELS.values()) - set(rows))
    if missing:
        raise KeyError(f"Missing variants: {missing}")
    keyed = {key: rows[label] for key, label in LABELS.items()}
    datasets = list(payload["config"]["dataset_names"])
    means = {
        key: {
            "mean_acc": overall_mean(row, datasets, "accuracy"),
            "mean_auc": overall_mean(row, datasets, "auc"),
        }
        for key, row in keyed.items()
    }
    control_keys = ("reference", "fold", "wrong")
    best_control = max(control_keys, key=lambda key: means[key]["mean_acc"])
    candidate = means["candidate"]["mean_acc"]
    best = means[best_control]["mean_acc"]
    per_dataset = {}
    for dataset in datasets:
        candidate_value = dataset_mean(keyed["candidate"], dataset, "accuracy")
        controls = {
            key: dataset_mean(keyed[key], dataset, "accuracy")
            for key in control_keys
        }
        dataset_best = max(controls, key=controls.get)
        per_dataset[dataset] = {
            "candidate": candidate_value,
            "controls": controls,
            "best_control": dataset_best,
            "delta_vs_best_control": candidate_value - controls[dataset_best],
        }
    minimum_dataset_delta = min(
        row["delta_vs_best_control"] for row in per_dataset.values()
    )
    fit_integrity = bool(
        fit["data_unchanged"]
        and fit["wrong_control"]["zero_shifts"] == 0
        and fit["wrong_control"]["unchanged_allocation_rows"] == 0
        and fit["wrong_control"]["allocation_multisets_preserved"]
        and fit["integrity"]["parent_recipient_disjoint"]
        and fit["integrity"]["same_exposed_feature_count"]
        and not fit["integrity"]["uses_saebench_labels_for_fitting"]
        and not fit["integrity"]["uses_eval_split_for_fitting"]
        and not fit["integrity"]["uses_one_vs_rest_targets_for_fitting"]
        and not fit["integrity"]["uses_mean_diff_selection_for_fitting"]
        and not fit["integrity"]["uses_test_feedback_for_fitting"]
    )
    gap_vs_reference = candidate - args.reference_initial3
    gates = {
        "fit_integrity_pass": fit_integrity,
        "candidate_beats_best_control_by_0p005": candidate - best
        >= args.minimum_control_delta,
        "no_initial3_dataset_drops_by_more_than_0p01": minimum_dataset_delta
        >= -args.maximum_dataset_drop,
        "candidate_within_0p01_of_same_dataset_reference": gap_vs_reference
        >= -args.maximum_reference_gap,
    }
    passed = all(gates.values())
    report = {
        "experiment": "ELUDe-inspired contribution-mode split frozen causal gate",
        "status": "mechanism-preflight-not-final-architecture",
        "summary": means,
        "candidate_mean_acc": candidate,
        "best_control": best_control,
        "best_control_mean_acc": best,
        "candidate_minus_best_control": candidate - best,
        "same_dataset_reference_initial3": args.reference_initial3,
        "gap_vs_reference": gap_vs_reference,
        "per_dataset": per_dataset,
        "minimum_dataset_delta": minimum_dataset_delta,
        "gates": gates,
        "pass": passed,
        "decision": (
            "allow-joint-contribution-routed-split-sae-development"
            if passed
            else "close-contribution-mode-split-family-before-training"
        ),
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Contribution-mode split Initial3 gate",
        "",
        "| Variant | Mean Acc | Mean AUC |",
        "|---|---:|---:|",
    ]
    for key, values in means.items():
        lines.append(
            f"| {key} | {values['mean_acc']:.6f} | {values['mean_auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Best control: `{best_control}` ({best:.6f})",
            f"Candidate delta: `{candidate - best:+.6f}`",
            f"Reference gap: `{gap_vs_reference:+.6f}`",
            f"Minimum dataset delta: `{minimum_dataset_delta:+.6f}`",
            "",
            "```json",
            json.dumps(gates, indent=2),
            "```",
            "",
            f"Decision: `{report['decision']}`",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
