#!/usr/bin/env python3
"""Apply the preregistered Initial3 gate to the fixed C2R preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "frozen": "Frozen ReLU reference",
    "finetune": "Matched ReLU finetune",
    "wrong": "Wrong-alignment C2R control",
    "candidate": "True C2R preflight",
}


def dataset_mean(row: dict[str, object], dataset: str, metric: str) -> float:
    values = row["dataset_results"][dataset]
    return sum(float(values[f"sae_top_{k}_test_{metric}"]) for k in (1, 2, 5)) / 3


def overall_mean(row: dict[str, object], datasets: list[str], metric: str) -> float:
    return sum(dataset_mean(row, dataset, metric) for dataset in datasets) / len(datasets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--training-integrity", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--minimum-control-delta", type=float, default=0.005)
    parser.add_argument("--maximum-dataset-drop", type=float, default=0.01)
    parser.add_argument("--reference-initial3", type=float, default=0.837543)
    parser.add_argument("--maximum-reference-gap", type=float, default=0.01)
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    integrity = json.loads(args.training_integrity.read_text(encoding="utf-8"))
    rows = {row["label"]: row for row in payload["architecture_results"]}
    missing = sorted(set(LABELS.values()) - set(rows))
    if missing:
        raise KeyError(f"Missing evaluated variants: {missing}")
    datasets = list(payload["config"]["dataset_names"])
    keyed = {key: rows[label] for key, label in LABELS.items()}
    means = {
        key: {
            "mean_acc": overall_mean(row, datasets, "accuracy"),
            "mean_auc": overall_mean(row, datasets, "auc"),
        }
        for key, row in keyed.items()
    }
    control_keys = ("frozen", "finetune", "wrong")
    best_control = max(control_keys, key=lambda key: means[key]["mean_acc"])
    candidate_mean = means["candidate"]["mean_acc"]
    best_control_mean = means[best_control]["mean_acc"]
    per_dataset = {}
    for dataset in datasets:
        candidate_value = dataset_mean(keyed["candidate"], dataset, "accuracy")
        control_values = {
            key: dataset_mean(keyed[key], dataset, "accuracy")
            for key in control_keys
        }
        dataset_best_key = max(control_values, key=control_values.get)
        per_dataset[dataset] = {
            "candidate": candidate_value,
            "controls": control_values,
            "best_control": dataset_best_key,
            "delta_vs_best_control": candidate_value
            - control_values[dataset_best_key],
        }
    minimum_dataset_delta = min(
        row["delta_vs_best_control"] for row in per_dataset.values()
    )
    gap_vs_reference = candidate_mean - args.reference_initial3
    gates = {
        "training_integrity_pass": bool(integrity["pass"]),
        "candidate_beats_best_control_by_0p005": candidate_mean
        - best_control_mean
        >= args.minimum_control_delta,
        "no_initial3_dataset_drops_by_more_than_0p01": minimum_dataset_delta
        >= -args.maximum_dataset_drop,
        "candidate_within_0p01_of_same_dataset_reference": gap_vs_reference
        >= -args.maximum_reference_gap,
    }
    passed = all(gates.values())
    report = {
        "experiment": "fixed C2R causal preflight Initial3 gate",
        "status": "mechanism-preflight-not-final-architecture",
        "summary": means,
        "candidate_mean_acc": candidate_mean,
        "best_control": best_control,
        "best_control_mean_acc": best_control_mean,
        "candidate_minus_best_control": candidate_mean - best_control_mean,
        "same_dataset_reference_initial3": args.reference_initial3,
        "gap_vs_reference": gap_vs_reference,
        "per_dataset": per_dataset,
        "minimum_dataset_delta": minimum_dataset_delta,
        "gates": gates,
        "pass": passed,
        "decision": (
            "allow-parameterized-c2r-router-development"
            if passed
            else "stop-before-parameterized-c2r-router-development"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fixed C2R causal preflight Initial3 gate",
        "",
        "| Variant | Mean Acc | Mean AUC |",
        "|---|---:|---:|",
    ]
    for key, values in means.items():
        lines.append(f"| {key} | {values['mean_acc']:.6f} | {values['mean_auc']:.6f} |")
    lines.extend(
        [
            "",
            f"- Candidate minus best control: {candidate_mean - best_control_mean:+.6f}",
            f"- Gap versus same-dataset reference: {gap_vs_reference:+.6f}",
            f"- Minimum dataset delta: {minimum_dataset_delta:+.6f}",
            f"- Decision: `{report['decision']}`",
            "",
            "| Dataset | Candidate | Best control | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset, row in per_dataset.items():
        best = row["best_control"]
        lines.append(
            f"| {dataset} | {row['candidate']:.6f} | "
            f"{row['controls'][best]:.6f} ({best}) | "
            f"{row['delta_vs_best_control']:+.6f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
