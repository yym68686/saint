#!/usr/bin/env python3
"""Apply the preregistered Initial3 gate to cross-layer concordance v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


K_VALUES = (1, 2, 5)


def dataset_mean(row: dict[str, object], dataset: str) -> float:
    metrics = row["dataset_results"][dataset]
    return sum(
        float(metrics[f"sae_top_{k}_test_accuracy"]) for k in K_VALUES
    ) / len(K_VALUES)


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
    rows = {
        row["variant_key"]: row for row in payload["architecture_results"]
    }
    summary = {row["variant_key"]: row for row in payload["summary"]}
    required = {"candidate", "reconstruction_control", "wrong_alignment"}
    missing = sorted(required - set(rows))
    if missing:
        raise KeyError(f"Missing eval variants: {missing}")
    controls = ("reconstruction_control", "wrong_alignment")
    candidate_mean = float(summary["candidate"]["mean_acc"])
    control_means = {key: float(summary[key]["mean_acc"]) for key in controls}
    best_control_key = max(control_means, key=control_means.get)
    best_control_mean = control_means[best_control_key]
    datasets = list(rows["candidate"]["dataset_results"])
    per_dataset = {}
    for dataset in datasets:
        candidate_value = dataset_mean(rows["candidate"], dataset)
        control_values = {key: dataset_mean(rows[key], dataset) for key in controls}
        best_key = max(control_values, key=control_values.get)
        per_dataset[dataset] = {
            "candidate": candidate_value,
            "controls": control_values,
            "best_control": best_key,
            "delta_vs_best_control": candidate_value - control_values[best_key],
        }
    minimum_dataset_delta = min(
        row["delta_vs_best_control"] for row in per_dataset.values()
    )
    gates = {
        "training_integrity_pass": bool(integrity["pass"]),
        "candidate_beats_best_control_by_0p005": candidate_mean - best_control_mean
        >= args.minimum_control_delta,
        "no_initial3_dataset_drops_by_more_than_0p01": minimum_dataset_delta
        >= -args.maximum_dataset_drop,
        "candidate_within_0p01_of_preregistered_reference": candidate_mean
        >= args.reference_initial3 - args.maximum_reference_gap,
    }
    passed = all(gates.values())
    report = {
        "experiment": "cross-layer concordance shared SAE v1 Initial3 gate",
        "candidate_mean_acc": candidate_mean,
        "control_mean_acc": control_means,
        "best_control": best_control_key,
        "delta_vs_best_control": candidate_mean - best_control_mean,
        "preregistered_reference_initial3": args.reference_initial3,
        "gap_vs_reference": candidate_mean - args.reference_initial3,
        "per_dataset": per_dataset,
        "minimum_dataset_delta": minimum_dataset_delta,
        "gates": gates,
        "pass": passed,
        "decision": "enter-long-training" if passed else "stop-before-long-training",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Cross-layer concordance v1 Initial3 gate",
        "",
        f"- Candidate Mean Acc: `{candidate_mean:.6f}`",
        f"- Best control: `{best_control_key}` / `{best_control_mean:.6f}`",
        f"- Delta vs best control: `{candidate_mean - best_control_mean:+.6f}`",
        f"- Gap vs preregistered reference: `{candidate_mean - args.reference_initial3:+.6f}`",
        f"- Decision: **{report['decision']}**",
        "",
        "| Dataset | Candidate | Best control | Delta |",
        "|---|---:|---:|---:|",
    ]
    for dataset, row in per_dataset.items():
        best_value = row["controls"][row["best_control"]]
        lines.append(
            f"| {dataset} | {row['candidate']:.6f} | {best_value:.6f} | "
            f"{row['delta_vs_best_control']:+.6f} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- [{'x' if value else ' '}] {key}" for key, value in gates.items()
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
