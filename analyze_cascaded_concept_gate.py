#!/usr/bin/env python3
"""Apply the preregistered Initial3 gate to Cascaded Concept SAE v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def mean_accuracy(metrics: dict[str, float]) -> float:
    return sum(metrics[f"sae_top_{k}_test_accuracy"] for k in (1, 2, 5)) / 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    train = json.loads(args.train_summary.read_text(encoding="utf-8"))
    evaluation = json.loads(args.eval_json.read_text(encoding="utf-8"))
    rows = {row["variant_key"]: row for row in evaluation["summary"]}
    details = {
        row["variant_key"]: row
        for row in evaluation["architecture_results"]
        if row["status"] == "ok"
    }
    required = {
        "v396_finetune",
        "cascaded_concept",
        "cascaded_concept_wrong_hierarchy",
        "cascaded_concept_level1_only",
    }
    missing = sorted(required - set(rows))
    if missing:
        raise KeyError(f"Evaluation is missing {missing}")

    candidate = rows["cascaded_concept"]["mean_acc"]
    base = rows["v396_finetune"]["mean_acc"]
    control_names = (
        "cascaded_concept_wrong_hierarchy",
        "cascaded_concept_level1_only",
    )
    best_control_name = max(control_names, key=lambda name: rows[name]["mean_acc"])
    best_control = rows[best_control_name]["mean_acc"]
    dataset_deltas = {}
    for dataset_name, metrics in details["cascaded_concept"][
        "dataset_results"
    ].items():
        dataset_deltas[dataset_name] = mean_accuracy(metrics) - mean_accuracy(
            details["v396_finetune"]["dataset_results"][dataset_name]
        )

    candidate_train = train["results"]["cascaded_concept"]
    report: dict[str, Any] = {
        "candidate_mean_acc": candidate,
        "v396_finetune_mean_acc": base,
        "best_hierarchy_control": best_control_name,
        "best_hierarchy_control_mean_acc": best_control,
        "delta_over_v396_finetune": candidate - base,
        "delta_over_best_hierarchy_control": candidate - best_control,
        "dataset_deltas_over_v396_finetune": dataset_deltas,
        "same_dataset_v396_reference": 0.837543,
        "delta_vs_same_dataset_v396_reference": candidate - 0.837543,
        "parameter_matched": train["parameter_matched"],
        "candidate_parameter_count": train["parameter_count_candidate"],
        "control_parameter_count": train["parameter_count_control"],
        "candidate_final_low_gradient_norm": candidate_train[
            "final_low_gradient_norm"
        ],
        "candidate_final_high_gradient_norm": candidate_train[
            "final_high_gradient_norm"
        ],
        "candidate_module_parameter_delta": candidate_train[
            "module_parameter_delta"
        ],
    }
    report["effect_pass"] = report["delta_over_v396_finetune"] >= 0.005
    report["hierarchy_control_pass"] = (
        report["delta_over_best_hierarchy_control"] >= 0.002
    )
    report["dataset_pass"] = min(dataset_deltas.values()) >= -0.01
    report["reference_pass"] = (
        report["delta_vs_same_dataset_v396_reference"] >= -0.01
    )
    report["parameter_pass"] = bool(report["parameter_matched"])
    report["gradient_pass"] = (
        report["candidate_final_low_gradient_norm"] > 0
        and report["candidate_final_high_gradient_norm"] > 0
        and report["candidate_module_parameter_delta"] > 0
    )
    report["pass"] = all(
        report[key]
        for key in (
            "effect_pass",
            "hierarchy_control_pass",
            "dataset_pass",
            "reference_pass",
            "parameter_pass",
            "gradient_pass",
        )
    )
    report["decision"] = (
        "authorize-long-training"
        if report["pass"]
        else "stop-before-long-training-and-full7"
    )
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Cascaded Concept SAE v1 Initial3 Gate",
        "",
        "| Variant | Mean Acc |",
        "|---|---:|",
    ]
    for name in sorted(required, key=lambda item: rows[item]["mean_acc"], reverse=True):
        lines.append(f"| {name} | {rows[name]['mean_acc']:.6f} |")
    lines.extend(
        [
            "",
            f"Decision: `{report['decision']}`",
            "",
            "| Gate | Pass |",
            "|---|:---:|",
            f"| candidate - V396 >= +0.005 | {report['effect_pass']} |",
            f"| candidate - hierarchy control >= +0.002 | {report['hierarchy_control_pass']} |",
            f"| each dataset delta >= -0.01 | {report['dataset_pass']} |",
            f"| within 0.01 of registered V396 reference | {report['reference_pass']} |",
            f"| exact parameter parity | {report['parameter_pass']} |",
            f"| nonzero gradients and parameter update | {report['gradient_pass']} |",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
