#!/usr/bin/env python3
"""Apply the preregistered Initial3 gate to next-token predictive SAE v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTROL_KEYS = ("detach_control", "wrong_alignment")


def mean_dataset_acc(metrics: dict[str, float]) -> float:
    return sum(
        metrics[f"sae_top_{k}_test_accuracy"] for k in (1, 2, 5)
    ) / 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {
        row["variant_key"]: row for row in payload["architecture_results"]
    }
    summary = {row["variant_key"]: row for row in payload["summary"]}
    required = {"candidate", *CONTROL_KEYS}
    missing = sorted(required - set(rows))
    if missing:
        raise KeyError(f"Missing variants: {missing}")

    candidate = summary["candidate"]
    best_control_key = max(
        CONTROL_KEYS,
        key=lambda key: summary[key]["mean_acc"],
    )
    best_control = summary[best_control_key]
    dataset_rows: dict[str, dict[str, float | str]] = {}
    minimum_delta = float("inf")
    for dataset, candidate_metrics in rows["candidate"]["dataset_results"].items():
        candidate_mean = mean_dataset_acc(candidate_metrics)
        control_means = {
            key: mean_dataset_acc(rows[key]["dataset_results"][dataset])
            for key in CONTROL_KEYS
        }
        best_dataset_control_key = max(control_means, key=control_means.__getitem__)
        best_dataset_control = control_means[best_dataset_control_key]
        delta = candidate_mean - best_dataset_control
        minimum_delta = min(minimum_delta, delta)
        dataset_rows[dataset] = {
            "candidate_mean_acc": candidate_mean,
            "best_control_key": best_dataset_control_key,
            "best_control_mean_acc": best_dataset_control,
            "delta": delta,
            **{
                f"{key}_mean_acc": value for key, value in control_means.items()
            },
        }

    delta_best = candidate["mean_acc"] - best_control["mean_acc"]
    delta_detach = candidate["mean_acc"] - summary["detach_control"]["mean_acc"]
    delta_wrong = candidate["mean_acc"] - summary["wrong_alignment"]["mean_acc"]
    gate = {
        "candidate_minus_best_control_at_least_0p005": delta_best >= 0.005,
        "candidate_minus_wrong_alignment_at_least_0p002": delta_wrong >= 0.002,
        "no_dataset_drop_below_minus_0p01_vs_best_control": minimum_delta >= -0.01,
    }
    gate["pass"] = all(gate.values())
    report = {
        "candidate_mean_acc": candidate["mean_acc"],
        "best_control_key": best_control_key,
        "best_control_mean_acc": best_control["mean_acc"],
        "candidate_minus_best_control": delta_best,
        "candidate_minus_detach_control": delta_detach,
        "candidate_minus_wrong_alignment": delta_wrong,
        "minimum_dataset_delta_vs_best_control": minimum_delta,
        "dataset_deltas": dataset_rows,
        "gate": gate,
        "decision": (
            "authorize-separately-preregistered-long-train"
            if gate["pass"]
            else "stop-next-token-predictive-v1-before-long-train"
        ),
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Next-token predictive SAE v1 Initial3 gate",
        "",
        "| Variant | Mean Acc | Mean AUC | Top-1 | Top-2 | Top-5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("candidate", "detach_control", "wrong_alignment"):
        row = summary[key]
        lines.append(
            f"| {key} | {row['mean_acc']:.6f} | {row['mean_auc']:.6f} | "
            f"{row['top_1_acc']:.6f} | {row['top_2_acc']:.6f} | "
            f"{row['top_5_acc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Dataset | Candidate | Best matched control | Control | Delta |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for dataset, row in dataset_rows.items():
        lines.append(
            f"| {dataset} | {row['candidate_mean_acc']:.6f} | "
            f"{row['best_control_mean_acc']:.6f} | {row['best_control_key']} | "
            f"{row['delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Candidate - best control: `{delta_best:+.6f}`",
            f"- Candidate - detach control: `{delta_detach:+.6f}`",
            f"- Candidate - wrong alignment: `{delta_wrong:+.6f}`",
            f"- Gate: `{gate['pass']}`",
            f"- Decision: **{report['decision']}**",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
