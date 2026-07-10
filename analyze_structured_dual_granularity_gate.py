#!/usr/bin/env python3
"""Apply the preregistered Initial3 gate to a dual-granularity candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    rows = {
        row["variant_key"]: row
        for row in payload["architecture_results"]
    }
    base = rows["base"]
    candidate = rows["candidate"]
    dataset_deltas = {}
    for dataset, candidate_metrics in candidate["dataset_results"].items():
        base_metrics = base["dataset_results"][dataset]
        candidate_mean = sum(
            candidate_metrics[f"sae_top_{k}_test_accuracy"]
            for k in (1, 2, 5)
        ) / 3
        base_mean = sum(
            base_metrics[f"sae_top_{k}_test_accuracy"]
            for k in (1, 2, 5)
        ) / 3
        dataset_deltas[dataset] = {
            "base_mean_acc": base_mean,
            "candidate_mean_acc": candidate_mean,
            "delta": candidate_mean - base_mean,
        }
    summary = {
        row["variant_key"]: row
        for row in payload["summary"]
    }
    overall_delta = (
        summary["candidate"]["mean_acc"] - summary["base"]["mean_acc"]
    )
    gate = {
        "overall_delta_at_least_0p005": overall_delta >= 0.005,
        "no_dataset_drop_below_minus_0p01": min(
            row["delta"] for row in dataset_deltas.values()
        )
        >= -0.01,
    }
    gate["pass"] = all(gate.values())
    report = {
        "initial3": [
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "fancyzhx/ag_news",
        ],
        "base_mean_acc": summary["base"]["mean_acc"],
        "candidate_mean_acc": summary["candidate"]["mean_acc"],
        "overall_delta": overall_delta,
        "dataset_deltas": dataset_deltas,
        "gate": gate,
        "decision": "allow-full7" if gate["pass"] else "reject-variant",
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Structured dual-granularity Initial3 gate",
        "",
        f"- Base Mean Acc: `{report['base_mean_acc']:.6f}`",
        f"- Candidate Mean Acc: `{report['candidate_mean_acc']:.6f}`",
        f"- Delta: `{overall_delta:+.6f}`",
        "",
        "| Dataset | Base ↑ | Candidate ↑ | Delta |",
        "|---|---:|---:|---:|",
    ]
    for dataset, row in dataset_deltas.items():
        lines.append(
            f"| {dataset} | {row['base_mean_acc']:.6f} | "
            f"{row['candidate_mean_acc']:.6f} | {row['delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Overall delta >= +0.005: `{gate['overall_delta_at_least_0p005']}`",
            f"- No dataset drop < -0.01: `{gate['no_dataset_drop_below_minus_0p01']}`",
            f"- Decision: **{report['decision']}**",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
