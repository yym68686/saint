#!/usr/bin/env python3
"""Validate training integrity before any probing evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    variants = payload["variants"]
    required = {"relu_finetune", "wrong_alignment_c2r", "true_c2r"}
    histories = {key: variants[key]["history"] for key in required}
    leakage_keys = (
        "uses_saebench_labels_for_training",
        "uses_eval_split_for_training",
        "uses_one_vs_rest_targets_for_training",
        "uses_mean_diff_selection_for_training",
        "uses_test_feedback_for_training",
    )
    checks = {
        "all_variants_present": required <= set(variants),
        "exact_parameter_count": all(
            int(variants[key]["parameter_count"]) == 402_721_792
            for key in required
        ),
        "exact_exposed_feature_count": all(
            int(variants[key]["exposed_feature_count"]) == 65_536
            for key in required
        ),
        "all_steps_completed": all(
            int(variants[key]["steps"]) == int(payload["arguments"]["steps"])
            for key in required
        ),
        "all_parameters_updated": all(
            all(float(delta) > 0.0 for delta in variants[key]["parameter_max_delta"].values())
            for key in required
        ),
        "true_c2r_encoder_gradient_nonzero": bool(
            variants["true_c2r"]["any_encoder_c2r_gradient"]
        ),
        "true_c2r_decoder_gradient_nonzero": bool(
            variants["true_c2r"]["any_decoder_c2r_gradient"]
        ),
        "wrong_c2r_encoder_gradient_nonzero": bool(
            variants["wrong_alignment_c2r"]["any_encoder_c2r_gradient"]
        ),
        "wrong_c2r_decoder_gradient_nonzero": bool(
            variants["wrong_alignment_c2r"]["any_decoder_c2r_gradient"]
        ),
        "wrong_alignment_has_no_fixed_pairs": int(
            variants["wrong_alignment_c2r"]["wrong_fixed_pair_count_total"]
        )
        == 0,
        "all_logged_values_finite": all(
            math.isfinite(float(value))
            for rows in histories.values()
            for row in rows
            for value in row.values()
        ),
        "data_manifest_unchanged": bool(payload["data_unchanged"]),
        "same_parameter_and_feature_budget": all(
            bool(payload["fairness"][key])
            for key in ("same_parameter_count", "same_exposed_feature_count")
        ),
        "no_label_or_eval_leakage": all(
            not bool(payload["fairness"][key]) for key in leakage_keys
        ),
    }
    report = {
        "summary": str(args.summary),
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit(f"Training integrity failed: {failed}")


if __name__ == "__main__":
    main()
