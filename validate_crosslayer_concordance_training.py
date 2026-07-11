#!/usr/bin/env python3
"""Reject a cross-layer run before probing if causal or training integrity fails."""

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
    required = {"candidate", "reconstruction_control", "wrong_alignment"}
    missing = sorted(required - set(variants))
    if missing:
        raise KeyError(f"Missing variants: {missing}")
    counts = {int(variants[key]["parameter_count"]) for key in required}
    source_count = int(payload["source_parameter_count"])
    histories = {
        key: variants[key].get("history", []) for key in required
    }
    all_logged_values_finite = all(
        math.isfinite(float(value))
        for rows in histories.values()
        for row in rows
        for value in row.values()
    )
    checks = {
        "three_variants_present": True,
        "same_parameter_count": len(counts) == 1,
        "matches_source_parameter_count": counts == {source_count},
        "same_exposed_feature_count": int(payload["exposed_feature_count_each"])
        == 65_536,
        "same_initial_tensors": bool(payload["same_initial_tensors"]),
        "cache_read_only": bool(payload["cache_read_only"]),
        "all_logged_values_finite": all_logged_values_finite,
        "all_variants_completed_steps": all(
            int(variants[key]["global_steps"])
            == int(payload["arguments"]["steps"])
            for key in required
        ),
        "all_backbones_updated": all(
            float(variants[key]["backbone_parameter_max_delta"]) > 0.0
            for key in required
        ),
        "all_calibration_modules_updated": all(
            float(variants[key]["calibration_parameter_max_delta"]) > 0.0
            for key in required
        ),
        "all_dictionary_gradients_nonzero": all(
            float(histories[key][-1]["dictionary_grad_norm"]) > 0.0
            for key in required
        ),
        "all_calibration_gradients_nonzero": all(
            float(histories[key][-1]["calibration_grad_norm"]) > 0.0
            for key in required
        ),
        "reconstruction_control_has_no_concordance": all(
            abs(float(row["concordance_loss"])) < 1.0e-12
            for row in histories["reconstruction_control"]
        ),
        "candidate_has_true_concordance": any(
            float(row["concordance_loss"]) > 0.0
            for row in histories["candidate"]
        ),
        "wrong_control_has_concordance": any(
            float(row["concordance_loss"]) > 0.0
            for row in histories["wrong_alignment"]
        ),
        "true_and_wrong_objective_scales_are_equal": all(
            abs(float(row["normalized_concordance_objective"]) - 1.0) < 1.0e-6
            for key in ("candidate", "wrong_alignment")
            for row in histories[key]
        ),
        "true_and_wrong_weighted_objectives_are_equal": all(
            abs(
                float(candidate_row["weighted_concordance_objective"])
                - float(wrong_row["weighted_concordance_objective"])
            )
            < 1.0e-8
            for candidate_row, wrong_row in zip(
                histories["candidate"],
                histories["wrong_alignment"],
                strict=True,
            )
        ),
        "wrong_alignment_no_fixed_pairs": int(
            variants["wrong_alignment"]["wrong_fixed_pair_count"]
        )
        == 0,
        "wrong_alignment_no_same_sample_pairs": int(
            variants["wrong_alignment"]["wrong_same_sample_pair_count"]
        )
        == 0,
        "no_label_or_eval_leakage": all(
            not bool(payload["fairness"][key])
            for key in (
                "uses_saebench_labels_for_training",
                "uses_eval_split_for_training",
                "uses_one_vs_rest_targets_for_training",
                "uses_mean_diff_selection_for_training",
                "uses_test_feedback_for_training",
            )
        ),
    }
    report = {
        "summary": str(args.summary),
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit(f"Training-integrity checks failed: {failed}")


if __name__ == "__main__":
    main()
