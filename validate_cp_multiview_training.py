#!/usr/bin/env python3
"""Validate CP multi-view training integrity before probing."""

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
    required = {"candidate", "mean_pooled_control", "wrong_alignment"}
    missing = sorted(required - set(variants))
    if missing:
        raise KeyError(f"Missing variants: {missing}")
    counts = {int(variants[key]["parameter_count"]) for key in required}
    histories = {key: variants[key]["history"] for key in required}
    checks = {
        "three_trained_variants_present": True,
        "same_parameter_count": len(counts) == 1,
        "matches_source_parameter_count": counts
        == {int(payload["source_parameter_count"])},
        "same_exposed_feature_count": int(payload["exposed_feature_count_each"])
        == 65_536,
        "cache_read_only": bool(payload["cache_read_only"]),
        "cp_rank_is_preregistered": int(payload["cp_initialization"]["rank"]) == 2_934,
        "all_logged_values_finite": all(
            math.isfinite(float(value))
            for rows in histories.values()
            for row in rows
            for value in row.values()
        ),
        "all_variants_completed_steps": all(
            int(variants[key]["global_steps"])
            == int(payload["arguments"]["steps"])
            for key in required
        ),
        "all_core_parameters_updated": all(
            float(variants[key]["core_parameter_max_delta"]) > 0.0
            for key in required
        ),
        "cp_modules_updated": all(
            float(variants[key]["module_parameter_max_delta"]) > 0.0
            for key in ("candidate", "wrong_alignment")
        ),
        "all_core_gradients_nonzero": all(
            float(histories[key][-1]["core_grad_norm"]) > 0.0
            for key in required
        ),
        "cp_module_gradients_nonzero": all(
            float(histories[key][-1]["module_grad_norm"]) > 0.0
            for key in ("candidate", "wrong_alignment")
        ),
        "wrong_control_has_no_fixed_view_pairs": int(
            variants["wrong_alignment"]["wrong_view_fixed_pair_count"]
        )
        == 0,
        "projection_errors_are_finite": all(
            math.isfinite(float(payload["cp_initialization"][key]))
            for key in (
                "encoder_projection_relative_frobenius_error",
                "decoder_projection_relative_frobenius_error",
            )
        ),
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
    report = {"summary": str(args.summary), "checks": checks, "pass": all(checks.values())}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit(f"Training integrity failed: {failed}")


if __name__ == "__main__":
    main()
