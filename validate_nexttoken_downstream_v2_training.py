#!/usr/bin/env python3
"""Validate training integrity before sparse probing evaluation."""

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
    exposed = int(payload["exposed_feature_count_each"])
    tolerance = float(payload["arguments"]["source_reproduction_tolerance"])
    checks: dict[str, bool] = {
        "three_variants_present": True,
        "same_parameter_count": len(counts) == 1,
        "same_exposed_feature_count_declared": bool(
            payload["fairness"]["same_exposed_feature_count"]
        )
        and exposed == 65536,
        "same_initial_tensors": bool(payload["same_initial_tensors"]),
        "cache_read_only": bool(payload["cache_read_only"]),
        "wrong_alignment_no_fixed_pairs": int(
            variants["wrong_alignment"]["wrong_fixed_pair_count"]
        )
        == 0,
        "wrong_alignment_no_same_sample_pairs": int(
            variants["wrong_alignment"]["wrong_same_sample_pair_count"]
        )
        == 0,
        "all_source_reproduction_checks_pass": all(
            float(variants[key]["source_reproduction_max_abs"]) <= tolerance
            for key in required
        ),
        "all_trunks_updated": all(
            float(variants[key]["trunk_parameter_max_delta"]) > 0.0
            for key in required
        ),
        "all_context_modules_updated": all(
            float(variants[key]["context_parameter_max_delta"]) > 0.0
            for key in required
        ),
        "all_logged_values_finite": all(
            math.isfinite(float(value))
            for key in required
            for row in variants[key]["logs"]
            for value in row.values()
            if isinstance(value, (int, float))
        ),
        "no_training_leakage_declared": not any(
            bool(payload["fairness"][field])
            for field in (
                "uses_saebench_labels_for_training",
                "uses_eval_split_for_training",
                "uses_one_vs_rest_targets_for_training",
                "uses_mean_diff_selection_for_training",
                "uses_test_feedback_for_training",
            )
        ),
        "downstream_lm_frozen": bool(payload["fairness"]["downstream_lm_frozen"]),
    }
    report = {
        "pass": all(checks.values()),
        "checks": checks,
        "parameter_count_each": sorted(counts),
        "exposed_feature_count_each": exposed,
        "maximum_source_reproduction_error": max(
            float(variants[key]["source_reproduction_max_abs"])
            for key in required
        ),
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
