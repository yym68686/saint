#!/usr/bin/env python3
"""Post-gate diagnosis for the fixed C2R mechanism preflight."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


LABEL_TO_KEY = {
    "Frozen ReLU reference": "frozen",
    "Matched ReLU finetune": "finetune",
    "Wrong-alignment C2R control": "wrong",
    "True C2R preflight": "candidate",
}


def selected_feature_overlap(eval_payload: dict[str, object]) -> dict[str, object]:
    rows = {
        LABEL_TO_KEY[row["label"]]: row
        for row in eval_payload["architecture_results"]
    }
    comparisons: dict[str, object] = {}
    for control in ("frozen", "finetune", "wrong"):
        overall = {"comparisons": 0, "exact_set": 0, "exact_order": 0, "jaccard_sum": 0.0}
        datasets: dict[str, object] = {}
        for dataset, candidate_classes in rows["candidate"]["per_class"].items():
            control_classes = rows[control]["per_class"][dataset]
            counts = {"comparisons": 0, "exact_set": 0, "exact_order": 0, "jaccard_sum": 0.0}
            for class_name, candidate_k in candidate_classes.items():
                control_k = control_classes[class_name]
                for k in (1, 2, 5):
                    name = f"top_{k}"
                    left = candidate_k[name]["selected_features"]
                    right = control_k[name]["selected_features"]
                    left_set, right_set = set(left), set(right)
                    union = left_set | right_set
                    jaccard = len(left_set & right_set) / len(union) if union else 1.0
                    for target in (counts, overall):
                        target["comparisons"] += 1
                        target["exact_set"] += int(left_set == right_set)
                        target["exact_order"] += int(left == right)
                        target["jaccard_sum"] += jaccard
            counts["exact_set_fraction"] = counts["exact_set"] / counts["comparisons"]
            counts["exact_order_fraction"] = counts["exact_order"] / counts["comparisons"]
            counts["mean_jaccard"] = counts.pop("jaccard_sum") / counts["comparisons"]
            datasets[dataset] = counts
        overall["exact_set_fraction"] = overall["exact_set"] / overall["comparisons"]
        overall["exact_order_fraction"] = overall["exact_order"] / overall["comparisons"]
        overall["mean_jaccard"] = overall.pop("jaccard_sum") / overall["comparisons"]
        comparisons[f"candidate_vs_{control}"] = {"overall": overall, "datasets": datasets}
    return comparisons


def pair_stats(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    chunk_size: int = 8_000_000,
) -> dict[str, object]:
    tensor_stats: dict[str, object] = {}
    totals = {"n": 0, "diff_sq": 0.0, "left_sq": 0.0, "right_sq": 0.0, "dot": 0.0, "max": 0.0}
    for key in sorted(left):
        a = left[key].detach().reshape(-1).float()
        b = right[key].detach().reshape(-1).float()
        acc = {"n": a.numel(), "diff_sq": 0.0, "left_sq": 0.0, "right_sq": 0.0, "dot": 0.0, "max": 0.0}
        for start in range(0, a.numel(), chunk_size):
            aa = a[start : start + chunk_size]
            bb = b[start : start + chunk_size]
            diff = aa - bb
            acc["diff_sq"] += float(torch.sum(diff * diff).item())
            acc["left_sq"] += float(torch.sum(aa * aa).item())
            acc["right_sq"] += float(torch.sum(bb * bb).item())
            acc["dot"] += float(torch.sum(aa * bb).item())
            acc["max"] = max(acc["max"], float(torch.max(torch.abs(diff)).item()))
        denom = math.sqrt(acc["left_sq"] * acc["right_sq"])
        tensor_stats[key] = {
            "numel": acc["n"],
            "rms_delta": math.sqrt(acc["diff_sq"] / acc["n"]),
            "relative_rms_to_left": math.sqrt(acc["diff_sq"] / max(acc["left_sq"], 1e-30)),
            "max_abs_delta": acc["max"],
            "cosine": max(-1.0, min(1.0, acc["dot"] / denom)) if denom else 1.0,
        }
        for name in totals:
            if name == "max":
                totals[name] = max(totals[name], acc[name])
            else:
                totals[name] += acc[name]
    denom = math.sqrt(totals["left_sq"] * totals["right_sq"])
    return {
        "overall": {
            "numel": totals["n"],
            "rms_delta": math.sqrt(totals["diff_sq"] / totals["n"]),
            "relative_rms_to_left": math.sqrt(totals["diff_sq"] / max(totals["left_sq"], 1e-30)),
            "max_abs_delta": totals["max"],
            "cosine": max(-1.0, min(1.0, totals["dot"] / denom)) if denom else 1.0,
        },
        "tensors": tensor_stats,
    }


def c2r_history_differences(summary: dict[str, object]) -> dict[str, float]:
    true_history = {
        int(row["step"]): row
        for row in summary["variants"]["true_c2r"]["history"]
        if row["c2r_selected_count"] > 0
    }
    wrong_history = {
        int(row["step"]): row
        for row in summary["variants"]["wrong_alignment_c2r"]["history"]
        if row["c2r_selected_count"] > 0
    }
    steps = sorted(set(true_history) & set(wrong_history))
    fields = ("reconstruction_loss", "c2r_raw", "c2r_scaled", "neighbor_cosine_mean")
    result = {"compared_logged_steps": len(steps)}
    for field in fields:
        diffs = [abs(float(true_history[s][field]) - float(wrong_history[s][field])) for s in steps]
        result[f"mean_abs_{field}_difference"] = sum(diffs) / len(diffs)
        result[f"max_abs_{field}_difference"] = max(diffs)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eval_payload = json.loads((args.root / "initial3-eval.json").read_text())
    gate = json.loads((args.root / "initial3-gate.json").read_text())
    summary = json.loads((args.root / "train/train-summary-c2r-preflight.json").read_text())
    checkpoints = {
        "finetune": args.root / "train/trained_sae-relu_finetune.pt",
        "wrong": args.root / "train/trained_sae-wrong_alignment_c2r.pt",
        "candidate": args.root / "train/trained_sae-true_c2r.pt",
    }
    states = {
        key: torch.load(path, map_location="cpu", weights_only=True)
        for key, path in checkpoints.items()
    }
    parameter_differences = {
        "candidate_vs_wrong": pair_stats(states["candidate"], states["wrong"]),
        "candidate_vs_finetune": pair_stats(states["candidate"], states["finetune"]),
        "wrong_vs_finetune": pair_stats(states["wrong"], states["finetune"]),
    }
    validations = {
        key: value["validation"] for key, value in summary["variants"].items()
    }
    report = {
        "experiment": "fixed C2R post-gate failure diagnosis",
        "gate": gate,
        "selected_feature_overlap": selected_feature_overlap(eval_payload),
        "parameter_differences": parameter_differences,
        "c2r_logged_history_differences": c2r_history_differences(summary),
        "validation": validations,
        "causal_conclusion": (
            "True C2R failed every preregistered promotion condition, matched the wrong-alignment "
            "control on Amazon category, and mostly preserved the same selected low-k coordinates. "
            "The C2R geometry therefore does not justify parameterized-router development under "
            "the current cache and budget."
        ),
        "decision": "close-c2r-family-no-full7-no-parameterized-router",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
