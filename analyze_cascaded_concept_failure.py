#!/usr/bin/env python3
"""Diagnose a failed Cascaded Concept SAE Initial3 gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def selected_cells(row: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output = {}
    for dataset, classes in row["per_class"].items():
        for class_name, k_results in classes.items():
            for k_name, values in k_results.items():
                output[(dataset, class_name, k_name)] = values
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--activity-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    evaluation = json.loads(args.eval_json.read_text(encoding="utf-8"))
    activity = json.loads(args.activity_json.read_text(encoding="utf-8"))
    rows = {row["variant_key"]: row for row in evaluation["architecture_results"]}
    summaries = {row["variant_key"]: row for row in evaluation["summary"]}
    required = {
        "v396_finetune",
        "cascaded_concept",
        "cascaded_concept_wrong_hierarchy",
        "cascaded_concept_level1_only",
    }
    if missing := sorted(required - set(rows)):
        raise KeyError(f"Missing evaluation rows: {missing}")

    parent = state["cascaded.parent"].long()
    wrong_parent = state["cascaded.wrong_parent"].long()
    counts = torch.bincount(parent, minlength=int(state["cascaded.n_high"]))
    active_counts = counts[counts > 0].float()
    probabilities = active_counts / active_counts.sum()
    entropy = -(probabilities * probabilities.log()).sum()
    effective_parents = entropy.exp()
    largest_count = int(active_counts.max().item())

    source_to_wrong_cardinality = []
    for source in torch.unique(parent):
        source_to_wrong_cardinality.append(
            int(torch.unique(wrong_parent[parent == source]).numel())
        )
    wrong_is_only_label_permutation = all(
        cardinality == 1 for cardinality in source_to_wrong_cardinality
    )

    n_low = int(state["cascaded.n_low"].item())
    selected = {}
    high_selected = {}
    for name in required:
        cells = selected_cells(rows[name])
        selected[name] = cells
        high_selected[name] = [
            {
                "dataset": key[0],
                "class": key[1],
                "k": key[2],
                "feature": int(feature),
                "test_accuracy": float(values["test_accuracy"]),
                "test_auc": float(values["test_auc"]),
            }
            for key, values in cells.items()
            for feature in values["selected_features"]
            if int(feature) >= n_low
        ]

    changed_cells = []
    base_cells = selected["v396_finetune"]
    candidate_cells = selected["cascaded_concept"]
    for key, base in base_cells.items():
        candidate = candidate_cells[key]
        if (
            base["selected_features"] != candidate["selected_features"]
            or base["test_accuracy"] != candidate["test_accuracy"]
            or base["test_auc"] != candidate["test_auc"]
        ):
            changed_cells.append(
                {
                    "dataset": key[0],
                    "class": key[1],
                    "k": key[2],
                    "base_features": base["selected_features"],
                    "candidate_features": candidate["selected_features"],
                    "accuracy_delta": candidate["test_accuracy"]
                    - base["test_accuracy"],
                    "auc_delta": candidate["test_auc"] - base["test_auc"],
                }
            )

    bottom = set(activity["bottom3072_indices"])
    base_selected_occurrences = [
        int(feature)
        for values in base_cells.values()
        for feature in values["selected_features"]
    ]
    report = {
        "candidate_mean_acc": summaries["cascaded_concept"]["mean_acc"],
        "base_mean_acc": summaries["v396_finetune"]["mean_acc"],
        "level1_only_mean_acc": summaries["cascaded_concept_level1_only"][
            "mean_acc"
        ],
        "learned_minus_base": summaries["cascaded_concept"]["mean_acc"]
        - summaries["v396_finetune"]["mean_acc"],
        "learned_minus_level1_only": summaries["cascaded_concept"]["mean_acc"]
        - summaries["cascaded_concept_level1_only"]["mean_acc"],
        "hierarchy": {
            "children": int(parent.numel()),
            "available_parents": int(counts.numel()),
            "active_parents": int((counts > 0).sum().item()),
            "effective_parents_entropy": float(effective_parents.item()),
            "largest_cluster_count": largest_count,
            "largest_cluster_share": largest_count / int(parent.numel()),
            "normalized_entropy": float(
                entropy.item() / math.log(int((counts > 0).sum().item()))
            ),
        },
        "wrong_hierarchy_control": {
            "fixed_children": int((parent == wrong_parent).sum().item()),
            "source_cluster_maps_to_one_wrong_cluster": wrong_is_only_label_permutation,
            "valid_membership_shuffle_control": not wrong_is_only_label_permutation,
            "metrics_identical_to_learned": summaries["cascaded_concept"][
                "mean_acc"
            ]
            == summaries["cascaded_concept_wrong_hierarchy"]["mean_acc"],
        },
        "high_feature_selections": high_selected,
        "changed_cells_vs_base": changed_cells,
        "label_free_activity_mask": {
            "tokens": activity["protocol"]["tokens"],
            "all_features_ever_active": activity["ever_active_features"]
            == activity["features"],
            "old_tail_zero_count": activity["old_tail_zero_count"],
            "bottom3072_overlap_old_tail": activity[
                "bottom3072_overlap_old_tail"
            ],
            "base_selected_occurrences_removed_by_bottom3072": sum(
                feature in bottom for feature in base_selected_occurrences
            ),
            "critical": activity["critical"],
        },
        "decision": (
            "close-v1; v2 is justified only by label-free low-activity slot "
            "reallocation plus an anti-collapse objective registered before eval"
        ),
    }
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Cascaded Concept SAE v1 Failure Diagnosis",
        "",
        f"- Candidate Initial3 Mean Acc: `{report['candidate_mean_acc']:.6f}`",
        f"- Same-parameter V396 Mean Acc: `{report['base_mean_acc']:.6f}`",
        f"- Delta: `{report['learned_minus_base']:+.6f}`",
        f"- Level-2 contribution over Level-1 only: "
        f"`{report['learned_minus_level1_only']:+.6f}`",
        f"- Active/effective Level-2 parents: "
        f"`{report['hierarchy']['active_parents']}` / "
        f"`{report['hierarchy']['effective_parents_entropy']:.2f}`",
        f"- Largest parent share: "
        f"`{report['hierarchy']['largest_cluster_share']:.2%}`",
        f"- Changed class-by-k cells versus V396: `{len(changed_cells)}`",
        f"- Lowest-activity replacement slots overlapping V396 selected "
        f"occurrences: "
        f"`{report['label_free_activity_mask']['base_selected_occurrences_removed_by_bottom3072']}`",
        "",
        "The registered wrong-hierarchy readout is a label permutation of the "
        "same partition, so its equal score is not causal evidence. The v1 "
        "gate failure against V396 remains valid independently of that control.",
        "",
        f"Decision: `{report['decision']}`",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
