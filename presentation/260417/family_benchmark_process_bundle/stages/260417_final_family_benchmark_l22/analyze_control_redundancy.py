import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_l22_family_benchmark import (
    METHOD_LABELS,
    build_batches,
    build_llama_model,
    find_budget_threshold,
    load_registry,
    load_sae_model_for_method,
    load_split_text_pool,
    method_sort_key,
    reject_rate,
    score_text_pool_for_method,
    set_seed,
    tokenize_texts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all ontology-matched candidate features to derive non-proxy control redundancy metrics."
    )
    parser.add_argument("--llama_model_dir", type=Path, required=True)
    parser.add_argument("--weights_dir", type=Path, required=True)
    parser.add_argument("--splits_root", type=Path, required=True)
    parser.add_argument("--feature_registry", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_token_length", type=int, default=192)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--max_batch_tokens", type=int, default=1024)
    parser.add_argument("--sae_top_k", type=int, default=64)
    parser.add_argument("--sae_normalization_eps", type=float, default=1e-6)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--target_thresholds", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, nargs="*", default=None)
    parser.add_argument("--families", type=str, nargs="*", default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_candidate_feature(
    score_column: np.ndarray,
    calibration_indices: list[int],
    evaluation_target_indices: list[int],
    evaluation_control_indices: list[int],
    budgets: list[float],
) -> list[dict[str, float]]:
    calibration_scores = score_column[np.array(calibration_indices)]
    evaluation_target_scores = score_column[np.array(evaluation_target_indices)]
    evaluation_control_scores = score_column[np.array(evaluation_control_indices)]

    rows = []
    for alpha in budgets:
        tau = find_budget_threshold(calibration_scores, alpha)
        target_reject = reject_rate(evaluation_target_scores, tau)
        control_reject = reject_rate(evaluation_control_scores, tau)
        rows.append(
            {
                "budget": float(alpha),
                "threshold": float(tau),
                "target_reject_rate": float(target_reject),
                "control_reject_rate": float(control_reject),
            }
        )
    return rows


def render_summary_markdown(
    metric_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    budget_2 = [row for row in metric_rows if abs(float(row["budget"]) - 0.02) < 1e-12]
    lines = [
        "# Control Redundancy Metric Sweep",
        "",
        "This report enumerates non-proxy control metrics derived from evaluating all ontology-matched candidate features on the finalized family benchmark splits.",
        "",
        "## 2% Budget Highlights",
        "",
    ]
    for threshold in sorted({float(row["target_threshold"]) for row in budget_2}):
        subset = [row for row in budget_2 if abs(float(row["target_threshold"]) - threshold) < 1e-12]
        subset.sort(key=lambda row: (-float(row["redundant_controller_family_rate"]), method_sort_key(row["method"])))
        top = subset[0]
        lines.append(
            f"- `budget=2%`, `min_target={threshold:.0%}`: best redundant-controller family rate is **{top['method_label']}** at {float(top['redundant_controller_family_rate']) * 100:.1f}%."
        )
    lines.extend(
        [
            "",
            "## Candidate Metric Definition",
            "",
            "- `Strict-Budget Redundant Controller Family Rate`: among the fixed 15 benchmark families, the share of families for which a method has at least 2 distinct candidate features that each satisfy the held-out control budget and the minimum target reject threshold.",
            "- This is a true control metric rather than a candidate-space proxy, because every counted controller is calibrated on `calibration_control` and validated on held-out `evaluation_target` and `evaluation_control`.",
        ]
    )
    (output_dir / "control_redundancy_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_methods = set(args.methods) if args.methods else None
    keep_families = set(args.families) if args.families else None

    set_seed(args.seed)

    registry_rows = load_registry(args.feature_registry, keep_methods=keep_methods, keep_families=keep_families)
    _, texts, split_indices, split_counts = load_split_text_pool(args.splits_root, keep_families=keep_families)

    if not registry_rows:
        raise RuntimeError("Feature registry is empty after applying filters.")

    device = torch.device(args.device)
    tokenizer, llama_model = build_llama_model(args.llama_model_dir, capture_layer_idx=22, device=device)
    tokenized_texts, lengths = tokenize_texts(tokenizer, texts, args.max_token_length)
    batches = build_batches(lengths, args.max_batch_size, args.max_batch_tokens)

    rows_by_method_family: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in registry_rows:
        rows_by_method_family[(row.method, row.family_id)].append(row)

    per_feature_rows: list[dict[str, Any]] = []

    methods = sorted({row.method for row in registry_rows}, key=method_sort_key)
    for method in methods:
        method_rows = [row for row in registry_rows if row.method == method]
        candidate_features = sorted({row.feature_index for row in method_rows})
        feature_to_column = {feature_index: idx for idx, feature_index in enumerate(candidate_features)}

        sae_model = load_sae_model_for_method(
            method=method,
            weights_dir=args.weights_dir,
            sae_top_k=args.sae_top_k,
            sae_normalization_eps=args.sae_normalization_eps,
            device=device,
        )
        score_matrix = score_text_pool_for_method(
            method=method,
            candidate_features=candidate_features,
            llama_model=llama_model,
            tokenizer=tokenizer,
            sae_model=sae_model,
            layer_idx=22,
            tokenized_texts=tokenized_texts,
            batches=batches,
            device=device,
        )

        families = sorted({row.family_id for row in method_rows})
        for family_id in families:
            selection_target_indices = split_indices.get((family_id, "selection_target"), [])
            selection_control_indices = split_indices.get((family_id, "selection_control"), [])
            calibration_indices = split_indices.get((family_id, "calibration_control"), [])
            evaluation_target_indices = split_indices.get((family_id, "evaluation_target"), [])
            evaluation_control_indices = split_indices.get((family_id, "evaluation_control"), [])

            for row in rows_by_method_family[(method, family_id)]:
                column = feature_to_column[row.feature_index]
                per_budget = evaluate_candidate_feature(
                    score_column=score_matrix[:, column],
                    calibration_indices=calibration_indices,
                    evaluation_target_indices=evaluation_target_indices,
                    evaluation_control_indices=evaluation_control_indices,
                    budgets=args.budgets,
                )
                for budget_row in per_budget:
                    per_feature_rows.append(
                        {
                            "family_id": family_id,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "feature_index": row.feature_index,
                            "certainty": row.certainty,
                            "display_name": row.display_name,
                            "category": row.category,
                            "common_semantic": row.common_semantic,
                            "experiment_id": row.experiment_id,
                            "selection_target_size": len(selection_target_indices),
                            "selection_control_size": len(selection_control_indices),
                            "calibration_control_size": len(calibration_indices),
                            "evaluation_target_size": len(evaluation_target_indices),
                            "evaluation_control_size": len(evaluation_control_indices),
                            **budget_row,
                        }
                    )

        del score_matrix, sae_model
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    per_feature_path = output_dir / "per_feature_budget_results.csv"
    write_csv(
        per_feature_path,
        per_feature_rows,
        fieldnames=[
            "family_id",
            "method",
            "method_label",
            "feature_index",
            "certainty",
            "display_name",
            "category",
            "common_semantic",
            "experiment_id",
            "selection_target_size",
            "selection_control_size",
            "calibration_control_size",
            "evaluation_target_size",
            "evaluation_control_size",
            "budget",
            "threshold",
            "target_reject_rate",
            "control_reject_rate",
        ],
    )

    family_count = len({row.family_id for row in registry_rows})
    metric_rows: list[dict[str, Any]] = []
    by_method_budget_family: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_feature_rows:
        by_method_budget_family[(row["method"], float(row["budget"]), row["family_id"])].append(row)

    methods_present = sorted({row["method"] for row in per_feature_rows}, key=method_sort_key)
    budgets_present = sorted({float(row["budget"]) for row in per_feature_rows})
    for budget in budgets_present:
        for target_threshold in sorted(args.target_thresholds):
            for method in methods_present:
                controllable_families = 0
                redundant_families = 0
                total_valid_controllers = 0
                per_family_counts = []
                families_for_method = sorted({key[2] for key in by_method_budget_family if key[0] == method and abs(key[1] - budget) < 1e-12})
                for family_id in families_for_method:
                    feature_rows = by_method_budget_family[(method, budget, family_id)]
                    valid_rows = [
                        row
                        for row in feature_rows
                        if float(row["control_reject_rate"]) <= budget + 1e-12
                        and float(row["target_reject_rate"]) >= target_threshold - 1e-12
                    ]
                    valid_count = len(valid_rows)
                    per_family_counts.append(
                        {
                            "family_id": family_id,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "budget": budget,
                            "target_threshold": target_threshold,
                            "valid_controller_count": valid_count,
                        }
                    )
                    if valid_count >= 1:
                        controllable_families += 1
                    if valid_count >= 2:
                        redundant_families += 1
                    total_valid_controllers += valid_count

                controllable_rate = controllable_families / family_count
                redundant_rate = redundant_families / family_count
                average_valid_count = total_valid_controllers / family_count
                metric_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "budget": budget,
                        "target_threshold": target_threshold,
                        "family_count": family_count,
                        "controllable_family_count": controllable_families,
                        "controllable_family_rate": controllable_rate,
                        "redundant_controller_family_count": redundant_families,
                        "redundant_controller_family_rate": redundant_rate,
                        "total_valid_controller_count": total_valid_controllers,
                        "average_valid_controller_count_per_family": average_valid_count,
                    }
                )
                write_csv(
                    output_dir / "per_family_counts" / f"{method}_b{int(round(budget * 100))}_t{int(round(target_threshold * 100))}.csv",
                    per_family_counts,
                    fieldnames=[
                        "family_id",
                        "method",
                        "method_label",
                        "budget",
                        "target_threshold",
                        "valid_controller_count",
                    ],
                )

    metric_path = output_dir / "control_metric_candidates.csv"
    write_csv(
        metric_path,
        metric_rows,
        fieldnames=[
            "method",
            "method_label",
            "budget",
            "target_threshold",
            "family_count",
            "controllable_family_count",
            "controllable_family_rate",
            "redundant_controller_family_count",
            "redundant_controller_family_rate",
            "total_valid_controller_count",
            "average_valid_controller_count_per_family",
        ],
    )

    render_summary_markdown(metric_rows, output_dir)

    metadata = {
        "budgets": args.budgets,
        "target_thresholds": args.target_thresholds,
        "methods": methods_present,
        "family_count": family_count,
        "registry_row_count": len(registry_rows),
        "text_pool_size": len(texts),
        "split_counts": split_counts,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
