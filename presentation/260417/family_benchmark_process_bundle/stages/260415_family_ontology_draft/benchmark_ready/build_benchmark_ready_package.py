from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


THIS_DIR = Path(__file__).resolve().parent
DEFINITIONS_PATH = THIS_DIR / "benchmark_ready_family_definitions.yaml"
SUMMARY_PATH = THIS_DIR.parent / "family_summary.csv"
OVERLAP_PATH = THIS_DIR.parent / "family_overlap_review.csv"

SELECTED_SUMMARY_PATH = THIS_DIR / "benchmark_ready_family_summary.csv"
EXCLUDED_SUMMARY_PATH = THIS_DIR / "excluded_family_notes.csv"
README_PATH = THIS_DIR / "README.md"
BLUEPRINT_PATH = THIS_DIR / "target_control_blueprint.md"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def bullet_list(items: Iterable[str], indent: str = "") -> List[str]:
    return [f"{indent}- {item}" for item in items]


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    payload = load_yaml(DEFINITIONS_PATH)
    family_summary = {row["family_id"]: row for row in load_csv(SUMMARY_PATH)}
    overlap_counter: Counter[str] = Counter()
    for row in load_csv(OVERLAP_PATH):
        for family_id in row["matched_families"].split(";"):
            overlap_counter[family_id] += 1

    selected_rows: List[Dict[str, Any]] = []
    for family in payload["selected_families"]:
        family_id = family["family_id"]
        if family_id not in family_summary:
            raise KeyError(f"Unknown family_id in selected_families: {family_id}")
        source = family_summary[family_id]
        selected_rows.append(
            {
                "family_id": family_id,
                "display_name": family["display_name"],
                "category": source["category"],
                "core_method_count": source["core_method_count"],
                "experiment_count": source["experiment_count"],
                "matched_feature_count": source["matched_feature_count"],
                "overlap_review_hits": overlap_counter.get(family_id, 0),
                "overlap_risk": family["overlap_risk"],
                "selection_reason": family["selection_reason"],
                "target_scope": family["target_scope"],
                "hard_negative_families": ";".join(family["hard_negative_families"]),
                "out_of_shortlist_hard_negatives": ";".join(
                    family["out_of_shortlist_hard_negatives"]
                ),
                "medium_negative_families": ";".join(
                    family["medium_negative_families"]
                ),
                "background_negative_families": ";".join(
                    family["background_negative_families"]
                ),
            }
        )

    excluded_rows: List[Dict[str, Any]] = []
    for family in payload["excluded_families"]:
        family_id = family["family_id"]
        if family_id not in family_summary:
            raise KeyError(f"Unknown family_id in excluded_families: {family_id}")
        source = family_summary[family_id]
        excluded_rows.append(
            {
                "family_id": family_id,
                "display_name": source["display_name"],
                "category": source["category"],
                "core_method_count": source["core_method_count"],
                "experiment_count": source["experiment_count"],
                "matched_feature_count": source["matched_feature_count"],
                "overlap_review_hits": overlap_counter.get(family_id, 0),
                "exclusion_reason": family["exclusion_reason"],
            }
        )

    write_csv(
        SELECTED_SUMMARY_PATH,
        selected_rows,
        [
            "family_id",
            "display_name",
            "category",
            "core_method_count",
            "experiment_count",
            "matched_feature_count",
            "overlap_review_hits",
            "overlap_risk",
            "selection_reason",
            "target_scope",
            "hard_negative_families",
            "out_of_shortlist_hard_negatives",
            "medium_negative_families",
            "background_negative_families",
        ],
    )
    write_csv(
        EXCLUDED_SUMMARY_PATH,
        excluded_rows,
        [
            "family_id",
            "display_name",
            "category",
            "core_method_count",
            "experiment_count",
            "matched_feature_count",
            "overlap_review_hits",
            "exclusion_reason",
        ],
    )

    selected_count = len(selected_rows)
    excluded_count = len(excluded_rows)
    budget_strings = [f"{int(alpha * 100)}%" for alpha in payload["benchmark_protocol"]["evaluation_budgets"]]
    split_recipe = payload["benchmark_protocol"]["split_recipe"]
    control_mix = payload["benchmark_protocol"]["control_mix"]

    readme_lines: List[str] = [
        "# Benchmark-Ready Family Package",
        "",
        "This folder compresses the 29-family draft ontology into a benchmark-ready family shortlist",
        "for complete control working-point evaluation.",
        "",
        "## Selection Rule",
        f"- {payload['metadata']['objective_rule']}",
        f"- Selected families: {selected_count}",
        f"- Excluded families: {excluded_count}",
        f"- Budgets supported by the blueprint: {', '.join(budget_strings)}",
        "",
        "## Files",
        "- `benchmark_ready_family_definitions.yaml`: manual shortlist and per-family construction blueprint.",
        "- `build_benchmark_ready_package.py`: regenerates the summary tables and markdown blueprint.",
        "- `benchmark_ready_family_summary.csv`: selected family table with support counts and negative-family assignments.",
        "- `excluded_family_notes.csv`: compressed-away families and why they are excluded from the first benchmark-ready pass.",
        "- `target_control_blueprint.md`: detailed protocol and per-family target/control construction guidance.",
        "- `family_retrieval_queries.yaml`: initial lexical retrieval query packs for all 15 benchmark-ready families.",
        "- `family_annotation_guidelines.md`: shared labeling rubric for family-level target/control annotation.",
        "- `family_annotation_sheet_template.csv`: header template for exported annotation batches.",
        "- `build_family_level_dataset_skeleton.py`: candidate-mining and annotation-batch scaffold built on top of the blueprint.",
        "",
        "## Global Benchmark Protocol",
        f"- Selection split: {split_recipe['selection_target']} target / {split_recipe['selection_control']} control.",
        f"- Calibration split: {split_recipe['calibration_control']} control only.",
        f"- Evaluation split: {split_recipe['evaluation_target']} target / {split_recipe['evaluation_control']} control.",
        "- Control mix:",
        f"  - hard negatives: {int(control_mix['hard_negative_share'] * 100)}%",
        f"  - medium negatives: {int(control_mix['medium_negative_share'] * 100)}%",
        f"  - background negatives: {int(control_mix['background_share'] * 100)}%",
        "",
        "## Selected Families",
        "| family_id | display_name | category | core_methods | exp_count | overlap_hits | overlap_risk |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in selected_rows:
        readme_lines.append(
            f"| {row['family_id']} | {row['display_name']} | {row['category']} | {row['core_method_count']} | {row['experiment_count']} | {row['overlap_review_hits']} | {row['overlap_risk']} |"
        )
    readme_lines.extend(
        [
            "",
            "## Excluded Families",
            "| family_id | display_name | core_methods | overlap_hits | reason |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in excluded_rows:
        readme_lines.append(
            f"| {row['family_id']} | {row['display_name']} | {row['core_method_count']} | {row['overlap_review_hits']} | {row['exclusion_reason']} |"
        )
    readme_lines.extend(
        [
            "",
            "Regenerate with:",
            "",
            "```bash",
            "uv run python build_benchmark_ready_package.py",
            "```",
            "",
        ]
    )
    README_PATH.write_text("\n".join(readme_lines))

    blueprint_lines: List[str] = [
        "# Family-Level Target / Control Blueprint",
        "",
        "This document translates the benchmark-ready shortlist into a concrete data-construction",
        "plan for a family-level complete control working-point benchmark.",
        "",
        "## Global Protocol",
        "",
        "### Final corpus policy",
    ]
    blueprint_lines.extend(
        bullet_list(payload["benchmark_protocol"]["final_corpus_policy"])
    )
    blueprint_lines.extend(
        [
            "",
            "### Split recipe",
            f"- Selection split: {split_recipe['selection_target']} target + {split_recipe['selection_control']} control.",
            f"- Calibration split: {split_recipe['calibration_control']} control only.",
            f"- Evaluation split: {split_recipe['evaluation_target']} target + {split_recipe['evaluation_control']} control.",
            "",
            "### Control mix",
            f"- Hard negatives: {int(control_mix['hard_negative_share'] * 100)}%",
            f"- Medium negatives: {int(control_mix['medium_negative_share'] * 100)}%",
            f"- Background negatives: {int(control_mix['background_share'] * 100)}%",
            "",
            "### Construction guardrails",
        ]
    )
    blueprint_lines.extend(
        bullet_list(payload["benchmark_protocol"]["construction_guardrails"])
    )
    blueprint_lines.extend(
        [
            "",
            "### Retrieval pipeline",
        ]
    )
    blueprint_lines.extend(
        bullet_list(payload["benchmark_protocol"]["retrieval_pipeline"])
    )

    for family in payload["selected_families"]:
        source = family_summary[family["family_id"]]
        blueprint_lines.extend(
            [
                "",
                f"## {family['display_name']} (`{family['family_id']}`)",
                "",
                f"- Category: {source['category']}",
                f"- Support: {source['core_method_count']} core methods, {source['experiment_count']} experiment outputs, {source['matched_feature_count']} matched features",
                f"- Overlap review hits: {overlap_counter.get(family['family_id'], 0)}",
                f"- Why kept: {family['selection_reason']}",
                "",
                "### Target definition",
                f"- Scope: {family['target_scope']}",
                "- Seed terms:",
            ]
        )
        blueprint_lines.extend(bullet_list(family["target_seed_terms"], indent="  "))
        blueprint_lines.extend(
            [
                "- Exclusions:",
            ]
        )
        blueprint_lines.extend(bullet_list(family["target_exclusions"], indent="  "))
        blueprint_lines.extend(
            [
                "",
                "### Control construction",
                "- Hard-negative benchmark families:",
            ]
        )
        blueprint_lines.extend(
            bullet_list(family["hard_negative_families"], indent="  ")
        )
        blueprint_lines.extend(
            [
                "- Hard negatives outside the benchmark shortlist:",
            ]
        )
        blueprint_lines.extend(
            bullet_list(family["out_of_shortlist_hard_negatives"], indent="  ")
        )
        blueprint_lines.extend(
            [
                "- Medium-negative benchmark families:",
            ]
        )
        blueprint_lines.extend(
            bullet_list(family["medium_negative_families"], indent="  ")
        )
        blueprint_lines.extend(
            [
                "- Background benchmark families:",
            ]
        )
        blueprint_lines.extend(
            bullet_list(family["background_negative_families"], indent="  ")
        )
        blueprint_lines.extend(
            [
                "",
                "### Construction notes",
            ]
        )
        blueprint_lines.extend(bullet_list(family["construction_notes"]))

    blueprint_lines.extend(
        [
            "",
            "## Usage",
            "",
            "- Use the selection split to choose one feature per family and method.",
            "- Use the calibration-control split to tune thresholds for each alpha budget.",
            "- Use the evaluation split to report family-level target reject and held-out control reject.",
            "- Aggregate across families only after every method has been evaluated on the same family set.",
            "",
        ]
    )
    BLUEPRINT_PATH.write_text("\n".join(blueprint_lines))


if __name__ == "__main__":
    main()
