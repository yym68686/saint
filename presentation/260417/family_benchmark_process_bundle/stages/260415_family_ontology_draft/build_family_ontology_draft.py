import argparse
import logging
import textwrap
from pathlib import Path
import re

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
THESIS_EXP_ROOT = REPO_ROOT.parent / "Thesis" / "exp"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_DEFINITIONS_PATH = DEFAULT_OUTPUT_DIR / "family_seed_definitions.yaml"

CORE_METHOD_MAP = {
    "baseline/main": "topk",
    "baseline/BatchTopK": "batchtopk",
    "baseline/relusae": "relu",
    "baseline/gatedsae": "gatedsae",
    "baseline/jumprelu": "jumprelu",
    "idea1-dense-success": "dense",
    "idea5-kernel": "kernel",
}
CORE_METHOD_ORDER = ["topk", "batchtopk", "relu", "gatedsae", "jumprelu", "dense", "kernel"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a draft family ontology from Thesis exp/**/parsed_responses.yaml files."
    )
    parser.add_argument("--exp-root", type=Path, default=THESIS_EXP_ROOT)
    parser.add_argument("--definitions", type=Path, default=DEFAULT_DEFINITIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def relative_experiment_id(path: Path, exp_root: Path) -> str:
    rel = path.relative_to(exp_root)
    if rel.parts[0] == "baseline":
        return f"{rel.parts[0]}/{rel.parts[1]}"
    return rel.parts[0]


def find_source_paths(exp_root: Path) -> list[Path]:
    return sorted(exp_root.glob("**/output/l22/output/parsed_responses.yaml"))


def load_family_definitions(definitions_path: Path) -> tuple[dict, list[dict]]:
    raw = yaml.safe_load(definitions_path.read_text(encoding="utf-8"))
    metadata = raw.get("metadata", {})
    families = raw.get("families", [])
    for family in families:
        family["compiled_regex_any"] = [re.compile(pattern) for pattern in family["regex_any"]]
    return metadata, families


def load_records(source_paths: list[Path], exp_root: Path) -> list[dict]:
    records: list[dict] = []
    for path in source_paths:
        experiment_id = relative_experiment_id(path, exp_root)
        method_label = CORE_METHOD_MAP.get(experiment_id, experiment_id)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for feature_id, row in payload.items():
            common_semantic = str(row.get("common_semantic", "")).strip()
            records.append(
                {
                    "experiment_id": experiment_id,
                    "method_label": method_label,
                    "core_method": CORE_METHOD_MAP.get(experiment_id),
                    "feature_id": int(feature_id),
                    "certainty": float(row.get("certainty", 0.0)),
                    "common_semantic": common_semantic,
                    "lower_common_semantic": common_semantic.lower(),
                    "source_path": str(path),
                }
            )
    return records


def matched_family_ids(record: dict, families: list[dict]) -> list[str]:
    hits: list[str] = []
    for family in families:
        if any(pattern.search(record["lower_common_semantic"]) for pattern in family["compiled_regex_any"]):
            hits.append(family["family_id"])
    return hits


def build_match_tables(records: list[dict], families: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_rows: list[dict] = []
    overlap_rows: list[dict] = []

    for record in records:
        hits = matched_family_ids(record, families)
        if len(hits) > 1:
            overlap_rows.append(
                {
                    "experiment_id": record["experiment_id"],
                    "method_label": record["method_label"],
                    "core_method": record["core_method"] or "",
                    "feature_id": record["feature_id"],
                    "certainty": record["certainty"],
                    "matched_families": ";".join(hits),
                    "common_semantic": record["common_semantic"],
                    "source_path": record["source_path"],
                }
            )

        for family_id in hits:
            family = next(item for item in families if item["family_id"] == family_id)
            match_rows.append(
                {
                    "family_id": family["family_id"],
                    "display_name": family["display_name"],
                    "category": family["category"],
                    "benchmark_priority": family["benchmark_priority"],
                    "experiment_id": record["experiment_id"],
                    "method_label": record["method_label"],
                    "core_method": record["core_method"] or "",
                    "feature_id": record["feature_id"],
                    "certainty": record["certainty"],
                    "common_semantic": record["common_semantic"],
                    "source_path": record["source_path"],
                }
            )

    return pd.DataFrame.from_records(match_rows), pd.DataFrame.from_records(overlap_rows)


def build_family_summary(
    family_metadata: dict,
    families: list[dict],
    records: list[dict],
    matches_df: pd.DataFrame,
    overlaps_df: pd.DataFrame,
    source_paths: list[Path],
) -> tuple[dict, pd.DataFrame]:
    family_rows: list[dict] = []

    for family in families:
        family_df = matches_df[matches_df["family_id"] == family["family_id"]].copy()
        experiment_ids = sorted(family_df["experiment_id"].unique().tolist()) if not family_df.empty else []
        core_methods = [method for method in CORE_METHOD_ORDER if method in set(family_df["core_method"].tolist())]
        example_rows = []
        if not family_df.empty:
            examples_df = (
                family_df.sort_values(["certainty", "experiment_id", "feature_id"], ascending=[False, True, True])
                .drop_duplicates(subset=["experiment_id"])
                .head(5)
            )
            for _, row in examples_df.iterrows():
                example_rows.append(
                    {
                        "experiment_id": row["experiment_id"],
                        "feature_id": int(row["feature_id"]),
                        "certainty": float(row["certainty"]),
                        "common_semantic": textwrap.shorten(str(row["common_semantic"]), width=180, placeholder="..."),
                    }
                )

        family_rows.append(
            {
                "family_id": family["family_id"],
                "display_name": family["display_name"],
                "category": family["category"],
                "benchmark_priority": family["benchmark_priority"],
                "description": family["description"],
                "notes": family.get("notes", ""),
                "regex_any": family["regex_any"],
                "matched_feature_count": int(len(family_df)),
                "experiment_count": int(len(experiment_ids)),
                "experiments_present": experiment_ids,
                "core_method_count": int(len(core_methods)),
                "core_methods_present": core_methods,
                "mean_certainty": float(family_df["certainty"].mean()) if not family_df.empty else 0.0,
                "max_certainty": float(family_df["certainty"].max()) if not family_df.empty else 0.0,
                "example_matches": example_rows,
            }
        )

    ontology_payload = {
        "metadata": {
            **family_metadata,
            "source_file_count": len(source_paths),
            "record_count": len(records),
            "matched_record_count": int(matches_df[["experiment_id", "feature_id"]].drop_duplicates().shape[0]),
            "overlap_record_count": int(overlaps_df.shape[0]),
            "family_count": len(families),
        },
        "families": family_rows,
    }

    summary_df = pd.DataFrame.from_records(
        [
            {
                "family_id": row["family_id"],
                "display_name": row["display_name"],
                "category": row["category"],
                "benchmark_priority": row["benchmark_priority"],
                "experiment_count": row["experiment_count"],
                "core_method_count": row["core_method_count"],
                "matched_feature_count": row["matched_feature_count"],
                "mean_certainty": row["mean_certainty"],
                "max_certainty": row["max_certainty"],
                "core_methods_present": ",".join(row["core_methods_present"]),
                "experiments_present": ",".join(row["experiments_present"]),
            }
            for row in family_rows
        ]
    )

    return ontology_payload, summary_df


def build_coverage_matrices(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment_matrix = (
        matches_df.pivot_table(
            index=["family_id", "display_name", "category", "benchmark_priority"],
            columns="experiment_id",
            values="feature_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    core_df = matches_df[matches_df["core_method"].astype(bool)].copy()
    core_matrix = (
        core_df.pivot_table(
            index=["family_id", "display_name", "category", "benchmark_priority"],
            columns="core_method",
            values="feature_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=CORE_METHOD_ORDER, fill_value=0)
        .reset_index()
    )
    return experiment_matrix, core_matrix


def write_readme(
    output_dir: Path,
    ontology_payload: dict,
    summary_df: pd.DataFrame,
    source_paths: list[Path],
) -> None:
    metadata = ontology_payload["metadata"]
    lines = [
        "# Family Ontology Draft",
        "",
        "This folder contains an editable first-pass family ontology built from",
        "`Thesis/exp/**/output/l22/output/parsed_responses.yaml`.",
        "",
        "## Scope",
        f"- Source files scanned: {len(source_paths)}",
        f"- Total L22 feature records: {metadata['record_count']}",
        f"- Families in shortlist: {metadata['family_count']}",
        f"- Records with at least one family match: {metadata['matched_record_count']}",
        f"- Overlap review queue size: {metadata['overlap_record_count']}",
        "",
        "## Files",
        "- `family_seed_definitions.yaml`: editable seed definitions and regex patterns.",
        "- `build_family_ontology_draft.py`: regeneration script.",
        "- `family_ontology_draft.yaml`: generated ontology payload with counts and examples.",
        "- `family_summary.csv`: compact family-level summary table.",
        "- `family_coverage_by_experiment.csv`: feature-count matrix across experiment outputs.",
        "- `family_coverage_by_core_method.csv`: feature-count matrix across the 7 core methods.",
        "- `family_match_details.csv`: one matched feature-family row per hit.",
        "- `family_overlap_review.csv`: records matching multiple families and needing manual review.",
        "",
        "## Current Shortlist",
        "| family_id | category | priority | exp_count | core_methods | matched_features |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['family_id']} | {row['category']} | {row['benchmark_priority']} | "
            f"{int(row['experiment_count'])} | {row['core_methods_present'] or '-'} | {int(row['matched_feature_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- The shortlist intentionally keeps only medium-granularity families that recur across multiple outputs.",
            "- Long-tail or artifact-heavy families such as `cybersecurity`, `maritime_naval`, `immigration_border`, and `cricket` are left out of this initial shortlist.",
            "- Overlapping records are expected at this stage. Use `family_overlap_review.csv` to split or tighten definitions before building a family-level control benchmark.",
        ]
    )

    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    family_metadata, families = load_family_definitions(args.definitions.resolve())
    source_paths = find_source_paths(args.exp_root.resolve())
    records = load_records(source_paths, args.exp_root.resolve())
    matches_df, overlaps_df = build_match_tables(records, families)
    ontology_payload, summary_df = build_family_summary(
        family_metadata=family_metadata,
        families=families,
        records=records,
        matches_df=matches_df,
        overlaps_df=overlaps_df,
        source_paths=source_paths,
    )
    experiment_matrix, core_matrix = build_coverage_matrices(matches_df)

    with (output_dir / "family_ontology_draft.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(ontology_payload, handle, sort_keys=False, allow_unicode=False, width=100)

    summary_df.to_csv(output_dir / "family_summary.csv", index=False)
    experiment_matrix.to_csv(output_dir / "family_coverage_by_experiment.csv", index=False)
    core_matrix.to_csv(output_dir / "family_coverage_by_core_method.csv", index=False)
    matches_df.sort_values(["family_id", "experiment_id", "feature_id"]).to_csv(
        output_dir / "family_match_details.csv",
        index=False,
    )
    overlaps_df.sort_values(["experiment_id", "feature_id"]).to_csv(
        output_dir / "family_overlap_review.csv",
        index=False,
    )
    write_readme(output_dir=output_dir, ontology_payload=ontology_payload, summary_df=summary_df, source_paths=source_paths)

    logging.info("Generated family ontology draft under %s", output_dir)


if __name__ == "__main__":
    main()
