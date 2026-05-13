from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = THIS_DIR / "family_dataset_round2" / "candidate_pools"
DEFAULT_OUTPUT_ROOT = THIS_DIR / "family_dataset_manualclean"
DEFAULT_RULES_PATH = THIS_DIR / "manual_cleaning_rules.yaml"
DEFAULT_DEFINITIONS_PATH = THIS_DIR / "benchmark_ready_family_definitions.yaml"

BUCKETS = [
    "positive",
    "local_hard_negative",
    "benchmark_hard_negative",
    "medium_negative",
    "background_negative",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def compile_patterns(patterns: List[str]) -> List[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def filter_positive_rows(
    rows: List[Dict[str, Any]],
    keep_patterns: List[re.Pattern[str]],
    drop_patterns: List[re.Pattern[str]],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        text = row["text"]
        normalized = normalize_text(text)
        if normalized in seen:
            continue
        if keep_patterns and not any(pattern.search(text) for pattern in keep_patterns):
            continue
        if any(pattern.search(text) for pattern in drop_patterns):
            continue
        filtered.append(row)
        seen.add(normalized)
    return filtered


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        normalized = normalize_text(row["text"])
        if normalized in seen:
            continue
        deduped.append(row)
        seen.add(normalized)
    return deduped


def build_reference_pool(
    family_ids: List[str],
    cleaned_positive_by_family: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family_id in family_ids:
        rows.extend(cleaned_positive_by_family.get(family_id, []))
    return dedupe_rows(rows)


def write_summary(output_root: Path, family_ids: List[str]) -> None:
    summary_path = output_root / "candidate_pools" / "candidate_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family_id", "bucket", "candidate_count"])
        writer.writeheader()
        for family_id in family_ids:
            family_dir = output_root / "candidate_pools" / family_id
            for bucket in BUCKETS:
                bucket_path = family_dir / f"{bucket}.jsonl"
                count = sum(1 for _ in bucket_path.open()) if bucket_path.exists() else 0
                writer.writerow(
                    {
                        "family_id": family_id,
                        "bucket": bucket,
                        "candidate_count": count,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply family-specific manual cleaning rules to round2 candidate pools and build a cleaner candidate round."
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Source candidate_pools directory, typically family_dataset_round2/candidate_pools.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for cleaned candidate pools.",
    )
    parser.add_argument(
        "--rules_path",
        type=Path,
        default=DEFAULT_RULES_PATH,
        help="Manual cleaning rules YAML path.",
    )
    parser.add_argument(
        "--definitions_path",
        type=Path,
        default=DEFAULT_DEFINITIONS_PATH,
        help="Benchmark-ready family definitions YAML path.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    cleaned_candidate_root = args.output_root / "candidate_pools"
    cleaned_candidate_root.mkdir(parents=True, exist_ok=True)

    rules_payload = load_yaml(args.rules_path)
    rule_map = rules_payload["families"]
    definitions_payload = load_yaml(args.definitions_path)
    selected_families = definitions_payload["selected_families"]
    family_ids = [family["family_id"] for family in selected_families]
    definition_map = {family["family_id"]: family for family in selected_families}

    cleaned_positive_by_family: Dict[str, List[Dict[str, Any]]] = {}
    local_negative_by_family: Dict[str, List[Dict[str, Any]]] = {}

    for family_id in family_ids:
        family_dir = args.source_root / family_id
        positive_rows = read_jsonl(family_dir / "positive.jsonl")
        local_rows = read_jsonl(family_dir / "local_hard_negative.jsonl")

        family_rules = rule_map.get(family_id, {})
        keep_patterns = compile_patterns(family_rules.get("positive_keep_regex_any", []))
        drop_patterns = compile_patterns(family_rules.get("positive_drop_regex_any", []))

        cleaned_positive_by_family[family_id] = filter_positive_rows(
            positive_rows,
            keep_patterns=keep_patterns,
            drop_patterns=drop_patterns,
        )
        local_negative_by_family[family_id] = dedupe_rows(local_rows)

    for family_id in family_ids:
        definition = definition_map[family_id]
        family_output_dir = cleaned_candidate_root / family_id
        family_output_dir.mkdir(parents=True, exist_ok=True)

        positive_rows = cleaned_positive_by_family[family_id]
        local_rows = local_negative_by_family[family_id]
        benchmark_hard_negative = build_reference_pool(
            definition.get("hard_negative_families", []),
            cleaned_positive_by_family,
        )
        medium_negative = build_reference_pool(
            definition.get("medium_negative_families", []),
            cleaned_positive_by_family,
        )
        background_negative = build_reference_pool(
            definition.get("background_negative_families", []),
            cleaned_positive_by_family,
        )

        write_jsonl(family_output_dir / "positive.jsonl", positive_rows)
        write_jsonl(family_output_dir / "local_hard_negative.jsonl", local_rows)
        write_jsonl(
            family_output_dir / "benchmark_hard_negative.jsonl",
            benchmark_hard_negative,
        )
        write_jsonl(family_output_dir / "medium_negative.jsonl", medium_negative)
        write_jsonl(family_output_dir / "background_negative.jsonl", background_negative)

    metadata = {
        "source_root": str(args.source_root),
        "rules_path": str(args.rules_path),
        "definitions_path": str(args.definitions_path),
        "round_name": "manualclean",
        "family_ids": family_ids,
    }
    with (cleaned_candidate_root / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    write_summary(args.output_root, family_ids)


if __name__ == "__main__":
    main()
