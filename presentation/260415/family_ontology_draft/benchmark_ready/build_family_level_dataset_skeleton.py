from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping

import yaml
from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DEFINITIONS_PATH = ROOT / "benchmark_ready_family_definitions.yaml"
DEFAULT_QUERY_PATH = ROOT / "family_retrieval_queries.yaml"
DEFAULT_TEMPLATE_PATH = ROOT / "family_annotation_sheet_template.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "family_dataset_skeleton"

CANDIDATE_BUCKETS = [
    "positive",
    "local_hard_negative",
    "benchmark_hard_negative",
    "medium_negative",
    "background_negative",
]
DEFAULT_BATCH_WEIGHTS = {
    "positive": 0.40,
    "benchmark_hard_negative": 0.30,
    "local_hard_negative": 0.10,
    "medium_negative": 0.12,
    "background_negative": 0.08,
}


@dataclass
class QueryBlock:
    name: str
    description: str
    terms: List[str]
    patterns: List[tuple[str, re.Pattern[str]]]


@dataclass
class FamilySpec:
    family_id: str
    display_name: str
    target_exclusions: List[str]
    hard_negative_families: List[str]
    medium_negative_families: List[str]
    background_negative_families: List[str]
    positive_blocks: List[QueryBlock]
    local_hard_negative_blocks: List[QueryBlock]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_annotation_columns(path: Path) -> List[str]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def literal_to_regex(term: str) -> re.Pattern[str]:
    if term.startswith("re:"):
        return re.compile(term[3:], re.IGNORECASE)
    escaped = re.escape(term.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    if term and term[0].isalnum():
        escaped = r"\b" + escaped
    if term and term[-1].isalnum():
        escaped = escaped + r"\b"
    return re.compile(escaped, re.IGNORECASE)


def compile_blocks(blocks: List[Dict[str, Any]]) -> List[QueryBlock]:
    compiled: List[QueryBlock] = []
    for block in blocks:
        compiled.append(
            QueryBlock(
                name=block["name"],
                description=block.get("description", ""),
                terms=list(block["terms"]),
                patterns=[(term, literal_to_regex(term)) for term in block["terms"]],
            )
        )
    return compiled


def flatten_hits(hit_map: Mapping[str, List[str]]) -> List[str]:
    terms: List[str] = []
    for hit_terms in hit_map.values():
        terms.extend(hit_terms)
    return sorted(set(terms))


def match_blocks(text: str, blocks: Iterable[QueryBlock]) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    for block in blocks:
        matched_terms = [term for term, pattern in block.patterns if pattern.search(text)]
        if matched_terms:
            hits[block.name] = matched_terms
    return hits


def match_terms(text: str, terms: Iterable[str]) -> List[str]:
    matched: List[str] = []
    for term in terms:
        if literal_to_regex(term).search(text):
            matched.append(term)
    return matched


def serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    return str(value)


def family_map_from_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {family["family_id"]: family for family in payload["selected_families"]}


def load_family_specs(
    definitions_path: Path,
    query_path: Path,
    selected_family_ids: List[str] | None,
) -> Dict[str, FamilySpec]:
    definitions_payload = load_yaml(definitions_path)
    query_payload = load_yaml(query_path)
    definitions_map = family_map_from_payload(definitions_payload)
    query_map = {family["family_id"]: family for family in query_payload["families"]}

    family_ids = selected_family_ids or list(definitions_map.keys())
    missing_from_queries = [family_id for family_id in family_ids if family_id not in query_map]
    if missing_from_queries:
        raise KeyError(f"Missing retrieval queries for: {', '.join(missing_from_queries)}")

    specs: Dict[str, FamilySpec] = {}
    for family_id in family_ids:
        definition = definitions_map[family_id]
        query_spec = query_map[family_id]
        specs[family_id] = FamilySpec(
            family_id=family_id,
            display_name=definition["display_name"],
            target_exclusions=list(definition.get("target_exclusions", [])),
            hard_negative_families=list(definition.get("hard_negative_families", [])),
            medium_negative_families=list(definition.get("medium_negative_families", [])),
            background_negative_families=list(definition.get("background_negative_families", [])),
            positive_blocks=compile_blocks(query_spec.get("positive_query_blocks", [])),
            local_hard_negative_blocks=compile_blocks(
                query_spec.get("local_hard_negative_blocks", [])
            ),
        )
    return specs


def load_records(
    dataset_path: Path,
    dataset_format: str,
    text_field: str,
    keep_fields: List[str],
    max_records: int | None,
) -> Iterator[Dict[str, Any]]:
    loader_name = "parquet" if dataset_format == "parquet" else "json"
    dataset = load_dataset(
        loader_name,
        data_files={"train": str(dataset_path)},
        split="train",
    )
    for row_index, row in enumerate(dataset):
        if max_records is not None and row_index >= max_records:
            break
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        extra = {
            key: serialize_value(row.get(key))
            for key in keep_fields
            if key in row and key != text_field
        }
        yield {
            "row_index": row_index,
            "text": text,
            "extra_fields": extra,
        }


def make_candidate_id(family_id: str, bucket: str, row_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{family_id}|{bucket}|{row_index}|{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def quota_from_weights(batch_size: int) -> Dict[str, int]:
    quotas = {
        bucket: int(math.floor(batch_size * weight))
        for bucket, weight in DEFAULT_BATCH_WEIGHTS.items()
    }
    assigned = sum(quotas.values())
    bucket_order = list(DEFAULT_BATCH_WEIGHTS.keys())
    idx = 0
    while assigned < batch_size:
        quotas[bucket_order[idx % len(bucket_order)]] += 1
        assigned += 1
        idx += 1
    return quotas


def build_candidate_row(
    family_id: str,
    bucket: str,
    row_index: int,
    text: str,
    matched_query_blocks: Dict[str, List[str]],
    matched_family_ids: List[str],
    bucket_reason: str,
    extra_fields: Dict[str, Any],
    exclusion_hits: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "candidate_id": make_candidate_id(family_id, bucket, row_index, text),
        "family_id": family_id,
        "proposed_bucket": bucket,
        "source_row_index": row_index,
        "text": text,
        "matched_query_blocks": sorted(matched_query_blocks.keys()),
        "matched_terms": flatten_hits(matched_query_blocks),
        "matched_family_ids": matched_family_ids,
        "bucket_reason": bucket_reason,
        "exclusion_hits": exclusion_hits or [],
        "extra_fields": extra_fields,
    }


def init_candidate_storage(
    specs: Mapping[str, FamilySpec],
) -> tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Dict[str, set[str]]]]:
    pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    dedup: Dict[str, Dict[str, set[str]]] = {}
    for family_id in specs:
        pools[family_id] = {bucket: [] for bucket in CANDIDATE_BUCKETS}
        dedup[family_id] = {bucket: set() for bucket in CANDIDATE_BUCKETS}
    return pools, dedup


def maybe_add_candidate(
    pools: Dict[str, Dict[str, List[Dict[str, Any]]]],
    dedup: Dict[str, Dict[str, set[str]]],
    family_id: str,
    bucket: str,
    row: Dict[str, Any],
    max_candidates_per_bucket: int,
) -> None:
    dedup_key = normalize_text(row["text"])
    if dedup_key in dedup[family_id][bucket]:
        return
    if len(pools[family_id][bucket]) >= max_candidates_per_bucket:
        return
    pools[family_id][bucket].append(row)
    dedup[family_id][bucket].add(dedup_key)


def mine_candidates(args: argparse.Namespace) -> None:
    specs = load_family_specs(
        definitions_path=args.definitions_path,
        query_path=args.query_path,
        selected_family_ids=args.family_ids,
    )
    ensure_dir(args.output_dir)
    candidate_root = args.output_dir / "candidate_pools"
    ensure_dir(candidate_root)

    pools, dedup = init_candidate_storage(specs)
    processed_records = 0

    for record in load_records(
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        text_field=args.text_field,
        keep_fields=args.keep_fields,
        max_records=args.max_records,
    ):
        processed_records += 1
        text = record["text"]
        positive_hits: Dict[str, Dict[str, Any]] = {}
        local_negative_hits: Dict[str, Dict[str, List[str]]] = {}

        for family_id, spec in specs.items():
            block_hits = match_blocks(text, spec.positive_blocks)
            exclusion_hits = match_terms(text, spec.target_exclusions)
            if block_hits and not exclusion_hits:
                positive_hits[family_id] = {
                    "block_hits": block_hits,
                    "exclusion_hits": exclusion_hits,
                }
            local_hits = match_blocks(text, spec.local_hard_negative_blocks)
            if local_hits:
                local_negative_hits[family_id] = local_hits

        for family_id, spec in specs.items():
            if family_id in positive_hits:
                row = build_candidate_row(
                    family_id=family_id,
                    bucket="positive",
                    row_index=record["row_index"],
                    text=text,
                    matched_query_blocks=positive_hits[family_id]["block_hits"],
                    matched_family_ids=[family_id],
                    bucket_reason="family_positive_query_hit",
                    extra_fields=record["extra_fields"],
                )
                maybe_add_candidate(
                    pools,
                    dedup,
                    family_id,
                    "positive",
                    row,
                    args.max_candidates_per_bucket,
                )
                continue

            matched_hard_families = [
                other_family
                for other_family in spec.hard_negative_families
                if other_family in positive_hits
            ]
            matched_medium_families = [
                other_family
                for other_family in spec.medium_negative_families
                if other_family in positive_hits
            ]
            matched_background_families = [
                other_family
                for other_family in spec.background_negative_families
                if other_family in positive_hits
            ]

            if family_id in local_negative_hits:
                row = build_candidate_row(
                    family_id=family_id,
                    bucket="local_hard_negative",
                    row_index=record["row_index"],
                    text=text,
                    matched_query_blocks=local_negative_hits[family_id],
                    matched_family_ids=[],
                    bucket_reason="local_hard_negative_query_hit",
                    extra_fields=record["extra_fields"],
                )
                maybe_add_candidate(
                    pools,
                    dedup,
                    family_id,
                    "local_hard_negative",
                    row,
                    args.max_candidates_per_bucket,
                )
                continue

            if matched_hard_families:
                merged_hits: Dict[str, List[str]] = {}
                for other_family in matched_hard_families:
                    merged_hits[other_family] = flatten_hits(
                        positive_hits[other_family]["block_hits"]
                    )
                row = build_candidate_row(
                    family_id=family_id,
                    bucket="benchmark_hard_negative",
                    row_index=record["row_index"],
                    text=text,
                    matched_query_blocks=merged_hits,
                    matched_family_ids=sorted(matched_hard_families),
                    bucket_reason="matched benchmark hard-negative family",
                    extra_fields=record["extra_fields"],
                )
                maybe_add_candidate(
                    pools,
                    dedup,
                    family_id,
                    "benchmark_hard_negative",
                    row,
                    args.max_candidates_per_bucket,
                )
                continue

            if matched_medium_families:
                merged_hits = {}
                for other_family in matched_medium_families:
                    merged_hits[other_family] = flatten_hits(
                        positive_hits[other_family]["block_hits"]
                    )
                row = build_candidate_row(
                    family_id=family_id,
                    bucket="medium_negative",
                    row_index=record["row_index"],
                    text=text,
                    matched_query_blocks=merged_hits,
                    matched_family_ids=sorted(matched_medium_families),
                    bucket_reason="matched benchmark medium-negative family",
                    extra_fields=record["extra_fields"],
                )
                maybe_add_candidate(
                    pools,
                    dedup,
                    family_id,
                    "medium_negative",
                    row,
                    args.max_candidates_per_bucket,
                )
                continue

            if matched_background_families:
                merged_hits = {}
                for other_family in matched_background_families:
                    merged_hits[other_family] = flatten_hits(
                        positive_hits[other_family]["block_hits"]
                    )
                row = build_candidate_row(
                    family_id=family_id,
                    bucket="background_negative",
                    row_index=record["row_index"],
                    text=text,
                    matched_query_blocks=merged_hits,
                    matched_family_ids=sorted(matched_background_families),
                    bucket_reason="matched benchmark background family",
                    extra_fields=record["extra_fields"],
                )
                maybe_add_candidate(
                    pools,
                    dedup,
                    family_id,
                    "background_negative",
                    row,
                    args.max_candidates_per_bucket,
                )

    summary_rows: List[Dict[str, Any]] = []
    for family_id in specs:
        family_dir = candidate_root / family_id
        ensure_dir(family_dir)
        for bucket in CANDIDATE_BUCKETS:
            bucket_rows = pools[family_id][bucket]
            write_jsonl(family_dir / f"{bucket}.jsonl", bucket_rows)
            summary_rows.append(
                {
                    "family_id": family_id,
                    "bucket": bucket,
                    "candidate_count": len(bucket_rows),
                }
            )

    with (candidate_root / "candidate_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family_id", "bucket", "candidate_count"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    metadata = {
        "dataset_path": str(args.dataset_path),
        "dataset_format": args.dataset_format,
        "text_field": args.text_field,
        "keep_fields": args.keep_fields,
        "max_records": args.max_records,
        "max_candidates_per_bucket": args.max_candidates_per_bucket,
        "processed_records": processed_records,
        "family_ids": list(specs.keys()),
    }
    with (candidate_root / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info("Processed %d records and wrote candidate pools to %s", processed_records, candidate_root)


def load_family_candidates(candidate_root: Path, family_id: str) -> Dict[str, List[Dict[str, Any]]]:
    family_dir = candidate_root / family_id
    return {
        bucket: read_jsonl(family_dir / f"{bucket}.jsonl")
        for bucket in CANDIDATE_BUCKETS
    }


def choose_rows(
    rng: random.Random,
    rows: List[Dict[str, Any]],
    take: int,
) -> List[Dict[str, Any]]:
    if take <= 0 or not rows:
        return []
    if len(rows) <= take:
        return list(rows)
    return rng.sample(rows, take)


def fill_remaining_slots(
    rng: random.Random,
    selected_rows: Dict[str, List[Dict[str, Any]]],
    available_rows: Dict[str, List[Dict[str, Any]]],
    missing_slots: int,
) -> None:
    refill_order = [
        "positive",
        "benchmark_hard_negative",
        "local_hard_negative",
        "medium_negative",
        "background_negative",
    ]
    for bucket in refill_order:
        if missing_slots <= 0:
            break
        remaining = [
            row
            for row in available_rows[bucket]
            if row["candidate_id"] not in {picked["candidate_id"] for picked in selected_rows[bucket]}
        ]
        if not remaining:
            continue
        extra_take = min(len(remaining), missing_slots)
        selected_rows[bucket].extend(choose_rows(rng, remaining, extra_take))
        missing_slots -= extra_take


def build_annotation_batches(args: argparse.Namespace) -> None:
    specs = load_family_specs(
        definitions_path=args.definitions_path,
        query_path=args.query_path,
        selected_family_ids=args.family_ids,
    )
    columns = load_annotation_columns(args.template_path)
    rng = random.Random(args.seed)

    ensure_dir(args.output_dir)
    manifest_rows: List[Dict[str, Any]] = []
    quotas = quota_from_weights(args.batch_size)

    for family_id in specs:
        family_candidates = load_family_candidates(args.candidate_root, family_id)
        selected_by_bucket: Dict[str, List[Dict[str, Any]]] = {}
        for bucket, quota in quotas.items():
            selected_by_bucket[bucket] = choose_rows(
                rng,
                family_candidates[bucket],
                quota,
            )
        filled = sum(len(rows) for rows in selected_by_bucket.values())
        if filled < args.batch_size:
            fill_remaining_slots(
                rng=rng,
                selected_rows=selected_by_bucket,
                available_rows=family_candidates,
                missing_slots=args.batch_size - filled,
            )

        batch_rows: List[Dict[str, Any]] = []
        for bucket in CANDIDATE_BUCKETS:
            for row in selected_by_bucket.get(bucket, []):
                batch_row = {column: "" for column in columns}
                batch_row.update(
                    {
                        "candidate_id": row["candidate_id"],
                        "family_id": row["family_id"],
                        "proposed_bucket": row["proposed_bucket"],
                        "source_row_index": row["source_row_index"],
                        "text": row["text"],
                        "matched_query_blocks": ";".join(row["matched_query_blocks"]),
                        "matched_terms": ";".join(row["matched_terms"]),
                        "matched_family_ids": ";".join(row["matched_family_ids"]),
                    }
                )
                batch_rows.append(batch_row)

        family_output_base = args.output_dir / family_id
        ensure_dir(family_output_base)

        with (family_output_base / "annotation_batch.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in batch_rows:
                writer.writerow(row)

        with (family_output_base / "annotation_batch.jsonl").open("w") as f:
            for row in batch_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest_row: Dict[str, Any] = {
            "family_id": family_id,
            "batch_size": len(batch_rows),
        }
        for bucket in CANDIDATE_BUCKETS:
            manifest_row[f"available_{bucket}"] = len(family_candidates[bucket])
            manifest_row[f"sampled_{bucket}"] = len(selected_by_bucket.get(bucket, []))
        manifest_rows.append(manifest_row)

    manifest_columns = ["family_id", "batch_size"]
    for bucket in CANDIDATE_BUCKETS:
        manifest_columns.append(f"available_{bucket}")
        manifest_columns.append(f"sampled_{bucket}")
    with (args.output_dir / "annotation_batch_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_columns)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    logging.info("Wrote annotation batches to %s", args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Skeleton workflow for building a family-level control benchmark candidate pool and annotation batches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine_parser = subparsers.add_parser(
        "mine-candidates",
        help="Scan a sentence corpus and produce per-family candidate pools using the benchmark-ready queries.",
    )
    mine_parser.add_argument("--dataset_path", type=Path, required=True, help="Path to the source sentence corpus.")
    mine_parser.add_argument(
        "--dataset_format",
        choices=["parquet", "json"],
        default="parquet",
        help="Dataset loader to use.",
    )
    mine_parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Name of the text field in the source dataset.",
    )
    mine_parser.add_argument(
        "--keep_fields",
        nargs="*",
        default=[],
        help="Optional extra fields to preserve in the candidate JSONL under extra_fields.",
    )
    mine_parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional cap on the number of records scanned.",
    )
    mine_parser.add_argument(
        "--max_candidates_per_bucket",
        type=int,
        default=400,
        help="Maximum number of deduplicated candidates to keep per family bucket.",
    )
    mine_parser.add_argument(
        "--family_ids",
        nargs="*",
        default=None,
        help="Optional subset of family IDs to process.",
    )
    mine_parser.add_argument(
        "--definitions_path",
        type=Path,
        default=DEFAULT_DEFINITIONS_PATH,
        help="Path to the benchmark-ready family definitions YAML.",
    )
    mine_parser.add_argument(
        "--query_path",
        type=Path,
        default=DEFAULT_QUERY_PATH,
        help="Path to the family retrieval queries YAML.",
    )
    mine_parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated candidate pools.",
    )
    mine_parser.set_defaults(func=mine_candidates)

    batch_parser = subparsers.add_parser(
        "build-annotation-batches",
        help="Sample annotation batches from existing candidate pools using the shared annotation template.",
    )
    batch_parser.add_argument(
        "--candidate_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "candidate_pools",
        help="Directory containing per-family candidate JSONL files.",
    )
    batch_parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "annotation_batches",
        help="Directory for sampled annotation batches.",
    )
    batch_parser.add_argument(
        "--batch_size",
        type=int,
        default=250,
        help="Target number of rows per family annotation batch.",
    )
    batch_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling annotation batches.",
    )
    batch_parser.add_argument(
        "--family_ids",
        nargs="*",
        default=None,
        help="Optional subset of family IDs to export.",
    )
    batch_parser.add_argument(
        "--definitions_path",
        type=Path,
        default=DEFAULT_DEFINITIONS_PATH,
        help="Path to the benchmark-ready family definitions YAML.",
    )
    batch_parser.add_argument(
        "--query_path",
        type=Path,
        default=DEFAULT_QUERY_PATH,
        help="Path to the family retrieval queries YAML.",
    )
    batch_parser.add_argument(
        "--template_path",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help="CSV file containing the annotation batch header.",
    )
    batch_parser.set_defaults(func=build_annotation_batches)

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
