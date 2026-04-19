import argparse
import csv
import re
from pathlib import Path


BENCHMARK_FAMILY_IDS = {
    "mlb_baseball",
    "nfl_football",
    "soccer",
    "nba_basketball",
    "nhl_hockey",
    "combat_sports",
    "gaming_general",
    "crypto_blockchain",
    "aviation_aerospace",
    "china",
    "japan",
    "russia_post_soviet",
    "middle_east_geopolitics",
    "us_electoral_politics",
    "us_legislative_governance",
}

LAYER_PATTERN = re.compile(r"/l(\d+)/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the L22 family benchmark feature registry from the canonical family match table."
    )
    parser.add_argument(
        "--family_match_details",
        type=Path,
        default=Path("presentation/260415/family_ontology_draft/family_match_details.csv"),
        help="Canonical family match table.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("presentation/260417/final_family_benchmark_l22/feature_registry_l22.csv"),
        help="Output registry CSV path.",
    )
    return parser.parse_args()


def parse_layer(source_path: str) -> int:
    match = LAYER_PATTERN.search(source_path)
    if not match:
        raise ValueError(f"Could not infer layer from source path: {source_path}")
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    rows = []
    seen = set()

    with args.family_match_details.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            family_id = row["family_id"]
            method = row["core_method"].strip()
            if family_id not in BENCHMARK_FAMILY_IDS or not method:
                continue

            feature_index = int(row["feature_id"])
            layer = parse_layer(row["source_path"])
            key = (family_id, method, layer, feature_index)
            if key in seen:
                continue
            seen.add(key)

            rows.append(
                {
                    "family_id": family_id,
                    "method": method,
                    "layer": layer,
                    "feature_index": feature_index,
                    "certainty": float(row["certainty"]),
                    "display_name": row["display_name"],
                    "category": row["category"],
                    "common_semantic": row["common_semantic"],
                    "source_path": row["source_path"],
                    "experiment_id": row["experiment_id"],
                }
            )

    rows.sort(
        key=lambda row: (
            row["family_id"],
            row["method"],
            row["layer"],
            -row["certainty"],
            row["feature_index"],
        )
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family_id",
                "method",
                "layer",
                "feature_index",
                "certainty",
                "display_name",
                "category",
                "common_semantic",
                "source_path",
                "experiment_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} feature registry rows to {args.output_path}")


if __name__ == "__main__":
    main()
