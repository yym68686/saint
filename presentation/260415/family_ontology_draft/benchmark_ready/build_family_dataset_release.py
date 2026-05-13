from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_ROOT = ROOT / "family_dataset_manualclean" / "annotation_batches"
DEFAULT_CANDIDATE_ROOT = ROOT / "family_dataset_manualclean" / "candidate_pools"
DEFAULT_DEFINITIONS_PATH = ROOT / "benchmark_ready_family_definitions.yaml"
DEFAULT_MANUAL_RULES_PATH = ROOT / "manual_cleaning_rules.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "family_dataset_release"

CANDIDATE_BUCKETS = [
    "positive",
    "local_hard_negative",
    "benchmark_hard_negative",
    "medium_negative",
    "background_negative",
]

KEEP_TRUE = {"yes", "true", "1"}
FINAL_LABEL_PRIORITY = {
    "target_positive": 5,
    "hard_negative": 4,
    "medium_negative": 3,
    "background_negative": 2,
    "ambiguous": 1,
    "drop": 0,
}
ANNOTATION_SOURCE_PRIORITY = {
    "manual_batch": 2,
    "auto_supplement": 1,
}
SOURCE_BUCKET_PRIORITY = {
    "positive": 5,
    "local_hard_negative": 4,
    "benchmark_hard_negative": 3,
    "medium_negative": 2,
    "background_negative": 1,
}
CONFIDENCE_PRIORITY = {"high": 3, "medium": 2, "low": 1, "": 0}
PREFERRED_SUPPLEMENT_SOURCES = {
    "target_positive": [
        "positive",
        "benchmark_hard_negative",
        "local_hard_negative",
        "medium_negative",
        "background_negative",
    ],
    "hard_negative": [
        "local_hard_negative",
        "benchmark_hard_negative",
        "positive",
        "medium_negative",
        "background_negative",
    ],
    "medium_negative": [
        "medium_negative",
        "positive",
        "benchmark_hard_negative",
        "local_hard_negative",
        "background_negative",
    ],
    "background_negative": [
        "background_negative",
        "positive",
        "medium_negative",
        "benchmark_hard_negative",
        "local_hard_negative",
    ],
}

GENERIC_ELECTION_CUES = [
    r"\bcandidate\b",
    r"\bturnout\b",
    r"\bendorsement\b",
    r"\bcaucus\b",
    r"\bpolling\b",
    r"\bcampaign manager\b",
    r"\bcampaign trail\b",
    r"\bpresidential race\b",
    r"\belection night\b",
    r"\bdelegate count\b",
    r"\bballot\b",
    r"\bswing state\b",
    r"\bprimary election\b",
    r"\brecount\b",
    r"\bvot(?:e|er|ing)\b",
    r"\bcampaign\b",
    r"\bpresident(?:ial)?\b",
]
US_ELECTORAL_US_CUES = [
    r"\bu\.s\.\b",
    r"\bunited states\b",
    r"\btrump\b",
    r"\bclinton\b",
    r"\bobama\b",
    r"\bsanders\b",
    r"\bted cruz\b",
    r"\bdonald\b",
    r"\brepublican\b",
    r"\bdemocratic?\b",
    r"\bgop\b",
    r"\bwhite house\b",
    r"\bcongress\b",
    r"\bsenate\b",
    r"\bgovern(or|ment|mental|orial)\b",
    r"\bmayor\b",
    r"\bdistrict\b",
    r"\bminnesota\b",
    r"\barkansas\b",
    r"\biowa\b",
    r"\bnew hampshire\b",
    r"\bdallas\b",
    r"\bpac\b",
    r"\bwashington post\b",
    r"\bcounty\b",
    r"\bmsnbc\b",
    r"\blibertarian candidate u\.s\. congress\b",
]
US_ELECTORAL_FOREIGN_MARKERS = [
    r"\blabour\b",
    r"\bwestern australia\b",
    r"\bmodi\b",
    r"\bottawa\b",
    r"\bcanada\b",
    r"\buruguay\b",
    r"\bsingfirst\b",
    r"\bang mo kio\b",
    r"\bvanathi\b",
    r"\biran\b",
    r"\btusc\b",
    r"\bstintz\b",
]
US_LEG_CORE_CUES = [
    r"\bcongress\b",
    r"\bsenate\b",
    r"\bhouse of representatives\b",
    r"\bexecutive order\b",
    r"\bappropriations?\b",
    r"\bfilibuster\b",
    r"\bcommittee\b",
    r"\bfederal agency\b",
    r"\bfederal register\b",
    r"\bbill passed\b",
    r"\bcommittee markup\b",
    r"\bbudget resolution\b",
    r"\bconfirmation vote\b",
    r"\bwhite house\b",
    r"\bhearing\b",
    r"\bfederal\b",
]
STRICT_TARGET_PATTERNS = {
    "mlb_baseball": [
        r"major league baseball",
        r"world series",
        r"american league",
        r"national league",
        r"new york yankees",
        r"los angeles dodgers",
        r"boston red sox",
        r"chicago cubs",
        r"new york mets",
        r"atlanta braves",
        r"\bshortstop\b",
        r"\bpitcher\b",
        r"\bbullpen\b",
        r"\binning\b",
        r"home plate",
        r"batting average",
    ],
    "nfl_football": [
        r"\bnfl\b",
        r"super bowl",
        r"nfl draft",
        r"\bquarterback\b",
        r"\btouchdown\b",
        r"\blinebacker\b",
        r"offensive line",
        r"green bay packers",
        r"new england patriots",
        r"kansas city chiefs",
        r"dallas cowboys",
        r"pittsburgh steelers",
        r"san francisco 49ers",
        r"\bred zone\b",
        r"\bafc\b",
        r"\bnfc\b",
    ],
    "soccer": [
        r"premier league",
        r"champions league",
        r"\bfifa\b(?!\s*1\d)",
        r"manchester united",
        r"real madrid",
        r"\bbundesliga\b",
        r"\bstriker\b",
        r"\bmidfielder\b",
        r"\bgoalkeeper\b",
        r"penalty area",
        r"fc barcelona",
        r"\bmls\b",
        r"major league soccer",
        r"\bserie a\b",
        r"\bla liga\b",
        r"transfer window",
        r"soccer federation",
        r"\bworld cup\b",
        r"\bac milan\b",
        r"ipswich town",
    ],
    "nba_basketball": [
        r"\bnba\b",
        r"nba finals",
        r"nba draft",
        r"point guard",
        r"triple-double",
        r"pick and roll",
        r"\bbackcourt\b",
        r"free throw",
        r"los angeles lakers",
        r"boston celtics",
        r"golden state warriors",
        r"new york knicks",
        r"milwaukee bucks",
        r"miami heat",
        r"\bcavaliers\b",
        r"\bplayoffs?\b",
    ],
    "nhl_hockey": [
        r"\bnhl\b",
        r"stanley cup",
        r"nhl draft",
        r"nhl playoffs",
        r"toronto maple leafs",
        r"vancouver canucks",
        r"boston bruins",
        r"chicago blackhawks",
        r"new york rangers",
        r"edmonton oilers",
        r"\bgoalie\b",
        r"\bpuck\b",
        r"power play",
        r"penalty kill",
        r"ice hockey",
    ],
    "combat_sports": [
        r"\bufc\b",
        r"\bmma\b",
        r"\bbellator\b",
        r"\boctagon\b",
        r"pay-per-view",
        r"main event",
        r"fight camp",
        r"\bsparring\b",
        r"title shot",
        r"\bweigh-in\b",
        r"\bfighter\b",
        r"\bknockout\b",
        r"boxing champion",
        r"boxing match",
        r"boxing promoter",
        r"heavyweight champion",
    ],
    "gaming_general": [
        r"video game",
        r"game developer",
        r"\bplaystation\b",
        r"\bxbox\b",
        r"\bnintendo\b",
        r"\besports\b",
        r"patch notes",
        r"\bmultiplayer\b",
        r"ranked play",
        r"battle pass",
        r"\bspeedrun\b",
        r"boss fight",
        r"skill tree",
        r"loot drop",
        r"open world",
        r"fps game",
        r"\bsteam\b",
    ],
    "crypto_blockchain": [
        r"\bbitcoin\b",
        r"\bethereum\b",
        r"\bblockchain\b",
        r"smart contract",
        r"\bdefi\b",
        r"crypto exchange",
        r"\bsolana\b",
        r"\bbinance\b",
        r"\bcoinbase\b",
        r"\bnft\b",
        r"\bon-chain\b",
        r"layer 2",
        r"\bcryptocurrenc(?:y|ies)\b",
        r"\bcrypto\b",
        r"token sale",
        r"mining rig",
        r"mining pool",
        r"private keys",
        r"digital assets",
    ],
    "aviation_aerospace": [
        r"\bairline\b",
        r"\brunway\b",
        r"\bfaa\b",
        r"\bboeing\b",
        r"\bairbus\b",
        r"\bspacex\b",
        r"\bnasa\b",
        r"rocket launch",
        r"\bsatellite\b",
        r"\bspacecraft\b",
        r"mission control",
        r"\bpilot\b",
        r"flight deck",
        r"air traffic control",
        r"\borbital\b",
        r"\bpayload\b",
        r"\bavionics\b",
        r"\baircraft\b(?! carriers?)",
    ],
}
STRICT_TARGET_EXCLUSIONS = {
    "mlb_baseball": [r"college baseball", r"softball", r"\bcricket\b", r"little league"],
    "nfl_football": [r"college football", r"\bncaa\b", r"\brugby\b"],
    "soccer": [
        r"pro evolution soccer",
        r"\bkonami\b",
        r"\bea sports\b",
        r"\bfifa 1\d\b",
        r"video game",
        r"high school soccer",
        r"soccer coach",
        r"captain of the soccer team",
        r"soccer fields",
        r"referees association",
        r"amateur soccer league",
        r"youth side",
        r"under-18 national team",
        r"women soccer",
        r"women's soccer",
    ],
    "nba_basketball": [r"college basketball", r"\bwnba\b", r"\bfiba\b"],
    "nhl_hockey": [r"field hockey", r"winter olympics", r"minor league hockey"],
    "combat_sports": [r"\bWWE\b", r"NJPW", r"\bIWGP\b", r"professional wrestling"],
    "gaming_general": [],
    "crypto_blockchain": [],
    "aviation_aerospace": [r"aircraft carriers?", r"star trek"],
}
CANONICAL_LABEL_PATTERNS = {
    "mlb_baseball": re.compile(r"\b(baseball|mlb|major league baseball)\b", re.I),
    "nfl_football": re.compile(r"\b(nfl|football|super bowl)\b", re.I),
    "soccer": re.compile(
        r"\b(soccer|premier league|champions league|fifa|mls|major league soccer|la liga|serie a|bundesliga)\b",
        re.I,
    ),
    "nba_basketball": re.compile(r"\b(nba|basketball)\b", re.I),
    "nhl_hockey": re.compile(r"\b(nhl|hockey|stanley cup)\b", re.I),
    "combat_sports": re.compile(r"\b(ufc|mma|boxing|bellator|fight|fighter)\b", re.I),
    "gaming_general": re.compile(r"\b(video game|playstation|xbox|nintendo|esports|steam|game)\b", re.I),
    "crypto_blockchain": re.compile(r"\b(bitcoin|crypto|blockchain|ethereum|wallet|nft|coinbase|binance|solana)\b", re.I),
    "aviation_aerospace": re.compile(r"\b(aviation|airline|aircraft|boeing|airbus|nasa|spacex|satellite|spacecraft)\b", re.I),
    "china": re.compile(r"\b(china|chinese|beijing|shanghai|hong kong)\b", re.I),
    "japan": re.compile(r"\b(japan|japanese|tokyo|osaka|kyoto)\b", re.I),
    "russia_post_soviet": re.compile(r"\b(russia|russian|ukraine|kremlin|moscow|crimea|belarus|donbas)\b", re.I),
    "middle_east_geopolitics": re.compile(r"\b(iran|iraq|syria|gaza|israel|palestinian|middle east|saudi arabia|tehran|hezbollah)\b", re.I),
    "us_electoral_politics": re.compile(r"\b(candidate|campaign|polling|caucus|ballot|voter|turnout|election|primary|delegate)\b", re.I),
    "us_legislative_governance": re.compile(r"\b(congress|senate|house|committee|hearing|executive order|appropriations|filibuster|confirmation|bill)\b", re.I),
}


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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def split_semicolon_field(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def is_keep(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("keep_for_benchmark", "")).strip().lower() in KEEP_TRUE
        and row.get("final_label") not in {"drop", "ambiguous", ""}
    )


def parse_batch_bucket(notes: str) -> str:
    for chunk in notes.split(";"):
        chunk = chunk.strip()
        if chunk.startswith("batch_bucket="):
            return chunk.split("=", 1)[1]
    return ""


def compile_patterns(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def any_pattern(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_family_seed(base_seed: int, family_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{family_id}".encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def row_priority(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        1 if is_keep(row) else 0,
        ANNOTATION_SOURCE_PRIORITY.get(str(row.get("annotation_source", "")), 0),
        FINAL_LABEL_PRIORITY.get(str(row.get("final_label", "")), -1),
        CONFIDENCE_PRIORITY.get(str(row.get("label_confidence", "")), 0),
        SOURCE_BUCKET_PRIORITY.get(str(row.get("sampled_from_bucket", "")), 0),
    )


def dedupe_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        normalized = normalize_text(str(row["text"]))
        candidate = dict(row)
        candidate["normalized_text"] = normalized
        existing = deduped.get(normalized)
        if existing is None or row_priority(candidate) > row_priority(existing):
            deduped[normalized] = candidate
    return list(deduped.values())


def summarize_by_label(rows: Iterable[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counter[(str(row["final_label"]), str(row["keep_for_benchmark"]))] += 1
    return counter


def required_counts_from_protocol(protocol: Mapping[str, Any]) -> tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    split_recipe = protocol["split_recipe"]
    control_mix = protocol["control_mix"]
    selection = {
        "target_positive": int(split_recipe["selection_target"]),
        "hard_negative": int(round(split_recipe["selection_control"] * control_mix["hard_negative_share"])),
        "medium_negative": int(round(split_recipe["selection_control"] * control_mix["medium_negative_share"])),
        "background_negative": int(round(split_recipe["selection_control"] * control_mix["background_share"])),
    }
    calibration = {
        "hard_negative": int(round(split_recipe["calibration_control"] * control_mix["hard_negative_share"])),
        "medium_negative": int(round(split_recipe["calibration_control"] * control_mix["medium_negative_share"])),
        "background_negative": int(round(split_recipe["calibration_control"] * control_mix["background_share"])),
    }
    evaluation = {
        "target_positive": int(split_recipe["evaluation_target"]),
        "hard_negative": int(round(split_recipe["evaluation_control"] * control_mix["hard_negative_share"])),
        "medium_negative": int(round(split_recipe["evaluation_control"] * control_mix["medium_negative_share"])),
        "background_negative": int(round(split_recipe["evaluation_control"] * control_mix["background_share"])),
    }
    required = {
        "target_positive": selection["target_positive"] + evaluation["target_positive"],
        "hard_negative": selection["hard_negative"] + calibration["hard_negative"] + evaluation["hard_negative"],
        "medium_negative": selection["medium_negative"] + calibration["medium_negative"] + evaluation["medium_negative"],
        "background_negative": selection["background_negative"] + calibration["background_negative"] + evaluation["background_negative"],
    }
    return required, {
        "selection": selection,
        "calibration": calibration,
        "evaluation": evaluation,
    }


def contains_canonical_label(family_id: str, text: str) -> bool:
    return bool(CANONICAL_LABEL_PATTERNS[family_id].search(text))


def is_too_short(text: str) -> bool:
    return len(text.strip()) < 25


def is_multi_topic_schedule(text: str) -> bool:
    lowered = text.lower()
    if "program net episode start end total viewers" in lowered:
        return True
    if len(text) > 1200 and text.count("NFL NETWORK") >= 2 and text.count("MLB NETWORK") >= 2:
        return True
    if len(text) > 1200 and text.count("NBA-TV") >= 2 and text.count("FOX SPORTS") >= 2:
        return True
    return False


def is_country_selector_page(text: str) -> bool:
    lowered = text.lower()
    return "please select your country:" in lowered and "not an american user?" in lowered


def is_world_leader_poll_tweet(text: str) -> bool:
    lowered = text.lower()
    return "net favorables of world leaders" in lowered and "putin:" in lowered


def is_us_electoral_target(text: str) -> bool:
    lowered = text.lower()
    foreign_patterns = compile_patterns(US_ELECTORAL_FOREIGN_MARKERS)
    if any_pattern(foreign_patterns, lowered):
        return False
    election_patterns = compile_patterns(GENERIC_ELECTION_CUES)
    us_patterns = compile_patterns(US_ELECTORAL_US_CUES)
    return any_pattern(election_patterns, lowered) and any_pattern(us_patterns, lowered)


def is_us_leg_target(text: str) -> bool:
    lowered = text.lower()
    core_patterns = compile_patterns(US_LEG_CORE_CUES)
    if any_pattern(core_patterns, lowered):
        return True
    if "oversight" in lowered and any(
        cue in lowered
        for cue in [
            "house",
            "congress",
            "senate",
            "committee",
            "hearing",
            "government reform",
            "white house",
            "sessions",
            "federal",
        ]
    ):
        return True
    return False


def is_crypto_target(text: str, strict_patterns: List[re.Pattern[str]]) -> bool:
    lowered = text.lower()
    if any_pattern(strict_patterns, lowered):
        return True
    if "wallet" in lowered and any(
        cue in lowered
        for cue in [
            "private keys",
            "digital assets",
            "cryptocurrency",
            "bitcoin",
            "ethereum",
            "blockchain",
            "coinbase",
            "binance",
            "solana",
            "nft",
            "trading the currency",
        ]
    ):
        return True
    return False


def is_aviation_target(text: str, strict_patterns: List[re.Pattern[str]]) -> bool:
    lowered = text.lower()
    if "aircraft carrier" in lowered or "aircraft carriers" in lowered:
        return any(
            cue in lowered
            for cue in ["airline", "boeing", "airbus", "nasa", "spacex", "satellite", "spacecraft", "pilot", "faa", "avionics"]
        )
    return any_pattern(strict_patterns, lowered)


def is_target_match(
    family_id: str,
    text: str,
    strict_patterns_by_family: Mapping[str, List[re.Pattern[str]]],
    strict_exclusions_by_family: Mapping[str, List[re.Pattern[str]]],
) -> bool:
    strict_patterns = strict_patterns_by_family.get(family_id, [])
    strict_exclusions = strict_exclusions_by_family.get(family_id, [])
    lowered = text.lower()
    if strict_exclusions and any_pattern(strict_exclusions, lowered):
        return False
    if family_id == "us_electoral_politics":
        return is_us_electoral_target(text)
    if family_id == "us_legislative_governance":
        return is_us_leg_target(text)
    if family_id == "crypto_blockchain":
        return is_crypto_target(text, strict_patterns)
    if family_id == "aviation_aerospace":
        return is_aviation_target(text, strict_patterns)
    return any_pattern(strict_patterns, lowered)


def default_annotation(source_bucket: str) -> Dict[str, str]:
    if source_bucket == "positive":
        return {
            "final_label": "target_positive",
            "keep_for_benchmark": "yes",
            "label_confidence": "high",
            "primary_reason": "clear_in_family_anchor",
            "exclusion_trigger": "",
        }
    if source_bucket in {"local_hard_negative", "benchmark_hard_negative"}:
        return {
            "final_label": "hard_negative",
            "keep_for_benchmark": "yes",
            "label_confidence": "high",
            "primary_reason": "nearby_confounder",
            "exclusion_trigger": "",
        }
    if source_bucket == "medium_negative":
        return {
            "final_label": "medium_negative",
            "keep_for_benchmark": "yes",
            "label_confidence": "high",
            "primary_reason": "topical_but_not_primary",
            "exclusion_trigger": "",
        }
    return {
        "final_label": "background_negative",
        "keep_for_benchmark": "yes",
        "label_confidence": "high",
        "primary_reason": "clean_background_control",
        "exclusion_trigger": "",
    }


def positive_demote_annotation(family_id: str, text: str, matched_terms: List[str]) -> Dict[str, str] | None:
    lowered = text.lower()

    if family_id == "soccer":
        if any(term in lowered for term in ["pro evolution soccer", "konami", "ea sports", "fifa 14", "fifa 15", "video game"]):
            return {
                "final_label": "medium_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "high",
                "primary_reason": "topical_but_not_primary",
                "exclusion_trigger": "video_game_context",
            }
        if any(
            term in lowered
            for term in [
                "soccer fields",
                "captain of the soccer team",
                "high school soccer",
                "soccer coach",
                "referees association",
                "amateur soccer league",
                "women's soccer",
                "women soccer",
                "youth side",
                "under-18 national team",
                "started getting into both soccer and hockey",
            ]
        ):
            return {
                "final_label": "medium_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "high",
                "primary_reason": "topical_but_not_primary",
                "exclusion_trigger": "non_benchmark_soccer_context",
            }
        if "author of the blog" in lowered or "senior fellow" in lowered:
            return {
                "final_label": "background_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "high",
                "primary_reason": "clean_background_control",
                "exclusion_trigger": "author_bio_fragment",
            }
        if any(term in lowered for term in ["soccer is markedly better", "most exciting aspect in soccer", "major soccer tournament"]):
            return {
                "final_label": "medium_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "medium",
                "primary_reason": "topical_but_not_primary",
                "exclusion_trigger": "generic_sport_commentary",
            }
        return {
            "final_label": "medium_negative",
            "keep_for_benchmark": "yes",
            "label_confidence": "medium",
            "primary_reason": "topical_but_not_primary",
            "exclusion_trigger": "generic_soccer_context",
        }

    if family_id == "us_electoral_politics":
        if any_pattern(compile_patterns(US_ELECTORAL_FOREIGN_MARKERS), lowered):
            return {
                "final_label": "medium_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "high",
                "primary_reason": "topical_but_not_primary",
                "exclusion_trigger": "non_us_electoral_context",
            }
        if any(term in lowered for term in ["financial endorsement", "help us finish off this month"]):
            return {
                "final_label": "background_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "high",
                "primary_reason": "clean_background_control",
                "exclusion_trigger": "generic_endorsement_request",
            }
        if any(
            term in lowered
            for term in [
                "breakout success",
                "authorization for the use of military force",
                "potential candidate",
                "pre-release candidate",
                "dedicated transit lane",
                "replace mullen",
                "head-coaching job",
                "young vince",
                "multipath tcp",
                "gradient in shell coiling",
            ]
        ):
            return {
                "final_label": "background_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "high",
                "primary_reason": "clean_background_control",
                "exclusion_trigger": "generic_candidate_usage",
            }
        if "turnout" in matched_terms or "candidate" in matched_terms or "endorsement" in matched_terms:
            return {
                "final_label": "medium_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "medium",
                "primary_reason": "topical_but_not_primary",
                "exclusion_trigger": "unclear_us_electoral_scope",
            }
        return {
            "final_label": "background_negative",
            "keep_for_benchmark": "yes",
            "label_confidence": "medium",
            "primary_reason": "clean_background_control",
            "exclusion_trigger": "non_benchmark_candidate_context",
        }

    if family_id == "us_legislative_governance":
        if any(
            term in lowered
            for term in [
                "board of trustees",
                "university",
                "lpl",
                "public procurement",
                "virtual currency",
                "public oversight",
            ]
        ):
            return {
                "final_label": "medium_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "high",
                "primary_reason": "topical_but_not_primary",
                "exclusion_trigger": "non_government_oversight_context",
            }
        return {
            "final_label": "medium_negative",
            "keep_for_benchmark": "yes",
            "label_confidence": "medium",
            "primary_reason": "topical_but_not_primary",
            "exclusion_trigger": "generic_governance_context",
        }

    if family_id == "crypto_blockchain":
        if "wallet" in matched_terms or "wallet" in lowered:
            if any(term in lowered for term in ["private keys", "digital assets", "cryptocurrency", "bitcoin", "ethereum", "blockchain", "coinbase", "binance", "solana"]):
                return None
            if any(term in lowered for term in ["amazon wallet", "google wallet", "apple pay", "credit cards", "debit cards", "nfc", "payment", "payment wallet", "trading the currency"]):
                return {
                    "final_label": "medium_negative",
                    "keep_for_benchmark": "yes",
                    "label_confidence": "high",
                    "primary_reason": "topical_but_not_primary",
                    "exclusion_trigger": "fintech_wallet_context",
                }
            return {
                "final_label": "background_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "high",
                "primary_reason": "clean_background_control",
                "exclusion_trigger": "generic_wallet_usage",
            }
        return {
            "final_label": "medium_negative",
            "keep_for_benchmark": "yes",
            "label_confidence": "medium",
            "primary_reason": "topical_but_not_primary",
            "exclusion_trigger": "generic_fintech_context",
        }

    if family_id == "aviation_aerospace":
        if "aircraft carrier" in lowered or "aircraft carriers" in lowered or "star trek" in lowered:
            return {
                "final_label": "medium_negative",
                "keep_for_benchmark": "yes",
                "label_confidence": "high",
                "primary_reason": "topical_but_not_primary",
                "exclusion_trigger": "carrier_or_pop_culture_context",
            }
        return {
            "final_label": "medium_negative",
            "keep_for_benchmark": "yes",
            "label_confidence": "medium",
            "primary_reason": "topical_but_not_primary",
            "exclusion_trigger": "generic_transport_context",
        }

    if family_id == "middle_east_geopolitics" and is_country_selector_page(text):
        return {
            "final_label": "drop",
            "keep_for_benchmark": "no",
            "label_confidence": "high",
            "primary_reason": "duplicate_or_low_quality",
            "exclusion_trigger": "country_list_fragment",
        }

    if family_id in {"mlb_baseball", "nfl_football", "nba_basketball", "nhl_hockey", "combat_sports"}:
        return {
            "final_label": "medium_negative",
            "keep_for_benchmark": "yes",
            "label_confidence": "medium",
            "primary_reason": "topical_but_not_primary",
            "exclusion_trigger": "non_benchmark_sports_context",
        }

    return None


def auto_label_candidate(
    family_id: str,
    source_bucket: str,
    row: Mapping[str, Any],
    strict_patterns_by_family: Mapping[str, List[re.Pattern[str]]],
    strict_exclusions_by_family: Mapping[str, List[re.Pattern[str]]],
) -> Dict[str, Any]:
    text = str(row["text"])
    matched_terms = [str(term).lower() for term in row.get("matched_terms", [])]
    base = default_annotation(source_bucket)

    if is_too_short(text):
        base.update(
            {
                "final_label": "drop",
                "keep_for_benchmark": "no",
                "label_confidence": "high",
                "primary_reason": "duplicate_or_low_quality",
                "exclusion_trigger": "too_short_or_fragment",
            }
        )
    elif is_multi_topic_schedule(text):
        base.update(
            {
                "final_label": "ambiguous",
                "keep_for_benchmark": "no",
                "label_confidence": "low",
                "primary_reason": "too_ambiguous",
                "exclusion_trigger": "multi_topic_schedule",
            }
        )
    elif family_id == "middle_east_geopolitics" and is_country_selector_page(text):
        base.update(
            {
                "final_label": "drop",
                "keep_for_benchmark": "no",
                "label_confidence": "high",
                "primary_reason": "duplicate_or_low_quality",
                "exclusion_trigger": "country_list_fragment",
            }
        )
    elif family_id == "russia_post_soviet" and is_world_leader_poll_tweet(text):
        base.update(
            {
                "final_label": "ambiguous",
                "keep_for_benchmark": "no",
                "label_confidence": "low",
                "primary_reason": "too_ambiguous",
                "exclusion_trigger": "multi_topic_poll",
            }
        )
    else:
        target_match = is_target_match(
            family_id=family_id,
            text=text,
            strict_patterns_by_family=strict_patterns_by_family,
            strict_exclusions_by_family=strict_exclusions_by_family,
        )
        if target_match:
            base.update(
                {
                    "final_label": "target_positive",
                    "keep_for_benchmark": "yes",
                    "label_confidence": "high" if source_bucket == "positive" else "medium",
                    "primary_reason": "clear_in_family_anchor" if source_bucket == "positive" else "entity_plus_context",
                    "exclusion_trigger": "",
                }
            )
        elif source_bucket == "positive":
            demoted = positive_demote_annotation(family_id, text, matched_terms)
            if demoted is not None:
                base.update(demoted)
            else:
                base.update(
                    {
                        "final_label": "medium_negative",
                        "keep_for_benchmark": "yes",
                        "label_confidence": "medium",
                        "primary_reason": "topical_but_not_primary",
                        "exclusion_trigger": "positive_pool_but_not_target_scope",
                    }
                )

    labeled = {
        "current_family_id": family_id,
        "candidate_id": row["candidate_id"],
        "source_family_id": row["family_id"],
        "source_candidate_bucket": row["proposed_bucket"],
        "sampled_from_bucket": source_bucket,
        "source_row_index": row["source_row_index"],
        "text": text,
        "matched_query_blocks": list(row.get("matched_query_blocks", [])),
        "matched_terms": list(row.get("matched_terms", [])),
        "matched_family_ids": list(row.get("matched_family_ids", [])),
        "final_label": base["final_label"],
        "keep_for_benchmark": base["keep_for_benchmark"],
        "label_confidence": base["label_confidence"],
        "primary_reason": base["primary_reason"],
        "exclusion_trigger": base["exclusion_trigger"],
        "notes": f"annotation_source=auto_supplement; sampled_from_bucket={source_bucket}",
        "annotation_source": "auto_supplement",
    }
    return labeled


def load_current_annotations(annotation_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    annotations: Dict[str, List[Dict[str, Any]]] = {}
    for family_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir()):
        rows: List[Dict[str, Any]] = []
        csv_path = family_dir / "annotation_batch.csv"
        with csv_path.open("r", newline="") as f:
            for raw in csv.DictReader(f):
                rows.append(
                    {
                        "current_family_id": family_dir.name,
                        "candidate_id": raw["candidate_id"],
                        "source_family_id": raw["family_id"],
                        "source_candidate_bucket": raw["proposed_bucket"],
                        "sampled_from_bucket": parse_batch_bucket(raw["notes"]),
                        "source_row_index": raw["source_row_index"],
                        "text": raw["text"],
                        "matched_query_blocks": split_semicolon_field(raw["matched_query_blocks"]),
                        "matched_terms": split_semicolon_field(raw["matched_terms"]),
                        "matched_family_ids": split_semicolon_field(raw["matched_family_ids"]),
                        "final_label": raw["final_label"],
                        "keep_for_benchmark": raw["keep_for_benchmark"],
                        "label_confidence": raw["label_confidence"],
                        "primary_reason": raw["primary_reason"],
                        "exclusion_trigger": raw["exclusion_trigger"],
                        "notes": raw["notes"],
                        "annotation_source": "manual_batch",
                    }
                )
        annotations[family_dir.name] = rows
    return annotations


def build_annotation_summary_rows(
    annotations_by_family: Mapping[str, List[Mapping[str, Any]]],
    dedupe_kept: bool,
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for family_id, rows in annotations_by_family.items():
        source_rows = dedupe_rows(rows) if dedupe_kept else [dict(row) for row in rows]
        if dedupe_kept:
            source_rows = [row for row in source_rows if is_keep(row)]
        counter = Counter()
        for row in source_rows:
            counter[(row["final_label"], row["keep_for_benchmark"])] += 1
        for (final_label, keep_for_benchmark), count in sorted(counter.items()):
            summary_rows.append(
                {
                    "family_id": family_id,
                    "final_label": final_label,
                    "keep_for_benchmark": keep_for_benchmark,
                    "count": count,
                    "summary_type": "dedup_keep_only" if dedupe_kept else "raw_annotations",
                }
            )
    return summary_rows


def write_csv(path: Path, rows: List[Mapping[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_gap_rows(
    family_ids: List[str],
    current_rows_by_family: Mapping[str, List[Mapping[str, Any]]],
    required_counts: Mapping[str, int],
) -> List[Dict[str, Any]]:
    gap_rows: List[Dict[str, Any]] = []
    for family_id in family_ids:
        deduped = [row for row in dedupe_rows(current_rows_by_family[family_id]) if is_keep(row)]
        counts = Counter(row["final_label"] for row in deduped)
        for final_label, required in required_counts.items():
            available = counts.get(final_label, 0)
            gap_rows.append(
                {
                    "family_id": family_id,
                    "final_label": final_label,
                    "available_count": available,
                    "required_count": required,
                    "gap_count": max(0, required - available),
                }
            )
    return gap_rows


def build_remaining_auto_pool(
    family_id: str,
    candidate_root: Path,
    existing_rows: List[Mapping[str, Any]],
    strict_patterns_by_family: Mapping[str, List[re.Pattern[str]]],
    strict_exclusions_by_family: Mapping[str, List[re.Pattern[str]]],
) -> List[Dict[str, Any]]:
    seen_candidate_ids = {str(row["candidate_id"]) for row in existing_rows}
    seen_texts = {normalize_text(str(row["text"])) for row in existing_rows}
    auto_rows: List[Dict[str, Any]] = []

    for source_bucket in CANDIDATE_BUCKETS:
        source_rows = read_jsonl(candidate_root / family_id / f"{source_bucket}.jsonl")
        for source_row in source_rows:
            normalized = normalize_text(str(source_row["text"]))
            if source_row["candidate_id"] in seen_candidate_ids or normalized in seen_texts:
                continue
            labeled = auto_label_candidate(
                family_id=family_id,
                source_bucket=source_bucket,
                row=source_row,
                strict_patterns_by_family=strict_patterns_by_family,
                strict_exclusions_by_family=strict_exclusions_by_family,
            )
            auto_rows.append(labeled)

    return dedupe_rows(auto_rows)


def count_label_free_targets(rows: Iterable[Mapping[str, Any]], family_id: str) -> int:
    return sum(
        1
        for row in rows
        if row["final_label"] == "target_positive" and not contains_canonical_label(family_id, str(row["text"]))
    )


def select_target_supplements(
    family_id: str,
    auto_rows: List[Dict[str, Any]],
    current_target_rows: List[Mapping[str, Any]],
    target_quota: int,
    min_label_free_needed: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if target_quota <= 0:
        return []
    target_candidates = [
        row
        for row in auto_rows
        if row["final_label"] == "target_positive" and row["sampled_from_bucket"] in PREFERRED_SUPPLEMENT_SOURCES["target_positive"]
    ]
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in target_candidates:
        by_bucket[row["sampled_from_bucket"]].append(row)
    for rows in by_bucket.values():
        rng.shuffle(rows)

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    label_free_candidates = [
        row
        for bucket in PREFERRED_SUPPLEMENT_SOURCES["target_positive"]
        for row in by_bucket.get(bucket, [])
        if not contains_canonical_label(family_id, str(row["text"]))
    ]
    rng.shuffle(label_free_candidates)
    need_no_label = max(0, min_label_free_needed - count_label_free_targets(current_target_rows, family_id))
    for row in label_free_candidates:
        if len(selected) >= target_quota:
            break
        if row["candidate_id"] in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row["candidate_id"])
        if len([item for item in selected if not contains_canonical_label(family_id, str(item["text"]))]) >= need_no_label:
            break

    for bucket in PREFERRED_SUPPLEMENT_SOURCES["target_positive"]:
        for row in by_bucket.get(bucket, []):
            if len(selected) >= target_quota:
                break
            if row["candidate_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["candidate_id"])
        if len(selected) >= target_quota:
            break

    return selected


def select_rows_by_label(
    family_id: str,
    auto_rows: List[Dict[str, Any]],
    final_label: str,
    quota: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if quota <= 0:
        return []
    candidates = [row for row in auto_rows if row["final_label"] == final_label]
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bucket[row["sampled_from_bucket"]].append(row)
    for rows in by_bucket.values():
        rng.shuffle(rows)

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    for bucket in PREFERRED_SUPPLEMENT_SOURCES[final_label]:
        for row in by_bucket.get(bucket, []):
            if len(selected) >= quota:
                break
            if row["candidate_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["candidate_id"])
        if len(selected) >= quota:
            break
    return selected


def write_canonical_rows_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    fieldnames = [
        "current_family_id",
        "candidate_id",
        "source_family_id",
        "source_candidate_bucket",
        "sampled_from_bucket",
        "source_row_index",
        "text",
        "matched_query_blocks",
        "matched_terms",
        "matched_family_ids",
        "final_label",
        "keep_for_benchmark",
        "label_confidence",
        "primary_reason",
        "exclusion_trigger",
        "notes",
        "annotation_source",
    ]
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                **{key: row.get(key, "") for key in fieldnames},
                "matched_query_blocks": ";".join(row.get("matched_query_blocks", [])),
                "matched_terms": ";".join(row.get("matched_terms", [])),
                "matched_family_ids": ";".join(row.get("matched_family_ids", [])),
            }
        )
    write_csv(path, normalized_rows, fieldnames)


def build_frozen_rows(
    current_rows: List[Mapping[str, Any]],
    supplement_rows: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    combined = [dict(row) for row in current_rows if is_keep(row)] + [dict(row) for row in supplement_rows if is_keep(row)]
    return [row for row in dedupe_rows(combined) if is_keep(row)]


def split_family_dataset(
    family_id: str,
    frozen_rows: List[Mapping[str, Any]],
    split_quota: Mapping[str, Dict[str, int]],
    rng: random.Random,
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in frozen_rows:
        by_label[str(row["final_label"])].append(dict(row))
    for rows in by_label.values():
        rng.shuffle(rows)

    required_counts = {
        "target_positive": split_quota["selection"]["target_positive"] + split_quota["evaluation"]["target_positive"],
        "hard_negative": split_quota["selection"]["hard_negative"] + split_quota["calibration"]["hard_negative"] + split_quota["evaluation"]["hard_negative"],
        "medium_negative": split_quota["selection"]["medium_negative"] + split_quota["calibration"]["medium_negative"] + split_quota["evaluation"]["medium_negative"],
        "background_negative": split_quota["selection"]["background_negative"] + split_quota["calibration"]["background_negative"] + split_quota["evaluation"]["background_negative"],
    }
    availability = {label: len(by_label.get(label, [])) for label in required_counts}
    ready = all(availability[label] >= required for label, required in required_counts.items())

    manifest = {
        "family_id": family_id,
        "split_ready": "yes" if ready else "no",
        "available_target_positive": availability["target_positive"],
        "available_hard_negative": availability["hard_negative"],
        "available_medium_negative": availability["medium_negative"],
        "available_background_negative": availability["background_negative"],
    }
    if not ready:
        return {}, manifest

    splits: Dict[str, List[Dict[str, Any]]] = {
        "selection_target": [],
        "selection_control": [],
        "calibration_control": [],
        "evaluation_target": [],
        "evaluation_control": [],
    }

    target_rows = list(by_label["target_positive"])
    label_free = [row for row in target_rows if not contains_canonical_label(family_id, str(row["text"]))]
    label_explicit = [row for row in target_rows if contains_canonical_label(family_id, str(row["text"]))]
    rng.shuffle(label_free)
    rng.shuffle(label_explicit)
    required_label_free = int(math.ceil(split_quota["evaluation"]["target_positive"] * 0.25))
    evaluation_target: List[Dict[str, Any]] = []
    evaluation_target.extend(label_free[:required_label_free])
    remaining_pool = label_free[required_label_free:] + label_explicit
    rng.shuffle(remaining_pool)
    evaluation_target.extend(remaining_pool[: split_quota["evaluation"]["target_positive"] - len(evaluation_target)])
    used_target_ids = {row["candidate_id"] for row in evaluation_target}
    remaining_targets = [row for row in target_rows if row["candidate_id"] not in used_target_ids]
    rng.shuffle(remaining_targets)
    selection_target = remaining_targets[: split_quota["selection"]["target_positive"]]
    splits["evaluation_target"] = evaluation_target
    splits["selection_target"] = selection_target
    manifest["evaluation_target_no_canonical_label_count"] = sum(
        1 for row in evaluation_target if not contains_canonical_label(family_id, str(row["text"]))
    )

    for label in ["hard_negative", "medium_negative", "background_negative"]:
        rows = list(by_label[label])
        rng.shuffle(rows)
        selection_n = split_quota["selection"][label]
        calibration_n = split_quota["calibration"][label]
        evaluation_n = split_quota["evaluation"][label]
        selection_rows = rows[:selection_n]
        calibration_rows = rows[selection_n : selection_n + calibration_n]
        evaluation_rows = rows[selection_n + calibration_n : selection_n + calibration_n + evaluation_n]
        splits["selection_control"].extend(selection_rows)
        splits["calibration_control"].extend(calibration_rows)
        splits["evaluation_control"].extend(evaluation_rows)

    rng.shuffle(splits["selection_control"])
    rng.shuffle(splits["calibration_control"])
    rng.shuffle(splits["evaluation_control"])
    return splits, manifest


def build_report(
    family_ids: List[str],
    current_gap_rows: List[Mapping[str, Any]],
    final_gap_rows: List[Mapping[str, Any]],
    split_manifest_rows: List[Mapping[str, Any]],
) -> str:
    current_gap_map = {(row["family_id"], row["final_label"]): row for row in current_gap_rows}
    final_gap_map = {(row["family_id"], row["final_label"]): row for row in final_gap_rows}
    split_map = {row["family_id"]: row for row in split_manifest_rows}
    lines = [
        "# Family Dataset Release Report",
        "",
        "This report summarizes the current manual annotations, targeted supplementation, and final split readiness.",
        "",
    ]
    for family_id in family_ids:
        split_row = split_map.get(family_id, {})
        lines.append(f"## {family_id}")
        lines.append(
            f"- Split ready: {split_row.get('split_ready', 'no')}"
        )
        for final_label in ["target_positive", "hard_negative", "medium_negative", "background_negative"]:
            current_gap = current_gap_map[(family_id, final_label)]["gap_count"]
            final_gap = final_gap_map[(family_id, final_label)]["gap_count"]
            lines.append(f"- {final_label}: initial gap {current_gap}, final gap {final_gap}")
        if split_row:
            lines.append(
                f"- Evaluation target no-canonical-label count: {split_row.get('evaluation_target_no_canonical_label_count', 0)}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a finalized family-level benchmark release from current annotation batches and remaining manual-clean candidate pools."
    )
    parser.add_argument("--annotation_root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--candidate_root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--definitions_path", type=Path, default=DEFAULT_DEFINITIONS_PATH)
    parser.add_argument("--manual_rules_path", type=Path, default=DEFAULT_MANUAL_RULES_PATH)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.output_root)
    definitions_payload = load_yaml(args.definitions_path)
    selected_families = definitions_payload["selected_families"]
    family_ids = [family["family_id"] for family in selected_families]
    protocol = definitions_payload["benchmark_protocol"]
    required_counts, split_quota = required_counts_from_protocol(protocol)
    manual_rules = load_yaml(args.manual_rules_path)["families"]

    strict_patterns_by_family: Dict[str, List[re.Pattern[str]]] = {}
    strict_exclusions_by_family: Dict[str, List[re.Pattern[str]]] = {}
    for family_id in family_ids:
        strict_patterns = list(STRICT_TARGET_PATTERNS.get(family_id, []))
        if not strict_patterns:
            strict_patterns = list(manual_rules.get(family_id, {}).get("positive_keep_regex_any", []))
        strict_patterns_by_family[family_id] = compile_patterns(strict_patterns)
        strict_exclusions = list(STRICT_TARGET_EXCLUSIONS.get(family_id, []))
        strict_exclusions.extend(manual_rules.get(family_id, {}).get("positive_drop_regex_any", []))
        strict_exclusions_by_family[family_id] = compile_patterns(strict_exclusions)

    current_annotations = load_current_annotations(args.annotation_root)

    summary_dir = args.output_root / "summary"
    ensure_dir(summary_dir)
    raw_summary_rows = build_annotation_summary_rows(current_annotations, dedupe_kept=False)
    dedup_summary_rows = build_annotation_summary_rows(current_annotations, dedupe_kept=True)
    write_csv(
        summary_dir / "annotation_summary_long.csv",
        raw_summary_rows + dedup_summary_rows,
        ["family_id", "final_label", "keep_for_benchmark", "count", "summary_type"],
    )

    current_gap_rows = build_gap_rows(family_ids, current_annotations, required_counts)
    write_csv(
        summary_dir / "initial_gap_report.csv",
        current_gap_rows,
        ["family_id", "final_label", "available_count", "required_count", "gap_count"],
    )

    rng = random.Random(args.seed)
    supplement_dir = args.output_root / "supplement_batches"
    ensure_dir(supplement_dir)
    supplement_manifest_rows: List[Dict[str, Any]] = []
    supplement_rows_by_family: Dict[str, List[Dict[str, Any]]] = {}
    auto_pool_summary_counter: Counter[tuple[str, str, str]] = Counter()

    for family_id in family_ids:
        current_dedup_kept = [row for row in dedupe_rows(current_annotations[family_id]) if is_keep(row)]
        current_counts = Counter(row["final_label"] for row in current_dedup_kept)
        current_target_gap = max(0, required_counts["target_positive"] - current_counts.get("target_positive", 0))
        current_target_no_label = count_label_free_targets(current_dedup_kept, family_id)
        target_no_label_needed = max(0, int(math.ceil(split_quota["evaluation"]["target_positive"] * 0.25)) - current_target_no_label)
        supplement_target_quota = max(current_target_gap, target_no_label_needed)

        remaining_auto_rows = build_remaining_auto_pool(
            family_id=family_id,
            candidate_root=args.candidate_root,
            existing_rows=current_annotations[family_id],
            strict_patterns_by_family=strict_patterns_by_family,
            strict_exclusions_by_family=strict_exclusions_by_family,
        )
        for row in remaining_auto_rows:
            auto_pool_summary_counter[(family_id, row["final_label"], row["sampled_from_bucket"])] += 1

        selected_supplements: List[Dict[str, Any]] = []
        target_rows = select_target_supplements(
            family_id=family_id,
            auto_rows=remaining_auto_rows,
            current_target_rows=current_dedup_kept,
            target_quota=supplement_target_quota,
            min_label_free_needed=int(math.ceil(split_quota["evaluation"]["target_positive"] * 0.25)),
            rng=rng,
        )
        selected_supplements.extend(target_rows)
        selected_ids = {row["candidate_id"] for row in target_rows}
        remaining_after_target = [row for row in remaining_auto_rows if row["candidate_id"] not in selected_ids]

        for final_label in ["hard_negative", "medium_negative", "background_negative"]:
            gap = max(0, required_counts[final_label] - current_counts.get(final_label, 0))
            chosen = select_rows_by_label(
                family_id=family_id,
                auto_rows=remaining_after_target,
                final_label=final_label,
                quota=gap,
                rng=rng,
            )
            selected_supplements.extend(chosen)
            chosen_ids = {row["candidate_id"] for row in chosen}
            remaining_after_target = [row for row in remaining_after_target if row["candidate_id"] not in chosen_ids]

        supplement_rows_by_family[family_id] = selected_supplements
        write_canonical_rows_csv(supplement_dir / family_id / "supplement_annotated.csv", selected_supplements)
        write_jsonl(supplement_dir / family_id / "supplement_annotated.jsonl", selected_supplements)

        supplement_counts = Counter(row["final_label"] for row in selected_supplements if is_keep(row))
        supply_counts = Counter(row["final_label"] for row in remaining_auto_rows if is_keep(row))
        supplement_manifest_rows.append(
            {
                "family_id": family_id,
                "auto_supply_target_positive": supply_counts.get("target_positive", 0),
                "auto_supply_hard_negative": supply_counts.get("hard_negative", 0),
                "auto_supply_medium_negative": supply_counts.get("medium_negative", 0),
                "auto_supply_background_negative": supply_counts.get("background_negative", 0),
                "selected_target_positive": supplement_counts.get("target_positive", 0),
                "selected_hard_negative": supplement_counts.get("hard_negative", 0),
                "selected_medium_negative": supplement_counts.get("medium_negative", 0),
                "selected_background_negative": supplement_counts.get("background_negative", 0),
                "selected_drop_or_ambiguous": sum(
                    1 for row in selected_supplements if row["final_label"] in {"drop", "ambiguous"}
                ),
            }
        )

    auto_pool_summary_rows = [
        {
            "family_id": family_id,
            "final_label": final_label,
            "sampled_from_bucket": sampled_from_bucket,
            "count": count,
        }
        for (family_id, final_label, sampled_from_bucket), count in sorted(auto_pool_summary_counter.items())
    ]
    write_csv(
        summary_dir / "auto_pool_summary_long.csv",
        auto_pool_summary_rows,
        ["family_id", "final_label", "sampled_from_bucket", "count"],
    )
    write_csv(
        args.output_root / "supplement_manifest.csv",
        supplement_manifest_rows,
        [
            "family_id",
            "auto_supply_target_positive",
            "auto_supply_hard_negative",
            "auto_supply_medium_negative",
            "auto_supply_background_negative",
            "selected_target_positive",
            "selected_hard_negative",
            "selected_medium_negative",
            "selected_background_negative",
            "selected_drop_or_ambiguous",
        ],
    )

    frozen_dir = args.output_root / "frozen_pool"
    ensure_dir(frozen_dir)
    final_gap_rows: List[Dict[str, Any]] = []
    frozen_manifest_rows: List[Dict[str, Any]] = []
    frozen_rows_by_family: Dict[str, List[Dict[str, Any]]] = {}

    for family_id in family_ids:
        frozen_rows = build_frozen_rows(
            current_rows=current_annotations[family_id],
            supplement_rows=supplement_rows_by_family[family_id],
        )
        frozen_rows_by_family[family_id] = frozen_rows
        counts = Counter(row["final_label"] for row in frozen_rows)
        frozen_manifest_rows.append(
            {
                "family_id": family_id,
                "target_positive_count": counts.get("target_positive", 0),
                "hard_negative_count": counts.get("hard_negative", 0),
                "medium_negative_count": counts.get("medium_negative", 0),
                "background_negative_count": counts.get("background_negative", 0),
                "target_positive_no_canonical_label_count": count_label_free_targets(frozen_rows, family_id),
            }
        )
        for final_label in ["target_positive", "hard_negative", "medium_negative", "background_negative"]:
            label_rows = [row for row in frozen_rows if row["final_label"] == final_label]
            write_jsonl(frozen_dir / family_id / f"{final_label}.jsonl", label_rows)
            final_gap_rows.append(
                {
                    "family_id": family_id,
                    "final_label": final_label,
                    "available_count": len(label_rows),
                    "required_count": required_counts[final_label],
                    "gap_count": max(0, required_counts[final_label] - len(label_rows)),
                }
            )

    write_csv(
        args.output_root / "frozen_manifest.csv",
        frozen_manifest_rows,
        [
            "family_id",
            "target_positive_count",
            "hard_negative_count",
            "medium_negative_count",
            "background_negative_count",
            "target_positive_no_canonical_label_count",
        ],
    )
    write_csv(
        summary_dir / "final_gap_report.csv",
        final_gap_rows,
        ["family_id", "final_label", "available_count", "required_count", "gap_count"],
    )

    split_dir = args.output_root / "splits"
    ensure_dir(split_dir)
    split_manifest_rows: List[Dict[str, Any]] = []
    for family_id in family_ids:
        family_rng = random.Random(stable_family_seed(args.seed, family_id))
        splits, split_manifest = split_family_dataset(
            family_id=family_id,
            frozen_rows=frozen_rows_by_family[family_id],
            split_quota=split_quota,
            rng=family_rng,
        )
        split_manifest_rows.append(split_manifest)
        if splits:
            for split_name, rows in splits.items():
                write_jsonl(split_dir / family_id / f"{split_name}.jsonl", rows)
    write_csv(
        args.output_root / "split_manifest.csv",
        split_manifest_rows,
        [
            "family_id",
            "split_ready",
            "available_target_positive",
            "available_hard_negative",
            "available_medium_negative",
            "available_background_negative",
            "evaluation_target_no_canonical_label_count",
        ],
    )

    report_text = build_report(
        family_ids=family_ids,
        current_gap_rows=current_gap_rows,
        final_gap_rows=final_gap_rows,
        split_manifest_rows=split_manifest_rows,
    )
    (args.output_root / "release_report.md").write_text(report_text)


if __name__ == "__main__":
    main()
