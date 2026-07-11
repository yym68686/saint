#!/usr/bin/env python3
"""Apply the preregistered label-free hierarchy gate before Initial3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    effective = float(state["cascaded.hard_effective_parents"].item())
    max_share = float(state["cascaded.hard_max_share"].item())
    fixed_fraction = float(state["cascaded.wrong_fixed_fraction"].item())
    parent = state["cascaded.parent"].long()
    wrong = state["cascaded.wrong_parent"].long()
    n_high = int(state["cascaded.n_high"].item())
    counts = torch.bincount(parent, minlength=n_high)
    wrong_counts = torch.bincount(wrong, minlength=n_high)
    report = {
        "hard_effective_parents": effective,
        "hard_max_parent_share": max_share,
        "wrong_hierarchy_fixed_fraction": fixed_fraction,
        "wrong_hierarchy_count_preserving": bool(torch.equal(counts, wrong_counts)),
        "effective_parent_pass": effective >= 64.0,
        "max_parent_share_pass": max_share <= 0.20,
    }
    report["pass"] = all(
        report[key]
        for key in (
            "wrong_hierarchy_count_preserving",
            "effective_parent_pass",
            "max_parent_share_pass",
        )
    )
    report["decision"] = (
        "authorize-initial3" if report["pass"] else "stop-before-initial3"
    )
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(
        "\n".join(
            [
                "# Cascaded Concept SAE v2 Unsupervised Gate",
                "",
                f"- Effective hard parents: `{effective:.4f}` (required >= 64)",
                f"- Largest hard parent share: `{max_share:.4%}` (required <= 20%)",
                f"- Wrong-control fixed fraction: `{fixed_fraction:.4%}`",
                f"- Count preserving: `{report['wrong_hierarchy_count_preserving']}`",
                f"- Decision: `{report['decision']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
