#!/usr/bin/env python3
"""Apply the v3 label-free transport gate before Initial3."""

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
    active_parents = int((state["cascaded.cluster_count"] > 0).sum().item())
    transport_column_cv = float(state["cascaded.transport_column_cv"].item())
    learned_coherence = float(
        state["cascaded.learned_partition_coherence"].item()
    )
    wrong_coherence = float(
        state["cascaded.wrong_partition_coherence"].item()
    )
    coherence_delta = learned_coherence - wrong_coherence
    fixed_fraction = float(state["cascaded.wrong_fixed_fraction"].item())
    parent = state["cascaded.parent"].long()
    wrong = state["cascaded.wrong_parent"].long()
    n_high = int(state["cascaded.n_high"].item())
    counts = torch.bincount(parent, minlength=n_high)
    wrong_counts = torch.bincount(wrong, minlength=n_high)
    report = {
        "hard_effective_parents": effective,
        "active_hard_parents": active_parents,
        "hard_max_parent_share": max_share,
        "transport_column_cv": transport_column_cv,
        "learned_partition_coherence": learned_coherence,
        "wrong_partition_coherence": wrong_coherence,
        "partition_coherence_delta": coherence_delta,
        "wrong_hierarchy_fixed_fraction": fixed_fraction,
        "wrong_hierarchy_count_preserving": bool(torch.equal(counts, wrong_counts)),
        "active_parent_pass": active_parents >= 2048,
        "effective_parent_pass": effective >= 256.0,
        "max_parent_share_pass": max_share <= 0.10,
        "transport_column_pass": transport_column_cv <= 0.05,
        "coherence_pass": coherence_delta >= 0.05,
    }
    report["pass"] = all(
        report[key]
        for key in (
            "wrong_hierarchy_count_preserving",
            "active_parent_pass",
            "effective_parent_pass",
            "max_parent_share_pass",
            "transport_column_pass",
            "coherence_pass",
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
                "# Cascaded Concept SAE v3 Unsupervised Transport Gate",
                "",
                f"- Active hard parents: `{active_parents}` (required >= 2048)",
                f"- Effective hard parents: `{effective:.4f}` (required >= 256)",
                f"- Largest hard parent share: `{max_share:.4%}` (required <= 10%)",
                f"- Transport column CV: `{transport_column_cv:.6f}` (required <= 0.05)",
                f"- Learned-minus-wrong coherence: `{coherence_delta:+.6f}` (required >= 0.05)",
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
