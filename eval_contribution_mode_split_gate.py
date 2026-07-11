#!/usr/bin/env python3
"""Patch the standard sparse-probing evaluator for the frozen split variants."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from contribution_mode_split import (
    apply_pair_transform,
    split_route_probability,
    v396_from_centered,
)


CUSTOM_KINDS = {
    "v396_reference",
    "cms_mass_fold",
    "cms_wrong",
    "cms_candidate",
}
TRAINABLE_KEYS = (
    "b_pre",
    "encoder.weight",
    "encoder.bias",
    "decoder.weight",
    "v396.raw_beta",
    "v396.log_gain",
)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_evaluator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("standard_sparse_probe", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-evaluator", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    base = load_base_evaluator(args.base_evaluator)
    original_load = base.load_sae_state
    original_encode = base.encode_features_for_tokens

    def patched_load_sae_state(
        target: dict[str, Any], config: Any
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        kind = str(target["kind"])
        if kind not in CUSTOM_KINDS:
            return original_load(target, config)
        checkpoint = Path(target["checkpoint"])
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        required = set(TRAINABLE_KEYS) | {
            "v396.max_beta",
            "v396.max_log_gain",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise KeyError(f"V396 checkpoint is missing {missing}")
        keys = (
            "b_pre",
            "encoder.weight",
            "encoder.bias",
            "v396.raw_beta",
            "v396.log_gain",
            "v396.max_beta",
            "v396.max_log_gain",
        )
        state = {
            key: raw[key].to(device=config.device, dtype=config.dtype)
            if raw[key].ndim > 0
            else raw[key].to(device=config.device)
            for key in keys
        }
        extra: dict[str, Any] = {
            "n_latents": int(raw["encoder.weight"].shape[0]),
            "parameter_count": int(sum(raw[key].numel() for key in TRAINABLE_KEYS)),
            "checkpoint_sha256": file_sha256(checkpoint),
        }
        if kind != "v396_reference":
            split_path = Path(target["split_spec"])
            split = torch.load(split_path, map_location="cpu", weights_only=True)
            extra.update(
                {
                    "parent_indices": split["parent_indices"].to(
                        device=config.device, dtype=torch.int64
                    ),
                    "recipient_indices": split["recipient_indices"].to(
                        device=config.device, dtype=torch.int64
                    ),
                    "candidate_allocation": split["candidate_allocation"].to(
                        device=config.device, dtype=config.dtype
                    ),
                    "wrong_allocation": split["wrong_allocation"].to(
                        device=config.device, dtype=config.dtype
                    ),
                    "split_spec_sha256": file_sha256(split_path),
                }
            )
        del raw
        return state, extra

    def patched_encode_features_for_tokens(
        x_flat: torch.Tensor,
        target: dict[str, Any],
        state: dict[str, torch.Tensor],
        extra: dict[str, Any],
        config: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        kind = str(target["kind"])
        if kind not in CUSTOM_KINDS:
            return original_encode(x_flat, target, state, extra, config)
        x_norm = base.normalize_activation(
            x_flat, config.dtype, config.normalize_eps
        )
        x_centered = x_norm - state["b_pre"]
        h, features = v396_from_centered(x_centered, state)
        if kind == "v396_reference":
            return None, None, features
        parents = extra["parent_indices"]
        recipients = extra["recipient_indices"]
        if kind == "cms_mass_fold":
            transformed = apply_pair_transform(
                features, parents, recipients, "fold"
            )
            return None, None, transformed
        allocation_key = (
            "candidate_allocation"
            if kind == "cms_candidate"
            else "wrong_allocation"
        )
        probability = split_route_probability(
            x_centered,
            h.index_select(1, parents),
            state["encoder.weight"].index_select(0, parents),
            state["encoder.bias"].index_select(0, parents),
            extra[allocation_key],
        )
        mode = "candidate" if kind == "cms_candidate" else "wrong"
        transformed = apply_pair_transform(
            features,
            parents,
            recipients,
            mode,
            probability,
        )
        return None, None, transformed

    base.load_sae_state = patched_load_sae_state
    base.encode_features_for_tokens = patched_encode_features_for_tokens
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
