#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-structured-attention-groups-v2}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-$MODEL_DIR/consolidated.00.pth}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
CHECKPOINT="${CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
ROOT="${ROOT:-/autodl-fs/data/structured_attention_groups_v2_gate_20260711}"
TRAIN_SIZE="${TRAIN_SIZE:-512}"
TEST_SIZE="${TEST_SIZE:-128}"

source /etc/network_turbo >/dev/null 2>&1 || true
mkdir -p "$ROOT"

"$PY" - <<PY | tee "$ROOT/preregistration.log"
import hashlib
import json
import stat
import subprocess
from pathlib import Path

root = Path("$ROOT")
code = Path("$CODE")
cache = Path("$CACHE_DIR")
checkpoint = Path("$CHECKPOINT")
model_weights = Path("$MODEL_WEIGHTS")
branch = subprocess.check_output(
    ["git", "-C", str(code), "branch", "--show-current"], text=True
).strip()
if branch != "scpg-structured-attention-groups-v2":
    raise SystemExit(f"Unexpected branch: {branch}")
if cache.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
    raise SystemExit("Structured cache is writable")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
if model_weights.stat().st_size != int(manifest["sources"]["model_weight_size_bytes"]):
    raise SystemExit("Model weight size differs from cache source")
model_weights_sha256 = sha256(model_weights)
if model_weights_sha256 != manifest["sources"]["model_weight_sha256"]:
    raise SystemExit("Model weight hash differs from cache source")
cache_files = [cache / "manifest.json"]
for shard in manifest["shards"]:
    cache_files.append(cache / shard["meta"]["path"])
    cache_files.extend(
        cache / layer_entry["path"] for layer_entry in shard["layers"].values()
    )
writable_cache_files = [
    str(path)
    for path in cache_files
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
]
if writable_cache_files:
    raise SystemExit(f"Structured cache contains writable files: {writable_cache_files}")

payload = {
    "experiment": "structured full-attention sequential token-group frozen gate",
    "diagnostic_family": "attention structure",
    "diagnostic_family_version": "2/3",
    "registered_before_diagnostic": True,
    "code_branch": branch,
    "code_commit": subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip(),
    "cache_dir": str(cache),
    "cache_read_only": True,
    "cache_files_checked": len(cache_files),
    "cache_manifest_sha256": sha256(cache / "manifest.json"),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "model_weights": str(model_weights),
    "model_weights_size": model_weights.stat().st_size,
    "model_weights_sha256": model_weights_sha256,
    "mechanism": {
        "layers": [20, 21, 22, 23],
        "group_count": 20,
        "attention_aggregation": "mean native causal attention over heads and L20-L23, then symmetrize",
        "sequential_prior": "normalized absolute token distance raised to 0.02",
        "clustering": "average-link agglomerative clustering on precomputed distance",
        "group_profile": "RMS of frozen ReLU features within each group",
        "sample_feature": "coordinatewise maximum group profile",
        "reference": "mean-pooled frozen ReLU",
        "matched_controls": [
            "global token RMS",
            "equal-width contiguous 20-group RMS",
            "maximally shifted true labels preserving exact group sizes",
        ],
        "true_sample_boundaries": True,
        "same_checkpoint": True,
        "same_parameter_count": True,
        "same_exposed_feature_count": True,
        "group_count_sweep": False,
        "layer_sweep": False,
        "positional_exponent_sweep": False,
        "pooling_formula_sweep": False,
        "training_prohibited_before_gate_pass": True,
    },
    "evidence": {
        "primary_source": "arXiv:2607.08605v1",
        "official_repository": "https://github.com/liaoweiduo/s2ae",
        "official_repository_commit_reviewed": "a26bf49e0adeb44c66349cb0075c53e26d5c5ff0",
        "adaptation_boundary": "replace 2D image-patch Manhattan distance with 1D token distance; retain all-layer attention, 20 groups, exponent 0.02, group RMS",
        "prior_final_query_attention_is_same_mechanism": False,
        "current_feature_width": 65536,
    },
    "evaluation": {
        "datasets": [
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "fancyzhx/ag_news",
        ],
        "train_size": int("$TRAIN_SIZE"),
        "test_size": int("$TEST_SIZE"),
        "k_values": [1, 2, 5],
        "random_seed": 42,
    },
    "gate": {
        "candidate_minus_reference_mean_acc": ">= 0.005",
        "minimum_per_dataset_delta": ">= -0.01",
        "candidate_minus_best_matched_control": ">= 0.003",
    },
    "leakage": {
        "uses_saebench_labels_to_define_transform": False,
        "uses_class_names_to_define_transform": False,
        "uses_eval_split_to_define_transform": False,
        "uses_mean_diff_to_define_transform": False,
        "uses_test_feedback_to_select_hyperparameters": False,
        "uses_attention_maps_from_same_frozen_llama": True,
        "modifies_checkpoint": False,
        "modifies_cache": False,
    },
}
path = root / "preregistration.json"
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit(f"Refusing to overwrite preregistration: {path}")
else:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

"$PY" "$CODE/diagnose_structured_attention_groups.py" \
  --checkpoint "$CHECKPOINT" \
  --cache-dir "$CACHE_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$ROOT" \
  --datasets \
    LabHC/bias_in_bios_class_set3 \
    canrager/amazon_reviews_mcauley_1and5 \
    fancyzhx/ag_news \
  --train-size "$TRAIN_SIZE" \
  --test-size "$TEST_SIZE" \
  --context-length 128 \
  --llm-batch-size 4 \
  --sae-seq-batch-size 1 \
  --k-values 1 2 5 \
  --random-seed 42 \
  --group-count 20 \
  --positional-exponent 0.02 \
  --dtype bfloat16 \
  --device cuda \
  > "$ROOT/diagnostic.log" 2>&1

"$PY" - <<PY | tee "$ROOT/decision.log"
import json
from pathlib import Path

payload = json.loads(Path("$ROOT/attention-group-gate.json").read_text())
report = {
    "gate_pass": payload["gate"]["pass"],
    "decision": payload["decision"],
    "training_ran": False,
}
Path("$ROOT/decision.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
