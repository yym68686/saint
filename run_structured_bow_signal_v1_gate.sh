#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-structured-bow-signal-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-$MODEL_DIR/consolidated.00.pth}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
CHECKPOINT="${CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
ROOT="${ROOT:-/autodl-fs/data/structured_bow_signal_v1_gate_20260711}"
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
if branch != "scpg-structured-bow-signal-v1":
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
if model_weights.stat().st_size != int(
    manifest["sources"]["model_weight_size_bytes"]
):
    raise SystemExit("Model weight size differs from cache source")
model_weights_sha256 = sha256(model_weights)
if model_weights_sha256 != manifest["sources"]["model_weight_sha256"]:
    raise SystemExit("Model weight hash differs from cache source")
cache_files = [cache / "manifest.json"]
for shard in manifest["shards"]:
    cache_files.append(cache / shard["meta"]["path"])
    cache_files.extend(
        cache / layer_entry["path"]
        for layer_entry in shard["layers"].values()
    )
writable_cache_files = [
    str(path)
    for path in cache_files
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
]
if writable_cache_files:
    raise SystemExit(f"Structured cache contains writable files: {writable_cache_files}")

payload = {
    "experiment": "structured true-sample binary bag-of-token frozen signal gate",
    "diagnostic_family_version": "1/3",
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
    "model_weights_sha256": model_weights_sha256,
    "mechanism": {
        "vocabulary": "all token IDs observed in the complete 50,000-sample OWT cache",
        "maximum_exposed_features": 65536,
        "candidate": "binary token-presence vector per true text sample",
        "wrong_control": "row-balanced cyclic derangement over all other classes",
        "wrong_control_same_class_pairs": 0,
        "uses_idf_or_frequency_weighting": False,
        "uses_nmf_svd_hash_cluster_stem_or_topn": False,
        "vocabulary_size_or_weight_sweep": False,
        "training_prohibited_before_gate_pass": True,
    },
    "evaluation": {
        "datasets": [
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "fancyzhx/ag_news",
        ],
        "train_size": int("$TRAIN_SIZE"),
        "test_size": int("$TEST_SIZE"),
        "context_length": 128,
        "k_values": [1, 2, 5],
        "random_seed": 42,
    },
    "gate": {
        "feature_count": "<= 65536",
        "candidate_minus_relu_mean_acc": ">= 0.005",
        "minimum_per_dataset_delta": ">= -0.01",
        "candidate_minus_wrong_control": ">= 0.05",
    },
    "leakage": {
        "uses_saebench_labels_to_define_vocabulary_or_features": False,
        "uses_class_names_to_define_vocabulary_or_features": False,
        "uses_eval_split_to_define_vocabulary_or_features": False,
        "uses_mean_diff_to_define_vocabulary_or_features": False,
        "uses_test_feedback_to_select_token_subset_or_weights": False,
        "candidate_uses_only_current_sample_token_ids": True,
        "modifies_checkpoint": False,
        "modifies_cache": False,
        "modifies_model_weights": False,
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

"$PY" "$CODE/diagnose_structured_bow_signal.py" \
  --checkpoint "$CHECKPOINT" \
  --cache-dir "$CACHE_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$ROOT" \
  --maximum-features 65536 \
  --datasets \
    LabHC/bias_in_bios_class_set3 \
    canrager/amazon_reviews_mcauley_1and5 \
    fancyzhx/ag_news \
  --train-size "$TRAIN_SIZE" \
  --test-size "$TEST_SIZE" \
  --context-length 128 \
  --llm-batch-size 4 \
  --sae-seq-batch-size 2 \
  --bow-batch-size 64 \
  --k-values 1 2 5 \
  --random-seed 42 \
  --dtype bfloat16 \
  --device cuda \
  > "$ROOT/diagnostic.log" 2>&1

"$PY" - <<PY | tee "$ROOT/decision.log"
import json
from pathlib import Path

payload = json.loads(Path("$ROOT/structured-bow-signal-gate.json").read_text())
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
