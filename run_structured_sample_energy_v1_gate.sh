#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-structured-sample-energy-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-$MODEL_DIR/consolidated.00.pth}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
CHECKPOINT="${CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
ROOT="${ROOT:-/autodl-fs/data/structured_sample_energy_v1_gate_20260711}"
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
if branch != "scpg-structured-sample-energy-v1":
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
writable = [
    str(path)
    for path in cache_files
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
]
if writable:
    raise SystemExit(f"Structured cache contains writable files: {writable}")

payload = {
    "experiment": "structured sample-energy cross-readout frozen gate",
    "diagnostic_family": "sample energy",
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
        "reference": "mean_t z_t",
        "candidate": "sqrt(mean_t z_t^2 + 1e-8)",
        "architecture_compatibility_control": "mean_t z_t^2",
        "causal_control": "row-balanced wrong-sample RMS with zero same-class pairs and full row-multiset preservation",
        "same_checkpoint": True,
        "same_parameter_count": True,
        "same_exposed_feature_count": True,
        "true_sample_boundaries": True,
        "transform_sweep": False,
        "training_prohibited_before_gate_pass": True,
    },
    "readouts": {
        "official": "absolute train mean-diff selection, k=1/2/5, logistic regression",
        "scale_invariant": "standardized effect-size selection, k=1/2/5, logistic regression",
        "nonofficial_k": "absolute train mean-diff selection, k=3/10/20, logistic regression",
        "all_feature": "train-feature standardized full-feature ridge classifier alpha=1",
    },
    "evaluation": {
        "datasets": [
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "fancyzhx/ag_news",
        ],
        "train_size": int("$TRAIN_SIZE"),
        "test_size": int("$TEST_SIZE"),
        "random_seed": 42,
    },
    "gate": {
        "official_rms_minus_mean": ">= 0.005",
        "official_minimum_dataset_delta": ">= -0.01",
        "effect_size_rms_minus_mean": ">= 0.003",
        "effect_size_minimum_dataset_delta": ">= -0.01",
        "wide_k_rms_minus_mean": ">= 0",
        "full_ridge_rms_minus_mean": ">= 0",
        "rms_minus_wrong_sample": ">= 0.05",
    },
    "leakage": {
        "uses_saebench_labels_to_define_representation": False,
        "uses_class_names_to_define_representation": False,
        "uses_eval_split_to_define_representation": False,
        "uses_mean_diff_to_define_representation": False,
        "uses_test_feedback_to_select_transform": False,
        "labels_used_only_inside_registered_readouts": True,
        "wrong_control_uses_class_partition_only_to_forbid_same_class_pairs": True,
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

"$PY" "$CODE/diagnose_structured_sample_energy.py" \
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
  --sae-seq-batch-size 2 \
  --random-seed 42 \
  --dtype bfloat16 \
  --device cuda \
  > "$ROOT/diagnostic.log" 2>&1

"$PY" - <<PY | tee "$ROOT/decision.log"
import json
from pathlib import Path

payload = json.loads(Path("$ROOT/sample-energy-gate.json").read_text())
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
