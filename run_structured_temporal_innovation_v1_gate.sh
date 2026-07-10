#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-structured-temporal-innovation-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
CHECKPOINT="${CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
ROOT="${ROOT:-/autodl-fs/data/structured_temporal_innovation_v1_gate_20260711}"
PAIR_COUNT="${PAIR_COUNT:-16384}"
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
branch = subprocess.check_output(
    ["git", "-C", str(code), "branch", "--show-current"], text=True
).strip()
if branch != "scpg-structured-temporal-innovation-v1":
    raise SystemExit(f"Unexpected branch: {branch}")
if cache.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
    raise SystemExit("Structured cache is writable")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "experiment": "structured true-sequence temporal-innovation frozen gate",
    "registered_before_diagnostic": True,
    "code_branch": branch,
    "code_commit": subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip(),
    "cache_dir": str(cache),
    "cache_read_only": True,
    "cache_manifest_sha256": sha256(cache / "manifest.json"),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "signal": {
        "layer": 22,
        "pair_count": int("$PAIR_COUNT"),
        "max_pairs_per_sample": 2,
        "position_seed": 46126,
        "pairs_never_cross_sample_boundary": True,
        "target": "normalize(x[t+1])-normalize(x[t]) within the same true OWT sample",
        "score": "diagonally whitened multivariate correlation norm between z[t] and target innovation",
        "weight": "rank map to [0.5,1.5] with mean 1.0",
        "evaluation_uses_only_standard_layer22_features": True,
    },
    "controls": {
        "raw_l22_relu": True,
        "feature_permuted_same_weight_distribution": True,
        "complete_innovation_cyclic_shift": True,
        "wrong_pairs_have_zero_fixed_tokens": True,
        "wrong_pairs_have_zero_same_sample_pairs": True,
        "permutation_seed": 46026,
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
        "candidate_minus_permuted": ">= 0.002",
        "candidate_minus_wrong_alignment": ">= 0.002",
        "training_prohibited_before_pass": True,
    },
    "leakage": {
        "uses_saebench_labels_to_construct_signal": False,
        "uses_class_names_to_construct_signal": False,
        "uses_eval_split_to_construct_signal": False,
        "uses_mean_diff_to_construct_signal": False,
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

"$PY" "$CODE/diagnose_structured_temporal_innovation.py" \
  --checkpoint "$CHECKPOINT" \
  --cache-dir "$CACHE_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$ROOT" \
  --pair-count "$PAIR_COUNT" \
  --max-pairs-per-sample 2 \
  --position-seed 46126 \
  --batch-pairs 128 \
  --score-feature-block 2048 \
  --permutation-seed 46026 \
  --datasets \
    LabHC/bias_in_bios_class_set3 \
    canrager/amazon_reviews_mcauley_1and5 \
    fancyzhx/ag_news \
  --train-size "$TRAIN_SIZE" \
  --test-size "$TEST_SIZE" \
  --context-length 128 \
  --llm-batch-size 4 \
  --sae-seq-batch-size 2 \
  --k-values 1 2 5 \
  --random-seed 42 \
  --dtype bfloat16 \
  --device cuda \
  > "$ROOT/diagnostic.log" 2>&1

"$PY" - <<PY | tee "$ROOT/decision.log"
import json
from pathlib import Path
payload = json.loads(
    Path("$ROOT/structured-temporal-innovation-gate.json").read_text()
)
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
