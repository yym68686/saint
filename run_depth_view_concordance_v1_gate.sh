#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-depth-view-concordance-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
CHECKPOINT="${CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
PERSISTENCE_WEIGHTS="${PERSISTENCE_WEIGHTS:-/autodl-fs/data/cross_layer_persistence_v1_gate_20260711/persistence-weights.pt}"
SPLITVIEW_WEIGHTS="${SPLITVIEW_WEIGHTS:-/autodl-fs/data/structured_splitview_reliability_v1_gate_20260711/splitview-reliability-weights.pt}"
PERSISTENCE_GATE="${PERSISTENCE_GATE:-/autodl-fs/data/cross_layer_persistence_v1_gate_20260711/cross-layer-persistence-gate.json}"
SPLITVIEW_GATE="${SPLITVIEW_GATE:-/autodl-fs/data/structured_splitview_reliability_v1_gate_20260711/structured-splitview-reliability-gate.json}"
ROOT="${ROOT:-/autodl-fs/data/depth_view_concordance_v1_gate_20260711}"
TRAIN_SIZE="${TRAIN_SIZE:-512}"
TEST_SIZE="${TEST_SIZE:-128}"

source /etc/network_turbo >/dev/null 2>&1 || true
mkdir -p "$ROOT"

"$PY" - <<PY | tee "$ROOT/preregistration.log"
import hashlib
import json
import subprocess
from pathlib import Path

root = Path("$ROOT")
code = Path("$CODE")
checkpoint = Path("$CHECKPOINT")
persistence_weights = Path("$PERSISTENCE_WEIGHTS")
splitview_weights = Path("$SPLITVIEW_WEIGHTS")
persistence_gate = Path("$PERSISTENCE_GATE")
splitview_gate = Path("$SPLITVIEW_GATE")
branch = subprocess.check_output(
    ["git", "-C", str(code), "branch", "--show-current"], text=True
).strip()
if branch != "scpg-depth-view-concordance-v1":
    raise SystemExit(f"Unexpected branch: {branch}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

source_persistence = json.loads(persistence_gate.read_text(encoding="utf-8"))
source_splitview = json.loads(splitview_gate.read_text(encoding="utf-8"))
checkpoint_hash = sha256(checkpoint)
if source_persistence["config"]["checkpoint_sha256"] != checkpoint_hash:
    raise SystemExit("Persistence source used a different checkpoint")
if source_splitview["config"]["checkpoint_sha256"] != checkpoint_hash:
    raise SystemExit("Split-view source used a different checkpoint")

payload = {
    "experiment": "equal-weight depth-view concordance frozen signal gate",
    "registered_before_diagnostic": True,
    "code_branch": branch,
    "code_commit": subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip(),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": checkpoint_hash,
    "sources": {
        "cross_layer_persistence_weights": str(persistence_weights),
        "cross_layer_persistence_weights_sha256": sha256(persistence_weights),
        "cross_layer_persistence_gate": str(persistence_gate),
        "cross_layer_persistence_gate_sha256": sha256(persistence_gate),
        "true_sample_splitview_weights": str(splitview_weights),
        "true_sample_splitview_weights_sha256": sha256(splitview_weights),
        "true_sample_splitview_gate": str(splitview_gate),
        "true_sample_splitview_gate_sha256": sha256(splitview_gate),
    },
    "signal": {
        "definition": "rank of the fixed 0.5/0.5 arithmetic mean of persistence and split-view reliability ranks",
        "weight_sweep": False,
        "concordance_seed": 45026,
        "evaluation_uses_only_standard_layer22_features": True,
    },
    "controls": {
        "raw_l22_relu": True,
        "final_weight_permutation": True,
        "splitview_signal_feature_mismatch": True,
        "dual_wrong_signal_concordance": True,
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
        "candidate_minus_each_control": ">= 0.002",
        "training_prohibited_before_pass": True,
    },
    "leakage": {
        "uses_saebench_labels_to_construct_signal": False,
        "uses_class_names_to_construct_signal": False,
        "uses_eval_split_to_construct_signal": False,
        "uses_mean_diff_to_construct_signal": False,
        "modifies_checkpoint": False,
        "modifies_source_weights": False,
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

"$PY" "$CODE/diagnose_depth_view_concordance.py" \
  --checkpoint "$CHECKPOINT" \
  --persistence-weights "$PERSISTENCE_WEIGHTS" \
  --splitview-weights "$SPLITVIEW_WEIGHTS" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$ROOT" \
  --concordance-seed 45026 \
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
payload = json.loads(Path("$ROOT/depth-view-concordance-gate.json").read_text())
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
