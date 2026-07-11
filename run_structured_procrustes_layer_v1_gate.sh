#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-structured-procrustes-layer-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-$MODEL_DIR/consolidated.00.pth}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
CHECKPOINT="${CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
ROOT="${ROOT:-/autodl-fs/data/structured_procrustes_layer_v1_gate_20260711}"
FIT_TOKENS="${FIT_TOKENS:-8192}"
HOLDOUT_TOKENS="${HOLDOUT_TOKENS:-8192}"
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
if branch != "scpg-structured-procrustes-layer-v1":
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
cache_files.extend(
    cache / entry["path"] for entry in manifest["layer_means"].values()
)
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
    "experiment": "structured cross-layer orthogonal Procrustes frozen gate",
    "diagnostic_family_version": "1/3",
    "registered_before_diagnostic": True,
    "code_branch": branch,
    "code_commit": subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip(),
    "external_primary_source": "arXiv:2607.08499v1",
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
        "source_layers": [20, 21, 23],
        "reference_layer": 22,
        "fit_tokens": int("$FIT_TOKENS"),
        "holdout_tokens": int("$HOLDOUT_TOKENS"),
        "centering": "fixed full-cache per-layer means from manifest",
        "fit": "Q_l = argmin ||(H_l-mu_l)Q-(H_22-mu_22)||_F subject to Q^TQ=I",
        "candidate_readout": "encode(mean_l((H_l-mu_l)Q_l+mu_22))",
        "identity_control": "encode(mean_l(H_l))",
        "wrong_control": "same fit after sample-separating cyclic permutation of H_22 targets",
        "alignment_or_layer_weight_sweep": False,
        "same_checkpoint": True,
        "same_parameter_count": True,
        "same_exposed_feature_count": True,
        "training_prohibited_before_both_gates_pass": True,
    },
    "owt_precheck": {
        "mean_true_minus_identity_cosine": ">= 0.01",
        "minimum_layer_true_minus_identity_cosine": ">= 0.005",
        "mean_true_minus_wrong_cosine": ">= 0.01",
        "benchmark_prohibited_before_pass": True,
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
    "benchmark_gate": {
        "candidate_minus_reference_mean_acc": ">= 0.005",
        "minimum_per_dataset_delta": ">= -0.01",
        "candidate_minus_identity_average": ">= 0.002",
        "candidate_minus_wrong_procrustes": ">= 0.002",
    },
    "leakage": {
        "uses_saebench_labels_to_define_alignment": False,
        "uses_class_names_to_define_alignment": False,
        "uses_eval_split_to_define_alignment": False,
        "uses_mean_diff_to_define_alignment": False,
        "uses_test_feedback_to_select_layers_or_weights": False,
        "uses_only_read_only_owt_aligned_layers": True,
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

"$PY" "$CODE/diagnose_structured_procrustes_layer_gate.py" \
  --checkpoint "$CHECKPOINT" \
  --cache-dir "$CACHE_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$ROOT" \
  --fit-tokens "$FIT_TOKENS" \
  --holdout-tokens "$HOLDOUT_TOKENS" \
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
  --dtype bfloat16 \
  --device cuda \
  > "$ROOT/diagnostic.log" 2>&1

"$PY" - <<PY | tee "$ROOT/decision.log"
import json
from pathlib import Path

payload = json.loads(Path("$ROOT/procrustes-layer-gate.json").read_text())
benchmark_gate = payload.get("benchmark_gate", {"pass": False})
report = {
    "owt_precheck_pass": payload["alignment_report"]["precheck"]["pass"],
    "benchmark_evaluation_ran": payload["benchmark_evaluation_ran"],
    "benchmark_gate_pass": benchmark_gate["pass"],
    "decision": payload["decision"],
    "training_ran": False,
}
Path("$ROOT/decision.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
