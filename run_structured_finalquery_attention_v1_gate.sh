#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-structured-finalquery-attention-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-$MODEL_DIR/consolidated.00.pth}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
CHECKPOINT="${CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
ROOT="${ROOT:-/autodl-fs/data/structured_finalquery_attention_v1_gate_20260711}"
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
if branch != "scpg-structured-finalquery-attention-v1":
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
    "experiment": "structured L22 final-query attention pooling frozen gate",
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
    "model_weights_size": model_weights.stat().st_size,
    "model_weights_sha256": model_weights_sha256,
    "mechanism": {
        "layer": 22,
        "reference": "mean_t ReLU(W_enc(normalize(x_t)-b_pre)+b_enc)",
        "candidate": "sum_t mean_heads softmax(q_last k_t / sqrt(d_h)) * ReLU(W_enc(normalize(x_t)-b_pre)+b_enc)",
        "control": "cyclically shift the true attention vector within every sample before pooling",
        "attention_query": "last valid token",
        "uses_actual_layer22_wq_wk": True,
        "uses_rope": True,
        "includes_self_attention": True,
        "head_aggregation": "mean after per-head softmax",
        "true_sample_boundaries": True,
        "same_checkpoint": True,
        "same_parameter_count": True,
        "same_exposed_feature_count": True,
        "pooling_rule_sweep": False,
        "attention_temperature_sweep": False,
        "native_attention_reconstruction_required": True,
        "native_attention_reconstruction_max_abs_error": "<= 0.01",
        "native_attention_reconstruction_min_cosine": ">= 0.9999",
        "training_prohibited_before_gate_pass": True,
    },
    "evidence": {
        "external_primary_source": "arXiv:2607.08605v1 S2AE",
        "external_mechanism": "Transformer attention similarity as a label-free structural prior for SAE sparsity",
        "prior_attention_projection_used_true_attention_maps": False,
        "prior_position_pooling_used_attention_weights": False,
        "prior_saliency_pooling_used_attention_weights": False,
        "prior_learned_probe_pooling_used_frozen_llm_attention": False,
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
        "candidate_minus_shifted_attention": ">= 0.002",
    },
    "leakage": {
        "uses_saebench_labels_to_define_transform": False,
        "uses_class_names_to_define_transform": False,
        "uses_eval_split_to_define_transform": False,
        "uses_mean_diff_to_define_transform": False,
        "uses_test_feedback_to_select_pooling_rule": False,
        "uses_attention_map_from_same_frozen_llama": True,
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

"$PY" "$CODE/diagnose_structured_finalquery_attention.py" \
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
  --k-values 1 2 5 \
  --random-seed 42 \
  --dtype bfloat16 \
  --device cuda \
  > "$ROOT/diagnostic.log" 2>&1

"$PY" - <<PY | tee "$ROOT/decision.log"
import json
from pathlib import Path
payload = json.loads(Path("$ROOT/finalquery-attention-gate.json").read_text())
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
