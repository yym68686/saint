#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-structured-sample-idf-centroid-v2}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-$MODEL_DIR/consolidated.00.pth}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
CHECKPOINT="${CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
ROOT="${ROOT:-/autodl-fs/data/structured_sample_idf_centroid_v2_gate_20260711}"
SAMPLE_COUNT="${SAMPLE_COUNT:-8192}"
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
if branch != "scpg-structured-sample-idf-centroid-v2":
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
expected_model_size = int(manifest["sources"]["model_weight_size_bytes"])
if model_weights.stat().st_size != expected_model_size:
    raise SystemExit("Model weight size differs from the captured cache source")
model_weights_sha256 = sha256(model_weights)
if model_weights_sha256 != manifest["sources"]["model_weight_sha256"]:
    raise SystemExit("Model weight hash differs from the captured cache source")
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
    "experiment": "structured true-sample IDF lexical-centroid frozen gate",
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
    "model_weights_sha256_from_cache_manifest": manifest["sources"]["model_weight_sha256"],
    "signal": {
        "layer": 22,
        "sample_count": int("$SAMPLE_COUNT"),
        "min_sample_tokens": 4,
        "sample_batch_size": 32,
        "token_batch_size": 128,
        "sample_boundaries_from_manifest": True,
        "document_frequency_corpus": "all 50000 unlabeled OWT samples in the immutable structured cache",
        "smooth_idf": "log((N+1)/(df+1))+1",
        "target": "L2-normalized smooth-IDF-weighted mean of per-token L2-normalized frozen output.weight rows within the same real OWT sample",
        "score": "diagonally whitened multivariate correlation norm between sample-mean SAE activation and the IDF lexical centroid",
        "weight": "rank map to [0.5,1.5] with mean 1.0",
        "target_variant_sweep": False,
        "evaluation_uses_only_standard_layer22_features": True,
    },
    "pre_gate_unlabeled_evidence": {
        "uses_saebench_data": False,
        "owt_document_count": 50000,
        "tokens_in_at_least_10pct_documents_token_mass": 0.2716706,
        "tokens_in_at_least_10pct_documents_idf_weighted_mass": 0.0987184,
        "split_view_sample_count": 4096,
        "uniform_same_minus_wrong_cosine": 0.0464753,
        "idf_same_minus_wrong_cosine": 0.0818911,
    },
    "controls": {
        "raw_l22_relu": True,
        "uniform_lexical_centroid": True,
        "token_idf_cyclic_permutation_same_global_weight_distribution": True,
        "feature_permuted_same_rank_weight_distribution": True,
        "complete_sample_target_cyclic_shift": True,
        "wrong_pairs_have_zero_fixed_samples": True,
        "wrong_pairs_have_zero_same_sample_pairs": True,
        "permutation_seed": 47026,
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
        "candidate_minus_uniform_centroid": ">= 0.002",
        "candidate_minus_token_idf_permuted": ">= 0.002",
        "candidate_minus_feature_permuted": ">= 0.002",
        "candidate_minus_wrong_alignment": ">= 0.002",
        "training_prohibited_before_pass": True,
    },
    "leakage": {
        "uses_saebench_labels_to_construct_signal": False,
        "uses_class_names_to_construct_signal": False,
        "uses_eval_split_to_construct_signal": False,
        "uses_mean_diff_to_construct_signal": False,
        "uses_test_feedback_to_choose_idf_formula": False,
        "uses_only_owt_sample_boundaries_and_token_ids": True,
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

"$PY" "$CODE/diagnose_structured_sample_idf_centroid.py" \
  --checkpoint "$CHECKPOINT" \
  --cache-dir "$CACHE_DIR" \
  --model-weights "$MODEL_WEIGHTS" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$ROOT" \
  --sample-count "$SAMPLE_COUNT" \
  --min-sample-tokens 4 \
  --sample-batch-size 32 \
  --token-batch-size 128 \
  --cross-batch-samples 32 \
  --score-feature-block 2048 \
  --permutation-seed 47026 \
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
    Path("$ROOT/sample-idf-centroid-gate.json").read_text()
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
