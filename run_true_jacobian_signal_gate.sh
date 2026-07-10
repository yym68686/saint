#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/root/saint-true-jacobian-workspace-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
ROOT="${ROOT:-/autodl-fs/data/true_jacobian_workspace_v1_20260710}"
JACOBIAN_DIR="$ROOT/jacobian_n10"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
mkdir -p "$ROOT"

"$PY" - <<PY | tee "$ROOT/preregistration.log"
import json
import subprocess
from pathlib import Path

payload = {
    "experiment": "true averaged downstream Jacobian workspace signal gate",
    "status": "registered-before-full-jacobian-and-evaluation",
    "code_branch": subprocess.check_output(
        ["git", "-C", "$CODE", "branch", "--show-current"], text=True
    ).strip(),
    "code_commit": subprocess.check_output(
        ["git", "-C", "$CODE", "rev-parse", "HEAD"], text=True
    ).strip(),
    "method_source": "https://transformer-circuits.pub/2026/workspace/index.html",
    "method": {
        "source_layer": 22,
        "source_representation": "attention-normalized residual stream at layer input",
        "target_layer": 26,
        "target_representation": "residual stream after penultimate transformer block",
        "sequence_length": 128,
        "prompt_count": 10,
        "prompt_seed": 42,
        "prompt_distribution": "deterministically sampled OWT cache prompts",
        "source_position_aggregation": "mean",
        "target_position_aggregation": "sum over all causal present/future positions",
        "prompt_aggregation": "elementwise mean",
        "attention_pattern_gradients": "enabled",
        "row_batch_size": 8,
    },
    "data": {
        "cache_dir": "$CACHE_DIR",
        "cache_read_only": True,
        "sample_boundaries_preserved": True,
        "token_ids_preserved": True,
        "attention_masks_preserved": True,
        "same_owt_source_as_flat_baselines": True,
    },
    "signal_gate": {
        "finite_nonzero_jacobian": True,
        "n5_to_n10_frobenius_cosine": ">= 0.95",
        "frozen_initial3_delta_over_logit_lens": ">= 0.005",
        "minimum_hard_dataset_delta_over_logit_lens": ">= -0.01",
        "must_beat_norm_matched_random_orthogonal_control": True,
        "train_architecture_only_after_all_pass": True,
    },
    "architecture_family_limit": 3,
    "uses_saebench_labels_for_jacobian": False,
    "uses_saebench_class_names_for_jacobian": False,
    "uses_eval_split_for_jacobian": False,
    "uses_test_feedback_for_jacobian": False,
}
path = Path("$ROOT/preregistration.json")
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit(f"Refusing to overwrite different preregistration: {path}")
else:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ ! -f "$JACOBIAN_DIR/average-jacobian-metadata.json" ]]; then
  "$PY" "$CODE/compute_true_jacobian_lens.py" \
    --model-dir "$MODEL_DIR" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$JACOBIAN_DIR" \
    --source-layer 22 \
    --target-layer 26 \
    --sequence-length 128 \
    --prompt-count 10 \
    --prompt-seed 42 \
    --row-batch-size 8 \
    --dtype bfloat16 \
    --device cuda \
    > "$ROOT/compute-jacobian.log" 2>&1
fi

"$PY" - <<PY | tee "$ROOT/jacobian-convergence-gate.log"
import json
from pathlib import Path

metadata = json.loads(
    Path("$JACOBIAN_DIR/average-jacobian-metadata.json").read_text(encoding="utf-8")
)
cosine = metadata["convergence"]["5"]["cosine_to_final"]
report = {
    "finite_nonzero": metadata["average_frobenius_norm"] > 0,
    "n5_to_n10_cosine": cosine,
    "n5_to_n10_cosine_at_least_0p95": cosine >= 0.95,
}
report["pass"] = all([
    report["finite_nonzero"],
    report["n5_to_n10_cosine_at_least_0p95"],
])
Path("$ROOT/jacobian-convergence-gate.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Jacobian convergence gate failed; frozen probing is prohibited")
PY
