#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-structured-aux-absorber-batchtopk-v3}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
SCREEN_ROOT="${SCREEN_ROOT:-/autodl-fs/data/structured_aux_absorber_batchtopk_v3_20260710}"
ROOT="${ROOT:-/autodl-fs/data/structured_aux_absorber_batchtopk_v3_long10_20260710}"
TRAIN_DIR="$ROOT/train_seed43"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
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
screen_gate_path = Path("$SCREEN_ROOT/initial3-gate.json")
screen_gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
if not screen_gate["gate"]["pass"]:
    raise SystemExit("The pre-registered three-epoch screen did not pass")
if cache.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
    raise SystemExit("Structured cache directory is writable")
branch = subprocess.check_output(
    ["git", "-C", str(code), "branch", "--show-current"], text=True
).strip()
if branch != "scpg-structured-aux-absorber-batchtopk-v3":
    raise SystemExit(f"Unexpected code branch: {branch}")

payload = {
    "experiment": "auxiliary-absorber BatchTopK SAE v3 matched long train",
    "registered_before_long_training_and_full7": True,
    "code_branch": branch,
    "code_branch_expected": "scpg-structured-aux-absorber-batchtopk-v3",
    "code_commit": subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip(),
    "screen_evidence": {
        "gate_path": str(screen_gate_path),
        "gate_sha256": hashlib.sha256(screen_gate_path.read_bytes()).hexdigest(),
        "base_mean_acc": screen_gate["base_mean_acc"],
        "candidate_mean_acc": screen_gate["candidate_mean_acc"],
        "overall_delta": screen_gate["overall_delta"],
        "dataset_deltas": screen_gate["dataset_deltas"],
    },
    "data": {
        "cache_dir": str(cache),
        "cache_manifest_sha256": hashlib.sha256(
            (cache / "manifest.json").read_bytes()
        ).hexdigest(),
        "cache_read_only": True,
        "same_owt_source_as_flat_baselines": True,
        "sample_boundaries_preserved": True,
        "layer": 22,
    },
    "architecture": {
        "total_features": 65536,
        "inner_sample_features": 32768,
        "outer_exposed_features": 32768,
        "top_k": 64,
        "parameter_count_each": 402721792,
        "exposed_feature_count_each": 32768,
        "token_reconstruction_uses_full_dictionary": True,
        "sample_mean_reconstruction_uses_inner_partition": True,
        "inner_partition_hidden_from_sparse_readout": True,
        "outer_partition_is_only_sparse_readout": True,
        "nested_loss_weight": 1.0,
        "l1_training_weight": 0.0,
    },
    "training": {
        "from_scratch": True,
        "warm_start_from_screen": False,
        "epochs": 10,
        "training_seed": 43,
        "initialization_seed": 430396,
        "learning_rate": 5e-5,
        "optimizer": "Adam(beta1=0.85,beta2=0.9999,eps=6.25e-10)",
        "batch_samples": 32,
        "train_fraction": 0.95,
        "threshold_calibration_batches": 128,
    },
    "fairness": {
        "ordinary_batchtopk_control_trained_from_scratch": True,
        "candidate_trained_from_scratch": True,
        "same_initial_tensors": True,
        "same_optimizer": True,
        "same_data_order": True,
        "same_epochs": True,
        "same_batchtopk_k": True,
        "same_threshold_calibration": True,
        "same_trainable_parameter_count": True,
        "same_exposed_feature_indices": True,
        "same_exposed_feature_count": True,
        "uses_saebench_labels_for_training": False,
        "uses_eval_split_for_training": False,
        "uses_one_vs_rest_targets_for_training": False,
        "uses_mean_diff_selection_for_training": False,
        "uses_test_feedback_for_training": False,
    },
    "evaluation": {
        "evaluation_seed": 42,
        "train_size_per_class": 512,
        "test_size_per_class": 128,
        "k_values": [1, 2, 5],
        "initial3_gate": {
            "candidate_minus_base_mean_acc": ">= 0.005",
            "minimum_per_dataset_delta": ">= -0.01",
        },
        "full7_only_after_long_initial3_pass": True,
        "full7_success_requires_candidate_mean_acc_strictly_above": 0.9,
        "full7_datasets": [
            "LabHC/bias_in_bios_class_set1",
            "LabHC/bias_in_bios_class_set2",
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "fancyzhx/ag_news",
            "Helsinki-NLP/europarl",
        ],
        "amazon_tasks": ["category", "sentiment"],
    },
}
path = root / "preregistration.json"
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit(f"Refusing to overwrite different preregistration: {path}")
else:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ ! -f "$TRAIN_DIR/train-summary-structured-sample-nested-batchtopk.json" ]]; then
  mkdir -p "$TRAIN_DIR"
  echo "== matched ten-epoch train start $(date -Is)"
  "$PY" "$CODE/train_structured_nested_batchtopk_sae.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$TRAIN_DIR" \
    --layer 22 \
    --n-total 65536 \
    --inner-features 32768 \
    --readout-mode outer \
    --top-k 64 \
    --nested-loss-weight 1.0 \
    --epochs 10 \
    --batch-samples 32 \
    --max-train-batches 0 \
    --max-validation-batches 0 \
    --threshold-calibration-batches 128 \
    --train-fraction 0.95 \
    --seed 43 \
    --initialization-seed 430396 \
    --lr 5e-5 \
    --beta1 0.85 \
    --beta2 0.9999 \
    --optimizer-eps 6.25e-10 \
    --l1-coeff 0 \
    --k-aux 2048 \
    --aux-loss-coeff 0.03125 \
    --dead-steps-threshold 0 \
    --log-every 100 \
    --device cuda \
    > "$ROOT/train.log" 2>&1
  echo "== matched ten-epoch train done $(date -Is)"
fi

"$PY" - <<PY | tee "$ROOT/train-gate.log"
import json
import math
from pathlib import Path

summary = json.loads(
    Path("$TRAIN_DIR/train-summary-structured-sample-nested-batchtopk.json").read_text()
)
base = summary["base_result"]
candidate = summary["candidate_result"]
targets = json.loads(Path(summary["targets_json"]).read_text())
all_rows = base["history"] + candidate["history"]
numeric_values = [
    value
    for row in all_rows
    for value in row.values()
    if isinstance(value, (int, float))
]
report = {
    "parameter_matched": summary["parameter_matched"],
    "base_parameter_count": summary["base_parameter_count"],
    "candidate_parameter_count": summary["candidate_parameter_count"],
    "parameter_count_is_expected": (
        summary["base_parameter_count"]
        == summary["candidate_parameter_count"]
        == 402721792
    ),
    "exposed_feature_count": summary["exposed_feature_count"],
    "exposed_count_is_expected": summary["exposed_feature_count"] == 32768,
    "same_outer_readout": all(
        target["feature_start"] == 32768
        and target["feature_end"] == 65536
        for target in targets
    ),
    "base_completed_10_epochs": len(base["history"]) == 10,
    "candidate_completed_10_epochs": len(candidate["history"]) == 10,
    "candidate_nested_gradient_nonzero": max(
        row["nested_module_grad"] for row in candidate["history"]
    ) > 0,
    "candidate_nested_parameters_updated": (
        candidate["nested_module_parameter_probe_max_delta"] > 0
    ),
    "base_threshold_finite_positive": (
        math.isfinite(base["threshold"]) and base["threshold"] > 0
    ),
    "candidate_threshold_finite_positive": (
        math.isfinite(candidate["threshold"])
        and candidate["threshold"] > 0
    ),
    "all_logged_numbers_finite": all(math.isfinite(v) for v in numeric_values),
}
report["pass"] = all(
    value for value in report.values() if isinstance(value, bool)
)
Path("$ROOT/train-gate.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Ten-epoch matched training gate failed")
PY

if [[ ! -f "$ROOT/initial3.json" ]]; then
  "$PY" "$CODE/saebench_sparse_probing_structured_batchtopk.py" \
    --targets-json "$TRAIN_DIR/targets-structured-sample-nested-batchtopk.json" \
    --output-json "$ROOT/initial3.json" \
    --output-md "$ROOT/initial3.md" \
    --model-dir "$MODEL_DIR" \
    --datasets \
      LabHC/bias_in_bios_class_set3 \
      canrager/amazon_reviews_mcauley_1and5 \
      fancyzhx/ag_news \
    --train-size 512 \
    --test-size 128 \
    --context-length 128 \
    --llm-batch-size 4 \
    --sae-seq-batch-size 2 \
    --k-values 1 2 5 \
    --random-seed 42 \
    --dtype bfloat16 \
    --device cuda \
    > "$ROOT/initial3.log" 2>&1
fi

"$PY" "$CODE/analyze_structured_batchtopk_gate.py" \
  --eval-json "$ROOT/initial3.json" \
  --output-json "$ROOT/initial3-gate.json" \
  --output-md "$ROOT/initial3-gate.md" \
  | tee "$ROOT/initial3-gate.log"

LONG_GATE_PASS="$($PY - <<PY
import json
print("1" if json.load(open("$ROOT/initial3-gate.json"))["gate"]["pass"] else "0")
PY
)"

if [[ "$LONG_GATE_PASS" != "1" ]]; then
  "$PY" - <<PY | tee "$ROOT/final-decision.log"
import json
from pathlib import Path
report = {
    "long_initial3_pass": False,
    "full7_ran": False,
    "decision": "stop-after-long-train-regression-and-close-family",
}
Path("$ROOT/final-decision.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
  exit 0
fi

if [[ ! -f "$ROOT/full7.json" ]]; then
  "$PY" "$CODE/saebench_sparse_probing_structured_batchtopk.py" \
    --targets-json "$TRAIN_DIR/targets-structured-sample-nested-batchtopk.json" \
    --output-json "$ROOT/full7.json" \
    --output-md "$ROOT/full7.md" \
    --model-dir "$MODEL_DIR" \
    --datasets \
      LabHC/bias_in_bios_class_set1 \
      LabHC/bias_in_bios_class_set2 \
      LabHC/bias_in_bios_class_set3 \
      canrager/amazon_reviews_mcauley_1and5 \
      fancyzhx/ag_news \
      Helsinki-NLP/europarl \
    --train-size 512 \
    --test-size 128 \
    --context-length 128 \
    --llm-batch-size 4 \
    --sae-seq-batch-size 2 \
    --k-values 1 2 5 \
    --random-seed 42 \
    --dtype bfloat16 \
    --device cuda \
    > "$ROOT/full7.log" 2>&1
fi

"$PY" - <<PY | tee "$ROOT/final-decision.log"
import json
from pathlib import Path

evaluation = json.loads(Path("$ROOT/full7.json").read_text())
by_key = {row["variant_key"]: row for row in evaluation["summary"]}
base = by_key["base"]
candidate = by_key["candidate"]
report = {
    "long_initial3_pass": True,
    "full7_ran": True,
    "base_mean_acc": base["mean_acc"],
    "candidate_mean_acc": candidate["mean_acc"],
    "candidate_minus_base": candidate["mean_acc"] - base["mean_acc"],
    "candidate_mean_auc": candidate["mean_auc"],
    "candidate_top_1_acc": candidate["top_1_acc"],
    "candidate_top_2_acc": candidate["top_2_acc"],
    "candidate_top_5_acc": candidate["top_5_acc"],
    "strictly_above_0p9": candidate["mean_acc"] > 0.9,
    "decision": (
        "advance-to-five-seed-confirmation"
        if candidate["mean_acc"] > 0.9
        else "record-full7-and-close-family-below-goal"
    ),
}
Path("$ROOT/final-decision.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
