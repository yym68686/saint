#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/root/saint-structured-nested-turn-sae-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
OLD_MEAN_PATH="${OLD_MEAN_PATH:-/root/autodl-tmp/activation_outputs_mean.pt}"
ROOT="${ROOT:-/autodl-fs/data/structured_nested_turn_sae_v1_20260710}"
SMOKE_DIR="$ROOT/smoke"
SCREEN_DIR="$ROOT/screen_seed42"
REUSED_BASE_CHECKPOINT="${REUSED_BASE_CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
REUSED_BASE_SUMMARY="${REUSED_BASE_SUMMARY:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/train-summary-structured-dual-granularity.json}"
REUSED_BASE_SHA256="${REUSED_BASE_SHA256:-407aa42b2bab27a8f1ab24369ea860649fd412aeb0d0239848800c2453197385}"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
source /etc/network_turbo >/dev/null 2>&1 || true
mkdir -p "$ROOT"

"$PY" - <<PY | tee "$ROOT/preregistration.log"
import hashlib
import json
import subprocess
from pathlib import Path

root = Path("$ROOT")
code = Path("$CODE")
cache = Path("$CACHE_DIR")
base = Path("$REUSED_BASE_CHECKPOINT")
digest = hashlib.sha256()
with base.open("rb") as handle:
    while chunk := handle.read(8 * 1024 * 1024):
        digest.update(chunk)
base_sha256 = digest.hexdigest()
if base_sha256 != "$REUSED_BASE_SHA256":
    raise SystemExit(
        f"Base hash mismatch: {base_sha256} != $REUSED_BASE_SHA256"
    )
payload = {
    "experiment": "sample-nested shared-dictionary ReLU SAE v1",
    "registered_before_training_and_evaluation": True,
    "code_branch": subprocess.check_output(
        ["git", "-C", str(code), "branch", "--show-current"], text=True
    ).strip(),
    "code_commit": subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip(),
    "method_basis": {
        "external_primary_source": "https://arxiv.org/abs/2606.28548",
        "measured_predecessor": "structured dual-granularity v3",
        "predecessor_failure": (
            "decoupled sample-mean/token-residual responsibilities lowered "
            "token-only Initial3 by 0.023394 and AG News by 0.052734"
        ),
        "controlled_mechanism_change": (
            "one shared encoder/decoder reconstructs tokens with the full "
            "dictionary while its first half also reconstructs the true "
            "within-sample mean"
        ),
    },
    "data": {
        "cache_dir": str(cache),
        "cache_manifest_sha256": hashlib.sha256(
            (cache / "manifest.json").read_bytes()
        ).hexdigest(),
        "cache_read_only": True,
        "same_owt_source_as_flat_baselines": True,
        "sample_boundaries_preserved": True,
        "token_ids_preserved": True,
        "attention_masks_preserved": True,
        "layer": 22,
    },
    "architecture": {
        "total_features": 65536,
        "inner_sample_features": 32768,
        "outer_features": 32768,
        "parameter_count": 402721792,
        "exposed_feature_count": 65536,
        "shared_encoder_decoder": True,
        "token_reconstruction_uses_full_dictionary": True,
        "sample_mean_reconstruction_uses_inner_partition": True,
        "turn_loss_weight": 1.0,
        "inner_l1_weight": 1.0,
        "frozen_bank_at_inference": False,
    },
    "fairness": {
        "base_checkpoint": str(base),
        "base_checkpoint_sha256": base_sha256,
        "same_initialization_seed": 420396,
        "same_optimizer": True,
        "same_data_order": True,
        "same_epochs": True,
        "same_trainable_parameter_count": True,
        "same_exposed_feature_count": True,
        "uses_saebench_labels_for_training": False,
        "uses_eval_split_for_training": False,
        "uses_one_vs_rest_targets_for_training": False,
        "uses_mean_diff_selection_for_training": False,
        "uses_test_feedback_for_training": False,
    },
    "initial3": [
        "LabHC/bias_in_bios_class_set3",
        "canrager/amazon_reviews_mcauley_1and5",
        "fancyzhx/ag_news",
    ],
    "gate": {
        "candidate_minus_base_mean_acc": ">= 0.005",
        "minimum_per_dataset_delta": ">= -0.01",
        "run_full7_only_after_pass": True,
    },
    "family": "sample-nested shared-dictionary",
    "family_version": 1,
    "maximum_family_versions": 3,
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

"$PY" "$CODE/validate_structured_activation_cache.py" \
  --cache-dir "$CACHE_DIR" \
  --require-read-only \
  --output-json "$ROOT/cache-validation.json" \
  > "$ROOT/cache-validation.log" 2>&1

"$PY" - <<PY | tee "$ROOT/cache-compatibility.log"
import json
from pathlib import Path
import torch
import torch.nn.functional as F

new = torch.load(
    Path("$CACHE_DIR") / "mean-layer-22.pt",
    map_location="cpu",
    weights_only=True,
).float()
old = torch.load(
    Path("$OLD_MEAN_PATH"), map_location="cpu", weights_only=True
).float()
delta = new - old
report = {
    "max_abs_difference": float(delta.abs().max().item()),
    "mean_abs_difference": float(delta.abs().mean().item()),
    "rmse": float(delta.square().mean().sqrt().item()),
    "cosine_similarity": float(
        F.cosine_similarity(new.unsqueeze(0), old.unsqueeze(0)).item()
    ),
}
report["pass"] = (
    report["mean_abs_difference"] <= 1e-5
    and report["cosine_similarity"] >= 0.999999
)
Path("$ROOT/cache-compatibility.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Structured cache is not compatible with old L22 data")
PY

if [[ ! -f "$SMOKE_DIR/train-summary-structured-sample-nested.json" ]]; then
  mkdir -p "$SMOKE_DIR"
  "$PY" "$CODE/train_structured_nested_turn_sae.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$SMOKE_DIR" \
    --reuse-base-checkpoint "$REUSED_BASE_CHECKPOINT" \
    --reuse-base-summary "$REUSED_BASE_SUMMARY" \
    --layer 22 \
    --n-total 65536 \
    --inner-features 32768 \
    --turn-loss-weight 1.0 \
    --inner-l1-weight 1.0 \
    --epochs 1 \
    --batch-samples 4 \
    --max-train-batches 2 \
    --max-validation-batches 1 \
    --train-fraction 0.95 \
    --seed 42 \
    --initialization-seed 420396 \
    --lr 5e-5 \
    --beta1 0.85 \
    --beta2 0.9999 \
    --optimizer-eps 6.25e-10 \
    --l1-coeff 1e-4 \
    --k-aux 2048 \
    --aux-loss-coeff 0.03125 \
    --dead-steps-threshold 0 \
    --log-every 1 \
    --device cuda \
    > "$ROOT/smoke-train.log" 2>&1
fi

"$PY" - <<PY | tee "$ROOT/smoke-gate.log"
import json
import math
from pathlib import Path

summary = json.loads(
    Path("$SMOKE_DIR/train-summary-structured-sample-nested.json").read_text()
)
candidate = summary["candidate_result"]
history = candidate["history"]
report = {
    "parameter_matched": summary["parameter_matched"],
    "base_parameter_count": summary["base_parameter_count"],
    "candidate_parameter_count": summary["candidate_parameter_count"],
    "exposed_feature_count": summary["exposed_feature_count"],
    "nested_module_gradient_nonzero": max(
        row["nested_module_grad"] for row in history
    ) > 0,
    "nested_module_parameter_delta_nonzero": (
        candidate["nested_module_parameter_probe_max_delta"] > 0
    ),
    "inner_mean_loss_finite_positive": all(
        math.isfinite(row["inner_mean_recon"])
        and row["inner_mean_recon"] > 0
        for row in history
    ),
    "token_loss_finite_positive": all(
        math.isfinite(row["token_recon"]) and row["token_recon"] > 0
        for row in history
    ),
}
report["pass"] = all(
    value for key, value in report.items() if isinstance(value, bool)
)
Path("$ROOT/smoke-gate.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Sample-nested implementation smoke failed")
PY

if [[ ! -f "$SCREEN_DIR/train-summary-structured-sample-nested.json" ]]; then
  mkdir -p "$SCREEN_DIR"
  echo "== full-budget training start $(date -Is)"
  "$PY" "$CODE/train_structured_nested_turn_sae.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$SCREEN_DIR" \
    --reuse-base-checkpoint "$REUSED_BASE_CHECKPOINT" \
    --reuse-base-summary "$REUSED_BASE_SUMMARY" \
    --layer 22 \
    --n-total 65536 \
    --inner-features 32768 \
    --turn-loss-weight 1.0 \
    --inner-l1-weight 1.0 \
    --epochs 10 \
    --batch-samples 32 \
    --max-train-batches 0 \
    --max-validation-batches 0 \
    --train-fraction 0.95 \
    --seed 42 \
    --initialization-seed 420396 \
    --lr 5e-5 \
    --beta1 0.85 \
    --beta2 0.9999 \
    --optimizer-eps 6.25e-10 \
    --l1-coeff 1e-4 \
    --k-aux 2048 \
    --aux-loss-coeff 0.03125 \
    --dead-steps-threshold 0 \
    --log-every 100 \
    --device cuda \
    > "$ROOT/screen-train.log" 2>&1
  echo "== full-budget training done $(date -Is)"
fi

if [[ ! -f "$ROOT/initial3.json" ]]; then
  "$PY" "$CODE/saebench_sparse_probing_structured_nested.py" \
    --targets-json "$SCREEN_DIR/targets-structured-sample-nested.json" \
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

"$PY" "$CODE/analyze_structured_nested_gate.py" \
  --eval-json "$ROOT/initial3.json" \
  --output-json "$ROOT/initial3-gate.json" \
  --output-md "$ROOT/initial3-gate.md" \
  | tee "$ROOT/initial3-gate.log"

GATE_PASS="$("$PY" - <<PY
import json
from pathlib import Path
print("true" if json.loads(Path("$ROOT/initial3-gate.json").read_text())["gate"]["pass"] else "false")
PY
)"

if [[ "$GATE_PASS" != "true" ]]; then
  echo "== Initial3 gate rejected sample-nested v1; full7 prohibited"
  exit 0
fi

if [[ ! -f "$ROOT/full7.json" ]]; then
  "$PY" "$CODE/saebench_sparse_probing_structured_nested.py" \
    --targets-json "$SCREEN_DIR/targets-structured-sample-nested.json" \
    --output-json "$ROOT/full7.json" \
    --output-md "$ROOT/full7.md" \
    --model-dir "$MODEL_DIR" \
    --datasets \
      LabHC/bias_in_bios_class_set1 \
      LabHC/bias_in_bios_class_set2 \
      LabHC/bias_in_bios_class_set3 \
      canrager/amazon_reviews_mcauley_1and5 \
      canrager/amazon_reviews_mcauley_1and5_sentiment \
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

"$PY" - <<PY | tee "$ROOT/full7-decision.log"
import json
from pathlib import Path

payload = json.loads(Path("$ROOT/full7.json").read_text())
rows = {row["variant_key"]: row for row in payload["summary"]}
report = {
    "base_mean_acc": rows["base"]["mean_acc"],
    "candidate_mean_acc": rows["candidate"]["mean_acc"],
    "candidate_minus_base": rows["candidate"]["mean_acc"] - rows["base"]["mean_acc"],
    "strictly_exceeds_0p9000": rows["candidate"]["mean_acc"] > 0.9000,
}
report["decision"] = (
    "enter-five-seed-validation"
    if report["strictly_exceeds_0p9000"]
    else "below-success-threshold"
)
Path("$ROOT/full7-decision.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
