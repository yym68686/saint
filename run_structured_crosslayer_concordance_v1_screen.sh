#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${ROOT:?ROOT must point to a new formal output directory}"
PYTHON_BIN="${PYTHON_BIN:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/root/autodl-tmp/saebench_sparse_probing_all_architectures.py}"

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing formal root: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT"
cd "$CODE"

cp structured_crosslayer_concordance_v1_preregistration.json "$ROOT/preregistration.json"
sha256sum "$ROOT/preregistration.json" > "$ROOT/preregistration.sha256"
git status --short --branch > "$ROOT/git-status-before-training.txt"
git rev-parse HEAD > "$ROOT/git-commit.txt"

"$PYTHON_BIN" test_structured_crosslayer_concordance_sae.py \
  | tee "$ROOT/unit-test.log"

"$PYTHON_BIN" validate_structured_activation_cache.py \
  --cache-dir "$CACHE_DIR" \
  --require-read-only \
  --output-json "$ROOT/cache-validation.json" \
  | tee "$ROOT/cache-validation.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train_structured_crosslayer_concordance_sae.py \
  --cache-dir "$CACHE_DIR" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$ROOT/train" \
  --layers 20 21 22 23 \
  --reference-layer 22 \
  --steps 600 \
  --batch-samples 8 \
  --train-fraction 0.95 \
  --seed 42 \
  --lr 1e-5 \
  --l1-coeff 1e-4 \
  --concordance-weight 0.005 \
  --calibration-groups 16 \
  --max-log-scale 0.25 \
  --validation-batches 32 \
  --log-every 50 \
  --device cuda \
  | tee "$ROOT/train.log"

"$PYTHON_BIN" validate_crosslayer_concordance_training.py \
  --summary "$ROOT/train/train-summary-crosslayer-concordance-v1.json" \
  --output-json "$ROOT/training-integrity.json" \
  | tee "$ROOT/training-integrity.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" saebench_sparse_probing_structured_dual_granularity.py \
  --eval-script "$EVAL_SCRIPT" \
  --targets-json "$ROOT/train/targets-crosslayer-concordance-v1.json" \
  --output-json "$ROOT/initial3-eval.json" \
  --output-md "$ROOT/initial3-eval.md" \
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
  | tee "$ROOT/initial3-eval.log"

"$PYTHON_BIN" analyze_crosslayer_concordance_gate.py \
  --eval-json "$ROOT/initial3-eval.json" \
  --training-integrity "$ROOT/training-integrity.json" \
  --output-json "$ROOT/initial3-gate.json" \
  --output-md "$ROOT/initial3-gate.md" \
  --minimum-control-delta 0.005 \
  --maximum-dataset-drop 0.01 \
  --reference-initial3 0.851389 \
  --maximum-reference-gap 0.01 \
  | tee "$ROOT/initial3-gate.log"

"$PYTHON_BIN" - <<PY | tee "$ROOT/decision.log"
import json
from pathlib import Path

report = json.loads(Path("$ROOT/initial3-gate.json").read_text(encoding="utf-8"))
print(report["decision"])
if report["pass"]:
    print("The short screen passed. Long training may now be scheduled; full7 is still prohibited before long training.")
else:
    print("The short screen failed. Long training and full7 are prohibited for v1.")
PY
