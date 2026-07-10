#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${ROOT:?ROOT must point to a new formal output directory}"
PYTHON_BIN="${PYTHON_BIN:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/root/autodl-tmp/v396_logcompanding_initial5/trained_sae-v396-logcompanding-relu.pt}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/root/autodl-tmp/saebench_sparse_probing_all_architectures.py}"

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing formal root: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT"
cd "$CODE"

cp structured_nexttoken_downstream_v2_preregistration.json "$ROOT/preregistration.json"
sha256sum "$ROOT/preregistration.json" > "$ROOT/preregistration.sha256"
git status --short --branch > "$ROOT/git-status-before-training.txt"
git rev-parse HEAD > "$ROOT/git-commit.txt"

"$PYTHON_BIN" test_structured_nexttoken_downstream_sae.py \
  | tee "$ROOT/unit-test.log"

"$PYTHON_BIN" validate_structured_activation_cache.py \
  --cache-dir "$CACHE_DIR" \
  --require-read-only \
  --output-json "$ROOT/cache-validation.json" \
  | tee "$ROOT/cache-validation.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train_structured_nexttoken_downstream_sae.py \
  --cache-dir "$CACHE_DIR" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$ROOT/train" \
  --steps 600 \
  --batch-sequences 2 \
  --sequence-length 32 \
  --context-rank 32 \
  --context-lr 5e-5 \
  --downstream-loss-weight 0.1 \
  --log-every 100 \
  | tee "$ROOT/train.log"

"$PYTHON_BIN" validate_nexttoken_downstream_v2_training.py \
  --summary "$ROOT/train/train-summary-nexttoken-downstream-v2.json" \
  --output-json "$ROOT/training-integrity.json" \
  | tee "$ROOT/training-integrity.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" saebench_sparse_probing_v396_causal_suite.py \
  --eval-script "$EVAL_SCRIPT" \
  --targets-json "$ROOT/train/targets-nexttoken-downstream-v2.json" \
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
  --k-values 1 2 5 \
  --random-seed 42 \
  | tee "$ROOT/initial3-eval.log"

"$PYTHON_BIN" analyze_nexttoken_downstream_v2_gate.py \
  --eval-json "$ROOT/initial3-eval.json" \
  --train-summary "$ROOT/train/train-summary-nexttoken-downstream-v2.json" \
  --output-json "$ROOT/initial3-gate.json" \
  --output-md "$ROOT/initial3-gate.md" \
  | tee "$ROOT/initial3-gate.log"
