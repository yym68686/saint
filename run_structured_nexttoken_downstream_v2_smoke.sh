#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${ROOT:?ROOT must point to a new smoke output directory}"
PYTHON_BIN="${PYTHON_BIN:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/root/autodl-tmp/v396_logcompanding_initial5/trained_sae-v396-logcompanding-relu.pt}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"

mkdir -p "$ROOT"
cd "$CODE"

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
  --steps 2 \
  --batch-sequences 2 \
  --sequence-length 24 \
  --context-rank 32 \
  --downstream-loss-weight 0.01 \
  --log-every 1 \
  --no-save-checkpoints \
  | tee "$ROOT/train-smoke.log"
