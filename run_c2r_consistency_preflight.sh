#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${ROOT:?ROOT must point to a new output directory}"
PYTHON_BIN="${PYTHON_BIN:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/activation_outputs_batched}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/root/saint/trained_sae-relu-l22.pt}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/root/autodl-tmp/saebench_sparse_probing_all_architectures.py}"

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing output: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT"
cd "$CODE"

cp c2r_consistency_preflight_preregistration.json "$ROOT/preregistration.json"
sha256sum "$ROOT/preregistration.json" > "$ROOT/preregistration.sha256"
git rev-parse HEAD > "$ROOT/git-commit.txt"
git status --short --branch > "$ROOT/git-status-before-training.txt"

"$PYTHON_BIN" test_c2r_consistency_preflight.py 2>&1 \
  | tee "$ROOT/unit-test.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train_c2r_consistency_preflight.py \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$ROOT/train" \
  --steps 600 \
  --batch-tokens 256 \
  --validation-batches 32 \
  --seed 42 \
  --lr 1e-6 \
  --l1-coeff 1e-6 \
  --c2r-interval 5 \
  --c2r-subset 2048 \
  --c2r-fraction 0.01 \
  --log-every 100 \
  --device cuda \
  | tee "$ROOT/train.log"

"$PYTHON_BIN" validate_c2r_consistency_preflight.py \
  --summary "$ROOT/train/train-summary-c2r-preflight.json" \
  --output-json "$ROOT/training-integrity.json" \
  | tee "$ROOT/training-integrity.log"

PYTHONPATH="$CODE" CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$EVAL_SCRIPT" \
  --targets-json "$ROOT/train/targets-c2r-preflight.json" \
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

"$PYTHON_BIN" analyze_c2r_consistency_gate.py \
  --eval-json "$ROOT/initial3-eval.json" \
  --training-integrity "$ROOT/training-integrity.json" \
  --output-json "$ROOT/initial3-gate.json" \
  --output-md "$ROOT/initial3-gate.md" \
  --minimum-control-delta 0.005 \
  --maximum-dataset-drop 0.01 \
  --reference-initial3 0.837543 \
  --maximum-reference-gap 0.01 \
  | tee "$ROOT/initial3-gate.log"

"$PYTHON_BIN" -c \
  'import json,sys; report=json.load(open(sys.argv[1])); print(report["decision"]); print("Parameterized C2R development is allowed." if report["pass"] else "C2R is stopped before parameterized architecture development and full7.")' \
  "$ROOT/initial3-gate.json" \
  | tee "$ROOT/decision.log"
