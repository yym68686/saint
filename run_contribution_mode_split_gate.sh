#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${ROOT:?ROOT must point to a new output directory}"
PYTHON_BIN="${PYTHON_BIN:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/activation_outputs_batched}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/v396_logcompanding_initial5/trained_sae-v396-logcompanding-relu.pt}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
BASE_EVALUATOR="${BASE_EVALUATOR:-/root/autodl-tmp/saebench_sparse_probing_all_architectures.py}"

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing output: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT"
cd "$CODE"

cp contribution_mode_split_preregistration.json "$ROOT/preregistration.json"
sha256sum "$ROOT/preregistration.json" > "$ROOT/preregistration.sha256"
sha256sum "$BASE_EVALUATOR" > "$ROOT/base-evaluator.sha256"
git rev-parse HEAD > "$ROOT/git-commit.txt"
git status --short --branch > "$ROOT/git-status-before-fit.txt"

"$PYTHON_BIN" test_contribution_mode_split.py 2>&1 | tee "$ROOT/unit-test.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" fit_contribution_mode_split_gate.py \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$ROOT/fit" \
  --sample-files 64 \
  --top-instances 64 \
  --scan-batch-tokens 256 \
  --feature-chunk 128 \
  --kmeans-iterations 8 \
  --minimum-cluster-size 16 \
  --split-pairs 4096 \
  --rho 0.5 \
  --seed 42 \
  --device cuda \
  | tee "$ROOT/fit.log"

PYTHONPATH="$CODE" CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  eval_contribution_mode_split_gate.py \
  --base-evaluator "$BASE_EVALUATOR" \
  --targets-json "$ROOT/fit/targets-contribution-mode-split.json" \
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

"$PYTHON_BIN" analyze_contribution_mode_split_gate.py \
  --eval-json "$ROOT/initial3-eval.json" \
  --fit-summary "$ROOT/fit/fit-summary.json" \
  --output-json "$ROOT/initial3-gate.json" \
  --output-md "$ROOT/initial3-gate.md" \
  --minimum-control-delta 0.005 \
  --maximum-dataset-drop 0.01 \
  --reference-initial3 0.837543 \
  --maximum-reference-gap 0.01 \
  | tee "$ROOT/initial3-gate.log"

"$PYTHON_BIN" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(d["decision"])' \
  "$ROOT/initial3-gate.json" | tee "$ROOT/decision.log"

sha256sum \
  "$ROOT/preregistration.json" \
  "$ROOT/fit/fit-summary.json" \
  "$ROOT/fit/contribution-mode-split-spec.pt" \
  "$ROOT/initial3-eval.json" \
  "$ROOT/initial3-gate.json" \
  > "$ROOT/result-artifacts.sha256"
