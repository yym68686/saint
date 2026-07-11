#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/root/saint-cascaded-concept-sae-v3}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
ROOT="${ROOT:-/autodl-fs/data/cascaded_concept_sae_v3_20260711}"
V396="${V396:-/root/autodl-tmp/v396_logcompanding_initial5/trained_sae-v396-logcompanding-relu.pt}"
DATA="${DATA:-/root/autodl-tmp/activation_outputs_batched}"
MODEL="${MODEL:-/root/saint/llama_3.2-3B_model/original}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/root/autodl-tmp/saebench_sparse_probing_all_architectures.py}"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$ROOT"

"$PY" -m py_compile \
  "$CODE/train_cascaded_concept_sae.py" \
  "$CODE/saebench_sparse_probing_cascaded_concept.py" \
  "$CODE/analyze_cascaded_unsupervised_gate.py" \
  "$CODE/analyze_cascaded_concept_gate.py" \
  "$CODE/test_cascaded_concept_sae.py"

"$PY" - <<'PY' | tee "$ROOT/unit-tests.log"
import test_cascaded_concept_sae as tests

for name in sorted(vars(tests)):
    if name.startswith("test_"):
        vars(tests)[name]()
        print("PASS", name)
PY

"$PY" "$CODE/train_cascaded_concept_sae.py" \
  --v396-checkpoint "$V396" \
  --data-dir "$DATA" \
  --output-dir "$ROOT" \
  --steps 600 \
  --batch-tokens 64 \
  --activity-steps 600 \
  --high-features 3072 \
  --active-atom-cap 1024 \
  --seed 42 \
  --low-lr 1e-6 \
  --high-lr 1e-5 \
  --l1-coeff 1e-6 \
  --hierarchy-weight 1.0 \
  --hierarchy-l1-coeff 1e-6 \
  --transport-weight 1e-3 \
  --transport-temperature 0.1 \
  --sinkhorn-iterations 100 \
  --beta-anchor-coeff 1e-3 \
  --gain-anchor-coeff 1e-4 \
  --log-every 50 \
  --device cuda \
  > "$ROOT/training.log" 2>&1

set +e
"$PY" "$CODE/analyze_cascaded_unsupervised_gate.py" \
  --checkpoint "$ROOT/trained_sae-cascaded_concept_v3.pt" \
  --output-json "$ROOT/unsupervised-gate.json" \
  --output-md "$ROOT/unsupervised-gate.md" \
  | tee "$ROOT/unsupervised-gate.log"
UNSUPERVISED_STATUS=${PIPESTATUS[0]}
set -e
if [[ "$UNSUPERVISED_STATUS" -eq 2 ]]; then
  echo "Stopped before Initial3: label-free hierarchy gate failed."
  exit 0
fi
if [[ "$UNSUPERVISED_STATUS" -ne 0 ]]; then
  exit "$UNSUPERVISED_STATUS"
fi

"$PY" "$CODE/saebench_sparse_probing_cascaded_concept.py" \
  --eval-script "$EVAL_SCRIPT" \
  --targets-json "$ROOT/targets-cascaded-concept.json" \
  --output-json "$ROOT/initial3.json" \
  --output-md "$ROOT/initial3.md" \
  --model-dir "$MODEL" \
  --datasets \
    LabHC/bias_in_bios_class_set3 \
    canrager/amazon_reviews_mcauley_1and5 \
    fancyzhx/ag_news \
  --train-size 512 \
  --test-size 128 \
  --context-length 128 \
  --llm-batch-size 4 \
  --sae-seq-batch-size 1 \
  --k-values 1 2 5 \
  --random-seed 42 \
  --dtype bfloat16 \
  --device cuda \
  > "$ROOT/initial3.log" 2>&1

"$PY" "$CODE/analyze_cascaded_concept_gate.py" \
  --train-summary "$ROOT/train-summary-cascaded-concept.json" \
  --eval-json "$ROOT/initial3.json" \
  --output-json "$ROOT/initial3-gate.json" \
  --output-md "$ROOT/initial3-gate.md" \
  | tee "$ROOT/initial3-gate.log"
