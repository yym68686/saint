#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/root/saint-structured-dual-granularity-v3}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
ROOT="${ROOT:-/autodl-fs/data/structured_dual_granularity_v3_20260710}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-$ROOT/screen_seed42/trained_sae-structured-dual-granularity-responsibility-split.pt}"
TARGETS="$ROOT/initial3-readout-diagnostic-targets.json"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
source /etc/network_turbo >/dev/null 2>&1 || true

"$PY" - <<PY
import json
from pathlib import Path

targets = [
    {
        "label": "structured-cache ReLU base-only",
        "kind": "relu",
        "layer": 22,
        "checkpoint": "$BASE_CHECKPOINT",
        "variant_key": "base",
    },
    {
        "label": "v3 all features",
        "kind": "structured_dual_granularity_responsibility_split",
        "layer": 22,
        "checkpoint": "$CANDIDATE_CHECKPOINT",
        "variant_key": "candidate_all",
        "readout_source": "all",
    },
    {
        "label": "v3 token branch only",
        "kind": "structured_dual_granularity_responsibility_split",
        "layer": 22,
        "checkpoint": "$CANDIDATE_CHECKPOINT",
        "variant_key": "candidate_token",
        "readout_source": "token",
    },
    {
        "label": "v3 semantic branch only",
        "kind": "structured_dual_granularity_responsibility_split",
        "layer": 22,
        "checkpoint": "$CANDIDATE_CHECKPOINT",
        "variant_key": "candidate_semantic",
        "readout_source": "semantic",
    },
]
Path("$TARGETS").write_text(
    json.dumps(targets, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

"$PY" "$CODE/saebench_sparse_probing_structured_dual_granularity.py" \
  --targets-json "$TARGETS" \
  --output-json "$ROOT/initial3-readout-diagnostic.json" \
  --output-md "$ROOT/initial3-readout-diagnostic.md" \
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
  > "$ROOT/initial3-readout-diagnostic.log" 2>&1

"$PY" - <<PY | tee "$ROOT/initial3-readout-diagnostic-summary.log"
import json
from pathlib import Path

payload = json.loads(
    Path("$ROOT/initial3-readout-diagnostic.json").read_text(encoding="utf-8")
)
rows = {row["variant_key"]: row for row in payload["summary"]}
report = {
    key: {
        metric: row[metric]
        for metric in ("mean_acc", "mean_auc", "top_1_acc", "top_2_acc", "top_5_acc")
    }
    for key, row in rows.items()
}
report["deltas_vs_base"] = {
    key: report[key]["mean_acc"] - report["base"]["mean_acc"]
    for key in ("candidate_all", "candidate_token", "candidate_semantic")
}
Path("$ROOT/initial3-readout-diagnostic-summary.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
