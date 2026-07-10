#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/root/saint-structured-dual-granularity-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
PARQUET_PATH="${PARQUET_PATH:-/root/autodl-tmp/train-00000-of-00082.parquet}"
OLD_MEAN_PATH="${OLD_MEAN_PATH:-/root/autodl-tmp/activation_outputs_mean.pt}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
ROOT="${ROOT:-/autodl-fs/data/structured_dual_granularity_v1_20260710}"
SCREEN_DIR="$ROOT/screen_seed42"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
source /etc/network_turbo >/dev/null 2>&1 || true
mkdir -p "$ROOT"

"$PY" - <<PY | tee "$ROOT/preregistration.log"
import json
import subprocess
from pathlib import Path

root = Path("$ROOT")
code = Path("$CODE")
payload = {
    "experiment": "structured-cache parameter-matched dual-granularity SAE v1",
    "registered_before_cache_capture": True,
    "code_branch": subprocess.check_output(
        ["git", "-C", str(code), "branch", "--show-current"],
        text=True,
    ).strip(),
    "code_commit": subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "source_parquet": "$PARQUET_PATH",
    "old_l22_mean": "$OLD_MEAN_PATH",
    "cache_dir": "$CACHE_DIR",
    "cache_configuration": {
        "samples": 50000,
        "max_token_length": 192,
        "layers": [20, 21, 22, 23],
        "seed": 42,
        "dtype": "bfloat16",
        "sample_boundaries_preserved": True,
        "token_ids_preserved": True,
        "attention_masks_preserved": True,
    },
    "screen_configuration": {
        "seed": 42,
        "initialization_seed": 420396,
        "epochs": 10,
        "n_total": 65536,
        "n_semantic": 4096,
        "layer": 22,
        "parameter_count_each": 402721792,
        "same_initial_tensors": True,
        "same_optimizer": True,
        "same_data_order": True,
        "same_exposed_feature_count": True,
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
    "family_version": 1,
    "maximum_family_versions": 3,
}
path = root / "preregistration.json"
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit(f"Refusing to overwrite a different preregistration: {path}")
else:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ ! -f "$CACHE_DIR/manifest.json" ]]; then
  echo "== structured cache capture start $(date -Is)"
  "$PY" "$CODE/capture_structured_activations.py" \
    --model-dir "$MODEL_DIR" \
    --parquet-path "$PARQUET_PATH" \
    --output-dir "$CACHE_DIR" \
    --layers 20 21 22 23 \
    --num-samples 50000 \
    --max-token-length 192 \
    --batch-size 32 \
    --shard-samples 256 \
    --seed 42 \
    --dtype bfloat16 \
    --device cuda \
    --num-workers 4 \
    --finalize-read-only \
    > "$ROOT/cache-capture.log" 2>&1
  echo "== structured cache capture done $(date -Is)"
fi

"$PY" "$CODE/validate_structured_activation_cache.py" \
  --cache-dir "$CACHE_DIR" \
  --require-read-only \
  --output-json "$ROOT/cache-validation.json" \
  | tee "$ROOT/cache-validation.log"

"$PY" - <<PY | tee "$ROOT/cache-compatibility-audit.log"
import json
from pathlib import Path

import torch
import torch.nn.functional as F

cache_dir = Path("$CACHE_DIR")
old_mean_path = Path("$OLD_MEAN_PATH")
new_mean = torch.load(
    cache_dir / "mean-layer-22.pt",
    map_location="cpu",
    weights_only=True,
).float()
old_mean = torch.load(
    old_mean_path,
    map_location="cpu",
    weights_only=True,
).float()
if new_mean.shape != old_mean.shape:
    raise SystemExit(
        f"L22 mean shape mismatch: new={tuple(new_mean.shape)} "
        f"old={tuple(old_mean.shape)}"
    )
difference = new_mean - old_mean
report = {
    "old_mean_path": str(old_mean_path),
    "new_mean_path": str(cache_dir / "mean-layer-22.pt"),
    "shape": list(new_mean.shape),
    "max_abs_difference": float(difference.abs().max().item()),
    "mean_abs_difference": float(difference.abs().mean().item()),
    "root_mean_squared_difference": float(difference.square().mean().sqrt().item()),
    "cosine_similarity": float(
        F.cosine_similarity(new_mean.unsqueeze(0), old_mean.unsqueeze(0)).item()
    ),
}
report["compatible"] = (
    report["mean_abs_difference"] <= 1.0e-5
    and report["cosine_similarity"] >= 0.999999
)
(Path("$ROOT") / "cache-compatibility-audit.json").write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2))
if not report["compatible"]:
    raise SystemExit("New structured cache is not distribution-compatible with the old L22 cache")
PY

if [[ ! -f "$SCREEN_DIR/train-summary-structured-dual-granularity.json" ]]; then
  echo "== parameter-matched screen training start $(date -Is)"
  "$PY" "$CODE/train_structured_dual_granularity_sae.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$SCREEN_DIR" \
    --layer 22 \
    --n-total 65536 \
    --n-semantic 4096 \
    --epochs 10 \
    --batch-samples 32 \
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
  echo "== parameter-matched screen training done $(date -Is)"
fi

if [[ ! -f "$ROOT/semantic-branch-diagnostic.json" ]]; then
  "$PY" "$CODE/diagnose_structured_semantic_branch.py" \
    --cache-dir "$CACHE_DIR" \
    --checkpoint "$SCREEN_DIR/trained_sae-structured-dual-granularity.pt" \
    --output-json "$ROOT/semantic-branch-diagnostic.json" \
    --layer 22 \
    --batch-samples 32 \
    --train-fraction 0.95 \
    --seed 42 \
    --max-batches 32 \
    --device cuda \
    | tee "$ROOT/semantic-branch-diagnostic.log"
fi

if [[ ! -f "$ROOT/initial3.json" ]]; then
  echo "== Initial3 eval start $(date -Is)"
  "$PY" "$CODE/saebench_sparse_probing_structured_dual_granularity.py" \
    --targets-json "$SCREEN_DIR/targets-structured-dual-granularity.json" \
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
  echo "== Initial3 eval done $(date -Is)"
fi

"$PY" "$CODE/analyze_structured_dual_granularity_gate.py" \
  --eval-json "$ROOT/initial3.json" \
  --output-json "$ROOT/initial3-gate.json" \
  --output-md "$ROOT/initial3-gate.md" \
  | tee "$ROOT/initial3-gate.log"

GATE_PASS="$("$PY" - <<PY
import json
from pathlib import Path
report = json.loads(Path("$ROOT/initial3-gate.json").read_text(encoding="utf-8"))
print("true" if report["gate"]["pass"] else "false")
PY
)"

if [[ "$GATE_PASS" != "true" ]]; then
  echo "== Initial3 gate rejected v1; full7 is prohibited"
  exit 0
fi

if [[ ! -f "$ROOT/full7.json" ]]; then
  echo "== Initial3 gate passed; full7 eval start $(date -Is)"
  "$PY" "$CODE/saebench_sparse_probing_structured_dual_granularity.py" \
    --targets-json "$SCREEN_DIR/targets-structured-dual-granularity.json" \
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
  echo "== full7 eval done $(date -Is)"
fi

"$PY" - <<PY | tee "$ROOT/full7-decision.log"
import json
from pathlib import Path

root = Path("$ROOT")
payload = json.loads((root / "full7.json").read_text(encoding="utf-8"))
rows = {row["variant_key"]: row for row in payload["summary"]}
report = {
    "base_mean_acc": rows["base"]["mean_acc"],
    "candidate_mean_acc": rows["candidate"]["mean_acc"],
    "delta": rows["candidate"]["mean_acc"] - rows["base"]["mean_acc"],
    "candidate_strictly_above_0p9": rows["candidate"]["mean_acc"] > 0.9,
    "decision": (
        "allow-five-seed-and-readout-validation"
        if rows["candidate"]["mean_acc"] > 0.9
        else "stop-after-full7"
    ),
}
(root / "full7-decision.json").write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2))
PY
