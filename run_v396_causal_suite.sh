#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/autodl-fs/data/v396_causal_attribution_20260710}"
CODE="${CODE:-/root/saint-v396-causal-attribution}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"

source /etc/network_turbo >/dev/null 2>&1 || true
mkdir -p "$ROOT"

"$PY" -c '
import json
import subprocess
from pathlib import Path

root = Path("'"$ROOT"'")
code = Path("'"$CODE"'")
payload = {
    "experiment": "V396 causal attribution warm-start five-seed suite",
    "registered_before_training": True,
    "code_branch": subprocess.check_output(
        ["git", "-C", str(code), "branch", "--show-current"],
        text=True,
    ).strip(),
    "code_commit": subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "seeds": [42, 43, 44, 45, 46],
    "steps": 600,
    "batch_tokens": 256,
    "fixed_betas": [0.10, 0.15, 0.20, 0.25],
    "variants": [
        "relu_finetune",
        "scaled_relu",
        "fixed_beta_0p10",
        "fixed_beta_0p15",
        "fixed_beta_0p20",
        "fixed_beta_0p25",
        "global_beta",
        "feature_beta",
        "full_beta_gain",
    ],
    "primary_gate": {
        "best_learned_minus_best_fixed_mean_acc": ">= 0.003",
        "paired_95pct_ci_low": "> 0",
        "top5_delta_vs_best_fixed": ">= 0",
        "top5_delta_vs_same_param_scaled_relu": ">= 0",
    },
}
path = root / "preregistration.json"
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit(f"Refusing to overwrite different preregistration: {path}")
else:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
' | tee "$ROOT/preregistration.log"

for seed in 42 43 44 45 46; do
  seed_dir="$ROOT/seed_${seed}"
  if [[ -f "$seed_dir/targets-v396-causal-seed${seed}.json" ]]; then
    echo "== seed ${seed} already complete"
    continue
  fi
  mkdir -p "$seed_dir"
  echo "== seed ${seed} train start $(date -Is)"
  "$PY" "$CODE/train_v396_causal_suite.py" \
    --output-dir "$seed_dir" \
    --seed "$seed" \
    --steps 600 \
    --batch-tokens 256 \
    --fixed-betas 0.10 0.15 0.20 0.25 \
    --log-every 100 \
    > "$seed_dir/train.log" 2>&1
  echo "== seed ${seed} train done $(date -Is)"
done

"$PY" -c '
import json
from pathlib import Path
root = Path("'"$ROOT"'")
targets = []
for seed in [42, 43, 44, 45, 46]:
    path = root / f"seed_{seed}" / f"targets-v396-causal-seed{seed}.json"
    targets.extend(json.loads(path.read_text(encoding="utf-8")))
missing = [target["checkpoint"] for target in targets if not Path(target["checkpoint"]).exists()]
if missing:
    raise SystemExit("Missing checkpoints:\n" + "\n".join(missing))
(root / "targets-v396-causal-5seed.json").write_text(
    json.dumps(targets, indent=2),
    encoding="utf-8",
)
print(json.dumps({"target_count": len(targets)}, indent=2))
'

echo "== full7 eval start $(date -Is)"
"$PY" "$CODE/saebench_sparse_probing_v396_causal_suite.py" \
  --targets-json "$ROOT/targets-v396-causal-5seed.json" \
  --output-json "$ROOT/v396-causal-5seed-full7.json" \
  --output-md "$ROOT/v396-causal-5seed-full7.md" \
  --datasets \
    LabHC/bias_in_bios_class_set1 LabHC/bias_in_bios_class_set2 LabHC/bias_in_bios_class_set3 \
    canrager/amazon_reviews_mcauley_1and5 canrager/amazon_reviews_mcauley_1and5_sentiment \
    fancyzhx/ag_news Helsinki-NLP/europarl \
  --train-size 512 \
  --test-size 128 \
  --context-length 128 \
  --llm-batch-size 4 \
  --sae-seq-batch-size 2 \
  --k-values 1 2 5 \
  --random-seed 42 \
  --dtype bfloat16 \
  --device cuda \
  > "$ROOT/eval.log" 2>&1
echo "== full7 eval done $(date -Is)"

"$PY" "$CODE/analyze_v396_causal_suite.py" \
  --eval-json "$ROOT/v396-causal-5seed-full7.json" \
  --output-json "$ROOT/v396-causal-decision.json" \
  --output-md "$ROOT/v396-causal-decision.md" \
  | tee "$ROOT/analysis.log"

echo "== suite complete $(date -Is)"
