#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/root/saint-true-jacobian-workspace-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/root/autodl-tmp/saebench_sparse_probing_all_architectures.py}"
ROOT="${ROOT:-/autodl-fs/data/true_jacobian_workspace_v1_20260710}"
JACOBIAN_DIR="$ROOT/jacobian_n10"
RELU_CHECKPOINT="${RELU_CHECKPOINT:-/autodl-fs/data/structured_dual_granularity_v1_20260710/screen_seed42/trained_sae-structured-relu-base.pt}"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"

"$PY" - <<PY | tee "$ROOT/frozen-eval-preregistration.log"
import json
import subprocess
from pathlib import Path

payload = {
    "experiment": "true Jacobian frozen Initial3 signal gate",
    "status": "registered-before-frozen-evaluation",
    "code_commit": subprocess.check_output(
        ["git", "-C", "$CODE", "rev-parse", "HEAD"], text=True
    ).strip(),
    "representations": [
        "same-protocol ReLU control",
        "identity logit lens",
        "true Jacobian lens N=5",
        "true Jacobian lens N=10",
        "norm-matched random signed-permutation orthogonal control",
    ],
    "random_orthogonal_seed": 42026,
    "evaluation": {
        "datasets": [
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "fancyzhx/ag_news",
        ],
        "train_size": 512,
        "test_size": 128,
        "context_length": 128,
        "k_values": [1, 2, 5],
        "random_seed": 42,
    },
    "gate": {
        "j_n10_minus_logit_lens_mean_acc": ">= 0.005",
        "minimum_dataset_delta_vs_logit_lens": ">= -0.01",
        "j_n10_must_beat_random_orthogonal": True,
        "training_allowed_only_after_pass": True,
    },
    "uses_labels_to_construct_jacobian": False,
    "uses_class_names_to_construct_jacobian": False,
    "uses_test_feedback_to_construct_jacobian": False,
}
path = Path("$ROOT/frozen-eval-preregistration.json")
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit(f"Refusing to overwrite different preregistration: {path}")
else:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ ! -f "$ROOT/jacobian-convergence-gate.json" ]]; then
  echo "Missing Jacobian convergence gate" >&2
  exit 1
fi

if [[ ! -f "$ROOT/frozen-initial3.json" ]]; then
  "$PY" "$CODE/evaluate_true_jacobian_signal.py" \
    --eval-script "$EVAL_SCRIPT" \
    --model-dir "$MODEL_DIR" \
    --relu-checkpoint "$RELU_CHECKPOINT" \
    --jacobian-checkpoint "$JACOBIAN_DIR/average-jacobian-n10.pt" \
    --per-prompt-dir "$JACOBIAN_DIR/per_prompt" \
    --output-json "$ROOT/frozen-initial3.json" \
    --output-md "$ROOT/frozen-initial3.md" \
    --datasets \
      LabHC/bias_in_bios_class_set3 \
      canrager/amazon_reviews_mcauley_1and5 \
      fancyzhx/ag_news \
    --train-size 512 \
    --test-size 128 \
    --context-length 128 \
    --llm-batch-size 4 \
    --seq-batch-size 1 \
    --k-values 1 2 5 \
    --random-seed 42 \
    --random-control-seed 42026 \
    --dtype bfloat16 \
    --device cuda \
    > "$ROOT/frozen-initial3.log" 2>&1
fi

"$PY" - <<PY | tee "$ROOT/frozen-initial3-gate.log"
import json
from pathlib import Path

payload = json.loads(Path("$ROOT/frozen-initial3.json").read_text(encoding="utf-8"))
summary = {row["representation"]: row for row in payload["summary"]}
logit = summary["logit_lens"]["mean_acc"]
jacobian = summary["true_jacobian_lens_n10"]["mean_acc"]
random_control = summary["random_orthogonal_control"]["mean_acc"]
dataset_deltas = {}
for dataset_name, dataset in payload["datasets"].items():
    reps = dataset["representations"]
    dataset_deltas[dataset_name] = (
        reps["true_jacobian_lens_n10"]["aggregate"]["mean_acc"]
        - reps["logit_lens"]["aggregate"]["mean_acc"]
    )
report = {
    "logit_lens_mean_acc": logit,
    "true_jacobian_lens_n10_mean_acc": jacobian,
    "random_orthogonal_mean_acc": random_control,
    "delta_over_logit_lens": jacobian - logit,
    "delta_over_random_orthogonal": jacobian - random_control,
    "dataset_deltas_over_logit_lens": dataset_deltas,
    "overall_delta_at_least_0p005": jacobian - logit >= 0.005,
    "minimum_dataset_delta_at_least_minus_0p01": min(dataset_deltas.values()) >= -0.01,
    "beats_random_orthogonal": jacobian > random_control,
}
report["pass"] = all([
    report["overall_delta_at_least_0p005"],
    report["minimum_dataset_delta_at_least_minus_0p01"],
    report["beats_random_orthogonal"],
])
report["decision"] = "allow-jacobian-sae-v1-training" if report["pass"] else "stop-before-training"
Path("$ROOT/frozen-initial3-gate.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
