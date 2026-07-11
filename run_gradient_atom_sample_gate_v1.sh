#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/root/saint-gradient-atom-sample-gate-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
STRUCTURED_CACHE="${STRUCTURED_CACHE:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
FLAT_CACHE="${FLAT_CACHE:-/root/autodl-tmp/activation_outputs_batched}"
V396="${V396:-/root/autodl-tmp/v396_logcompanding_initial5/trained_sae-v396-logcompanding-relu.pt}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/root/autodl-tmp/saebench_sparse_probing_all_architectures.py}"
ROOT="${ROOT:-/autodl-fs/data/gradient_atom_sample_gate_v1_20260711}"
EXTRACTION="$ROOT/owt-document-gradients.pt"
HEADS="$ROOT/gradient-atom-heads.pt"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
mkdir -p "$ROOT"

"$PY" - <<PY | tee "$ROOT/preregistration.log"
import json
import subprocess
from pathlib import Path

payload = {
    "experiment": "activation-only document-gradient atom sample head gate v1",
    "status": "registered-before-gradient-extraction-and-saebench-evaluation",
    "branch": subprocess.check_output(
        ["git", "-C", "$CODE", "branch", "--show-current"], text=True
    ).strip(),
    "commit": subprocess.check_output(
        ["git", "-C", "$CODE", "rev-parse", "HEAD"], text=True
    ).strip(),
    "external_sources": {
        "gradient_atoms": "arXiv:2603.14665",
        "gradient_atoms_official_code_commit": "a83858557b411249097861f25d57eb13020eb334",
        "gradient_sae": "arXiv:2411.10397",
    },
    "novelty_vs_local_nexttoken_family": (
        "v1/v2 next-token SAEs optimized token prediction directly; this gate first "
        "decomposes per-document activation gradients into sparse functional atoms, "
        "then learns an activation-only student"
    ),
    "gradient_extraction": {
        "owt_samples": 1024,
        "sequence_length": 32,
        "source_layer": 22,
        "source_coordinate": "attention-normalized cache coordinate",
        "target": "mean self-supervised next-token cross-entropy",
        "microbatch_size": 2,
        "sample_seed": 42,
    },
    "atom_fit": {
        "train_samples": 768,
        "holdout_samples": 256,
        "preconditioned_gradient_dimensions": 256,
        "gradient_atoms": 256,
        "dictionary_alpha": 0.1,
        "dictionary_max_iter": 100,
        "ridge_scale": 0.01,
        "head_top_abs_k": 4,
    },
    "frozen_representations": [
        "V396 reference",
        "V396 with 256 registered slots replaced by activation-only gradient-atom student",
        "same slots with wrong-alignment student",
        "same slots with gradient-PCA student",
        "same slots with activation-PCA control",
        "same slots with random-orthogonal control",
        "head-only versions for diagnosis",
    ],
    "evaluation_requires_gradient": False,
    "unsupervised_gate": {
        "true_minus_wrong_holdout_centered_cosine": ">= 0.02",
        "atom_minus_random_gradient_coherence": ">= 0.01",
        "coefficient_zero_row_fraction": "<= 0.10",
    },
    "initial3_gate": {
        "candidate_minus_v396_mean_acc": ">= 0.005",
        "candidate_minus_best_matched_control": ">= 0.002",
        "minimum_dataset_delta_vs_v396": ">= -0.01",
        "train_end_to_end_sae_only_after_pass": True,
    },
    "family_limit": 3,
    "data": {
        "structured_cache": "$STRUCTURED_CACHE",
        "flat_cache": "$FLAT_CACHE",
        "v396_checkpoint": "$V396",
        "read_only": True,
        "same_owt_activation_source": True,
    },
    "uses_saebench_labels_for_training_signal": False,
    "uses_saebench_class_names_for_training_signal": False,
    "uses_eval_split_for_training_signal": False,
    "uses_mean_diff_for_training_signal": False,
    "uses_test_feedback_for_training_signal": False,
}
path = Path("$ROOT/preregistration.json")
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit(f"Refusing to overwrite different preregistration: {path}")
else:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ ! -f "$EXTRACTION" ]]; then
  "$PY" "$CODE/extract_document_activation_gradients.py" \
    --model-dir "$MODEL_DIR" \
    --cache-dir "$STRUCTURED_CACHE" \
    --output "$EXTRACTION" \
    --source-layer 22 \
    --sequence-length 32 \
    --sample-count 1024 \
    --sample-seed 42 \
    --microbatch-size 2 \
    --normalize-eps 1e-6 \
    --source-tolerance 0 \
    --dtype bfloat16 \
    --device cuda \
    > "$ROOT/gradient-extraction.log" 2>&1
fi

if [[ ! -f "$HEADS" ]]; then
  "$PY" "$CODE/fit_gradient_atom_sample_heads.py" \
    --extraction "$EXTRACTION" \
    --v396-checkpoint "$V396" \
    --flat-cache-dir "$FLAT_CACHE" \
    --output "$HEADS" \
    --train-count 768 \
    --atom-count 256 \
    --dict-alpha 0.1 \
    --dict-max-iter 100 \
    --ridge-scale 0.01 \
    --head-top-k 4 \
    --slot-seed 42026 \
    --slot-stat-tokens 32768 \
    --slot-stat-batch-size 64 \
    --random-seed 42 \
    --device cuda \
    > "$ROOT/atom-fit.log" 2>&1
fi

"$PY" - <<PY | tee "$ROOT/unsupervised-gate.log"
import json
from pathlib import Path

fit = json.loads(Path("${HEADS%.pt}.json").read_text(encoding="utf-8"))
prediction = fit["holdout_prediction"]
true_cosine = prediction["gradient_atom_student"]["centered_cosine_mean"]
wrong_cosine = prediction["wrong_alignment_student"]["centered_cosine_mean"]
coherence = fit["atom_coherence"]
report = {
    "true_holdout_centered_cosine": true_cosine,
    "wrong_holdout_centered_cosine": wrong_cosine,
    "true_minus_wrong_centered_cosine": true_cosine - wrong_cosine,
    "atom_gradient_coherence": coherence["atom_mean"],
    "random_gradient_coherence": coherence["random_mean"],
    "atom_minus_random_coherence": coherence["atom_mean"] - coherence["random_mean"],
    "coefficient_zero_row_fraction": fit["coefficient_zero_row_fraction"],
}
report["prediction_pass"] = report["true_minus_wrong_centered_cosine"] >= 0.02
report["coherence_pass"] = report["atom_minus_random_coherence"] >= 0.01
report["nonzero_pass"] = report["coefficient_zero_row_fraction"] <= 0.10
report["pass"] = all([
    report["prediction_pass"],
    report["coherence_pass"],
    report["nonzero_pass"],
])
report["decision"] = "allow-initial3" if report["pass"] else "stop-before-initial3"
Path("$ROOT/unsupervised-gate.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Unsupervised gradient-atom gate failed")
PY

if [[ ! -f "$ROOT/initial3.json" ]]; then
  "$PY" "$CODE/evaluate_gradient_atom_sample_heads.py" \
    --eval-script "$EVAL_SCRIPT" \
    --model-dir "$MODEL_DIR" \
    --v396-checkpoint "$V396" \
    --head-artifact "$HEADS" \
    --output-json "$ROOT/initial3.json" \
    --output-md "$ROOT/initial3.md" \
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
    --dtype bfloat16 \
    --device cuda \
    > "$ROOT/initial3.log" 2>&1
fi

"$PY" - <<PY | tee "$ROOT/initial3-gate.log"
import json
from pathlib import Path

payload = json.loads(Path("$ROOT/initial3.json").read_text(encoding="utf-8"))
summary = {row["representation"]: row for row in payload["summary"]}
candidate_name = "replace_gradient_atom_student"
control_names = [
    "replace_wrong_alignment_student",
    "replace_gradient_pca_student",
    "replace_activation_pca_control",
    "replace_random_orthogonal_control",
]
base = summary["v396_reference"]["mean_acc"]
candidate = summary[candidate_name]["mean_acc"]
best_control_name = max(control_names, key=lambda name: summary[name]["mean_acc"])
best_control = summary[best_control_name]["mean_acc"]
dataset_deltas = {}
for dataset_name, dataset in payload["datasets"].items():
    reps = dataset["representations"]
    dataset_deltas[dataset_name] = (
        reps[candidate_name]["aggregate"]["mean_acc"]
        - reps["v396_reference"]["aggregate"]["mean_acc"]
    )
report = {
    "v396_mean_acc": base,
    "candidate_mean_acc": candidate,
    "best_control_name": best_control_name,
    "best_control_mean_acc": best_control,
    "delta_over_v396": candidate - base,
    "delta_over_best_control": candidate - best_control,
    "dataset_deltas_over_v396": dataset_deltas,
    "effect_pass": candidate - base >= 0.005,
    "control_pass": candidate - best_control >= 0.002,
    "dataset_pass": min(dataset_deltas.values()) >= -0.01,
}
report["pass"] = all([
    report["effect_pass"],
    report["control_pass"],
    report["dataset_pass"],
])
report["decision"] = (
    "allow-gradient-atom-sample-sae-v1-training"
    if report["pass"]
    else "stop-before-training"
)
Path("$ROOT/initial3-gate.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
