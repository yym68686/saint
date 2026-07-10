#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-nexttoken-predictive-sae-v1}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-$MODEL_DIR/consolidated.00.pth}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
PRIOR_VALIDATION="${PRIOR_VALIDATION:-/autodl-fs/data/structured_nested_turn_sae_v1_20260710/cache-validation.json}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/root/autodl-tmp/v396_logcompanding_initial5/trained_sae-v396-logcompanding-relu.pt}"
ROOT="${ROOT:-/autodl-fs/data/structured_nexttoken_predictive_v1_screen_20260711}"
PREDICTION_LOSS_WEIGHT="${PREDICTION_LOSS_WEIGHT:-0.02}"
SCREEN_STEPS="${SCREEN_STEPS:-600}"
SMOKE_DIR="$ROOT/smoke"
SCREEN_DIR="$ROOT/screen_seed49"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
source /etc/network_turbo >/dev/null 2>&1 || true
mkdir -p "$ROOT"

"$PY" - <<PY | tee "$ROOT/preregistration.log"
import hashlib
import json
import math
import stat
import subprocess
from pathlib import Path

root = Path("$ROOT")
code = Path("$CODE")
cache = Path("$CACHE_DIR")
base = Path("$BASE_CHECKPOINT")
weights = Path("$MODEL_WEIGHTS")
validation_path = Path("$PRIOR_VALIDATION")
manifest_path = cache / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
validation = json.loads(validation_path.read_text(encoding="utf-8"))
if validation["status"] != "valid":
    raise SystemExit("Prior structured-cache validation is not valid")
if Path(validation["cache_dir"]).resolve() != cache.resolve():
    raise SystemExit("Prior validation references a different cache")
if cache.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
    raise SystemExit("Structured cache directory is writable")
branch = subprocess.check_output(
    ["git", "-C", str(code), "branch", "--show-current"], text=True
).strip()
commit = subprocess.check_output(
    ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
).strip()
dirty = subprocess.check_output(
    ["git", "-C", str(code), "status", "--porcelain"], text=True
).strip()
if branch != "scpg-nexttoken-predictive-sae-v1":
    raise SystemExit(f"Unexpected branch: {branch}")
if dirty:
    raise SystemExit(f"Remote code worktree is dirty: {dirty}")
if weights.stat().st_size != manifest["sources"]["model_weight_size_bytes"]:
    raise SystemExit("Model weight size differs from structured-cache manifest")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

predictor_rank = 64
d_model = 3072
predictor_parameters = (
    predictor_rank * d_model
    + predictor_rank
    + d_model * predictor_rank
    + d_model
)
base_parameters = 402_852_864
payload = {
    "experiment": "next-token predictive V396 SAE v1 warm-start screen",
    "registered_before_training_and_evaluation": True,
    "code_branch": branch,
    "code_commit": commit,
    "code_worktree_clean": True,
    "evidence_basis": {
        "frozen_gate_root": "/autodl-fs/data/structured_nexttoken_output_v1_gate_20260711",
        "frozen_candidate_mean_acc": 0.8272135416666666,
        "frozen_raw_mean_acc": 0.8235677083333334,
        "frozen_wrong_alignment_mean_acc": 0.8210069444444444,
        "frozen_candidate_minus_raw": 0.0036458333333332,
        "frozen_candidate_minus_wrong_alignment": 0.0062065972222222,
        "controlled_change": (
            "replace failed post-hoc rank calibration with a trainable low-rank "
            "prediction head whose loss changes formation of the unfrozen SAE trunk"
        ),
    },
    "data": {
        "cache_dir": str(cache),
        "cache_manifest_sha256": sha256(manifest_path),
        "cache_read_only": True,
        "prior_full_validation": str(validation_path),
        "prior_full_validation_sha256": sha256(validation_path),
        "layers_present": manifest["configuration"]["layers"],
        "training_layer": 22,
        "sample_boundaries_preserved": True,
        "token_ids_preserved": True,
        "actual_next_tokens_only": True,
        "same_owt_source_as_flat_baselines": True,
    },
    "initialization": {
        "base_checkpoint": str(base),
        "base_checkpoint_sha256": sha256(base),
        "base_architecture": "V396 LogCompanding ReLU SAE",
        "base_checkpoint_full7_mean_acc": 0.884728,
        "sae_trunk_unfrozen": True,
        "predictor_seed": 490396,
        "training_seed": 49,
        "evaluation_seed": 42,
    },
    "model_weights": {
        "path": str(weights),
        "size_bytes": weights.stat().st_size,
        "sha256_from_cache_manifest": manifest["sources"]["model_weight_sha256"],
        "target": "L2-normalized frozen output.weight row for actual next token",
    },
    "architecture": {
        "predictor": "LayerNorm(reconstruction residual) -> Linear(3072,64) -> SiLU -> Linear(64,3072)",
        "predictor_rank": predictor_rank,
        "base_parameter_count": base_parameters,
        "new_predictor_parameter_count": predictor_parameters,
        "trainable_parameter_count_each": base_parameters + predictor_parameters,
        "exposed_feature_count_each": 65536,
        "prediction_loss": "one minus cosine similarity to actual next-token output direction",
        "prediction_loss_weight": float("$PREDICTION_LOSS_WEIGHT"),
        "loss_weight_selected_from_label_free_loss_scale": True,
        "target_initial_weighted_prediction_to_reconstruction_ratio": "0.1 to 0.5 in smoke",
    },
    "variants": {
        "candidate": "true next-token target, prediction gradient reaches SAE trunk",
        "detach_control": "true next-token target, identical trainable head, prediction input detached from SAE trunk",
        "wrong_alignment": "exact same target marginal permuted across different samples, prediction gradient reaches SAE trunk",
    },
    "fairness": {
        "same_base_checkpoint": True,
        "same_initial_trunk_tensors": True,
        "same_initial_predictor_tensors": True,
        "same_trainable_parameter_count": True,
        "same_exposed_feature_count": True,
        "same_cache": True,
        "same_data_order": True,
        "same_optimizer": True,
        "same_steps": True,
        "same_batch_tokens": True,
        "same_target_marginal_candidate_and_wrong_control": True,
        "uses_saebench_labels_for_training": False,
        "uses_eval_split_for_training": False,
        "uses_one_vs_rest_targets_for_training": False,
        "uses_mean_diff_selection_for_training": False,
        "uses_test_feedback_for_training": False,
    },
    "screen": {
        "steps": int("$SCREEN_STEPS"),
        "batch_tokens": 256,
        "max_transitions_per_sample": 64,
        "datasets": [
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "fancyzhx/ag_news",
        ],
        "candidate_minus_best_matched_control": ">= 0.005",
        "candidate_minus_wrong_alignment": ">= 0.002",
        "minimum_dataset_delta_vs_best_matched_control": ">= -0.01",
        "long_train_only_after_pass": True,
        "full7_prohibited_in_screen_runner": True,
    },
    "family": "next-token predictive auxiliary SAE",
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

"$PY" - <<PY | tee "$ROOT/numerical-unit-test.log"
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

source = Path("$CODE/train_structured_nexttoken_predictive_sae.py")
spec = importlib.util.spec_from_file_location("nexttoken_train", source)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot import {source}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

torch.manual_seed(7)
d_model = 12
n_latents = 16
base_state = {
    "b_pre": torch.randn(d_model),
    "encoder.weight": torch.randn(n_latents, d_model) * 0.05,
    "encoder.bias": torch.zeros(n_latents),
    "decoder.weight": torch.randn(d_model, n_latents) * 0.05,
    "raw_beta": torch.full((n_latents,), -1.2586915),
    "log_gain": torch.zeros(n_latents),
    "init_beta": 0.25,
    "max_beta": 4.0,
    "max_log_gain": 2.0,
}
candidate = module.NextTokenPredictiveV396SAE(base_state, 4, 123)
detach = module.NextTokenPredictiveV396SAE(base_state, 4, 123)
initial_equal = all(
    torch.equal(left, right)
    for left, right in zip(candidate.parameters(), detach.parameters(), strict=True)
)
x = torch.randn(8, d_model)
target = F.normalize(torch.randn(8, d_model), dim=-1)

candidate.zero_grad(set_to_none=True)
candidate_out = candidate(x, False)
candidate_loss = 1.0 - (candidate_out["prediction"] * target).sum(dim=-1).mean()
candidate_loss.backward()
candidate_trunk_grad = module.grad_norm(candidate.trunk_parameters())
candidate_predictor_grad = module.grad_norm(candidate.predictor_parameters())

detach.zero_grad(set_to_none=True)
detach_out = detach(x, True)
detach_loss = 1.0 - (detach_out["prediction"] * target).sum(dim=-1).mean()
detach_loss.backward()
detach_trunk_grad = module.grad_norm(detach.trunk_parameters())
detach_predictor_grad = module.grad_norm(detach.predictor_parameters())

sample_ids = torch.repeat_interleave(torch.arange(8), 4)
permutation = module.sample_separating_permutation(sample_ids, 456)
exported = candidate.export_state()
report = {
    "initial_parameter_tensors_exactly_equal": initial_equal,
    "parameter_count_equal": module.parameter_count(candidate) == module.parameter_count(detach),
    "candidate_prediction_gradient_reaches_trunk": candidate_trunk_grad > 0,
    "candidate_prediction_gradient_reaches_head": candidate_predictor_grad > 0,
    "detach_prediction_gradient_does_not_reach_trunk": detach_trunk_grad == 0,
    "detach_prediction_gradient_reaches_head": detach_predictor_grad > 0,
    "wrong_permutation_has_zero_fixed_pairs": int(
        (permutation == torch.arange(permutation.numel())).sum().item()
    ) == 0,
    "wrong_permutation_has_zero_same_sample_pairs": int(
        (sample_ids[permutation] == sample_ids).sum().item()
    ) == 0,
    "candidate_trunk_grad_norm": candidate_trunk_grad,
    "candidate_predictor_grad_norm": candidate_predictor_grad,
    "detach_trunk_grad_norm": detach_trunk_grad,
    "detach_predictor_grad_norm": detach_predictor_grad,
    "export_contains_standard_sae_keys": {
        "b_pre", "encoder.weight", "encoder.bias", "decoder.weight"
    }.issubset(exported),
    "export_contains_v396_shape_keys": {
        "causal.raw_beta", "causal.log_gain", "causal.max_beta", "causal.max_log_gain"
    }.issubset(exported),
    "export_contains_predictor_keys": {
        "nexttoken.predictor_down_weight", "nexttoken.predictor_up_weight"
    }.issubset(exported),
}
report["pass"] = all(value for value in report.values() if isinstance(value, bool))
Path("$ROOT/numerical-unit-test.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Numerical unit test failed")
PY

if [[ ! -f "$SMOKE_DIR/train-summary-nexttoken-predictive.json" ]]; then
  mkdir -p "$SMOKE_DIR"
  "$PY" "$CODE/train_structured_nexttoken_predictive_sae.py" \
    --cache-dir "$CACHE_DIR" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --model-weights "$MODEL_WEIGHTS" \
    --output-dir "$SMOKE_DIR" \
    --layer 22 \
    --predictor-rank 64 \
    --predictor-seed 490396 \
    --steps 2 \
    --batch-tokens 64 \
    --max-transitions-per-sample 16 \
    --train-fraction 0.95 \
    --seed 49 \
    --trunk-lr 1e-6 \
    --predictor-lr 1e-5 \
    --prediction-loss-weight "$PREDICTION_LOSS_WEIGHT" \
    --l1-coeff 1e-6 \
    --beta-anchor-coeff 1e-3 \
    --gain-l2-coeff 1e-4 \
    --grad-clip 1 \
    --log-every 1 \
    --no-save-checkpoints \
    --device cuda \
    > "$ROOT/smoke-train.log" 2>&1
fi

"$PY" - <<PY | tee "$ROOT/smoke-gate.log"
import json
import math
from pathlib import Path

summary = json.loads(
    Path("$SMOKE_DIR/train-summary-nexttoken-predictive.json").read_text()
)
variants = summary["variants"]
counts = {row["parameter_count"] for row in variants.values()}
ratios = {}
for key, row in variants.items():
    first = row["logs"][0]
    ratios[key] = (
        float("$PREDICTION_LOSS_WEIGHT") * first["prediction_loss"]
        / first["reconstruction_loss"]
    )
report = {
    "parameter_counts_equal": len(counts) == 1,
    "parameter_count_each": next(iter(counts)),
    "expected_parameter_count": 403249216,
    "parameter_count_matches_expected": next(iter(counts)) == 403249216,
    "predictor_parameter_count": summary["new_predictor_parameter_count"],
    "predictor_parameter_count_matches_expected": summary["new_predictor_parameter_count"] == 396352,
    "all_trunks_updated": all(row["trunk_parameter_max_delta"] > 0 for row in variants.values()),
    "all_predictors_updated": all(row["predictor_parameter_max_delta"] > 0 for row in variants.values()),
    "all_logged_losses_finite": all(
        all(
            math.isfinite(entry[name])
            for entry in row["logs"]
            for name in (
                "loss", "reconstruction_loss", "prediction_loss",
                "explained_variance", "trunk_grad_norm", "predictor_grad_norm",
            )
        )
        for row in variants.values()
    ),
    "all_logged_gradients_positive": all(
        all(entry["trunk_grad_norm"] > 0 and entry["predictor_grad_norm"] > 0 for entry in row["logs"])
        for row in variants.values()
    ),
    "wrong_alignment_fixed_pairs_zero": variants["wrong_alignment"]["wrong_fixed_pair_count"] == 0,
    "wrong_alignment_same_sample_pairs_zero": variants["wrong_alignment"]["wrong_same_sample_pair_count"] == 0,
    "weighted_prediction_to_reconstruction_ratio": ratios,
    "candidate_ratio_in_registered_range": 0.1 <= ratios["candidate"] <= 0.5,
}
report["pass"] = all(value for value in report.values() if isinstance(value, bool))
Path("$ROOT/smoke-gate.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Implementation smoke failed; formal screen prohibited")
PY

if [[ ! -f "$SCREEN_DIR/train-summary-nexttoken-predictive.json" ]]; then
  mkdir -p "$SCREEN_DIR"
  echo "== matched screen start $(date -Is)"
  "$PY" "$CODE/train_structured_nexttoken_predictive_sae.py" \
    --cache-dir "$CACHE_DIR" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --model-weights "$MODEL_WEIGHTS" \
    --output-dir "$SCREEN_DIR" \
    --layer 22 \
    --predictor-rank 64 \
    --predictor-seed 490396 \
    --steps "$SCREEN_STEPS" \
    --batch-tokens 256 \
    --max-transitions-per-sample 64 \
    --train-fraction 0.95 \
    --seed 49 \
    --trunk-lr 1e-6 \
    --predictor-lr 1e-5 \
    --prediction-loss-weight "$PREDICTION_LOSS_WEIGHT" \
    --l1-coeff 1e-6 \
    --beta-anchor-coeff 1e-3 \
    --gain-l2-coeff 1e-4 \
    --grad-clip 1 \
    --log-every 100 \
    --save-checkpoints \
    --device cuda \
    > "$ROOT/screen-train.log" 2>&1
  echo "== matched screen done $(date -Is)"
fi

"$PY" - <<PY | tee "$ROOT/training-integrity-gate.log"
import json
import math
from pathlib import Path

summary = json.loads(
    Path("$SCREEN_DIR/train-summary-nexttoken-predictive.json").read_text()
)
variants = summary["variants"]
counts = {row["parameter_count"] for row in variants.values()}
report = {
    "parameter_counts_equal": len(counts) == 1,
    "same_exposed_feature_count": summary["exposed_feature_count_each"] == 65536,
    "all_trunks_updated": all(row["trunk_parameter_max_delta"] > 0 for row in variants.values()),
    "all_predictors_updated": all(row["predictor_parameter_max_delta"] > 0 for row in variants.values()),
    "all_final_gradients_positive": all(
        row["logs"][-1]["trunk_grad_norm"] > 0
        and row["logs"][-1]["predictor_grad_norm"] > 0
        for row in variants.values()
    ),
    "all_metrics_finite": all(
        all(
            math.isfinite(entry[name])
            for entry in row["logs"]
            for name in (
                "loss", "reconstruction_loss", "prediction_loss",
                "explained_variance", "active_features_per_token",
                "dead_feature_ratio_seen", "trunk_grad_norm", "predictor_grad_norm",
            )
        )
        for row in variants.values()
    ),
    "wrong_alignment_fixed_pairs_zero": variants["wrong_alignment"]["wrong_fixed_pair_count"] == 0,
    "wrong_alignment_same_sample_pairs_zero": variants["wrong_alignment"]["wrong_same_sample_pair_count"] == 0,
    "cache_read_only": summary["cache_read_only"],
    "no_label_or_eval_training_signal": all(
        not summary["fairness"][key]
        for key in (
            "uses_saebench_labels_for_training",
            "uses_eval_split_for_training",
            "uses_one_vs_rest_targets_for_training",
            "uses_mean_diff_selection_for_training",
            "uses_test_feedback_for_training",
        )
    ),
}
report["pass"] = all(value for value in report.values() if isinstance(value, bool))
Path("$ROOT/training-integrity-gate.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Formal training integrity gate failed; evaluation prohibited")
PY

if [[ ! -f "$ROOT/initial3.json" ]]; then
  "$PY" "$CODE/saebench_sparse_probing_v396_causal_suite.py" \
    --targets-json "$SCREEN_DIR/targets-nexttoken-predictive.json" \
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
fi

"$PY" "$CODE/analyze_nexttoken_predictive_gate.py" \
  --eval-json "$ROOT/initial3.json" \
  --output-json "$ROOT/initial3-gate.json" \
  --output-md "$ROOT/initial3-gate.md" \
  | tee "$ROOT/initial3-gate.log"

"$PY" - <<PY | tee "$ROOT/screen-decision.log"
import json
from pathlib import Path

gate = json.loads(Path("$ROOT/initial3-gate.json").read_text())
report = {
    "screen_gate_pass": gate["gate"]["pass"],
    "decision": gate["decision"],
    "full7_ran": False,
}
Path("$ROOT/screen-decision.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
