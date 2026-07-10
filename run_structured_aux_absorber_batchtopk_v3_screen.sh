#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-/autodl-fs/data/worktrees/saint-structured-aux-absorber-batchtopk-v3}"
PY="${PY:-/root/.cache/pypoetry/virtualenvs/llama3-interpretability-sae-d40co3fS-py3.12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/saint/llama_3.2-3B_model/original}"
CACHE_DIR="${CACHE_DIR:-/autodl-fs/data/structured_activation_cache_owt50k_l20-l23_v1}"
OLD_MEAN_PATH="${OLD_MEAN_PATH:-/root/autodl-tmp/activation_outputs_mean.pt}"
PRIOR_VALIDATION="${PRIOR_VALIDATION:-/autodl-fs/data/structured_nested_turn_sae_v1_20260710/cache-validation.json}"
ROOT="${ROOT:-/autodl-fs/data/structured_aux_absorber_batchtopk_v3_20260710}"
SMOKE_DIR="$ROOT/smoke"
SCREEN_DIR="$ROOT/screen_seed43"

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
source /etc/network_turbo >/dev/null 2>&1 || true
mkdir -p "$ROOT"

"$PY" - <<PY | tee "$ROOT/preregistration.log"
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

root = Path("$ROOT")
code = Path("$CODE")
cache = Path("$CACHE_DIR")
prior_validation = Path("$PRIOR_VALIDATION")
validation = json.loads(prior_validation.read_text(encoding="utf-8"))
if validation["status"] != "valid":
    raise SystemExit("Prior full cache validation is not valid")
if Path(validation["cache_dir"]).resolve() != cache.resolve():
    raise SystemExit("Prior validation references a different cache")
if cache.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
    raise SystemExit("Structured cache directory is writable")

payload = {
    "experiment": "hidden auxiliary-absorber BatchTopK SAE v3 screen",
    "registered_before_training_and_evaluation": True,
    "code_branch": subprocess.check_output(
        ["git", "-C", str(code), "branch", "--show-current"], text=True
    ).strip(),
    "code_branch_expected": "scpg-structured-aux-absorber-batchtopk-v3",
    "code_commit": subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip(),
    "method_basis": {
        "external_primary_source": "https://arxiv.org/abs/2606.28548",
        "measured_predecessor": "sample-nested shared-dictionary BatchTopK v2",
        "predecessor_full_readout_initial3_delta": -0.0156684027777777,
        "diagnostic_only_outer_readout": {
            "candidate_mean_acc": 0.7993489583333333,
            "same_index_base_suffix_mean_acc": 0.7862847222222222,
            "delta": 0.013064236111111072,
            "bias_set3_delta": 0.023958333333333304,
            "amazon_category_delta": 0.004166666666666652,
            "ag_news_delta": 0.01106770833333326,
        },
        "label_free_partition_evidence": {
            "base_prefix_event_share": 0.49840279403544147,
            "base_suffix_event_share": 0.5015972059645586,
            "candidate_inner_event_share": 0.24923703047937074,
            "candidate_outer_event_share": 0.7507629695206293,
        },
        "controlled_mechanism_change": (
            "pre-register the directly supervised inner partition as a hidden "
            "sample-mean absorber and expose only the outer partition; training "
            "mechanism and all optimization hyperparameters remain unchanged"
        ),
    },
    "data": {
        "cache_dir": str(cache),
        "cache_manifest_sha256": hashlib.sha256(
            (cache / "manifest.json").read_bytes()
        ).hexdigest(),
        "cache_read_only": True,
        "prior_full_validation": str(prior_validation),
        "prior_full_validation_sha256": hashlib.sha256(
            prior_validation.read_bytes()
        ).hexdigest(),
        "same_owt_source_as_flat_baselines": True,
        "sample_boundaries_preserved": True,
        "layer": 22,
    },
    "architecture": {
        "total_features": 65536,
        "inner_sample_features": 32768,
        "outer_features": 32768,
        "top_k": 64,
        "parameter_count_each": 402721792,
        "exposed_feature_count_each": 32768,
        "shared_encoder_decoder": True,
        "token_reconstruction_uses_full_dictionary": True,
        "sample_mean_reconstruction_uses_inner_partition": True,
        "inner_partition_hidden_from_sparse_readout": True,
        "outer_partition_is_only_sparse_readout": True,
        "nested_loss_weight": 1.0,
        "l1_training_weight": 0.0,
        "threshold_calibration_batches": 128,
        "frozen_bank_at_inference": False,
    },
    "fairness": {
        "ordinary_batchtopk_control_trained_from_scratch": True,
        "candidate_trained_from_scratch": True,
        "training_seed": 43,
        "initialization_seed": 430396,
        "evaluation_seed": 42,
        "same_optimizer": True,
        "same_data_order": True,
        "same_epochs": True,
        "same_batchtopk_k": True,
        "same_threshold_calibration": True,
        "same_trainable_parameter_count": True,
        "same_exposed_feature_count": True,
        "uses_saebench_labels_for_training": False,
        "uses_eval_split_for_training": False,
        "uses_one_vs_rest_targets_for_training": False,
        "uses_mean_diff_selection_for_training": False,
        "uses_test_feedback_for_training": False,
    },
    "screen": {
        "epochs": 3,
        "initial3": [
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "fancyzhx/ag_news",
        ],
        "candidate_minus_base_mean_acc": ">= 0.005",
        "minimum_per_dataset_delta": ">= -0.01",
        "long_train_only_after_pass": True,
        "full7_prohibited_in_screen_runner": True,
    },
    "family": "sample-nested hidden-absorber shared-dictionary",
    "family_version": 3,
    "maximum_family_versions": 3,
    "fresh_confirmation_run": True,
    "does_not_reuse_v2_checkpoint": True,
}
path = root / "preregistration.json"
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit(f"Refusing to overwrite different preregistration: {path}")
else:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

"$PY" - <<PY | tee "$ROOT/cache-compatibility.log"
import json
from pathlib import Path
import torch
import torch.nn.functional as F

new = torch.load(
    Path("$CACHE_DIR") / "mean-layer-22.pt",
    map_location="cpu",
    weights_only=True,
).float()
old = torch.load(
    Path("$OLD_MEAN_PATH"), map_location="cpu", weights_only=True
).float()
delta = new - old
report = {
    "max_abs_difference": float(delta.abs().max().item()),
    "mean_abs_difference": float(delta.abs().mean().item()),
    "rmse": float(delta.square().mean().sqrt().item()),
    "cosine_similarity": float(
        F.cosine_similarity(new.unsqueeze(0), old.unsqueeze(0)).item()
    ),
}
report["pass"] = (
    report["mean_abs_difference"] <= 1e-5
    and report["cosine_similarity"] >= 0.999999
)
Path("$ROOT/cache-compatibility.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Structured cache is not compatible with old L22 data")
PY

"$PY" - <<PY | tee "$ROOT/numerical-unit-test.log"
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

source = Path("$CODE/train_structured_nested_batchtopk_sae.py")
spec = importlib.util.spec_from_file_location("nested_batchtopk_train", source)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot import {source}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

torch.manual_seed(9)
d_model = 12
n_latents = 16
inner_features = 8
top_k = 2
b_pre = torch.linspace(-0.2, 0.2, d_model)
base = module.BatchTopKFromScratchSAE(
    b_pre=b_pre,
    n_latents=n_latents,
    top_k=top_k,
    seed=1234,
)
candidate = module.SampleNestedBatchTopKSAE(
    b_pre=b_pre,
    n_latents=n_latents,
    inner_features=inner_features,
    top_k=top_k,
    seed=1234,
    nested_loss_weight=1.0,
)

initial_tensors_equal = all(
    torch.equal(left, right)
    for left, right in (
        (base.b_pre, candidate.b_pre),
        (base.encoder_weight, candidate.encoder_weight),
        (base.encoder_bias, candidate.encoder_bias),
        (base.decoder_weight, candidate.decoder_weight),
    )
)
parameter_count_equal = (
    module.parameter_count(base) == module.parameter_count(candidate)
)

x = torch.randn(7, d_model)
sample_index = torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.long)
lengths = torch.tensor([3, 2, 2], dtype=torch.long)
out = candidate(x, sample_index, lengths)
token_positive = int((out["z_token"] > 0).sum().item())
inner_positive = int((out["z_semantic"] > 0).sum().item())

candidate.zero_grad(set_to_none=True)
inner_loss = candidate.reconstruction_losses(out, x, lengths)["semantic"]
inner_loss.backward()
inner_encoder_grad = float(
    candidate.encoder_weight.grad[:inner_features].abs().max().item()
)
outer_encoder_grad = float(
    candidate.encoder_weight.grad[inner_features:].abs().max().item()
)
inner_decoder_grad = float(
    candidate.decoder_weight.grad[:, :inner_features].abs().max().item()
)
outer_decoder_grad = float(
    candidate.decoder_weight.grad[:, inner_features:].abs().max().item()
)

dead_mask = torch.ones(n_latents, dtype=torch.bool)
base_out = base(x, sample_index, lengths)
aux = base.auxiliary_reconstruction(
    base_out,
    dead_mask,
    k_aux=4,
    sample_index=sample_index,
)
h_masked = base_out["h_token"] * dead_mask.to(base_out["h_token"].dtype)
values, indices = torch.topk(torch.relu(h_masked), k=4, dim=1)
expected_sparse = torch.zeros_like(h_masked).scatter_(1, indices, values)
expected_aux = F.linear(expected_sparse, base.decoder_weight)

cache = module.StructuredActivationCache(
    cache_dir=Path("$CACHE_DIR"),
    layer=22,
    batch_samples=4,
    train_fraction=0.95,
    seed=42,
)
batch_a = next(cache.iter_batches(0, "train"))
batch_b = next(cache.iter_batches(0, "train"))
data_order_reproducible = (
    torch.equal(batch_a.activations, batch_b.activations)
    and torch.equal(batch_a.sample_index, batch_b.sample_index)
    and torch.equal(batch_a.lengths, batch_b.lengths)
)

exported = base.export_state(torch.tensor(0.125))
report = {
    "initial_tensors_exactly_equal": initial_tensors_equal,
    "parameter_count_equal": parameter_count_equal,
    "base_parameter_count": module.parameter_count(base),
    "candidate_parameter_count": module.parameter_count(candidate),
    "token_batchtopk_budget_holds": token_positive <= top_k * x.shape[0],
    "inner_batchtopk_budget_holds": inner_positive <= top_k * lengths.numel(),
    "token_positive_count": token_positive,
    "inner_positive_count": inner_positive,
    "inner_encoder_gradient_nonzero": inner_encoder_grad > 0,
    "inner_decoder_gradient_nonzero": inner_decoder_grad > 0,
    "outer_encoder_gradient_zero_for_inner_loss": outer_encoder_grad == 0,
    "outer_decoder_gradient_zero_for_inner_loss": outer_decoder_grad == 0,
    "inner_encoder_gradient_max": inner_encoder_grad,
    "outer_encoder_gradient_max": outer_encoder_grad,
    "inner_decoder_gradient_max": inner_decoder_grad,
    "outer_decoder_gradient_max": outer_decoder_grad,
    "auxiliary_reconstruction_excludes_b_pre": torch.equal(aux, expected_aux),
    "data_order_reproducible": data_order_reproducible,
    "export_has_standard_batchtopk_keys": set(exported) == {
        "b_pre",
        "encoder.weight",
        "encoder.bias",
        "decoder.weight",
        "threshold",
    },
}
report["pass"] = all(
    value for value in report.values() if isinstance(value, bool)
)
Path("$ROOT/numerical-unit-test.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Nested BatchTopK numerical unit test failed")
PY

if [[ ! -f "$SMOKE_DIR/train-summary-structured-sample-nested-batchtopk.json" ]]; then
  mkdir -p "$SMOKE_DIR"
  "$PY" "$CODE/train_structured_nested_batchtopk_sae.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$SMOKE_DIR" \
    --layer 22 \
    --n-total 65536 \
    --inner-features 32768 \
    --readout-mode outer \
    --top-k 64 \
    --nested-loss-weight 1.0 \
    --epochs 1 \
    --batch-samples 4 \
    --max-train-batches 2 \
    --max-validation-batches 1 \
    --threshold-calibration-batches 2 \
    --train-fraction 0.95 \
    --seed 43 \
    --initialization-seed 430396 \
    --lr 5e-5 \
    --beta1 0.85 \
    --beta2 0.9999 \
    --optimizer-eps 6.25e-10 \
    --l1-coeff 0 \
    --k-aux 2048 \
    --aux-loss-coeff 0.03125 \
    --dead-steps-threshold 0 \
    --log-every 1 \
    --device cuda \
    > "$ROOT/smoke-train.log" 2>&1
fi

"$PY" - <<PY | tee "$ROOT/smoke-gate.log"
import json
import math
from pathlib import Path

summary = json.loads(
    Path("$SMOKE_DIR/train-summary-structured-sample-nested-batchtopk.json").read_text()
)
base = summary["base_result"]
candidate = summary["candidate_result"]
base_history = base["history"]
candidate_history = candidate["history"]
targets = json.loads(Path(summary["targets_json"]).read_text())
report = {
    "parameter_matched": summary["parameter_matched"],
    "base_parameter_count": summary["base_parameter_count"],
    "candidate_parameter_count": summary["candidate_parameter_count"],
    "exposed_feature_count": summary["exposed_feature_count"],
    "outer_readout_has_32768_features": (
        summary["exposed_feature_count"] == 32768
    ),
    "both_targets_use_identical_outer_readout": all(
        target["feature_start"] == 32768
        and target["feature_end"] == 65536
        for target in targets
    ),
    "base_threshold_finite_positive": (
        math.isfinite(base["threshold"]) and base["threshold"] > 0
    ),
    "candidate_threshold_finite_positive": (
        math.isfinite(candidate["threshold"])
        and candidate["threshold"] > 0
    ),
    "nested_module_gradient_nonzero": max(
        row["nested_module_grad"] for row in candidate_history
    ) > 0,
    "nested_module_parameter_delta_nonzero": (
        candidate["nested_module_parameter_probe_max_delta"] > 0
    ),
    "base_batchtopk_l0_bounded": all(
        0 < row["active_token"] <= 64.0001 for row in base_history
    ),
    "candidate_batchtopk_l0_bounded": all(
        0 < row["active_token"] <= 64.0001 for row in candidate_history
    ),
    "inner_batchtopk_l0_bounded": all(
        0 < row["active_inner_mean"] <= 64.0001
        for row in candidate_history
    ),
    "inner_mean_loss_finite_positive": all(
        math.isfinite(row["inner_mean_recon"])
        and row["inner_mean_recon"] > 0
        for row in candidate_history
    ),
    "token_loss_finite_positive": all(
        math.isfinite(row["token_recon"]) and row["token_recon"] > 0
        for row in candidate_history
    ),
}
report["pass"] = all(
    value for value in report.values() if isinstance(value, bool)
)
Path("$ROOT/smoke-gate.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["pass"]:
    raise SystemExit("Nested BatchTopK implementation smoke failed")
PY

if [[ ! -f "$SCREEN_DIR/train-summary-structured-sample-nested-batchtopk.json" ]]; then
  mkdir -p "$SCREEN_DIR"
  echo "== matched three-epoch screen start $(date -Is)"
  "$PY" "$CODE/train_structured_nested_batchtopk_sae.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$SCREEN_DIR" \
    --layer 22 \
    --n-total 65536 \
    --inner-features 32768 \
    --readout-mode outer \
    --top-k 64 \
    --nested-loss-weight 1.0 \
    --epochs 3 \
    --batch-samples 32 \
    --max-train-batches 0 \
    --max-validation-batches 0 \
    --threshold-calibration-batches 128 \
    --train-fraction 0.95 \
    --seed 43 \
    --initialization-seed 430396 \
    --lr 5e-5 \
    --beta1 0.85 \
    --beta2 0.9999 \
    --optimizer-eps 6.25e-10 \
    --l1-coeff 0 \
    --k-aux 2048 \
    --aux-loss-coeff 0.03125 \
    --dead-steps-threshold 0 \
    --log-every 100 \
    --device cuda \
    > "$ROOT/screen-train.log" 2>&1
  echo "== matched three-epoch screen done $(date -Is)"
fi

if [[ ! -f "$ROOT/initial3.json" ]]; then
  "$PY" "$CODE/saebench_sparse_probing_structured_batchtopk.py" \
    --targets-json "$SCREEN_DIR/targets-structured-sample-nested-batchtopk.json" \
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

"$PY" "$CODE/analyze_structured_batchtopk_gate.py" \
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
    "decision": (
        "authorize-separately-preregistered-long-train"
        if gate["gate"]["pass"]
        else "stop-v3-and-close-family-before-long-train"
    ),
    "full7_ran": False,
}
Path("$ROOT/screen-decision.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
