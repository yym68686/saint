# PT Weight Audit For Family Benchmark

## Scope

This audit is for the current family-level working-point benchmark built on the finalized L22 split release:

- dataset release: `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260415/family_ontology_draft/benchmark_ready/family_dataset_release`
- benchmark families: 15
- core methods to evaluate: `topk`, `batchtopk`, `relu`, `gatedsae`, `jumprelu`, `dense`, `kernel`
- layer to test: `L22`

## Local Inventory

- The current repo `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae` does **not** contain tracked `.pt` weights.
- The canonical local reference weights are in `/Users/yanyuming/Downloads/GitHub/Thesis/exp/**`.

## Suggested Server Test Directory

Use one dedicated directory for the actual benchmark smoke tests and full runs:

`~/saint/weights/family_benchmark_l22/`

Recommended symlink names inside that directory:

- `topk_l22.pt`
- `batchtopk_l22.pt`
- `relu_l22.pt`
- `gatedsae_l22.pt`
- `jumprelu_l22.pt`
- `dense_l22.pt`
- `kernel_l22.pt`

This keeps the runner simple and separates benchmark-critical weights from old experiment artifacts.

## Must-Have Weights For The Current Benchmark

These seven are the weights you actually need for the current full family benchmark:

| Method | Architecture | Layer | Canonical local reference |
|---|---|---:|---|
| TopK 2024 | `topk` | 22 | `/Users/yanyuming/Downloads/GitHub/Thesis/exp/baseline/main/output/l22/trained_sae-main-l22.pt` |
| BatchTopK 2024 | `batchtopk` | 22 | `/Users/yanyuming/Downloads/GitHub/Thesis/exp/baseline/BatchTopK/output/l22/trained_sae-batchtopk-l22.pt` |
| ReLU SAE 2023 | `relu` | 22 | `/Users/yanyuming/Downloads/GitHub/Thesis/exp/baseline/relusae/output/l22/trained_sae-relu-l22.pt` |
| Gated SAE 2024 | `gatedsae` | 22 | `/Users/yanyuming/Downloads/GitHub/Thesis/exp/baseline/gatedsae/output/l22/trained_sae-gatedsae-l22.pt` |
| JumpReLU 2024 | `jumprelu` | 22 | `/Users/yanyuming/Downloads/GitHub/Thesis/exp/baseline/jumprelu/output/l22/trained_sae-jumprelu-l22.pt` |
| PLRDC SAE | `dense` | 22 | `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea1-dense-success/output/l22/trained_sae-dense-l22.pt` |
| SUR SAE | `kernel` | 22 | `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea5-kernel/output/l22/kernel.pt` |

## Server Files You Mentioned

Current server-side files:

- `./exp-sigreg-dict-reg-gradfix.pt`
- `./exp-dict-sigreg-repulsion-both.pt`
- `./activation_outputs_mean.pt`
- `./exp-imq-mmd-dict-reg.pt`
- `./4_d_model.pt`
- `./025_d_model.pt`
- `./trained_sae.pt`
- `./kernel-active.pt`

Interpretation against the local experiment inventory:

- `trained_sae.pt`
  - likely the main SUR / kernel model used in current chapter-4 style runs
  - probably corresponds to local canonical `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea5-kernel/output/l22/kernel.pt`
  - **must keep**
  - but the filename is historically ambiguous, so verify once before using it as `kernel_l22.pt`

- `kernel-active.pt`
  - likely corresponds to the `idea11-kernel-active` diagnostic variant
  - **not required** for the main 7-method family benchmark

- `exp-sigreg-dict-reg-gradfix.pt`
  - corresponds to SIGReg variant
  - local canonical reference:
    `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea3-lejepa/output/l22/exp-sigreg-dict-reg-gradfix.pt`
  - **optional only**

- `exp-dict-sigreg-repulsion-both.pt`
  - corresponds to SIGReg + RepReg combined variant
  - local canonical reference:
    `/Users/yanyuming/Downloads/GitHub/Thesis/exp/idea4-lejepa-leech/output/l22/exp-dict-sigreg-repulsion-both.pt`
  - **optional only**

- `exp-imq-mmd-dict-reg.pt`
  - corresponds to IMQ-MMD family
  - local thesis output references exist as `idea6-imq*` experiments
  - **optional only**

- `4_d_model.pt`
  - corresponds to IMQ `4/d` scale variant
  - aligns with `idea6-imq_4_d`
  - **optional only**

- `025_d_model.pt`
  - corresponds to IMQ `0.25/d` scale variant
  - aligns with `idea6-imq_025_d`
  - **optional only**

- `activation_outputs_mean.pt`
  - training-only `b_pre` initialization artifact
  - **not needed** for inference or benchmark evaluation because `load_sae_model` loads `b_pre` from the SAE state dict itself

## What Is Missing On The Server For A Full Benchmark

From your current server list, these benchmark-critical L22 weights are still missing:

- `trained_sae-main-l22.pt`
- `trained_sae-batchtopk-l22.pt`
- `trained_sae-relu-l22.pt`
- `trained_sae-gatedsae-l22.pt`
- `trained_sae-jumprelu-l22.pt`
- `trained_sae-dense-l22.pt`

If `trained_sae.pt` is verified to be the SUR model, then only `kernel_l22.pt` is already covered by the current server inventory.

## Deletion / Archival Recommendation

If your goal is **only** to run the current 15-family L22 benchmark, then:

### Keep in the active benchmark directory

- `trained_sae.pt` after verifying it is the SUR / kernel L22 model
- plus the six missing L22 benchmark weights once copied in

### Safe to move out of the active benchmark directory

- `activation_outputs_mean.pt`
- `exp-sigreg-dict-reg-gradfix.pt`
- `exp-dict-sigreg-repulsion-both.pt`
- `exp-imq-mmd-dict-reg.pt`
- `4_d_model.pt`
- `025_d_model.pt`
- `kernel-active.pt`

These are better treated as `archive/optional-experiments`, not as benchmark-critical weights.

### Only delete, not just archive, if you are sure you no longer need these analyses

- SIGReg supplementary analysis
- RepReg or SIGReg+RepReg supplementary analysis
- IMQ scale-sensitivity supplementary analysis
- kernel-active diagnostic analysis

## One Important Non-PT Reminder

The actual runner also needs the base Llama checkpoint:

- `llama_3.2-3B_model/original/consolidated.00.pth`

That file is required for activation recomputation, but it is a `.pth`, not a `.pt`.
