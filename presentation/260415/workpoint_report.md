# Working-Point Benchmark Report

## Protocol
- Source data: `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/ablation_datasets-dense`
- Score definition: `max` sequence aggregation, to match the existing AUC benchmark figures.
- Threshold calibration: deterministic 50/50 split on the control set only, with seed `42` and the same split reused across methods within each concept.
- Evaluation: realized control reject rate on the held-out control split; target reject rate on the full target set (target samples are never used for threshold selection).
- Fixed operating points: `1%`, `2%`, `5%`, `10%`, `20%`.
- Error bars / intervals: 95% bootstrap confidence intervals over concepts.

## Data Limits
- `JumpReLU SAE` is available only for `indian_politics`, so it is excluded from multi-concept comparison figures.
- `SUR SAE` (`kernel`) does not cover `female_subjects` or `photo_captions`, so the SUR panel is restricted to `football` and `indian_politics`.
- `ReLU SAE` and `Gated SAE` are unavailable for `canadian_political` and `female_subjects`, so the mainstream panel uses `football`, `indian_politics`, and `photo_captions`.
- Several methods have highly zero-inflated control-score distributions, so with the strict `score > tau` decision rule some concepts cannot realize exactly 5% on held-out control data; this is why a few realized control rates sit below the nominal target.
- The per-concept CSV covers every method found under `ablation_datasets-dense`, but the figures intentionally use only method subsets with shared concept coverage so that the comparisons remain valid under the thesis plotting requirement.

## Output Files
- Main figure: `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260415/pic/fixed_control_5_target_reject_rate_panels.png`
- Operating curves: `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260415/pic/operating_curves_panels.png`
- AUC vs working point: `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260415/pic/auc_vs_target_reject_at_5_panels.png`
- Per-concept table: `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260415/tables/working_point_per_concept.csv`
- Summary tables: `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260415/tables/working_point_summary_long.csv` and `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/presentation/260415/tables/working_point_summary_wide.csv`

## Group Summaries

### PLRDC vs Mainstream Methods
- Shared concepts: football, indian_politics, photo_captions
- PLRDC SAE: AUC=1.0000, Target@1=100.0%, Target@5=100.0%, Target@10=100.0%, RealizedControl@5=2.3%, n_concepts=3
- ReLU SAE: AUC=0.9734, Target@1=69.3%, Target@5=87.0%, Target@10=93.3%, RealizedControl@5=6.3%, n_concepts=3
- Gated SAE: AUC=0.9592, Target@1=76.7%, Target@5=86.7%, Target@10=89.3%, RealizedControl@5=5.3%, n_concepts=3
- BatchTopK SAE: AUC=0.9995, Target@1=66.3%, Target@5=66.7%, Target@10=66.7%, RealizedControl@5=2.7%, n_concepts=3

### SUR vs Mainstream Methods
- Shared concepts: football, indian_politics
- SUR SAE: AUC=0.9994, Target@1=98.5%, Target@5=100.0%, Target@10=100.0%, RealizedControl@5=3.0%, n_concepts=2
- PLRDC SAE: AUC=0.9999, Target@1=100.0%, Target@5=100.0%, Target@10=100.0%, RealizedControl@5=3.0%, n_concepts=2
- BatchTopK SAE: AUC=0.9992, Target@1=99.5%, Target@5=100.0%, Target@10=100.0%, RealizedControl@5=4.0%, n_concepts=2
- ReLU SAE: AUC=0.9601, Target@1=54.0%, Target@5=80.5%, Target@10=90.0%, RealizedControl@5=3.5%, n_concepts=2
- Gated SAE: AUC=0.9387, Target@1=65.0%, Target@5=80.0%, Target@10=84.0%, RealizedControl@5=4.5%, n_concepts=2

