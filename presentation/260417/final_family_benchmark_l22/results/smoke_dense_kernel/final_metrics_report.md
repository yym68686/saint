# L22 Family Benchmark Final Metrics

## Headline Metrics

- `FCOS@2%,5%,10%`: mean penalized target reject rate across all 15 benchmark families. Missing family coverage scores 0.
- `Covered Strict Quality@2%,5%`: mean penalized target reject rate over covered families only, focusing on strict deployment budgets.

## Key Findings

- Highest overall FCOS: **PLRDC SAE** at 78.9% with benchmark family coverage 2/15.
- Highest covered strict quality: **PLRDC SAE** at 76.6% over 2 covered families.
- Supporting decomposition: PLRDC SAE carries the strongest total benchmark utility via coverage + deployable quality, while PLRDC SAE is strongest when quality is conditioned on families it already covers.

## Method Summary

- PLRDC SAE: coverage 2/15, FCOS 78.9%, covered strict quality 76.6%
- SUR SAE: coverage 2/15, FCOS 78.0%, covered strict quality 73.7%

## Family Notes

- PLRDC SAE strongest strict-budget families: china, soccer
- SUR SAE strongest overall families: china, soccer

## Output Files

- `method_summary.csv`
- `per_family_summary.csv`
- `per_budget_results.csv`
- `selected_features.csv`
- `run_metadata.json`
- `plots/headline_metrics_panels.png`
- `plots/coverage_vs_quality.png`
