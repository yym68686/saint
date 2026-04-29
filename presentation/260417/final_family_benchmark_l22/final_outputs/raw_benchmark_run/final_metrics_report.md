# L22 Family Benchmark Final Metrics

## Headline Metrics

- `FCOS@2%,5%,10%`: mean penalized target reject rate across all 15 benchmark families. Missing family coverage scores 0.
- `Covered Strict Quality@2%,5%`: mean penalized target reject rate over covered families only, focusing on strict deployment budgets.

## Key Findings

- Highest overall FCOS: **ReLU SAE** at 65.5% with benchmark family coverage 15/15.
- Highest covered strict quality: **ReLU SAE** at 60.6% over 15 covered families.
- Supporting decomposition: ReLU SAE carries the strongest total benchmark utility via coverage + deployable quality, while ReLU SAE is strongest when quality is conditioned on families it already covers.

## Method Summary

- ReLU SAE: coverage 15/15, FCOS 65.5%, covered strict quality 60.6%
- SUR SAE: coverage 15/15, FCOS 62.9%, covered strict quality 57.1%
- Gated SAE: coverage 14/15, FCOS 60.5%, covered strict quality 59.1%
- BatchTopK SAE: coverage 14/15, FCOS 58.3%, covered strict quality 57.1%
- PLRDC SAE: coverage 13/15, FCOS 55.7%, covered strict quality 59.6%
- TopK SAE: coverage 11/15, FCOS 45.1%, covered strict quality 57.3%
- JumpReLU SAE: coverage 8/15, FCOS 22.0%, covered strict quality 33.3%

## Family Notes

- PLRDC SAE strongest strict-budget families: crypto_blockchain, china, nfl_football, russia_post_soviet, soccer
- SUR SAE strongest overall families: china, crypto_blockchain, nfl_football, nhl_hockey, russia_post_soviet

## Output Files

- `method_summary.csv`
- `per_family_summary.csv`
- `per_budget_results.csv`
- `selected_features.csv`
- `run_metadata.json`
- `plots/headline_metrics_panels.png`
- `plots/coverage_vs_quality.png`
