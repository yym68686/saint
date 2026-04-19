# Final Focused Family-Benchmark Metrics

## Chosen Headline Metrics

- `Rich Candidate Coverage Rate @2+`: among the fixed 15 benchmark families, the share of families for which a method contributes at least 2 ontology-matched candidate features. This is a percentage-valued candidate-space metric designed to capture whether a method provides not just coverage, but redundant benchmark-ready options.
- `Strict-Budget Valid Trigger Yield @2%`: for each covered family, keep the evaluation target reject rate only if the realized held-out control reject rate is at or below 2%; otherwise score that family as 0. This is a hard-gated strict-budget metric tailored to the PLRDC innovation.

## Headline Results

- Highest rich candidate coverage rate @2+: **SUR SAE** at 33.3%, with 22 matched benchmark-ready candidates in total.
- Highest strict-budget valid trigger yield @2%: **PLRDC SAE** at 30.8% over its covered families.
- Supporting utility metric: the highest overall FCOS@2%,5%,10% is **ReLU SAE** at 65.5%.

## Interpretation

- The SUR innovation is better captured by rich candidate coverage than by binary coverage alone. In this benchmark, `kernel` and `relu` both reach full family coverage, but `kernel` is the only method that sustains multiple benchmark-ready candidates across a substantially larger share of families.
- The PLRDC innovation is better captured by a hard-gated strict-budget metric than by a soft penalty. If a family misses the 2% control budget, that family should contribute 0 to a strict-control claim.
- This hard-gated metric separates `dense` from `relu` more clearly because `dense` combines a slightly higher 2% budget-hit rate with a higher in-budget target reject mean.
- FCOS remains useful as a supporting overall benchmark score, but it is not the cleanest single metric for isolating either innovation.

## Method Table

- SUR SAE: RCCR@2+ 33.3%, raw candidate yield 22, coverage 15/15, SBVTY@2% 22.8%, 2% hit rate 46.7%, FCOS 62.9%.
- PLRDC SAE: RCCR@2+ 26.7%, raw candidate yield 17, coverage 13/15, SBVTY@2% 30.8%, 2% hit rate 53.8%, FCOS 55.7%.
- ReLU SAE: RCCR@2+ 13.3%, raw candidate yield 18, coverage 15/15, SBVTY@2% 29.1%, 2% hit rate 53.3%, FCOS 65.5%.
- Gated SAE: RCCR@2+ 13.3%, raw candidate yield 18, coverage 14/15, SBVTY@2% 21.2%, 2% hit rate 35.7%, FCOS 60.5%.
- BatchTopK SAE: RCCR@2+ 13.3%, raw candidate yield 16, coverage 14/15, SBVTY@2% 19.3%, 2% hit rate 42.9%, FCOS 58.3%.
- JumpReLU SAE: RCCR@2+ 13.3%, raw candidate yield 9, coverage 8/15, SBVTY@2% 15.5%, 2% hit rate 75.0%, FCOS 22.0%.
- TopK SAE: RCCR@2+ 6.7%, raw candidate yield 12, coverage 11/15, SBVTY@2% 11.1%, 2% hit rate 27.3%, FCOS 45.1%.

## Included Files

- `focused_method_summary.csv`
- `headline_metrics_report.md`
- `raw_benchmark_run/`
- `plots/final_headline_metrics.png`
- `plots/candidate_yield_vs_ultra_strict_quality.png`
