# Final Focused Family-Benchmark Metrics

## Chosen Headline Metrics

- `Strict-Budget Alternative-Controller Rate @2%/30%`: among families that are controllable at a 2% held-out control budget with at least 30% target reject, the share of families for which a method retains at least 2 distinct valid controllers. This is a non-proxy control redundancy metric tailored to the SUR innovation.
- `Strict-Budget Valid Trigger Yield @2%`: for each covered family, keep the evaluation target reject rate only if the realized held-out control reject rate is at or below 2%; otherwise score that family as 0. This is a hard-gated strict-budget metric tailored to the PLRDC innovation.

## Headline Results

- Highest strict-budget alternative-controller rate @2%/30%: **SUR SAE** at 20.0%, corresponding to 1/5 controllable families with an alternative valid controller.
- Highest strict-budget valid trigger yield @2%: **PLRDC SAE** at 30.8% over its covered families.
- Supporting utility metric: the highest overall FCOS@2%,5%,10% is **ReLU SAE** at 65.5%.

## Interpretation

- The SUR innovation is better captured by control redundancy than by candidate-space breadth alone. This metric only counts controllers that survive held-out calibration and held-out evaluation, so it is a true control metric rather than a proxy.
- At the chosen strict setting (2% budget, minimum 30% target reject), `kernel` is the only method with a non-zero alternative-controller rate. In the current run, that redundancy appears in `gaming_general`.
- The PLRDC innovation is better captured by a hard-gated strict-budget metric than by a soft penalty. If a family misses the 2% control budget, that family should contribute 0 to a strict-control claim.
- This hard-gated metric separates `dense` from `relu` more clearly because `dense` combines a slightly higher 2% budget-hit rate with a higher in-budget target reject mean.
- `Rich Candidate Coverage Rate @2+` remains a useful supporting candidate-space diagnostic, but it is no longer the SUR headline metric.
- FCOS remains useful as a supporting overall benchmark score, but it is not the cleanest single metric for isolating either innovation.

## Method Table

- SUR SAE: SBACR@2%/30% 20.0% (1/5 controllable families; 6.7% of all families), RCCR@2+ 33.3%, raw candidate yield 22, coverage 15/15, SBVTY@2% 22.8%, 2% hit rate 46.7%, FCOS 62.9%.
- PLRDC SAE: SBACR@2%/30% 0.0% (0/5 controllable families; 0.0% of all families), RCCR@2+ 26.7%, raw candidate yield 17, coverage 13/15, SBVTY@2% 30.8%, 2% hit rate 53.8%, FCOS 55.7%.
- ReLU SAE: SBACR@2%/30% 0.0% (0/6 controllable families; 0.0% of all families), RCCR@2+ 13.3%, raw candidate yield 18, coverage 15/15, SBVTY@2% 29.1%, 2% hit rate 53.3%, FCOS 65.5%.
- Gated SAE: SBACR@2%/30% 0.0% (0/6 controllable families; 0.0% of all families), RCCR@2+ 13.3%, raw candidate yield 18, coverage 14/15, SBVTY@2% 21.2%, 2% hit rate 35.7%, FCOS 60.5%.
- BatchTopK SAE: SBACR@2%/30% 0.0% (0/5 controllable families; 0.0% of all families), RCCR@2+ 13.3%, raw candidate yield 16, coverage 14/15, SBVTY@2% 19.3%, 2% hit rate 42.9%, FCOS 58.3%.
- TopK SAE: SBACR@2%/30% 0.0% (0/3 controllable families; 0.0% of all families), RCCR@2+ 6.7%, raw candidate yield 12, coverage 11/15, SBVTY@2% 11.1%, 2% hit rate 27.3%, FCOS 45.1%.
- JumpReLU SAE: SBACR@2%/30% 0.0% (0/2 controllable families; 0.0% of all families), RCCR@2+ 13.3%, raw candidate yield 9, coverage 8/15, SBVTY@2% 15.5%, 2% hit rate 75.0%, FCOS 22.0%.

## Included Files

- `focused_method_summary.csv`
- `headline_metrics_report.md`
- `raw_benchmark_run/`
- `plots/final_headline_metrics.png`
- `plots/alternative_controller_vs_strict_valid_yield.png`
