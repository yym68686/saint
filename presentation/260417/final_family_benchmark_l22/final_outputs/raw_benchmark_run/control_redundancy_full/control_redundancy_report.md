# Control Redundancy Metric Sweep

This report enumerates non-proxy control metrics derived from evaluating all ontology-matched candidate features on the finalized family benchmark splits.

## 2% Budget Highlights

- `budget=2%`, `min_target=10%`: best redundant-controller family rate is **PLRDC SAE** at 13.3%.
- `budget=2%`, `min_target=20%`: best redundant-controller family rate is **SUR SAE** at 6.7%.
- `budget=2%`, `min_target=30%`: best redundant-controller family rate is **SUR SAE** at 6.7%.
- `budget=2%`, `min_target=40%`: best redundant-controller family rate is **SUR SAE** at 0.0%.

## Candidate Metric Definition

- `Strict-Budget Redundant Controller Family Rate`: among the fixed 15 benchmark families, the share of families for which a method has at least 2 distinct candidate features that each satisfy the held-out control budget and the minimum target reject threshold.
- This is a true control metric rather than a candidate-space proxy, because every counted controller is calibrated on `calibration_control` and validated on held-out `evaluation_target` and `evaluation_control`.
