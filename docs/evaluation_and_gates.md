# Evaluation and Ablation Plan

## Core metrics

| Category | Measurement | Purpose |
|---|---|---|
| Reconstruction | Accuracy, completeness, overall distance on DTU | Primary baseline comparison |
| Generalization | F-score on an additional benchmark when available | Checks whether gains transfer beyond DTU |
| Hard regions | Completeness and depth error under auto-generated low-texture, high-residual, occlusion masks | Tests the stated research problem directly |
| Calibration | Depth entropy, expected calibration error, error vs. gate strength | Determines whether the reliability mechanism is meaningful |
| Efficiency | Runtime, peak memory, parameters, cost-volume size | Verifies GeoCorr Lite remains practical |

When comparing against other GeoMVS methods, use identical image resolution, view
count, depth range, and fusion thresholds whenever the implementation permits. **Label
results as published or reproduced and never combine the two categories in one ranking
without qualification.**

## Required ablations

| ID | Variant | Question | Priority |
|---|---|---|---|
| A0 | Reference GeoMVS baseline | Establish the reproducible target | Required |
| A1 | A0 + GeoFusion GGF | Measure feature conditioning alone | Required |
| A2 | A1 + normal-conditioned warp | Measure sampling change independently | Conditional |
| A3 | A1 + fixed GeoCorr energy | Verify candidate geometry score | Required |
| A4 | A3 + learned reliability gate | Test adaptive geometry influence | Required |
| A5 | A4 + source-view weighting | Test view reliability | Second phase |
| A6 | A4 + deformable sampling | Measure value of local search | Later |
| N1 | A4 with shuffled geometry | Confirm geometry carries causal information | Required |
| N2 | A4 with noisy or dropped priors | Test fallback behavior | Required |

Ablation configs live in `configs/ablations/`; run all of them with
`ablations/run_ablation.py` (driven by `scripts/run_all_ablations.sh`).

## Acceptance gates

These are engineering recommendations, not reported experimental results — adjust to
the variance and compute budget of the selected baseline.

| Gate | Pass condition | Interpretation |
|---|---|---|
| Baseline parity | Hybrid score lies within baseline mean +/- one standard deviation | If only one run is possible, use no more than ~1% relative degradation as a temporary rule |
| Hard-region value | Completeness or depth error improves in at least one predefined ambiguity mask | Mask and threshold must be fixed before model selection |
| Efficiency | Runtime and peak memory remain within the project budget | Practical initial target for GeoCorr Lite: < 10% overhead |
| Prior robustness | Moderate prior noise does not cause catastrophic loss | Model should approach RGB-only behavior as confidence decreases |
| Repeatability | Direction of change is consistent across seeds | Do not select a module from a single favorable run |

These gates are implemented as checks in `evaluation/eval_dtu.py` and
`evaluation/efficiency.py`, and are enforced by `ablations/run_ablation.py` before a
stage is marked as "passed" in its run log.
