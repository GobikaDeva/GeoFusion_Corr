# GeoFusionNet + GeoCorr Lite

Efficient integration of **GeoCorr** into **GeoFusionNet** for Multi-View Stereo (MVS),
following the staged baseline-recovery strategy described in
`docs/architecture_comparison.md` and `docs/baseline_recovery_plan.md`.

## Objective

Beat the established **GeoMVS baseline** (accuracy / completeness / overall distance on
DTU, plus generalization and hard-region metrics) by adding **GeoCorr Lite** — a
zero-initialized residual geometry-compatibility correction to GeoFusionNet's matching
score — without regressing the baseline itself. The project is deliberately staged so
that architectural gains can be attributed correctly, rather than confounded with
baseline-reproduction noise.

> **Ground rule:** No architectural comparison against other GeoMVS methods is valid
> until Stage 0 (baseline reproduction) is confirmed within normal training variation.
> See `docs/evaluation_and_gates.md`.

## Repository layout

```
geofusionnet_geocorr/
├── configs/                  Stage configs (stage0..stage5) + ablation configs (A0-A6, N1-N2)
├── data/                     Dataset loading (DTU, generalization sets) & transforms
├── models/                   GeoFusionNet backbone, GeoCorr Lite module, gate, losses
├── training/                 Training loop, seeding, alpha/gate ramp schedules
├── evaluation/                DTU metrics, hard-region masks, calibration, efficiency
├── ablations/                 Ablation + control-experiment drivers (A0-A6, N1, N2)
├── scripts/                   Shell entry points for reproduction / staged runs / ablations
├── docs/                      Condensed design report (source of truth for decisions)
└── tests/                     Unit tests for zero-init, schedules, geometry cues
```

## Staged integration plan (see `docs/baseline_recovery_plan.md`)

| Stage | Model | Change | Advance when |
|---|---|---|---|
| 0 | Baseline | Reproduce reference GeoMVS result, no new geometry modules | Baseline checkpoint matches published/reproduced target |
| 1 | GeoFusion path | Add GGF as a zero-initialized residual at one feature level | Parity with baseline |
| 2 | Fixed GeoCorr Lite | Add geometry energy with a small fixed residual scale | Correct sign and cue behavior |
| 3 | Learned gate | Predict candidate reliability, ramp the maximum gate | Hard-region gain without global loss |
| 4 | View weighting | Add source reliability and occlusion cues | Better handling of weak source views |
| 5 | Optional sampling | Test normal warp / deformable epipolar offsets separately | Incremental value exceeds cost |

## Quickstart

```bash
pip install -r requirements.txt

# Stage 0: reproduce the GeoMVS baseline (run this first, always)
bash scripts/reproduce_baseline.sh

# Stage 1-5: staged integration of GeoCorr Lite
bash scripts/run_staged_integration.sh

# Full ablation + control sweep (A0-A6, N1, N2)
bash scripts/run_all_ablations.sh
```

## Design source

This codebase is a direct implementation of the technical design report
*"Architecture Comparison, Implementation Strategy and Evaluation Gates"*
(GeoFusionNet v2 Design and GeoCorr MVS Research Proposal). See `docs/` for the
condensed version of each section:

- `docs/architecture_comparison.md` — GeoFusionNet vs. GeoCorr, shared principles, core mathematical difference
- `docs/baseline_recovery_plan.md` — Stage 0 protocol, staged integration table, training safeguards
- `docs/evaluation_and_gates.md` — core metrics, required ablations, acceptance gates
- `docs/risks_and_mitigations.md` — risk register (double use of normals, prior copying, gate collapse, etc.)
