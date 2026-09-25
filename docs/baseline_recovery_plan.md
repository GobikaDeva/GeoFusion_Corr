# Baseline Recovery Plan

## Stage 0: baseline reproduction

Before adding GeoCorr, reproduce the chosen GeoMVS baseline with the **same** train/test
split, view count, input scale, depth interval, cost metric, regularizer, loss
schedule, augmentation, and post-processing.

- Published and reproduced results **must be shown separately** — a published number is
  not a valid control for a different implementation environment.
- Run at least three seeds when feasible; record mean and standard deviation.
- Verify depth-range conversion, camera scaling, and validity masks before tuning any
  architectural component.
- Confirm the evaluation script and point-cloud filtering match the selected baseline
  method exactly.
- Record wall-clock time, peak GPU memory, parameter count, and the exact number of
  source views.

## Staged integration

| Stage | Model | Change | Advance when |
|---|---|---|---|
| 0 | Baseline | Reproduce the reference GeoMVS result, no new geometry modules | Baseline checkpoint |
| 1 | GeoFusion path | Add GGF as a zero-initialized residual at one feature level | Parity with baseline |
| 2 | Fixed GeoCorr Lite | Add geometry energy with a small fixed residual scale | Correct sign and cue behavior |
| 3 | Learned gate | Predict candidate reliability, ramp the maximum gate | Hard-region gain without global loss |
| 4 | View weighting | Add source reliability and occlusion cues | Better handling of weak source views |
| 5 | Optional sampling | Test normal warp or deformable epipolar offsets in separate runs | Incremental value exceeds cost |

Each stage's config lives under `configs/stageN_*.yaml` and is meant to be launched
only after the previous stage's "advance when" condition is met — see
`scripts/run_staged_integration.sh`.

## Training safeguards

- Initialize the geometry-energy output and residual scale at zero (or near zero).
- Warm up the baseline path before increasing the maximum geometry gate — roughly
  0.05 to 0.10.
- Apply geometry-prior dropout on ~20-30% of training samples, and perturb prior scale
  and noise so the network cannot simply copy the prior.
- Monitor the gate mean and distribution separately for textured / textureless /
  reflective / invalid-prior pixels.
- Use a single score convention consistently: if higher scores mean better matches,
  *subtract* positive geometry energy; if lower scores mean better costs, *add* positive
  geometry energy. (See `models/geocorr_lite.py::SCORE_CONVENTION`.)

Keep the baseline objective unchanged during the first GeoCorr Lite experiment.

## GeoCorr Lite design (baseline-safe)

- Reuse the projection grid and cost volume already computed by GeoFusionNet — do not
  construct a second high-dimensional geometry cost volume.
- Represent geometry with a small number of scalar compatibility cues (6-8 channels)
  rather than a second feature volume: reference prior error, source reprojection
  error, normal disagreement, reference/source prior confidence, entropy of the RGB
  depth scores. Occlusion and camera-angle cues are added only after baseline parity
  (Stage 4).
- Process cues with a compact shared 1x1 convolution / shallow MLP head.
- Apply the correction at the final two cascade stages first, before extending to the
  coarse stage.
- Insert as an **additive residual on the matching score**, scaled by a schedule
  variable `alpha` that starts at 0 so the forward path at initialization is exactly
  the existing GeoFusionNet model (see `training/schedule.py`).

## Components: keep / replace / defer

| Decision | Component | Reason |
|---|---|---|
| Keep | Separate encoders, existing projection grid, cost regularizer, cascade, compact losses | Preserves the validated GeoFusion computation path |
| Keep conditionally | GGF residual attention at one low-resolution feature level | Retain only if it has already reached baseline parity |
| Add first | Scalar geometry energy, reliability gate, residual scale | Directly modifies candidate-depth ranking with small overhead |
| Replace | Simple multiplicative normal weighting of the cost volume | Avoids ambiguous cost signs and repeated use of the same normal cue |
| Defer | Geometry-aware view weighting and occlusion reasoning | Useful only after the score correction is independently validated |
| Defer | Deformable epipolar sampling, contrastive loss, KL loss, distillation | These increase cost and make baseline attribution difficult |

## Efficiency implications

- Asymptotic cost stays tied to the existing source-view x depth x height x width
  volume; GeoCorr Lite adds only a small constant number of scalar channels, not
  another learned feature volume.
- Apply the module at 1/2 or 1/4 resolution, at the final two cascade stages first.
- Vectorize all cues on the existing warp grid; share the small scoring head across
  stages where dimensions permit.
- Avoid global spatial GGF attention; use windowed attention, channel gating, or one
  low-resolution GGF block if the current implementation is quadratic in image area.
- Do not activate local deformable offsets — each additional offset multiplies feature
  sampling and memory traffic.
