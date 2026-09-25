# Architecture Comparison: GeoFusionNet vs. GeoCorr

## Executive summary

The most efficient adaptation is **not** the full GeoCorr proposal. The preferred first
implementation adds only a candidate-depth geometry energy and a reliability gate to the
existing GeoFusionNet matching score. The module starts as a no-op and is introduced
only after the GeoFusionNet baseline is reproducible.

GeoFusionNet and GeoCorr address adjacent parts of the same failure:

- **GeoFusionNet** uses geometry to improve descriptors and warping *before* the cost
  volume is regularized (representation-level).
- **GeoCorr** uses reference and source geometry to change the score of each candidate
  depth (scoring-level).

The hybrid is useful when GeoCorr is treated as a **compact matching correction**
inside GeoFusionNet, not a second complete architecture.

Guiding constraints:

- Baseline parity comes first: match the GeoMVS training protocol (data split, input
  resolution, view selection, depth sampling, evaluation code) before interpreting any
  architectural gain.
- The first GeoCorr module preserves the baseline at initialization via a residual
  scale set to zero and a small, gradually increased geometry gate.
- Apply geometry conditioning at the later cascade stages first (narrower candidate
  range, better-aligned priors).
- Do not add deformable epipolar attention, source-view weighting, contrastive losses,
  and normal-conditioned warping all in one experiment.

**Decision criterion:** the hybrid is retained only if it reaches baseline accuracy
within normal training variation and then improves completeness or hard-region
performance without unacceptable runtime/memory growth.

## Shared design principles

Both designs keep RGB and geometry encoders separate at the input and place geometry
before final depth regression. Both target failures from weak photometric evidence and
can use normal/depth priors without concatenating them directly with RGB at the first
layer. Their main difference is *which decision* geometry controls.

## Architectural differences

| Aspect | GeoFusionNet v2 | GeoCorr |
|---|---|---|
| Primary role | Construct geometry-guided features and warps | Score each candidate correspondence with geometric compatibility |
| First interaction | Cross-modal interaction before volume construction | Reference/source modalities interact inside matching |
| Matching unit | Feature location and normal-aware warp | Pixel, source view, candidate depth tuple |
| Geometry cues | Mainly encoded normal features and normal agreement | Depth prior confidence, normals, reprojection, visibility, camera cues |
| Reliability | Simple normal-agreement weighting, advanced confidence deferred | Learned reliability gate for every candidate hypothesis |
| View handling | Mostly conventional aggregation | Optional geometry-aware source-view weighting |
| Training | Compact loss for stable baseline recovery | Full proposal has several additional objectives |
| Best property | Clear implementation path, lower optimization risk | Direct answer to geometry-guided correspondence selection |
| Main risk | Geometry may affect the final match only indirectly | Wrong priors can dominate; training cost can grow rapidly |

## Core mathematical difference

GeoFusionNet first constructs a geometry-guided feature representation used for
matching. GeoCorr then operates at a finer level: it assigns a geometric energy and
reliability weight to each (reference pixel, source view, candidate depth) triple.

**Interpretation:** GeoFusionNet changes the *representation* used for matching.
GeoCorr changes *which depth hypothesis wins*. This makes GeoCorr a suitable matching
module inside GeoFusionNet rather than a replacement for the entire backbone — the
basis for the GeoCorr Lite design in `baseline_recovery_plan.md`.

## Full GeoCorr vs. recommended GeoCorr Lite

| Design choice | Full GeoCorr | Recommended GeoCorr Lite |
|---|---|---|
| Geometry representation | Encoded reference and source geometry fields | Six to eight scalar compatibility maps |
| Scoring head | Candidate-conditioned MLP plus uncertainty and view heads | One shared 1x1 convolution head with two outputs |
| Sampling | Optional deformable epipolar offsets | Reuse the current plane-sweep sampling grid |
| Cascade placement | Potentially every stage | Later stages first |
| Losses | Photometric, feature, contrastive, KL, geometry, cycle, structure, distillation | Existing GeoFusion loss during baseline recovery |
| Expected overhead | High, dependent on depth/views/offsets | Low to moderate — scalar maps only |

## Expected effects by region/condition

| Region / condition | Observed ambiguity | Desired hybrid behavior |
|---|---|---|
| Reliable texture | RGB scores already have a clear peak | Gate stays small; GeoFusionNet behavior preserved |
| Textureless surface | Several depth candidates have similar RGB scores | Reliable geometry penalizes implausible candidates, sharpens distribution |
| Reflective surface | Photometric reliability decreases | Geometry gets more influence only when its own confidence stays high |
| Occlusion | One source view produces an inconsistent candidate | Later view weighting suppresses that source instead of corrupting the aggregate |
| Incorrect prior | Geometry conflicts with multiview evidence | Confidence gate + prior dropout return control to RGB |
| Calibration error | Nominal projection is slightly displaced | Do not add deformable correction until the baseline matcher is stable |

**Note:** the expected benefit (completeness gain in ambiguous regions rather than a
universal per-pixel improvement) must be tested and must not be reported as a measured
result before experiments are complete.
