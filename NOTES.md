# Research notes

## Stage 0 baseline (frozen 2026-09-27): A0f

The Stage 0 baseline is `configs/ablations/A0f_fixes_only_short.yaml`, which
`configs/stage0_baseline.yaml` mirrors. It is the end of a three-step chain per seed N:

1. `A0_baseline` seedN: 20k steps, constant lr 1e-3, pre-fix cost volume.
2. `A0_finetune_clmvs` seedN: 6k steps with CL-MVSNet's LR recipe.
3. `A0f_fixes_only_short` seedN: 1500 steps with the cost-volume fixes on. The fixes
   mask out-of-view source samples and use trilinear regularizer upsampling.

Final numbers come from `scripts/eval_stage0_a0f.sh`: full-res 1600x1152, all 22 test
scans, 3 seeds. They are written to `runs/A0f_fixes_only_short/aggregated_1600x1152.json`.
They replace the old A0 figure (0.517 overall), which was measured before the cost-volume
fixes.

We tested CL-MVSNet's consistency loss (ICC + SCC) against A0f, which is the same config
with that loss off. On aggregate validation depth, the two differed by at most 0.05 mm,
which is noise. On scan48 the consistency run ended worse (fine median 4.34 vs 3.45 mm).
We dropped the loss. The code stays in `models/losses.py`, off by default.

## scan48-type scans: photometric ambiguity

On scan48-type scans (the outliers are 11, 13, 48, 62 and 77), wrong depths come from
photometric ambiguity that this training regime does not remove. The self-supervised
photometric objective cannot tell the true surface apart from wrong depth hypotheses.

The evidence comes from `scripts/costvol_profile_probe.py` and `scripts/costvol_diag.py`,
run on pixels with fine-stage error above 4 mm:

- **Raw cost:** GT is the global minimum at only about 30% of scan48's wrong pixels,
  against 89% of correct ones. At the rest, GT is a secondary minimum (about 41%) or not a
  minimum at all (about 28%).
- **Loss preference:** at those pixels the photometric loss prefers the wrong depth over
  GT. Ours picks GT at 42% of them, the min-view variant at 15%, and CL-MVSNet's L0.5
  min-view loss at 12%. A more robust loss does not fix this.
- **Top-k view aggregation:** best-2-view aggregation helps scan1 (GT is the global min at
  36% → 53% of wrong pixels) but not scan48 (26%).
- **Pipeline:** the failure is not in the evaluation pipeline, cascade narrowing or LR.
  - Widening the fine depth window at test time makes scan48 worse.
  - CL-MVSNet's LR recipe gives only about 5%.
  - The cost-volume fixes remove the sawtooth (41% → 0.1% of wrong pixels) but barely
    change the ambiguous share (18.4% → 18.8% secondary-min of valid pixels).
- **Consistency distillation (CL-MVSNet ICC/SCC):** no gain (see above).

This motivates Stage 2 (GeoCorr Lite). We can't pull a correct match out of the image
evidence alone, so the correction has to use geometry: the mono+sparse prior G, and
normal-aware warping or correction of the cost volume.

## Geometry prior experiments (2026-10-06 .. 10-07)

All seed 2, 1500-step continuations of A0f unless noted.

- **B1 (G as input via zero-init GGF) and L_geo off:** both are within run-to-run
  noise of the matched control (`A0f_continued_short`). Val fine median: 1.552 / 1.565 vs
  1.540 mm. scan48 competing-dips share: 18.53 / 18.57 vs 18.42% of valid pixels.
- **Prior quality (22 test scans):** median |G - GT| is 5-69 mm per scan. On ambiguous
  pixels G is nearer the true dip only 52.4% of the time and within 4 mm only 11.5%.
- **Test 1, confidence gates** (`scripts/prior_gate_valfrozen.py`,
  `runs/prior_confidence_val/gates_valfrozen.json`): thresholds fit on dtu_val, applied
  unchanged to test.
  - Only reversed coarse disagreement passes the 20%/80% rule:
    |G - coarse| >= 99.3 mm, val 20.4%/90.7%, test 21.2%/86.7%.
  - Harm with depth := G, test, all valid coarse pixels:
    - fires on 5.29% of pixels, 14.9% of which the model already had right
    - mean error -2.52 mm (from 18.28 mm)
    - share of pixels within 4 mm -0.46 pt
  - So the gate fixes gross outliers but makes almost no pixel accurate.
- **Gate through official fusion** (`scripts/gated_fusion_diag.py`,
  `runs/prior_confidence_val/fusion_diag/`): a diagnostic, not a paper number. A0f seed2,
  1600x1152, GeoMVSNet fusion, outlier scans 11/13/48/62/77, run on CPU. The
  CPU model-only numbers are within 0.025 mm of the GPU evaluation per scan.
  Mean overall: model 0.828, gated (confidence kept) 0.777, gated (confidence := 1) 0.773.
  - Gains come from completeness: scan48 1.93 -> 1.70, scan77 0.80 -> 0.73.
  - scan62 gets slightly worse (0.712 -> 0.724).
  - Not yet measured on the other 17 scans, where firing could cost accuracy.
- **Test 2, B2_capacity** (2.03M params vs 121k; FPN encoder, 8-group cost, 3D U-Net):
  running as of 2026-10-07. Results go to `runs/B2_capacity/seed2/` (diag + probe_final).
  No full DTU evaluation is scheduled.
