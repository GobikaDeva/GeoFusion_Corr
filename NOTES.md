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
