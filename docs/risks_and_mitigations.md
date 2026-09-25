# Risks and Mitigations

| Risk | Failure mode | Mitigation |
|---|---|---|
| Double use of normals | Normals influence GGF warping cost, weighting, and GeoCorr simultaneously | Remove multiplicative cost weighting first; add one normal mechanism per ablation |
| Prior copying | Network follows monocular depth prior even when multiview evidence disagrees | Confidence gating, prior dropout, noise, and shuffled-prior controls (N1/N2) |
| Gate collapse | Gate stays near zero or saturates near its maximum | Warm start, ramp the maximum, inspect gate histograms by region |
| Baseline drift | New parameters alter the cost distribution before learning | Zero-initialized residual heads, unchanged regularization |
| Compute growth | Per-candidate cues and optional offsets increase memory traffic | Scalar cues, later stages, shared heads, no deformable offsets initially |
| Unclear cost sign | Normal disagreement accidentally lowers a dissimilarity cost | Define a similarity/energy convention once (see `SCORE_CONVENTION`) and test with synthetic candidates |
| Self-supervised claim | External priors contain learned geometric knowledge | Describe the system as self-supervised MVS with auxiliary geometric priors; disclose prior provenance |

## Final recommendation

Use **GeoCorr Lite** as a zero-initialized residual correction to GeoFusionNet
candidate depth scores. Apply it at the later cascade stages, reuse the current warp
grid, and keep the cost regularizer and loss unchanged until baseline parity is
confirmed. Only then proceed to learned gating, view weighting, and — last, and
optionally — deformable/normal-conditioned sampling.
