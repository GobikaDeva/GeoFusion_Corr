"""Scalar geometry compatibility cues for GeoCorr Lite.

Per docs/baseline_recovery_plan.md ("Candidate Projection and Geometry Cues"):
the first implementation must use only LOW-COST cues sampled on the SAME plane-sweep
grid GeoFusionNet already computes -- no second high-dimensional geometry volume.

Recommended cues (6 channels, extendable to 8 with occlusion/camera-angle later):
    1. reference prior error       -- |prior_depth_ref - candidate_depth|
    2. source reprojection error   -- photometric/geometric reprojection residual
    3. normal disagreement         -- 1 - cos_angle(normal_ref, normal_src_warped)
    4. reference prior confidence  -- confidence map for the reference depth prior
    5. source prior confidence     -- confidence map for the source depth prior
    6. RGB depth-score entropy     -- entropy of the (pre-GeoCorr) matching distribution

Deferred until after baseline parity (Stage 4+):
    7. occlusion cue
    8. camera-angle cue
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

NUM_BASE_CUES = 6
NUM_EXTENDED_CUES = 8  # includes occlusion + camera-angle, Stage 4+


def _softmax_entropy(scores: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Entropy of the softmax distribution of `scores` along `dim`, normalized to [0, 1]."""
    log_probs = F.log_softmax(scores, dim=dim)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=dim, keepdim=True)
    max_entropy = torch.log(torch.tensor(float(scores.shape[dim]), device=scores.device))
    return entropy / max_entropy.clamp_min(1e-6)


def compute_geometry_cues(
    candidate_depth: torch.Tensor,          # (B, D, H, W)
    prior_depth_ref: torch.Tensor,          # (B, 1, H, W)
    prior_confidence_ref: torch.Tensor,     # (B, 1, H, W)
    warped_prior_depth_src: List[torch.Tensor],   # each (B, D, H, W), reused warp grid
    warped_prior_confidence_src: List[torch.Tensor],  # each (B, D, H, W)
    normal_ref: torch.Tensor,               # (B, 3, H, W)
    warped_normal_src: List[torch.Tensor],  # each (B, 3, D, H, W)
    raw_matching_scores: torch.Tensor,      # (B, D, H, W), GeoFusionNet's pre-residual score
    extended_cues: Optional[dict] = None,   # {"occlusion": ..., "camera_angle": ...}, Stage 4+
) -> torch.Tensor:
    """Assemble the scalar geometry-cue tensor, shape (B, C, D, H, W), C in {6, 8}.

    All inputs are expected to already live on the exact plane-sweep grid produced by
    `models.geofusionnet.differentiable_homography_warp` -- computing a second grid
    here would violate the "reuse the projection grid" efficiency constraint.
    """
    B, D, H, W = candidate_depth.shape

    # 1. reference prior error
    ref_prior_error = (prior_depth_ref - candidate_depth).abs().unsqueeze(1)  # placeholder dim fix below
    ref_prior_error = (prior_depth_ref.expand(-1, D, -1, -1) - candidate_depth).abs().unsqueeze(1)

    # 2. source reprojection error (mean across source views)
    src_reproj_errors = [
        (wpd - candidate_depth).abs() for wpd in warped_prior_depth_src
    ]
    src_reproj_error = torch.stack(src_reproj_errors, dim=0).mean(dim=0).unsqueeze(1)

    # 3. normal disagreement (mean cosine disagreement across source views)
    ref_n = F.normalize(normal_ref, dim=1).unsqueeze(2).expand(-1, -1, D, -1, -1)  # (B,3,D,H,W)
    disagreements = []
    for wn in warped_normal_src:
        wn_norm = F.normalize(wn, dim=1)
        cos_sim = (ref_n * wn_norm).sum(dim=1, keepdim=True)  # (B,1,D,H,W)
        disagreements.append(1.0 - cos_sim)
    normal_disagreement = torch.stack(disagreements, dim=0).mean(dim=0)

    # 4/5. prior confidences
    ref_conf = prior_confidence_ref.expand(-1, D, -1, -1).unsqueeze(1)
    src_conf = torch.stack(warped_prior_confidence_src, dim=0).mean(dim=0).unsqueeze(1)

    # 6. entropy of the RGB-only matching distribution (broadcast over D)
    entropy = _softmax_entropy(raw_matching_scores, dim=1)  # (B,1,H,W)
    entropy = entropy.unsqueeze(1).expand(-1, 1, D, -1, -1).squeeze(1).unsqueeze(1) \
        if entropy.dim() == 4 else entropy
    entropy = entropy.expand(-1, -1, D, -1, -1) if entropy.shape[2] != D else entropy

    cues = [ref_prior_error, src_reproj_error, normal_disagreement, ref_conf, src_conf, entropy]

    if extended_cues is not None:
        # Stage 4+: occlusion + camera-angle cues, each (B, 1, D, H, W)
        cues.append(extended_cues["occlusion"])
        cues.append(extended_cues["camera_angle"])

    return torch.cat(cues, dim=1)  # (B, C, D, H, W)
