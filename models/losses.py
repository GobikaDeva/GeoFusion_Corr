"""Losses.

Per docs/baseline_recovery_plan.md: "Keep the baseline objective unchanged during the
first GeoCorr Lite experiment." This module intentionally keeps a single compact
self-supervised loss for Stages 0-3 and only exposes hooks for the deferred
objectives (contrastive / KL / distillation) referenced in the design report, which
belong to the "Defer" row of the keep/replace/defer table and to Stage 5.

The compact self-supervised loss below (photometric consistency + edge-aware
smoothness) is the standard unsupervised-MVS objective (Khot et al. 2019-style):
it needs no ground-truth depth, matching the report's framing of GeoFusionNet as a
"self supervised MVS with auxiliary geometric priors" (docs/risks_and_mitigations.md,
"Self supervised claim" mitigation). `supervised_l1_loss` below remains available for
runs that use DTU's ground-truth depth (e.g. for validation-time metrics, or if the
adopted GeoMVS baseline is itself supervised) -- pick ONE and keep it fixed across
Stage 0-3 per the training safeguards.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp_utils import warp_image_by_depth


def _ssim(x: torch.Tensor, y: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Simplified single-scale SSIM (per-pixel dissimilarity map), C1/C2 as in the
    original SSIM paper. Cheap 3x3 average-pool window keeps this "compact"."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, 1, pad)
    mu_y = F.avg_pool2d(y, window, 1, pad)

    sigma_x = F.avg_pool2d(x * x, window, 1, pad) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, window, 1, pad) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window, 1, pad) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim = ssim_n / ssim_d.clamp(min=1e-8)
    return torch.clamp((1 - ssim) / 2, 0, 1)


def photometric_consistency_loss(
    pred_depth: torch.Tensor,        # (B, 1, H, W) or (B, H, W)
    ref_img: torch.Tensor,           # (B, C, H, W)
    src_imgs: List[torch.Tensor],    # each (B, C, H, W)
    ref_proj: torch.Tensor,          # (B, 4, 4), matching this resolution
    src_projs: List[torch.Tensor],   # each (B, 4, 4), matching this resolution
    ssim_weight: float = 0.85,
) -> dict:
    """Warps each source image into the reference view using `pred_depth`, then
    combines L1 + SSIM photometric error, masked to in-bounds/in-front pixels.
    Returns per-source-view losses averaged, plus the aggregate valid mask (useful
    for smoothness/logging).
    """
    total_loss = 0.0
    total_valid = None
    for src_img, src_proj in zip(src_imgs, src_projs):
        warped, valid = warp_image_by_depth(src_img, src_proj, ref_proj, pred_depth)
        l1 = (warped - ref_img).abs().mean(dim=1, keepdim=True)
        ssim_map = _ssim(warped, ref_img).mean(dim=1, keepdim=True)
        photo_error = ssim_weight * ssim_map + (1 - ssim_weight) * l1
        masked_error = photo_error * valid
        denom = valid.sum().clamp(min=1.0)
        total_loss = total_loss + masked_error.sum() / denom
        total_valid = valid if total_valid is None else torch.max(total_valid, valid)
    total_loss = total_loss / max(1, len(src_imgs))
    return {"photometric_loss": total_loss, "valid_mask": total_valid}


def edge_aware_smoothness_loss(pred_depth: torch.Tensor, ref_img: torch.Tensor) -> torch.Tensor:
    """Depth-gradient penalty, downweighted where the reference image itself has a
    strong gradient (i.e. an edge, where a depth discontinuity is expected).

    Depth is divided by its per-image mean first (Monodepth2-style). Raw DTU depths
    are ~425-935mm, so unnormalized gradients made this term outweigh the
    photometric term and rewarded a constant depth map over the true one.
    """
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.unsqueeze(1)
    pred_depth = pred_depth / pred_depth.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)

    depth_dx = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])
    depth_dy = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])

    img_dx = torch.mean(torch.abs(ref_img[:, :, :, :-1] - ref_img[:, :, :, 1:]), dim=1, keepdim=True)
    img_dy = torch.mean(torch.abs(ref_img[:, :, :-1, :] - ref_img[:, :, 1:, :]), dim=1, keepdim=True)

    weight_x = torch.exp(-img_dx)
    weight_y = torch.exp(-img_dy)

    return (depth_dx * weight_x).mean() + (depth_dy * weight_y).mean()


def geometric_prior_loss(
    pred_depth: torch.Tensor,    # (B, 1, H, W) mm
    prior_depth: torch.Tensor,   # (B, 1, H, W) mm, auxiliary prior G
    prior_valid: torch.Tensor,   # (B, 1, H, W) {0, 1}
    normalize: bool = True,
) -> torch.Tensor:
    """L_geo = ||D - G||_1 over valid prior pixels.

    With `normalize` the residual is divided by the per-image mean of G, as the
    smoothness term does, so the paper's lambda_geo is on the same unitless scale as
    the photometric term -- raw DTU depths (~425-935mm) would otherwise make a
    lambda=0.5 L_geo ~100x the photometric loss and reduce training to copying G.
    """
    mask = prior_valid > 0
    if mask.sum() == 0:
        return pred_depth.sum() * 0.0
    residual = (pred_depth - prior_depth).abs()
    if normalize:
        g_mean = (prior_depth * mask).sum(dim=(1, 2, 3), keepdim=True) / mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
        residual = residual / g_mean.clamp(min=1e-6)
    return residual[mask].mean()


def compact_self_supervised_loss(
    pred_depth: torch.Tensor,
    ref_img: torch.Tensor,
    src_imgs: List[torch.Tensor],
    ref_proj: torch.Tensor,
    src_projs: List[torch.Tensor],
    smoothness_weight: float = 0.1,
    photometric_weight: float = 1.0,
    prior_depth: Optional[torch.Tensor] = None,
    prior_valid: Optional[torch.Tensor] = None,
    geo_weight: float = 0.0,
    geo_normalize: bool = True,
) -> dict:
    """GeoFusionNet's compact self-supervised loss:

        L_total = photometric_weight * L_photo + smoothness_weight * L_smooth + geo_weight * L_geo

    (paper: 1.0 / 0.1 / 0.5). L_geo needs the auxiliary prior (`prior_depth`,
    `prior_valid`) and is skipped when `geo_weight` is 0, so Stage 0 configs can
    ablate it via loss.use_geo_term. Keep the chosen objective fixed across
    Stages 0-3 (docs/baseline_recovery_plan.md, "Training Safeguards") -- do not add
    contrastive/KL/distillation terms until Stage 5, and only in separate runs.

    Returns a dict with the total loss plus its components, for logging.
    """
    photo = photometric_consistency_loss(pred_depth, ref_img, src_imgs, ref_proj, src_projs)
    smooth = edge_aware_smoothness_loss(pred_depth, ref_img)
    total = photometric_weight * photo["photometric_loss"] + smoothness_weight * smooth
    out = {
        "photometric_loss": photo["photometric_loss"].detach(),
        "smoothness_loss": smooth.detach(),
        "valid_mask": photo["valid_mask"],
    }
    if geo_weight > 0:
        if prior_depth is None or prior_valid is None:
            raise ValueError("geo_weight > 0 requires prior_depth/prior_valid (data.geometry_prior: mono_sparse)")
        geo = geometric_prior_loss(pred_depth, prior_depth, prior_valid, normalize=geo_normalize)
        total = total + geo_weight * geo
        out["geo_loss"] = geo.detach()
    out["loss"] = total
    return out


def supervised_l1_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standard masked L1 depth loss, for datasets with ground truth (e.g. DTU)."""
    mask = mask.bool()
    if mask.sum() == 0:
        return pred_depth.sum() * 0.0
    return F.l1_loss(pred_depth[mask], gt_depth[mask])


# --- Deferred objectives (Stage 5 / full-GeoCorr comparison only; NOT used in
#     baseline recovery, Stages 0-4) -------------------------------------------------

def contrastive_loss(*args, **kwargs):
    raise NotImplementedError("Deferred until after baseline parity (docs/baseline_recovery_plan.md).")


def geometry_kl_loss(*args, **kwargs):
    raise NotImplementedError("Deferred until after baseline parity (docs/baseline_recovery_plan.md).")


def distillation_loss(*args, **kwargs):
    raise NotImplementedError("Deferred until after baseline parity (docs/baseline_recovery_plan.md).")
