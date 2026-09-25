"""Automatically generated hard-region masks: low texture, high residual, occlusion.

Per docs/evaluation_and_gates.md, the mask and threshold MUST be fixed before model
selection ("Hard-region value" gate) -- do not retune thresholds after seeing results
for a candidate model.
"""
from __future__ import annotations

import numpy as np


def low_texture_mask(ref_img_gray: np.ndarray, window: int = 7, var_threshold: float = 15.0) -> np.ndarray:
    """Pixels with low local intensity variance (textureless regions)."""
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(ref_img_gray.astype(np.float32), size=window)
    sq_mean = uniform_filter(ref_img_gray.astype(np.float32) ** 2, size=window)
    local_var = np.clip(sq_mean - mean ** 2, 0, None)
    return local_var < var_threshold


def high_residual_mask(pred_depth: np.ndarray, prior_depth: np.ndarray, residual_threshold: float) -> np.ndarray:
    """Pixels where the prediction disagrees strongly with the geometric prior --
    used to isolate cases the geometry cues are meant to help with, independent of
    whether they actually do (that independence is what N1/N2 controls test)."""
    return np.abs(pred_depth - prior_depth) > residual_threshold


def occlusion_mask(consistency_count: np.ndarray, min_consistent_views: int = 2) -> np.ndarray:
    """Pixels visible in fewer than `min_consistent_views` source views after
    geometric consistency checking (standard MVS multi-view consistency filter)."""
    return consistency_count < min_consistent_views


def build_all_masks(
    ref_img_gray: np.ndarray,
    pred_depth: np.ndarray,
    prior_depth: np.ndarray,
    consistency_count: np.ndarray,
    residual_threshold: float,
    min_consistent_views: int = 2,
) -> dict:
    return {
        "textureless": low_texture_mask(ref_img_gray),
        "high_residual": high_residual_mask(pred_depth, prior_depth, residual_threshold),
        "occlusion": occlusion_mask(consistency_count, min_consistent_views),
        "textured": ~low_texture_mask(ref_img_gray),
    }
