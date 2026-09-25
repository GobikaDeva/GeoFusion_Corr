"""Shared image/geometry transforms and augmentation.

Per Stage 0's baseline-reproduction requirement, augmentation must MATCH the reference
GeoMVS method exactly. Keep any GeoCorr-specific perturbation (prior noise/dropout)
OUT of this file -- that lives in training/schedule.py::PriorDropoutSchedule and is
applied only from Stage 2+ onward, never during Stage 0 baseline reproduction.
"""
from __future__ import annotations

import numpy as np


def normalize_image(img: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    return (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)


def build_depth_hypotheses(depth_min: float, depth_interval: float, num_depth: int) -> np.ndarray:
    """Uniform depth-plane hypotheses, matching the reference GeoMVS depth sampling."""
    return depth_min + depth_interval * np.arange(num_depth, dtype=np.float32)
