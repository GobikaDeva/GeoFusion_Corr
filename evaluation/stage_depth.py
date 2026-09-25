"""Per-cascade-stage depth diagnostics against DTU GT depth maps.

For each stage: absolute depth error |pred - gt| (mm) on valid GT pixels at that
stage's resolution, and GT coverage -- the fraction of valid GT pixels whose true
depth lies inside the stage's per-pixel search window [first, last hypothesis].
Coverage < 100% at a narrowed stage means cascade narrowing has already locked the
pixel out of the correct depth.

Used by scripts/costvol_diag.py (per-scan breakdown) and training/train.py
(validation depth curve).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models import regress_depth


def _collate_one(sample: dict, device: str) -> dict:
    """Adds a batch dim to a single DTUDataset sample's tensors."""
    def up(x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0).to(device)
        if isinstance(x, list):
            return [up(v) for v in x]
        if isinstance(x, dict):
            return {k: up(v) for k, v in x.items()}
        return x
    return up(sample)


@torch.no_grad()
def sample_stage_stats(model, sample: dict, device: str) -> dict:
    """Runs one DTUDataset sample (must carry gt_depth) through `model` and returns
    {stage: {"abs_err": 1-D np.ndarray of per-pixel errors (mm), "covered": 1-D bool
    np.ndarray, same pixels}}."""
    batch = _collate_one(sample, device)
    outputs = model(**batch["model_inputs"], alpha=1.0, max_gate=1.0)
    stats = {}
    for stage, scores in outputs["scores"].items():
        hyp = outputs["depth_hypotheses"][stage]
        pred = regress_depth(scores, hyp)  # (1, 1, H, W)
        H, W = pred.shape[-2:]
        gt = F.interpolate(batch["gt_depth"], size=(H, W), mode="nearest")
        valid = gt > 0
        covered = (gt >= hyp[:, :1]) & (gt <= hyp[:, -1:])
        stats[stage] = {
            "abs_err": (pred - gt).abs()[valid].cpu().numpy(),
            "covered": covered[valid].cpu().numpy(),
        }
    return stats


def summarize(per_sample: list) -> dict:
    """Aggregates a list of sample_stage_stats results.

    median_err_mm: median over all pooled valid pixels.
    median_view_err_mm: median over views of each view's median error.
    coverage_pct: pooled % of valid pixels whose GT lies inside the stage window.
    pct_gt_4mm: pooled % of valid pixels with error > 4mm.
    """
    out = {}
    for stage in per_sample[0]:
        errs = [s[stage]["abs_err"] for s in per_sample if s[stage]["abs_err"].size]
        cov = [s[stage]["covered"] for s in per_sample if s[stage]["covered"].size]
        pooled = np.concatenate(errs)
        out[stage] = {
            "median_err_mm": float(np.median(pooled)),
            "median_view_err_mm": float(np.median([np.median(e) for e in errs])),
            "pct_gt_4mm": float(100.0 * (pooled > 4.0).mean()),
            "coverage_pct": float(100.0 * np.concatenate(cov).mean()),
            "n_views": len(errs),
        }
    return out
