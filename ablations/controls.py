"""Control-experiment perturbations used by N1 and N2 (docs/evaluation_and_gates.md).

N1: shuffled geometry -- confirms geometry cues carry causal information (if
    shuffling them doesn't hurt performance, GeoCorr Lite isn't really using geometry).
N2: noisy or dropped priors -- tests fallback behavior; performance should degrade
    gracefully toward RGB-only behavior, not collapse.
"""
from __future__ import annotations

import torch


def shuffle_geometry_cues(cues: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """N1: randomly permute cues across the batch dimension so per-sample geometry no
    longer corresponds to the matching RGB content, while keeping tensor statistics
    (mean/std/shape) identical."""
    perm = torch.randperm(cues.shape[dim], device=cues.device)
    return cues.index_select(dim, perm)


def perturb_priors(
    prior_depth: torch.Tensor,
    prior_confidence: torch.Tensor,
    noise_std: float = 0.05,
    dropout_prob: float = 0.25,
    drop_value: float = 0.0,
) -> dict:
    """N2: additive Gaussian noise on prior depth/confidence, plus outright dropping
    the prior for a fraction of samples (mirrors training/schedule.PriorDropoutSchedule,
    but used here as a held-out evaluation-time stress test rather than a training
    augmentation)."""
    noisy_depth = prior_depth + noise_std * torch.randn_like(prior_depth)
    noisy_conf = (prior_confidence + noise_std * torch.randn_like(prior_confidence)).clamp(0, 1)

    drop_mask = (torch.rand(prior_depth.shape[0], device=prior_depth.device) < dropout_prob)
    drop_mask = drop_mask.view(-1, *([1] * (prior_depth.dim() - 1)))
    dropped_depth = torch.where(drop_mask, torch.full_like(noisy_depth, drop_value), noisy_depth)
    dropped_conf = torch.where(drop_mask, torch.zeros_like(noisy_conf), noisy_conf)

    return {"prior_depth": dropped_depth, "prior_confidence": dropped_conf, "drop_mask": drop_mask}
