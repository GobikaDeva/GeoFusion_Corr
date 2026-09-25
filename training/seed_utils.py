"""Seeding utilities.

Per docs/baseline_recovery_plan.md: "Run at least three seeds when feasible, and
record the mean and standard deviation." Used by scripts/reproduce_baseline.sh and
ablations/run_ablation.py to launch multi-seed runs and aggregate results.
"""
from __future__ import annotations

import random

import numpy as np
import torch

DEFAULT_SEEDS = [0, 1, 2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def aggregate_over_seeds(values: list) -> dict:
    """Returns {mean, std, n} for a list of per-seed metric values."""
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.shape[0])}
