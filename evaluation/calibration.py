"""Calibration metrics: depth entropy, expected calibration error, error-vs-gate.

Per docs/evaluation_and_gates.md: "Determines whether the reliability mechanism is
meaningful" -- i.e., does the learned gate actually track cases where geometry
correction helps, or is it uncorrelated / collapsed (docs/risks_and_mitigations.md).
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE over a binary correctness indicator (e.g. |error| < threshold)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        acc_in_bin = correct[in_bin].mean()
        conf_in_bin = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc_in_bin - conf_in_bin)
    return float(ece)


def error_vs_gate_strength(gate: np.ndarray, abs_error: np.ndarray, n_bins: int = 10) -> dict:
    """Bins pixels by gate strength and reports mean depth error per bin.

    A meaningful reliability gate should show LOWER error correction benefit
    correlating with HIGHER gate confidence where geometry is trustworthy; use
    alongside N1 (shuffled geometry) and N2 (noisy priors) controls to confirm the
    correlation is causal, not incidental.
    """
    bin_edges = np.linspace(gate.min(), gate.max() + 1e-8, n_bins + 1)
    bin_means = []
    bin_centers = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (gate >= lo) & (gate < hi)
        if in_bin.sum() == 0:
            bin_means.append(float("nan"))
        else:
            bin_means.append(float(abs_error[in_bin].mean()))
        bin_centers.append(float((lo + hi) / 2))
    return {"bin_centers": bin_centers, "mean_error": bin_means}
