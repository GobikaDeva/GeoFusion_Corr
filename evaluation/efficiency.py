"""Efficiency measurement: runtime, peak memory, parameter count, cost-volume size.

Feeds the "Efficiency" acceptance gate in docs/evaluation_and_gates.md:
    "Runtime and peak memory remain within the project budget. A practical initial
    target for GeoCorr Lite is less than ten percent overhead."
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


@contextmanager
def measure_runtime_and_memory(device: str = "cuda"):
    """Usage:
        with measure_runtime_and_memory() as m:
            model(...)
        print(m["elapsed_sec"], m["peak_mem_mb"])
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    result = {}
    t0 = time.time()
    yield result
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        result["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
    else:
        result["peak_mem_mb"] = None
    result["elapsed_sec"] = time.time() - t0


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def relative_overhead(baseline_value: float, candidate_value: float) -> float:
    """Fractional overhead of `candidate_value` relative to `baseline_value`
    (e.g. runtime, memory). Used directly against the <10% efficiency gate."""
    if baseline_value == 0:
        return float("inf")
    return (candidate_value - baseline_value) / baseline_value
