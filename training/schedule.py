"""Alpha (residual scale) and gate ceiling ramp schedules.

Per docs/baseline_recovery_plan.md:
    "Initialize the geometry energy output and residual scale at zero or near zero."
    "Warm up the baseline path before increasing the maximum geometry gate from
    approximately 0.05 to 0.10."
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlphaSchedule:
    """Linear ramp of alpha (the GeoCorr residual scale multiplier) from 0 -> alpha_max,
    starting only after `warmup_steps` baseline-path-only steps.
    """
    warmup_steps: int = 2000
    ramp_steps: int = 8000
    alpha_max: float = 1.0

    def value(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.0
        t = min(1.0, (step - self.warmup_steps) / max(1, self.ramp_steps))
        return t * self.alpha_max


@dataclass
class GateCeilingSchedule:
    """Ramp of the learned gate's maximum value, from 0 up to `gate_max`
    (recommended range ~0.05-0.10), after the alpha schedule has started.
    """
    warmup_steps: int = 2000
    ramp_steps: int = 10000
    gate_min_ceiling: float = 0.0
    gate_max: float = 0.10

    def value(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.gate_min_ceiling
        t = min(1.0, (step - self.warmup_steps) / max(1, self.ramp_steps))
        return self.gate_min_ceiling + t * (self.gate_max - self.gate_min_ceiling)


@dataclass
class PriorDropoutSchedule:
    """Geometry-prior dropout applied to ~20-30% of training samples, with prior
    scale/noise perturbation, so the network cannot simply copy the prior
    (docs/baseline_recovery_plan.md, "Training Safeguards").
    """
    dropout_prob: float = 0.25
    noise_std: float = 0.05

    def sample_mask(self, batch_size: int, device) -> "torch.Tensor":
        import torch
        return (torch.rand(batch_size, device=device) < self.dropout_prob)
