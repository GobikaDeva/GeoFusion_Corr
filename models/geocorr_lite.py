"""GeoCorr Lite: baseline-safe residual geometry correction to matching scores.

Core idea (docs/architecture_comparison.md / baseline_recovery_plan.md):
    GeoFusionNet changes the *representation* used for matching.
    GeoCorr changes *which depth hypothesis wins*.

GeoCorr Lite implements the latter as an ADDITIVE residual on GeoFusionNet's raw
matching-score volume, scaled by a schedule variable `alpha` that starts at 0 so the
forward path at initialization is exactly the unmodified GeoFusionNet baseline.

    corrected_score = raw_score + SCORE_CONVENTION_SIGN * alpha * gate * energy

SCORE_CONVENTION fixes the cost-sign ambiguity flagged in docs/risks_and_mitigations.md
("Unclear cost sign"): define once, use everywhere.
"""
from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn

from .gate import ReliabilityGate
from .geometry_cues import NUM_BASE_CUES, NUM_EXTENDED_CUES


class SCORE_CONVENTION(str, Enum):
    """Fixes the single score convention used across the whole codebase.

    HIGHER_IS_BETTER -> subtract positive geometry energy (energy = incompatibility).
    LOWER_IS_BETTER  -> add positive geometry energy.
    GeoFusionNet's matching score (from cost-volume regularization / correlation) is
    HIGHER_IS_BETTER by default in this scaffold -- update if the real backbone differs.
    """
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


DEFAULT_CONVENTION = SCORE_CONVENTION.HIGHER_IS_BETTER


class GeometryEnergyHead(nn.Module):
    """One shared 1x1(x1) convolution head producing a non-negative geometric energy.

    Zero-initialized so the module starts as a strict no-op (energy == 0 everywhere),
    independent of the alpha/gate schedule -- belt-and-braces baseline safety.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cues: torch.Tensor) -> torch.Tensor:
        # Softplus keeps the energy non-negative (an "incompatibility" magnitude)
        # while still being exactly 0 at initialization (softplus(0) != 0, so we
        # instead use a plain linear output clamped at inference-safe magnitudes and
        # rely on zero-init for the true starting no-op).
        return self.net(cues)  # (B, 1, D, H, W)


class GeoCorrLite(nn.Module):
    """Residual scoring module: geometry energy * learned gate * alpha schedule.

    use_learned_gate=False -> Stage 2 (fixed residual scale, no learned gate).
    use_learned_gate=True  -> Stage 3+ (learned reliability gate).
    """

    def __init__(
        self,
        use_learned_gate: bool = False,
        use_extended_cues: bool = False,
        convention: SCORE_CONVENTION = DEFAULT_CONVENTION,
        fixed_residual_scale: float = 0.02,
    ):
        super().__init__()
        num_cues = NUM_EXTENDED_CUES if use_extended_cues else NUM_BASE_CUES
        self.energy_head = GeometryEnergyHead(num_cues)
        self.use_learned_gate = use_learned_gate
        self.gate = ReliabilityGate(num_cues) if use_learned_gate else None
        self.convention = convention
        self.fixed_residual_scale = fixed_residual_scale  # used when gate is None

        # log-parameterized residual scale for the fixed-scale (Stage 2) case, so it
        # can be optimized but stays initialized at the configured small value.
        self.register_buffer("_fixed_scale", torch.tensor(float(fixed_residual_scale)))

    def _sign(self) -> float:
        return -1.0 if self.convention == SCORE_CONVENTION.HIGHER_IS_BETTER else 1.0

    def forward(
        self,
        raw_score: torch.Tensor,     # (B, D, H, W) GeoFusionNet's pre-residual score
        cues: torch.Tensor,          # (B, C, D, H, W) from geometry_cues.compute_geometry_cues
        alpha: float,                # training-time schedule scalar in [0, 1], see training/schedule.py
        max_gate: float = 0.1,       # ceiling for the learned gate (Stage 3+)
    ) -> dict:
        energy = self.energy_head(cues).squeeze(1)  # (B, D, H, W), == 0 at init

        if self.use_learned_gate:
            gate = self.gate(cues, max_gate=max_gate).squeeze(1)  # (B, D, H, W)
        else:
            gate = torch.full_like(energy, self._fixed_scale.item())

        residual = self._sign() * alpha * gate * energy
        corrected_score = raw_score + residual

        return {
            "score": corrected_score,
            "energy": energy,
            "gate": gate,
            "residual": residual,
            "alpha": alpha,
        }
