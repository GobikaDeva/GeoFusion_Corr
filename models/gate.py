"""Learned reliability gate for GeoCorr Lite (Stage 3).

Per docs/baseline_recovery_plan.md: "Predict candidate reliability and ramp the
maximum gate." The gate must be monitored separately for textured / textureless /
reflective / invalid-prior pixels (see training/trainer.py logging hooks) to catch
gate collapse (docs/risks_and_mitigations.md).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReliabilityGate(nn.Module):
    """Predicts a per-candidate reliability in [0, max_gate] from geometry cues.

    Zero/near-zero initialized so that at the start of Stage 3 training the gate
    contributes negligibly, consistent with the baseline-safety requirement.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )
        # Near-zero init: gate starts small, not exactly zero, to keep gradients alive.
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.constant_(self.net[-1].bias, -4.0)  # sigmoid(-4) ~ 0.018 -> small initial gate

    def forward(self, cues: torch.Tensor, max_gate: float) -> torch.Tensor:
        """
        Args:
            cues: (B, C, D, H, W) scalar geometry cues.
            max_gate: current schedule ceiling (see training/schedule.py), e.g. ramped
                from 0.0 up to ~0.05-0.10.
        Returns:
            (B, 1, D, H, W) gate in [0, max_gate].
        """
        logits = self.net(cues)
        return torch.sigmoid(logits) * max_gate
