"""Assembles GeoFusionNet (+ optional GeoCorr Lite) from a stage config dict.

This is the single place that maps a `configs/stageN_*.yaml` / `configs/ablations/*.yaml`
file to a concrete model, so that stage/ablation configs stay declarative and the
staged-integration table (docs/baseline_recovery_plan.md) is directly executable.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .geofusionnet import GeoFusionNet, GeoFusionNetConfig
from .geocorr_lite import GeoCorrLite, SCORE_CONVENTION


class GeoFusionGeoCorrModel(nn.Module):
    """Wraps GeoFusionNet and applies GeoCorr Lite's residual, when configured,
    to each cascade stage's raw matching score before depth regression.
    """

    def __init__(self, backbone: GeoFusionNet, geocorr: Optional[GeoCorrLite], apply_stages):
        super().__init__()
        self.backbone = backbone
        self.geocorr = geocorr
        self.apply_stages = set(apply_stages or [])

    def forward(self, *args, alpha: float = 0.0, max_gate: float = 0.1, cues_fn=None, **kwargs):
        backbone_out = self.backbone(*args, **kwargs)
        raw_scores = backbone_out["scores"]
        hypotheses = backbone_out["depth_hypotheses"]  # per stage, (B, D, H, W)
        if self.geocorr is None:
            return {"scores": raw_scores, "depth_hypotheses": hypotheses, "geocorr": None}

        corrected = {}
        geocorr_debug = {}
        for stage_name, score in raw_scores.items():
            if stage_name not in self.apply_stages:
                corrected[stage_name] = score
                continue
            if cues_fn is None:
                raise ValueError(
                    "GeoCorr Lite is configured but no `cues_fn` was provided to compute "
                    "geometry cues for this stage. See models/geometry_cues.py."
                )
            cues = cues_fn(stage_name, score)
            out = self.geocorr(score, cues, alpha=alpha, max_gate=max_gate)
            corrected[stage_name] = out["score"]
            geocorr_debug[stage_name] = out
        return {"scores": corrected, "depth_hypotheses": hypotheses, "geocorr": geocorr_debug}


def build_model(cfg: dict) -> GeoFusionGeoCorrModel:
    """Build a model from a stage/ablation config dict (see configs/*.yaml).

    Expected keys (see configs/stage2_geocorr_lite_fixed.yaml for a full example):
        backbone: {in_channels, geometry_channels, base_channels, use_ggf_residual, ggf_stage_index}
        geocorr:
          enabled: bool
          use_learned_gate: bool
          use_extended_cues: bool
          score_convention: "higher_is_better" | "lower_is_better"
          fixed_residual_scale: float
          apply_stages: [str]  # e.g. ["mid", "fine"] -- later stages first
    """
    backbone_cfg = GeoFusionNetConfig(**cfg.get("backbone", {}))
    backbone = GeoFusionNet(backbone_cfg)

    geocorr_cfg = cfg.get("geocorr", {"enabled": False})
    if not geocorr_cfg.get("enabled", False):
        return GeoFusionGeoCorrModel(backbone, None, apply_stages=[])

    geocorr = GeoCorrLite(
        use_learned_gate=geocorr_cfg.get("use_learned_gate", False),
        use_extended_cues=geocorr_cfg.get("use_extended_cues", False),
        convention=SCORE_CONVENTION(geocorr_cfg.get("score_convention", "higher_is_better")),
        fixed_residual_scale=geocorr_cfg.get("fixed_residual_scale", 0.02),
    )
    apply_stages = geocorr_cfg.get("apply_stages", ["fine", "mid"])
    return GeoFusionGeoCorrModel(backbone, geocorr, apply_stages=apply_stages)
