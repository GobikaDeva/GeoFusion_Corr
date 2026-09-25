"""Confirms the single most important baseline-safety property:
GeoCorr Lite is an exact no-op at initialization, at alpha=0, and at any alpha with
zero-initialized energy weights, so the baseline forward path is preserved
(docs/baseline_recovery_plan.md, "Training Safeguards").
"""
import torch

from models.geocorr_lite import GeoCorrLite


def _dummy_inputs(B=2, D=4, H=8, W=8, C=6):
    raw_score = torch.randn(B, D, H, W)
    cues = torch.randn(B, C, D, H, W)
    return raw_score, cues


def test_fixed_scale_zero_at_init():
    raw_score, cues = _dummy_inputs()
    module = GeoCorrLite(use_learned_gate=False, use_extended_cues=False, fixed_residual_scale=0.02)
    out = module(raw_score, cues, alpha=1.0)
    # energy head is zero-initialized -> energy == 0 everywhere regardless of gate/alpha
    assert torch.allclose(out["energy"], torch.zeros_like(out["energy"]))
    assert torch.allclose(out["score"], raw_score, atol=1e-6)


def test_learned_gate_zero_at_init():
    raw_score, cues = _dummy_inputs()
    module = GeoCorrLite(use_learned_gate=True, use_extended_cues=False)
    out = module(raw_score, cues, alpha=1.0, max_gate=0.1)
    assert torch.allclose(out["energy"], torch.zeros_like(out["energy"]))
    assert torch.allclose(out["score"], raw_score, atol=1e-6)


def test_alpha_zero_is_noop_even_with_nonzero_energy():
    raw_score, cues = _dummy_inputs()
    module = GeoCorrLite(use_learned_gate=False, fixed_residual_scale=0.02)
    # force non-zero energy to simulate a partially trained head
    with torch.no_grad():
        module.energy_head.net[-1].bias.fill_(1.0)
    out = module(raw_score, cues, alpha=0.0)
    assert torch.allclose(out["score"], raw_score, atol=1e-6)
