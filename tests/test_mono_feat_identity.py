"""B2_mono_feat (configs/ablations/B2_mono_feat.yaml) adds zero-initialized frozen
Depth-Anything-V2 features to B2's reference encoder features. Loaded with B2's weights
and before any training, its scores must be bit-identical to B2's at every cascade
stage; the frozen network must stay out of the optimizer and the checkpoint, and the
injection must still receive a gradient (it is not a dead branch)."""
import torch

from models import build_model
from training.train import load_config
from tests.test_ggf_identity import _inputs


def _b2_and_mono():
    torch.manual_seed(0)
    b2 = build_model(load_config("configs/ablations/B2_nogeo.yaml")["model"]).eval()
    mono = build_model(load_config("configs/ablations/B2_mono_feat.yaml")["model"]).eval()
    result = mono.load_state_dict(b2.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all(k.startswith("backbone.mono_injection.proj.") for k in result.missing_keys)
    return b2, mono


def test_b2_mono_feat_matches_b2_at_init():
    b2, mono = _b2_and_mono()
    inputs = _inputs()
    with torch.no_grad():
        ref = b2(**inputs)["scores"]
        out = mono(**inputs)["scores"]
    assert set(out) == {"coarse", "mid", "fine"}
    for stage in out:
        assert torch.equal(out[stage], ref[stage]), stage


def test_frozen_network_not_trained_or_saved_and_injection_gets_gradient():
    _, mono = _b2_and_mono()
    frozen = mono.backbone.mono_injection._frozen[0]
    assert all(not p.requires_grad for p in frozen.parameters())
    own = {id(p) for p in mono.parameters()}
    assert not any(id(p) in own for p in frozen.parameters())
    assert not any(k.startswith("backbone.mono_injection._frozen") for k in mono.state_dict())

    mono.train()
    sum(s.sum() for s in mono(**_inputs())["scores"].values()).backward()
    assert not frozen.training
    for stage in ("coarse", "mid"):
        grad = mono.backbone.mono_injection.proj[stage].weight.grad
        assert grad is not None and grad.abs().sum() > 0, stage
