import pytest
import torch

from models.losses import compact_self_supervised_loss, geometric_prior_loss


def _inputs(B=2, H=16, W=20):
    torch.manual_seed(0)
    K = torch.eye(4)
    K[0, 0] = K[1, 1] = 20.0
    K[0, 2], K[1, 2] = W / 2, H / 2
    src = K.clone()
    src[0, 3] = 20.0 * 5.0  # small baseline along x
    return {
        "pred_depth": torch.full((B, 1, H, W), 600.0) + torch.randn(B, 1, H, W),
        "ref_img": torch.rand(B, 3, H, W),
        "src_imgs": [torch.rand(B, 3, H, W)],
        "ref_proj": K.expand(B, 4, 4).clone(),
        "src_projs": [src.expand(B, 4, 4).clone()],
    }


def test_geo_loss_zero_when_depth_matches_prior():
    d = torch.rand(2, 1, 8, 8) * 400 + 450
    assert geometric_prior_loss(d, d.clone(), torch.ones_like(d)).item() == 0.0


def test_geo_loss_normalized_by_prior_mean_and_masked():
    g = torch.full((1, 1, 4, 4), 500.0)
    d = g + 50.0
    valid = torch.ones_like(g)
    valid[..., 0, 0] = 0
    d[..., 0, 0] = 1e6  # masked-out pixel must not contribute
    assert geometric_prior_loss(d, g, valid, normalize=False).item() == pytest.approx(50.0)
    assert geometric_prior_loss(d, g, valid, normalize=True).item() == pytest.approx(0.1)


def test_zero_geo_weight_keeps_original_objective():
    x = _inputs()
    base = compact_self_supervised_loss(**x)
    gated = compact_self_supervised_loss(
        **x, prior_depth=torch.full_like(x["pred_depth"], 900.0),
        prior_valid=torch.ones_like(x["pred_depth"]), geo_weight=0.0,
    )
    assert torch.equal(base["loss"], gated["loss"]) and "geo_loss" not in gated


def test_geo_term_added_with_weight():
    x = _inputs()
    prior = x["pred_depth"] + 30.0
    out = compact_self_supervised_loss(**x, prior_depth=prior, prior_valid=torch.ones_like(prior), geo_weight=0.5)
    expected = out["photometric_loss"] + 0.1 * out["smoothness_loss"] + 0.5 * out["geo_loss"]
    assert out["loss"].item() == pytest.approx(expected.item(), rel=1e-5)
    assert out["geo_loss"].item() == pytest.approx(30.0 / 630.0, rel=1e-2)


def test_geo_weight_without_prior_raises():
    with pytest.raises(ValueError):
        compact_self_supervised_loss(**_inputs(), geo_weight=0.5)
