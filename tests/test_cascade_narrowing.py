"""Cascade narrowing: later stages sweep a narrow per-pixel window around the
previous stage's depth (CasMVSNet-style), instead of re-sweeping the full range."""
import torch

from models.build_model import build_model
from models.geofusionnet import GeoFusionNet, differentiable_homography_warp, regress_depth


def _proj(B):
    P = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    P[:, 0, 0] = P[:, 1, 1] = 50.0
    P[:, 0, 2], P[:, 1, 2] = 16.0, 12.0
    return P


def test_per_pixel_warp_matches_shared_hypotheses():
    B, C, H, W, D = 2, 4, 24, 32, 5
    feat = torch.randn(B, C, H, W)
    ref = _proj(B)
    src = ref.clone()
    src[:, 0, 3] = 30.0  # baseline along x
    shared = torch.linspace(400, 800, D).unsqueeze(0).repeat(B, 1)
    per_pixel = shared.view(B, D, 1, 1).expand(-1, -1, H, W).contiguous()
    a = differentiable_homography_warp(feat, src, ref, shared)
    b = differentiable_homography_warp(feat, src, ref, per_pixel)
    assert torch.allclose(a, b, atol=1e-5)


def test_regress_depth_accepts_per_pixel_hypotheses():
    B, D, H, W = 1, 4, 3, 3
    hyp = torch.arange(D, dtype=torch.float32).view(1, D, 1, 1).expand(B, D, H, W) * 10 + 500
    scores = torch.full((B, D, H, W), -1e4)
    scores[:, 2] = 0.0
    assert torch.allclose(regress_depth(scores, hyp), torch.full((B, 1, H, W), 520.0))


def test_narrowed_hypotheses_centered_with_stage_spacing():
    prev = torch.full((2, 1, 4, 5), 600.0)
    interval = torch.tensor([2.5, 5.0])
    hyp = GeoFusionNet._narrowed_hypotheses(prev, (8, 10), num_depth=8, interval=interval)
    assert hyp.shape == (2, 8, 8, 10)
    assert torch.allclose(hyp[0, 1] - hyp[0, 0], torch.full((8, 10), 2.5))
    assert torch.allclose(hyp[1, 1] - hyp[1, 0], torch.full((8, 10), 5.0))
    assert torch.allclose(hyp[0].mean(0), torch.full((8, 10), 600.0 - 1.25))  # window centred on prev


def test_forward_chains_stages_and_narrows():
    model = build_model({"backbone": {"base_channels": 8}, "geocorr": {"enabled": False}}).eval()
    B, H, W = 1, 64, 80
    imgs = [torch.randn(B, 3, H, W) for _ in range(3)]
    proj = _proj(B)
    src_proj = proj.clone()
    src_proj[:, 0, 3] = 20.0
    coarse = torch.linspace(425, 425 + 48 * 10, 48).unsqueeze(0)
    with torch.no_grad():
        out = model(
            ref_img=imgs[0], src_imgs=imgs[1:], ref_geom=torch.zeros(B, 4, H, W),
            ref_proj=proj, src_projs=[src_proj, src_proj],
            depth_hypotheses_per_stage={"coarse": coarse}, depth_interval=torch.tensor([2.5]),
        )
    hyp = out["depth_hypotheses"]
    assert set(out["scores"]) == {"coarse", "mid", "fine"}
    assert hyp["fine"].shape == (B, 8, H, W) and hyp["mid"].shape == (B, 32, H // 2, W // 2)
    fine_span = (hyp["fine"][:, -1] - hyp["fine"][:, 0]).max().item()
    assert abs(fine_span - 7 * 2.5) < 1e-3  # fine window is 8 x 2.5mm, not the full range
    assert abs((hyp["mid"][:, 1] - hyp["mid"][:, 0]).mean().item() - 5.0) < 1e-3


def test_smoothness_is_depth_scale_invariant():
    # Raw-mm depth gradients once outweighed the photometric term and rewarded
    # constant depth maps; the term must not depend on the depth's absolute scale.
    from models.losses import edge_aware_smoothness_loss

    depth = torch.rand(2, 1, 16, 20) + 1.0
    img = torch.randn(2, 3, 16, 20)
    assert torch.allclose(edge_aware_smoothness_loss(depth, img), edge_aware_smoothness_loss(depth * 700.0, img), rtol=1e-4)
