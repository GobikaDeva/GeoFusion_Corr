"""Sanity checks for the plane-sweep and per-pixel warps.

Uses the simplest possible case: src camera == ref camera (identity relative pose).
In that case the projection is independent of depth, and the warp grid should equal
the identity pixel grid -- i.e. warping should return the input unchanged (up to
grid_sample's boundary handling), regardless of which depth hypothesis is used.
This does not validate the full camera math against real DTU calibration, but it
does catch sign/axis errors in the warp before spending GPU time on real training.
"""
import torch

from models.geofusionnet import differentiable_homography_warp
from models.warp_utils import warp_image_by_depth


def _identity_proj(batch=1):
    proj = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    # simple intrinsic so pixel coords roughly match a small image; extrinsic identity
    proj[:, 0, 0] = 50.0  # fx
    proj[:, 1, 1] = 50.0  # fy
    proj[:, 0, 2] = 4.0   # cx (half of an 8-wide image)
    proj[:, 1, 2] = 4.0   # cy
    return proj


def test_plane_sweep_identity_camera_is_noop_in_interior():
    B, C, H, W, D = 1, 2, 8, 8, 4
    feat = torch.rand(B, C, H, W)
    proj = _identity_proj(B)
    depths = torch.tensor([[1.0, 2.0, 5.0, 10.0]])

    warped = differentiable_homography_warp(feat, proj, proj, depths)
    assert warped.shape == (B, C, D, H, W)

    # interior pixels (away from the border, where grid_sample boundary effects and
    # the H/W-1 pixel-center convention can shift values slightly) should match the
    # original feature closely, for every depth hypothesis, since src==ref camera.
    interior = warped[:, :, :, 2:-2, 2:-2]
    original_interior = feat[:, :, 2:-2, 2:-2].unsqueeze(2).expand_as(interior)
    assert torch.allclose(interior, original_interior, atol=1e-3)


def test_warp_image_by_depth_identity_camera_is_noop():
    B, C, H, W = 1, 3, 8, 8
    img = torch.rand(B, C, H, W)
    proj = _identity_proj(B)
    pred_depth = torch.rand(B, 1, H, W) * 5 + 1

    warped, valid = warp_image_by_depth(img, proj, proj, pred_depth)
    assert warped.shape == img.shape
    assert valid.shape == (B, 1, H, W)

    interior = warped[:, :, 2:-2, 2:-2]
    original_interior = img[:, :, 2:-2, 2:-2]
    assert torch.allclose(interior, original_interior, atol=1e-3)
