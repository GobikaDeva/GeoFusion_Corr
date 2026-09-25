"""Single-depth (per-pixel) image warp, used by the compact self-supervised loss.

Distinct from models.geofusionnet.differentiable_homography_warp, which sweeps a
whole batch of discrete depth hypotheses (used for the cost volume). This warps a
source image into the reference view using one continuous predicted depth value per
reference pixel -- the standard photometric-consistency warp for unsupervised MVS.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_image_by_depth(
    src_img: torch.Tensor,     # (B, C, H, W)
    src_proj: torch.Tensor,    # (B, 4, 4), same resolution as src_img
    ref_proj: torch.Tensor,    # (B, 4, 4), same resolution as ref pixel grid
    ref_depth: torch.Tensor,   # (B, 1, H, W) or (B, H, W) predicted depth at ref pixels
) -> tuple:
    """Warps `src_img` into the reference frame using `ref_depth`.

    Returns (warped_src_img, valid_mask) where valid_mask marks reference pixels whose
    projected source coordinate falls inside the source image (in-bounds and in front
    of the camera).
    """
    if ref_depth.dim() == 3:
        ref_depth = ref_depth.unsqueeze(1)
    B, C, H, W = src_img.shape

    proj = torch.matmul(src_proj, torch.inverse(ref_proj))  # (B, 4, 4)
    rot = proj[:, :3, :3]
    trans = proj[:, :3, 3:4]

    y, x = torch.meshgrid(
        torch.arange(0, H, dtype=torch.float32, device=src_img.device),
        torch.arange(0, W, dtype=torch.float32, device=src_img.device),
        indexing="ij",
    )
    xyz = torch.stack((x, y, torch.ones_like(x)), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,3,H,W)
    xyz = xyz.view(B, 3, H * W)

    rot_xyz = torch.matmul(rot, xyz)  # (B, 3, H*W)
    depth_flat = ref_depth.view(B, 1, H * W)
    proj_xyz = rot_xyz * depth_flat + trans  # (B, 3, H*W)

    z = proj_xyz[:, 2:3, :]
    in_front = (z.squeeze(1) > 1e-6)
    z = z.clamp(min=1e-6)
    proj_xy = proj_xyz[:, :2, :] / z  # (B, 2, H*W)

    proj_x_norm = proj_xy[:, 0, :] / ((W - 1) / 2) - 1
    proj_y_norm = proj_xy[:, 1, :] / ((H - 1) / 2) - 1
    grid = torch.stack((proj_x_norm, proj_y_norm), dim=2).view(B, H, W, 2)

    in_bounds = (
        (proj_x_norm.view(B, H, W) >= -1) & (proj_x_norm.view(B, H, W) <= 1)
        & (proj_y_norm.view(B, H, W) >= -1) & (proj_y_norm.view(B, H, W) <= 1)
    )
    valid = (in_bounds & in_front.view(B, H, W)).unsqueeze(1).float()

    warped = F.grid_sample(src_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped, valid
