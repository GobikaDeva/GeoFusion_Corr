"""Depth-map back-projection for DTU evaluation.

`backproject_depth_map` + `voxel_downsample` is the legacy unfiltered fusion (kept for
comparison); `geometric_consistency_fusion` below is the reference baseline's fusion and
the default in evaluation/eval_dtu.py.

LEGACY SIMPLIFICATION (documented, matching the "no second cost volume" spirit of
this scaffold): per-view predicted depth maps are back-projected and concatenated
into a single point cloud with no cross-view photometric/geometric consistency
filtering. A full CasMVSNet-style fusion (reprojection-consistency pruning across
views) would remove more outliers before the accuracy metric's nearest-neighbor
query, but isn't required for the pipeline to run end-to-end or for the metrics in
evaluation/metrics.py to be well-defined.
"""
from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keeps one representative point per occupied voxel.

    Fusing many heavily-overlapping DTU views (49 per scan) back-projects the same
    physical surface many times over, producing tens of millions of near-duplicate
    points -- which does not change point-to-point accuracy/completeness distances
    (evaluation/metrics.py) but makes the nearest-neighbor query pathologically
    slow. Downsampling before the metric is standard practice for exactly this
    reason (not just a speed hack): it removes redundancy the metric is invariant
    to anyway.
    """
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[unique_idx]


def backproject_depth_map(depth: np.ndarray, ref_proj: np.ndarray, min_depth: float = 1e-3) -> np.ndarray:
    """Inverts the (u*z, v*z, z, 1) = ref_proj @ (x, y, z, 1) mapping used to build
    the model's cost volume (see models.geofusionnet.differentiable_homography_warp
    and data.datasets.camera_io.build_projection_matrix) to recover world points
    from a predicted depth map.

    Args:
        depth: (H, W) predicted depth, in the reference camera's z-axis units.
        ref_proj: (4, 4) full projection matrix at the SAME resolution as `depth`.
        min_depth: depths at or below this are treated as invalid and dropped.

    Returns:
        (N, 3) float32 world-space points, N <= H*W.
    """
    H, W = depth.shape
    ys, xs = np.nonzero(depth > min_depth)
    d = depth[ys, xs].astype(np.float64)
    ones = np.ones_like(d)
    h = np.stack([xs.astype(np.float64) * d, ys.astype(np.float64) * d, d, ones], axis=0)  # (4, N)
    world = np.linalg.inv(ref_proj.astype(np.float64)) @ h  # (4, N)
    return world[:3].T.astype(np.float32)


# --- Reference-baseline fusion ---------------------------------------------------
# Port of GeoMVSNet's recommended DTU fusion (GeoMVSNet fusions/dtu/_open3d.py, run by
# scripts/dtu/fusion_dtu.sh with --prob_thresh=0.3 --dist_thresh=0.2 --num_consist=4).
# Same algorithm and thresholds. One deliberate difference: the reference normalizes
# sampling coordinates with the align_corners=True formula but calls grid_sample with
# the (default) align_corners=False, a position-dependent shift of up to 0.5px. At the
# reference's 1600px test width that is ~0.05mm on DTU, but at our 640px it is ~0.2mm --
# the whole dist_thresh -- so it would reject correct points for reasons unrelated to
# the model. `reference_sampling_quirk=True` reproduces it exactly.

GEOMVSNET_PROB_THRESH = 0.3
GEOMVSNET_DIST_THRESH = 0.2   # mm, world-space distance
GEOMVSNET_NUM_CONSIST = 4     # consistent views, counting the reference view itself


def _points_from_depth(depth, proj):
    """depth (B, 1, H, W), proj (B, 4, 4) -> world points (B, 3, H, W)."""
    import torch

    B, _, H, W = depth.shape
    inv = torch.inverse(proj)
    y, x = torch.meshgrid(torch.arange(H, dtype=depth.dtype, device=depth.device),
                          torch.arange(W, dtype=depth.dtype, device=depth.device), indexing="ij")
    xyz = torch.stack((x.reshape(-1), y.reshape(-1), torch.ones(H * W, dtype=depth.dtype, device=depth.device)))
    rot_xyz = torch.matmul(inv[:, :3, :3], xyz.unsqueeze(0).expand(B, -1, -1))
    pts = rot_xyz * depth.view(B, 1, -1) + inv[:, :3, 3:4]
    return pts.view(B, 3, H, W)


def _warp_to_ref(src_fea, src_proj, ref_proj, ref_depth, align_corners: bool = True):
    """Samples per-pixel source values at where each ref pixel (at its ref depth)
    projects into each source view. src_fea (B, C, H, W); ref_depth (1, 1, H, W)."""
    import torch
    import torch.nn.functional as F

    B, C, H, W = src_fea.shape
    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
    y, x = torch.meshgrid(torch.arange(H, dtype=src_fea.dtype, device=src_fea.device),
                          torch.arange(W, dtype=src_fea.dtype, device=src_fea.device), indexing="ij")
    xyz = torch.stack((x.reshape(-1), y.reshape(-1), torch.ones(H * W, dtype=src_fea.dtype, device=src_fea.device)))
    p = torch.matmul(proj[:, :3, :3], xyz.unsqueeze(0).expand(B, -1, -1)) * ref_depth.view(1, 1, -1) + proj[:, :3, 3:4]
    px = p[:, 0] / p[:, 2] / ((W - 1) / 2) - 1
    py = p[:, 1] / p[:, 2] / ((H - 1) / 2) - 1
    grid = torch.stack((px, py), dim=-1).view(B, H, W, 2)
    return F.grid_sample(src_fea, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)


def geometric_consistency_fusion(
    depths: np.ndarray,       # (V, H, W) predicted depth per view
    confidences: np.ndarray,  # (V, H, W) photometric confidence per view
    projs: np.ndarray,        # (V, 4, 4) projection matrices at depth-map resolution
    prob_thresh: float = GEOMVSNET_PROB_THRESH,
    dist_thresh: float = GEOMVSNET_DIST_THRESH,
    num_consist: int = GEOMVSNET_NUM_CONSIST,
    device: str = "cpu",
    batch_size: int = 20,
    reference_sampling_quirk: bool = False,
) -> np.ndarray:
    """GeoMVSNet DTU fusion. For each reference view: zero depths with confidence
    <= prob_thresh, back-project every view (the reference included), sample each
    view's point at the reference pixel's reprojection, count views whose point lies
    within dist_thresh of the reference point, keep pixels with >= num_consist such
    views and output the average of the consistent points. Returns (N, 3) points.
    """
    import torch

    with torch.no_grad():
        d = torch.from_numpy(depths * (confidences > prob_thresh)).float().unsqueeze(1).to(device)
        P = torch.from_numpy(projs).float().to(device)
        V, _, H, W = d.shape
        out = []
        for i in range(V):
            ref_pc = _points_from_depth(d[i:i + 1], P[i:i + 1])
            pc_sum = torch.zeros(3, H, W, device=device)
            cnt = torch.zeros(1, H, W, device=device)
            for j in range(0, V, batch_size):
                src_pcs = _points_from_depth(d[j:j + batch_size], P[j:j + batch_size])
                aligned = _warp_to_ref(src_pcs, P[j:j + batch_size], P[i:i + 1], d[i:i + 1],
                                       align_corners=not reference_sampling_quirk)
                dist = torch.sqrt(((ref_pc - aligned) ** 2).sum(dim=1, keepdim=True))
                m = (dist < dist_thresh).float()
                pc_sum += (aligned * m).sum(dim=0)
                cnt += m.sum(dim=0)
            keep = (cnt >= num_consist)[0]
            avg = (pc_sum / cnt).permute(1, 2, 0)
            out.append(avg[keep].cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)
