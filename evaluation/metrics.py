"""Core reconstruction metrics: accuracy, completeness, overall distance, F-score.

Matches docs/evaluation_and_gates.md's "Core Metrics" table. These operate on point
clouds (post depth-fusion), consistent with the standard DTU evaluation protocol
(distance-based accuracy/completeness against the DTU reference point clouds).
"""
from __future__ import annotations

from typing import Optional

import numba
import numpy as np
from scipy.spatial import cKDTree


def chamfer_distances(pred_points: np.ndarray, gt_points: np.ndarray, max_dist: Optional[float] = None) -> dict:
    """Returns per-point nearest-neighbor distances in both directions.

    accuracy: for each predicted point, distance to nearest GT point (lower is better).
    completeness: for each GT point, distance to nearest predicted point (lower is better).

    `max_dist`: distances beyond it are returned as inf. Callers that clip or exclude
    at a threshold should pass it -- a nearest-neighbour search from a point far off a
    densely sampled surface visits a huge number of candidates, which made unfiltered
    DTU clouds take ~30 min per scan.
    """
    bound = np.inf if max_dist is None else max_dist
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)

    acc_dists, _ = gt_tree.query(pred_points, k=1, distance_upper_bound=bound, workers=-1)
    comp_dists, _ = pred_tree.query(gt_points, k=1, distance_upper_bound=bound, workers=-1)

    return {"accuracy_dists": acc_dists, "completeness_dists": comp_dists}


def overall_distance(accuracy_dists: np.ndarray, completeness_dists: np.ndarray) -> dict:
    accuracy = float(np.mean(accuracy_dists))
    completeness = float(np.mean(completeness_dists))
    overall = 0.5 * (accuracy + completeness)
    return {"accuracy": accuracy, "completeness": completeness, "overall": overall}


def f_score(accuracy_dists: np.ndarray, completeness_dists: np.ndarray, threshold: float = 2.0) -> dict:
    """F-score at a distance threshold (mm for DTU-scale point clouds), matching the
    "Generalization" row of the core-metrics table."""
    precision = float(np.mean(accuracy_dists < threshold))
    recall = float(np.mean(completeness_dists < threshold))
    denom = precision + recall
    f = 0.0 if denom == 0 else 2 * precision * recall / denom
    return {"precision": precision, "recall": recall, "f_score": f, "threshold": threshold}


def masked_depth_error(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray) -> dict:
    """Absolute depth error restricted to a region mask (used for hard-region metrics
    and calibration -- see hard_region_masks.py and calibration.py)."""
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return {"mae": float("nan"), "n_valid": 0}
    err = np.abs(pred_depth[mask] - gt_depth[mask])
    return {"mae": float(err.mean()), "n_valid": int(mask.sum())}


def _radius_downsample_reference(points: np.ndarray, radius: float = 0.2, seed: int = 0) -> np.ndarray:
    """DTUeval-python's greedy downsampling, verbatim in structure (slow, O(N) Python
    loop): shuffle, then keep a point and drop every other point within `radius` of
    it, in shuffled order. Kept as the reference that `radius_downsample` is tested
    against."""
    pts = points.copy()
    np.random.default_rng(seed).shuffle(pts, axis=0)
    tree = cKDTree(pts)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if keep[i]:
            keep[tree.query_ball_point(pts[i], radius)] = False
            keep[i] = True
    return pts[keep]


@numba.njit(cache=True)
def _greedy_keep(pts, order, starts, ends, keys_sorted, cells, nx, ny, nz, radius):
    r2 = radius * radius
    n = pts.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        cx, cy, cz = cells[i, 0], cells[i, 1], cells[i, 2]
        ok = True
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    x, y, z = cx + dx, cy + dy, cz + dz
                    if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
                        continue
                    key = (x * ny + y) * nz + z
                    k = np.searchsorted(keys_sorted, key)
                    if k >= keys_sorted.shape[0] or keys_sorted[k] != key:
                        continue
                    for m in range(starts[k], ends[k]):
                        j = order[m]
                        if keep[j]:
                            d0 = pts[i, 0] - pts[j, 0]
                            d1 = pts[i, 1] - pts[j, 1]
                            d2 = pts[i, 2] - pts[j, 2]
                            if d0 * d0 + d1 * d1 + d2 * d2 <= r2:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        keep[i] = ok
    return keep


def radius_downsample(points: np.ndarray, radius: float = 0.2, seed: int = 0) -> np.ndarray:
    """Exactly the DTUeval-python greedy downsampling (same kept set as
    `_radius_downsample_reference`), compiled: after the shuffle, a point is kept iff
    no EARLIER kept point lies within `radius` -- equivalent to the official "keep,
    then drop all neighbours" loop because the radius relation is symmetric. Kept
    points are found through a radius-sized cell grid. (The official script leaves
    the shuffle unseeded; a fixed seed keeps our numbers reproducible.)"""
    pts = points.astype(np.float64, copy=True)
    np.random.default_rng(seed).shuffle(pts, axis=0)
    if len(pts) == 0:
        return pts
    cells = np.floor((pts - pts.min(axis=0)) / radius).astype(np.int64)
    nx, ny, nz = (cells.max(axis=0) + 1).tolist()
    keys = (cells[:, 0] * ny + cells[:, 1]) * nz + cells[:, 2]
    order = np.argsort(keys, kind="stable")
    keys_sorted, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    keep = _greedy_keep(pts, order, starts, starts + counts, keys_sorted, cells, nx, ny, nz, radius)
    return pts[keep]


def dtu_official_metrics(
    pred_points: np.ndarray,
    stl_points: np.ndarray,
    obs_mask: np.ndarray,
    bb: np.ndarray,
    res: float,
    plane: np.ndarray,
    downsample_density: float = 0.2,
    patch_size: float = 60.0,
    max_dist: float = 20.0,
    f_threshold: float = 2.0,
    seed: int = 0,
) -> dict:
    """Official DTU point-cloud protocol, as in DTUeval-python (the Python port of the
    MATLAB evaluation used by MVSNet/CasMVSNet/GeoMVSNet):

      accuracy     = mean over d < max_dist of dist(pred -> ALL stl points), where pred is
                     radius-downsampled then kept inside BB (+patch) and the ObsMask;
      completeness = mean over d < max_dist of dist(stl above Plane -> pred inside BB+patch).

    Distances >= max_dist are EXCLUDED (not clipped). F-score (not part of the official
    protocol) uses the same point sets without the max_dist exclusion.
    """
    bb = bb.astype(np.float32)
    data = radius_downsample(pred_points.astype(np.float64), downsample_density, seed)

    inbound = ((data >= bb[:1] - patch_size) & (data < bb[1:] + patch_size * 2)).sum(axis=-1) == 3
    data_in = data[inbound]
    grid = np.around((data_in - bb[:1]) / res).astype(np.int32)
    grid_inbound = ((grid >= 0) & (grid < np.expand_dims(obs_mask.shape, 0))).sum(axis=-1) == 3
    g = grid[grid_inbound]
    in_obs = obs_mask[g[:, 0], g[:, 1], g[:, 2]].astype(bool)
    data_in_obs = data_in[grid_inbound][in_obs]

    # Distances >= max_dist are excluded below, so the search can stop there (inf).
    d2s, _ = cKDTree(stl_points).query(data_in_obs, k=1, distance_upper_bound=max_dist, workers=-1)
    stl_hom = np.concatenate([stl_points, np.ones_like(stl_points[:, :1])], -1)
    stl_above = stl_points[(plane.reshape(1, 4) * stl_hom).sum(-1) > 0]
    s2d, _ = cKDTree(data_in).query(stl_above, k=1, distance_upper_bound=max_dist, workers=-1)

    accuracy = float(d2s[d2s < max_dist].mean())
    completeness = float(s2d[s2d < max_dist].mean())
    out = {"accuracy": accuracy, "completeness": completeness, "overall": 0.5 * (accuracy + completeness),
           "n_pred_eval": int(len(data_in_obs)), "n_gt_eval": int(len(stl_above))}
    out.update(f_score(d2s, s2d, f_threshold))
    return out
