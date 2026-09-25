"""Official DTU ground-truth loading: STL reference point clouds, ObsMask
(observed-region voxel grid) and Plane (foreground half-space) filtering.

Matches the standard DTU MVS evaluation protocol's inputs (the same "SampleSet" /
"Points" data used by the reference GeoMVS baseline), so accuracy/completeness/
overall/F-score (evaluation/metrics.py) are computed on the same footing as the
published numbers -- not a self-consistent proxy.

Expected layout (confirmed against the user's local GT copy):
    <gt_root>/Points/stl/stl{scan:03d}_total.ply   -- laser-scanned reference points
    <gt_root>/ObsMask/ObsMask{scan}_10.mat         -- {ObsMask, BB, Res} observed-voxel grid
    <gt_root>/Plane/Plane{scan}.mat                -- {P} foreground/background plane
"""
from __future__ import annotations

import os
import struct

import numpy as np
import scipy.io as sio

_PLY_TYPE_MAP = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "short": "i2", "ushort": "u2", "uchar": "u1", "char": "i1", "uint8": "u1", "int8": "i1",
}


def read_ply_points(path: str) -> np.ndarray:
    """Reads vertex x/y/z from a binary_little_endian or ascii PLY file.

    Only vertex positions are needed for point-to-point distance metrics, so other
    per-vertex properties (normals, color) are parsed (to keep the record layout
    correct) but discarded.
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="strict").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        header_end = f.tell()

        fmt = None
        n_vertices = 0
        properties = []
        in_vertex_element = False
        for line in header_lines:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                in_vertex_element = tokens[1] == "vertex"
                if in_vertex_element:
                    n_vertices = int(tokens[2])
            elif tokens[0] == "property" and in_vertex_element:
                properties.append((tokens[1], tokens[2]))  # (type, name)
            elif tokens[0] == "element":
                in_vertex_element = False

        if fmt is None or n_vertices == 0:
            raise ValueError(f"Could not parse PLY header in {path}")

        xyz_indices = [i for i, (_, name) in enumerate(properties) if name in ("x", "y", "z")]
        if len(xyz_indices) != 3:
            raise ValueError(f"PLY {path} does not have x/y/z vertex properties")

        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n_vertices)
            return data[:, xyz_indices].astype(np.float32)

        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")
        endian = "<" if fmt == "binary_little_endian" else ">"

        dtype = np.dtype([(name, endian + _PLY_TYPE_MAP[ptype]) for ptype, name in properties])
        f.seek(header_end)
        raw = np.fromfile(f, dtype=dtype, count=n_vertices)
        xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
        return xyz


def load_obs_mask(gt_root: str, scan_num: int) -> dict:
    path = os.path.join(gt_root, "ObsMask", f"ObsMask{scan_num}_10.mat")
    mat = sio.loadmat(path)
    return {
        "mask": mat["ObsMask"].astype(bool),  # (Dx, Dy, Dz)
        "bb": mat["BB"].astype(np.float64),  # (2, 3): [min; max]
        "res": float(np.asarray(mat["Res"]).reshape(-1)[0]),
    }


def load_plane(gt_root: str, scan_num: int) -> np.ndarray:
    path = os.path.join(gt_root, "Plane", f"Plane{scan_num}.mat")
    mat = sio.loadmat(path)
    return mat["P"].reshape(4).astype(np.float64)


def filter_by_obs_mask(points: np.ndarray, obs_mask: np.ndarray, bb: np.ndarray, res: float) -> np.ndarray:
    """Keeps only points that fall inside the observed-region voxel grid.

    `bb` is [min_xyz; max_xyz] (world units matching the STL points), `res` is the
    grid's voxel size, and `obs_mask` is indexed [x, y, z].
    """
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    idx = np.round((points - bb[0]) / res).astype(np.int64)
    shape = np.array(obs_mask.shape)
    in_bounds = np.all((idx >= 0) & (idx < shape), axis=1)
    keep = np.zeros(points.shape[0], dtype=bool)
    valid_idx = idx[in_bounds]
    keep[in_bounds] = obs_mask[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
    return keep


def filter_by_plane(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Keeps points on the foreground side of `plane` (ax+by+cz+d >= 0), matching the
    official DTU evaluation convention -- Plane.mat's sign is fixed by construction,
    not something to (re-)calibrate per scan.

    An earlier version of this function inferred the sign from the median of a
    reference point set (typically the GT STL points). That heuristic is fragile:
    scan29's raw STL cloud has a near-zero median (-3.09mm) because it includes a
    sizeable amount of background/platform geometry roughly balanced across the
    plane, so the median landed (barely) on the wrong side and inverted the filter
    for that scan -- rejecting 100% of valid foreground points instead of keeping
    them. Verified against scan1/4/9/11 and scan29's own real (non-predicted) depth:
    the true foreground side is always positive, so no per-scan calibration is
    needed or correct.
    """
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    normal, d = plane[:3], plane[3]
    side = points @ normal + d
    return side >= 0


def load_scan_ground_truth(gt_root: str, scan_num: int) -> dict:
    stl_points = read_ply_points(os.path.join(gt_root, "Points", "stl", f"stl{scan_num:03d}_total.ply"))
    obs = load_obs_mask(gt_root, scan_num)
    plane = load_plane(gt_root, scan_num)
    return {"stl_points": stl_points, "obs_mask": obs["mask"], "bb": obs["bb"], "res": obs["res"], "plane": plane}
