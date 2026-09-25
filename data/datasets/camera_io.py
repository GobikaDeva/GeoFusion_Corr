"""MVSNet-format DTU camera (.txt) file parser.

Standard cam.txt layout (confirmed against the user's DTU copy at
data/DTU/Cameras/{:08d}_cam.txt):

    extrinsic
    E00 E01 E02 E03
    E10 E11 E12 E13
    E20 E21 E22 E23
    E30 E31 E32 E33

    intrinsic
    K00 K01 K02
    K10 K11 K12
    K20 K21 K22

    DEPTH_MIN DEPTH_INTERVAL [NUM_DEPTH DEPTH_MAX]

Returns extrinsic (world-to-camera, 4x4), intrinsic (3x3), and the depth range.
"""
from __future__ import annotations

import numpy as np


def read_cam_file(path: str) -> dict:
    with open(path) as f:
        lines = [line.rstrip() for line in f.readlines()]

    assert lines[0] == "extrinsic", f"Unexpected cam file header in {path}: {lines[0]}"
    extrinsic = np.array(
        [[float(x) for x in lines[i].split()] for i in range(1, 5)], dtype=np.float32
    )

    assert lines[6] == "intrinsic", f"Unexpected cam file layout in {path}: {lines[6]}"
    intrinsic = np.array(
        [[float(x) for x in lines[i].split()] for i in range(7, 10)], dtype=np.float32
    )

    depth_line = lines[11].split() if len(lines) > 11 else lines[10].split()
    depth_min = float(depth_line[0])
    depth_interval = float(depth_line[1]) if len(depth_line) > 1 else 1.0
    num_depth = int(float(depth_line[2])) if len(depth_line) > 2 else None
    depth_max = float(depth_line[3]) if len(depth_line) > 3 else None

    return {
        "extrinsic": extrinsic,        # (4, 4) world-to-camera
        "intrinsic": intrinsic,        # (3, 3)
        "depth_min": depth_min,
        "depth_interval": depth_interval,
        "num_depth": num_depth,
        "depth_max": depth_max,
    }


def build_projection_matrix(intrinsic: np.ndarray, extrinsic: np.ndarray, scale=1.0) -> np.ndarray:
    """Combines intrinsic + extrinsic into a single 4x4 projection matrix, with the
    intrinsic's focal length / principal point scaled to match a resized image
    (e.g. img_wh in the dataset config vs. the camera file's native resolution).
    `scale` is a scalar or an (x, y) pair for an anisotropic resize.
    """
    sx, sy = scale if isinstance(scale, (tuple, list)) else (scale, scale)
    intrinsic_scaled = intrinsic.copy()
    intrinsic_scaled[0, :] *= sx
    intrinsic_scaled[1, :] *= sy
    proj = np.eye(4, dtype=np.float32)
    proj[:3, :3] = intrinsic_scaled
    return proj @ extrinsic
