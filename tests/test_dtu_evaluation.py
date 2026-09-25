"""Unit tests for the DTU evaluation wiring added to evaluation/eval_dtu.py:
depth-map back-projection (evaluation/fusion.py) and ground-truth loading/filtering
(evaluation/dtu_gt.py). Uses synthetic fixtures only -- the real DTU GT files (STL
points / ObsMask / Plane) live outside this repo and are not required to run tests.
"""
import struct

import numpy as np

from evaluation.dtu_gt import filter_by_obs_mask, filter_by_plane, read_ply_points
from evaluation.fusion import backproject_depth_map, voxel_downsample


def test_backproject_depth_map_roundtrip():
    """Forward-projecting the recovered world points through the same camera should
    reproduce the exact (u*d, v*d, d, 1) values the depth map was built from --
    this is the inverse of the SAME projection convention used by
    models.geofusionnet.differentiable_homography_warp."""
    H, W = 4, 5
    K = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.array([
        [0.9363, -0.2896, 0.1987],
        [0.2896, 0.9564, 0.0295],
        [-0.1987, 0.0295, 0.9797],
    ])
    extrinsic[:3, 3] = [1.0, -2.0, 5.0]
    proj = np.eye(4)
    proj[:3, :3] = K
    proj = proj @ extrinsic

    depth = np.random.RandomState(0).uniform(1.0, 10.0, size=(H, W)).astype(np.float32)
    points = backproject_depth_map(depth, proj)
    assert points.shape == (H * W, 3)

    ones = np.ones((points.shape[0], 1))
    world_h = np.concatenate([points, ones], axis=1)
    reproj = (proj @ world_h.T).T

    ys, xs = np.nonzero(depth > 1e-3)
    d = depth[ys, xs].astype(np.float64)
    expected = np.stack([xs * d, ys * d, d, np.ones_like(d)], axis=1)
    assert np.allclose(reproj, expected, atol=1e-3)


def test_backproject_depth_map_drops_invalid_depths():
    depth = np.array([[0.0, 2.0], [-1.0, 3.0]], dtype=np.float32)
    proj = np.eye(4)
    points = backproject_depth_map(depth, proj)
    assert points.shape == (2, 3)  # only the two positive-depth pixels survive


def test_voxel_downsample_collapses_duplicates_in_same_voxel():
    points = np.array([
        [0.1, 0.1, 0.1], [0.4, 0.2, 0.3],  # both in voxel (0,0,0) at voxel_size=1.0
        [5.0, 5.0, 5.0],  # its own voxel
    ])
    result = voxel_downsample(points, voxel_size=1.0)
    assert result.shape[0] == 2  # one representative per occupied voxel


def test_voxel_downsample_empty_input():
    assert voxel_downsample(np.zeros((0, 3)), voxel_size=1.0).shape == (0, 3)


def test_filter_by_obs_mask_keeps_only_observed_voxels():
    obs_mask = np.zeros((3, 3, 3), dtype=bool)
    obs_mask[1, 1, 1] = True
    bb = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    points = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    keep = filter_by_obs_mask(points, obs_mask, bb, res=1.0)
    assert keep.tolist() == [True, False, False]


def test_filter_by_plane_keeps_foreground_side():
    plane = np.array([0.0, 0.0, 1.0, 0.0])  # z=0 plane; foreground is the +z side (fixed convention)
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    keep = filter_by_plane(points, plane)
    assert keep.tolist() == [True, False]


def test_read_ply_points_binary(tmp_path):
    verts = [(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 10, 20, 30), (4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 40, 50, 60)]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    body = b"".join(struct.pack("<6f3B", *v) for v in verts)

    path = tmp_path / "tiny.ply"
    with open(path, "wb") as f:
        f.write(header + body)

    pts = read_ply_points(str(path))
    assert pts.shape == (2, 3)
    assert np.allclose(pts, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
