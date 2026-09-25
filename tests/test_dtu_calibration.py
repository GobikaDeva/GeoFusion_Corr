"""Guards the DTU loader's camera calibration against the image resolution.

The raw Cameras/{:08d}_cam.txt files are 1600x1200 calibrations; using them at
640x512 put the principal point outside the image and scored even GT depth at
~16mm overall. These checks use DTU's own GT depth, so they are model-independent.
"""
import os

import numpy as np
import pytest

from data.datasets.camera_io import read_cam_file
from data.datasets.dtu import DTUDataset
from data.datasets.pfm_io import read_pfm

DTU_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "DTU")
CFG = {"n_views": 5, "img_wh": [640, 512], "num_depth": 192}

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DTU_ROOT, "Cameras", "train")), reason="DTU data not available"
)


def _dataset():
    return DTUDataset(root=DTU_ROOT, split="test", cfg=CFG)


def test_principal_point_inside_image():
    ds = _dataset()
    cam = read_cam_file(ds._cam_path(0))
    scale = CFG["img_wh"][0] / ds.cam_calib_wh[0]
    cx, cy = cam["intrinsic"][0, 2] * scale, cam["intrinsic"][1, 2] * scale
    assert 0.3 * 640 < cx < 0.7 * 640 and 0.3 * 512 < cy < 0.7 * 512, (cx, cy)


def test_gt_depth_reprojects_consistently_across_views():
    ds = _dataset()
    idx = next(i for i, (scan, ref, _) in enumerate(ds.metas) if scan == "scan1" and ref == 0)
    sample = ds[idx]
    ref_proj = sample["model_inputs"]["ref_proj"].numpy().astype(np.float64)
    src_proj = sample["model_inputs"]["src_projs"][0].numpy().astype(np.float64)
    src_view = sample["src_views"][0]

    # GT depth is stored at cam_calib_wh (160x128); evaluate on that grid.
    s = ds.cam_calib_wh[0] / CFG["img_wh"][0]
    down = np.diag([s, s, 1.0, 1.0])
    ref_depth = read_pfm(ds._depth_path("scan1", 0))
    src_depth = read_pfm(ds._depth_path("scan1", src_view))

    ys, xs = np.nonzero(ref_depth > 0)
    d = ref_depth[ys, xs].astype(np.float64)
    world = np.linalg.inv(down @ ref_proj) @ np.stack([xs * d, ys * d, d, np.ones_like(d)])
    uvz = down @ src_proj @ world
    z = uvz[2]
    u, v = np.round(uvz[0] / z).astype(int), np.round(uvz[1] / z).astype(int)
    h, w = src_depth.shape
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    sd = src_depth[v[inb], u[inb]]
    valid = sd > 0
    assert valid.sum() > 1000, "too few overlapping pixels -- calibration misaligned"
    rel_err = np.abs(z[inb][valid] - sd[valid]) / sd[valid]
    assert np.median(rel_err) < 0.01, np.median(rel_err)
