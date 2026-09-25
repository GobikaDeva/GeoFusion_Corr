import numpy as np
import pytest

from evaluation.fusion import geometric_consistency_fusion
from evaluation.metrics import _radius_downsample_reference, dtu_official_metrics, radius_downsample

H, W, F_PX, Z = 48, 64, 400.0, 500.0


def _cams(n=6, baseline=4.0):
    K = np.eye(4, dtype=np.float32)
    K[0, 0] = K[1, 1] = F_PX
    K[0, 2], K[1, 2] = (W - 1) / 2, (H - 1) / 2
    projs = []
    for i in range(n):
        E = np.eye(4, dtype=np.float32)
        E[0, 3] = -baseline * i  # camera i centred at x = baseline * i
        projs.append(K @ E)
    return np.stack(projs)


def _plane_depths(n):
    return np.full((n, H, W), Z, np.float32)  # fronto-parallel plane z = Z seen by all views


def test_consistent_plane_kept_and_on_plane():
    n = 6
    pts = geometric_consistency_fusion(_plane_depths(n), np.ones((n, H, W), np.float32), _cams(n))
    assert len(pts) > 0.5 * n * H * W  # overlap region survives
    assert np.allclose(pts[:, 2], Z, atol=1e-3)


def test_inconsistent_view_rejected():
    n = 6
    depths = _plane_depths(n)
    depths[0] += 30.0  # view 0 disagrees with every other view by 30mm
    pts = geometric_consistency_fusion(depths, np.ones((n, H, W), np.float32), _cams(n))
    assert not np.any(np.abs(pts[:, 2] - (Z + 30.0)) < 1.0)  # its points never reach 4 consistent views
    assert np.allclose(pts[:, 2], Z, atol=1e-3)


def test_low_confidence_dropped():
    n = 6
    conf = np.ones((n, H, W), np.float32)
    conf[:, :, : W // 2] = 0.1  # below prob_thresh=0.3 in the left half of every view
    pts_all = geometric_consistency_fusion(_plane_depths(n), np.ones((n, H, W), np.float32), _cams(n))
    pts = geometric_consistency_fusion(_plane_depths(n), conf, _cams(n))
    assert 0 < len(pts) < 0.7 * len(pts_all)


def test_num_consist_counts_reference_view():
    # 3 views only: with num_consist=4 nothing can survive; with 3 the overlap does.
    assert len(geometric_consistency_fusion(_plane_depths(3), np.ones((3, H, W), np.float32), _cams(3))) == 0
    assert len(geometric_consistency_fusion(_plane_depths(3), np.ones((3, H, W), np.float32), _cams(3),
                                            num_consist=3)) > 0


def test_radius_downsample_spacing():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 5, size=(4000, 3))
    down = radius_downsample(pts, 0.5)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(down).query(down, k=2)
    assert d[:, 1].min() >= 0.5 - 1e-9
    assert cKDTree(down).query(pts)[0].max() <= 0.5 + 1e-9


def test_radius_downsample_matches_official_loop_exactly():
    rng = np.random.default_rng(1)
    # clustered + uniform points, so many neighbourhoods overlap
    pts = np.vstack([rng.uniform(0, 3, size=(3000, 3)), rng.normal(1.5, 0.1, size=(2000, 3))])
    for radius in (0.05, 0.2, 0.5):
        fast = radius_downsample(pts, radius, seed=7)
        ref = _radius_downsample_reference(pts, radius, seed=7)
        assert np.array_equal(fast, ref)


def _grid_scene(top=1.0):
    xs = np.arange(0, 20, 0.25)
    gt = np.array([[x, y, 0.0] for x in xs for y in xs])
    bb = np.array([[-1, -1, -1], [21, 21, top]], dtype=np.float64)
    res = 0.5
    obs = np.ones(np.ceil((bb[1] - bb[0]) / res).astype(int) + 1, dtype=bool)
    plane = np.array([0, 0, 1.0, 1.0])  # z > -1 is "above"
    return gt, bb, res, obs, plane


def test_official_metric_perfect_prediction():
    gt, bb, res, obs, plane = _grid_scene()
    m = dtu_official_metrics(gt.copy(), gt, obs, bb, res, plane)
    assert m["accuracy"] == pytest.approx(0.0, abs=1e-9)
    assert m["completeness"] < 0.2  # only the 0.2mm downsampling gap remains


def test_official_metric_excludes_not_clips_outliers():
    # One point 25mm off the surface, inside BB and ObsMask: evaluated, but excluded
    # from the accuracy mean (>= max_dist) -- a clipped metric would add 20/N instead.
    gt, bb, res, obs, plane = _grid_scene(top=30.0)
    clean = dtu_official_metrics(gt.copy(), gt, obs, bb, res, plane)
    noisy = dtu_official_metrics(np.vstack([gt, [[10.0, 10.0, 25.0]]]), gt, obs, bb, res, plane)
    assert noisy["n_pred_eval"] == clean["n_pred_eval"] + 1
    assert noisy["accuracy"] == pytest.approx(clean["accuracy"], abs=1e-9)
    assert noisy["precision"] < clean["precision"]  # F-score still sees it
