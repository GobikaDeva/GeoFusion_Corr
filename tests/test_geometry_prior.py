import numpy as np

from data.datasets.dtu import depth_to_normals

K = np.array([[500.0, 0, 64], [0, 500.0, 48], [0, 0, 1]], dtype=np.float32)


def test_fronto_parallel_plane_faces_camera():
    n = depth_to_normals(np.full((96, 128), 600.0, np.float32), K)
    inner = n[:, 1:-1, 1:-1].reshape(3, -1)
    assert np.allclose(inner, np.array([[0], [0], [-1]]), atol=1e-5)
    assert np.all(n[:, 0, :] == 0) and np.all(n[:, :, -1] == 0)  # borders invalid


def test_tilted_plane_matches_analytic_normal():
    # Plane z = 600 + 0.5 * X  ->  normal ∝ (0.5, 0, -1).
    h, w = 96, 128
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    a = (u - K[0, 2]) / K[0, 0]
    depth = 600.0 / (1 - 0.5 * a)
    n = depth_to_normals(depth.astype(np.float32), K)[:, 1:-1, 1:-1].reshape(3, -1)
    expected = np.array([0.5, 0.0, -1.0]) / np.linalg.norm([0.5, 0.0, -1.0])
    assert np.allclose(n, expected[:, None], atol=1e-3)


def test_invalid_depth_zeroes_normal_and_neighbours():
    depth = np.full((32, 32), 600.0, np.float32)
    depth[10, 10] = 0
    n = depth_to_normals(depth, K)
    for y, x in [(10, 10), (9, 10), (11, 10), (10, 9), (10, 11)]:
        assert np.all(n[:, y, x] == 0)
    assert np.allclose(n[:, 20, 20], [0, 0, -1], atol=1e-5)
