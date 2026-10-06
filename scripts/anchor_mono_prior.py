#!/usr/bin/env python
"""Converts the cached relative monocular prior (scripts/cache_mono_prior.py) into a
METRIC depth prior G by anchoring it to sparse, triangulated points -- no GT used.

Per (scan, ref view):
  1. SIFT-match the ref image against its top `--n_src` pair.txt source views
     (lighting 3, 640x512), Lowe ratio test.
  2. Triangulate each match with the known DTU cameras; keep points in front of both
     cameras, inside the depth sweep, and with reprojection error < `--max_reproj` px
     in both views.
  3. RANSAC-fit the affine inverse-depth model  1/z = s * g + t  (g = relative mono
     prior at the keypoint), then least-squares refit on the inliers.
  4. G = 1 / (s * g + t), valid where finite and inside the depth sweep.

Writes, next to the relative prior:
    <root>/MonoPrior/<scan>_train/<view:04d>_metric.npy   float16 (256, 320) mm, 0 = invalid
    <root>/MonoPrior/<scan>_train/fits.json               per-view s, t, #points, #inliers, residual

With `--anchors_only`, nothing above is rewritten: the fit is recomputed (it is
deterministic), checked against fits.json, and the RANSAC-inlier anchors are saved as
    <root>/MonoPrior/<scan>_train/<view:04d>_anchors.npz   xy (N, 2) SIFT-image px, z (N,) mm
(views without a valid fit get N = 0). Used by scripts/prior_confidence.py.

With `--layout test` (MVSNet dtu-test layout) SIFT runs at 800x600 against the raw
1600x1200 Cameras/*.txt, and the prior is read from / written to <prior_root>/<scan>/.

Usage:
    python scripts/anchor_mono_prior.py --root data/DTU [--scans scan1] [--workers 8]
    python scripts/anchor_mono_prior.py --root <dtu-test-1200> --layout test --prior_root data/DTU_test/MonoPrior
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.datasets.camera_io import build_projection_matrix, read_cam_file  # noqa: E402

NUM_VIEWS = 49
LIGHT = 3
# Per layout: scan dir suffix, SIFT image size, calibration dir and its width.
# Cameras/train intrinsics are at 160x128; the test layout's Cameras/ are at 1600x1200.
LAYOUTS = {
    "train": {"suffix": "_train", "img_wh": (640, 512), "cam_dir": "train", "calib_w": 160},
    "test": {"suffix": "", "img_wh": (800, 600), "cam_dir": "", "calib_w": 1600},
}
NUM_DEPTH = 192
MIN_INLIERS = 50

ARGS = None


def _pair_file(root):
    pairs = {}
    with open(os.path.join(root, "Cameras", "pair.txt")) as f:
        for _ in range(int(f.readline())):
            ref = int(f.readline())
            pairs[ref] = [int(x) for x in f.readline().split()[1::2]]
    return pairs


def _sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample a (h, w) map stored at 1/k of the SIFT image size at its pixel coords."""
    k = LAYOUTS[ARGS.layout]["img_wh"][0] / img.shape[1]
    mx = ((xy[:, 0] + 0.5) / k - 0.5).astype(np.float32)[None]
    my = ((xy[:, 1] + 0.5) / k - 0.5).astype(np.float32)[None]
    return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]


def _ransac_affine_inv(g, z, tau_rel, iters=1000, seed=0):
    """Fits 1/z = s*g + t robustly. Inlier: |1/(s g + t) - z| < tau_rel * z."""
    rng = np.random.default_rng(seed)
    inv_z = 1.0 / z
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(g), 2, replace=False)
        if abs(g[i] - g[j]) < 1e-6:
            continue
        s = (inv_z[i] - inv_z[j]) / (g[i] - g[j])
        t = inv_z[i] - s * g[i]
        if s <= 0:  # larger mono value must mean nearer
            continue
        pred = s * g + t
        inl = (pred > 0) & (np.abs(1.0 / np.maximum(pred, 1e-12) - z) < tau_rel * z)
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_INLIERS:
        return None
    for _ in range(2):  # least-squares refit on inliers, then re-select
        A = np.stack([g[best], np.ones(best.sum())], 1)
        s, t = np.linalg.lstsq(A, inv_z[best], rcond=None)[0]
        pred = s * g + t
        new = (pred > 0) & (np.abs(1.0 / np.maximum(pred, 1e-12) - z) < tau_rel * z)
        if new.sum() < MIN_INLIERS:
            break
        best = new
    return float(s), float(t), best


def process_scan(scan: str) -> str:
    cv2.setNumThreads(1)  # parallelism comes from the scan-level Pool
    root = ARGS.root
    layout = LAYOUTS[ARGS.layout]
    img_wh, scan_dir = layout["img_wh"], scan + layout["suffix"]
    pairs = _pair_file(root)
    scale = img_wh[0] / layout["calib_w"]
    cams = [read_cam_file(os.path.join(root, "Cameras", layout["cam_dir"], f"{v:08d}_cam.txt")) for v in range(NUM_VIEWS)]
    P = [build_projection_matrix(c["intrinsic"], c["extrinsic"], scale=scale)[:3] for c in cams]
    sift = cv2.SIFT_create(nfeatures=8000)
    kps, descs = [], []
    for v in range(NUM_VIEWS):
        img = cv2.imread(os.path.join(root, "Rectified", scan_dir, f"rect_{v + 1:03d}_{LIGHT}_r5000.png"),
                         cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_wh, interpolation=cv2.INTER_AREA)
        kp, d = sift.detectAndCompute(img, None)
        kps.append(np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32))
        descs.append(d)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    fits = {}
    out_dir = os.path.join(ARGS.prior_root, scan_dir)
    if ARGS.anchors_only:
        with open(os.path.join(out_dir, "fits.json")) as f:
            old_fits = json.load(f)
    for ref in range(NUM_VIEWS):
        dmin = cams[ref]["depth_min"]
        dmax = dmin + NUM_DEPTH * cams[ref]["depth_interval"]
        pts, zs = [], []
        for src in pairs[ref][: ARGS.n_src]:
            if descs[ref] is None or descs[src] is None:
                continue
            ms = [m for m, n in matcher.knnMatch(descs[ref], descs[src], k=2) if m.distance < 0.75 * n.distance]
            if len(ms) < 8:
                continue
            x1 = kps[ref][[m.queryIdx for m in ms]]
            x2 = kps[src][[m.trainIdx for m in ms]]
            Xh = cv2.triangulatePoints(P[ref], P[src], x1.T.astype(np.float64), x2.T.astype(np.float64))
            X = np.vstack([Xh[:3] / Xh[3:], np.ones((1, Xh.shape[1]))])
            ok = np.ones(len(ms), bool)
            for Pm, x in ((P[ref], x1), (P[src], x2)):
                p = Pm @ X
                ok &= p[2] > 0
                ok &= np.linalg.norm(p[:2] / p[2:] - x.T, axis=0) < ARGS.max_reproj
            z = (cams[ref]["extrinsic"][:3] @ X)[2]
            ok &= (z > dmin) & (z < dmax)
            pts.append(x1[ok]); zs.append(z[ok])
        pts = np.concatenate(pts) if pts else np.zeros((0, 2))
        zs = np.concatenate(zs) if zs else np.zeros(0)

        g_rel = np.load(os.path.join(out_dir, f"{ref:04d}.npy")).astype(np.float32)
        fit = _ransac_affine_inv(_sample(g_rel, pts), zs, ARGS.tau_rel, seed=ref) if len(zs) >= MIN_INLIERS else None
        if ARGS.anchors_only:
            cached = old_fits[str(ref)]
            if (fit is None) != (not cached["valid"]) or (
                    fit is not None and not np.allclose([fit[0], fit[1]], [cached["s"], cached["t"]], rtol=1e-4)):
                raise RuntimeError(f"{scan} view {ref}: recomputed fit differs from fits.json")
            inl = fit[2] if fit is not None else np.zeros(len(zs), bool)
            np.savez(os.path.join(out_dir, f"{ref:04d}_anchors.npz"),
                     xy=pts[inl].astype(np.float32), z=zs[inl].astype(np.float32))
            continue
        if fit is None:
            G = np.zeros_like(g_rel)
            fits[ref] = {"n_points": int(len(zs)), "n_inliers": 0, "valid": False}
        else:
            s, t, inl = fit
            inv = s * g_rel + t
            G = np.where(inv > 0, 1.0 / np.maximum(inv, 1e-12), 0.0)
            G = np.where((G > dmin) & (G < dmax), G, 0.0)
            res = np.abs(1.0 / (s * _sample(g_rel, pts[inl]) + t) - zs[inl])
            fits[ref] = {"s": s, "t": t, "n_points": int(len(zs)), "n_inliers": int(inl.sum()),
                         "median_abs_res_mm": float(np.median(res)), "valid_frac": float((G > 0).mean()), "valid": True}
        np.save(os.path.join(out_dir, f"{ref:04d}_metric.npy"), G.astype(np.float16))
    if ARGS.anchors_only:
        return f"{scan}: anchors saved, fits match fits.json"
    with open(os.path.join(out_dir, "fits.json"), "w") as f:
        json.dump(fits, f, indent=1)
    n_ok = sum(v["valid"] for v in fits.values())
    return f"{scan}: {n_ok}/{NUM_VIEWS} views anchored"


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scans", nargs="*", default=None)
    ap.add_argument("--n_src", type=int, default=4)
    ap.add_argument("--max_reproj", type=float, default=1.5)
    ap.add_argument("--tau_rel", type=float, default=0.01, help="RANSAC inlier threshold, fraction of depth")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="train")
    ap.add_argument("--prior_root", default=None, help="default: <root>/MonoPrior")
    ap.add_argument("--anchors_only", action="store_true",
                    help="only save the inlier anchors (see docstring); the prior is not rewritten")
    ARGS = ap.parse_args()
    ARGS.prior_root = ARGS.prior_root or os.path.join(ARGS.root, "MonoPrior")
    suffix = LAYOUTS[ARGS.layout]["suffix"]
    scans = ARGS.scans or sorted(
        d[: len(d) - len(suffix)] for d in os.listdir(ARGS.prior_root) if d.startswith("scan") and d.endswith(suffix)
    )
    with Pool(min(ARGS.workers, len(scans))) as pool:
        for msg in pool.imap_unordered(process_scan, scans):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
