#!/usr/bin/env python
"""Caches a monocular depth prior for every DTU view (auxiliary geometric prior G).

Runs Depth-Anything-V2-Small once per (scan, view) on the lighting-3 image and stores
its raw output -- RELATIVE inverse depth (affine-invariant: larger = nearer, unknown
scale/shift) -- as float16 at `--out_wh`, under:

    <root>/MonoPrior/<scan>_train/<view:04d>.npy

The prior is lighting-independent by construction (one fixed lighting per view), so
train-time lighting augmentation does not change G. Metric alignment is NOT done here;
see data/datasets/dtu.py.

With `--layout test` (MVSNet dtu-test layout, 1600x1200 images under Rectified/<scan>/)
the prior is written to <prior_root>/<scan>/<view:04d>.npy at 800x600 by default.

Usage:
    python scripts/cache_mono_prior.py --root data/DTU [--scans scan1 scan4] [--overwrite]
    python scripts/cache_mono_prior.py --root <dtu-test-1200> --layout test --prior_root data/DTU_test/MonoPrior
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
NUM_VIEWS = 49
LIGHT = 3
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Per layout: scan dir suffix, Depth-Anything inference (h, w) keeping the image aspect
# ratio at a multiple of the ViT patch (14), and default stored prior size (w, h).
LAYOUTS = {
    "train": {"suffix": "_train", "infer_hw": (518, 644), "out_wh": (320, 256)},
    "test": {"suffix": "", "infer_hw": (518, 686), "out_wh": (800, 600)},
}


def prior_path(prior_root: str, scan_dir: str, view: int) -> str:
    return os.path.join(prior_root, scan_dir, f"{view:04d}.npy")


@torch.no_grad()
def predict(model, img_rgb: np.ndarray, device: str, infer_h: int = 518, infer_w: int = 644) -> torch.Tensor:
    """(H, W, 3) uint8 -> (H, W) relative inverse depth at the input resolution."""
    x = cv2.resize(img_rgb, (infer_w, infer_h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = torch.from_numpy((x - MEAN) / STD).permute(2, 0, 1)[None].to(device)
    out = model(pixel_values=x).predicted_depth  # (1, h, w)
    return F.interpolate(out[None].float(), size=img_rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scans", nargs="*", default=None, help="default: every scan under Rectified/")
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="train")
    ap.add_argument("--prior_root", default=None, help="default: <root>/MonoPrior")
    ap.add_argument("--out_wh", type=int, nargs=2, default=None, help="default: per --layout")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    layout = LAYOUTS[args.layout]
    suffix = layout["suffix"]
    prior_root = args.prior_root or os.path.join(args.root, "MonoPrior")
    scans = args.scans or sorted(
        d[: len(d) - len(suffix)] for d in os.listdir(os.path.join(args.root, "Rectified"))
        if d.startswith("scan") and d.endswith(suffix)
    )
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(args.device).eval()
    out_w, out_h = args.out_wh or layout["out_wh"]
    infer_h, infer_w = layout["infer_hw"]
    for si, scan in enumerate(scans):
        os.makedirs(os.path.join(prior_root, scan + suffix), exist_ok=True)
        for v in range(NUM_VIEWS):
            dst = prior_path(prior_root, scan + suffix, v)
            if os.path.exists(dst) and not args.overwrite:
                continue
            src = os.path.join(args.root, "Rectified", scan + suffix, f"rect_{v + 1:03d}_{LIGHT}_r5000.png")
            img = cv2.cvtColor(cv2.imread(src, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            inv = predict(model, img, args.device, infer_h, infer_w)
            inv = F.interpolate(inv[None, None], size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
            np.save(dst, inv.cpu().numpy().astype(np.float16))
        print(f"[{si + 1}/{len(scans)}] {scan}", flush=True)


if __name__ == "__main__":
    main()
