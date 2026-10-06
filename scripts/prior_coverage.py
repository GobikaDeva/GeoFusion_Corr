#!/usr/bin/env python
"""Does the geometric prior G cover the pixels the model gets wrong, and is it right there?

Pixel sets come from scripts/costvol_profile_probe.py::probe_view (coarse grid,
640x512 val images, lighting 3, every pair.txt reference view):
  valid      GT > 0 and inside the coarse sweep
  wrong      valid and |fine-stage depth - GT| > --wrong_mm
  ambiguous  wrong and the GT bin is only a secondary local minimum of the raw
             (pre-regularizer) cost -- the probe's "competing dips" class
For each set: share of pixels with a valid prior, and the median |G - GT| (mm) over
those. G is nearest-downsampled to the coarse grid, as L_geo does per stage.

Also: valid-prior share per view at 640x512, over all pixels and over GT-valid pixels.

Usage:
    python scripts/prior_coverage.py --config configs/ablations/A0f_fixes_only_short.yaml \
        --ckpt runs/A0f_fixes_only_short/seed2/ckpt_final.pt --out runs/.../prior_coverage.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from costvol_profile_probe import probe_view  # noqa: E402
from data.datasets.dtu import DTUDataset  # noqa: E402
from models import build_model  # noqa: E402


def set_stats(prior_ok: np.ndarray, prior_err: np.ndarray) -> dict:
    n = len(prior_ok)
    return {
        "n_pixels": int(n),
        "prior_valid_pct": 100 * float(prior_ok.mean()) if n else float("nan"),
        "median_abs_prior_err_mm": float(np.median(prior_err[prior_ok])) if prior_ok.any() else float("nan"),
        "pct_prior_err_gt_4mm": 100 * float((prior_err[prior_ok] > 4).mean()) if prior_ok.any() else float("nan"),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scans", nargs="+", default=["scan48", "scan1"])
    ap.add_argument("--wrong_mm", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_views", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False)["model"])
    model.to(args.device).eval()

    summary = {"ckpt": args.ckpt, "wrong_mm": args.wrong_mm, "scans": {}}
    for scan in args.scans:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(scan + "\n")
        ds = DTUDataset(root=cfg["data"]["root"], split="val", cfg={**cfg["data"], "scan_list_file": f.name})
        os.unlink(f.name)
        acc = {k: ([], []) for k in ("valid", "wrong", "ambiguous")}
        per_view = []
        for i in range(min(len(ds), args.max_views or len(ds))):
            sample = ds[i]
            res = probe_view(model, sample, args.device, args.wrong_mm)
            pv, pd, gt = sample["prior_valid"][0], sample["prior_depth"][0], sample["gt_depth"][0]
            gt_ok = gt > 0
            per_view.append({
                "view": i,
                "prior_valid_pct_all": 100 * float(pv.mean()),
                "prior_valid_pct_gt": 100 * float(pv[gt_ok].mean()) if gt_ok.any() else float("nan"),
            })
            scale = model.backbone.cfg.stages[0].resolution_scale
            size = (round(gt.shape[0] * scale), round(gt.shape[1] * scale))
            down = lambda x: F.interpolate(x[None, None], size=size, mode="nearest")[0, 0].numpy()  # noqa: E731
            pv_c, pd_c, gt_c = down(pv) > 0.5, down(pd), down(gt)
            err_c = np.abs(pd_c - gt_c)
            # valid set as the probe defines it: GT > 0 and inside the coarse sweep
            hyps = sample["model_inputs"]["depth_hypotheses_per_stage"]["coarse"].numpy()
            step = hyps[1] - hyps[0]
            valid = (gt_c > 0) & (gt_c > hyps[0] - step / 2) & (gt_c < hyps[-1] + step / 2)
            acc["valid"][0].append(pv_c[valid]); acc["valid"][1].append(err_c[valid])
            ex = res.get("examples")
            if ex is not None:
                ys, xs = ex["ys"], ex["xs"]
                amb = res["wrong"]["raw_mean_class"] == 1
                acc["wrong"][0].append(pv_c[ys, xs]); acc["wrong"][1].append(err_c[ys, xs])
                acc["ambiguous"][0].append(pv_c[ys[amb], xs[amb]]); acc["ambiguous"][1].append(err_c[ys[amb], xs[amb]])
        stats = {k: set_stats(np.concatenate(v[0]) if v[0] else np.zeros(0, bool),
                              np.concatenate(v[1]) if v[1] else np.zeros(0))
                 for k, v in acc.items()}
        all_pct = np.array([v["prior_valid_pct_all"] for v in per_view])
        stats["per_view"] = per_view
        stats["per_view_summary"] = {
            "median_prior_valid_pct_all": float(np.median(all_pct)),
            "min_prior_valid_pct_all": float(all_pct.min()),
            "n_views_no_prior": int((all_pct == 0).sum()),
            "n_views_lt_50pct": int((all_pct < 50).sum()),
        }
        summary["scans"][scan] = stats
        print(scan, json.dumps({k: v for k, v in stats.items() if k != "per_view"}, indent=1), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
