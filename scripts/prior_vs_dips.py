#!/usr/bin/env python
"""Could a prior of this quality pick the right cost dip where the model fails?

Pixel sets are scripts/costvol_profile_probe.py::probe_view's (coarse grid, 640x512
val images, lighting 3, every pair.txt reference view):
  wrong      GT > 0, inside the coarse sweep, |final depth - GT| > --wrong_mm
  ambiguous  wrong and the GT bin is only a secondary local minimum of the raw
             (pre-regularizer) coarse cost -- "competing dips"
G is nearest-downsampled to the coarse grid (as L_geo does per stage).

1. wrong & prior-valid: share where |G - GT| < |D - GT| (D = final prediction).
2. ambiguous & prior-valid, on the raw coarse cost:
     true dip    lowest-cost bin within +-1 bin of GT (the probe's GT minimum)
     chosen dip  the local minimum nearest the model's final prediction
     separation  |depth(chosen dip) - depth(true dip)| in mm (0 when they coincide)
   reports separation median / p25 / p75, median |G - GT|, and
     sep_gt_prior_err_pct   separation > |G - GT|   (the requested criterion)
     prior_nearer_true_pct  |G - true dip| < |G - chosen dip|, the condition for G to
                            actually prefer the true dip (stricter)
3. the same, restricted to pixels where |G - GT| < 4mm.

Usage:
    python scripts/prior_vs_dips.py --config configs/ablations/A0f_fixes_only_short.yaml \
        --ckpt runs/A0f_fixes_only_short/seed2/ckpt_final.pt --out runs/prior_vs_dips/A0f_seed2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from costvol_profile_probe import probe_view  # noqa: E402
from data.datasets.dtu import DTUDataset  # noqa: E402
from models import build_model  # noqa: E402


def dips(raw: np.ndarray, hyps: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    """raw (N, D) lower = better; hyps (D,); gt, pred (N,) mm.
    Returns (true dip depth, chosen dip depth), each (N,)."""
    N, D = raw.shape
    step = hyps[1] - hyps[0]
    gt_bin = np.clip(np.round((gt - hyps[0]) / step).astype(int), 0, D - 1)
    near = np.clip(gt_bin[:, None] + np.array([-1, 0, 1])[None], 0, D - 1)
    true_bin = near[np.arange(N), raw[np.arange(N)[:, None], near].argmin(1)]
    left = np.concatenate([np.full((N, 1), np.inf), raw[:, :-1]], 1)
    right = np.concatenate([raw[:, 1:], np.full((N, 1), np.inf)], 1)
    is_min = (raw <= left) & (raw <= right)
    pred_bin = (pred - hyps[0]) / step
    dist = np.where(is_min, np.abs(np.arange(D)[None] - pred_bin[:, None]), np.inf)
    chosen_bin = dist.argmin(1)
    return hyps[true_bin], hyps[chosen_bin]


def wrong_stats(e_prior, e_pred):
    n = len(e_prior)
    return {"n_pixels": int(n),
            "prior_closer_pct": 100 * float((e_prior < e_pred).mean()) if n else float("nan")}


def dip_stats(sep, e_prior, d_true, d_chosen, g):
    n = len(sep)
    if n == 0:
        return {"n_pixels": 0}
    return {
        "n_pixels": int(n),
        "same_dip_pct": 100 * float((sep == 0).mean()),
        "sep_median_mm": float(np.median(sep)),
        "sep_p25_mm": float(np.percentile(sep, 25)),
        "sep_p75_mm": float(np.percentile(sep, 75)),
        "prior_err_median_mm": float(np.median(e_prior)),
        "sep_gt_prior_err_pct": 100 * float((sep > e_prior).mean()),
        "prior_nearer_true_pct": 100 * float((np.abs(g - d_true) < np.abs(g - d_chosen)).mean()),
    }


def summarize(c, wrong_mm=4.0):
    pv = c["w_pv"]
    out = {"wrong_n": int(len(pv)), "wrong_prior_valid_pct": 100 * float(pv.mean()) if len(pv) else float("nan")}
    out["q1_wrong"] = wrong_stats(c["w_ep"][pv], c["w_ed"][pv])
    a = c["a_pv"]
    sep = np.abs(c["a_chosen"] - c["a_true"])
    out["ambiguous_n"] = int(len(a))
    out["ambiguous_prior_valid_pct"] = 100 * float(a.mean()) if len(a) else float("nan")
    out["q2_ambiguous"] = dip_stats(sep[a], c["a_ep"][a], c["a_true"][a], c["a_chosen"][a], c["a_g"][a])
    good_w = pv & (c["w_ep"] < 4.0)
    good_a = a & (c["a_ep"] < 4.0)
    out["q3_prior_within_4mm"] = {
        "wrong_pct_of_prior_valid": 100 * float(good_w.sum() / max(pv.sum(), 1)),
        "ambiguous_pct_of_prior_valid": 100 * float(good_a.sum() / max(a.sum(), 1)),
        "wrong": wrong_stats(c["w_ep"][good_w], c["w_ed"][good_w]),
        "ambiguous": dip_stats(sep[good_a], c["a_ep"][good_a], c["a_true"][good_a], c["a_chosen"][good_a], c["a_g"][good_a]),
    }
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scan_list", default="data/datasets/splits/dtu_test.txt")
    ap.add_argument("--wrong_mm", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_views", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False)["model"])
    model.to(args.device).eval()
    scans = [s.strip() for s in open(args.scan_list) if s.strip()]
    keys = ("w_pv", "w_ep", "w_ed", "a_pv", "a_ep", "a_g", "a_true", "a_chosen")

    out = {"ckpt": args.ckpt, "wrong_mm": args.wrong_mm, "scans": {}}
    pooled = {k: [] for k in keys}
    for scan in scans:
        t0 = time.time()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(scan + "\n")
        ds = DTUDataset(root=cfg["data"]["root"], split="val", cfg={**cfg["data"], "scan_list_file": f.name})
        os.unlink(f.name)
        cols = {k: [] for k in keys}
        for i in range(min(len(ds), args.max_views or len(ds))):
            sample = ds[i]
            res = probe_view(model, sample, args.device, args.wrong_mm)
            ex = res.get("examples")
            if ex is None:
                continue
            scale = model.backbone.cfg.stages[0].resolution_scale
            size = (round(sample["gt_depth"].shape[-2] * scale), round(sample["gt_depth"].shape[-1] * scale))
            down = lambda x: F.interpolate(x[None], size=size, mode="nearest")[0, 0].numpy()  # noqa: E731
            pv_c, g_c = down(sample["prior_valid"]) > 0.5, down(sample["prior_depth"])
            ys, xs, gt, pred = ex["ys"], ex["xs"], ex["gt"], ex["fine_pred"]
            g, pv = g_c[ys, xs], pv_c[ys, xs]
            amb = res["wrong"]["raw_mean_class"] == 1
            d_true, d_chosen = dips(ex["raw_mean"][amb], ex["hyps"], gt[amb], pred[amb])
            for k, v in (("w_pv", pv), ("w_ep", np.abs(g - gt)), ("w_ed", np.abs(pred - gt)),
                         ("a_pv", pv[amb]), ("a_ep", np.abs(g - gt)[amb]), ("a_g", g[amb]),
                         ("a_true", d_true), ("a_chosen", d_chosen)):
                cols[k].append(v)
        c = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
        for k in ("w_pv", "a_pv"):
            c[k] = c[k].astype(bool)
        for k in keys:
            pooled[k].append(c[k])
        s = summarize(c, args.wrong_mm)
        out["scans"][scan] = s
        q2 = s["q2_ambiguous"]
        print(f"{scan:<8} wrong {s['wrong_n']:>7} prior-valid {s['wrong_prior_valid_pct']:5.1f}% "
              f"G closer {s['q1_wrong']['prior_closer_pct']:5.1f}% | amb {s['ambiguous_n']:>6} "
              f"sep med {q2.get('sep_median_mm', float('nan')):5.1f} |G-GT| med {q2.get('prior_err_median_mm', float('nan')):5.1f} "
              f"sep>err {q2.get('sep_gt_prior_err_pct', float('nan')):5.1f}% G nearer true {q2.get('prior_nearer_true_pct', float('nan')):5.1f}% "
              f"({time.time() - t0:.0f}s)", flush=True)
    c = {k: np.concatenate(v) for k, v in pooled.items()}
    for k in ("w_pv", "a_pv"):
        c[k] = c[k].astype(bool)
    out["pooled_all_scans"] = summarize(c, args.wrong_mm)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("pooled", json.dumps(out["pooled_all_scans"], indent=1), flush=True)


if __name__ == "__main__":
    main()
