#!/usr/bin/env python
"""Per-scan quality of the geometric prior G against GT and against the model's depth.

Runs on the train-layout 640x512 images (lighting 3, every pair.txt reference view),
where DTU GT depth exists for the test scans too. The prior is the train-layout
build (data/DTU/MonoPrior); the full-res test inputs use a separate build
(data/DTU_test/MonoPrior) that has no GT depth to score against.

Pixels: GT > 0. Per scan, pooled over views:
  prior_valid_pct           share of GT pixels with a valid prior
  med_prior_vs_model_mm     median |G - D_model| over prior-valid pixels
  med_prior_vs_gt_mm        median |G - GT| over prior-valid pixels
  med_model_vs_gt_mm        median |D_model - GT| over the SAME prior-valid pixels
  prior_closer_pct          share of those pixels where |G - GT| < |D_model - GT|
  model_wrong_*             the same, restricted to pixels where |D_model - GT| > --wrong_mm
D_model is the final (fine-stage) depth.

Usage:
    python scripts/prior_quality.py --config configs/ablations/A0f_fixes_only_short.yaml \
        --ckpt runs/A0f_fixes_only_short/seed2/ckpt_final.pt --out runs/prior_quality/A0f_seed2.json
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
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets.dtu import DTUDataset  # noqa: E402
from evaluation.stage_depth import _collate_one  # noqa: E402
from models import build_model, regress_depth  # noqa: E402


def stats(prior_ok, e_prior, e_model, d_pm) -> dict:
    m = prior_ok
    return {
        "n_pixels": int(len(m)),
        "prior_valid_pct": 100 * float(m.mean()) if len(m) else float("nan"),
        "med_prior_vs_model_mm": float(np.median(d_pm[m])) if m.any() else float("nan"),
        "med_prior_vs_gt_mm": float(np.median(e_prior[m])) if m.any() else float("nan"),
        "med_model_vs_gt_mm": float(np.median(e_model[m])) if m.any() else float("nan"),
        "med_model_vs_gt_all_mm": float(np.median(e_model)) if len(m) else float("nan"),
        "prior_closer_pct": 100 * float((e_prior[m] < e_model[m]).mean()) if m.any() else float("nan"),
    }


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

    out = {"ckpt": args.ckpt, "wrong_mm": args.wrong_mm, "scans": {}}
    for scan in scans:
        t0 = time.time()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(scan + "\n")
        ds = DTUDataset(root=cfg["data"]["root"], split="val", cfg={**cfg["data"], "scan_list_file": f.name})
        os.unlink(f.name)
        cols = {k: [] for k in ("prior_ok", "e_prior", "e_model", "d_pm")}
        per_view = []
        for i in range(min(len(ds), args.max_views or len(ds))):
            batch = _collate_one(ds[i], args.device)
            o = model(**batch["model_inputs"], alpha=1.0, max_gate=1.0)
            pred = regress_depth(o["scores"]["fine"], o["depth_hypotheses"]["fine"])[0, 0].cpu().numpy()
            gt = batch["gt_depth"][0, 0].cpu().numpy()
            pd = batch["prior_depth"][0, 0].cpu().numpy()
            pv = batch["prior_valid"][0, 0].cpu().numpy() > 0.5
            g = gt > 0
            cols["prior_ok"].append(pv[g]); cols["e_prior"].append(np.abs(pd - gt)[g])
            cols["e_model"].append(np.abs(pred - gt)[g]); cols["d_pm"].append(np.abs(pd - pred)[g])
            per_view.append({"view": i, "prior_valid_pct_all": 100 * float(pv.mean()),
                             "prior_valid_pct_gt": 100 * float(pv[g].mean()) if g.any() else float("nan")})
        c = {k: np.concatenate(v) for k, v in cols.items()}
        wrong = c["e_model"] > args.wrong_mm
        s = stats(c["prior_ok"], c["e_prior"], c["e_model"], c["d_pm"])
        s["model_wrong"] = {"pct_of_gt": 100 * float(wrong.mean()),
                            **stats(c["prior_ok"][wrong], c["e_prior"][wrong], c["e_model"][wrong], c["d_pm"][wrong])}
        pva = np.array([v["prior_valid_pct_all"] for v in per_view])
        s["n_views"] = len(per_view)
        s["n_views_no_prior"] = int((pva == 0).sum())
        s["per_view"] = per_view
        out["scans"][scan] = s
        print(f"{scan:<8} prior valid {s['prior_valid_pct']:5.1f}%  |G-D| {s['med_prior_vs_model_mm']:6.2f}  "
              f"|G-GT| {s['med_prior_vs_gt_mm']:6.2f}  |D-GT| {s['med_model_vs_gt_mm']:5.2f}  "
              f"G closer {s['prior_closer_pct']:5.1f}%  no-prior views {s['n_views_no_prior']}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
