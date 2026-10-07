#!/usr/bin/env python
"""Diagnostic (not a paper number): does the val-frozen prior gate's harm-check gain
survive DTU fusion?

For each scan, runs the model through the official evaluation path
(evaluation/eval_dtu.py: eval.test_data, 1600x1152, fine-stage depth and max-softmax
confidence), then fuses and scores three depth-map sets with the official fusion
(geometric_consistency_fusion, GeoMVSNet thresholds) and dtu_official_metrics:
  model          the model's fine depth (= the official evaluation)
  gated_keepconf depth := G where the gate fires; confidence unchanged
  gated_conf1    depth := G where the gate fires; confidence := 1 there, so overridden
                 pixels always pass the prob_thresh filter
Gate = coarse_disagr_rev from gates_valfrozen.json: |G - coarse depth| >= thr (mm), G
valid, evaluated on the coarse grid exactly as prior_confidence.view_signals (G sampled
nearest each coarse cell centre) and broadcast nearest to the fine grid. The override
uses the full-res G at each fine pixel; fine pixels with G = 0 keep the model depth.

Runs on CPU, one process per scan; per-scan results go to <out_dir>/<scan>.json.

Usage:
    python scripts/gated_fusion_diag.py --config configs/ablations/A0f_fixes_only_short.yaml \
        --ckpt runs/A0f_fixes_only_short/seed2/ckpt_final.pt \
        --gates runs/prior_confidence_val/gates_valfrozen.json \
        --out_dir runs/prior_confidence_val/fusion_diag --scans scan11 scan13 scan48 scan62 scan77
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARGS = None
VARIANTS = ("model", "gated_keepconf", "gated_conf1")


def process_scan(scan):
    path = os.path.join(ARGS.out_dir, f"{scan}.json")
    if os.path.exists(path):
        return json.load(open(path))
    import torch
    import yaml
    from data.datasets.dtu import DTUDataset
    from evaluation.eval_dtu import _scan_number, official_scan_metrics
    from evaluation.dtu_gt import load_scan_ground_truth
    from evaluation.fusion import (GEOMVSNET_DIST_THRESH, GEOMVSNET_NUM_CONSIST, GEOMVSNET_PROB_THRESH,
                                   geometric_consistency_fusion)
    from evaluation.stage_depth import _collate_one
    from models import build_model, regress_depth

    torch.set_num_threads(ARGS.threads)
    cfg = yaml.safe_load(open(ARGS.config))
    thr = json.load(open(ARGS.gates))["gates"]["coarse_disagr_rev"]["threshold"]
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(ARGS.ckpt, map_location="cpu", weights_only=False)["model"])
    model.eval()
    eval_cfg = cfg.get("eval", {})
    test_data = {**cfg["data"], **eval_cfg.get("test_data", {})}
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(scan + "\n")
    ds = DTUDataset(root=test_data["root"], split="test", cfg={**test_data, "scan_list_file": f.name})
    os.unlink(f.name)

    t0 = time.time()
    depths, confs, gated, fired, projs = [], [], [], [], []
    for i in range(min(len(ds), ARGS.max_views or len(ds))):
        sample = ds[i]
        inp = _collate_one(sample, "cpu")["model_inputs"]
        with torch.no_grad():
            out = model(**inp, alpha=1.0, max_gate=1.0)
            score = out["scores"]["fine"]
            d = regress_depth(score, out["depth_hypotheses"]["fine"])[0, 0].numpy()
            c = torch.softmax(score, dim=1).max(dim=1).values[0].numpy()
            coarse = regress_depth(out["scores"]["coarse"], out["depth_hypotheses"]["coarse"])[0, 0].numpy()
        G = sample["prior_depth"][0].numpy()
        H, W = G.shape
        Hc, Wc = coarse.shape
        k = H // Hc
        yc, xc = np.mgrid[0:Hc, 0:Wc]
        G_c = G[yc * k + k // 2, xc * k + k // 2]
        fire_c = (G_c > 0) & (np.abs(G_c - coarse) >= thr)
        fire = np.repeat(np.repeat(fire_c, k, 0), k, 1)[:H, :W] & (G > 0)
        depths.append(d)
        confs.append(c)
        gated.append(np.where(fire, G, d))
        fired.append(fire)
        projs.append(sample["model_inputs"]["ref_proj"].numpy())
    depths, confs, gated, fired, projs = map(np.stack, (depths, confs, gated, fired, projs))
    t_inf = time.time() - t0

    gt = load_scan_ground_truth(cfg["data"]["gt_root"], _scan_number(scan))
    sets = {"model": (depths, confs), "gated_keepconf": (gated, confs),
            "gated_conf1": (gated, np.where(fired, 1.0, confs).astype(np.float32))}
    res = {"scan": scan, "threshold_mm": thr, "n_views": len(depths), "img_hw": list(depths.shape[1:]),
           "fired_pct_of_px": 100 * float(fired.mean()),
           "fired_pct_of_conf_pass_px": 100 * float(fired[confs > GEOMVSNET_PROB_THRESH].mean()),
           "fired_conf_pass_pct": 100 * float((confs[fired] > GEOMVSNET_PROB_THRESH).mean()) if fired.any() else 0.0,
           "inference_sec": t_inf}
    for name, (dm, cm) in sets.items():
        pts = geometric_consistency_fusion(dm, cm, projs, prob_thresh=GEOMVSNET_PROB_THRESH,
                                           dist_thresh=GEOMVSNET_DIST_THRESH, num_consist=GEOMVSNET_NUM_CONSIST,
                                           device="cpu")
        m = official_scan_metrics(pts, gt)
        res[name] = {k: float(m[k]) for k in ("accuracy", "completeness", "overall")}
        res[name]["n_points"] = int(len(pts))
    res["total_sec"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=2)
    return res


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gates", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scans", nargs="+", required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max_views", type=int, default=0, help="smoke test only")
    ARGS = p.parse_args()
    os.makedirs(ARGS.out_dir, exist_ok=True)
    with Pool(len(ARGS.scans)) as pool:
        results = pool.map(process_scan, ARGS.scans)
    rows = {r["scan"]: r for r in results}
    summary = {v: {k: float(np.mean([r[v][k] for r in results])) for k in ("accuracy", "completeness", "overall")}
               for v in VARIANTS}
    json.dump({"scans": rows, "mean": summary}, open(os.path.join(ARGS.out_dir, "summary.json"), "w"), indent=2)
    print(f"{'scan':<8} {'fired%':>7} " + " ".join(f"{v:>26}" for v in VARIANTS))
    for s, r in rows.items():
        print(f"{s:<8} {r['fired_pct_of_px']:7.2f} " + " ".join(
            f"{r[v]['accuracy']:8.3f}{r[v]['completeness']:9.3f}{r[v]['overall']:9.3f}" for v in VARIANTS))
    print("mean              " + " ".join(
        f"{summary[v]['accuracy']:8.3f}{summary[v]['completeness']:9.3f}{summary[v]['overall']:9.3f}" for v in VARIANTS))
    print("FUSION_DIAG_DONE")


if __name__ == "__main__":
    main()
