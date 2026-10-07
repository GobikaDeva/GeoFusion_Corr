#!/usr/bin/env python
"""Diagnostic: does B2's fused-completeness gain come from its confidence letting more
pixels through fusion, rather than from better depth?

For each scan, runs both models through the official evaluation path
(eval_dtu._predict_scan_views: eval.test_data, 1600x1152, fine depth, max-softmax
confidence) and fuses with the GeoMVSNet fusion (evaluation/fusion.py, same loop,
instrumented to also return per-pixel stage counts). Per model it records
  conf quantiles, % of pixels with conf > prob_thresh (stage 1),
  % of those that pass the >= num_consist geometric check (stage 2), fused point count,
and official Acc/Comp/Overall for three fusions:
  default    prob_thresh = 0.3 (the official evaluation; must reproduce eval_1600x1152.json)
  matched    prob_thresh set per scan so this model's conf-pass fraction equals the OTHER
             model's at 0.3 (quantile of this model's confidences)
  noconf     prob_thresh = 0 (geometric consistency only; depth quality alone)
If B2's gain is confidence-driven, B2@matched should lose most of it and A0f@matched
should gain most of it; under noconf the two should be close.

Per-scan results go to <out_dir>/<scan>.json (skipped if present, so reruns resume).

Usage:
    python scripts/fusion_pass_diag.py --out_dir runs/fusion_pass_diag --device cuda --scans scan48 ...
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
from evaluation.dtu_gt import load_scan_ground_truth  # noqa: E402
from evaluation.eval_dtu import _predict_scan_views, _scan_number, official_scan_metrics  # noqa: E402
from evaluation.fusion import (GEOMVSNET_DIST_THRESH, GEOMVSNET_NUM_CONSIST, GEOMVSNET_PROB_THRESH,  # noqa: E402
                               _points_from_depth, _warp_to_ref)
from models import build_model  # noqa: E402

MODELS = {
    "A0f": ("configs/ablations/A0f_fixes_only_short.yaml", "runs/A0f_fixes_only_short/seed2/ckpt_final.pt"),
    "B2": ("configs/ablations/B2_capacity.yaml", "runs/B2_capacity/seed2/ckpt_final.pt"),
}


def fuse_with_stats(depths, confs, projs, prob_thresh, device, batch_size=20):
    """evaluation.fusion.geometric_consistency_fusion, same arithmetic, plus stage counts."""
    with torch.no_grad():
        d = torch.from_numpy(depths * (confs > prob_thresh)).float().unsqueeze(1).to(device)
        P = torch.from_numpy(projs).float().to(device)
        V, _, H, W = d.shape
        out, n_conf, n_keep = [], 0, 0
        for i in range(V):
            ref_pc = _points_from_depth(d[i:i + 1], P[i:i + 1])
            pc_sum = torch.zeros(3, H, W, device=device)
            cnt = torch.zeros(1, H, W, device=device)
            for j in range(0, V, batch_size):
                src_pcs = _points_from_depth(d[j:j + batch_size], P[j:j + batch_size])
                aligned = _warp_to_ref(src_pcs, P[j:j + batch_size], P[i:i + 1], d[i:i + 1], align_corners=True)
                dist = torch.sqrt(((ref_pc - aligned) ** 2).sum(dim=1, keepdim=True))
                m = (dist < GEOMVSNET_DIST_THRESH).float()
                pc_sum += (aligned * m).sum(dim=0)
                cnt += m.sum(dim=0)
            keep = (cnt >= GEOMVSNET_NUM_CONSIST)[0]
            n_conf += int((d[i, 0] > 0).sum())
            n_keep += int(keep.sum())
            out.append((pc_sum / cnt).permute(1, 2, 0)[keep].cpu().numpy())
    n_px = V * H * W
    return np.concatenate(out, axis=0).astype(np.float32), {
        "prob_thresh": float(prob_thresh),
        "conf_pass_pct": 100 * n_conf / n_px,
        "geo_pass_pct_of_conf_pass": 100 * n_keep / max(n_conf, 1),
        "kept_pct_of_px": 100 * n_keep / n_px,
    }


def predict(name, scan, device, max_views):
    cfg_path, ckpt = MODELS[name]
    cfg = yaml.safe_load(open(cfg_path))
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    model.to(device).eval()
    test_data = {**cfg["data"], **cfg.get("eval", {}).get("test_data", {})}
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(scan + "\n")
    ds = DTUDataset(root=test_data["root"], split="test", cfg={**test_data, "scan_list_file": f.name})
    os.unlink(f.name)
    if max_views:
        ds.metas = ds.metas[:max_views]
    with torch.no_grad():
        depths, confs, projs = _predict_scan_views(model, ds, scan, device)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return depths, confs, projs, cfg["data"]["gt_root"]


def process_scan(scan, args):
    path = os.path.join(args.out_dir, f"{scan}.json")
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    preds = {n: predict(n, scan, args.device, args.max_views) for n in MODELS}
    gt = load_scan_ground_truth(preds["A0f"][3], _scan_number(scan))
    pass_frac = {n: float((c > GEOMVSNET_PROB_THRESH).mean()) for n, (_, c, _, _) in preds.items()}
    res = {"scan": scan, "n_views": int(preds["A0f"][0].shape[0])}
    for n, (depths, confs, projs, _) in preds.items():
        other = [o for o in MODELS if o != n][0]
        # threshold whose pass fraction equals the other model's at 0.3
        matched = float(np.quantile(confs, 1 - pass_frac[other]))
        r = {"conf_quantiles": {q: float(np.quantile(confs, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
             "conf_mean": float(confs.mean())}
        for variant, thr in (("default", GEOMVSNET_PROB_THRESH), ("matched", matched), ("noconf", 0.0)):
            pts, st = fuse_with_stats(depths, confs, projs, thr, args.device)
            m = official_scan_metrics(pts, gt)
            r[variant] = {**st, "n_points": int(len(pts)),
                          **{k: float(m[k]) for k in ("accuracy", "completeness", "overall")}}
        res[n] = r
    res["total_sec"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=2)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scans", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_views", type=int, default=0, help="smoke test only")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for s in args.scans:
        r = process_scan(s, args)
        rows.append(r)
        print(s, " ".join(f"{n}:{v}={r[n][v]['conf_pass_pct']:.1f}%/{r[n][v]['geo_pass_pct_of_conf_pass']:.1f}%"
                          f"/{r[n][v]['overall']:.3f}" for n in MODELS for v in ("default", "matched", "noconf")),
              flush=True)
    summary = {n: {v: {k: float(np.mean([r[n][v][k] for r in rows]))
                       for k in ("conf_pass_pct", "geo_pass_pct_of_conf_pass", "kept_pct_of_px", "n_points",
                                 "accuracy", "completeness", "overall")}
                   for v in ("default", "matched", "noconf")} for n in MODELS}
    json.dump({"scans": [r["scan"] for r in rows], "mean": summary},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))
    print("FUSION_PASS_DIAG_DONE")


if __name__ == "__main__":
    main()
