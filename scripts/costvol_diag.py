#!/usr/bin/env python
"""Per-stage cost-volume diagnostic: for each scan, median depth error at the
coarse / mid / fine cascade stages plus GT coverage (% of valid GT pixels whose true
depth lies inside that stage's depth search window). See evaluation/stage_depth.py.

Runs on the train-layout 640x512 images (lighting 3) for every reference view in
pair.txt, against Depths/scan{N}_train GT.

Usage:
    python scripts/costvol_diag.py --config configs/ablations/A0_baseline.yaml \
        --ckpt runs/A0_baseline/seed2/ckpt_final.pt --scans scan1 scan48 [--out diag.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets.dtu import DTUDataset  # noqa: E402
from evaluation.stage_depth import sample_stage_stats, summarize  # noqa: E402
from models import build_model  # noqa: E402
from training.provenance import log_git_commit  # noqa: E402


def main():
    log_git_commit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--scans", nargs="+", default=["scan1", "scan48"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="JSON output path (default: next to ckpt)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device)["model"])
    model.to(args.device).eval()

    results = {}
    for scan in args.scans:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(scan + "\n")
        data_cfg = {**cfg["data"], "scan_list_file": f.name}
        dataset = DTUDataset(root=data_cfg["root"], split="val", cfg=data_cfg)
        os.unlink(f.name)
        per_view = [sample_stage_stats(model, dataset[i], args.device) for i in range(len(dataset))]
        results[scan] = summarize(per_view)

    print(f"ckpt: {args.ckpt}")
    print(f"{'scan':<8}{'stage':<8}{'median err mm':>15}{'view-median mm':>16}{'>4mm %':>9}{'GT coverage %':>15}")
    for scan, stages in results.items():
        for stage, s in stages.items():
            print(f"{scan:<8}{stage:<8}{s['median_err_mm']:>15.2f}{s['median_view_err_mm']:>16.2f}"
                  f"{s['pct_gt_4mm']:>9.1f}{s['coverage_pct']:>15.1f}")

    out = args.out or os.path.splitext(args.ckpt)[0] + "_costvol_diag.json"
    with open(out, "w") as f:
        json.dump({"ckpt": args.ckpt, "scans": results}, f, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
