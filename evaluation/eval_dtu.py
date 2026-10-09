#!/usr/bin/env python
"""Evaluate a trained checkpoint on DTU and check it against the acceptance gates.

Usage:
    python -m evaluation.eval_dtu --config configs/stage0_baseline.yaml --ckpt runs/A0f_fixes_only_short/seed0/ckpt_final.pt
    python -m evaluation.eval_dtu --config configs/ablations/A3_geocorr_fixed.yaml \\
        --ckpt runs/A3_geocorr_fixed/seed0/ckpt_final.pt --baseline_stats baseline_stats.json

`--baseline_stats` should be the JSON produced by running this script on Stage 0
(A0f) across >=3 seeds (scripts/eval_stage0_a0f.sh) -- required to
evaluate the "Baseline parity" and "Efficiency" gates for any later stage.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch
import yaml

from data.datasets.dtu import DTUDataset
from evaluation.dtu_gt import filter_by_obs_mask, filter_by_plane, load_scan_ground_truth
from evaluation.efficiency import count_parameters, measure_runtime_and_memory, relative_overhead
from evaluation.fusion import (
    GEOMVSNET_DIST_THRESH, GEOMVSNET_NUM_CONSIST, GEOMVSNET_PROB_THRESH,
    backproject_depth_map, geometric_consistency_fusion, voxel_downsample,
)
from evaluation.metrics import chamfer_distances, dtu_official_metrics, f_score, overall_distance
from models import build_model, regress_depth
from training.provenance import log_git_commit

MAX_EVAL_DIST = 20.0  # mm; standard DTU-scale outlier clip before averaging accuracy/completeness
FUSION_STAGE = "fine"  # full-resolution cascade stage; matches training/train.py's eval_stage default
FUSION_VOXEL_SIZE = 1.0  # mm; matches ObsMask's voxel resolution (evaluation/dtu_gt.py)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _scan_number(scan_name: str) -> int:
    match = re.search(r"\d+", scan_name)
    if not match:
        raise ValueError(f"Could not parse a scan number out of '{scan_name}'")
    return int(match.group())


def _predict_scan_views(model, dataset: DTUDataset, scan: str, device: str):
    """Runs inference over every reference view of `scan` (per pair.txt, matching the
    reference protocol's view selection). Returns (depths (V, H, W), confidences
    (V, H, W), projs (V, 4, 4)) at the FUSION_STAGE resolution.

    Confidence is the max softmax probability over the FUSION_STAGE hypotheses --
    GeoMVSNet's `photometric_confidence` (prob_volume.max(1) at its last stage), which
    the reference fusion's prob_thresh is defined against.
    """
    depths, confs, projs = [], [], []
    for idx, (meta_scan, ref_view, _) in enumerate(dataset.metas):
        if meta_scan != scan:
            continue
        sample = dataset[idx]
        model_inputs = sample["model_inputs"]
        batched = {
            "ref_img": model_inputs["ref_img"].unsqueeze(0).to(device),
            "src_imgs": [s.unsqueeze(0).to(device) for s in model_inputs["src_imgs"]],
            "ref_geom": model_inputs["ref_geom"].unsqueeze(0).to(device),
            "ref_proj": model_inputs["ref_proj"].unsqueeze(0).to(device),
            "src_projs": [p.unsqueeze(0).to(device) for p in model_inputs["src_projs"]],
            "depth_hypotheses_per_stage": {
                k: v.unsqueeze(0).to(device) for k, v in model_inputs["depth_hypotheses_per_stage"].items()
            },
            "depth_interval": model_inputs["depth_interval"].view(1).to(device),
        }
        with torch.no_grad():
            outputs = model(**batched, alpha=1.0, max_gate=1.0)
            score = outputs["scores"][FUSION_STAGE]
            depth_hyp = outputs["depth_hypotheses"][FUSION_STAGE]  # per-pixel, narrowed
            depths.append(regress_depth(score, depth_hyp)[0, 0].cpu().numpy())  # (H, W)
            confs.append(torch.softmax(score, dim=1).max(dim=1).values[0].cpu().numpy())
        projs.append(model_inputs["ref_proj"].numpy())

    if not depths:
        raise ValueError(f"No test-split samples found for scan '{scan}' -- check pair.txt/split lists.")
    return np.stack(depths), np.stack(confs), np.stack(projs)


def _fuse_scan_point_cloud(model, dataset: DTUDataset, scan: str, device: str, fusion_cfg: dict | None = None) -> np.ndarray:
    """Predicts every view of `scan` and fuses them into one point cloud.

    fusion_cfg["method"]:
      "geometric" (default) -- the reference GeoMVSNet fusion (evaluation/fusion.py::
          geometric_consistency_fusion) with its published DTU thresholds unless
          overridden (prob_thresh / dist_thresh / num_consist).
      "none" -- legacy: back-project every pixel, no consistency filtering.
    """
    fusion_cfg = fusion_cfg or {}
    depths, confs, projs = _predict_scan_views(model, dataset, scan, device)
    if fusion_cfg.get("method", "geometric") == "none":
        fused = np.concatenate([backproject_depth_map(d, p) for d, p in zip(depths, projs)], axis=0)
        return voxel_downsample(fused, FUSION_VOXEL_SIZE)
    return geometric_consistency_fusion(
        depths, confs, projs,
        prob_thresh=fusion_cfg.get("prob_thresh", GEOMVSNET_PROB_THRESH),
        dist_thresh=fusion_cfg.get("dist_thresh", GEOMVSNET_DIST_THRESH),
        num_consist=fusion_cfg.get("num_consist", GEOMVSNET_NUM_CONSIST),
        device=device,
        reference_sampling_quirk=fusion_cfg.get("reference_sampling_quirk", False),
    )


def legacy_scan_metrics(pred_points: np.ndarray, gt: dict) -> dict:
    """Pre-audit metric: ObsMask+Plane-filtered prediction vs ObsMask-filtered GT, both
    directions clipped (not excluded) at MAX_EVAL_DIST. Kept only to compare against
    numbers reported before the protocol fix; use dtu_official_metrics."""
    pred_keep = filter_by_obs_mask(pred_points, gt["obs_mask"], gt["bb"], gt["res"])
    pred_keep &= filter_by_plane(pred_points, gt["plane"])
    gt_keep = filter_by_obs_mask(gt["stl_points"], gt["obs_mask"], gt["bb"], gt["res"])
    filtered_pred = pred_points[pred_keep]
    filtered_gt = gt["stl_points"][gt_keep]
    if filtered_pred.shape[0] == 0 or filtered_gt.shape[0] == 0:
        raise ValueError("No points survived GT/ObsMask filtering; check calibration.")
    dists = chamfer_distances(filtered_pred, filtered_gt, max_dist=MAX_EVAL_DIST)
    acc_dists = np.clip(dists["accuracy_dists"], 0, MAX_EVAL_DIST)
    comp_dists = np.clip(dists["completeness_dists"], 0, MAX_EVAL_DIST)
    m = overall_distance(acc_dists, comp_dists)
    m.update(f_score(acc_dists, comp_dists))
    return m


def official_scan_metrics(pred_points: np.ndarray, gt: dict) -> dict:
    return dtu_official_metrics(pred_points, gt["stl_points"], gt["obs_mask"], gt["bb"], gt["res"], gt["plane"])


def evaluate_checkpoint(cfg: dict, ckpt_path: str, device: str) -> dict:
    """Runs inference over the DTU test split, fuses each scan's per-view depth
    maps into a point cloud (evaluation/fusion.py), filters both the prediction and
    the official DTU reference point cloud to the observed region and foreground
    half-space (evaluation/dtu_gt.py, using the same ObsMask/Plane data as the
    official DTU evaluation protocol), and computes accuracy/completeness/overall/
    F-score (evaluation/metrics.py).
    """
    model = build_model(cfg["model"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    gt_root = cfg["data"].get("gt_root")
    if not gt_root:
        raise ValueError(
            "cfg['data']['gt_root'] is required for evaluation -- point it at the official "
            "DTU Points/ObsMask/Plane data (see evaluation/dtu_gt.py)."
        )

    eval_cfg = cfg.get("eval", {})
    # eval.test_data overrides cfg["data"] for the test set only, e.g. the full-res
    # dtu-test layout (layout: test, img_wh: [1600, 1152]) published numbers use.
    test_data = {**cfg["data"], **eval_cfg.get("test_data", {})}
    dataset = DTUDataset(root=test_data["root"], split="test", cfg=test_data)
    scans = sorted({scan for scan, _, _ in dataset.metas})

    per_scan = []
    with measure_runtime_and_memory(device) as timing:
        for scan in scans:
            pred_points = _fuse_scan_point_cloud(model, dataset, scan, device, eval_cfg.get("fusion"))
            gt = load_scan_ground_truth(gt_root, _scan_number(scan))
            metric_fn = legacy_scan_metrics if eval_cfg.get("metric", "official") == "legacy" else official_scan_metrics
            per_scan.append(metric_fn(pred_points, gt))

    result = {
        "accuracy": float(np.mean([m["accuracy"] for m in per_scan])),
        "completeness": float(np.mean([m["completeness"] for m in per_scan])),
        "overall": float(np.mean([m["overall"] for m in per_scan])),
        "f_score": float(np.mean([m["f_score"] for m in per_scan])),
        "runtime_sec": timing["elapsed_sec"],
        "peak_mem_mb": timing["peak_mem_mb"],
        "param_count": count_parameters(model),
        "n_scans": len(per_scan),
        "test_img_wh": list(dataset.img_wh),
        "test_layout": dataset.layout,
        "per_scan": {scan: {k: float(m[k]) for k in ("accuracy", "completeness", "overall")}
                     for scan, m in zip(scans, per_scan)},
    }
    return result


def check_gates(result: dict, baseline_stats: dict | None) -> dict:
    """Applies the acceptance gates from docs/evaluation_and_gates.md."""
    gates = {}
    if baseline_stats is not None:
        b_mean, b_std = baseline_stats["overall"]["mean"], baseline_stats["overall"]["std"]
        gates["baseline_parity"] = bool(
            (b_mean - b_std) <= result["overall"] <= (b_mean + b_std)
        ) or (
            # temporary single-run rule: <=1% relative degradation
            relative_overhead(b_mean, result["overall"]) <= 0.01
        )
        gates["efficiency"] = relative_overhead(
            baseline_stats["runtime_sec"]["mean"], result["runtime_sec"]
        ) < 0.10
    return gates


def main():
    log_git_commit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--baseline_stats", default=None, help="JSON from a Stage 0 (A0) multi-seed run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="Where to write the result JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = evaluate_checkpoint(cfg, args.ckpt, args.device)

    baseline_stats = None
    if args.baseline_stats:
        with open(args.baseline_stats) as f:
            baseline_stats = json.load(f)
    result["gates"] = check_gates(result, baseline_stats)

    out_path = args.out or os.path.splitext(args.ckpt)[0] + "_eval.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
