#!/usr/bin/env python
"""Coarse-stage cost-volume probe at pixels where the fine stage is wrong.

For each view of a scan, pixels at coarse resolution are labelled "wrong" when the
final fine-stage depth there is off by more than --wrong_mm. At every such pixel
(and, for contrast, at "right" pixels with error < 1mm) this reports:

  raw cost (pre-regularizer, variance aggregated over all 5 views -- what the model
  sees): is the GT bin (+-1 bin) the global minimum, only a secondary local minimum,
  or not a minimum at all?
  robust aggregation of the same features: per-source-view pair costs combined as
  mean-of-best-2 views (occlusion-robust); does that recover the GT minimum?
  regularized scores: probability mass within +-1 bin of GT, argmax at GT?
  training objective: at the fine resolution, is the photometric loss lower at GT
  depth than at the predicted depth? Ours (0.85 SSIM + 0.15 L1, mean over views)
  vs a min-over-views variant (occlusion-robust) and CL-MVSNet-like L0.5.

Also saves the wrong-pixel cost profiles for scripts/plot_costvol_profiles.py.
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

from data.datasets.dtu import DTUDataset  # noqa: E402
from evaluation.stage_depth import _collate_one  # noqa: E402
from models import build_model, regress_depth  # noqa: E402
from models.geofusionnet import differentiable_homography_warp  # noqa: E402
from models.losses import _ssim  # noqa: E402
from models.warp_utils import warp_image_by_depth  # noqa: E402


def local_min(cost, idx):
    """cost (N, D); idx (N,). True where cost[idx] <= both neighbours."""
    D = cost.shape[1]
    c = cost.gather(1, idx[:, None])[:, 0]
    left = cost.gather(1, (idx - 1).clamp(min=0)[:, None])[:, 0]
    right = cost.gather(1, (idx + 1).clamp(max=D - 1)[:, None])[:, 0]
    return (c <= left) & (c <= right)


def sawtooth_flips(prob, min_step=0.01):
    """prob (N, D). Number of consecutive probability steps that reverse sign with
    both steps > min_step -- the stride-2 transposed-conv checkerboard shows up as
    many alternations between neighbouring depth bins."""
    dp = prob[:, 1:] - prob[:, :-1]
    big = dp.abs() > min_step
    return ((dp[:, 1:] * dp[:, :-1] < 0) & big[:, 1:] & big[:, :-1]).sum(1)


def classify(cost, gt_bin):
    """cost (N, D), lower = better. Returns labels: 0 = GT (+-1 bin) is global min,
    1 = GT near a secondary local min (ambiguous), 2 = no minimum near GT."""
    argmin = cost.argmin(1)
    is_global = (argmin - gt_bin).abs() <= 1
    near = torch.stack([gt_bin + o for o in (-1, 0, 1)], 1).clamp(0, cost.shape[1] - 1)
    best_near = near.gather(1, cost.gather(1, near).argmin(1, keepdim=True))[:, 0]
    is_local = local_min(cost, best_near)
    lab = torch.full_like(gt_bin, 2)
    lab[is_local] = 1
    lab[is_global] = 0
    return lab


@torch.no_grad()
def photo_errors(inputs, depth):
    """Per-pixel, per-source-view photometric error maps (V, H, W) at full res for
    `depth`: ours = 0.85 SSIM + 0.15 L1 (models/losses.py), l05 = sqrt|diff| (CL-MVSNet
    smooth_l0_5 for |x| > ~1e-3). Invalid (out-of-view) pixels are inf."""
    ref = inputs["ref_img"]
    ours, l05 = [], []
    for s, p in zip(inputs["src_imgs"], inputs["src_projs"]):
        w, valid = warp_image_by_depth(s, p, inputs["ref_proj"], depth)
        l1 = (w - ref).abs().mean(1, keepdim=True)
        e = 0.85 * _ssim(w, ref).mean(1, keepdim=True) + 0.15 * l1
        inv = valid < 0.5
        ours.append(torch.where(inv, torch.full_like(e, float("inf")), e)[0, 0])
        e2 = (w - ref).abs().clamp(min=1e-3).sqrt().mean(1, keepdim=True)
        l05.append(torch.where(inv, torch.full_like(e2, float("inf")), e2)[0, 0])
    return torch.stack(ours), torch.stack(l05)


def agg(err, how):
    """err (V, N) with inf = invalid view. mean over valid / min over valid."""
    if how == "min":
        return err.min(0).values
    finite = torch.isfinite(err)
    return torch.where(finite, err, torch.zeros_like(err)).sum(0) / finite.sum(0).clamp(min=1)


@torch.no_grad()
def probe_view(model, sample, device, wrong_mm):
    batch = _collate_one(sample, device)
    inp = batch["model_inputs"]
    out = model(**inp, alpha=1.0, max_gate=1.0)
    fine_pred = regress_depth(out["scores"]["fine"], out["depth_hypotheses"]["fine"])  # (1,1,H,W)
    gt_full = batch["gt_depth"]

    bb = model.backbone
    stage = bb.cfg.stages[0]
    assert stage.name == "coarse"
    # the features the model's own coarse cost volume uses (incl. the GGF residual)
    ref_feats, src_feats = bb.extract_features(inp["ref_img"], inp["src_imgs"], inp["ref_geom"])
    ref_f = ref_feats["coarse"]
    src_fs = [sf["coarse"] for sf in src_feats]
    Hc, Wc = ref_f.shape[-2:]
    hyp = out["depth_hypotheses"]["coarse"]  # (1, D, Hc, Wc), same every pixel
    ref_proj = bb._scale_projection(inp["ref_proj"], stage.resolution_scale)
    src_projs = [bb._scale_projection(p, stage.resolution_scale) for p in inp["src_projs"]]

    # mean over cost channels: the single channel-mean variance channel, or for a
    # group-wise volume (cost_groups > 1) the mean of the equal-size groups, which is
    # the same channel-mean variance -- so the dip statistics stay comparable
    raw_mean = bb.build_cost_volume(ref_f, src_fs, ref_proj, src_projs, hyp).mean(1)  # (1, D, Hc, Wc)
    pair = []
    for sf, sp in zip(src_fs, src_projs):
        w = torch.cat([differentiable_homography_warp(sf, sp, ref_proj, hyp[:, d:d + 8])
                       for d in range(0, hyp.shape[1], 8)], 2)
        ones = torch.cat([differentiable_homography_warp(torch.ones_like(sf[:, :1]), sp, ref_proj, hyp[:, d:d + 8])
                          for d in range(0, hyp.shape[1], 8)], 2)
        c = ((ref_f.unsqueeze(2) - w) ** 2).mean(1) / 2  # pair variance, (1, D, Hc, Wc)
        pair.append(torch.where(ones[:, 0] > 0.99, c, torch.full_like(c, float("inf"))))
    pair = torch.stack(pair)  # (V, 1, D, Hc, Wc)
    best2 = pair.sort(0).values[:2]
    best2 = torch.where(torch.isfinite(best2), best2, torch.nan).nanmean(0)  # (1, D, Hc, Wc)
    prob = torch.softmax(out["scores"]["coarse"], 1)

    gt_c = F.interpolate(gt_full, size=(Hc, Wc), mode="nearest")[0, 0]
    err_c = F.interpolate((fine_pred - gt_full).abs(), size=(Hc, Wc), mode="nearest")[0, 0]
    hyps = hyp[0, :, 0, 0]
    step = (hyps[1] - hyps[0]).item()
    in_range = (gt_c > hyps[0] - step / 2) & (gt_c < hyps[-1] + step / 2)
    valid = (gt_c > 0) & in_range

    coarse_pred = regress_depth(out["scores"]["coarse"], hyp)[0, 0]
    res = {"n_valid": int(valid.sum()), "coarse_pred_map": coarse_pred.cpu().numpy()}
    for name, mask in (("wrong", valid & (err_c > wrong_mm)), ("right", valid & (err_c < 1.0))):
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            continue
        gt_bin = ((gt_c[ys, xs] - hyps[0]) / step).round().long().clamp(0, len(hyps) - 1)
        rm = raw_mean[0][:, ys, xs].T  # (N, D)
        b2 = best2[0][:, ys, xs].T
        b2 = torch.where(torch.isnan(b2), torch.full_like(b2, float("inf")), b2)
        pr = prob[0][:, ys, xs].T
        near = torch.stack([gt_bin + o for o in (-1, 0, 1)], 1).clamp(0, len(hyps) - 1)
        res[name] = {
            "raw_mean_class": classify(rm, gt_bin).cpu().numpy(),
            "best2_class": classify(b2, gt_bin).cpu().numpy(),
            "reg_mass_near_gt": pr.gather(1, near).sum(1).cpu().numpy(),
            "reg_argmax_ok": ((pr.argmax(1) - gt_bin).abs() <= 1).cpu().numpy(),
            "sawtooth": (sawtooth_flips(pr) >= 6).cpu().numpy(),
        }
        if name == "wrong":
            res["examples"] = {
                "ys": ys.cpu().numpy(), "xs": xs.cpu().numpy(), "hyps": hyps.cpu().numpy(),
                "gt": gt_c[ys, xs].cpu().numpy(), "fine_pred": F.interpolate(fine_pred, size=(Hc, Wc), mode="nearest")[0, 0][ys, xs].cpu().numpy(),
                "raw_mean": rm.cpu().numpy(), "best2": b2.cpu().numpy(), "prob": pr.cpu().numpy(),
                "pair": pair[:, 0, :, ys, xs].permute(2, 0, 1).cpu().numpy(),  # (N, V, D)
            }

    # Training-objective check at full res: loss(GT depth) < loss(pred depth)?
    gt_depth = torch.where(gt_full > 0, gt_full, fine_pred)
    e_gt, l_gt = photo_errors(inp, gt_depth)
    e_pr, l_pr = photo_errors(inp, fine_pred)
    err_full = (fine_pred - gt_full).abs()[0, 0]
    for name, m in (("wrong", (gt_full[0, 0] > 0) & (err_full > wrong_mm)), ("right", (gt_full[0, 0] > 0) & (err_full < 1.0))):
        if name not in res or m.sum() == 0:
            continue
        idx = torch.nonzero(m.flatten(), as_tuple=True)[0]
        V = e_gt.shape[0]
        f = lambda t: t.reshape(V, -1)[:, idx]  # noqa: E731
        both = torch.isfinite(agg(f(e_gt), "min")) & torch.isfinite(agg(f(e_pr), "min"))
        res[name]["loss_prefers_gt"] = {
            "ours_mean": (agg(f(e_gt), "mean") < agg(f(e_pr), "mean"))[both].cpu().numpy(),
            "ours_minview": (agg(f(e_gt), "min") < agg(f(e_pr), "min"))[both].cpu().numpy(),
            "l05_minview": (agg(f(l_gt), "min") < agg(f(l_pr), "min"))[both].cpu().numpy(),
        }
    return res


def summarize(results):
    n_valid = sum(r["n_valid"] for r in results)
    out = {"n_valid_pixels": n_valid}
    for name in ("wrong", "right"):
        rs = [r[name] for r in results if name in r]
        cat = lambda k: np.concatenate([r[k] for r in rs])  # noqa: E731
        s = {"n_pixels": int(len(cat("raw_mean_class")))}
        s["pct_of_valid"] = 100 * s["n_pixels"] / max(n_valid, 1)
        for k in ("raw_mean_class", "best2_class"):
            c = cat(k)
            s[k] = {"gt_global_min_pct": 100 * float((c == 0).mean()),
                    "gt_secondary_min_pct": 100 * float((c == 1).mean()),
                    "no_min_at_gt_pct": 100 * float((c == 2).mean())}
        # the same classes as a share of ALL valid in-range GT pixels, which (unlike
        # the shares above) is comparable across models with different wrong sets
        c = cat("raw_mean_class")
        s["raw_mean_class_pct_of_valid"] = {
            "gt_secondary_min": 100 * float((c == 1).sum()) / max(n_valid, 1),
            "no_min_at_gt": 100 * float((c == 2).sum()) / max(n_valid, 1),
        }
        s["reg_argmax_at_gt_pct"] = 100 * float(cat("reg_argmax_ok").mean())
        s["reg_mean_mass_near_gt"] = float(cat("reg_mass_near_gt").mean())
        s["sawtooth_pct"] = 100 * float(cat("sawtooth").mean())
        lp = [r["loss_prefers_gt"] for r in rs if "loss_prefers_gt" in r]
        s["loss_prefers_gt_pct"] = {k: 100 * float(np.concatenate([x[k] for x in lp]).mean()) for k in lp[0]}
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scans", nargs="+", default=["scan48", "scan1"])
    ap.add_argument("--wrong_mm", type=float, default=4.0)
    ap.add_argument("--plot_scan", default="scan48")
    ap.add_argument("--no_examples", dest="save_examples", action="store_false")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False)["model"])
    model.to(args.device).eval()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for scan in args.scans:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(scan + "\n")
        ds = DTUDataset(root=cfg["data"]["root"], split="val", cfg={**cfg["data"], "scan_list_file": f.name})
        os.unlink(f.name)
        results = []
        for i in range(len(ds)):
            r = probe_view(model, ds[i], args.device, args.wrong_mm)
            r["view"] = ds.metas[i][1]
            results.append(r)
        summary[scan] = summarize(results)
        if scan == args.plot_scan and args.save_examples:
            # Plotted by scripts/plot_costvol_profiles.py (needs matplotlib, which the
            # training env lacks).
            import pickle
            with open(os.path.join(args.out_dir, f"{scan}_examples.pkl"), "wb") as f:
                pickle.dump([{"view": r["view"], "examples": r["examples"]} for r in results if "examples" in r], f)
        print(scan, json.dumps(summary[scan], indent=1), flush=True)
    with open(os.path.join(args.out_dir, "costvol_profile_probe.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "wrong_mm": args.wrong_mm, "scans": summary}, f, indent=2)


if __name__ == "__main__":
    main()
