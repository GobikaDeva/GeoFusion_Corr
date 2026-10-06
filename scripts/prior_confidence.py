#!/usr/bin/env python
"""Can a GT-free confidence signal pick out the pixels where the prior G chooses the
right cost dip? (Test 1 of the confidence-gated prior question.)

Pre-registered rule: a gated prior is worth building only if some gate covers >= 20%
of ambiguous pixels with the prior picking the correct dip >= 80% of the time.

Pixel sets, dips and "correct dip" are scripts/prior_vs_dips.py's (coarse grid,
640x512 val images, lighting 3, all 22 test scans):
  ambiguous    wrong (> --wrong_mm) and the GT bin is only a secondary local min of
               the raw coarse cost
  correct dip  |G - true dip| < |G - chosen dip| (prior_nearer_true)
  sep > err    |chosen dip - true dip| > |G - GT|
Coverage is a share of ALL ambiguous pixels, prior-invalid ones included.

Confidence signals, all GT-free, evaluated on the coarse grid (lower = more
confident unless noted):
  a anchor_dist   distance (640x512 px) to the nearest RANSAC-inlier SIFT anchor
  b anchor_dens   inlier anchors in the 64x64 px window around the pixel (higher = more)
  c fit_resid     the view's RANSAC median |residual| (mm), one value per view
  d mono_grad     max |grad log G| (Sobel, 640x512) over the pixel's 4x4 block
  e coarse_disagr |G - model coarse-stage depth| (mm)
  f mv_incons     median over the 4 source views of |z_j - G_j(p_j)| / z_j: G
                  reprojected into source view j vs that view's own prior
Plus "combined": a logistic regression on the rank-transformed signals predicting
"correct dip", scored with 2-fold cross-validation split by scan.

Anchors come from `scripts/anchor_mono_prior.py --anchors_only`.

Usage:
    python scripts/prior_confidence.py --config configs/ablations/A0f_fixes_only_short.yaml \
        --ckpt runs/A0f_fixes_only_short/seed2/ckpt_final.pt --out_dir runs/prior_confidence
    (--analyze_only re-runs the analysis on the saved pixels.npz)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIGNALS = ["anchor_dist", "anchor_dens", "fit_resid", "mono_grad", "coarse_disagr", "mv_incons"]
HIGHER_IS_CONFIDENT = {"anchor_dens"}
KEEP_FRACS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
ALL_PIXEL_SUBSAMPLE = 0.25
RULE_COVERAGE, RULE_CORRECT = 20.0, 80.0


# ----------------------------------------------------------------------------- signals
def _prior_full(prior_root, scan, view, wh):
    """The dataset's 640x512 prior (data/datasets/dtu.py::_load_prior), depth only."""
    p = np.load(os.path.join(prior_root, f"{scan}_train", f"{view:04d}_metric.npy")).astype(np.float32)
    valid = cv2.resize((p > 0).astype(np.uint8), wh, interpolation=cv2.INTER_NEAREST) > 0
    p = cv2.resize(p, wh, interpolation=cv2.INTER_LINEAR)
    return np.where(valid & (p > 0), p, 0.0).astype(np.float32)


def view_signals(sample, prior_root, fits, coarse_pred, Hc, Wc):
    """(len(SIGNALS), Hc, Wc) float32 maps; nan = no evidence (least confident)."""
    scan, ref = sample["scan"], sample["ref_view"]
    G = sample["prior_depth"][0].numpy()
    H, W = G.shape
    k = H // Hc
    yc, xc = np.mgrid[0:Hc, 0:Wc]
    yf, xf = yc * k + k // 2, xc * k + k // 2  # full-res pixel nearest each coarse centre
    G_c = G[yf, xf]
    out = np.full((len(SIGNALS), Hc, Wc), np.nan, np.float32)

    a = np.load(os.path.join(prior_root, f"{scan}_train", f"{ref:04d}_anchors.npz"))
    xy = np.round(a["xy"]).astype(int)
    xy[:, 0] = xy[:, 0].clip(0, W - 1)
    xy[:, 1] = xy[:, 1].clip(0, H - 1)
    if len(xy):
        mask = np.ones((H, W), np.uint8)
        mask[xy[:, 1], xy[:, 0]] = 0
        out[0] = cv2.distanceTransform(mask, cv2.DIST_L2, 5)[yf, xf]
        cnt = np.zeros((H, W), np.float32)
        np.add.at(cnt, (xy[:, 1], xy[:, 0]), 1)
        out[1] = cv2.boxFilter(cnt, -1, (64, 64), normalize=False, borderType=cv2.BORDER_CONSTANT)[yf, xf]
    f = fits[str(ref)]
    if f["valid"]:
        out[2] = f["median_abs_res_mm"]

    logG = np.log(np.where(G > 0, G, 1.0))
    grad = np.hypot(cv2.Sobel(logG, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(logG, cv2.CV_32F, 0, 1, ksize=3)) / 8
    grad[cv2.dilate((G <= 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = np.nan  # invalid neighbour
    out[3] = grad.reshape(Hc, k, Wc, k).max(axis=(1, 3))  # nan propagates: any invalid in block -> nan
    out[4] = np.abs(G_c - coarse_pred)

    # multi-view consistency of G
    P_ref = sample["model_inputs"]["ref_proj"].numpy().astype(np.float64)
    pts = np.stack([xf * G_c, yf * G_c, G_c, np.ones_like(G_c)]).reshape(4, -1).astype(np.float64)
    X = np.linalg.inv(P_ref) @ pts
    rels = []
    for v, P in zip(sample["src_views"], sample["model_inputs"]["src_projs"]):
        q = P.numpy().astype(np.float64) @ X
        z = q[2]
        u, w = q[0] / np.maximum(z, 1e-6), q[1] / np.maximum(z, 1e-6)
        ui, wi = np.round(u).astype(int), np.round(w).astype(int)
        ok = (z > 0) & (ui >= 0) & (ui < W) & (wi >= 0) & (wi < H)
        Gj = np.zeros_like(z)
        Gj[ok] = _prior_full(prior_root, scan, v, (W, H))[wi[ok], ui[ok]]
        ok &= Gj > 0
        rels.append(np.where(ok, np.abs(z - Gj) / np.maximum(z, 1e-6), np.nan))
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        out[5] = np.nanmedian(np.stack(rels), 0).reshape(Hc, Wc)
    out[:, G_c <= 0] = np.nan
    return out


# ----------------------------------------------------------------------------- collect
def collect(args):
    import torch
    import torch.nn.functional as F
    import yaml
    from costvol_profile_probe import probe_view
    from prior_vs_dips import dips
    from data.datasets.dtu import DTUDataset
    from models import build_model

    cfg = yaml.safe_load(open(args.config))
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False)["model"])
    model.to(args.device).eval()
    prior_root = cfg["data"].get("prior_root", os.path.join(cfg["data"]["root"], "MonoPrior"))
    scans = [s.strip() for s in open(args.scan_list) if s.strip()]
    rng = np.random.default_rng(0)
    A = {k: [] for k in ("sig", "err", "scan")}  # all valid prior-valid pixels (subsampled)
    B = {k: [] for k in ("sig", "pv", "err", "g", "gt", "true", "chosen", "scan")}  # ambiguous
    for si, scan in enumerate(scans):
        t0 = time.time()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(scan + "\n")
        ds = DTUDataset(root=cfg["data"]["root"], split="val", cfg={**cfg["data"], "scan_list_file": f.name})
        os.unlink(f.name)
        fits = json.load(open(os.path.join(prior_root, f"{scan}_train", "fits.json")))
        n_amb = 0
        for i in range(min(len(ds), args.max_views or len(ds))):
            sample = ds[i]
            with torch.no_grad():
                res = probe_view(model, sample, args.device, args.wrong_mm)
            Hc, Wc = res["coarse_pred_map"].shape
            sig = view_signals(sample, prior_root, fits, res["coarse_pred_map"], Hc, Wc)
            down = lambda x: F.interpolate(x[None], size=(Hc, Wc), mode="nearest")[0, 0].numpy()  # noqa: E731
            gt_c, g_c, pv_c = down(sample["gt_depth"]), down(sample["prior_depth"]), down(sample["prior_valid"]) > 0.5
            hyps = sample["model_inputs"]["depth_hypotheses_per_stage"]["coarse"].numpy()
            step = hyps[1] - hyps[0]
            valid = (gt_c > 0) & (gt_c > hyps[0] - step / 2) & (gt_c < hyps[-1] + step / 2) & pv_c
            ys, xs = np.nonzero(valid & (rng.random(valid.shape) < ALL_PIXEL_SUBSAMPLE))
            A["sig"].append(sig[:, ys, xs].T); A["err"].append(np.abs(g_c - gt_c)[ys, xs])
            A["scan"].append(np.full(len(ys), si, np.int16))
            ex = res.get("examples")
            if ex is None:
                continue
            amb = res["wrong"]["raw_mean_class"] == 1
            ys, xs, gt, pred = ex["ys"][amb], ex["xs"][amb], ex["gt"][amb], ex["fine_pred"][amb]
            d_true, d_chosen = dips(ex["raw_mean"][amb], ex["hyps"], gt, pred)
            B["sig"].append(sig[:, ys, xs].T); B["pv"].append(pv_c[ys, xs]); B["g"].append(g_c[ys, xs])
            B["err"].append(np.abs(g_c[ys, xs] - gt)); B["gt"].append(gt)
            B["true"].append(d_true); B["chosen"].append(d_chosen); B["scan"].append(np.full(len(ys), si, np.int16))
            n_amb += len(ys)
        print(f"{scan:<8} ambiguous {n_amb:>6} ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(args.out_dir, "pixels.npz"), scans=np.array(scans),
                        **{f"all_{k}": np.concatenate(v) for k, v in A.items()},
                        **{f"amb_{k}": np.concatenate(v) for k, v in B.items()})


# ----------------------------------------------------------------------------- analyze
def confidence(sig, name):
    """Higher = more confident; no evidence (nan) -> -inf."""
    c = sig if name in HIGHER_IS_CONFIDENT else -sig
    return np.where(np.isnan(c), -np.inf, c)


def rank_order(conf):
    """Indices sorted most- to least-confident; ties broken by a seeded random key, not
    by pixel order (which groups scans, and tied signals such as fit_resid are common)."""
    tie = np.random.default_rng(0).random(len(conf))
    return np.lexsort((tie, -conf))


def sweep(conf, correct, sep_gt_err, n_total, fracs):
    """Keep the top `frac` (of finite-confidence pixels) by confidence."""
    fin = np.isfinite(conf)
    rows = []
    order = rank_order(conf[fin])
    cor, sge = correct[fin][order], sep_gt_err[fin][order]
    for frac in fracs:
        n = int(round(frac * fin.sum()))
        if n == 0:
            continue
        rows.append({"keep_frac_of_scored": frac, "n": n,
                     "coverage_pct": 100 * n / n_total,
                     "correct_dip_pct": 100 * float(cor[:n].mean()),
                     "sep_gt_err_pct": 100 * float(sge[:n].mean())})
    return rows


def operating_points(conf, correct, sep_gt_err, n_total):
    fine = sweep(conf, correct, sep_gt_err, n_total, np.linspace(0.01, 1.0, 100))
    at20 = [r for r in fine if r["coverage_pct"] >= RULE_COVERAGE]
    best20 = max(at20, key=lambda r: r["correct_dip_pct"]) if at20 else None
    ok80 = [r for r in fine if r["correct_dip_pct"] >= RULE_CORRECT]
    max_cov80 = max(ok80, key=lambda r: r["coverage_pct"]) if ok80 else None
    peak = max((r for r in fine if r["coverage_pct"] >= 2.0), key=lambda r: r["correct_dip_pct"], default=None)
    return {"best_correct_at_cov_ge_20": best20, "max_coverage_at_correct_ge_80": max_cov80,
            "peak_correct_at_cov_ge_2": peak,
            "meets_rule": bool(best20 and best20["correct_dip_pct"] >= RULE_CORRECT)}


def rank01(x):
    """Rank-transform to [0, 1], nan -> 0 (with a separate missing indicator)."""
    out = np.zeros(len(x))
    fin = np.isfinite(x)
    out[fin] = (np.argsort(np.argsort(x[fin])) + 0.5) / max(fin.sum(), 1)
    return out


def logistic_fit(X, y, l2=1e-3, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):  # Newton / IRLS
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(Hm, g)
    return w


def combined_conf(sig, scan_id, correct, scored):
    """2-fold (by scan) CV logistic-regression confidence; -inf outside `scored`."""
    feats = [rank01(confidence(sig[:, j], s)) for j, s in enumerate(SIGNALS)]
    feats += [np.isnan(sig[:, j]).astype(float) for j in range(len(SIGNALS))]
    X = np.column_stack(feats + [np.ones(len(sig))])
    conf = np.full(len(sig), -np.inf)
    folds = scan_id % 2
    for k in (0, 1):
        tr, te = scored & (folds != k), scored & (folds == k)
        w = logistic_fit(X[tr], correct[tr].astype(float))
        conf[te] = X[te] @ w
    return conf


def analyze(args):
    d = np.load(os.path.join(args.out_dir, "pixels.npz"))
    scans = list(d["scans"])
    a_sig, a_err = d["all_sig"], d["all_err"]
    sig, pv, err = d["amb_sig"], d["amb_pv"].astype(bool), d["amb_err"]
    g, true, chosen, scan_id = d["amb_g"], d["amb_true"], d["amb_chosen"], d["amb_scan"]
    n_amb = len(pv)
    correct = np.abs(g - true) < np.abs(g - chosen)
    sep_gt_err = np.abs(chosen - true) > err
    sig_pv = np.where(pv[:, None], sig, np.nan)  # prior-invalid pixels can't be gated in
    out = {"n_ambiguous": int(n_amb), "ambiguous_prior_valid_pct": 100 * float(pv.mean()),
           "baseline_correct_dip_pct_prior_valid": 100 * float(correct[pv].mean()),
           "rule": f"coverage >= {RULE_COVERAGE}% of all ambiguous and correct dip >= {RULE_CORRECT}%",
           "signals": {}}
    print(f"ambiguous pixels {n_amb}, prior-valid {out['ambiguous_prior_valid_pct']:.1f}%, "
          f"correct dip on prior-valid {out['baseline_correct_dip_pct_prior_valid']:.1f}%\n")

    for j, name in enumerate(SIGNALS + ["combined"]):
        if name == "combined":
            conf_amb = combined_conf(sig_pv, scan_id, correct, pv)
            deciles = None
        else:
            conf_all = confidence(a_sig[:, j], name)
            fin = np.isfinite(conf_all)
            c, e = conf_all[fin], a_err[fin]
            bins = np.empty(len(c), int)
            bins[rank_order(c)] = np.arange(len(c)) * 10 // len(c)  # 0 = most confident
            sgn = 1 if name in HIGHER_IS_CONFIDENT else -1
            deciles = [{"decile": b + 1, "median_prior_err_mm": float(np.median(e[bins == b])),
                        "pct_err_lt_4mm": 100 * float((e[bins == b] < 4).mean()),
                        "signal_range": [float(sgn * c[bins == b].min()), float(sgn * c[bins == b].max())]}
                       for b in range(10)]
            deciles.append({"no_evidence_pct_of_prior_valid": 100 * float((~fin).mean()),
                            "median_prior_err_mm_no_evidence": float(np.median(a_err[~fin])) if (~fin).any() else None})
            conf_amb = confidence(sig_pv[:, j], name)
        rows = sweep(conf_amb, correct, sep_gt_err, n_amb, KEEP_FRACS)
        ops = operating_points(conf_amb, correct, sep_gt_err, n_amb)
        # per-scan spread at the >=20%-coverage operating point (same pooled threshold)
        spread = None
        if ops["best_correct_at_cov_ge_20"]:
            fin = np.isfinite(conf_amb)
            keep = np.zeros(len(conf_amb), bool)
            keep[np.flatnonzero(fin)[rank_order(conf_amb[fin])[:ops["best_correct_at_cov_ge_20"]["n"]]]] = True
            per = [100 * correct[keep & (scan_id == s)].mean() for s in range(len(scans)) if (keep & (scan_id == s)).sum() >= 50]
            spread = {"min": float(np.min(per)), "median": float(np.median(per)), "max": float(np.max(per)), "n_scans": len(per)}
        out["signals"][name] = {"prior_err_by_confidence_decile": deciles, "ambiguous_sweep": rows,
                                "operating_points": ops, "per_scan_correct_at_op20": spread}
        print(f"== {name}")
        if deciles:
            print("  prior error by confidence decile (1 = most confident), all valid prior-valid px:")
            print("   " + " ".join(f"{r['median_prior_err_mm']:6.1f}" for r in deciles[:10]) + "  mm median")
            print("   " + " ".join(f"{r['pct_err_lt_4mm']:6.1f}" for r in deciles[:10]) + "  % <4mm")
        print(f"  {'keep':>6} {'cover%':>7} {'correct%':>9} {'sep>err%':>9}")
        for r in rows:
            print(f"  {r['keep_frac_of_scored']:6.2f} {r['coverage_pct']:7.1f} {r['correct_dip_pct']:9.1f} {r['sep_gt_err_pct']:9.1f}")
        b = ops["best_correct_at_cov_ge_20"]
        m = ops["max_coverage_at_correct_ge_80"]
        print(f"  best correct at cover>=20%: {b['correct_dip_pct']:.1f}% (cover {b['coverage_pct']:.1f}%)" if b else "  cover>=20% unreachable")
        print(f"  max cover at correct>=80%: {m['coverage_pct']:.1f}%" if m else "  correct>=80% never reached")
        print(f"  meets rule: {ops['meets_rule']}  per-scan correct at that op: {spread}\n", flush=True)
    out["any_meets_rule"] = any(v["operating_points"]["meets_rule"] for v in out["signals"].values())
    print("ANY GATE MEETS RULE:", out["any_meets_rule"])
    with open(os.path.join(args.out_dir, "prior_confidence.json"), "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--scan_list", default="data/datasets/splits/dtu_test.txt")
    ap.add_argument("--wrong_mm", type=float, default=4.0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_views", type=int, default=None, help="smoke tests only")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.analyze_only:
        collect(args)
    analyze(args)


if __name__ == "__main__":
    main()
