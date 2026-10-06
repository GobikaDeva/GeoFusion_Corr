#!/usr/bin/env python
"""Deployment check for the confidence gates of scripts/prior_confidence.py.

prior_confidence.py scores gates on ambiguous pixels only, which are selected with GT.
At inference a gate fires wherever its signal passes the threshold, including pixels the
model already gets right. This applies the gates that met the pre-registered rule, at
their >=20%-coverage operating point, to ALL valid prior-valid coarse pixels of the 22
test scans and counts:
  help  fired ambiguous pixels where the prior picks the correct dip (from pixels.npz)
  harm  fired pixels where the model is right (fine error <= --wrong_mm) and the prior
        is not (|G - GT| > --wrong_mm), i.e. following the prior would move off GT
Gates:
  coarse_disagr_rev  |G - coarse depth| above the threshold (disagreement = trust prior)
  combined           the logistic gate refit on all ambiguous pixels; ranks are taken
                     against the ambiguous-set distribution so they transfer
Runs on CPU (the GPU is busy with training), one process per scan.

Usage:
    python scripts/prior_gate_harm.py --config configs/ablations/A0f_fixes_only_short.yaml \
        --ckpt runs/A0f_fixes_only_short/seed2/ckpt_final.pt --pixels runs/prior_confidence/pixels.npz \
        --out runs/prior_confidence/gate_harm.json
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prior_confidence as pc  # noqa: E402

ARGS = None
GATES = None  # {name: (kind, params)}, set in main and inherited by fork


def fit_gates(pixels):
    d = np.load(pixels)
    sig, pv, err = d["amb_sig"], d["amb_pv"].astype(bool), d["amb_err"]
    g, true, chosen = d["amb_g"], d["amb_true"], d["amb_chosen"]
    n = len(pv)
    correct = np.abs(g - true) < np.abs(g - chosen)
    sge = np.abs(chosen - true) > err
    sp = np.where(pv[:, None], sig, np.nan)
    gates, info = {}, {}

    e = sp[:, 4]
    conf = np.where(np.isnan(e), -np.inf, e)
    op = pc.operating_points(conf, correct, sge, n)["best_correct_at_cov_ge_20"]
    fin = np.isfinite(conf)
    thr = float(np.sort(conf[fin])[::-1][op["n"] - 1])
    gates["coarse_disagr_rev"] = ("threshold_e", thr)
    info["coarse_disagr_rev"] = {"threshold_mm": thr, "ambiguous_op": op,
                                 "fired_ambiguous": int((conf >= thr).sum()),
                                 "help": int(((conf >= thr) & correct).sum())}

    # combined: features = rank against the ambiguous prior-valid distribution + missing flags
    ref = [np.sort(pc.confidence(sp[pv, j], s)[np.isfinite(pc.confidence(sp[pv, j], s))]) for j, s in enumerate(pc.SIGNALS)]

    def feats(s):
        cols = []
        for j, name in enumerate(pc.SIGNALS):
            c = pc.confidence(s[:, j], name)
            r = np.zeros(len(s))
            f = np.isfinite(c)
            r[f] = np.searchsorted(ref[j], c[f]) / max(len(ref[j]), 1)
            cols.append(r)
        cols += [np.isnan(s[:, j]).astype(float) for j in range(len(pc.SIGNALS))]
        return np.column_stack(cols + [np.ones(len(s))])

    X = feats(sp)
    w = pc.logistic_fit(X[pv], correct[pv].astype(float))
    c = np.where(pv, X @ w, -np.inf)
    op = pc.operating_points(c, correct, sge, n)["best_correct_at_cov_ge_20"]
    thr = float(np.sort(c[np.isfinite(c)])[::-1][op["n"] - 1])
    gates["combined"] = ("logistic", (w, thr, ref))
    info["combined"] = {"threshold_logit": thr, "ambiguous_op_in_sample": op,
                        "weights": dict(zip([f"rank_{s}" for s in pc.SIGNALS] + [f"nan_{s}" for s in pc.SIGNALS] + ["bias"],
                                            map(float, w))),
                        "fired_ambiguous": int((c >= thr).sum()), "help": int(((c >= thr) & correct).sum())}
    return gates, info, feats


FEATS = None


def gate_fires(name, s):
    kind, p = GATES[name]
    if kind == "threshold_e":
        e = s[:, 4]
        return np.isfinite(e) & (e >= p)
    w, thr, _ = p
    return (FEATS(s) @ w) >= thr


def process_scan(scan):
    import torch
    import torch.nn.functional as F
    import yaml
    from data.datasets.dtu import DTUDataset
    from evaluation.stage_depth import _collate_one
    from models import build_model, regress_depth

    torch.set_num_threads(ARGS.threads)
    cfg = yaml.safe_load(open(ARGS.config))
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(ARGS.ckpt, map_location="cpu", weights_only=False)["model"])
    model.eval()
    prior_root = cfg["data"].get("prior_root", os.path.join(cfg["data"]["root"], "MonoPrior"))
    fits = json.load(open(os.path.join(prior_root, f"{scan}_train", "fits.json")))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(scan + "\n")
    ds = DTUDataset(root=cfg["data"]["root"], split="val", cfg={**cfg["data"], "scan_list_file": f.name})
    os.unlink(f.name)
    t0 = time.time()
    counts = {name: {"fired": 0, "fired_model_right": 0, "harm": 0, "fired_model_wrong": 0,
                     "fired_model_wrong_prior_better": 0} for name in GATES}
    n_valid = n_right = 0
    for i in range(min(len(ds), ARGS.max_views or len(ds))):
        sample = ds[i]
        batch = _collate_one(sample, "cpu")
        with torch.no_grad():
            out = model(**batch["model_inputs"], alpha=1.0, max_gate=1.0)
        hyp_c = out["depth_hypotheses"]["coarse"]
        coarse = regress_depth(out["scores"]["coarse"], hyp_c)[0, 0].numpy()
        fine = regress_depth(out["scores"]["fine"], out["depth_hypotheses"]["fine"])
        Hc, Wc = coarse.shape
        down = lambda x: F.interpolate(x, size=(Hc, Wc), mode="nearest")[0, 0].numpy()  # noqa: E731
        gt_c = down(batch["gt_depth"])
        err_c = down((fine - batch["gt_depth"]).abs())  # as in costvol_profile_probe
        g_c, pv_c = down(batch["prior_depth"]), down(batch["prior_valid"]) > 0.5
        hyps = hyp_c[0, :, 0, 0].numpy()
        step = hyps[1] - hyps[0]
        valid = (gt_c > 0) & (gt_c > hyps[0] - step / 2) & (gt_c < hyps[-1] + step / 2) & pv_c
        sig = pc.view_signals(sample, prior_root, fits, coarse, Hc, Wc)
        ys, xs = np.nonzero(valid)
        s = sig[:, ys, xs].T
        right = err_c[ys, xs] <= ARGS.wrong_mm
        e_prior = np.abs(g_c - gt_c)[ys, xs]
        n_valid += len(ys)
        n_right += int(right.sum())
        for name in GATES:
            fire = gate_fires(name, s)
            c = counts[name]
            c["fired"] += int(fire.sum())
            c["fired_model_right"] += int((fire & right).sum())
            c["harm"] += int((fire & right & (e_prior > ARGS.wrong_mm)).sum())
            c["fired_model_wrong"] += int((fire & ~right).sum())
            c["fired_model_wrong_prior_better"] += int((fire & ~right & (e_prior < err_c[ys, xs])).sum())
    print(f"{scan:<8} valid {n_valid:>8} ({time.time() - t0:.0f}s)", flush=True)
    return scan, {"n_valid": n_valid, "n_model_right": n_right, "gates": counts}


def main():
    global ARGS, GATES, FEATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pixels", required=True)
    ap.add_argument("--scan_list", default="data/datasets/splits/dtu_test.txt")
    ap.add_argument("--wrong_mm", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max_views", type=int, default=None, help="smoke tests only")
    ap.add_argument("--out", required=True)
    ARGS = ap.parse_args()
    GATES, info, FEATS = fit_gates(ARGS.pixels)
    scans = [s.strip() for s in open(ARGS.scan_list) if s.strip()]
    with Pool(min(ARGS.workers, len(scans))) as pool:
        per_scan = dict(pool.imap_unordered(process_scan, scans))
    out = {"wrong_mm": ARGS.wrong_mm, "gates": {}, "scans": per_scan}
    n_valid = sum(v["n_valid"] for v in per_scan.values())
    n_right = sum(v["n_model_right"] for v in per_scan.values())
    for name in GATES:
        tot = {k: sum(v["gates"][name][k] for v in per_scan.values()) for k in per_scan[scans[0]]["gates"][name]}
        help_ = info[name]["help"]
        out["gates"][name] = {**info[name], **tot,
                              "fired_pct_of_valid": 100 * tot["fired"] / n_valid,
                              "fired_model_right_pct_of_fired": 100 * tot["fired_model_right"] / max(tot["fired"], 1),
                              "harm_pct_of_model_right": 100 * tot["harm"] / max(n_right, 1),
                              "harm_to_help_ratio": tot["harm"] / max(help_, 1)}
        print(name, json.dumps({k: v for k, v in out["gates"][name].items() if k != "weights"}, indent=1), flush=True)
    out["n_valid"], out["n_model_right"] = n_valid, n_right
    with open(ARGS.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
