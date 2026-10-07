#!/usr/bin/env python
"""Test 1 redone without test-set selection: confidence-gate thresholds are chosen on
dtu_val and frozen, then scored once on the 22 test scans, together with the
deployment harm check.

scripts/prior_confidence.py and scripts/prior_gate_harm.py picked each gate's
operating point on the test scans themselves. Here:
  val   for every gate, the threshold (and, for "combined", the logistic weights and
        rank references) is fit on val ambiguous pixels: the operating point is the
        best correct-dip % at >= 20% coverage, as before. Each single signal is tried
        in both orientations, since the reversed one was itself found on test.
  test  that frozen threshold is applied unchanged. Rule (pre-registered): coverage
        >= 20% of ambiguous pixels and correct dip >= 80%.
Harm (test, every valid coarse pixel, not only ambiguous ones): wherever a gate
fires, suppose the prior overrides the model (depth := G) and report the share of
fired pixels the model already had right (fine error <= --wrong_mm) and the net change
in error, |G - GT| - |model - GT|.

Definitions match prior_confidence.py: coarse grid, 640x512 val-mode images
(lighting 3); ambiguous = model wrong (> --wrong_mm) and GT only a secondary local
minimum of the raw channel-mean coarse cost; correct dip = |G - true dip| <
|G - chosen dip|; coverage is a share of ALL ambiguous pixels, prior-invalid included.
The model error is the full-res fine error sampled nearest onto the coarse grid
(as in costvol_profile_probe.py). Runs on CPU, one process per scan; per-scan pixel
files are cached in <out_dir>/pixels/ so collection resumes.

Usage:
    python scripts/prior_gate_valfrozen.py --config configs/ablations/A0f_fixes_only_short.yaml \
        --ckpt runs/A0f_fixes_only_short/seed2/ckpt_final.pt --out_dir runs/prior_confidence_val
    (--analyze_only skips collection)
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

SPLITS = {"val": "data/datasets/splits/dtu_val.txt", "test": "data/datasets/splits/dtu_test.txt"}
ARGS = None


# ----------------------------------------------------------------------------- collect
def process_scan(job):
    split, scan = job
    path = os.path.join(ARGS.out_dir, "pixels", f"{split}_{scan}.npz")
    if os.path.exists(path):
        return f"{split} {scan:<8} cached"
    import torch
    import torch.nn.functional as F
    import yaml
    from costvol_profile_probe import classify
    from data.datasets.dtu import DTUDataset
    from evaluation.stage_depth import _collate_one
    from models import build_model, regress_depth
    from prior_vs_dips import dips

    torch.set_num_threads(ARGS.threads)
    cfg = yaml.safe_load(open(ARGS.config))
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(ARGS.ckpt, map_location="cpu", weights_only=False)["model"])
    model.eval()
    bb = model.backbone
    prior_root = cfg["data"].get("prior_root", os.path.join(cfg["data"]["root"], "MonoPrior"))
    fits = json.load(open(os.path.join(prior_root, f"{scan}_train", "fits.json")))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(scan + "\n")
    ds = DTUDataset(root=cfg["data"]["root"], split="val", cfg={**cfg["data"], "scan_list_file": f.name})
    os.unlink(f.name)
    t0 = time.time()
    cols = {k: [] for k in ("sig", "pv", "e_model", "e_prior", "amb", "correct", "sep_gt_err", "view")}
    for i in range(min(len(ds), ARGS.max_views or len(ds))):
        sample = ds[i]
        batch = _collate_one(sample, "cpu")
        inp = batch["model_inputs"]
        with torch.no_grad():
            out = model(**inp, alpha=1.0, max_gate=1.0)
            fine = regress_depth(out["scores"]["fine"], out["depth_hypotheses"]["fine"])
            hyp = out["depth_hypotheses"]["coarse"]
            coarse = regress_depth(out["scores"]["coarse"], hyp)[0, 0]
            # the raw channel-mean coarse cost, exactly as costvol_profile_probe.probe_view
            stage = bb.cfg.stages[0]
            ref_feats, src_feats = bb.extract_features(inp["ref_img"], inp["src_imgs"], inp["ref_geom"])
            raw = bb.build_cost_volume(ref_feats["coarse"], [sf["coarse"] for sf in src_feats],
                                       bb._scale_projection(inp["ref_proj"], stage.resolution_scale),
                                       [bb._scale_projection(p, stage.resolution_scale) for p in inp["src_projs"]],
                                       hyp).mean(1)[0]  # (D, Hc, Wc)
        Hc, Wc = coarse.shape
        down = lambda x: F.interpolate(x, size=(Hc, Wc), mode="nearest")[0, 0]  # noqa: E731
        gt_c = down(batch["gt_depth"])
        err_c = down((fine - batch["gt_depth"]).abs())
        fine_c = down(fine)
        g_c, pv_c = down(batch["prior_depth"]), down(batch["prior_valid"]) > 0.5
        hyps = hyp[0, :, 0, 0]
        step = (hyps[1] - hyps[0]).item()
        valid = (gt_c > 0) & (gt_c > hyps[0] - step / 2) & (gt_c < hyps[-1] + step / 2)
        ys, xs = torch.nonzero(valid, as_tuple=True)
        e_model = err_c[ys, xs]
        wrong = e_model > ARGS.wrong_mm
        gt = gt_c[ys, xs]
        amb = torch.zeros_like(wrong)
        if wrong.any():
            gt_bin = ((gt[wrong] - hyps[0]) / step).round().long().clamp(0, len(hyps) - 1)
            amb[wrong] = torch.as_tensor(classify(raw[:, ys[wrong], xs[wrong]].T, gt_bin) == 1)
        sig = pc.view_signals(sample, prior_root, fits, coarse.numpy(), Hc, Wc)
        ys_n, xs_n = ys.numpy(), xs.numpy()
        g, pv = g_c.numpy()[ys_n, xs_n], pv_c.numpy()[ys_n, xs_n]
        e_prior = np.where(pv, np.abs(g - gt.numpy()), np.nan)
        correct = np.zeros(len(ys_n), bool)
        sep = np.zeros(len(ys_n), bool)
        a = amb.numpy()
        if a.any():
            d_true, d_chosen = dips(raw[:, ys[amb], xs[amb]].T.numpy(), hyps.numpy(), gt[amb].numpy(),
                                    fine_c[ys[amb], xs[amb]].numpy())
            correct[a] = np.abs(g[a] - d_true) < np.abs(g[a] - d_chosen)
            sep[a] = np.abs(d_chosen - d_true) > e_prior[a]
        for k, v in (("sig", sig[:, ys_n, xs_n].T), ("pv", pv), ("e_model", e_model.numpy()), ("e_prior", e_prior),
                     ("amb", a), ("correct", correct), ("sep_gt_err", sep), ("view", np.full(len(ys_n), i, np.int16))):
            cols[k].append(v)
    np.savez_compressed(path + ".tmp.npz", **{k: np.concatenate(v) for k, v in cols.items()})
    os.replace(path + ".tmp.npz", path)
    n = sum(len(v) for v in cols["pv"])
    n_amb = int(sum(v.sum() for v in cols["amb"]))
    return f"{split} {scan:<8} valid {n:>8} ambiguous {n_amb:>6} ({time.time() - t0:.0f}s)"


# ----------------------------------------------------------------------------- analyze
def load(split):
    scans = [s.strip() for s in open(SPLITS[split]) if s.strip()]
    parts = [dict(np.load(os.path.join(ARGS.out_dir, "pixels", f"{split}_{s}.npz"))) for s in scans]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d["scan"] = np.concatenate([np.full(len(p["pv"]), i, np.int16) for i, p in enumerate(parts)])
    d["sig_pv"] = np.where(d["pv"][:, None], d["sig"], np.nan)  # prior-invalid pixels never fire
    d["scans"] = scans
    return d


def single_conf(sig_pv, j, orient):
    """Higher = more confident; orient 'fwd' = the pre-specified direction."""
    c = pc.confidence(sig_pv[:, j], pc.SIGNALS[j])
    return c if orient == "fwd" else np.where(np.isfinite(c), -c, -np.inf)


def fit_combined(v):
    """Logistic gate fit on all val ambiguous prior-valid pixels; features are ranks
    against the val ambiguous distribution, so they transfer to test unchanged."""
    m = v["amb"] & v["pv"]
    ref = []
    for j, s in enumerate(pc.SIGNALS):
        c = pc.confidence(v["sig_pv"][m, j], s)
        ref.append(np.sort(c[np.isfinite(c)]))

    def feats(sig_pv):
        cols = []
        for j, name in enumerate(pc.SIGNALS):
            c = pc.confidence(sig_pv[:, j], name)
            r = np.zeros(len(c))
            f = np.isfinite(c)
            r[f] = np.searchsorted(ref[j], c[f]) / max(len(ref[j]), 1)
            cols.append(r)
        cols += [np.isnan(sig_pv[:, j]).astype(float) for j in range(len(pc.SIGNALS))]
        return np.column_stack(cols + [np.ones(len(sig_pv))])

    w = pc.logistic_fit(feats(v["sig_pv"][m]), v["correct"][m].astype(float))
    names = [f"rank_{s}" for s in pc.SIGNALS] + [f"nan_{s}" for s in pc.SIGNALS] + ["bias"]
    return (lambda d: np.where(d["pv"], feats(d["sig_pv"]) @ w, -np.inf)), dict(zip(names, map(float, w)))


def ambiguous_score(conf_amb, correct, sep, n_amb, thr):
    fire = conf_amb >= thr
    n = int(fire.sum())
    return {"n_fired": n, "coverage_pct": 100 * n / n_amb,
            "correct_dip_pct": 100 * float(correct[fire].mean()) if n else float("nan"),
            "sep_gt_err_pct": 100 * float(sep[fire].mean()) if n else float("nan"),
            "meets_rule": bool(n and 100 * n / n_amb >= pc.RULE_COVERAGE and 100 * correct[fire].mean() >= pc.RULE_CORRECT)}


def harm(conf_all, d, wrong_mm):
    """Override = depth := G on every fired pixel (all valid pixels, test)."""
    fire = np.isfinite(conf_all) & (conf_all >= 0)  # conf_all is already conf - thr
    n_valid = len(fire)
    em, ep = d["e_model"][fire], d["e_prior"][fire]
    right = em <= wrong_mm
    delta = ep - em
    n = int(fire.sum())
    if n == 0:
        return {"n_fired": 0}
    return {
        "n_fired": n, "fired_pct_of_valid": 100 * n / n_valid,
        "fired_model_right_pct": 100 * float(right.mean()),
        "fired_model_right_prior_wrong_pct": 100 * float((right & (ep > wrong_mm)).mean()),
        "fired_model_wrong_prior_right_pct": 100 * float((~right & (ep <= wrong_mm)).mean()),
        "net_delta_err_mean_mm_per_fired": float(delta.mean()),
        "net_delta_err_median_mm_per_fired": float(np.median(delta)),
        "delta_err_mean_mm_on_model_right": float(delta[right].mean()) if right.any() else float("nan"),
        "delta_err_mean_mm_on_model_wrong": float(delta[~right].mean()) if (~right).any() else float("nan"),
        "net_delta_mean_err_mm_all_valid": float(delta.sum() / n_valid),
        "model_mean_err_mm_all_valid": float(d["e_model"].mean()),
        "net_delta_pct_px_within_wrong_mm_all_valid":
            100 * (int((~right & (ep <= wrong_mm)).sum()) - int((right & (ep > wrong_mm)).sum())) / n_valid,
    }


def analyze():
    v, t = load("val"), load("test")
    out = {"wrong_mm": ARGS.wrong_mm, "selection": "thresholds, orientations and logistic weights fit on dtu_val only",
           "rule": f"coverage >= {pc.RULE_COVERAGE}% of all ambiguous and correct dip >= {pc.RULE_CORRECT}%",
           "n": {s: {"valid": int(len(d["pv"])), "ambiguous": int(d["amb"].sum()),
                     "ambiguous_prior_valid_pct": 100 * float(d["pv"][d["amb"]].mean()),
                     "baseline_correct_dip_pct_prior_valid": 100 * float(d["correct"][d["amb"] & d["pv"]].mean())}
                 for s, d in (("val", v), ("test", t))},
           "gates": {}}
    gates = [(f"{s}_{o}", (lambda d, j=j, o=o: single_conf(d["sig_pv"], j, o)))
             for j, s in enumerate(pc.SIGNALS) for o in ("fwd", "rev")]
    comb, weights = fit_combined(v)
    gates.append(("combined", comb))
    for name, conf_fn in gates:
        cv = conf_fn(v)
        a = v["amb"]
        op = pc.operating_points(cv[a], v["correct"][a], v["sep_gt_err"][a], int(a.sum()))["best_correct_at_cov_ge_20"]
        g = {"val_op": op}
        if op is None:
            g["threshold"] = None
            out["gates"][name] = g
            continue
        fin = np.isfinite(cv[a])
        thr = float(np.sort(cv[a][fin])[::-1][op["n"] - 1])
        g["threshold"] = thr
        g["val"] = ambiguous_score(cv[a], v["correct"][a], v["sep_gt_err"][a], int(a.sum()), thr)
        ct = conf_fn(t)
        b = t["amb"]
        g["test"] = ambiguous_score(ct[b], t["correct"][b], t["sep_gt_err"][b], int(b.sum()), thr)
        g["test_harm"] = harm(ct - thr, t, ARGS.wrong_mm)
        g["val_harm"] = harm(cv - thr, v, ARGS.wrong_mm)
        if name == "combined":
            g["weights"] = weights
        out["gates"][name] = g
    out["passing_on_test"] = [n for n, g in out["gates"].items() if g.get("test", {}).get("meets_rule")]
    out["passing_on_val"] = [n for n, g in out["gates"].items() if g.get("val", {}).get("meets_rule")]

    print(json.dumps(out["n"], indent=1))
    print(f"\n{'gate':<20} {'thr':>9} | {'val cov':>7} {'val cor':>7} | {'test cov':>8} {'test cor':>8} {'rule':>5} | "
          f"{'fired%':>6} {'mdlOK%':>6} {'dErr/fired':>10} {'dErr/all':>8} {'d<4mm%':>7}")
    for name, g in out["gates"].items():
        if g["threshold"] is None:
            print(f"{name:<20} {'cover>=20% unreachable on val':>40}")
            continue
        h = g["test_harm"]
        print(f"{name:<20} {g['threshold']:9.3g} | {g['val']['coverage_pct']:7.1f} {g['val']['correct_dip_pct']:7.1f} | "
              f"{g['test']['coverage_pct']:8.1f} {g['test']['correct_dip_pct']:8.1f} {str(g['test']['meets_rule']):>5} | "
              f"{h.get('fired_pct_of_valid', 0):6.2f} {h.get('fired_model_right_pct', float('nan')):6.1f} "
              f"{h.get('net_delta_err_mean_mm_per_fired', float('nan')):10.2f} {h.get('net_delta_mean_err_mm_all_valid', float('nan')):8.3f} "
              f"{h.get('net_delta_pct_px_within_wrong_mm_all_valid', float('nan')):7.3f}")
    print("\npassing on val:", out["passing_on_val"], " passing on test (val-frozen):", out["passing_on_test"])
    with open(os.path.join(ARGS.out_dir, "gates_valfrozen.json"), "w") as f:
        json.dump(out, f, indent=2)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wrong_mm", type=float, default=4.0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max_views", type=int, default=None, help="smoke tests only")
    ap.add_argument("--analyze_only", action="store_true")
    ARGS = ap.parse_args()
    os.makedirs(os.path.join(ARGS.out_dir, "pixels"), exist_ok=True)
    if not ARGS.analyze_only:
        jobs = [(s, scan.strip()) for s in ARGS.splits for scan in open(SPLITS[s]) if scan.strip()]
        with Pool(min(ARGS.workers, len(jobs))) as pool:
            for msg in pool.imap_unordered(process_scan, jobs):
                print(msg, flush=True)
    if ARGS.max_views is None:
        analyze()


if __name__ == "__main__":
    main()
