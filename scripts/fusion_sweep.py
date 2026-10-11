#!/usr/bin/env python
"""Fusion-threshold sweep: does C1's (B2_capacity's) completeness gain over A0f survive
when both models are fused at matched point counts?

Two steps, both resumable (outputs skipped if present):

  dump   runs each model through the official evaluation path
         (eval_dtu._predict_scan_views: eval.test_data, 1600x1152, fine depth,
         max-softmax confidence) and saves <out_dir>/maps/<model>_<scan>.npz
         (depth, conf, proj). Needs inference; --device cpu works but is slow.
  sweep  CPU only. Re-fuses the saved maps under a grid of thresholds and scores
         each with the official metric. Writes <out_dir>/<model>_<scan>.json.

The fusion is evaluation/fusion.py's GeoMVSNet loop (every view checked against every
reference view, consistent points averaged, the reference counts as one view, conf >
prob_thresh first). Per (ref pixel, view) it records three quantities at once, so all
settings come from one pass:
  dist    world distance between the ref point and the view's point sampled at the
          ref pixel's reprojection (the quantity the official fusion thresholds)
  reproj  pixel distance between the ref pixel and that view's point projected back
          into the ref view (MVSNet/CasMVSNet-style check)
  reldep  |depth of that point in the ref view - ref depth| / ref depth (ditto)
Families (a view is consistent if it passes every test of the family):
  geo     dist < dist_thresh                         (official: 0.2 mm, >= 4 views)
  mvs     reproj < px and reldep < rel                (MVSNet-style, same view count rule)
Settings are crossed with num_consist (views incl. the reference). The official
setting (geo, 0.2 mm, 4 views, prob 0.3) must reproduce eval_1600x1152.json.

Usage:
    python scripts/fusion_sweep.py dump  --out_dir runs/fusion_sweep --device cpu --scans scan1 ...
    python scripts/fusion_sweep.py sweep --out_dir runs/fusion_sweep --scans scan1 ... --workers 4
    python scripts/fusion_sweep.py report --out_dir runs/fusion_sweep --scans scan1 ...
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

from evaluation.fusion import GEOMVSNET_PROB_THRESH, _points_from_depth, _warp_to_ref  # noqa: E402

MODELS = {
    "A0f": ("configs/ablations/A0f_fixes_only_short.yaml", "runs/A0f_fixes_only_short/seed2/ckpt_final.pt"),
    "C1": ("configs/ablations/B2_capacity.yaml", "runs/B2_capacity/seed2/ckpt_final.pt"),
}
GEO_DIST = (0.1, 0.2, 0.4, 0.8)            # mm
MVS_PX_REL = ((0.5, 0.001), (1.0, 0.001), (1.0, 0.01), (2.0, 0.01))
NUM_CONSIST = (2, 3, 4, 6)


def settings(keep=None):
    """The full grid, or only the settings whose keys are in `keep` (--settings)."""
    out = []
    for n in NUM_CONSIST:
        out += [{"family": "geo", "dist": d, "n": n} for d in GEO_DIST]
        out += [{"family": "mvs", "px": px, "rel": rel, "n": n} for px, rel in MVS_PX_REL]
    if keep:
        unknown = set(keep) - {setting_key(s) for s in out}
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        out = [s for s in out if setting_key(s) in keep]
    return out


def setting_key(s):
    return f"geo_d{s['dist']}_n{s['n']}" if s["family"] == "geo" else f"mvs_px{s['px']}_rel{s['rel']}_n{s['n']}"


# ---------------------------------------------------------------- dump
def dump(args):
    from data.datasets.dtu import DTUDataset
    from evaluation.eval_dtu import _predict_scan_views
    from models import build_model

    os.makedirs(os.path.join(args.out_dir, "maps"), exist_ok=True)
    for name, (cfg_path, ckpt) in MODELS.items():
        cfg = yaml.safe_load(open(cfg_path))
        model = build_model(cfg["model"])
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        model.to(args.device).eval()
        test_data = {**cfg["data"], **cfg.get("eval", {}).get("test_data", {})}
        for scan in args.scans:
            path = os.path.join(args.out_dir, "maps", f"{name}_{scan}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write(scan + "\n")
            ds = DTUDataset(root=test_data["root"], split="test", cfg={**test_data, "scan_list_file": f.name})
            os.unlink(f.name)
            if args.max_views:
                ds.metas = ds.metas[:args.max_views]
            depths, confs, projs = _predict_scan_views(model, ds, scan, args.device)
            np.savez(path + ".tmp.npz", depth=depths.astype(np.float32), conf=confs.astype(np.float16),
                     proj=projs.astype(np.float32), gt_root=cfg["data"]["gt_root"])
            os.replace(path + ".tmp.npz", path)
            print(f"dump {name} {scan} views={len(depths)} {time.time() - t0:.0f}s", flush=True)
        del model


# ---------------------------------------------------------------- sweep
def fuse_all_settings(depths, confs, projs, prob_thresh, sets, batch_size=7):
    """One pass over all (ref, view) pairs; returns {key: (N, 3) points}."""
    d = torch.from_numpy(depths * (confs.astype(np.float32) > prob_thresh)).float().unsqueeze(1)
    P = torch.from_numpy(projs).float()
    V, _, H, W = d.shape
    pts = {setting_key(s): [] for s in sets}
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    with torch.no_grad():
        for i in range(V):
            ref_pc = _points_from_depth(d[i:i + 1], P[i:i + 1])           # (1, 3, H, W)
            ref_d = d[i, 0]
            valid_ref = ref_d > 0
            sums = {k: torch.zeros(3, H, W) for k in pts}
            cnts = {k: torch.zeros(H, W) for k in pts}
            for j in range(0, V, batch_size):
                src_pcs = _points_from_depth(d[j:j + batch_size], P[j:j + batch_size])
                aligned = _warp_to_ref(src_pcs, P[j:j + batch_size], P[i:i + 1], d[i:i + 1], align_corners=True)
                dist = torch.sqrt(((ref_pc - aligned) ** 2).sum(dim=1))   # (b, H, W)
                # project the aligned world point back into the ref view
                hom = torch.cat([aligned, torch.ones_like(aligned[:, :1])], dim=1).view(aligned.shape[0], 4, -1)
                q = torch.matmul(P[i, :3], hom).view(aligned.shape[0], 3, H, W)
                z = q[:, 2]
                zs = torch.where(z.abs() > 1e-6, z, torch.full_like(z, 1e-6))
                reproj = torch.sqrt((q[:, 0] / zs - xs) ** 2 + (q[:, 1] / zs - ys) ** 2)
                reldep = (z - ref_d).abs() / ref_d.clamp(min=1e-6)
                # a zero (filtered/out-of-view) source point is never consistent
                src_ok = aligned.abs().sum(dim=1) > 0
                for s in sets:
                    k = setting_key(s)
                    if s["family"] == "geo":
                        m = dist < s["dist"]
                    else:
                        m = (reproj < s["px"]) & (reldep < s["rel"]) & (z > 0)
                    m = (m & src_ok & valid_ref).float()
                    sums[k] += (aligned * m.unsqueeze(1)).sum(dim=0)
                    cnts[k] += m.sum(dim=0)
            for s in sets:
                k = setting_key(s)
                keep = cnts[k] >= s["n"]
                avg = (sums[k] / cnts[k].clamp(min=1)).permute(1, 2, 0)
                pts[k].append(avg[keep].numpy())
            del sums, cnts
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in pts.items()}


def sweep_one(job):
    name, scan, out_dir, threads, keep = job
    torch.set_num_threads(threads)
    from evaluation.dtu_gt import load_scan_ground_truth
    from evaluation.eval_dtu import _scan_number, official_scan_metrics

    path = os.path.join(out_dir, f"{name}_{scan}.json")
    if os.path.exists(path):
        return path
    t0 = time.time()
    z = np.load(os.path.join(out_dir, "maps", f"{name}_{scan}.npz"))
    gt = load_scan_ground_truth(str(z["gt_root"]), _scan_number(scan))
    depth, conf, proj = z["depth"], z["conf"], z["proj"]
    res = {"model": name, "scan": scan, "n_views": int(depth.shape[0]), "settings": {}}
    t_fuse = 0.0
    for n in NUM_CONSIST:  # one fusion pass per view count keeps ~8 point clouds in memory, not 32
        sets = [s for s in settings(keep) if s["n"] == n]
        if not sets:
            continue
        t1 = time.time()
        fused = fuse_all_settings(depth, conf, proj, GEOMVSNET_PROB_THRESH, sets)
        t_fuse += time.time() - t1
        for s in sets:
            k = setting_key(s)
            m = official_scan_metrics(fused[k], gt)
            res["settings"][k] = {**s, "n_points": int(len(fused[k])),
                                  **{q: float(m[q]) for q in ("accuracy", "completeness", "overall")}}
        del fused
    res["fuse_sec"], res["total_sec"] = t_fuse, time.time() - t0
    json.dump(res, open(path + ".tmp", "w"), indent=2)
    os.replace(path + ".tmp", path)
    print(f"sweep {name} {scan} fuse {t_fuse:.0f}s total {res['total_sec']:.0f}s", flush=True)
    return path


def sweep(args):
    import multiprocessing as mp
    jobs = [(n, s, args.out_dir, args.threads, args.settings) for s in args.scans for n in MODELS]
    if args.workers <= 1:
        for j in jobs:
            sweep_one(j)
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for _ in pool.imap_unordered(sweep_one, jobs):
                pass


# ---------------------------------------------------------------- report
def report(args):
    """Per setting: mean Acc/Comp/Overall/points over scans, per model. Then a matched-count
    comparison: for each scan and each C1 setting, A0f's acc/comp interpolated (in log point
    count, within one family's curve at fixed n) to C1's point count, and the reverse."""
    R = {n: {s: json.load(open(os.path.join(args.out_dir, f"{n}_{s}.json"))) for s in args.scans} for n in MODELS}
    keys = [setting_key(s) for s in settings(args.settings)]
    official = {"A0f": "runs/A0f_fixes_only_short/seed2/eval_1600x1152.json", "C1": "runs/B2_capacity/seed2/eval_1600x1152.json"}
    for n, p in official.items():
        ref = json.load(open(p))["per_scan"]
        dev = max(abs(R[n][s]["settings"]["geo_d0.2_n4"]["overall"] - ref[s]["overall"]) for s in args.scans)
        print(f"reproduction check {n}: max |overall - {p}| over scans = {dev:.4f} mm")
    table = {}
    for k in keys:
        table[k] = {n: {q: float(np.mean([R[n][s]["settings"][k][q] for s in args.scans]))
                        for q in ("n_points", "accuracy", "completeness", "overall")} for n in MODELS}
    print(f"{'setting':<26}" + "".join(f"{n + ' ' + q:>13}" for n in MODELS for q in ("Mpts", "acc", "comp", "ovr")))
    for k in keys:
        print(f"{k:<26}" + "".join(
            f"{table[k][n]['n_points'] / 1e6:>13.2f}{table[k][n]['accuracy']:>13.3f}"
            f"{table[k][n]['completeness']:>13.3f}{table[k][n]['overall']:>13.3f}" for n in MODELS))

    # matched point count: pool every setting of a model on a scan into one
    # (log count -> metric) curve; compare the other model at the same count.
    def curve(n, scan, q):
        rows = sorted((v["n_points"], v[q]) for v in R[n][scan]["settings"].values() if v["n_points"] > 0)
        return np.log([r[0] for r in rows]), np.array([r[1] for r in rows])

    def at(n, scan, q, count):
        x, y = curve(n, scan, q)
        lc = np.log(count)
        if lc < x.min() or lc > x.max():
            return None
        return float(np.interp(lc, x, y))

    matched = []
    ref_key = "geo_d0.2_n4"
    for scan in args.scans:
        for src, dst in (("A0f", "C1"), ("C1", "A0f")):
            c = R[src][scan]["settings"][ref_key]["n_points"]
            row = {"scan": scan, "count_from": src, "count": c}
            for n in MODELS:
                row[n] = {q: at(n, scan, q, c) for q in ("accuracy", "completeness", "overall")}
            matched.append(row)
    summary = {"scans": args.scans, "per_setting_mean": table, "matched_count": matched}
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print("\nMatched point count (each model's metric at the other's official-setting count; None = outside range)")
    for src in ("A0f", "C1"):
        rows = [r for r in matched if r["count_from"] == src]
        ok = [r for r in rows if all(r[n]["completeness"] is not None for n in MODELS)]
        print(f"count from {src}: {len(ok)}/{len(rows)} scans in range")
        for q in ("accuracy", "completeness", "overall"):
            print(f"  {q:<13}" + "  ".join(f"{n} {np.mean([r[n][q] for r in ok]):.3f}" for n in MODELS)
                  + f"  C1 better on {sum(r['C1'][q] < r['A0f'][q] for r in ok)}/{len(ok)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("dump", "sweep", "report"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scans", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_views", type=int, default=0, help="smoke test only")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--settings", nargs="+", default=None,
                   help="sweep/report only these setting keys (e.g. geo_d0.2_n4); default: full grid")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    {"dump": dump, "sweep": sweep, "report": report}[args.step](args)


if __name__ == "__main__":
    main()
