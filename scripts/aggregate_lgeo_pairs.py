"""L_geo on/off matched pairs: C1 (runs/B2_capacity, geo_weight 0.5) vs B2_nogeo (geo_weight 0.0).

For each seed with both evals present (seeds 0 and 2 by default): Acc/Comp/Overall of both
arms, the paired difference (nogeo - C1; positive means L_geo helps, since lower is better),
and the per-scan Overall win count over the 22 test scans. Across the available pairs it prints
the mean paired difference and its std with ddof=0 (population std), the convention used for
the A0f and C1 3-seed tables (scripts/aggregate_b2_3seed.sh). CPU only; reads eval JSONs only.

Usage: python scripts/aggregate_lgeo_pairs.py [--seeds 0 2] [--out runs/B2_nogeo/lgeo_pairs.json]
A seed whose B2_nogeo eval is missing is reported as PENDING and left out of the aggregate.
"""
import argparse
import json
import os

import numpy as np

METRICS = ("accuracy", "completeness", "overall")
ON, OFF = "runs/B2_capacity", "runs/B2_nogeo"


def load(path):
    r = json.load(open(path))
    assert r["n_scans"] == 22 and list(r["test_img_wh"]) == [1600, 1152], (path, r["n_scans"], r["test_img_wh"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--out", default=f"{OFF}/lgeo_pairs.json")
    args = ap.parse_args()

    pairs, pending = {}, []
    for s in args.seeds:
        on_p, off_p = f"{ON}/seed{s}/eval_1600x1152.json", f"{OFF}/seed{s}/eval_1600x1152.json"
        if not (os.path.exists(on_p) and os.path.exists(off_p)):
            pending.append(s)
            continue
        on, off = load(on_p), load(off_p)
        assert set(on["per_scan"]) == set(off["per_scan"])
        scans = sorted(on["per_scan"])
        on_wins = sum(on["per_scan"][k]["overall"] < off["per_scan"][k]["overall"] for k in scans)
        off_wins = sum(off["per_scan"][k]["overall"] < on["per_scan"][k]["overall"] for k in scans)
        pairs[s] = {
            "C1_lgeo_on": {m: on[m] for m in METRICS},
            "B2_nogeo_lgeo_off": {m: off[m] for m in METRICS},
            "diff_off_minus_on": {m: off[m] - on[m] for m in METRICS},
            "scan_wins_overall": {"C1_lgeo_on": on_wins, "B2_nogeo_lgeo_off": off_wins,
                                  "ties": len(scans) - on_wins - off_wins, "n_scans": len(scans)},
        }

    w = (8, 8, 9)
    for s, p in pairs.items():
        print(f"\nseed {s}{'':<10}" + "".join(f"{h:>{x}}" for h, x in zip(("Acc", "Comp", "Overall"), w)))
        for name, key in (("C1 (L_geo on)", "C1_lgeo_on"), ("B2_nogeo (off)", "B2_nogeo_lgeo_off")):
            print(f"{name:<16}" + "".join(f"{p[key][m]:>{x}.4f}" for m, x in zip(METRICS, w)))
        print(f"{'off - on':<16}" + "".join(f"{p['diff_off_minus_on'][m]:>+{x}.4f}" for m, x in zip(METRICS, w)))
        sw = p["scan_wins_overall"]
        print(f"per-scan Overall wins: L_geo on {sw['C1_lgeo_on']}/{sw['n_scans']}, "
              f"off {sw['B2_nogeo_lgeo_off']}/{sw['n_scans']}, ties {sw['ties']}")
    for s in pending:
        print(f"\nseed {s}: PENDING (eval JSON missing)")

    agg = {}
    if pairs:
        for m in METRICS:
            d = np.array([p["diff_off_minus_on"][m] for p in pairs.values()], dtype=np.float64)
            agg[m] = {"mean": float(d.mean()), "std_ddof0": float(d.std(ddof=0)), "n": int(d.size)}
        print(f"\npaired diff (off - on), mean +- std (ddof=0), n={len(pairs)}: "
              + "  ".join(f"{m} {agg[m]['mean']:+.4f}+-{agg[m]['std_ddof0']:.4f}" for m in METRICS))
        same_sign = all(np.sign(p["diff_off_minus_on"]["overall"]) == np.sign(agg["overall"]["mean"])
                        for p in pairs.values())
        print(f"Overall difference has the same sign in every pair: {same_sign}")
        agg["overall_same_sign_all_pairs"] = bool(same_sign)

    json.dump({"ddof": 0, "seeds": args.seeds, "pending": pending,
               "pairs": {str(k): v for k, v in pairs.items()}, "aggregate": agg},
              open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
