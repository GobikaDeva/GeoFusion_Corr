#!/usr/bin/env bash
# B2_capacity 3-seed numbers, modelled on the aggregate step of scripts/eval_stage0_a0f.sh.
# Reads only runs/B2_capacity/seed{0,1,2}/eval_1600x1152.json and
# runs/B2_nogeo/seed2/eval_1600x1152.json; no GPU. Prints per-seed and mean +- std
# (population std, ddof=0, as aggregate_over_seeds uses for A0f) for Acc/Comp/Overall,
# writes runs/B2_capacity/aggregated_1600x1152.json, and prints B2_nogeo seed2 next to
# B2_capacity seed2 as a matched pair. Refuses to run until scripts/queue_b2_seeds_nogeo.sh
# has logged QUEUE_DONE and its process group has exited.
set -euo pipefail
cd "$(dirname "$0")/.."
QLOG=runs/queue_b2_seeds_nogeo.log
QPID_FILE=runs/queue_b2_seeds_nogeo.pid

if [ -f "$QPID_FILE" ]; then
  QPID=$(cat "$QPID_FILE")
  if [ -n "$(ps -o pid= -g "$QPID" 2>/dev/null)" ]; then
    echo "queue process group $QPID is still active; not running" >&2; exit 1
  fi
fi
if ! grep -q QUEUE_DONE "$QLOG" 2>/dev/null; then
  echo "$QLOG has no QUEUE_DONE line; not running" >&2; exit 1
fi

CUDA_VISIBLE_DEVICES="" python - <<'PY'
import json
import numpy as np

def load(path):
    r = json.load(open(path))
    assert r["n_scans"] == 22 and list(r["test_img_wh"]) == [1600, 1152], (path, r["n_scans"], r["test_img_wh"])
    return r

def agg(values):
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.shape[0])}

out = "runs/B2_capacity"
seeds = (0, 1, 2)
results = [load(f"{out}/seed{s}/eval_1600x1152.json") for s in seeds]
metrics = ("accuracy", "completeness", "overall")

print(f"{'B2_capacity':<14}{'Acc':>8}{'Comp':>8}{'Overall':>9}")
for s, r in zip(seeds, results):
    print(f"{'seed' + str(s):<14}" + "".join(f"{r[m]:>{w}.4f}" for m, w in zip(metrics, (8, 8, 9))))
aggregated = {k: agg([r[k] for r in results]) for k in metrics + ("runtime_sec",)}
print(f"{'mean +- std':<14}" + "  ".join(f"{aggregated[m]['mean']:.4f}+-{aggregated[m]['std']:.4f}" for m in metrics))
aggregated["test_img_wh"] = results[0]["test_img_wh"]
aggregated["n_scans"] = [r["n_scans"] for r in results]
json.dump(aggregated, open(f"{out}/aggregated_1600x1152.json", "w"), indent=2)
print(f"wrote {out}/aggregated_1600x1152.json")

nogeo = load("runs/B2_nogeo/seed2/eval_1600x1152.json")
b2 = results[seeds.index(2)]
print("\nMatched pair, seed 2 (only loss.geo_weight differs: 0.5 vs 0.0)")
print(f"{'':<16}{'Acc':>8}{'Comp':>8}{'Overall':>9}")
for name, r in (("B2_capacity", b2), ("B2_nogeo", nogeo)):
    print(f"{name:<16}" + "".join(f"{r[m]:>{w}.4f}" for m, w in zip(metrics, (8, 8, 9))))
print(f"{'nogeo - B2':<16}" + "".join(f"{nogeo[m] - b2[m]:>+{w}.4f}" for m, w in zip(metrics, (8, 8, 9))))
better = sum(nogeo["per_scan"][k]["overall"] < b2["per_scan"][k]["overall"] for k in b2["per_scan"])
print(f"B2_nogeo better overall on {better}/{len(b2['per_scan'])} scans")
PY
