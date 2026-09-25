#!/usr/bin/env python
"""Small-multiples plot of coarse-stage cost profiles saved by
scripts/costvol_profile_probe.py. Needs only numpy + matplotlib.

Usage: python scripts/plot_costvol_profiles.py <scan>_examples.pkl out.png [--n 6] [--seed 0]
"""
import argparse
import pickle

import numpy as np


def plot_examples(results, path, n=6, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pool = [(vi, j) for vi, r in enumerate(results) if "examples" in r for j in range(len(r["examples"]["ys"]))]
    picks = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    fig, axes = plt.subplots(2, len(picks), figsize=(3.1 * len(picks), 5.6), sharex=False)
    for col, (vi, j) in enumerate(picks):
        ex = results[vi]["examples"]
        h = ex["hyps"]
        norm = lambda c: (c - np.nanmin(c[np.isfinite(c)])) / (np.ptp(c[np.isfinite(c)]) + 1e-12)  # noqa: E731
        ax = axes[0, col]
        ax.plot(h, norm(ex["raw_mean"][j]), color=blue, lw=2, label="all 5 views (model input)")
        ax.plot(h, norm(ex["best2"][j]), color=orange, lw=2, label="best-2 source views")
        ax2 = axes[1, col]
        ax2.plot(h, ex["prob"][j], color=aqua, lw=2, label="after 3D regularizer")
        for a in (ax, ax2):
            a.axvline(ex["gt"][j], color=ink, lw=1.2)
            a.axvline(ex["fine_pred"][j], color=muted, lw=1.2, ls="--")
            a.grid(color=grid, lw=0.8)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.tick_params(colors=muted, labelsize=8)
        ax.set_title(f"view {results[vi]['view']} px ({ex['ys'][j]},{ex['xs'][j]})\n"
                     f"GT {ex['gt'][j]:.0f}mm, fine {ex['fine_pred'][j]:.0f}mm", fontsize=8.5, color=ink)
        ax2.set_xlabel("depth hypothesis (mm)", fontsize=8, color=muted)
    axes[0, 0].set_ylabel("raw matching cost\n(normalized, low = match)", fontsize=8.5, color=muted)
    axes[1, 0].set_ylabel("coarse-stage probability", fontsize=8.5, color=muted)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    extra = [Line2D([], [], color=ink, lw=1.2), Line2D([], [], color=muted, lw=1.2, ls="--")]
    fig.legend(h1 + h2 + extra, l1 + l2 + ["GT depth", "final fine-stage depth"], loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle("scan48: coarse cost profiles at pixels where the fine stage is off by >4mm",
                 y=0.995, fontsize=10.5, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(a.pkl, "rb") as f:
        plot_examples(pickle.load(f), a.out, n=a.n, seed=a.seed)
