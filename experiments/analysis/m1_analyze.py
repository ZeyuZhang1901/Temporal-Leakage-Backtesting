#!/usr/bin/env python3
"""M1: provably-clean flagships still 'fail' the naive check.

Each model is scored ONLY on questions resolving after its documented
cutoff + 30-day guard band; the pre/post boundary is the median resolution
date of that clean window. Output: m1/results.json + figures/m1_forest.pdf
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting_common import (MODEL_CUTOFF, ROOT, cluster_bootstrap_gap,
                                joined, load_panel, naive_gap)

ROWS = [
    ("gpt5",     "GPT-5",             1),
    ("gemini31", "Gemini-3.1-Pro",    1),
    ("kimi26",   "Kimi-K2.6",         2),   # tier-2 cutoff, footnoted
    ("gpt55",    "GPT-5.5",           1),
    ("minimax",  "MiniMax-M3",        1),
]
GUARD_DAYS = 30


def main():
    panel = load_panel()
    results = []
    for tag, name, tier in ROWS:
        cut = date.fromisoformat(MODEL_CUTOFF[tag])
        lo = cut + timedelta(days=GUARD_DAYS)
        rows = [r for r in joined(tag, panel) if r["d"] >= lo]
        if len(rows) < 60:
            print(f"{name}: too few clean questions ({len(rows)}), skipped")
            continue
        dates = sorted(r["d"] for r in rows)
        boundary = dates[len(dates) // 2]
        gap, n_pre, n_post = naive_gap(rows, boundary)
        ci = cluster_bootstrap_gap(rows, boundary, seed=1)
        star = ci[0] > 0 or ci[1] < 0
        results.append({
            "tag": tag, "name": name, "tier": tier,
            "cutoff": str(cut), "clean_from": str(lo),
            "boundary": str(boundary), "n_pre": n_pre, "n_post": n_post,
            "gap": gap, "ci": ci, "star": bool(star),
        })
        print(f"{name:16s} cutoff {cut}  n={n_pre}/{n_post}  "
              f"gap={gap:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"{'*' if star else ''}")

    # matched-window check: every model restricted to the SHORTEST clean
    # window (MiniMax's), same boundary -- shows the gap is a property of the
    # window, not the model
    short = max(results, key=lambda r: r["clean_from"])
    lo_m = date.fromisoformat(short["clean_from"])
    bd_m = date.fromisoformat(short["boundary"])
    matched = []
    print(f"\nmatched-window check (window from {lo_m}, split {bd_m}):")
    for tag, name, tier in ROWS:
        rows = [r for r in joined(tag, panel) if r["d"] >= lo_m]
        gap, n_pre, n_post = naive_gap(rows, bd_m)
        ci = cluster_bootstrap_gap(rows, bd_m, seed=1)
        star = ci[0] > 0 or ci[1] < 0
        matched.append({"tag": tag, "name": name, "n_pre": n_pre,
                        "n_post": n_post, "gap": gap, "ci": ci,
                        "star": bool(star)})
        print(f"  {name:16s} n={n_pre}/{n_post}  gap={gap:+.4f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}] {'*' if star else ''}")

    out = ROOT / "m1" / "results.json"
    out.write_text(json.dumps(
        {"per_model": results,
         "matched_window": {"clean_from": str(lo_m), "boundary": str(bd_m),
                            "rows": matched}}, indent=2))

    # forest plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(results) + 1.4))
    ys = np.arange(len(results))[::-1]
    for yy, r in zip(ys, results):
        ax.errorbar(r["gap"], yy,
                    xerr=[[r["gap"] - r["ci"][0]], [r["ci"][1] - r["gap"]]],
                    fmt="o", color="#1f5fa8", capsize=3, lw=1.8, ms=6)
        lab = f"{r['gap']:+.3f}" + ("*" if r["star"] else "")
        ax.annotate(lab, (r["ci"][1], yy), textcoords="offset points",
                    xytext=(8, -3), fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['name']}  (cutoff {r['cutoff'][:7]})"
                        for r in results])
    ax.set_xlabel("naive pre/post gap in crowd-anchored Brier reduction "
                  "(clean window)")
    ax.set_title("Provably-clean models 'fail' the naive contamination check")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "m1_forest.pdf")
    fig.savefig(ROOT / "figures" / "m1_forest.png", dpi=160)
    print("saved m1/results.json and figures/m1_forest.*")


if __name__ == "__main__":
    main()
