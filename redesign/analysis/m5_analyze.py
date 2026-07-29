#!/usr/bin/env python3
"""M5: the disciplined null on the forecasting panel.

- Boundary 2025-07-01 (inside the plausible training window of the target).
- Control selection between {gemini31, gpt5} by the pre-registered
  profile-matching protocol: minimize post-boundary-clean profile distance
  (calibration-curve L2 + source-stratified mean-L L2) to the target.
- Paired DiD per target vs the selected control; weak-control arm (gpt35t);
  boundary sweep; semi-synthetic power injection.

Output: m5/results.json + figures/m5_table.txt
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting_common import (REDESIGN, cluster_bootstrap_gap, did_gap,
                                cluster_bootstrap_did, joined, load_panel,
                                naive_gap)

BOUNDARY = date(2025, 7, 1)
LO, HI = date(2025, 1, 1), date(2026, 7, 1)
PRIMARY_TARGET = "qwen35"
CONTROL_CANDIDATES = ["gemini31", "gpt5"]
SECONDARY = ["kimi26", "dsv32", "glm47", "minimax"]
WEAK_CONTROL = "gpt35t"
NAMES = {"qwen35": "Qwen3.5-35B-A3B", "kimi26": "Kimi-K2.6",
         "dsv32": "DeepSeek-V3.2", "glm47": "GLM-4.7",
         "minimax": "MiniMax-M3", "gemini31": "Gemini-3.1-Pro",
         "gpt5": "GPT-5", "gpt35t": "GPT-3.5-Turbo"}


def profile_distance(rows_t, rows_c):
    """Post-boundary profile match: calibration L2 + per-source mean-L L2."""
    ct = {r["id"]: r for r in rows_c}
    pairs = [(t, ct[t["id"]]) for t in rows_t
             if t["id"] in ct and t["d"] >= BOUNDARY and t["d"] < HI]
    if len(pairs) < 50:
        return None
    # calibration: bin target/control probs, compare freq(y)
    bins = np.linspace(0, 1, 6)
    cal = 0.0
    for lohi in zip(bins[:-1], bins[1:]):
        for which in (0, 1):
            sel = [p for p in pairs if lohi[0] <= p[which]["p"] < lohi[1]]
            if len(sel) < 5:
                continue
            ft = np.mean([p[0]["y"] for p in sel])
            fc = np.mean([p[1]["y"] for p in sel])
            cal += (ft - fc) ** 2
    srcs = sorted({t["src"] for t, _ in pairs})
    strat = 0.0
    for s in srcs:
        sel = [p for p in pairs if p[0]["src"] == s]
        if len(sel) < 10:
            continue
        strat += (np.mean([p[0]["L"] for p in sel])
                  - np.mean([p[1]["L"] for p in sel])) ** 2
    return float(cal + strat)


def power_injection(rows_t, dl_ctrl_map, effects=(0.05, 0.09),
                    n_rep=300, seed=5):
    """Semi-synthetic: pull pre-boundary probs toward outcome to create a
    known naive-gap increase; measure adjusted-DiD detection rate."""
    rng = np.random.default_rng(seed)
    out = {}
    base = [dict(r) for r in rows_t if LO <= r["d"] < HI]
    for eff in effects:
        det = 0
        for _ in range(n_rep):
            sample_idx = rng.integers(0, len(base), len(base))
            rows = [dict(base[i]) for i in sample_idx]
            for r in rows:
                if r["d"] < BOUNDARY:
                    w = min(1.0, eff / 0.09 * 0.35)
                    p_new = (1 - w) * r["p"] + w * r["y"]
                    r["L"] = (r["fz"] - r["y"])**2 - (p_new - r["y"])**2
                c = dl_ctrl_map.get(r["id"])
                r["Ld"] = r["L"] - c if c is not None else None
            dl = [{"d": r["d"], "L": r["Ld"], "cl": r["cl"], "id": r["id"]}
                  for r in rows if r["Ld"] is not None]
            g, npre, npost = naive_gap(dl, BOUNDARY, LO, HI)
            if g is None:
                continue
            # normal-approx significance via cluster-free bootstrap-lite
            pre = [x["L"] for x in dl if x["d"] < BOUNDARY]
            post = [x["L"] for x in dl if x["d"] >= BOUNDARY]
            se = np.sqrt(np.var(pre)/len(pre) + np.var(post)/len(post))
            if g - 1.96 * se > 0:
                det += 1
        out[str(eff)] = det / n_rep
    return out


def main():
    panel = load_panel()
    t_rows = joined(PRIMARY_TARGET, panel)

    # control selection
    sel_table = {}
    for c in CONTROL_CANDIDATES:
        sel_table[c] = profile_distance(t_rows, joined(c, panel))
    valid = {c: v for c, v in sel_table.items() if v is not None}
    control = min(valid, key=valid.get)
    print("control selection:", sel_table, "->", control)
    c_rows = joined(control, panel)
    c_map = {r["id"]: r["L"] for r in c_rows}

    results = {"boundary": str(BOUNDARY), "window": [str(LO), str(HI)],
               "control_selection": sel_table, "control": control,
               "rows": []}

    for tag in [PRIMARY_TARGET] + SECONDARY:
        rows = joined(tag, panel)
        gap, n_pre, n_post = naive_gap(rows, BOUNDARY, LO, HI)
        gci = cluster_bootstrap_gap(rows, BOUNDARY, LO, HI, seed=2)
        (did, dn_pre, dn_post), dl = did_gap(rows, c_rows, BOUNDARY, LO, HI)
        dci = cluster_bootstrap_did(dl, BOUNDARY, LO, HI, seed=3)
        upper = dci[1]
        results["rows"].append({
            "tag": tag, "name": NAMES[tag], "n": [n_pre, n_post],
            "naive_gap": gap, "naive_ci": gci,
            "naive_star": bool(gci[0] > 0 or gci[1] < 0),
            "did": did, "did_ci": dci,
            "did_star": bool(dci[0] > 0 or dci[1] < 0),
            "one_sided_upper": upper,
        })
        r = results["rows"][-1]
        print(f"{NAMES[tag]:18s} naive {gap:+.4f} [{gci[0]:+.4f},{gci[1]:+.4f}]"
              f"{'*' if r['naive_star'] else ' '}  "
              f"DiD {did:+.4f} [{dci[0]:+.4f},{dci[1]:+.4f}]"
              f"{'*' if r['did_star'] else ' '}")

    # weak-control arm
    w_rows = joined(WEAK_CONTROL, panel)
    (wdid, _, _), wdl = did_gap(t_rows, w_rows, BOUNDARY, LO, HI)
    wci = cluster_bootstrap_did(wdl, BOUNDARY, LO, HI, seed=4)
    results["weak_control"] = {"tag": WEAK_CONTROL, "did": wdid, "ci": wci,
                               "star": bool(wci[0] > 0 or wci[1] < 0)}
    print(f"weak-control arm  DiD {wdid:+.4f} [{wci[0]:+.4f},{wci[1]:+.4f}]")

    # boundary sweep (target, adjusted DiD at shifted boundaries)
    sweep = []
    for off in range(-180, 181, 30):
        B = BOUNDARY + timedelta(days=off)
        (d, npre, npost), dl = did_gap(t_rows, c_rows, B, LO, HI)
        sweep.append({"offset": off, "did": d, "n": [npre, npost]})
    results["boundary_sweep"] = sweep

    # power
    results["power"] = power_injection(t_rows, c_map)
    print("power:", results["power"])

    (REDESIGN / "m5").mkdir(exist_ok=True)
    (REDESIGN / "m5" / "results.json").write_text(
        json.dumps(results, indent=1))
    print("saved m5/results.json")


if __name__ == "__main__":
    main()
