#!/usr/bin/env python3
"""M4-F: cutoff-specificity matrix on the forecasting panel.

For each target model and each ASSUMED boundary (the set of target cutoffs),
compute the boundary jump DiD'd against the clean-control pool:

  diff series  dL(q) = L_target(q) - mean_ctrl L(q)   (same questions)
  jump(D, h)   = mean dL[D-h, D) - mean dL[D, D+h)
                 (positive = target's relative edge disappears after D
                  = leakage signature at D)

A real leakage signal must peak at the model's OWN cutoff (diagonal) and be
absent elsewhere. Permutation test: shuffle question dates (one joint
shuffle for all models), recompute the mean diagonal-minus-offdiagonal
statistic, 10,000 draws.

Output: m4f/results.json + figures/m4f_matrix.txt (paper table source)
"""
import json
import zlib
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting_common import (MODEL_CUTOFF, ROOT, joined, load_panel)

TARGETS = ["dsv31", "kimi26", "gpt54", "gpt55"]
EXPLORATORY = ["gemini31"]
CONTROLS = ["minimax", "opus47"]
NAMES = {"dsv31": "DeepSeek-V3.1", "kimi26": "Kimi-K2.6", "gpt54": "GPT-5.4",
         "gpt55": "GPT-5.5", "gemini31": "Gemini-3.1-Pro",
         "minimax": "MiniMax-M3", "opus47": "Claude-Opus-4.7"}
BW = 90          # primary bandwidth (days); sensitivity in results
N_PERM = 10000


def diff_series(rows_t, ctrl_by_q):
    out = []
    for r in rows_t:
        c = ctrl_by_q.get(r["id"])
        if c is None:
            continue
        out.append({"id": r["id"], "d": r["d"], "dL": r["L"] - c,
                    "cl": r["cl"]})
    return out


def jump(series, D, h):
    lo, hi = D - timedelta(days=h), D + timedelta(days=h)
    pre = [s["dL"] for s in series if lo <= s["d"] < D]
    post = [s["dL"] for s in series if D <= s["d"] < hi]
    if len(pre) < 15 or len(post) < 15:
        return None, len(pre), len(post)
    return float(np.mean(pre) - np.mean(post)), len(pre), len(post)


def boot_ci(series, D, h, n=4000, seed=0):
    lo, hi = D - timedelta(days=h), D + timedelta(days=h)
    sel = [s for s in series if lo <= s["d"] < hi]
    clusters = {}
    for s in sel:
        clusters.setdefault(s["cl"], []).append(s)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pre, post = [], []
        for k in rng.choice(keys, size=len(keys), replace=True):
            for s in clusters[k]:
                (pre if s["d"] < D else post).append(s["dL"])
        if pre and post:
            vals.append(np.mean(pre) - np.mean(post))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    panel = load_panel()
    all_tags = TARGETS + EXPLORATORY + CONTROLS
    data = {t: joined(t, panel) for t in all_tags}
    for t, rows in data.items():
        print(f"{t}: {len(rows)} scored questions")

    # control pool per question
    ctrl_by_q = {}
    for q in {r["id"] for t in CONTROLS for r in data[t]}:
        vals = [r["L"] for t in CONTROLS for r in data[t] if r["id"] == q]
        if vals:
            ctrl_by_q[q] = float(np.mean(vals))
    # faster: rebuild via dicts
    ctrl_maps = [{r["id"]: r["L"] for r in data[t]} for t in CONTROLS]
    ctrl_by_q = {}
    for q in set().union(*[set(m) for m in ctrl_maps]):
        vals = [m[q] for m in ctrl_maps if q in m]
        ctrl_by_q[q] = float(np.mean(vals))

    boundaries = {t: date.fromisoformat(MODEL_CUTOFF[t]) for t in TARGETS}
    series = {t: diff_series(data[t], ctrl_by_q)
              for t in TARGETS + EXPLORATORY}

    matrix = {}
    for t in TARGETS + EXPLORATORY:
        row = {}
        for bt, D in boundaries.items():
            for h in (60, BW, 120):
                j, npre, npost = jump(series[t], D, h)
                cell = {"jump": j, "n_pre": npre, "n_post": npost}
                if j is not None and h == BW:
                    cell["ci"] = boot_ci(series[t], D, h,
                                         seed=zlib.crc32(f"{t}|{bt}".encode()))
                    cell["star"] = bool(cell["ci"][0] > 0 or cell["ci"][1] < 0)
                row.setdefault(bt, {})[f"h{h}"] = cell
        matrix[t] = row

    # placebo boundaries +-6 months around own cutoff (diagonal targets only)
    placebo = {}
    for t in TARGETS:
        D = boundaries[t]
        placebo[t] = {}
        for off in (-180, 180):
            Dp = D + timedelta(days=off)
            j, npre, npost = jump(series[t], Dp, BW)
            placebo[t][str(off)] = {"jump": j, "n_pre": npre, "n_post": npost}

    # permutation test on the diagonal-vs-offdiagonal contrast
    def diag_stat(series_map):
        diag, off = [], []
        for t in TARGETS:
            for bt, D in boundaries.items():
                j, _, _ = jump(series_map[t], D, BW)
                if j is None:
                    continue
                (diag if bt == t else off).append(j)
        if not diag or not off:
            return None
        return float(np.mean(diag) - np.mean(off))

    obs = diag_stat(series)
    rng = np.random.default_rng(11)
    qids = sorted({s["id"] for t in TARGETS for s in series[t]})
    date_of = {}
    for t in TARGETS:
        for s in series[t]:
            date_of[s["id"]] = s["d"]
    perm_ge = 0
    n_valid = 0
    dates_pool = [date_of[q] for q in qids]
    for _ in range(N_PERM):
        perm = rng.permutation(len(qids))
        newdate = {q: dates_pool[perm[i]] for i, q in enumerate(qids)}
        shuffled = {t: [{**s, "d": newdate[s["id"]]} for s in series[t]]
                    for t in TARGETS}
        st = diag_stat(shuffled)
        if st is None:
            continue
        n_valid += 1
        if st >= obs:
            perm_ge += 1
    p_perm = (perm_ge + 1) / (n_valid + 1) if n_valid else None

    out = {"bandwidth_days": BW, "targets": TARGETS,
           "exploratory": EXPLORATORY, "controls": CONTROLS,
           "matrix": {t: {bt: v for bt, v in row.items()}
                      for t, row in matrix.items()},
           "placebo": placebo,
           "diag_stat": obs, "perm_p": p_perm, "n_perm_valid": n_valid}

    def default(o):
        if isinstance(o, (date,)):
            return str(o)
        raise TypeError

    (ROOT / "m4f").mkdir(exist_ok=True)
    (ROOT / "m4f" / "results.json").write_text(
        json.dumps(out, indent=1, default=default))

    # text table
    lines = ["M4-F specificity matrix (jump at assumed boundary, DiD vs "
             f"control pool, h={BW}d). * = 95% CI excludes 0.",
             "model (own cutoff)      " + "".join(
                 f"@{MODEL_CUTOFF[bt][:7]}   " for bt in TARGETS)]
    for t in TARGETS + EXPLORATORY:
        cells = []
        for bt in TARGETS:
            c = matrix[t][bt][f"h{BW}"]
            if c["jump"] is None:
                cells.append("   n/a    ")
                continue
            s = f"{c['jump']:+.3f}" + ("*" if c.get("star") else " ")
            if MODEL_CUTOFF.get(t) == MODEL_CUTOFF.get(bt):
                s = f"[{s}]"
            cells.append(f"{s:>10s}")
        lines.append(f"{NAMES[t]:22s}  " + " ".join(cells))
    lines.append(f"diag-vs-offdiag stat = "
                 f"{obs if obs is None else round(obs, 4)}, "
                 f"permutation p = {p_perm}")
    txt = "\n".join(lines)
    (ROOT / "figures" / "m4f_matrix.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
