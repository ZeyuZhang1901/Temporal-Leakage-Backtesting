#!/usr/bin/env python3
"""App. E.7 and E.9: power floors under the reported inference, alternative
controls for the nulls, and a trend check of the control-selection rule.

E3. Semi-synthetic injection as in m5_analyze.power_injection (pull
    pre-boundary target probabilities toward the outcome), but
    (i) the pull w is calibrated per target so the realized mean increase of
        pre-boundary paired differences equals the nominal effect, and
    (ii) the paired series is first centred at zero boundary effect, so power
        does not depend on the observed estimate, and
    (iii) detection = lower bound of the source x month cluster-bootstrap 95%
        CI > 0, the inference used for every reported estimate.
    Replicates resample clusters. Also: analytic MDE80 = 2.8 x cluster SE for
    M5 targets and the M4-F own-cutoff cells.

E2b. M5 with Gemini-3.1-Pro (runner-up control); M4-F own-cutoff cells with
    each control alone; clean-region (>= 2026-02-24) trend gap of each
    candidate control against Qwen3.5.

Output: e9/power_controls.json + .txt
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting_common import ROOT, did_gap, joined, load_panel
import e9_trends as e1

OUT = ROOT / "e9"
B5 = date(2025, 7, 1)
LO, HI = date(2025, 1, 1), date(2026, 7, 1)
TARGETS5 = ["qwen35", "kimi26", "dsv32", "glm47", "minimax"]
EFFECTS = (0.03, 0.05, 0.07, 0.09, 0.11)
N_REP, N_BOOT = 300, 500
CUT = e1.CUT


def paired_rows(t_rows, c_rows):
    c = {r["id"]: r for r in c_rows}
    return [(t, c[t["id"]]) for t in t_rows
            if t["id"] in c and LO <= t["d"] < HI]


def injected_diff(pairs, w):
    out = []
    for t, c in pairs:
        L = t["L"]
        if t["d"] < B5 and w > 0:
            p = (1 - w) * t["p"] + w * t["y"]
            L = (t["fz"] - t["y"]) ** 2 - (p - t["y"]) ** 2
        out.append(L - c["L"])
    return np.array(out)


def calibrate_w(pairs, eff):
    pre = np.array([t["d"] < B5 for t, _ in pairs])
    base = injected_diff(pairs, 0.0)[pre].mean()
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        gain = injected_diff(pairs, mid)[pre].mean() - base
        lo, hi = (mid, hi) if gain < eff else (lo, mid)
    return (lo + hi) / 2


def cluster_arrays(pairs, vals):
    keys = sorted({t["cl"] for t, _ in pairs})
    idx = {k: i for i, k in enumerate(keys)}
    K = len(keys)
    sp, np_, sq, nq = (np.zeros(K) for _ in range(4))
    for (t, _), v in zip(pairs, vals):
        i = idx[t["cl"]]
        if t["d"] < B5:
            sp[i] += v; np_[i] += 1
        else:
            sq[i] += v; nq[i] += 1
    return sp, np_, sq, nq


def boot_ci(sp, np_, sq, nq, rng, n=N_BOOT):
    K = len(sp)
    W = rng.multinomial(K, np.full(K, 1 / K), size=n)
    a, b = W @ np_, W @ nq
    ok = (a > 0) & (b > 0)
    d = (W @ sp)[ok] / a[ok] - (W @ sq)[ok] / b[ok]
    return np.percentile(d, [2.5, 97.5]), d.std()


def power(pairs, eff, seed):
    rng = np.random.default_rng(seed)
    w = calibrate_w(pairs, eff) if eff > 0 else 0.0
    pre = np.array([t["d"] < B5 for t, _ in pairs])
    v0 = injected_diff(pairs, 0.0)
    gap0 = v0[pre].mean() - v0[~pre].mean()
    vals = injected_diff(pairs, w) - gap0 * pre   # null-centred, then injected
    sp, np_, sq, nq = cluster_arrays(pairs, vals)
    K = len(sp)
    det = 0
    for _ in range(N_REP):
        m = rng.multinomial(K, np.full(K, 1 / K))
        ci, _ = boot_ci(sp * m, np_ * m, sq * m, nq * m, rng)
        det += ci[0] > 0
    return det / N_REP, w


def main():
    OUT.mkdir(exist_ok=True)
    panel = load_panel()
    data = {t: joined(t, panel) for t in
            TARGETS5 + ["gpt5", "gemini31", "opus47"] + list(CUT)}
    res = {"power_m5": {}, "mde_m5": {}, "mde_m4f": {}, "m5_alt_control": {},
           "m4f_single_control": {}, "control_trend_gap": {}}
    L = []

    # E3: M5 power and MDE (control GPT-5)
    L.append("M5 power (cluster-bootstrap detection, calibrated injection)")
    for i, t in enumerate(TARGETS5):
        pairs = paired_rows(data[t], data["gpt5"])
        vals = injected_diff(pairs, 0.0)
        ci0, _ = boot_ci(*cluster_arrays(pairs, vals), np.random.default_rng(7), n=4000)
        se = (ci0[1] - ci0[0]) / 3.92
        res["mde_m5"][t] = {"cluster_se": float(se), "mde80": float(2.8 * se)}
        row = {}
        for j, eff in enumerate(EFFECTS):
            pw, w = power(pairs, eff, seed=500 + 10 * i + j)
            row[str(eff)] = {"power": pw, "w": w}
        res["power_m5"][t] = row
        L.append(f"  {t:8s} MDE80={2.8*se:.3f} | " + " ".join(
            f"{e}:{row[str(e)]['power']:.2f}" for e in EFFECTS))

    # E3: M4-F own-cutoff MDEs (pooled controls, +-90)
    ctrl = [{r["id"]: r["L"] for r in data[c]} for c in ("minimax", "opus47")]
    pool = {q: float(np.mean([m[q] for m in ctrl if q in m]))
            for q in set().union(*map(set, ctrl))}
    L.append("M4-F own-cutoff MDE80 (cluster SE, +-90d, pooled controls)")
    for k, (t, D) in enumerate(CUT.items()):
        items = e1.series_pair(data[t], pool)
        j = e1.jump_boot(items, D, D - timedelta(days=90), D + timedelta(days=90), 900 + k)
        lo_, hi_ = np.percentile(j["draws"], [2.5, 97.5])
        se = float((hi_ - lo_) / 3.92)
        res["mde_m4f"][t] = {"cluster_se": se, "mde80": 2.8 * se}
        L.append(f"  {t:8s} MDE80={2.8*se:.3f}")

    # E2b: M5 with Gemini-3.1-Pro control, incl. boundary sweep
    L.append("M5 DiD, control Gemini-3.1-Pro (vs paper's GPT-5)")
    for i, t in enumerate(TARGETS5):
        (g, npre, npost), dl = did_gap(data[t], data["gemini31"], B5, LO, HI)
        items = [{"d": x["d"], "v": x["L"], "cl": x["cl"]} for x in dl]
        jb = e1.jump_boot(items, B5, LO, HI, 1200 + i)
        ci = np.percentile(jb["draws"], [2.5, 97.5]).tolist()
        sweep = []
        for off in range(-180, 181, 30):
            (d, a, b), _ = did_gap(data[t], data["gemini31"],
                                   B5 + timedelta(days=off), LO, HI)
            sweep.append({"offset": off, "did": d, "n": [a, b]})
        res["m5_alt_control"][t] = {"did": g, "ci": ci, "n": [npre, npost],
                                    "star": bool(ci[0] > 0 or ci[1] < 0), "sweep": sweep}
        rng_ = [s["did"] for s in sweep if s["did"] is not None and s["n"][0] >= 62]
        L.append(f"  {t:8s} DiD={g:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
                 f"{'*' if ci[0] > 0 or ci[1] < 0 else ' '} | sweep (n_pre>=62) "
                 f"{min(rng_):+.3f}..{max(rng_):+.3f}")

    # E2b: M4-F own-cutoff cells with each control alone
    L.append("M4-F own-cutoff cells, single controls (+-90d)")
    for c in ("minimax", "opus47"):
        cmap = {r["id"]: r["L"] for r in data[c]}
        for k, (t, D) in enumerate(CUT.items()):
            items = e1.series_pair(data[t], cmap)
            j = e1.jump_boot(items, D, D - timedelta(days=90), D + timedelta(days=90),
                             1300 + k + (10 if c == "opus47" else 0))
            if j is None:
                L.append(f"  {c:8s} {t:8s} n/a"); continue
            ci = np.percentile(j["draws"], [2.5, 97.5]).tolist()
            res["m4f_single_control"][f"{c}:{t}"] = {"jump": j["est"], "ci": ci, "n": j["n"]}
            L.append(f"  {c:8s} {t:8s} {j['est']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
                     f"{'*' if ci[0] > 0 or ci[1] < 0 else ' '} n={j['n'][0]}/{j['n'][1]}")

    # E2b: clean-region trend gap of candidate controls vs Qwen3.5
    L.append("Trend gap vs Qwen3.5 on clean region (>= 2026-02-24), per month")
    for k, c in enumerate(("gpt5", "gemini31")):
        cmap = {r["id"]: r["L"] for r in data[c]}
        items = [i for i in e1.series_pair(data["qwen35"], cmap)
                 if date(2026, 2, 24) <= i["d"] < date(2026, 8, 1)]
        s = e1.slope(items, date(2026, 2, 24))
        ci = np.percentile(e1.slope_boot(items, date(2026, 2, 24), 1400 + k), [2.5, 97.5])
        res["control_trend_gap"][c] = {"slope": s, "ci": ci.tolist(), "n": len(items)}
        L.append(f"  {c:8s} s={s:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] n={len(items)}")

    (OUT / "power_controls.json").write_text(json.dumps(res, indent=1))
    (OUT / "power_controls.txt").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
