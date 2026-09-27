#!/usr/bin/env python3
"""App. E.9 trend test: parallel-recency diagnostic and bias bound.

For a difference-of-means boundary estimator J = mean(pre) - mean(post) of a
paired target-minus-reference series, a linear honest trend gap s (per month)
contributes -s * dt, where dt = mean post date - mean pre date (months).
Hence B = J + s * dt.

1. Estimate s where target and reference are both clean by construction
   (OLS slope of the paired series on resolution date; source x month
   cluster bootstrap).
2. Bias-adjusted estimate J + s*dt with a CI from independent bootstrap draws.
3. Breakdown: the slope gap (and its multiple of |s_hat|) that would
   (a) remove a detection (adjusted lower CI = 0) or
   (b) hide a 0.05 effect (bias of -0.05).
4. Clean-region placebo jumps with the same estimator.

Output: e9/trends.json + e9/trends.txt
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4f_anchor as anc
from forecasting_common import ROOT, joined, load_panel

OUT = ROOT / "e9"
MONTH = 30.44
N_BOOT = 4000
EFFECT = 0.05
CUT = {"dsv31": date(2025, 3, 31), "kimi26": date(2025, 4, 30),
       "gpt54": date(2025, 8, 31), "gpt55": date(2025, 12, 1)}


def series_pair(t_rows, ref):
    return [{"d": r["d"], "v": r["L"] - ref[r["id"]], "cl": r["cl"]}
            for r in t_rows if r["id"] in ref]


def clusters_of(items):
    cl = {}
    for it in items:
        cl.setdefault(it["cl"], []).append(it)
    return cl


def slope(items, origin):
    x = np.array([(it["d"] - origin).days / MONTH for it in items])
    y = np.array([it["v"] for it in items])
    if len(x) < 30 or np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def slope_boot(items, origin, seed):
    cl = clusters_of(items)
    keys = sorted(cl)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOT):
        smp = [it for k in rng.choice(keys, len(keys)) for it in cl[k]]
        s = slope(smp, origin)
        if s is not None:
            draws.append(s)
    return np.array(draws)


def jump_items(items, D, lo, hi):
    sel = [it for it in items if lo <= it["d"] < hi]
    pre = [it for it in sel if it["d"] < D]
    post = [it for it in sel if it["d"] >= D]
    return sel, pre, post


def jump_boot(items, D, lo, hi, seed):
    sel, pre, post = jump_items(items, D, lo, hi)
    if len(pre) < 10 or len(post) < 10:
        return None
    est = np.mean([i["v"] for i in pre]) - np.mean([i["v"] for i in post])
    dpre = np.mean([(i["d"] - D).days for i in pre]) / MONTH
    dpost = np.mean([(i["d"] - D).days for i in post]) / MONTH
    cl = clusters_of(sel)
    keys = sorted(cl)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOT):
        p, q = [], []
        for k in rng.choice(keys, len(keys)):
            for i in cl[k]:
                (p if i["d"] < D else q).append(i["v"])
        if p and q:
            draws.append(np.mean(p) - np.mean(q))
    return {"est": float(est), "draws": np.array(draws), "dt": float(dpost - dpre),
            "n": [len(pre), len(post)]}


def analyse(name, items_all, D, lo, hi, clean_lo, clean_hi, detection, seed):
    j = jump_boot(items_all, D, lo, hi, seed)
    clean = [i for i in items_all if clean_lo <= i["d"] < clean_hi]
    s_hat = slope(clean, clean_lo)
    s_d = slope_boot(clean, clean_lo, seed + 1)
    s_ci = np.percentile(s_d, [2.5, 97.5])
    m = min(len(j["draws"]), len(s_d))
    adj = j["draws"][:m] + s_d[:m] * j["dt"]
    j_ci = np.percentile(j["draws"], [2.5, 97.5])
    adj_ci = np.percentile(adj, [2.5, 97.5])
    s_abs_max = float(max(abs(s_ci[0]), abs(s_ci[1])))
    if detection:
        s_star = float(j_ci[0] / j["dt"])           # slope that removes lower CI
    else:
        s_star = float(EFFECT / j["dt"])            # slope that hides 0.05
    rec = {"design": name, "boundary": str(D), "window": [str(lo), str(hi)],
           "n_pre_post": j["n"], "estimate": j["est"], "ci": j_ci.tolist(),
           "dt_months": j["dt"], "clean_region": [str(clean_lo), str(clean_hi)],
           "n_clean": len(clean), "slope_per_month": s_hat, "slope_ci": s_ci.tolist(),
           "adjusted": float(j["est"] + s_hat * j["dt"]), "adjusted_ci": adj_ci.tolist(),
           "breakdown_slope": s_star,
           "breakdown_multiple_of_slope": abs(s_star) / abs(s_hat) if s_hat else None,
           "breakdown_multiple_of_slope_ci_bound": abs(s_star) / s_abs_max,
           "kind": "detection" if detection else "null"}
    return rec


def placebo(name, items_all, D, h, seed):
    j = jump_boot(items_all, D, D - timedelta(days=h), D + timedelta(days=h), seed)
    if j is None:
        return None
    ci = np.percentile(j["draws"], [2.5, 97.5])
    return {"design": name, "placebo": str(D), "h": h, "estimate": j["est"],
            "ci": ci.tolist(), "star": bool(ci[0] > 0 or ci[1] < 0), "n": j["n"]}


def main():
    OUT.mkdir(exist_ok=True)
    panel = load_panel()
    res, plac = [], []
    seed = 1000

    # 1. M4-F own-cutoff cells: target vs control pool, +-90 days
    ctrl = [{r["id"]: r["L"] for r in joined(c, panel)} for c in ("minimax", "opus47")]
    pool = {q: float(np.mean([m[q] for m in ctrl if q in m]))
            for q in set().union(*map(set, ctrl))}
    for t, D in CUT.items():
        items = series_pair(joined(t, panel), pool)
        seed += 10
        res.append(analyse(f"M4-F {t} vs control pool", items, D,
                           D - timedelta(days=90), D + timedelta(days=90),
                           date(2026, 2, 1), date(2026, 8, 1), False, seed))
        for P in (date(2026, 4, 1), date(2026, 5, 1)):
            seed += 1
            plac.append(placebo(f"M4-F {t}", items, P, 60, seed))

    # 2. GPT-5.5 archived-forecast arm (pooled anchors), +-90 days
    retro = {r["id"]: r for r in joined("gpt55", panel)}
    maps = {rel: anc.load_realtime(rel) for rel in
            ["OpenAI.gpt-5-mini-2025-08-07", "OpenAI.gpt-5.1-2025-11-13"]}
    for vname, kw in (("unbanded", {}),
                      ("banded_h10_75", dict(h_lo=10, h_hi=75, h_target=40))):
        items = []
        for row in panel:
            r = retro.get(row["id"])
            if r is None:
                continue
            a = [g[0] for m in maps.values()
                 if (g := anc.realtime_for(row, m, **kw)) is not None]
            if not a:
                continue
            y, fz = row["outcome"], row["freeze_earliest"]
            L_rt = (fz - y) ** 2 - (float(np.mean(a)) - y) ** 2
            items.append({"d": row["_d"], "v": r["L"] - L_rt, "cl": r["cl"]})
        D = CUT["gpt55"]
        seed += 10
        res.append(analyse(f"GPT-5.5 archived-forecast arm ({vname})", items, D,
                           D - timedelta(days=90), D + timedelta(days=90),
                           D, date(2026, 8, 1), True, seed))

    # 3. M5: targets vs GPT-5, asymmetric window around 2025-07-01
    g5 = {r["id"]: r["L"] for r in joined("gpt5", panel)}
    for t in ("qwen35", "kimi26", "dsv32", "glm47", "minimax"):
        items = series_pair(joined(t, panel), g5)
        seed += 10
        res.append(analyse(f"M5 {t} vs GPT-5", items, date(2025, 7, 1),
                           date(2025, 1, 1), date(2026, 7, 1),
                           date(2026, 2, 24), date(2026, 8, 1), False, seed))
        seed += 1
        plac.append(placebo(f"M5 {t}", items, date(2026, 5, 1), 60, seed))

    lines = ["Trend test: B = J + s*dt. s = clean-region slope of paired series (per month)."]
    for r in res:
        lines.append(
            f"{r['design']:44s} J={r['estimate']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
            f"n={r['n_pre_post'][0]}/{r['n_pre_post'][1]} dt={r['dt_months']:.2f}mo | "
            f"s={r['slope_per_month']:+.4f} [{r['slope_ci'][0]:+.4f},{r['slope_ci'][1]:+.4f}] "
            f"(n={r['n_clean']}) | adj={r['adjusted']:+.3f} "
            f"[{r['adjusted_ci'][0]:+.3f},{r['adjusted_ci'][1]:+.3f}] | "
            f"breakdown s*={r['breakdown_slope']:+.4f}/mo = "
            f"{r['breakdown_multiple_of_slope']:.1f}x |s_hat|, "
            f"{r['breakdown_multiple_of_slope_ci_bound']:.1f}x CI bound ({r['kind']})")
    lines.append("Clean-region placebos (h=60):")
    for p in plac:
        if p:
            lines.append(f"  {p['design']:22s} @{p['placebo']} {p['estimate']:+.3f} "
                         f"[{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}]{'*' if p['star'] else ' '} "
                         f"n={p['n'][0]}/{p['n'][1]}")
    (OUT / "trends.json").write_text(json.dumps({"designs": res, "placebos": plac},
                                                   indent=1))
    (OUT / "trends.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
