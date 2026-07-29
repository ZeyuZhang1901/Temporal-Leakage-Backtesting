#!/usr/bin/env python3
"""M4-F clean-anchor arm: archived REAL-TIME ForecastBench forecasts vs our
retrospective queries.

Two designs:

1. Leakage test (family anchor). Target model T with cutoff D; relative R
   from the same family submitted real-time forecasts to ForecastBench in
   rounds before D. On each panel question q with a real-time forecast from R
   made before q's resolution, define
       dA(q) = L_retro_T(q) - L_realtime_R(q)
   (both crowd-anchored Brier reductions). A jump of dA at D
   (pre-cutoff-resolved minus post-cutoff-resolved) is a leakage signature in
   T's retrospective answers that no cross-model control can fake: the
   real-time anchor is clean BY CONSTRUCTION (forecast predates resolution).

2. Protocol check (within-model). Models that submitted real-time forecasts
   themselves (GPT-5, DeepSeek-V3.1): compare our retrospective probability
   with their own archived real-time forecast on post-cutoff questions.
   Agreement validates the elicitation protocol (frozen weights => the only
   difference is protocol).

Output: m4f/anchor_results.json + printout.
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting_common import MODEL_CUTOFF, REDESIGN, joined, load_panel

FSETS = Path("/tmp/fb_check/fsets/forecastbench-processed-forecast-sets")

# leakage arms: target tag -> list of real-time relative model files
FAMILY_ANCHORS = {
    "gpt55": ["OpenAI.gpt-5-mini-2025-08-07", "OpenAI.gpt-5.1-2025-11-13"],
    "gpt54": ["OpenAI.gpt-4.1-2025-04-14", "OpenAI.o4-mini-2025-04-16"],
}
# protocol arms: our tag -> its own real-time file prefix
SELF_ANCHORS = {
    "gpt5": "OpenAI.gpt-5-2025-08-07",
    "dsv31": "DeepSeek.DeepSeek-V3.1",
}
H = 120           # days around cutoff for the leakage jump
N_BOOT = 10000


def load_realtime(model_name):
    """(source, qid) -> list of (due_date, forecast); zero_shot files only."""
    out = {}
    for rdir in sorted(FSETS.iterdir()):
        f = rdir / f"{rdir.name}.{model_name}_zero_shot.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        due = date.fromisoformat(d["forecast_due_date"][:10])
        for fc in d["forecasts"]:
            if fc.get("imputed"):
                continue
            if fc.get("forecast") is None:
                continue
            if not isinstance(fc["id"], str):   # combo questions
                continue
            out.setdefault((fc["source"], fc["id"]), []).append(
                (due, float(fc["forecast"])))
    return out


def realtime_for(panel_row, rt_index, h_lo=None, h_hi=None, h_target=None):
    """Real-time forecast strictly before resolution.

    Default: latest one. With a horizon band (h_lo, h_hi in days): only
    forecasts whose horizon (resolution - due) falls in the band, choosing
    the one closest to h_target. Returns (prob, horizon_days) or None."""
    key = (panel_row["source"], panel_row["id"])
    if key not in rt_index:
        return None
    res_d = panel_row["_d"]
    cands = []
    for due, p in rt_index[key]:
        h = (res_d - due).days
        if h <= 0:
            continue
        if h_lo is not None and not (h_lo <= h <= h_hi):
            continue
        cands.append((h, p))
    if not cands:
        return None
    if h_target is not None:
        h, p = min(cands, key=lambda x: abs(x[0] - h_target))
    else:
        h, p = min(cands)          # smallest horizon = latest forecast
    return p, h


def boot_jump(items, boundary, n=N_BOOT, seed=0):
    """items: list of (date, value, cluster). Jump = mean pre - mean post,
    cluster bootstrap CI."""
    clusters = {}
    for d, v, cl in items:
        clusters.setdefault(cl, []).append((d, v))
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    pre = [v for d, v, _ in items if d < boundary]
    post = [v for d, v, _ in items if d >= boundary]
    if len(pre) < 10 or len(post) < 10:
        return None
    obs = float(np.mean(pre) - np.mean(post))
    draws = []
    for _ in range(n):
        p, q = [], []
        for k in rng.choice(keys, size=len(keys), replace=True):
            for d, v in clusters[k]:
                (p if d < boundary else q).append(v)
        if p and q:
            draws.append(np.mean(p) - np.mean(q))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"jump": obs, "ci": [float(lo), float(hi)],
            "star": bool(lo > 0 or hi < 0),
            "n_pre": len(pre), "n_post": len(post)}


def main():
    panel = load_panel()
    out = {"leakage_arms": {}, "protocol_arms": {}}

    for tag, relatives in FAMILY_ANCHORS.items():
        D = date.fromisoformat(MODEL_CUTOFF[tag])
        retro = {r["id"]: r for r in joined(tag, panel)}
        rt_maps = {rel: load_realtime(rel) for rel in relatives}
        arm = {"cutoff": str(D), "relatives": relatives, "variants": {}}
        # primary: horizon band 10-75d (rules out anchor-recency artifact);
        # secondary: latest-before-resolution (max data)
        for vname, kw in (("banded_h10_75", dict(h_lo=10, h_hi=75, h_target=40)),
                          ("latest", {})):
            items, hor_pre, hor_post = [], [], []
            used = {rel: 0 for rel in relatives}
            for row in panel:
                r = retro.get(row["id"])
                if r is None:
                    continue
                if not (D - timedelta(days=H) <= row["_d"]
                        < D + timedelta(days=H)):
                    continue
                anchors, hs = [], []
                for rel, m in rt_maps.items():
                    got = realtime_for(row, m, **kw)
                    if got is not None:
                        anchors.append(got[0])
                        hs.append(got[1])
                        used[rel] += 1
                if not anchors:
                    continue
                y = row["outcome"]
                fz = row["freeze_earliest"]
                L_rt = (fz - y) ** 2 - (float(np.mean(anchors)) - y) ** 2
                items.append((row["_d"], r["L"] - L_rt, r["cl"]))
                (hor_pre if row["_d"] < D else hor_post).append(
                    float(np.mean(hs)))
            res = boot_jump(items, D, seed=hash((tag, vname)) % 2**31)
            arm["variants"][vname] = {
                "anchored_questions": len(items), "per_relative": used,
                "mean_horizon_pre": float(np.mean(hor_pre)) if hor_pre else None,
                "mean_horizon_post": float(np.mean(hor_post)) if hor_post else None,
                "result": res}
            print(f"[leakage] {tag} {vname}: {len(items)} qs, horizon "
                  f"pre/post {arm['variants'][vname]['mean_horizon_pre']}/"
                  f"{arm['variants'][vname]['mean_horizon_post']} -> {res}")
        out["leakage_arms"][tag] = arm

    for tag, own in SELF_ANCHORS.items():
        retro = {r["id"]: r for r in joined(tag, panel)}
        rt = load_realtime(own)
        diffs, briers = [], []
        for row in panel:
            r = retro.get(row["id"])
            if r is None:
                continue
            got = realtime_for(row, rt)
            if got is None:
                continue
            p_rt = got[0]
            y = row["outcome"]
            diffs.append(r["p"] - p_rt)
            briers.append((r["p"] - y) ** 2 - (p_rt - y) ** 2)
        rng = np.random.default_rng(3)
        def ci(x):
            x = np.asarray(x)
            ms = [np.mean(rng.choice(x, len(x))) for _ in range(4000)]
            return [float(np.mean(x)), float(np.percentile(ms, 2.5)),
                    float(np.percentile(ms, 97.5))]
        out["protocol_arms"][tag] = {
            "own_file": own, "n": len(diffs),
            "prob_shift": ci(diffs) if diffs else None,
            "brier_diff_retro_minus_rt": ci(briers) if briers else None}
        print(f"[protocol] {tag}: n={len(diffs)} prob shift "
              f"{out['protocol_arms'][tag]['prob_shift']}")

    (REDESIGN / "m4f").mkdir(exist_ok=True)
    (REDESIGN / "m4f" / "anchor_results.json").write_text(
        json.dumps(out, indent=1))
    print("saved m4f/anchor_results.json")


if __name__ == "__main__":
    main()
