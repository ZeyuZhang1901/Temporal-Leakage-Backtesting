#!/usr/bin/env python3
"""App. E.9 (Table G1): sensitivity of the GPT-5.5 boundary result.

Grid over (i) window half-width, (ii) comparison anchor / control, and
(iii) assumed boundary date, for two designs:

  A. within-family archived-forecast arm (as in m4f_anchor.py):
     dA(q) = L_retro_GPT55(q) - L_realtime_anchor(q), jump at boundary.
  B. cross-model arm (as in m4f_analyze.py): dL(q) = L_GPT55(q) - L_ctrl(q).

Reuses the original loaders and estimators unchanged; only the seeds are
fixed integers so every CI is reproducible.

Output: e9/gpt55_sensitivity.json + .txt
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4f_anchor as anc
import m4f_analyze as mx
from forecasting_common import ROOT, joined, load_panel

OUT = ROOT / "e9"
CUTOFF = date(2025, 12, 1)
ANCHORS = {"pooled": ["OpenAI.gpt-5-mini-2025-08-07",
                      "OpenAI.gpt-5.1-2025-11-13"],
           "gpt5mini": ["OpenAI.gpt-5-mini-2025-08-07"],
           "gpt51": ["OpenAI.gpt-5.1-2025-11-13"]}
VARIANTS = {"unbanded": {}, "banded_h10_75": dict(h_lo=10, h_hi=75, h_target=40)}
WINDOWS = (60, 90, 120)
SHIFTS = (-60, -30, 0, 30, 60)
PLACEBOS = (date(2026, 3, 1), date(2026, 4, 1))
N_BOOT = 10000


def anchor_items(panel, retro, rt_maps, D, h, kw):
    items = []
    for row in panel:
        r = retro.get(row["id"])
        if r is None or not (D - timedelta(days=h) <= row["_d"] < D + timedelta(days=h)):
            continue
        a = [got[0] for m in rt_maps.values()
             if (got := anc.realtime_for(row, m, **kw)) is not None]
        if not a:
            continue
        y, fz = row["outcome"], row["freeze_earliest"]
        L_rt = (fz - y) ** 2 - (float(np.mean(a)) - y) ** 2
        items.append((row["_d"], r["L"] - L_rt, r["cl"]))
    return items


def main():
    OUT.mkdir(exist_ok=True)
    panel = load_panel()
    retro = {r["id"]: r for r in joined("gpt55", panel)}
    rt_all = {rel: anc.load_realtime(rel) for rel in ANCHORS["pooled"]}
    res = {"within_family": [], "cross_model": []}
    lines = ["GPT-5.5 sensitivity. jump = mean pre - mean post; * = 95% CI excludes 0"]

    seed = 100
    boundaries = [("cutoff%+d" % s, CUTOFF + timedelta(days=s)) for s in SHIFTS]
    boundaries += [("placebo_" + str(p), p) for p in PLACEBOS]
    for aname, rels in ANCHORS.items():
        maps = {r: rt_all[r] for r in rels}
        for vname, kw in VARIANTS.items():
            for bname, D in boundaries:
                for h in WINDOWS:
                    seed += 1
                    items = anchor_items(panel, retro, maps, D, h, kw)
                    out = anc.boot_jump(items, D, n=N_BOOT, seed=seed)
                    rec = {"anchor": aname, "variant": vname, "boundary": bname,
                           "date": str(D), "h": h, "result": out}
                    res["within_family"].append(rec)
                    if out:
                        lines.append(f"WF {aname:8s} {vname:13s} {bname:18s} h={h:3d} "
                                     f"{out['jump']:+.3f} [{out['ci'][0]:+.3f},{out['ci'][1]:+.3f}]"
                                     f"{'*' if out['star'] else ' '} n={out['n_pre']}/{out['n_post']}")
                    else:
                        lines.append(f"WF {aname:8s} {vname:13s} {bname:18s} h={h:3d} n/a (n<10)")

    target = joined("gpt55", panel)
    ctrl_rows = {c: {r["id"]: r["L"] for r in joined(c, panel)} for c in ("minimax", "opus47")}
    pools = {"pool": None, "minimax": ["minimax"], "opus47": ["opus47"]}
    for pname, members in pools.items():
        mem = members or list(ctrl_rows)
        cmap = {}
        for q in set().union(*[set(ctrl_rows[m]) for m in mem]):
            v = [ctrl_rows[m][q] for m in mem if q in ctrl_rows[m]]
            cmap[q] = float(np.mean(v))
        series = mx.diff_series(target, cmap)
        for bname, D in boundaries:
            for h in WINDOWS:
                seed += 1
                j, npre, npost = mx.jump(series, D, h)
                rec = {"control": pname, "boundary": bname, "date": str(D), "h": h,
                       "jump": j, "n_pre": npre, "n_post": npost}
                if j is not None:
                    lo, hi = mx.boot_ci(series, D, h, n=4000, seed=seed)
                    rec.update(ci=[lo, hi], star=bool(lo > 0 or hi < 0))
                    lines.append(f"CM {pname:8s} {bname:18s} h={h:3d} {j:+.3f} "
                                 f"[{lo:+.3f},{hi:+.3f}]{'*' if rec['star'] else ' '} n={npre}/{npost}")
                else:
                    lines.append(f"CM {pname:8s} {bname:18s} h={h:3d} n/a n={npre}/{npost}")
                res["cross_model"].append(rec)

    (OUT / "gpt55_sensitivity.json").write_text(json.dumps(res, indent=1))
    (OUT / "gpt55_sensitivity.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
