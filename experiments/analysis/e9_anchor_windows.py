#!/usr/bin/env python3
"""App. D.5: archived-forecast statistics at the ±90- and ±120-day windows.

The submitted anchor arm (m4f_anchor.py) used H = 120 days for every
statistic. The revision uses ±90 days, the window stated in Table 2. This
script recomputes, at both windows:

  1. GPT-5.4 archived-forecast jump at its cutoff (unbanded and banded).
  2. DeepSeek-V3.1 protocol placebo-jump: its retrospective score minus its
     own archived real-time score, tested at monthly boundaries
     (DEVIATIONS.md item 7). The ±120 run must reproduce the submitted
     +0.026 [-0.011, +0.064], n = 42/419 at 1 Dec 2025.

Output: e9/anchor_windows.json + .txt
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
WINDOWS = (90, 120)
VARIANTS = {"unbanded": {}, "banded_h10_75": dict(h_lo=10, h_hi=75, h_target=40)}
N_BOOT = 10000


def diff_items(panel, retro, maps, kw):
    """(date, retro L minus real-time L, cluster) for every covered question."""
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
        items.append((row["_d"], r["L"] - L_rt, r["cl"]))
    return items


def windowed(items, D, h):
    return [it for it in items
            if D - timedelta(days=h) <= it[0] < D + timedelta(days=h)]


def main():
    OUT.mkdir(exist_ok=True)
    panel = load_panel()
    res = {"gpt54": [], "dsv31_placebo": []}
    lines = ["Archived-forecast statistics at ±90 and ±120 days"]
    seed = 4000

    # 1. GPT-5.4 at its documented cutoff
    retro = {r["id"]: r for r in joined("gpt54", panel)}
    maps = {rel: anc.load_realtime(rel) for rel in anc.FAMILY_ANCHORS["gpt54"]}
    D = date(2025, 8, 31)
    for vname, kw in VARIANTS.items():
        items = diff_items(panel, retro, maps, kw)
        for h in WINDOWS:
            seed += 1
            out = anc.boot_jump(windowed(items, D, h), D, n=N_BOOT, seed=seed)
            res["gpt54"].append({"variant": vname, "h": h, "result": out})
            lines.append(f"GPT-5.4 {vname:13s} h={h:3d} {out['jump']:+.3f} "
                         f"[{out['ci'][0]:+.3f},{out['ci'][1]:+.3f}]"
                         f"{'*' if out['star'] else ' '} n={out['n_pre']}/{out['n_post']}")

    # 2. DeepSeek-V3.1 protocol placebo-jump at monthly boundaries
    retro = {r["id"]: r for r in joined("dsv31", panel)}
    maps = {"own": anc.load_realtime(anc.SELF_ANCHORS["dsv31"])}
    items = diff_items(panel, retro, maps, {})
    boundaries = [date(y, m, 1) for y, m in
                  [(2025, 6), (2025, 7), (2025, 8), (2025, 9), (2025, 10), (2025, 11),
                   (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5)]]
    for h in WINDOWS:
        for B in boundaries:
            seed += 1
            out = anc.boot_jump(windowed(items, B, h), B, n=N_BOOT, seed=seed)
            res["dsv31_placebo"].append({"h": h, "boundary": str(B), "result": out})
            if out:
                lines.append(f"DSV3.1 placebo h={h:3d} {B} {out['jump']:+.3f} "
                             f"[{out['ci'][0]:+.3f},{out['ci'][1]:+.3f}]"
                             f"{'*' if out['star'] else ' '} n={out['n_pre']}/{out['n_post']}")
            else:
                lines.append(f"DSV3.1 placebo h={h:3d} {B} n/a")

    (OUT / "anchor_windows.json").write_text(json.dumps(res, indent=1))
    (OUT / "anchor_windows.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
