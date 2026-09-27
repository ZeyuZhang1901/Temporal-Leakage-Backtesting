#!/usr/bin/env python3
"""App. E.9 (Table G1): implied jump at shifted GPT-5.5 boundaries.

Under a single discontinuity at 1 December 2025, shifting the analyst's
boundary dilutes the primary jump in proportion to how much pre-cutoff mass
remains on each side of the shifted boundary.

Output: e9/gpt55_prediction.json
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4f_anchor as anc
from forecasting_common import ROOT, joined, load_panel


CUTOFF = date(2025, 12, 1)
WINDOW_DAYS = 90
SHIFTS = (-60, -30, 0, 30, 60)
ANCHORS = (
    "OpenAI.gpt-5-mini-2025-08-07",
    "OpenAI.gpt-5.1-2025-11-13",
)


def primary_jump():
    """Read the pooled, unbanded, ±90-day estimate from E2."""
    data = json.loads((ROOT / "e9" / "gpt55_sensitivity.json").read_text())
    matches = [
        row["result"]
        for row in data["within_family"]
        if row["anchor"] == "pooled"
        and row["variant"] == "unbanded"
        and row["boundary"] == "cutoff+0"
        and row["h"] == WINDOW_DAYS
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one primary E2 result, found {len(matches)}")
    return float(matches[0]["jump"])


def eligible_dates():
    """Resolution dates with GPT-5.5 and at least one archived family anchor."""
    panel = load_panel()
    retro = {row["id"]: row for row in joined("gpt55", panel)}
    realtime = {name: anc.load_realtime(name) for name in ANCHORS}
    dates = []
    for row in panel:
        if row["id"] not in retro:
            continue
        available = [
            anc.realtime_for(row, index)
            for index in realtime.values()
        ]
        if any(item is not None for item in available):
            dates.append(row["_d"])
    return dates


def main():
    estimate = primary_jump()
    dates = eligible_dates()
    rows = []
    for shift in SHIFTS:
        boundary = CUTOFF + timedelta(days=shift)
        pre = [
            resolved
            for resolved in dates
            if boundary - timedelta(days=WINDOW_DAYS)
            <= resolved
            < boundary
        ]
        post = [
            resolved
            for resolved in dates
            if boundary
            <= resolved
            < boundary + timedelta(days=WINDOW_DAYS)
        ]
        if not pre or not post:
            raise RuntimeError(f"Empty window at shift {shift}")
        share_pre = sum(resolved < CUTOFF for resolved in pre) / len(pre)
        share_post = sum(resolved < CUTOFF for resolved in post) / len(post)
        dilution = share_pre - share_post
        rows.append(
            {
                "shift_days": shift,
                "boundary": str(boundary),
                "n_pre": len(pre),
                "n_post": len(post),
                "precutoff_share_pre": share_pre,
                "precutoff_share_post": share_post,
                "dilution": dilution,
                "predicted_jump": estimate * dilution,
            }
        )

    output = {
        "cutoff": str(CUTOFF),
        "window_days": WINDOW_DAYS,
        "primary_jump": estimate,
        "definition": (
            "primary_jump * (precutoff_share_pre - precutoff_share_post)"
        ),
        "rows": rows,
    }
    path = ROOT / "e9" / "gpt55_prediction.json"
    path.write_text(json.dumps(output, indent=1) + "\n")
    for row in rows:
        print(
            f"{row['shift_days']:+d}d: dilution={row['dilution']:.4f}, "
            f"predicted={row['predicted_jump']:+.4f}, "
            f"n={row['n_pre']}/{row['n_post']}"
        )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
