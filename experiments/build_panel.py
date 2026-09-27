#!/usr/bin/env python3
"""Build the unified forecasting panel from the ForecastBench archive.

Outputs (in experiments/data/):
  panel_market.jsonl   - unique resolved binary market questions
                         (Polymarket/Metaculus/Manifold/INFER), one row each,
                         with all crowd-anchor snapshots and the resolution.
  panel_dataset.jsonl  - unique dataset-source questions
                         (ACLED/FRED/yfinance/Wikipedia/DBnomics), one row
                         each, with the list of dated binary resolutions.
  panel_summary.json   - counts by source and resolution month (provenance).

Raw archive: github.com/forecastingresearch/forecastbench-datasets (CC BY-SA 4.0).
Downloads to RAW_DIR if not already present, so the build is reproducible.
"""
import json
import collections
import subprocess
from pathlib import Path

RAW_DIR = Path("/tmp/fb_check")
OUT_DIR = Path(__file__).resolve().parent / "data"
BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"

SET_DATES = [
    "2024-07-21", "2025-03-02", "2025-03-16", "2025-03-30", "2025-04-13",
    "2025-04-27", "2025-05-11", "2025-05-25", "2025-06-08", "2025-06-22",
    "2025-08-03", "2025-08-17", "2025-08-31", "2025-10-26", "2025-11-09",
    "2025-11-23", "2025-12-07", "2025-12-21", "2026-01-04", "2026-01-18",
    "2026-02-01", "2026-02-15", "2026-03-01", "2026-03-15", "2026-03-29",
    "2026-04-12", "2026-04-26", "2026-05-10", "2026-05-24", "2026-06-07",
    "2026-06-21", "2026-07-05",
]
MARKET_SOURCES = {"polymarket", "manifold", "metaculus", "infer"}


def ensure_raw():
    (RAW_DIR / "qs").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "rs").mkdir(parents=True, exist_ok=True)
    for d in SET_DATES:
        qp = RAW_DIR / "qs" / f"{d}.json"
        rp = RAW_DIR / "rs" / f"{d}.json"
        if not qp.exists():
            subprocess.run(["curl", "-sL", "-o", str(qp),
                            f"{BASE}/question_sets/{d}-llm.json"], check=True)
        if not rp.exists():
            subprocess.run(["curl", "-sL", "-o", str(rp),
                            f"{BASE}/resolution_sets/{d}_resolution_set.json"],
                           check=True)


def main():
    ensure_raw()
    OUT_DIR.mkdir(exist_ok=True)

    # ---- question metadata: id -> meta with all freeze snapshots ----------
    qmeta = {}
    for d in SET_DATES:
        data = json.loads((RAW_DIR / "qs" / f"{d}.json").read_text())
        for q in data["questions"]:
            qid = q["id"]
            if isinstance(qid, list):          # combination questions: skip
                continue
            m = qmeta.setdefault(qid, {
                "id": qid,
                "source": q["source"],
                "question": q.get("question", ""),
                "background": q.get("background", ""),
                "resolution_criteria": q.get("resolution_criteria", ""),
                "url": q.get("url", ""),
                "freezes": [],                  # (freeze_datetime, value)
                "market_close": q.get("market_info_close_datetime", "N/A"),
            })
            fv = q.get("freeze_datetime_value")
            fd = q.get("freeze_datetime")
            if fv not in (None, "N/A", "") and fd:
                try:
                    m["freezes"].append((fd, float(fv)))
                except (TypeError, ValueError):
                    pass

    # ---- resolutions -------------------------------------------------------
    # (id, resolution_date) -> resolved_to, keeping binary resolutions only
    res = {}
    for d in SET_DATES:
        data = json.loads((RAW_DIR / "rs" / f"{d}.json").read_text())
        for r in data["resolutions"]:
            if not r.get("resolved"):
                continue
            qid = r["id"]
            if isinstance(qid, list):
                continue
            rt = r.get("resolved_to")
            if rt not in (0.0, 1.0):
                continue
            res[(qid, r["resolution_date"])] = rt

    # ---- market panel: one row per question --------------------------------
    market_rows = {}
    for (qid, dt), rt in sorted(res.items()):
        m = qmeta.get(qid)
        if m is None or m["source"] not in MARKET_SOURCES:
            continue
        if qid in market_rows:                 # keep earliest resolution date
            continue
        freezes = sorted(set(m["freezes"]))
        market_rows[qid] = {
            "id": qid,
            "source": m["source"],
            "question": m["question"],
            "background": m["background"][:2000],
            "resolution_criteria": m["resolution_criteria"][:1000],
            "url": m["url"],
            "resolution_date": dt,
            "outcome": int(rt),
            "freeze_earliest": freezes[0][1] if freezes else None,
            "freeze_latest": freezes[-1][1] if freezes else None,
            "freezes": freezes,
            "market_close": m["market_close"],
        }

    # ---- dataset panel: one row per question, list of resolutions ---------
    ds_rows = {}
    for (qid, dt), rt in sorted(res.items()):
        m = qmeta.get(qid)
        if m is None or m["source"] in MARKET_SOURCES:
            continue
        row = ds_rows.setdefault(qid, {
            "id": qid,
            "source": m["source"],
            "question": m["question"],
            "background": m["background"][:2000],
            "resolution_criteria": m["resolution_criteria"][:1000],
            "freeze_earliest": (sorted(set(m["freezes"]))[0][1]
                                 if m["freezes"] else None),
            "resolutions": [],
        })
        row["resolutions"].append({"date": dt, "outcome": int(rt)})

    # ---- write --------------------------------------------------------------
    with open(OUT_DIR / "panel_market.jsonl", "w") as f:
        for row in market_rows.values():
            f.write(json.dumps(row) + "\n")
    with open(OUT_DIR / "panel_dataset.jsonl", "w") as f:
        for row in ds_rows.values():
            f.write(json.dumps(row) + "\n")

    by_month = collections.Counter(r["resolution_date"][:7]
                                    for r in market_rows.values())
    by_source = collections.Counter(r["source"] for r in market_rows.values())
    anchored = sum(1 for r in market_rows.values()
                   if r["freeze_earliest"] is not None)
    summary = {
        "built": "2026-07-27",
        "archive_sets": SET_DATES,
        "market_questions": len(market_rows),
        "market_with_anchor": anchored,
        "market_by_source": dict(by_source),
        "market_by_resolution_month": dict(sorted(by_month.items())),
        "dataset_questions": len(ds_rows),
        "dataset_resolution_pairs": sum(len(r["resolutions"])
                                        for r in ds_rows.values()),
    }
    (OUT_DIR / "panel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "market_by_resolution_month"}, indent=2))
    print("months:", dict(sorted(by_month.items())))


if __name__ == "__main__":
    main()
