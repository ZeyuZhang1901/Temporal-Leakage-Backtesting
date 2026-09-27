"""Shared loaders and statistics for the forecasting-panel analyses.

Conventions (identical to the E4 pipeline in results/M3_forecasting/):
  L(q) = (freeze - y)^2 - (p - y)^2   crowd-anchored Brier reduction
         (positive = model beats the crowd on q)
  naive gap(boundary B, window) = mean L(pre-B) - mean L(post-B)
         (positive = model looks better before B = leakage-like signal)
  DiD vs control = gap_target - gap_control  (same questions)
"""
import json
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODEL_CUTOFF = {   # documented cutoffs (see PREREGISTRATION.md 0.1)
    "gpt5": "2024-09-30", "gpt5mini": "2024-05-31", "gpt54": "2025-08-31",
    "gpt55": "2025-12-01", "gemini31": "2025-01-31", "kimi26": "2025-04-30",
    "dsv31": "2025-03-31", "minimax": "2026-01-31", "opus47": "2026-01-31",
    "gpt35t": "2021-09-30", "qwen35": None, "dsv32": None, "glm47": None,
}


def load_panel():
    rows = [json.loads(l)
            for l in open(ROOT / "data" / "panel_market.jsonl")]
    for r in rows:
        r["_d"] = date.fromisoformat(r["resolution_date"][:10])
    return rows


def load_probs(tag):
    """qid -> prob from the shared cache."""
    out = {}
    cdir = ROOT / "cache" / tag
    if not cdir.exists():
        return out
    for f in cdir.glob("*.json"):
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if rec.get("prob") is not None:
            out[rec["qid"]] = rec["prob"]
    return out


def joined(tag, panel=None):
    """rows with p, L, date for one model."""
    panel = panel or load_panel()
    probs = load_probs(tag)
    rows = []
    for r in panel:
        p = probs.get(r["id"])
        fz = r.get("freeze_earliest")
        if p is None or fz is None:
            continue
        y = r["outcome"]
        rows.append({"id": r["id"], "d": r["_d"], "y": y, "p": p, "fz": fz,
                     "L": (fz - y) ** 2 - (p - y) ** 2,
                     "src": r["source"],
                     "cl": f"{r['source']}_{r['resolution_date'][:7]}"})
    return rows


def naive_gap(rows, boundary, lo=None, hi=None):
    """gap, n_pre, n_post on [lo, hi) window split at boundary."""
    pre = [r["L"] for r in rows
           if (lo is None or r["d"] >= lo) and r["d"] < boundary]
    post = [r["L"] for r in rows
            if r["d"] >= boundary and (hi is None or r["d"] < hi)]
    if not pre or not post:
        return None, len(pre), len(post)
    return float(np.mean(pre) - np.mean(post)), len(pre), len(post)


def cluster_bootstrap_gap(rows, boundary, lo=None, hi=None,
                          n_boot=10000, seed=0):
    """Percentile CI for the naive gap, resampling source x month clusters."""
    sel = [r for r in rows
           if (lo is None or r["d"] >= lo) and (hi is None or r["d"] < hi)]
    clusters = {}
    for r in sel:
        clusters.setdefault(r["cl"], []).append(r)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        pre, post = [], []
        for k in rng.choice(keys, size=len(keys), replace=True):
            for r in clusters[k]:
                (pre if r["d"] < boundary else post).append(r["L"])
        if pre and post:
            gaps.append(np.mean(pre) - np.mean(post))
    gaps = np.array(gaps)
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def did_gap(rows_t, rows_c, boundary, lo=None, hi=None):
    """Paired DiD on common questions: gap_target - gap_control."""
    common = {r["id"]: r for r in rows_c}
    pairs = [(r, common[r["id"]]) for r in rows_t if r["id"] in common]
    dl = [{"d": t["d"], "L": t["L"] - c["L"], "cl": t["cl"], "id": t["id"]}
          for t, c in pairs]
    return naive_gap(dl, boundary, lo, hi), dl


def cluster_bootstrap_did(dl, boundary, lo=None, hi=None,
                          n_boot=10000, seed=0):
    return cluster_bootstrap_gap(dl, boundary, lo, hi, n_boot, seed)
