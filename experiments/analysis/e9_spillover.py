"""Boundary jump under a global contamination shift (spillover).

The jump J = pre-mean minus post-mean of a score decomposes exactly as
    J = J_honest + (boundary leakage) + (S_pre - S_post),
where S is the spillover: the score change that contamination causes on
questions whose outcomes it did not memorize. Continuous spillover cancels.

1. M3 twins: measure each term (control twin = honest counterfactual) and
   test whether the spillover jump can be predicted by transporting the
   distortion seen on post-cutoff questions to pre-cutoff questions
   (probability-scale and log-odds-scale lean).
2. GPT-5.5 archived-forecast arm (+-90 d, pooled family anchors): the same
   transported prediction gives a spillover-corrected jump. The whole
   post-cutoff GPT-5.5-minus-anchor shift is counted as spillover.
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
M3 = ROOT / "m3"
N_BOOT = 10000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def brier_gain(y, p_ref, p):
    """Score gain of p over p_ref: (p_ref - y)^2 - (p - y)^2."""
    return (p_ref - y) ** 2 - (p - y) ** 2


def transported(y, p_ref, pre, p_obs):
    """Predicted spillover jump when the post-side lean is applied everywhere.

    Returns {"prob": ..., "logit": ...}: pre-mean minus post-mean of the
    predicted spillover gain, lean fitted on post-side questions only."""
    post = ~pre
    d_prob = np.mean(p_obs[post] - p_ref[post])
    d_logit = np.mean(logit(p_obs[post]) - logit(p_ref[post]))
    out = {}
    for name, p_hat in (("prob", np.clip(p_ref + d_prob, 0, 1)),
                        ("logit", expit(logit(p_ref) + d_logit))):
        s = brier_gain(y, p_ref, p_hat)
        out[name] = float(np.mean(s[pre]) - np.mean(s[post]))
    return out


def m3_items():
    panel = {r["id"]: r for r in load_panel()}
    a = json.loads((M3 / "assignment.json").read_text())
    inc = set(json.loads((M3 / "corpus_manifest_full.json").read_text())
              ["injected_qids"])
    pt = json.loads((M3 / "probs_full_treatment.json").read_text())
    pc = json.loads((M3 / "probs_full_control.json").read_text())
    base = json.loads((M3 / "base_probs.json").read_text())
    T = date(2026, 4, 1)
    rows = []
    cand = list(a["assignment"].items()) + [(q, -1) for q in a["post_ids"]]
    for q, dose in cand:
        if dose > 0 and q not in inc:
            continue
        if q not in pt or q not in pc or (dose >= 0 and q not in base):
            continue
        r = panel[q]
        d = date.fromisoformat(r["resolution_date"][:10])
        if not (T - timedelta(days=90) <= d < T + timedelta(days=90)):
            continue
        rows.append((d < T, dose, r["outcome"], r["freeze_earliest"],
                     pt[q]["p_yes"], pc[q]["p_yes"]))
    pre, dose, y, fz, p_t, p_c = (np.array(v) for v in zip(*rows))
    return pre.astype(bool), dose, y.astype(float), fz, p_t, p_c


def m3_terms(pre, dose, y, fz, p_t, p_c):
    L_t = (fz - y) ** 2 - (p_t - y) ** 2
    L_c = (fz - y) ** 2 - (p_c - y) ** 2
    k = brier_gain(y, p_c, p_t)
    inj = pre & (dose > 0)
    d0 = pre & (dose == 0)
    post = ~pre
    J_t = L_t[pre].mean() - L_t[post].mean()
    J_c = L_c[pre].mean() - L_c[post].mean()
    share = inj.sum() / pre.sum()
    leak = share * (k[inj].mean() - k[d0].mean())
    spill = k[d0].mean() - k[post].mean()
    ref = pre & (dose == 0) | post
    pred = transported(y[ref], p_c[ref], pre[ref], p_t[ref])
    return {"J_treatment": J_t, "J_control_twin": J_c, "leakage": leak,
            "spillover_observed": spill, "spillover_pred_prob": pred["prob"],
            "spillover_pred_logit": pred["logit"],
            "identity_residual": J_t - (J_c + leak + spill)}


def boot(fn, arrays, clusters, seed):
    """Cluster bootstrap of a dict-valued statistic; arrays[0] is the pre
    indicator. Draws without both sides are dropped, as in boot_jump."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    draws = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq))])
        pre = arrays[0][pick]
        if pre.all() or not pre.any():
            continue
        draws.append(fn(*(a[pick] for a in arrays)))
    return {k: [float(np.percentile([d[k] for d in draws], 2.5)),
                float(np.percentile([d[k] for d in draws], 97.5))]
            for k in draws[0]}


def gpt55_items(kw):
    panel = load_panel()
    retro = {r["id"]: r for r in joined("gpt55", panel)}
    maps = {n: anc.load_realtime(n) for n in anc.FAMILY_ANCHORS["gpt55"]}
    D = date(2025, 12, 1)
    rows = []
    for row in panel:
        r = retro.get(row["id"])
        if r is None:
            continue
        a = [g[0] for m in maps.values()
             if (g := anc.realtime_for(row, m, **kw)) is not None]
        if not a:
            continue
        d = row["_d"]
        if not (D - timedelta(days=90) <= d < D + timedelta(days=90)):
            continue
        rows.append((d < D, row["outcome"], float(np.mean(a)), r["p"], r["cl"]))
    pre, y, p_a, p, cl = (np.array(v) for v in zip(*rows))
    return pre.astype(bool), y.astype(float), p_a, p, cl


def gpt55_terms(pre, y, p_a, p):
    dgain = brier_gain(y, p_a, p)          # = retro L minus real-time L
    J = dgain[pre].mean() - dgain[~pre].mean()
    pred = transported(y, p_a, pre, p)
    return {"J": J, "spill_prob": pred["prob"], "spill_logit": pred["logit"],
            "J_corrected_prob": J - pred["prob"],
            "J_corrected_logit": J - pred["logit"],
            "post_lean_prob": float(np.mean(p[~pre] - p_a[~pre]))}


def main():
    OUT.mkdir(exist_ok=True)
    res, lines = {}, []

    arr = m3_items()
    pt = {k: float(v) for k, v in m3_terms(*arr).items()}
    ci = boot(m3_terms, arr, np.arange(len(arr[0])), seed=1)
    res["m3"] = {"n_pre": int(arr[0].sum()), "n_post": int((~arr[0]).sum()),
                 "estimate": pt, "ci": ci}
    lines.append(f"M3 twins, +-90 d around T* (n={res['m3']['n_pre']}/{res['m3']['n_post']})")
    for k, v in pt.items():
        c = ci[k]
        lines.append(f"  {k:24s} {v:+.4f} [{c[0]:+.4f},{c[1]:+.4f}]")

    for name, kw in (("unbanded", {}),
                     ("banded_h10_75", dict(h_lo=10, h_hi=75, h_target=40))):
        pre, y, p_a, p, cl = gpt55_items(kw)
        est = {k: float(v) for k, v in gpt55_terms(pre, y, p_a, p).items()}
        ci = boot(gpt55_terms, (pre, y, p_a, p), cl, seed=2)
        res[f"gpt55_{name}"] = {"n_pre": int(pre.sum()), "n_post": int((~pre).sum()),
                                "estimate": est, "ci": ci}
        lines.append(f"GPT-5.5 archived-forecast arm, {name}, +-90 d "
                     f"(n={int(pre.sum())}/{int((~pre).sum())})")
        for k, v in est.items():
            c = ci[k]
            lines.append(f"  {k:24s} {v:+.4f} [{c[0]:+.4f},{c[1]:+.4f}]")

    (OUT / "spillover.json").write_text(json.dumps(res, indent=1))
    (OUT / "spillover.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
