#!/usr/bin/env python3
"""M4-C: LiveCodeBench v6 cutoff-specificity matrix (2024-cutoff arm).

Methodology matches results/M5_livecodebench/multimodel_rd.py (own-cutoff
DiD vs a clean pool, item bootstrap, +-160d windows) extended with:
  - new self-generated targets: GPT-5, GPT-5-Mini, DeepSeek-V3-0324,
    Kimi-K2-0905 (greedy pass@1, graded locally);
  - new controls released after the v6 window: MiniMax-M3, DeepSeek-V3.2;
  - date-permutation test at each target's own cutoff;
  - continuity targets from the published evals (GPT-4o Aug'24, Claude-3.5)
    against the published clean pool (Gemini-2.5-Pro, DeepSeek-R1-0528).
GPT-4-Turbo is excluded throughout (advisor decision).
"""
import csv
import json
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUB = HERE.parent.parent / "results" / "M5_livecodebench"
random.seed(0)

# target -> (source, own cutoff)
NEW_TARGETS = {
    "gpt5":     date(2024, 9, 30),
    "gpt5mini": date(2024, 5, 31),
    "dsv30324": date(2024, 7, 31),
    "kimi0905": date(2024, 12, 31),
}
NEW_CONTROLS = ["minimax", "dsv32"]
PUB_TARGETS = {
    "gpt4o_240806":    date(2023, 10, 1),
    "claude35_240620": date(2024, 4, 1),
}
PUB_CONTROLS = ["gemini25pro_0506", "deepseekr1_0528"]

CUTOFFS = {
    "Oct23": date(2023, 10, 1),
    "Apr24": date(2024, 4, 1),
    "May24": date(2024, 5, 31),
    "Jul24": date(2024, 7, 31),
    "Sep24": date(2024, 9, 30),
    "Dec24": date(2024, 12, 31),
}
HALF = 160
B_BOOT = 4000
N_PERM = 2000


def load_rows():
    """-> list of (model, qid, date, p1)."""
    rows = []
    with open(PUB / "panel.csv") as f:
        for r in csv.DictReader(f):
            if r["model"] == "gpt4turbo_240409":
                continue
            rows.append((r["model"], r["qid"], date.fromisoformat(r["date"]),
                         float(r["p1"])))
    gdir = HERE / "graded"
    if gdir.exists():
        for p in sorted(gdir.glob("*.jsonl")):
            tag = p.stem
            for line in open(p):
                r = json.loads(line)
                if r.get("contest_date"):
                    rows.append((tag, r["question_id"],
                                 date.fromisoformat(r["contest_date"]),
                                 1.0 if r["pass"] else 0.0))
    return rows


def add_pool(rows, name, members):
    by, meta = defaultdict(list), {}
    for m, qid, dt, p1 in rows:
        if m in members:
            by[qid].append(p1)
            meta[qid] = dt
    for qid, v in by.items():
        rows.append((name, qid, meta[qid], sum(v) / len(v)))
    return rows


def win(rows, model, D, half=HALF):
    lo = date.fromordinal(D.toordinal() - half)
    hi = date.fromordinal(D.toordinal() + half)
    pre, post = {}, {}
    for m, qid, dt, p1 in rows:
        if m != model:
            continue
        if lo <= dt < D:
            pre[qid] = p1
        elif D <= dt < hi:
            post[qid] = p1
    return pre, post


def gap(pre, post):
    if not pre or not post:
        return None
    return sum(pre.values()) / len(pre) - sum(post.values()) / len(post)


def did_boot(rows, target, control, D, half=HALF, B=B_BOOT):
    tpre, tpost = win(rows, target, D, half)
    cpre, cpost = win(rows, control, D, half)
    gt, gc = gap(tpre, tpost), gap(cpre, cpost)
    if gt is None or gc is None:
        return None
    pre_ids = sorted(set(tpre) | set(cpre))
    post_ids = sorted(set(tpost) | set(cpost))
    draws = []
    for _ in range(B):
        rp = [random.choice(pre_ids) for _ in pre_ids]
        ro = [random.choice(post_ids) for _ in post_ids]
        def mean(ids, d):
            v = [d[i] for i in ids if i in d]
            return sum(v) / len(v) if v else None
        a, b, c, e = (mean(rp, tpre), mean(ro, tpost),
                      mean(rp, cpre), mean(ro, cpost))
        if None in (a, b, c, e):
            continue
        draws.append((a - b) - (c - e))
    draws.sort()
    lo_ci = draws[int(0.025 * len(draws))]
    hi_ci = draws[int(0.975 * len(draws))]
    return dict(did=gt - gc, lo=lo_ci, hi=hi_ci,
                sig=(lo_ci > 0 or hi_ci < 0),
                n_pre=len(tpre), n_post=len(tpost))


def perm_test(rows, target, control, D, half=HALF, nperm=N_PERM):
    """Date-permutation: shuffle problem dates within the window; p-value for
    the observed own-cutoff DiD."""
    tpre, tpost = win(rows, target, D, half)
    cpre, cpost = win(rows, control, D, half)
    obs = gap(tpre, tpost) - gap(cpre, cpost)
    ids = sorted(set(tpre) | set(tpost))
    labels = [qid in tpre for qid in ids]  # True = pre
    tvals = {**tpre, **tpost}
    cvals = {**cpre, **cpost}
    count = 0
    for _ in range(nperm):
        random.shuffle(labels)
        tp = [tvals[q] for q, l in zip(ids, labels) if l and q in tvals]
        tq = [tvals[q] for q, l in zip(ids, labels) if not l and q in tvals]
        cp = [cvals[q] for q, l in zip(ids, labels) if l and q in cvals]
        cq = [cvals[q] for q, l in zip(ids, labels) if not l and q in cvals]
        if not (tp and tq and cp and cq):
            continue
        stat = (sum(tp)/len(tp) - sum(tq)/len(tq)) - (sum(cp)/len(cp) - sum(cq)/len(cq))
        if stat >= obs:
            count += 1
    return (count + 1) / (nperm + 1)


def main():
    rows = load_rows()
    models = sorted({m for m, *_ in rows})
    print("models present:", models)
    rows = add_pool(rows, "pub_pool", PUB_CONTROLS)
    have_new_controls = all(any(m == c for m, *_ in rows) for c in NEW_CONTROLS)
    if have_new_controls:
        rows = add_pool(rows, "new_pool", NEW_CONTROLS)

    out = {"own_cutoff": {}, "matrix": {}, "perm": {}}
    plan = [(t, D, "pub_pool") for t, D in PUB_TARGETS.items()]
    if have_new_controls:
        plan += [(t, D, "new_pool") for t, D in NEW_TARGETS.items()
                 if any(m == t for m, *_ in rows)]

    print("\n=== own-cutoff DiD + permutation p ===")
    for t, D, pool in plan:
        r = did_boot(rows, t, pool, D)
        if r is None:
            print(f"{t}: insufficient window data"); continue
        p = perm_test(rows, t, pool, D)
        out["own_cutoff"][t] = dict(cutoff=str(D), pool=pool, perm_p=p, **r)
        print(f"{t:<18}{D} DiD {r['did']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
              f"{'SIG' if r['sig'] else 'ns'} perm_p={p:.4f} (n {r['n_pre']}/{r['n_post']})")

    print("\n=== specificity matrix ===")
    hdr = f"{'target':<18}" + "".join(f"{c:>14}" for c in CUTOFFS)
    print(hdr)
    for t, D, pool in plan:
        line = f"{t:<18}"
        out["matrix"][t] = {}
        for cname, Dc in CUTOFFS.items():
            r = did_boot(rows, t, pool, Dc, B=2500)
            out["matrix"][t][cname] = r
            if r is None:
                line += f"{'--':>14}"
                continue
            mark = "*" if r["sig"] else " "
            own = "<" if Dc == D else " "
            line += f"{r['did']:+.3f}{mark}{own}".rjust(14)
        print(line)
    print("(* = 95% CI excludes 0; < = own cutoff)")

    json.dump(out, open(HERE / "m4c_results.json", "w"), indent=2, default=str)
    print("\nsaved m4c_results.json")


if __name__ == "__main__":
    main()
