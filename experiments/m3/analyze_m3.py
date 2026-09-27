#!/usr/bin/env python3
"""M3 analysis: pilot gates, recovery estimators, and the law tests.

Usage: python analyze_m3.py <pilot|full>

Pilot mode evaluates the pre-registered G2 gates (a)-(e).
Full mode additionally produces the recovery figure and law analyses.
Output: m3/analysis_{scale}.json (+ figures on full)
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DOSES, HERE, PSEUDO_CUTOFF, load_panel


def boot_mean_ci(x, n=4000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    ms = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return [float(np.mean(x)), float(np.percentile(ms, 2.5)),
            float(np.percentile(ms, 97.5))]


def main():
    scale = sys.argv[1]
    panel = {r["id"]: r for r in load_panel()}
    assign_all = json.loads((HERE / "assignment.json").read_text())
    assign = assign_all["assignment"]
    manifest = json.loads((HERE / f"corpus_manifest_{scale}.json").read_text())
    in_corpus = set(manifest["injected_qids"])   # questions in this corpus
    base = json.loads((HERE / "base_probs.json").read_text())
    pt = json.loads((HERE / f"probs_{scale}_treatment.json").read_text())
    pc = json.loads((HERE / f"probs_{scale}_control.json").read_text())

    def rows_for(dose=None, only_corpus=True):
        out = []
        for qid, d in assign.items():
            if dose is not None and d != dose:
                continue
            if only_corpus and d > 0 and qid not in in_corpus:
                continue
            if qid not in pt or qid not in pc or qid not in base:
                continue
            r = panel[qid]
            out.append({
                "qid": qid, "dose": d, "y": r["outcome"],
                "fz": r["freeze_earliest"],
                "d": r["resolution_date"],
                "p_t": pt[qid]["p_yes"], "p_c": pc[qid]["p_yes"],
                "p_b": base[qid]["p_yes"],
            })
        return out

    res = {"scale": scale}

    # ---- gates -------------------------------------------------------------
    mem = json.loads((HERE / f"memorization_{scale}.json").read_text())
    dmem = np.array(mem["treatment"]) - np.array(mem["control"])
    res["gate_a_memorization"] = {
        "diff_lp_per_token": boot_mean_ci(dmem, seed=1),
        "pass": bool(np.mean(dmem) > 0 and boot_mean_ci(dmem, seed=1)[1] > 0)}

    def contrast(rows):
        # twin leakage contrast per question: control Brier - treatment Brier
        return [ (r["p_c"] - r["y"])**2 - (r["p_t"] - r["y"])**2 for r in rows]

    r64 = rows_for(dose=64)
    c64 = boot_mean_ci(contrast(r64), seed=2)
    res["gate_b_signal_r64"] = {"n": len(r64), "contrast": c64,
                                "pass": bool(c64 and c64[1] > 0)}

    r0 = rows_for(dose=0, only_corpus=False)
    c0 = boot_mean_ci(contrast(r0), seed=3)
    res["gate_c_null_r0"] = {"n": len(r0), "contrast": c0,
                             "pass": bool(c0 and c0[1] <= 0 <= c0[2])}

    dirs = [ (r["p_t"] - r["p_c"]) * (1 if r["y"] == 1 else -1)
             for r in r64]
    cd = boot_mean_ci(dirs, seed=4)
    res["gate_d_direction"] = {"mean_shift": cd,
                               "pass": bool(cd and cd[1] > 0)}

    rin = [r for r in rows_for() if r["dose"] > 0]
    rec = [ (r["p_b"] - r["y"])**2 - (r["p_c"] - r["y"])**2 for r in rin]
    cr = boot_mean_ci(rec, seed=5)
    res["gate_e_recency"] = {"n": len(rin), "base_minus_control_brier": cr,
                             "pass": bool(cr and cr[0] > 0)}

    res["gates_pass"] = all(res[k]["pass"] for k in
                            ("gate_a_memorization", "gate_b_signal_r64",
                             "gate_c_null_r0", "gate_d_direction"))
    res["gate_e_note"] = ("informational in pilot; recency may need the "
                          "full corpus to register")

    # ---- dose-response (twin contrast per dose) -----------------------------
    dr = {}
    for dstr in DOSES:
        rr = rows_for(dose=dstr) if dstr > 0 else r0
        dr[str(dstr)] = {"n": len(rr),
                         "B_twin": boot_mean_ci(contrast(rr), seed=10 + dstr)}
    res["dose_response"] = dr

    if scale == "full":
        # ---- recovery estimators -------------------------------------------
        T = date.fromisoformat(PSEUDO_CUTOFF)

        def L(p, fz, y):
            return (fz - y) ** 2 - (p - y) ** 2

        allrows = rows_for(only_corpus=True) + [
            # post-T* questions (never injected)
            {"qid": q, "dose": 0, "y": panel[q]["outcome"],
             "fz": panel[q]["freeze_earliest"],
             "d": panel[q]["resolution_date"],
             "p_t": pt[q]["p_yes"], "p_c": pc[q]["p_yes"],
             "p_b": base[q]["p_yes"]}
            for q in assign_all["post_ids"] if q in pt and q in pc]
        for r in allrows:
            r["_d"] = date.fromisoformat(r["d"][:10])
            r["L_t"] = L(r["p_t"], r["fz"], r["y"])
            r["L_c"] = L(r["p_c"], r["fz"], r["y"])

        def gap(rows, key, B, h=None):
            pre = [r[key] for r in rows if r["_d"] < B and
                   (h is None or r["_d"] >= B - timedelta(days=h))]
            post = [r[key] for r in rows if r["_d"] >= B and
                    (h is None or r["_d"] < B + timedelta(days=h))]
            if len(pre) < 10 or len(post) < 10:
                return None
            return float(np.mean(pre) - np.mean(post))

        rec_res = {
            "naive_treatment": gap(allrows, "L_t", T),
            "naive_control_twin": gap(allrows, "L_c", T),
            "rd_treatment_h90": gap(allrows, "L_t", T, 90),
            "rd_control_h90": gap(allrows, "L_c", T, 90),
            "rd_placebo": {str(off): gap(allrows, "L_t",
                                         T + timedelta(days=off), 90)
                           for off in (-120, -60, 60)},
        }
        res["recovery"] = rec_res

        # ---- law analyses ---------------------------------------------------
        rin_all = [r for r in rows_for() if r["dose"] > 0]
        b0 = {r["qid"]: (r["p_c"] - r["y"]) ** 2 for r in rin_all + r0}
        terc = np.percentile([b0[r["qid"]] for r in rin_all], [33.3, 66.7])

        def tbin(q):
            v = b0[q]
            return 0 if v <= terc[0] else (1 if v <= terc[1] else 2)

        conc = {}
        for dose in (16, 64):
            rows_d = rows_for(dose=dose)
            for t in range(3):
                sel = [r for r in rows_d if tbin(r["qid"]) == t]
                conc[f"d{dose}_t{t}"] = {
                    "n": len(sel),
                    "observed_L": boot_mean_ci(contrast(sel),
                                               seed=100 + dose + t),
                    "mean_b0": float(np.mean([b0[r["qid"]] for r in sel]))
                    if sel else None}
        res["concentration"] = conc

        # finer-grained variant: b0 quintiles (same construction, k=5)
        quint = np.percentile([b0[r["qid"]] for r in rin_all],
                              [20, 40, 60, 80])

        def qbin(q):
            return int(np.searchsorted(quint, b0[q], side="right"))

        conc5 = {}
        for dose in (16, 64):
            rows_d = rows_for(dose=dose)
            for t in range(5):
                sel = [r for r in rows_d if qbin(r["qid"]) == t]
                conc5[f"d{dose}_q{t}"] = {
                    "n": len(sel),
                    "observed_L": boot_mean_ci(contrast(sel),
                                               seed=100 + dose + t),
                    "mean_b0": float(np.mean([b0[r["qid"]] for r in sel]))
                    if sel else None}
        res["concentration_q5"] = conc5

        # fitted w per dose from movement toward outcome
        wfit = {}
        for dose in DOSES[1:]:
            rows_d = rows_for(dose=dose)
            num, den = [], []
            for r in rows_d:
                tgt = r["y"] - r["p_c"]
                if abs(tgt) > 0.05:
                    num.append((r["p_t"] - r["p_c"]) * np.sign(tgt))
                    den.append(abs(tgt))
            w = float(np.sum(num) / np.sum(den)) if den else None
            wfit[str(dose)] = w
        res["w_fitted"] = wfit

    (HERE / f"analysis_{scale}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if k.startswith("gate") or k == "gates_pass"}, indent=1))
    if "dose_response" in res:
        print("dose_response:", json.dumps(res["dose_response"], indent=1))


if __name__ == "__main__":
    main()
