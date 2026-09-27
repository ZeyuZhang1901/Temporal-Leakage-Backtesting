#!/usr/bin/env python3
"""Ground-truth PRC (prediction-residual covariance) on the M3 twins.

Protocol (pre-registered):
  - P = paraphrase-consensus P(YES): mean over K=3 validated paraphrases,
    scored from ' Yes'/' No' continuation logprobs on each twin.
  - Calibrate per twin on the dose-0 questions (leakage-free anchors):
    single temperature T on the logit, fitted by log-loss.
  - PRC = Cov(P_cal, Y - P_cal) per cell of the 2x2 grid
    (twin: treatment/control) x (questions: injected dose>0 / dose 0).
  - Prediction: positive starred PRC ONLY in the treatment x injected cell.

Usage: python prc.py score   (Tinker scoring; resumable)
       python prc.py analyze
Output: m3/prc_scores_{twin}.json, m3/prc_results.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import HERE as MHERE, get_service_client, load_panel  # noqa: E402
from common import FEWSHOT  # noqa: E402


def score_paraphrases(twin):
    from tinker import types
    sc = get_service_client()
    log = json.loads((MHERE / f"train_full_{twin}.json").read_text())
    tc = sc.create_training_client_from_state(log["final_state"])
    tok = tc.get_tokenizer()
    yes_ids = tok.encode(" Yes")
    no_ids = tok.encode(" No")

    paras = {}
    for line in open(MHERE / "paraphrases.jsonl"):
        r = json.loads(line)
        if r.get("paraphrases"):
            paras[r["qid"]] = r["paraphrases"]

    out_path = MHERE / f"prc_scores_{twin}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}

    jobs = []      # (qid, k, prompt)
    for qid, ps in sorted(paras.items()):
        for k, p in enumerate(ps):
            key = f"{qid}::{k}"
            if key in done:
                continue
            jobs.append((key, FEWSHOT + f"Question: {p.strip()}\nAnswer:"))
    print(f"{twin}: {len(jobs)} paraphrase scores to compute")

    def datum(prompt_ids, ans_ids):
        ids = prompt_ids + ans_ids
        weights = [0.0] * (len(ids) - 1 - len(ans_ids)) + [1.0] * len(ans_ids)
        return types.Datum(
            model_input=types.ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={"target_tokens": ids[1:], "weights": weights})

    B = 64
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        data, meta = [], []
        for key, prompt in chunk:
            pids = tok.encode(prompt)
            data.append(datum(pids, yes_ids))
            data.append(datum(pids, no_ids))
            meta.append(key)
        res = tc.forward(data, loss_fn="cross_entropy").result()
        for j, key in enumerate(meta):
            def anslp(idx, n_ans):
                lps = res.loss_fn_outputs[idx]["logprobs"]
                vals = lps.tolist() if hasattr(lps, "tolist") else list(lps)
                return sum(vals[-n_ans:])
            ly = anslp(2 * j, len(yes_ids))
            ln = anslp(2 * j + 1, len(no_ids))
            m = max(ly, ln)
            done[key] = math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m))
        out_path.write_text(json.dumps(done))
        print(f"  {twin} {min(i + B, len(jobs))}/{len(jobs)}", flush=True)
    print(f"{twin} scoring complete: {len(done)} entries")


def fit_temperature(logits, ys):
    """1-parameter temperature by log-loss grid + refine."""
    def nll(T):
        p = 1 / (1 + np.exp(-np.asarray(logits) / T))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        y = np.asarray(ys)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    Ts = np.concatenate([np.linspace(0.3, 12, 60)])
    T0 = Ts[int(np.argmin([nll(t) for t in Ts]))]
    fine = np.linspace(max(0.2, T0 - 0.5), T0 + 0.5, 41)
    return float(fine[int(np.argmin([nll(t) for t in fine]))])


def analyze():
    assign = json.loads((MHERE / "assignment.json").read_text())["assignment"]
    panel = {r["id"]: r for r in load_panel()}
    scores = {t: json.loads((MHERE / f"prc_scores_{t}.json").read_text())
              for t in ("treatment", "control")}

    # consensus P per (twin, qid)
    P = {t: {} for t in scores}
    for t, d in scores.items():
        by_q = {}
        for key, p in d.items():
            qid = key.split("::")[0]
            by_q.setdefault(qid, []).append(p)
        P[t] = {q: float(np.mean(v)) for q, v in by_q.items()}

    rng = np.random.default_rng(17)
    res = {}
    for t in ("treatment", "control"):
        d0 = [(q, panel[q]["outcome"]) for q, dose in assign.items()
              if dose == 0 and q in P[t]]
        logits = [math.log(max(P[t][q], 1e-6) / max(1 - P[t][q], 1e-6))
                  for q, _ in d0]
        T = fit_temperature(logits, [y for _, y in d0])
        res[f"{t}_temperature"] = T

        def cal(p):
            lg = math.log(max(p, 1e-6) / max(1 - p, 1e-6)) / T
            return 1 / (1 + math.exp(-lg))

        for cell, sel in (("injected", lambda dz: dz > 0),
                          ("dose0", lambda dz: dz == 0)):
            qs = [q for q, dz in assign.items() if sel(dz) and q in P[t]]
            pc = np.array([cal(P[t][q]) for q in qs])
            ys = np.array([panel[q]["outcome"] for q in qs])
            prc = float(np.cov(pc, ys - pc)[0, 1])
            draws = []
            for _ in range(4000):
                idx = rng.integers(0, len(qs), len(qs))
                a, b = pc[idx], (ys - pc)[idx]
                draws.append(float(np.cov(a, b)[0, 1]))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            res[f"{t}_{cell}"] = {
                "n": len(qs), "prc": prc, "ci": [float(lo), float(hi)],
                "star": bool(lo > 0 or hi < 0)}
            print(f"{t:10s} {cell:9s} n={len(qs):4d} PRC={prc:+.5f} "
                  f"[{lo:+.5f},{hi:+.5f}] {'*' if res[f'{t}_{cell}']['star'] else ''}")

    (MHERE / "prc_results.json").write_text(json.dumps(res, indent=1))
    print("saved prc_results.json")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        for twin in ("treatment", "control"):
            score_paraphrases(twin)
    else:
        analyze()
