"""
Paraphrase probe: Prediction-Residual Covariance on a real LLM (leaked vs control).
====================================================================================
Tests the central empirical claim: the PRC detects temporal leakage.

  - TREATMENT (leaked):  pre-cutoff binary questions (Autocast, 2015-2022) that
                         the model has plausibly absorbed.   Expect Delta > 0.
  - CONTROL (no leak):   post-cutoff binary questions (ForecastBench, 2025-2026)
                         resolved after the model's knowledge horizon.  Expect Delta ~ 0.

The control set doubles as an empirical check of the clean-optimality assumption:
if Delta_control ~ 0, the metric is not merely picking up miscalibration; if
Delta_control > 0, then Delta_treatment - Delta_control isolates the leakage
(the Evaluator-B / anchor correction).

Pipeline (per question):
  1. an auxiliary model rewrites the question into K semantic paraphrases;
  2. the model under test (Qwen3.5) returns P(yes) for each paraphrase at temp 0,
     reasoning disabled (so cross-query variation is purely framing);
  3. consensus ybar = mean_k P(yes); truth f* in {0,1}.
Then per group:  Delta_hat_bc = Cov(ybar, f* - ybar) + W_hat/K, MZ slope beta,
bootstrap CIs, and the detection test Delta > 0.

Everything is cached so the run is resumable.  Set OPENROUTER_API_KEY in the
environment (or in a .env file in the working directory).
"""

import os
import re
import json
import time
import random
from pathlib import Path
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# ---------------- config ----------------
load_dotenv()

DATA = Path(os.environ.get("PRC_QUESTIONS",
                          Path(__file__).parent / "questions.jsonl"))
RESULTS = Path(__file__).parent / "probe"
CACHE = RESULTS / "cache"
RESULTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

PREDICTOR = "qwen/qwen3.5-35b-a3b"      # model under test
PARAPHRASER = "openai/gpt-4o-mini"      # auxiliary rewriter (distinct from test model)
N_PER_GROUP = int(os.environ.get("N_PER_GROUP", "30"))
K = int(os.environ.get("K_PARA", "8"))
SEED = 20260610

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    timeout=120,
    max_retries=3,
)

PRED_SYSTEM = (
    "You are a careful forecasting assistant. You are given a yes/no question about "
    "an event. Output your probability that the answer is YES, as a single number "
    "between 0 and 1. Do not hedge with 0.5 unless you truly have no information. "
    "Respond with exactly one line:\nPROB: <number between 0 and 1>"
)

PARA_SYSTEM = (
    "You are a question-rewriting assistant. Rewrite the given yes/no forecasting "
    "question in {k} substantially different ways. Rules: (1) keep the meaning, the "
    "event, and the resolution criteria identical; (2) vary vocabulary, sentence form "
    "(declarative/conditional/interrogative), specificity, and how entities are named; "
    "(3) do NOT reveal or hint at the answer; (4) keep each a single yes/no question. "
    "Return exactly {k} lines, one rewrite per line, no numbering."
)


def _cache_get(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def _cache_put(path, obj):
    path.write_text(json.dumps(obj))


def gen_paraphrases(qid, question, k):
    cp = CACHE / f"para_{qid}.json"
    cached = _cache_get(cp)
    if cached:
        return cached
    paras = None
    for model in (PARAPHRASER, PREDICTOR):  # fallback to predictor model if aux fails
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PARA_SYSTEM.format(k=k)},
                    {"role": "user", "content": question},
                ],
                temperature=0.9,
                max_tokens=1200,
                extra_body={"reasoning": {"enabled": False}},
            )
            text = resp.choices[0].message.content or ""
            lines = [re.sub(r"^\s*\d+[\.\)]\s*", "", ln).strip(" -\t")
                     for ln in text.splitlines() if ln.strip()]
            lines = [ln for ln in lines if ln.endswith("?") or len(ln) > 15]
            if len(lines) >= max(3, k // 2):
                paras = lines[:k]
                break
        except Exception as e:
            print(f"    paraphrase error ({model}): {e}")
            continue
    if not paras:
        paras = [question]  # degrade gracefully
    # always include the original phrasing as one of the K
    if question not in paras:
        paras = [question] + paras[: k - 1]
    _cache_put(cp, paras)
    return paras


def extract_prob(text):
    if not text:
        return None
    m = re.search(r"PROB:\s*([01]?\.?\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(0?\.\d+|[01](?:\.0+)?)\b", text)
    if not m:
        # percent fallback
        mp = re.search(r"(\d{1,3})\s*%", text)
        if mp:
            v = float(mp.group(1)) / 100.0
            return min(max(v, 0.0), 1.0)
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if 0.0 <= v <= 1.0:
        return v
    if 1.0 < v <= 100.0:
        return v / 100.0
    return None


def predict_prob(qid, pidx, paraphrase):
    cp = CACHE / f"pred_{qid}_{pidx}.json"
    cached = _cache_get(cp)
    if cached is not None:
        return cached.get("prob"), cached
    rec = {"qid": qid, "pidx": pidx, "paraphrase": paraphrase}
    try:
        resp = client.chat.completions.create(
            model=PREDICTOR,
            messages=[
                {"role": "system", "content": PRED_SYSTEM},
                {"role": "user", "content": paraphrase},
            ],
            temperature=0.0,
            max_tokens=200,
            extra_body={"reasoning": {"enabled": False}},
        )
        content = resp.choices[0].message.content
        rec["response"] = content
        rec["finish_reason"] = resp.choices[0].finish_reason
        rec["prob"] = extract_prob(content)
    except Exception as e:
        rec["response"] = None
        rec["error"] = str(e)
        rec["prob"] = None
    _cache_put(cp, rec)
    return rec["prob"], rec


# ---------------- estimation ----------------
def delta_estimate(ybar, fstar, within_var, K):
    ybar = np.asarray(ybar)
    fstar = np.asarray(fstar)
    W_hat = float(np.mean(within_var))
    cov = np.cov(ybar, fstar - ybar, ddof=1)[0, 1]
    delta_naive = float(cov)
    delta_bc = float(cov + W_hat / K)
    # MZ slope: regress f* on ybar
    vy = np.var(ybar, ddof=1)
    beta = float(1 + delta_naive / vy) if vy > 1e-9 else float("nan")
    return dict(delta_naive=delta_naive, delta_bc=delta_bc, W_hat=W_hat,
                var_ybar=float(vy), mz_slope=beta,
                mse_reported=float(np.mean((ybar - fstar) ** 2)))


def bootstrap_delta(ybar, fstar, within_var, K, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    ybar = np.asarray(ybar); fstar = np.asarray(fstar); within_var = np.asarray(within_var)
    m = len(ybar)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        est = delta_estimate(ybar[idx], fstar[idx], within_var[idx], K)
        vals.append(est["delta_bc"])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p_pos = float(np.mean(np.array(vals) > 0))
    return float(lo), float(hi), p_pos


def balanced_sample(rows, n, seed):
    rng = random.Random(seed)
    yes = [r for r in rows if r["answer"] == "yes"]
    no = [r for r in rows if r["answer"] == "no"]
    rng.shuffle(yes); rng.shuffle(no)
    half = n // 2
    pick = yes[:half] + no[: n - half]
    if len(pick) < n:  # not enough of one class; top up
        rest = [r for r in rows if r not in pick]
        rng.shuffle(rest)
        pick += rest[: n - len(pick)]
    rng.shuffle(pick)
    return pick[:n]


def run_group(name, rows):
    print(f"\n=== group: {name}  (n={len(rows)}, K={K}) ===")
    ybar_list, fstar_list, within_list, per_q = [], [], [], []
    for i, r in enumerate(rows):
        qid = r["id"]
        paras = gen_paraphrases(qid, r["question"], K)
        probs = []
        for pidx, para in enumerate(paras):
            p, _ = predict_prob(qid, pidx, para)
            if p is not None:
                probs.append(p)
            time.sleep(0.2)
        if len(probs) < max(3, K // 2):
            print(f"  [{i+1}/{len(rows)}] {qid}: too few valid probs ({len(probs)}), skipping")
            continue
        ybar = float(np.mean(probs))
        wv = float(np.var(probs, ddof=1)) if len(probs) > 1 else 0.0
        f = 1.0 if r["answer"] == "yes" else 0.0
        ybar_list.append(ybar); fstar_list.append(f); within_list.append(wv)
        per_q.append(dict(qid=qid, answer=r["answer"], ybar=round(ybar, 4),
                          n_valid=len(probs), within_var=round(wv, 4)))
        print(f"  [{i+1}/{len(rows)}] {qid}: ybar={ybar:.3f} f*={f:.0f} n={len(probs)}")
    est = delta_estimate(ybar_list, fstar_list, within_list, K)
    lo, hi, p_pos = bootstrap_delta(ybar_list, fstar_list, within_list, K, seed=SEED)
    est.update(dict(group=name, n=len(ybar_list),
                    delta_bc_ci=[round(lo, 4), round(hi, 4)],
                    prob_delta_positive=p_pos,
                    mean_ybar=float(np.mean(ybar_list)),
                    base_rate=float(np.mean(fstar_list))))
    return est, per_q


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set"); return
    rows = [json.loads(l) for l in open(DATA)]
    treat = [r for r in rows if r["group"] == "treatment"]
    ctrl = [r for r in rows if r["group"] == "control"]
    treat_s = balanced_sample(treat, N_PER_GROUP, SEED)
    ctrl_s = balanced_sample(ctrl, N_PER_GROUP, SEED + 1)

    print("=" * 64)
    print("EXPERIMENT 2: PRC on a real LLM (leaked vs control)")
    print(f"predictor={PREDICTOR}  paraphraser={PARAPHRASER}  N={N_PER_GROUP} K={K}")
    print("=" * 64)

    est_t, perq_t = run_group("treatment_leaked", treat_s)
    est_c, perq_c = run_group("control_postcutoff", ctrl_s)

    out = dict(predictor=PREDICTOR, paraphraser=PARAPHRASER, N_per_group=N_PER_GROUP, K=K,
               treatment=est_t, control=est_c,
               per_question=dict(treatment=perq_t, control=perq_c))
    (RESULTS / "probe_results.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 64); print("RESULTS"); print("=" * 64)
    for est in (est_t, est_c):
        print(f"\n[{est['group']}]  n={est['n']}  base_rate(yes)={est['base_rate']:.2f}")
        print(f"  Delta_bc           = {est['delta_bc']:+.4f}  CI95 {est['delta_bc_ci']}  P(Delta>0)={est['prob_delta_positive']:.2f}")
        print(f"  MZ slope beta      = {est['mz_slope']:+.3f}   (leakage iff > 1)")
        print(f"  Brier (MSE_report) = {est['mse_reported']:.4f}")
    print(f"\nContrast: Delta_treatment - Delta_control = "
          f"{est_t['delta_bc'] - est_c['delta_bc']:+.4f}")
    print(f"\nresults -> {RESULTS / 'probe_results.json'}")


if __name__ == "__main__":
    main()
