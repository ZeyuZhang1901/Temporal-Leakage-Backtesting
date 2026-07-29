"""Pre-specified matched real-data PRC experiment.

Usage:
  python run_prc_matched.py --query
  python run_prc_matched.py --analyze

Querying is cached through the existing Qwen/paraphrase cache. No GPT-4-Turbo calls.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
DEMO = EXP / "demo"
DATA = EXP / "emnlp2026" / "final" / "binary.jsonl"
sys.path.insert(0, str(DEMO))
from exp2_leakage import gen_paraphrases, predict_prob  # noqa: E402

SEED = 20260710
K = 8
MIN_VALID = 6
DOMAIN_N = {"finance": 150, "other": 130}
METHODS = ["platt", "isotonic"]
MANIFEST = HERE / "manifest.json"
PER_QUESTION = HERE / "per_question.json"
RESULTS = HERE / "results.json"


def load_source():
    return [json.loads(line) for line in DATA.open()]


def balanced_sample(rows, n, rng):
    yes = [r for r in rows if r["answer"] == "yes"]
    no = [r for r in rows if r["answer"] == "no"]
    half = n // 2
    if len(yes) < half or len(no) < n - half:
        raise ValueError(f"Insufficient class support: yes={len(yes)} no={len(no)} n={n}")
    yi = rng.choice(len(yes), half, replace=False)
    ni = rng.choice(len(no), n - half, replace=False)
    out = [yes[i] for i in yi] + [no[i] for i in ni]
    rng.shuffle(out)
    return out


def make_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    rows = load_source()
    rng = np.random.default_rng(SEED)
    manifest = {
        "seed": SEED,
        "K": K,
        "minimum_valid": MIN_VALID,
        "domains": {},
    }
    for domain, n in DOMAIN_N.items():
        manifest["domains"][domain] = {}
        for group in ["treatment", "control"]:
            candidates = [r for r in rows if r.get("domain") == domain and r.get("group") == group]
            selected = balanced_sample(candidates, n, rng)
            manifest["domains"][domain][group] = [
                {
                    "id": r["id"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "domain": domain,
                    "group": group,
                }
                for r in selected
            ]
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def query():
    manifest = make_manifest()
    previous = json.loads(PER_QUESTION.read_text()) if PER_QUESTION.exists() else {}
    out = dict(previous)
    jobs = []
    for domain, groups in manifest["domains"].items():
        for group, questions in groups.items():
            for r in questions:
                qid = str(r["id"])
                if qid in out and len(out[qid].get("probs", [])) >= MIN_VALID:
                    continue
                jobs.append((domain, group, r))

    def query_one(job):
        domain, group, r = job
        qid = str(r["id"])
        paras = gen_paraphrases(qid, r["question"], K)
        probs = []
        for pidx, para in enumerate(paras[:K]):
            prob, _ = predict_prob(qid, pidx, para)
            if prob is not None:
                probs.append(float(prob))
            time.sleep(0.01)
        return qid, {
            "id": qid,
            "domain": domain,
            "group": group,
            "answer": r["answer"],
            "probs": probs,
            "n_valid": len(probs),
        }

    total = sum(len(v) for d in manifest["domains"].values() for v in d.values())
    complete_before = total - len(jobs)
    print(f"resuming with {complete_before}/{total} usable cached questions; pending={len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(query_one, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            qid, record = future.result()
            out[qid] = record
            if completed % 10 == 0 or completed == len(futures):
                PER_QUESTION.write_text(json.dumps(out, indent=2))
                print(f"queried {complete_before + completed}/{total}", flush=True)
    PER_QUESTION.write_text(json.dumps(out, indent=2))
    print(f"saved {PER_QUESTION}; usable={sum(len(v.get('probs', [])) >= MIN_VALID for v in out.values())}/{total}")


def arrays(domain, group, records, ids=None):
    vals = [
        r for r in records.values()
        if r["domain"] == domain and r["group"] == group and len(r["probs"]) >= MIN_VALID
    ]
    vals = sorted(vals, key=lambda r: r["id"])
    if ids is not None:
        vals = [vals[i] for i in ids]
    probs = [np.asarray(r["probs"], dtype=float) for r in vals]
    y = np.asarray([1.0 if r["answer"] == "yes" else 0.0 for r in vals])
    return probs, y


def flatten_questions(probs, y):
    x, yy, weights = [], [], []
    for p, outcome in zip(probs, y):
        x.extend(p.tolist())
        yy.extend([outcome] * len(p))
        weights.extend([1.0 / len(p)] * len(p))
    return np.asarray(x), np.asarray(yy), np.asarray(weights)


def fit_calibrator(method, probs, y):
    x, yy, weights = flatten_questions(probs, y)
    if method == "platt":
        eps = 1e-8
        z = np.log(np.clip(x, eps, 1 - eps) / (1 - np.clip(x, eps, 1 - eps))).reshape(-1, 1)
        model = LogisticRegression(C=1e4, solver="lbfgs").fit(z, yy, sample_weight=weights)
        return ("platt", model)
    model = IsotonicRegression(out_of_bounds="clip").fit(x, yy, sample_weight=weights)
    return ("isotonic", model)


def apply_calibrator(calibrator, p):
    kind, model = calibrator
    p = np.asarray(p, dtype=float)
    if kind == "platt":
        eps = 1e-8
        z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
        return model.predict_proba(z)[:, 1]
    return model.predict(p)


def prc_metrics(probs, y):
    consensus = np.asarray([p.mean() for p in probs])
    correction = float(np.mean([
        np.var(p, ddof=1) / len(p) if len(p) > 1 else 0.0 for p in probs
    ]))
    raw = float(np.cov(consensus, y - consensus, ddof=1)[0, 1])
    return {
        "raw": raw,
        "finite_k_correction": correction,
        "bc": raw + correction,
        "n": len(y),
        "mean_consensus": float(consensus.mean()),
        "brier": float(np.mean((consensus - y) ** 2)),
    }


def crossfit_calibrate(method, treatment_probs, control_probs, control_y, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    control_cal = [None] * len(control_probs)
    treatment_by_fold = []
    idx = np.arange(len(control_probs))
    for train, test in kf.split(idx):
        cal = fit_calibrator(method, [control_probs[i] for i in train], control_y[train])
        for i in test:
            control_cal[i] = apply_calibrator(cal, control_probs[i])
        treatment_by_fold.append([apply_calibrator(cal, p) for p in treatment_probs])
    treatment_cal = [
        np.mean(np.vstack([fold[i] for fold in treatment_by_fold]), axis=0)
        for i in range(len(treatment_probs))
    ]
    return treatment_cal, control_cal


def estimate_domain(method, treatment_probs, treatment_y, control_probs, control_y, seed):
    treatment_cal, control_cal = crossfit_calibrate(
        method, treatment_probs, control_probs, control_y, seed
    )
    mt = prc_metrics(treatment_cal, treatment_y)
    mc = prc_metrics(control_cal, control_y)
    return {
        "treatment": mt,
        "control": mc,
        "excess_raw": mt["raw"] - mc["raw"],
        "excess_bc": mt["bc"] - mc["bc"],
    }


def bootstrap_domain(method, treatment_probs, treatment_y, control_probs, control_y,
                     reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    draws = {"treatment_bc": [], "control_bc": [], "excess_bc": [], "excess_raw": []}
    for b in range(reps):
        it = rng.integers(0, len(treatment_probs), len(treatment_probs))
        ic = rng.integers(0, len(control_probs), len(control_probs))
        est = estimate_domain(
            method,
            [treatment_probs[i] for i in it],
            treatment_y[it],
            [control_probs[i] for i in ic],
            control_y[ic],
            seed + b + 1,
        )
        draws["treatment_bc"].append(est["treatment"]["bc"])
        draws["control_bc"].append(est["control"]["bc"])
        draws["excess_bc"].append(est["excess_bc"])
        draws["excess_raw"].append(est["excess_raw"])
    return {k: np.asarray(v) for k, v in draws.items()}


def interval(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def analyze():
    if not PER_QUESTION.exists():
        raise FileNotFoundError("Run --query first")
    records = json.loads(PER_QUESTION.read_text())
    output = {
        "protocol": {
            "seed": SEED,
            "K": K,
            "minimum_valid": MIN_VALID,
            "domains": DOMAIN_N,
            "bootstrap_reps": 1000,
        },
        "methods": {},
    }
    pooled_draws = {}
    for method in METHODS:
        output["methods"][method] = {"domains": {}}
        domain_draws = {}
        for di, domain in enumerate(DOMAIN_N):
            tp, ty = arrays(domain, "treatment", records)
            cp, cy = arrays(domain, "control", records)
            point = estimate_domain(method, tp, ty, cp, cy, SEED + di)
            draws = bootstrap_domain(
                method, tp, ty, cp, cy, reps=1000, seed=SEED + 1000 * (di + 1)
            )
            domain_draws[domain] = draws
            output["methods"][method]["domains"][domain] = {
                "point": point,
                "ci": {k: interval(v) for k, v in draws.items()},
                "p_excess_positive": float(np.mean(draws["excess_bc"] > 0)),
            }
        pooled = {
            k: 0.5 * domain_draws["finance"][k] + 0.5 * domain_draws["other"][k]
            for k in domain_draws["finance"]
        }
        pooled_draws[method] = pooled
        output["methods"][method]["pooled"] = {
            "point_excess_bc": float(np.mean([
                output["methods"][method]["domains"][d]["point"]["excess_bc"] for d in DOMAIN_N
            ])),
            "ci": {k: interval(v) for k, v in pooled.items()},
            "p_excess_positive": float(np.mean(pooled["excess_bc"] > 0)),
        }

    platt = output["methods"]["platt"]
    iso = output["methods"]["isotonic"]
    platt_domain_positive = all(
        platt["domains"][d]["point"]["excess_bc"] > 0 for d in DOMAIN_N
    )
    iso_domain_not_reversed = all(
        iso["domains"][d]["point"]["excess_bc"] > -0.002 for d in DOMAIN_N
    )
    output["success_gate"] = {
        "platt_pooled_ci_above_zero": platt["pooled"]["ci"]["excess_bc"][0] > 0,
        "platt_control_ci_contains_zero": all(
            platt["domains"][d]["ci"]["control_bc"][0] <= 0 <=
            platt["domains"][d]["ci"]["control_bc"][1] for d in DOMAIN_N
        ),
        "platt_both_domain_points_positive": platt_domain_positive,
        "isotonic_same_direction": (
            iso["pooled"]["point_excess_bc"] > 0 and iso_domain_not_reversed
        ),
        "finite_k_not_sole_source": platt["pooled"]["ci"]["excess_raw"][0] > 0,
    }
    output["success_gate"]["passed"] = all(output["success_gate"].values())
    RESULTS.write_text(json.dumps(output, indent=2))
    make_plot(output)
    print(json.dumps(output["success_gate"], indent=2))
    for method in METHODS:
        print(f"\n{method.upper()}")
        for domain, r in output["methods"][method]["domains"].items():
            p = r["point"]
            print(
                f"  {domain}: treatment={p['treatment']['bc']:+.4f} "
                f"control={p['control']['bc']:+.4f} excess={p['excess_bc']:+.4f} "
                f"CI{r['ci']['excess_bc']}"
            )
        pooled = output["methods"][method]["pooled"]
        print(f"  pooled excess={pooled['point_excess_bc']:+.4f} CI{pooled['ci']['excess_bc']}")


def make_plot(output):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    domains = list(DOMAIN_N)
    x = np.arange(len(domains))
    width = 0.35
    for mi, method in enumerate(METHODS):
        vals = [output["methods"][method]["domains"][d]["point"]["excess_bc"] for d in domains]
        cis = [output["methods"][method]["domains"][d]["ci"]["excess_bc"] for d in domains]
        err = [
            [vals[i] - cis[i][0] for i in range(len(vals))],
            [cis[i][1] - vals[i] for i in range(len(vals))],
        ]
        ax[0].bar(x + (mi - 0.5) * width, vals, width, yerr=err, capsize=4, label=method)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(domains)
    ax[0].set_ylabel("treatment − control PRC")
    ax[0].set_title("(a) Domain-matched PRC excess")
    ax[0].legend()

    labels, vals, lo, hi = [], [], [], []
    for method in METHODS:
        r = output["methods"][method]["pooled"]
        labels.append(method)
        vals.append(r["point_excess_bc"])
        lo.append(r["point_excess_bc"] - r["ci"]["excess_bc"][0])
        hi.append(r["ci"]["excess_bc"][1] - r["point_excess_bc"])
    ax[1].bar(labels, vals, yerr=[lo, hi], capsize=5)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("pooled treatment − control PRC")
    ax[1].set_title("(b) Calibration sensitivity")
    fig.tight_layout()
    fig.savefig(HERE / "prc_matched.png", dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    make_manifest()
    if args.query:
        query()
    if args.analyze:
        analyze()
    if not args.query and not args.analyze:
        parser.error("Choose --query and/or --analyze")


if __name__ == "__main__":
    main()
