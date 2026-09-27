"""
E1: synthetic validation of the leakage estimator against known ground truth.
Four results, each tied to a contribution:
  T1 Recovery + CI coverage (C1,C5): estimator recovers injected leakage, ~95% coverage.
  T2 RD recovery + placebo (C3): jump tracks injected boundary leakage; no-leak placebo ~0.
  T3 Complementarity (C4): Delta_PRC = w(1-w)sigma^2 (->0 at w=1) vs L = b0 w(2-w) (max at w=1).
  T4 Calibration requirement (C5): calibration suppresses overconfidence bias when enough
     leakage-free anchors are available; residual calibration error is visible at small n_cal.
Outputs: results.json, m1.png, analysis.md.  Self-contained (numpy, scipy, matplotlib).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(0)


# ---------- shared estimator pieces ----------
def fit_temp(P, Y):
    p = np.clip(P, 1e-12, 1 - 1e-12); z = np.log(p / (1 - p))
    def nll(T):
        pc = np.clip(1 / (1 + np.exp(-z / T)), 1e-6, 1 - 1e-6)
        return -np.mean(Y * np.log(pc) + (1 - Y) * np.log(1 - pc))
    return minimize_scalar(nll, bounds=(0.3, 20.0), method="bounded").x

def apply_temp(P, T):
    p = np.clip(P, 1e-12, 1 - 1e-12); z = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-z / T))

def gen(n, w, s, leaked, Tover, rng):
    c0 = rng.uniform(0.15, 0.85, n)
    Y = (rng.uniform(size=n) < c0).astype(float)
    Phon = np.clip(c0 + s * (Y - c0) + rng.normal(0, 0.05, n), 1e-3, 1 - 1e-3)
    Pl = np.clip((1 - w) * Phon + w * Y, 1e-3, 1 - 1e-3) if leaked else Phon
    z = np.log(Pl / (1 - Pl)); Pobs = 1 / (1 + np.exp(-z * Tover))
    return dict(c0=c0, Y=Y, Phon=Phon, Pl=Pl, Pobs=Pobs)

def bgap(P, c0, Y):  # Brier reduction vs crowd
    return np.mean((c0 - Y) ** 2 - (P - Y) ** 2)


# ---------- T1: recovery + coverage ----------
def T1():
    res = []
    for w in [0.0, 0.2, 0.4, 0.7, 1.0]:
        biases, covers, trues = [], [], []
        for s in range(300):
            rng = np.random.default_rng(s)
            lk = gen(400, w, 0.3, True, 1.0, rng); cl = gen(400, 0.0, 0.3, False, 1.0, rng)
            Bt = np.mean((lk["Phon"] - lk["Y"]) ** 2 - (lk["Pl"] - lk["Y"]) ** 2)
            Bhat = bgap(lk["Pl"], lk["c0"], lk["Y"]) - bgap(cl["Pl"], cl["c0"], cl["Y"])
            bs = []
            for _ in range(200):
                a = lk["Pl"]; b = cl["Pl"]
                ia = rng.integers(0, 400, 400); ib = rng.integers(0, 400, 400)
                bs.append(bgap(a[ia], lk["c0"][ia], lk["Y"][ia]) - bgap(b[ib], cl["c0"][ib], cl["Y"][ib]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            biases.append(Bhat - Bt); covers.append(lo <= Bt <= hi); trues.append(Bt)
        res.append(dict(w=w, true_B=float(np.mean(trues)), bias=float(np.mean(biases)), coverage=float(np.mean(covers))))
    return res


# ---------- T2: RD recovery + placebo ----------
def local_jump(delta, b, h):
    m = np.abs(delta) <= h; d, bb = delta[m], b[m]; left = (d < 0).astype(float)
    w = np.maximum(0, 1 - np.abs(d) / h); sw = np.sqrt(w)
    A = np.column_stack([np.ones_like(d), left, d, d * left])
    coef, *_ = np.linalg.lstsq(A * sw[:, None], bb * sw, rcond=None)
    return -coef[1]

def gen_rd(n, w0, leak, rng):
    delta = rng.uniform(-180, 180, n)
    c0 = rng.uniform(0.15, 0.85, n); Y = (rng.uniform(size=n) < c0).astype(float)
    s = 0.4 - 0.25 * np.abs(delta) / 180.0
    Phon = np.clip(c0 + s * (Y - c0) + rng.normal(0, 0.05, n), 1e-3, 1 - 1e-3)
    wv = np.where((delta < 0) & leak, w0, 0.0)
    P = np.clip((1 - wv) * Phon + wv * Y, 1e-3, 1 - 1e-3)
    return delta, (P - Y) ** 2

def T2():
    res = []
    for w0 in [0.0, 0.3, 0.6]:
        jl, jp = [], []
        for s in range(300):
            rng = np.random.default_rng(s)
            d, b = gen_rd(1500, w0, True, rng); jl.append(local_jump(d, b, 120))
            dp, bp = gen_rd(1500, w0, False, rng); jp.append(local_jump(dp, bp, 120))
        res.append(dict(w0=w0, jump_leak=float(np.mean(jl)), jump_placebo=float(np.mean(jp))))
    return res


# ---------- T3: complementarity ----------
def T3():
    res = []; K = 12
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rng = np.random.default_rng(1); n = 8000
        c0 = rng.uniform(0.15, 0.85, n); Y = (rng.uniform(size=n) < c0).astype(float)
        ybar = np.zeros(n)
        for _ in range(K):
            e = rng.normal(0, 0.12, n)
            Pk = np.clip((1 - w) * np.clip(c0 + e, 0, 1) + w * Y, 0, 1); ybar += Pk / K
        Delta = float(np.cov(ybar, Y - ybar)[0, 1]); L = float(np.mean((c0 - Y) ** 2 - (ybar - Y) ** 2))
        res.append(dict(w=w, Delta_PRC=Delta, leakage_L=L))
    return res


# ---------- T4: overconfidence robustness ----------
def T4():
    """Calibration sample-size sweep under overconfidence.

    The previous version fit temperature on the same clean sample used in the DiD contrast and
    reported a single residual bias. This version uses an independent leakage-free calibration
    set and varies n_cal, which tests the actual practical requirement: enough clean anchors.

    This sub-test intentionally uses a calibrated clean forecaster by construction:
    c0 = P_hon = P(Y=1 | information). Overconfidence is then a pure temperature distortion
    of the emitted probabilities, so temperature scaling is correctly specified. This isolates
    the calibration question from the separate question of whether an LLM's clean forecasts are
    calibratable by one global temperature.
    """
    def gen_calibrated(n, w, leaked, Tover, rng):
        c0 = rng.uniform(0.15, 0.85, n)
        Y = (rng.uniform(size=n) < c0).astype(float)
        Phon = c0.copy()
        Pl = np.clip((1 - w) * Phon + w * Y, 1e-3, 1 - 1e-3) if leaked else Phon
        z = np.log(Pl / (1 - Pl))
        Pobs = 1 / (1 + np.exp(-z * Tover))
        return dict(c0=c0, Y=Y, Phon=Phon, Pl=Pl, Pobs=Pobs)

    rows = []
    n_eval = 600
    n_reps = 200
    for Tover in [1.0, 2.0, 4.0, 8.0]:
        for n_cal in [100, 250, 500, 1000, 2000, 5000]:
            nb, cb, tb, temp = [], [], [], []
            for s in range(n_reps):
                rng = np.random.default_rng(10_000 * int(Tover) + n_cal + s)
                tr = gen_calibrated(n_eval, 0.8, True, Tover, rng)
                ct = gen_calibrated(n_eval, 0.0, False, Tover, rng)
                calset = gen_calibrated(n_cal, 0.0, False, Tover, rng)

                Bt = np.mean((tr["Phon"] - tr["Y"]) ** 2 - (tr["Pl"] - tr["Y"]) ** 2)
                naive = bgap(tr["Pobs"], tr["c0"], tr["Y"]) - bgap(ct["Pobs"], ct["c0"], ct["Y"])

                T = fit_temp(calset["Pobs"], calset["Y"])
                calibrated = (
                    bgap(apply_temp(tr["Pobs"], T), tr["c0"], tr["Y"])
                    - bgap(apply_temp(ct["Pobs"], T), ct["c0"], ct["Y"])
                )
                nb.append(naive - Bt)
                cb.append(calibrated - Bt)
                tb.append(Bt)
                temp.append(T)

            nb = np.asarray(nb)
            cb = np.asarray(cb)
            rows.append(dict(
                Tover=Tover,
                n_cal=n_cal,
                true_B=float(np.mean(tb)),
                fitted_T=float(np.mean(temp)),
                naive_bias=float(np.mean(nb)),
                calibrated_bias=float(np.mean(cb)),
                naive_abs_bias=float(np.mean(np.abs(nb))),
                calibrated_abs_bias=float(np.mean(np.abs(cb))),
                naive_rmse=float(np.sqrt(np.mean(nb ** 2))),
                calibrated_rmse=float(np.sqrt(np.mean(cb ** 2))),
            ))
    return rows


# ---------- T5: no free inflation (outcome-orthogonal noise cannot fake leakage) ----------
def T5():
    """Corollary 'no free inflation': perturbations independent of Y must not inflate the score.

    'Leaked'-side forecasts are the honest forecasts perturbed by zero-mean logit-space noise
    independent of Y (logit-space avoids clipping-induced dependence at the boundaries). True
    leakage is exactly zero. The same Brier-reduction DiD estimator is applied. Expectation:
    B-hat <= 0 at every noise scale (noise strictly deflates the contrast).
    """
    res = []
    for sig in [0.25, 0.5, 1.0]:
        bhats = []
        for s in range(300):
            rng = np.random.default_rng(50_000 + s)
            tr = gen(400, 0.0, 0.3, False, 1.0, rng)   # honest, no leakage
            ct = gen(400, 0.0, 0.3, False, 1.0, rng)   # clean control
            z = np.log(tr["Pl"] / (1 - tr["Pl"])) + rng.normal(0, sig, 400)
            Pnoise = 1 / (1 + np.exp(-z))
            bhats.append(bgap(Pnoise, tr["c0"], tr["Y"]) - bgap(ct["Pl"], ct["c0"], ct["Y"]))
        bhats = np.asarray(bhats)
        se = float(np.std(bhats, ddof=1) / np.sqrt(len(bhats)))
        res.append(dict(noise_sigma=sig,
                        Bhat_mean=float(np.mean(bhats)),
                        Bhat_mean_ci_lo=float(np.mean(bhats) - 1.96 * se),
                        Bhat_mean_ci_hi=float(np.mean(bhats) + 1.96 * se),
                        Bhat_lo=float(np.percentile(bhats, 2.5)),
                        Bhat_hi=float(np.percentile(bhats, 97.5)),
                        frac_positive=float(np.mean(bhats > 0))))
    return res


# ---------- T6: concentration across crowd-surprise strata ----------
def T6():
    """Stakes-times-extraction concentration: at fixed w, the stratified DiD estimate should
    track E[b0 w(2-w) | stratum] across crowd-surprise strata (observable (c0-Y)^2 quartiles).
    """
    w = 0.5
    reps = 300
    strat_obs = np.zeros((reps, 4))
    strat_pred = np.zeros((reps, 4))
    for s in range(reps):
        rng = np.random.default_rng(60_000 + s)
        lk = gen(4000, w, 0.3, True, 1.0, rng)
        cl = gen(4000, 0.0, 0.3, False, 1.0, rng)
        surprise_lk = (lk["c0"] - lk["Y"]) ** 2
        surprise_cl = (cl["c0"] - cl["Y"]) ** 2
        qs = np.quantile(np.concatenate([surprise_lk, surprise_cl]), [0.25, 0.5, 0.75])
        for j in range(4):
            lo = -np.inf if j == 0 else qs[j - 1]
            hi = np.inf if j == 3 else qs[j]
            ml = (surprise_lk > lo) & (surprise_lk <= hi)
            mc = (surprise_cl > lo) & (surprise_cl <= hi)
            gl = np.mean((lk["c0"][ml] - lk["Y"][ml]) ** 2 - (lk["Pl"][ml] - lk["Y"][ml]) ** 2)
            gc = np.mean((cl["c0"][mc] - cl["Y"][mc]) ** 2 - (cl["Pl"][mc] - cl["Y"][mc]) ** 2)
            strat_obs[s, j] = gl - gc
            b0 = (lk["Phon"][ml] - lk["Y"][ml]) ** 2
            strat_pred[s, j] = np.mean(b0 * w * (2 - w))
    res = []
    for j in range(4):
        res.append(dict(stratum=j + 1,
                        observed_mean=float(np.mean(strat_obs[:, j])),
                        observed_lo=float(np.percentile(strat_obs[:, j], 2.5)),
                        observed_hi=float(np.percentile(strat_obs[:, j], 97.5)),
                        predicted_mean=float(np.mean(strat_pred[:, j]))))
    return res


def functional_form(t3):
    """L(w)/L(1) vs the predicted w(2-w) from the T3 sweep."""
    L1 = [r["leakage_L"] for r in t3 if r["w"] == 1.0][0]
    return [dict(w=r["w"],
                 ratio_observed=float(r["leakage_L"] / L1),
                 ratio_predicted=float(r["w"] * (2 - r["w"])))
            for r in t3]


def main():
    t3 = T3()
    results = dict(T1_recovery_coverage=T1(), T2_rd_placebo=T2(),
                   T3_complementarity=t3, T4_overconfidence=T4(),
                   T5_no_free_inflation=T5(), T6_concentration=T6(),
                   T3_functional_form=functional_form(t3))
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    t1 = results["T1_recovery_coverage"]
    ax[0,0].plot([r["true_B"] for r in t1], [r["true_B"]+r["bias"] for r in t1], "o-", label="estimated")
    ax[0,0].plot([r["true_B"] for r in t1], [r["true_B"] for r in t1], "k--", label="truth")
    covs = [r["coverage"] for r in t1]
    ax[0,0].set_title(f"T1 recovery (coverage {min(covs):.0%}\u2013{max(covs):.0%})"); ax[0,0].set_xlabel("true B"); ax[0,0].set_ylabel("estimated B"); ax[0,0].legend()
    t2 = results["T2_rd_placebo"]
    x = np.arange(len(t2)); ax[0,1].bar(x-0.2,[r["jump_leak"] for r in t2],0.4,label="leak"); ax[0,1].bar(x+0.2,[r["jump_placebo"] for r in t2],0.4,label="placebo")
    ax[0,1].set_xticks(x); ax[0,1].set_xticklabels([f"w0={r['w0']}" for r in t2]); ax[0,1].set_title("T2 RD jump vs placebo"); ax[0,1].legend()
    t3 = results["T3_complementarity"]
    ax[1,0].plot([r["w"] for r in t3],[r["Delta_PRC"] for r in t3],"o-",label="Δ_PRC"); ax[1,0].plot([r["w"] for r in t3],[r["leakage_L"] for r in t3],"s-",label="L (RD/score)")
    ax[1,0].set_title("T3 complementarity"); ax[1,0].set_xlabel("w (extraction)"); ax[1,0].set_ylabel("population signal"); ax[1,0].legend()
    t4 = results["T4_overconfidence"]
    for i, (Tover, marker) in enumerate([(2.0, "o"), (4.0, "s"), (8.0, "^")]):
        rr = [r for r in t4 if r["Tover"] == Tover]
        ax[1,1].plot([r["n_cal"] for r in rr], [r["calibrated_abs_bias"] for r in rr],
                     marker + "-", label=f"calibrated T={Tover:g}")
        ax[1,1].hlines(rr[0]["naive_abs_bias"], rr[0]["n_cal"], rr[-1]["n_cal"],
                       colors="gray", linestyles="dotted", linewidth=1,
                       label="naive (uncalibrated)" if i == 0 else None)
    ax[1,1].set_xscale("log")
    ax[1,1].set_title("T4 calibration needs clean anchors")
    ax[1,1].set_xlabel("calibration sample size")
    ax[1,1].set_ylabel("mean |bias|")
    ax[1,1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "m1.png", dpi=130)

    # supplementary figure: T5 + T6
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
    t5 = results["T5_no_free_inflation"]
    xs = [r["noise_sigma"] for r in t5]
    means = [r["Bhat_mean"] for r in t5]
    los = [r["Bhat_mean"] - r["Bhat_lo"] for r in t5]
    his = [r["Bhat_hi"] - r["Bhat_mean"] for r in t5]
    ax2[0].errorbar(xs, means, yerr=[los, his], fmt="o-", capsize=4, color="tab:blue")
    ax2[0].axhline(0, color="k", linestyle="--", linewidth=1)
    ax2[0].set_title("T5: outcome-orthogonal noise, zero true leakage")
    ax2[0].set_xlabel("logit-noise scale $\\sigma$")
    ax2[0].set_ylabel("estimated $\\widehat{B}$")
    t6 = results["T6_concentration"]
    xs6 = np.arange(len(t6))
    obs = [r["observed_mean"] for r in t6]
    lo6 = [r["observed_mean"] - r["observed_lo"] for r in t6]
    hi6 = [r["observed_hi"] - r["observed_mean"] for r in t6]
    pred = [r["predicted_mean"] for r in t6]
    ax2[1].bar(xs6, obs, 0.55, yerr=[lo6, hi6], capsize=4, label="stratified DiD estimate", color="tab:blue", alpha=0.75)
    ax2[1].plot(xs6, pred, "k^--", label="predicted $\\mathbb{E}[b_0 w(2-w)]$")
    ax2[1].set_xticks(xs6)
    ax2[1].set_xticklabels([f"Q{r['stratum']}" for r in t6])
    ax2[1].set_title("T6: leakage concentrates in surprising strata ($w=0.5$)")
    ax2[1].set_xlabel("crowd-surprise quartile")
    ax2[1].set_ylabel("leakage per question")
    ax2[1].legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(OUT / "e1_supp.png", dpi=130)
    print("saved results.json + m1.png + e1_supp.png")
    for k, v in results.items():
        print(f"\n{k}:"); [print("  ", r) for r in v]


if __name__ == "__main__":
    main()
