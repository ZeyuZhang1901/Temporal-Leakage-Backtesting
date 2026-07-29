"""
M2 (main): Hubble controlled-contamination validation. Zero-compute
(uses published per-(model,task,dup) accuracy in data/hubble_testset.csv).

Standard model = minimal-pair clean control. Perturbed = contaminated at known dup rates.
  B_twin(r) = acc_pert(dup=r) - acc_std(dup=r)        [perfect-twin contrast]
  B_self(r) = acc_pert(dup=r) - acc_pert(dup=0)       [no twin: own clean examples]

Reviewer-facing diagnostics:
  * raw and placebo-centered twin contrasts (handles small r=0 model-pair offset)
  * approximate aggregate uncertainty from published SEMs
  * monotone dose-response trend test
  * robustness across scale/token-count pairs
  * per-task consistency and standard-model flatness checks
"""
import csv, itertools, json, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[2]
CSV = EXP / "data" / "hubble_testset.csv"
OUT = Path(__file__).resolve().parent
MCQ = ["mmlu", "piqa", "hellaswag", "winogrande_mcq"]
DUPS = [0, 1, 4, 16, 64, 256]
PAIRS = [("8b", "500b"), ("8b", "100b"), ("1b", "500b")]


def load():
    d = {}
    for r in csv.DictReader(open(CSV)):
        try: d[(r["model"], r["task"], int(r["duplicates"]), r["metric"])] = (float(r["mean"]), float(r["sem"]))
        except ValueError: pass
    return d

def acc(d, model, task, dup):
    for m in ("acc", "acc_norm", "exact_match"):
        if (model, task, dup, m) in d: return d[(model, task, dup, m)]
    return None


def analyze_pair(d, size, toks):
    std = f"allegrolab__hubble-{size}-{toks}_toks-standard-hf"
    pert = f"allegrolab__hubble-{size}-{toks}_toks-perturbed-hf"
    out = {"tasks": {}, "std_flatness": {}}
    for task in MCQ:
        p0 = acc(d, pert, task, 0); s0 = acc(d, std, task, 0)
        if p0 is None or s0 is None: continue
        twin, selfb, stds = {}, {}, []
        for r in DUPS:
            pr = acc(d, pert, task, r); sr = acc(d, std, task, r)
            if pr is None or sr is None: continue
            twin[r] = (pr[0] - sr[0], math.sqrt(pr[1] ** 2 + sr[1] ** 2))
            selfb[r] = (pr[0] - p0[0], math.sqrt(pr[1] ** 2 + p0[1] ** 2))
            stds.append(sr[0])
        out["tasks"][task] = {"B_twin": twin, "B_self": selfb}
        out["std_flatness"][task] = float(np.ptp(stds))  # range of std acc across dup (should be small)
    return out


def aggregate_pair(pair_result):
    """Unweighted task aggregate with approximate SEM propagation.

    The input entries are per-task means with published SEMs. For each duplication
    rate we average over tasks and propagate SEMs as sqrt(sum(se_i^2))/num_tasks.
    For placebo-centered twin estimates, we subtract each task's r=0 twin offset
    before aggregating; SEMs are conservatively combined assuming independence.
    """
    out = {
        "raw_twin": {},
        "placebo_centered_twin": {},
        "self": {},
        "self_minus_raw_twin": {},
        "num_tasks": {},
    }
    tasks = pair_result["tasks"]
    for r in DUPS:
        raw_vals, raw_vars = [], []
        centered_vals, centered_vars = [], []
        self_vals, self_vars = [], []
        diffs = []
        for _, t in tasks.items():
            key = r
            zero_key = 0
            if key not in t["B_twin"] or zero_key not in t["B_twin"]:
                continue
            bt, se_t = t["B_twin"][key]
            b0, se_0 = t["B_twin"][zero_key]
            bs, se_s = t["B_self"][key]
            raw_vals.append(bt)
            raw_vars.append(se_t ** 2)
            centered_vals.append(bt - b0)
            centered_vars.append(se_t ** 2 + se_0 ** 2)
            self_vals.append(bs)
            self_vars.append(se_s ** 2)
            diffs.append(bs - bt)
        if not raw_vals:
            continue
        n = len(raw_vals)
        out["raw_twin"][str(r)] = [float(np.mean(raw_vals)), float(math.sqrt(sum(raw_vars)) / n)]
        out["placebo_centered_twin"][str(r)] = [
            float(np.mean(centered_vals)),
            float(math.sqrt(sum(centered_vars)) / n),
        ]
        out["self"][str(r)] = [float(np.mean(self_vals)), float(math.sqrt(sum(self_vars)) / n)]
        out["self_minus_raw_twin"][str(r)] = float(np.mean(diffs))
        out["num_tasks"][str(r)] = n
    return out


def spearman_rho(xs, ys):
    """Spearman rho with average ranks for small arrays."""
    def ranks(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = rank
            i = j + 1
        return np.asarray(r)
    rx, ry = ranks(xs), ranks(ys)
    return float(np.corrcoef(rx, ry)[0, 1])


def trend_test(agg, series_name="placebo_centered_twin", exclude_zero=True):
    keys = [r for r in DUPS if str(r) in agg[series_name]]
    if exclude_zero:
        keys = [r for r in keys if r > 0]
    x = np.log1p(np.asarray(keys, dtype=float))
    y = np.asarray([agg[series_name][str(r)][0] for r in keys], dtype=float)
    rho = spearman_rho(list(x), list(y))
    # Exact one-sided permutation p-value over y assignments (small n).
    count = 0
    total = 0
    for yp in itertools.permutations(y):
        total += 1
        if spearman_rho(list(x), list(yp)) >= rho - 1e-12:
            count += 1
    monotone_nondec = all(y[i] <= y[i + 1] + 1e-12 for i in range(len(y) - 1))
    return {
        "duplications": keys,
        "rho": rho,
        "one_sided_permutation_p": count / total,
        "monotone_non_decreasing": monotone_nondec,
    }


def main():
    d = load()
    results = {}
    for size, toks in PAIRS:
        key = f"{size}-{toks}"
        results[key] = analyze_pair(d, size, toks)
        results[key]["aggregate"] = aggregate_pair(results[key])
        results[key]["trend_placebo_centered"] = trend_test(results[key]["aggregate"], "placebo_centered_twin")
        results[key]["trend_self"] = trend_test(results[key]["aggregate"], "self")
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))

    head = results["8b-500b"]
    print("=== M2 Hubble 8B-500B (headline) ===")
    print("control-degradation: B_twin (perfect control) vs B_self (own dup=0 baseline, NO twin)")
    # aggregate across MCQ tasks
    agg_twin = {r: [] for r in DUPS}; agg_self = {r: [] for r in DUPS}
    for task, t in head["tasks"].items():
        bt = {r: v[0] for r, v in t["B_twin"].items()}
        bs = {r: v[0] for r, v in t["B_self"].items()}
        print(f"  {task:15s} twin: " + " ".join(f"r{r}:{bt.get(r,float('nan')):+.2f}" for r in DUPS))
        print(f"  {'':15s} self: " + " ".join(f"r{r}:{bs.get(r,float('nan')):+.2f}" for r in DUPS))
        for r in DUPS:
            if r in bt: agg_twin[r].append(bt[r]); agg_self[r].append(bs[r])
    print("  std-acc flatness across dup (range; small => random assignment, no difficulty confound):")
    print("   ", {k: round(v, 3) for k, v in head["std_flatness"].items()})
    mt = {r: results["8b-500b"]["aggregate"]["raw_twin"][str(r)][0] for r in DUPS}
    mt_se = {r: results["8b-500b"]["aggregate"]["raw_twin"][str(r)][1] for r in DUPS}
    mc = {r: results["8b-500b"]["aggregate"]["placebo_centered_twin"][str(r)][0] for r in DUPS}
    mc_se = {r: results["8b-500b"]["aggregate"]["placebo_centered_twin"][str(r)][1] for r in DUPS}
    ms = {r: results["8b-500b"]["aggregate"]["self"][str(r)][0] for r in DUPS}
    ms_se = {r: results["8b-500b"]["aggregate"]["self"][str(r)][1] for r in DUPS}
    print(f"\n  AGGREGATE B_twin: " + " ".join(f"r{r}:{mt[r]:+.3f}" for r in mt))
    print(f"  PLACEBO-CENTERED B_twin: " + " ".join(f"r{r}:{mc[r]:+.3f}" for r in mc))
    print(f"  AGGREGATE B_self: " + " ".join(f"r{r}:{ms[r]:+.3f}" for r in ms))
    print(f"  placebo B_twin(0)={mt.get(0):+.3f} ; max diff |twin-self| over r>0 = "
          f"{max(abs(mt[r]-ms[r]) for r in mt if r>0):.3f}  (small => recover WITHOUT twin)")
    print("  trend placebo-centered:",
          results["8b-500b"]["trend_placebo_centered"])
    print("  trend self:", results["8b-500b"]["trend_self"])

    # figure: aggregate dose-response, robustness, task consistency, placebo flatness
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    xs = [r for r in DUPS if r in mt]
    xplot = [max(x, 0.5) for x in xs]
    ax[0,0].errorbar(xplot, [mt[r] for r in xs], yerr=[1.96 * mt_se[r] for r in xs],
                     fmt="o-", capsize=3, label="raw minimal-pair")
    ax[0,0].errorbar(xplot, [mc[r] for r in xs], yerr=[1.96 * mc_se[r] for r in xs],
                     fmt="^-", capsize=3, label="placebo-centered")
    ax[0,0].errorbar(xplot, [ms[r] for r in xs], yerr=[1.96 * ms_se[r] for r in xs],
                     fmt="s--", capsize=3, label="self-baseline")
    ax[0,0].axhline(0, color="gray", lw=.8)
    ax[0,0].set_xscale("log")
    ax[0,0].set_xlabel("duplication rate $r$")
    ax[0,0].set_ylabel("estimated score inflation")
    ax[0,0].set_title("(a) Dose-response (8B, 500B)")
    ax[0,0].legend(fontsize=8)

    for pair, marker in [("8b-500b", "o"), ("8b-100b", "s"), ("1b-500b", "^")]:
        agg = results[pair]["aggregate"]["placebo_centered_twin"]
        ax[0,1].plot(xplot, [agg[str(r)][0] for r in xs], marker + "-", label=pair)
    ax[0,1].axhline(0, color="gray", lw=.8)
    ax[0,1].set_xscale("log")
    ax[0,1].set_xlabel("duplication rate $r$")
    ax[0,1].set_ylabel("placebo-centered estimate")
    ax[0,1].set_title("(b) Robustness across scale/token count")
    ax[0,1].legend(fontsize=8)

    for task, t in head["tasks"].items():
        y = [t["B_twin"][r][0] - t["B_twin"][0][0] for r in xs]
        ax[1,0].plot(xplot, y, "o-", label=task.replace("_mcq", ""))
    ax[1,0].axhline(0, color="gray", lw=.8)
    ax[1,0].set_xscale("log")
    ax[1,0].set_xlabel("duplication rate $r$")
    ax[1,0].set_ylabel("placebo-centered task estimate")
    ax[1,0].set_title("(c) Per-task consistency (8B, 500B)")
    ax[1,0].legend(fontsize=8)

    tasks = list(head["std_flatness"].keys())
    ax[1,1].bar(range(len(tasks)), [head["std_flatness"][t] for t in tasks])
    ax[1,1].set_xticks(range(len(tasks)))
    ax[1,1].set_xticklabels([t.replace("_mcq", "") for t in tasks], rotation=30, ha="right")
    ax[1,1].set_title("(d) Clean-control accuracy range across $r$")
    ax[1,1].set_ylabel("accuracy range")
    fig.tight_layout()
    fig.savefig(OUT / "m2.png", dpi=150)
    print("\nsaved results.json + m2.png")


if __name__ == "__main__":
    main()
