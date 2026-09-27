#!/usr/bin/env python3
"""Regenerate the appendix figures with the unified paper style.

Reads stored result JSONs only (no recomputation):
  synthetic/results.json            -> m1 (E1 synthetic), e1_supp
  livecodebench/e3_robustness.json  -> e3_robustness (GPT-4-Turbo dropped)
  livecodebench/e3_global_sensitivity.json -> e3_global_sensitivity
  prc_matched/results.json          -> prc_matched
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (BLUE, GREEN, GREY, ORANGE, PURPLE, RED, ROOT,
                         apply_style, panel_letter, save)

apply_style()
import matplotlib.pyplot as plt

RESULTS = ROOT.parent


# ---- E1 synthetic: main figure (T1-T4) --------------------------------------
def fig_e1():
    res = json.loads((RESULTS / "synthetic" / "results.json").read_text())
    fig, ax = plt.subplots(2, 2, figsize=(6.0, 4.6))

    a = ax[0, 0]
    t1 = res["T1_recovery_coverage"]
    a.plot([r["true_B"] for r in t1], [r["true_B"] + r["bias"] for r in t1],
           "o-", color=BLUE, label="estimated")
    a.plot([r["true_B"] for r in t1], [r["true_B"] for r in t1],
           "k--", lw=1, label="truth")
    a.set_xlabel("true $B$")
    a.set_ylabel("estimated $B$")
    a.legend(handlelength=1.6)
    panel_letter(a, "a")

    a = ax[0, 1]
    t2 = res["T2_rd_placebo"]
    x = np.arange(len(t2))
    a.bar(x - 0.2, [r["jump_leak"] for r in t2], 0.4, color=BLUE,
          label="leak boundary")
    a.bar(x + 0.2, [r["jump_placebo"] for r in t2], 0.4, color=GREY,
          label="placebo")
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xticks(x)
    a.set_xticklabels([f"$w_0$={r['w0']}" for r in t2])
    a.set_ylabel("RD jump")
    a.legend(handlelength=1.2)
    panel_letter(a, "b")

    a = ax[1, 0]
    t3 = res["T3_complementarity"]
    a.plot([r["w"] for r in t3], [r["Delta_PRC"] for r in t3], "o-",
           color=BLUE, label=r"$\Delta$PRC")
    a.plot([r["w"] for r in t3], [r["leakage_L"] for r in t3], "s-",
           color=RED, label="$L$ (RD/score)")
    a.set_xlabel("$w$ (extraction)")
    a.set_ylabel("population signal")
    a.legend(handlelength=1.6)
    panel_letter(a, "c")

    a = ax[1, 1]
    t4 = res["T4_overconfidence"]
    for i, (tover, marker, col) in enumerate(
            [(2.0, "o", BLUE), (4.0, "s", RED), (8.0, "^", GREEN)]):
        rr = [r for r in t4 if r["Tover"] == tover]
        a.plot([r["n_cal"] for r in rr],
               [r["calibrated_abs_bias"] for r in rr],
               marker + "-", color=col, label=f"calibrated $T$={tover:g}")
        a.hlines(rr[0]["naive_abs_bias"], rr[0]["n_cal"], rr[-1]["n_cal"],
                 colors="gray", linestyles="dotted", lw=1,
                 label="uncalibrated" if i == 0 else None)
    a.set_xscale("log")
    a.set_xlabel("calibration sample size")
    a.set_ylabel("mean $|$bias$|$")
    a.legend(handlelength=1.6)
    panel_letter(a, "d")
    save(fig, "m1")


# ---- E1 synthetic: supplementary (T5-T6) ------------------------------------
def fig_e1_supp():
    res = json.loads((RESULTS / "synthetic" / "results.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.6))

    a = ax[0]
    t5 = res["T5_no_free_inflation"]
    xs = [r["noise_sigma"] for r in t5]
    means = [r["Bhat_mean"] for r in t5]
    los = [r["Bhat_mean"] - r["Bhat_lo"] for r in t5]
    his = [r["Bhat_hi"] - r["Bhat_mean"] for r in t5]
    a.errorbar(xs, means, yerr=[los, his], fmt="o-", color=BLUE)
    a.axhline(0, color="0.35", ls="--", lw=1)
    a.set_xlabel(r"logit-noise scale $\sigma$")
    a.set_ylabel(r"estimated $\widehat B$")
    panel_letter(a, "a")

    a = ax[1]
    t6 = res["T6_concentration"]
    xs6 = np.arange(len(t6))
    obs = [r["observed_mean"] for r in t6]
    lo6 = [r["observed_mean"] - r["observed_lo"] for r in t6]
    hi6 = [r["observed_hi"] - r["observed_mean"] for r in t6]
    pred = [r["predicted_mean"] for r in t6]
    a.bar(xs6, obs, 0.55, yerr=[lo6, hi6], color=BLUE, alpha=0.8,
          label="stratified DiD", error_kw=dict(ecolor="0.35", lw=1))
    a.plot(xs6, pred, "k^--", lw=1,
           label=r"predicted $\mathbb{E}[b_0 w(2-w)]$")
    a.set_xticks(xs6)
    a.set_xticklabels([f"Q{r['stratum']}" for r in t6])
    a.set_xlabel("crowd-surprise quartile")
    a.set_ylabel("leakage per question")
    a.legend(handlelength=1.4)
    panel_letter(a, "b")
    save(fig, "e1_supp")


# ---- M4-C robustness battery (GPT-4-Turbo dropped) --------------------------
TARGET_LABEL = {"gpt4o_240513": "4o-0513", "gpt4o_240806": "4o-0806",
                "claude35_240620": "Claude-3.5"}
CONTROL_LABEL = {"gemini25pro_0506": "Gemini-2.5-Pro",
                 "deepseekr1_0528": "DeepSeek-R1", "clean_pool": "clean pool"}


def fig_e3_robustness():
    out = json.loads(
        (RESULTS / "livecodebench" / "e3_robustness.json").read_text())
    targets = list(TARGET_LABEL)
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 5.4))

    a = ax[0, 0]
    cuts = ["Oct2023", "Dec2023", "Apr2024"]
    x, y, lo, hi, labels = [], [], [], [], []
    for i, control in enumerate(CONTROL_LABEL):
        for j, cut in enumerate(cuts):
            r = out["control_smoothness"][control][cut]
            x.append(i * (len(cuts) + 1) + j)
            y.append(r["jump"])
            lo.append(r["jump"] - r["lo"])
            hi.append(r["hi"] - r["jump"])
            short = {"gemini25pro_0506": "Gem", "deepseekr1_0528": "DS",
                     "clean_pool": "Pool"}[control]
            labels.append(f"{short} {cut[:3]}'{cut[-2:]}")
    a.errorbar(x, y, yerr=[lo, hi], fmt="o", color=BLUE)
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xticks(x)
    a.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    a.set_ylabel("local-linear jump")
    panel_letter(a, "a")

    a = ax[0, 1]
    xs = np.arange(len(targets))
    width = 0.25
    for k, (control, col) in enumerate(zip(CONTROL_LABEL, (BLUE, RED, GREEN))):
        vals = [out["separate_controls"][t][control]["did"] for t in targets]
        los = [vals[i] - out["separate_controls"][t][control]["lo"]
               for i, t in enumerate(targets)]
        his = [out["separate_controls"][t][control]["hi"] - vals[i]
               for i, t in enumerate(targets)]
        a.bar(xs + (k - 1) * width, vals, width=width, yerr=[los, his],
              color=col, label=CONTROL_LABEL[control],
              error_kw=dict(ecolor="0.35", lw=1))
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xticks(xs)
    a.set_xticklabels([TARGET_LABEL[t] for t in targets], fontsize=7)
    a.set_ylabel("DiD at own cutoff")
    a.legend(handlelength=1.2, fontsize=6, ncol=3, loc="lower left",
             bbox_to_anchor=(0.14, 1.0), frameon=False, columnspacing=0.8)
    panel_letter(a, "b")

    a = ax[1, 0]
    for t, col in zip(targets, (BLUE, RED, GREEN)):
        monthly = out["month_scan"][t]["monthly"]
        months = sorted(monthly)
        a.plot(range(len(months)), [monthly[m]["did"] for m in months],
               "o-", color=col, label=TARGET_LABEL[t], ms=3.5)
        own = out["month_scan"][t]["own_cutoff"]
        if own in months:
            a.axvline(months.index(own), color=col, ls=":", alpha=0.6)
    a.axhline(0, color="0.35", lw=0.9)
    months = sorted(out["month_scan"][targets[0]]["monthly"])
    a.set_xticks(range(0, len(months), 2))
    a.set_xticklabels([m[2:] for m in months[::2]], rotation=45, fontsize=6.5)
    a.set_ylabel("DiD vs clean pool")
    a.legend(handlelength=1.4, fontsize=6.5, ncol=3, loc="lower left",
             bbox_to_anchor=(0.14, 1.0), frameon=False, columnspacing=0.8)
    panel_letter(a, "c")

    a = ax[1, 1]
    raw = [out["separate_controls"][t]["clean_pool"]["did"] for t in targets]
    strat = [out["difficulty_stratified"][t]["did_stratified"]
             for t in targets]
    xpos = np.arange(len(targets))
    a.bar(xpos - 0.18, raw, width=0.36, color=BLUE, label="raw DiD")
    a.bar(xpos + 0.18, strat, width=0.36, color=RED,
          label="difficulty-stratified")
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xticks(xpos)
    a.set_xticklabels([TARGET_LABEL[t] for t in targets], fontsize=7)
    a.set_ylabel("DiD")
    a.set_ylim(top=0.12)
    a.legend(handlelength=1.2, fontsize=6.5, loc="upper right")
    panel_letter(a, "d")
    save(fig, "e3_robustness")


# ---- M4-C global-B sensitivity ----------------------------------------------
def fig_e3_global():
    out = json.loads(
        (RESULTS / "livecodebench" / "e3_global_sensitivity.json")
        .read_text())
    targets = list(out["strict_old_control"])
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.6))

    a = ax[0]
    horizons = [160, 240, 365, 730]
    for t, col in zip(targets, (BLUE, RED)):
        vals = [out["strict_old_control"][t][str(h)]["estimate"]
                for h in horizons]
        a.plot(horizons, vals, "o-", color=col, label=TARGET_LABEL.get(t, t))
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xlabel("post-cutoff horizon (days)")
    a.set_ylabel("DiD, strict old control")
    a.legend(handlelength=1.4)
    panel_letter(a, "a")

    a = ax[1]
    x = np.arange(9)
    labels = [f"{h}/{d}" for h in [240, 365, 730] for d in [1, 2, 3]]
    for t, col in zip(targets, (BLUE, RED)):
        vals = [out["extrapolation"][t][f"h{h}_degree{d}"]["estimate"]
                for h in [240, 365, 730] for d in [1, 2, 3]]
        a.plot(x, vals, "o-", color=col, label=TARGET_LABEL.get(t, t))
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xticks(x)
    a.set_xticklabels(labels, rotation=45, fontsize=6.5)
    a.set_xlabel("post horizon / poly degree")
    a.set_ylabel("extrapolated global $B$")
    a.legend(handlelength=1.4)
    panel_letter(a, "b")
    save(fig, "e3_global_sensitivity")


# ---- Matched real-data PRC ---------------------------------------------------
def fig_prc_matched():
    out = json.loads(
        (RESULTS / "prc_matched" / "results.json").read_text())
    methods = list(out["methods"])
    domains = list(out["methods"][methods[0]]["domains"])
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.6))

    a = ax[0]
    x = np.arange(len(domains))
    width = 0.35
    for mi, (method, col) in enumerate(zip(methods, (BLUE, RED))):
        vals = [out["methods"][method]["domains"][d]["point"]["excess_bc"]
                for d in domains]
        cis = [out["methods"][method]["domains"][d]["ci"]["excess_bc"]
               for d in domains]
        err = [[vals[i] - cis[i][0] for i in range(len(vals))],
               [cis[i][1] - vals[i] for i in range(len(vals))]]
        a.bar(x + (mi - 0.5) * width, vals, width, yerr=err, color=col,
              label=method, error_kw=dict(ecolor="0.35", lw=1))
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xticks(x)
    a.set_xticklabels(domains)
    a.set_ylabel(r"treatment $-$ control PRC")
    a.legend(handlelength=1.2)
    panel_letter(a, "a")

    a = ax[1]
    labels, vals, lo, hi = [], [], [], []
    for method in methods:
        r = out["methods"][method]["pooled"]
        labels.append(method)
        vals.append(r["point_excess_bc"])
        lo.append(r["point_excess_bc"] - r["ci"]["excess_bc"][0])
        hi.append(r["ci"]["excess_bc"][1] - r["point_excess_bc"])
    a.bar(labels, vals, yerr=[lo, hi], color=BLUE, width=0.5,
          error_kw=dict(ecolor="0.35", lw=1))
    a.axhline(0, color="0.35", lw=0.9)
    a.set_ylabel(r"pooled excess PRC")
    panel_letter(a, "b")
    save(fig, "prc_matched")


if __name__ == "__main__":
    fig_e1()
    fig_e1_supp()
    fig_e3_robustness()
    fig_e3_global()
    fig_prc_matched()
