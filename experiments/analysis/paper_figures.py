#!/usr/bin/env python3
"""Regenerate ALL main-text figures with the unified paper style.

Reads existing result JSONs (no recomputation):
  m1/results.json                       -> m1_forest
  hubble/results.json        -> m2
  m3/analysis_full.json                 -> m3_dose_response, m3_concentration
  m3/prc_differenced.json               -> m3_prc
  m5/results.json                       -> m5_dissolve
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (BLUE, GREEN, GREY, RED, ROOT, apply_style,
                         panel_letter, save)

apply_style()
import matplotlib.pyplot as plt

PACKAGE = ROOT.parent


# ---- M1: forest plot --------------------------------------------------------
def fig_m1():
    results = json.loads((ROOT / "m1" / "results.json").read_text())
    if isinstance(results, dict):
        results = results["per_model"]
    n = len(results)
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * n + 0.9))
    ys = np.arange(n)[::-1]
    hi_max = max(r["ci"][1] for r in results)
    for yy, r in zip(ys, results):
        ax.errorbar(r["gap"], yy,
                    xerr=[[r["gap"] - r["ci"][0]], [r["ci"][1] - r["gap"]]],
                    fmt="o", color=BLUE, lw=1.4)
        lab = f"{r['gap']:+.3f}" + ("*" if r["star"] else "")
        ax.text(hi_max + 0.012, yy, lab, va="center", fontsize=8)
    ax.axvline(0, color="0.35", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['name']} ({r['cutoff'][:7]})"
                        for r in results])
    ax.set_xlim(right=hi_max + 0.045)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(r"naive gap $\Delta$")
    save(fig, "m1_forest")


# ---- M2: Hubble 2x2 ---------------------------------------------------------
def fig_m2():
    res = json.loads(
        (PACKAGE / "hubble" / "results.json").read_text())
    head = res["8b-500b"]
    DUPS = [0, 1, 4, 16, 64, 256]
    mt = {r: head["aggregate"]["raw_twin"][str(r)] for r in DUPS}
    mc = {r: head["aggregate"]["placebo_centered_twin"][str(r)] for r in DUPS}
    ms = {r: head["aggregate"]["self"][str(r)] for r in DUPS}
    xs = DUPS
    xplot = [max(x, 0.5) for x in xs]

    fig, ax = plt.subplots(2, 2, figsize=(6.0, 4.4))
    a = ax[0, 0]
    for dat, fmtm, lab, col in ((mt, "o-", "raw minimal-pair", BLUE),
                                (mc, "^-", "placebo-centered", RED),
                                (ms, "s--", "self-baseline", GREEN)):
        a.errorbar(xplot, [dat[r][0] for r in xs],
                   yerr=[1.96 * dat[r][1] for r in xs],
                   fmt=fmtm, color=col, label=lab, ms=4)
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xscale("log")
    a.set_xlabel("duplication rate $r$")
    a.set_ylabel("estimated inflation")
    a.legend(loc="upper left", handlelength=1.6)
    panel_letter(a, "a")

    a = ax[0, 1]
    for (pair, marker), col in zip(
            [("8b-500b", "o"), ("8b-100b", "s"), ("1b-500b", "^")],
            (BLUE, RED, GREEN)):
        agg = res[pair]["aggregate"]["placebo_centered_twin"]
        a.plot(xplot, [agg[str(r)][0] for r in xs], marker + "-",
               color=col, label=pair, ms=4)
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xscale("log")
    a.set_xlabel("duplication rate $r$")
    a.set_ylabel("placebo-centered estimate")
    a.legend(loc="upper left", handlelength=1.6)
    panel_letter(a, "b")

    a = ax[1, 0]
    for task, col in zip(head["tasks"], (BLUE, RED, GREEN, "#7b5aa6")):
        t = head["tasks"][task]
        y = [t["B_twin"][str(r)][0] - t["B_twin"]["0"][0] for r in xs]
        a.plot(xplot, y, "o-", color=col,
               label=task.replace("_mcq", ""), ms=4)
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xscale("log")
    a.set_xlabel("duplication rate $r$")
    a.set_ylabel("per-task estimate")
    a.legend(loc="upper left", handlelength=1.6)
    panel_letter(a, "c")

    a = ax[1, 1]
    tasks = list(head["std_flatness"].keys())
    a.bar(range(len(tasks)), [head["std_flatness"][t] for t in tasks],
          color=BLUE, width=0.6)
    a.set_xticks(range(len(tasks)))
    a.set_xticklabels([t.replace("_mcq", "") for t in tasks],
                      rotation=20, ha="right")
    a.set_ylabel("clean-control acc. range")
    panel_letter(a, "d")
    save(fig, "m2")


# ---- M2: main-text dose response (headline panel) ---------------------------
def fig_m2_main():
    res = json.loads(
        (PACKAGE / "hubble" / "results.json").read_text())
    head = res["8b-500b"]
    DUPS = [0, 1, 4, 16, 64, 256]
    mc = {r: head["aggregate"]["placebo_centered_twin"][str(r)] for r in DUPS}
    xplot = [max(r, 0.5) for r in DUPS]
    vals = [mc[r][0] for r in DUPS]
    errs = [1.96 * mc[r][1] for r in DUPS]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.errorbar(xplot, vals, yerr=errs, fmt="o-", color=BLUE, lw=1.6)
    # skip r=0: it is the centering reference (zero by construction)
    for x, v, e in list(zip(xplot, vals, errs))[1:]:
        star = "*" if v - e > 0 else ""
        ax.annotate(f"{v:+.3f}{star}", (x, v + e),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(xplot)
    ax.set_xticklabels([str(r) for r in DUPS])
    ax.minorticks_off()
    ax.set_ylim(top=max(v + e for v, e in zip(vals, errs)) + 0.06)
    ax.set_xlabel("dose $r$")
    ax.set_ylabel(r"$\widetilde B_{\mathrm{pair}}$")
    save(fig, "m2_main")


# ---- M3: dose-response ------------------------------------------------------
def fig_m3_dose():
    res = json.loads((ROOT / "m3" / "analysis_full.json").read_text())
    dr = res["dose_response"]
    doses = [0, 1, 4, 16, 64]
    means = [dr[str(d)]["B_twin"][0] for d in doses]
    los = [dr[str(d)]["B_twin"][1] for d in doses]
    his = [dr[str(d)]["B_twin"][2] for d in doses]
    x = np.arange(len(doses))

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.errorbar(x, means, yerr=[np.array(means) - los, np.array(his) - means],
                fmt="o-", color=BLUE, lw=1.6)
    for xi, (m, lo, hi) in enumerate(zip(means, los, his)):
        star = "*" if lo > 0 or hi < 0 else ""
        ax.annotate(f"{m:+.3f}{star}", (xi, hi), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in doses])
    ax.set_ylim(top=max(his) + 0.014)
    ax.set_xlabel("dose $r$")
    ax.set_ylabel(r"$\widehat B_{\mathrm{twin}}$")
    save(fig, "m3_dose_response")


# ---- M3: concentration law --------------------------------------------------
def fig_m3_conc():
    res = json.loads((ROOT / "m3" / "analysis_full.json").read_text())
    conc = res["concentration_q5"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    ymax = 0.0
    for ax, dose in ((axes[0], 16), (axes[1], 64)):
        w = res["w_fitted"][str(dose)]
        obs, lo, hi, pred, b0s = [], [], [], [], []
        for t in range(5):
            c = conc[f"d{dose}_q{t}"]
            obs.append(c["observed_L"][0])
            lo.append(c["observed_L"][1])
            hi.append(c["observed_L"][2])
            b0s.append(c["mean_b0"])
            pred.append(c["mean_b0"] * w * (2 - w))
        xx = np.arange(5)
        ax.errorbar(xx - 0.1, obs,
                    yerr=[np.array(obs) - lo, np.array(hi) - obs],
                    fmt="o", color=BLUE, label="observed")
        ax.plot(xx + 0.1, pred, "s", color=RED, label="law prediction")
        ax.set_xticks(xx)
        ax.set_xticklabels([f"{b0s[t]:.2f}" for t in range(5)])
        ax.set_xlabel(r"$\bar b_0$")
        ax.axhline(0, color="0.35", lw=0.9)
        ax.text(0.03, 0.97, f"$r={dose}$", transform=ax.transAxes,
                fontsize=9, va="top")
        ymax = max(ymax, max(hi), max(pred))
    axes[0].set_ylim(top=ymax + 0.025)
    axes[0].set_ylabel(r"$\widehat B_{\mathrm{twin}}$")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, ncol=2, loc="outside upper center",
               frameon=False, handlelength=1.2, columnspacing=1.2)
    save(fig, "m3_concentration")


# ---- M3: differenced PRC ----------------------------------------------------
def fig_m3_prc():
    dprc = json.loads((ROOT / "m3" / "prc_differenced.json").read_text())
    cells = [("dose0", "$r=0$"), ("d1", "$r=1$"), ("d4", "$r=4$"),
             ("d16", "$r=16$"), ("d64", "$r=64$")]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    xx = np.arange(len(cells))
    vals = [dprc[k][0] for k, _ in cells]
    los = [dprc[k][1] for k, _ in cells]
    his = [dprc[k][2] for k, _ in cells]
    colors = [GREY if k == "dose0" else BLUE for k, _ in cells]
    ax.bar(xx, vals, color=colors, width=0.55)
    ax.errorbar(xx, vals, yerr=[np.array(vals) - los, np.array(his) - vals],
                fmt="none", ecolor="0.3", lw=1)
    for xi, (v, lo, hi) in enumerate(zip(vals, los, his)):
        if lo > 0 or hi < 0:
            ax.annotate("*", (xi, hi), ha="center", fontsize=11,
                        textcoords="offset points", xytext=(0, 1))
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xticks(xx)
    ax.set_xticklabels([lab for _, lab in cells])
    ax.set_xlabel("injected dose of the question cell")
    ax.set_ylabel(r"calibrated $\Delta$PRC")
    save(fig, "m3_prc")


# ---- M5: dissolve plot ------------------------------------------------------
def fig_m5():
    res = json.loads((ROOT / "m5" / "results.json").read_text())
    rows = res["rows"]
    names = [r["name"] for r in rows]
    n = len(rows)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    for i, r in enumerate(rows):
        yi = y[i]
        ng, nc = r["naive_gap"], r["naive_ci"]
        dd, dc = r["did"], r["did_ci"]
        ax.annotate("", xy=(dd, yi - 0.18), xytext=(ng, yi + 0.18),
                    arrowprops=dict(arrowstyle="->", color="0.75", lw=0.9))
        ax.errorbar([ng], [yi + 0.18], xerr=[[ng - nc[0]], [nc[1] - ng]],
                    fmt="o", color=RED,
                    label=r"naive $\Delta$" if i == 0 else None)
        ax.errorbar([dd], [yi - 0.18], xerr=[[dd - dc[0]], [dc[1] - dd]],
                    fmt="s", color=BLUE,
                    label=r"adjusted $\widehat\Delta_{\mathrm{DiD}}$" if i == 0 else None)
        for v, ci, yy in ((ng, nc, yi + 0.18), (dd, dc, yi - 0.18)):
            if ci[0] > 0 or ci[1] < 0:
                ax.annotate("*", (ci[1], yy), textcoords="offset points",
                            xytext=(2, -3), fontsize=10)
    wc = res["weak_control"]
    ax.errorbar([wc["did"]], [-1],
                xerr=[[wc["did"] - wc["ci"][0]], [wc["ci"][1] - wc["did"]]],
                fmt="D", color=GREY, label="weak control")
    ax.axvline(0, color="0.35", lw=1.0)
    ax.set_yticks(list(y) + [-1])
    ax.set_yticklabels(names + ["Qwen3.5, weak control"])
    ax.set_xlabel(r"boundary gap $\Delta$")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2, columnspacing=1.0, frameon=False)
    save(fig, "m5_dissolve")


if __name__ == "__main__":
    fig_m1()
    fig_m2()
    fig_m2_main()
    fig_m3_dose()
    fig_m3_conc()
    fig_m3_prc()
    fig_m5()
