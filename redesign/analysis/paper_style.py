"""Shared matplotlib style for all paper figures.

Principles: no in-figure titles (captions carry the explanation), small
consistent fonts, one palette, no top/right spines, panel letters for
multi-panel figures. Figures are saved to both the experiment figures/
directory and the paper's figures/ directory.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REDESIGN = Path(__file__).resolve().parent.parent
FIG_DIRS = [
    REDESIGN / "figures",
    REDESIGN.parent.parent / "Theory" / "tmlr_paper" / "figures",
]

BLUE = "#3A6EA5"
RED = "#D1495B"
GREEN = "#2E8B6E"
PURPLE = "#7A5FA8"
ORANGE = "#E8A13A"
GREY = "#9AA0A6"
PALETTE = [BLUE, RED, GREEN, PURPLE, ORANGE, GREY]


def apply_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "#d8dce0",
        "legend.fancybox": True,
        "axes.facecolor": "#FBFBFD",
        "axes.edgecolor": "#4a4a4a",
        "axes.grid": True,
        "grid.color": "#dfe3e8",
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#4a4a4a",
        "ytick.color": "#4a4a4a",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "lines.linewidth": 1.4,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0.9,
        "lines.markeredgecolor": "white",
        "errorbar.capsize": 2.5,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "figure.constrained_layout.use": True,
    })


def zero_line(ax, axis="y"):
    """Soft reference line at zero (grey, under the data)."""
    if axis == "y":
        ax.axhline(0, color="0.35", lw=0.9, zorder=1.5)
    else:
        ax.axvline(0, color="0.35", lw=0.9, zorder=1.5)


def panel_letter(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontsize=9, fontweight="bold")


def save(fig, name):
    for d in FIG_DIRS:
        d.mkdir(exist_ok=True)
        fig.savefig(d / f"{name}.pdf")
        fig.savefig(d / f"{name}.png", dpi=200)
    print(f"saved {name}")
