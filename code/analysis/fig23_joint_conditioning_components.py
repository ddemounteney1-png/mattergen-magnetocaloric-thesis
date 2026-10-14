import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os

# ============================================================
# EDITABLE: Step labels (shared x-axis across all panels)
# ============================================================
STEP_LABELS = ["Median\n(ΔS=5.3, $T_C$=242)", "+1σ\n(ΔS=18.9, $T_C$=335)", "+2σ\n(ΔS=32.5, $T_C$=428)"]

# ============================================================
# EDITABLE: Per-strategy data — from mattergen-evaluate metrics.json
# frac_stable_structures / frac_unique_structures / frac_novel_structures
# frac_novel_unique_stable_structures = SUN
# ============================================================
STRATEGIES = [
    {
        "label":  "Fixed $\\Delta S$=10.5, sweep $T_C$",
        "colour": "#0072B2",
        "stable": [0.2500, 0.2054, 0.1964],
        "unique": [0.9732, 0.9821, 0.9911],
        "novel":  [0.7946, 0.7411, 0.7589],
        "sun":    [0.1250, 0.0982, 0.0804],
    },
    {
        "label":  "Fixed $T_C$=300 K, sweep $\\Delta S$",
        "colour": "#E69F00",
        "stable": [0.1875, 0.1786, 0.1786],
        "unique": [0.9732, 0.9821, 0.9911],
        "novel":  [0.7500, 0.8125, 0.8393],
        "sun":    [0.0268, 0.0714, 0.0982],
    },
    {
        "label":  "Simultaneous escalation",
        "colour": "#009E73",
        "stable": [0.1786, 0.2054, 0.2054],
        "unique": [0.9821, 0.9732, 1.0000],
        "novel":  [0.7946, 0.7411, 0.8661],
        "sun":    [0.0893, 0.0714, 0.1071],
    },
]

# ============================================================
# EDITABLE: Metric line styles and colours (shared across panels)
# ============================================================
METRIC_LINES = [
    ("stable", "Stable (<0.1 eV/atom)",  "#D55E00", "-"),
    ("novel",  "Novel",                  "#CC79A7", "--"),
    ("unique", "Unique",                 "#999999", ":"),
    ("sun",    "S.U.N. (combined)",      "black",   "-."),
]

# ============================================================
# EDITABLE: Y-axis — show full 0–1 range so scale is honest
# ============================================================
Y_MAX = 1.05

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.08, left=0.07, right=0.97, top=0.88, bottom=0.18)
fig.suptitle(
    "Generation Quality Components Across Joint $\\Delta S$+$T_C$ Conditioning Strategies",
    fontsize=11, y=0.97)

x = np.arange(3)
BAR_WIDTH = 0.55

panel_letters = ["A", "B", "C"]

for ax, strat, letter in zip(axes, STRATEGIES, panel_letters):
    # SUN bar (shaded, behind lines)
    ax.bar(x, strat["sun"], width=BAR_WIDTH,
           color=strat["colour"], alpha=0.30, edgecolor="white",
           linewidth=0.7, zorder=2, label="_nolegend_")

    # Component lines
    for metric_key, _, line_col, ls in METRIC_LINES:
        ax.plot(x, strat[metric_key], color=line_col, ls=ls,
                lw=1.8, marker="o", ms=5, zorder=4,
                markerfacecolor="white", markeredgewidth=1.4,
                markeredgecolor=line_col)

    # Annotate the SUN bar tops with values
    for xi, val in zip(x, strat["sun"]):
        ax.text(xi, val + 0.012, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, color=strat["colour"],
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(STEP_LABELS, fontsize=7.5)
    ax.set_ylim(0, Y_MAX)
    ax.set_title(strat["label"], fontsize=9.5, pad=6, color=strat["colour"],
                 fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    ax.text(0.04, 0.97, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

axes[0].set_ylabel("Fraction", fontsize=10)

# ── Shared legend ─────────────────────────────────────────────
line_handles = [
    mlines.Line2D([], [], color=col, ls=ls, lw=1.8, marker="o", ms=5,
                  markerfacecolor="white", markeredgecolor=col,
                  markeredgewidth=1.4, label=lbl)
    for _, lbl, col, ls in METRIC_LINES
]
# Add a patch for the SUN bars
import matplotlib.patches as mpatches
bar_handle = mpatches.Patch(facecolor="#888888", alpha=0.30,
                             edgecolor="white", label="S.U.N. (bar height)")
# Override the SUN line handle label to distinguish line vs bar
line_handles[-1] = mlines.Line2D([], [], color="black", ls="-.", lw=1.8,
                                  marker="o", ms=5, markerfacecolor="white",
                                  markeredgecolor="black", markeredgewidth=1.4,
                                  label="S.U.N. (line = same data as bar)")

fig.legend(handles=line_handles, fontsize=8.5, loc="lower center",
           ncol=4, bbox_to_anchor=(0.5, -0.01), framealpha=0.95)

os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{FIGURES}/fig23_joint_conditioning_components.{ext}",
                dpi=300, bbox_inches="tight")
print("Saved fig23 — stable/unique/novel/SUN components for 3 joint conditioning strategies")
plt.close()
