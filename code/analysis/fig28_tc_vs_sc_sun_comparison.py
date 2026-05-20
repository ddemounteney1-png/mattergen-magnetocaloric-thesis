import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# ============================================================
# EDITABLE: TC-only run data (from mattergen-evaluate)
# ============================================================
TC_LEVELS = ["Median\n(242 K)", "+1σ\n(335 K)", "+2σ\n(428 K)", "+3σ\n(521 K)", "+4σ\n(614 K)"]
TC_DATA = {
    "stable": [0.2692, 0.1923, 0.1971, 0.2212, 0.2404],
    "unique": [0.9615, 0.9856, 0.9856, 0.9760, 0.9808],
    "novel":  [0.7885, 0.8365, 0.8173, 0.8173, 0.8173],
    "sun":    [0.1298, 0.0962, 0.0769, 0.1058, 0.1010],
}

# ============================================================
# EDITABLE: Superconductor run data (from mattergen-evaluate)
# Conditioning values: median=3.8K, +1σ=40.3K, +2σ=76.9K, +3σ=113.4K, +4σ=149.9K
# ============================================================
SC_LEVELS = ["Median\n(3.8 K)", "+1σ\n(40.3 K)", "+2σ\n(76.9 K)", "+3σ\n(113.4 K)", "+4σ\n(149.9 K)"]
SC_DATA = {
    "stable": [0.3750, 0.3510, 0.4904, 0.5817, 0.5962],
    "unique": [0.9904, 0.9663, 0.8125, 0.6490, 0.7019],
    "novel":  [0.7163, 0.7837, 0.6587, 0.6587, 0.6154],
    "sun":    [0.1394, 0.1779, 0.1538, 0.1731, 0.2115],
}

# ============================================================
# EDITABLE: Colours and styles
# ============================================================
C_TC  = "#3700FF"   # blue  — TC-only
C_SC  = "#FF3300"   # green — superconductor
BAR_WIDTH = 0.32

# Dotted line styles for individual metrics
METRIC_STYLES = {
    "stable": {"ls": "-",  "marker": "o", "label": "Stable"},
    "unique": {"ls": "--", "marker": "s", "label": "Unique"},
    "novel":  {"ls": ":",  "marker": "^", "label": "Novel"},
}

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12.0, 5.5))

x = np.arange(5)

# ── SUN bars ─────────────────────────────────────────────────
bars_tc = ax.bar(x - BAR_WIDTH/2, [v*100 for v in TC_DATA["sun"]],
                 width=BAR_WIDTH, color=C_TC, alpha=0.85,
                 edgecolor="white", linewidth=0.7,
                 label="S.U.N. — $T_C$-only conditioning")
bars_sc = ax.bar(x + BAR_WIDTH/2, [v*100 for v in SC_DATA["sun"]],
                 width=BAR_WIDTH, color=C_SC, alpha=0.85,
                 edgecolor="white", linewidth=0.7,
                 label="S.U.N. — Superconductor conditioning")

# Value labels on bars
for bar in bars_tc:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", va="bottom",
            fontsize=7.5, color=C_TC, fontweight="bold")
for bar in bars_sc:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", va="bottom",
            fontsize=7.5, color=C_SC, fontweight="bold")

# ── Component dotted lines ────────────────────────────────────
for metric, style in METRIC_STYLES.items():
    ax.plot(x - BAR_WIDTH/2, [v*100 for v in TC_DATA[metric]],
            color=C_TC, ls=style["ls"], lw=1.5,
            marker=style["marker"], ms=5,
            markerfacecolor="white", markeredgewidth=2,
            markeredgecolor=C_TC, alpha=0.80, zorder=4)
    ax.plot(x + BAR_WIDTH/2, [v*100 for v in SC_DATA[metric]],
            color=C_SC, ls=style["ls"], lw=1.5,
            marker=style["marker"], ms=5,
            markerfacecolor="white", markeredgewidth=2,
            markeredgecolor=C_SC, alpha=0.80, zorder=4)

# ── X-axis: paired level labels ───────────────────────────────
ax.set_xticks(x)
# Two-line tick labels: TC level on top, SC level below
combo_labels = [f"$T_C$: {tc.replace(chr(10), ' ')}\nSC: {sc.replace(chr(10), ' ')}"
                for tc, sc in zip(TC_LEVELS, SC_LEVELS)]
ax.set_xticklabels(combo_labels, fontsize=7.5)

ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_ylim(0, 115)
ax.set_title(
    "Generation Quality: $T_C$-only vs Superconductor Conditioning\n"
    "(bars = S.U.N.; dotted lines = individual components)",
    fontsize=11, pad=10)
ax.tick_params(axis="y", labelsize=9)

# ── Legend ────────────────────────────────────────────────────
bar_handles = [
    mpatches.Patch(color=C_TC, alpha=0.85, label="S.U.N. — $T_C$-only"),
    mpatches.Patch(color=C_SC, alpha=0.85, label="S.U.N. — Superconductor"),
]
line_handles = [
    mlines.Line2D([], [], color="gray", ls=style["ls"], lw=1.5,
                  marker=style["marker"], ms=5, markerfacecolor="white",
                  markeredgecolor="gray", markeredgewidth=2,
                  label=style["label"])
    for style in METRIC_STYLES.values()
]
ax.legend(handles=bar_handles + line_handles,
          fontsize=8.5, loc="upper right", framealpha=0.92,
          title="Bars = S.U.N.  |  Lines = components (blue=$T_C$, red=SC)",
          title_fontsize=7.5)

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"figures/fig28_tc_vs_sc_sun_comparison.{ext}", dpi=300, bbox_inches="tight")
print("Saved fig28 — TC-only vs SC conditioning SUN comparison")
plt.close()
