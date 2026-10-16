import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES, METRICS  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import json
import os

# ============================================================
# EDITABLE: Run paths grouped by conditioning level
# Each tuple: (strategy_label, path_to_detailed_metrics.json)
# ============================================================
LEVEL_RUNS = {
    "Median":   [
        ("Fixed $\\Delta S$",  f"{METRICS}/06_fixed_deltaS_TC_sweep/ds10p5_tc242/detailed_metrics.json"),
        ("Fixed $T_C$",        f"{METRICS}/fixed_Tc_delta_S_sweep/deltaS_5.3_TC_300_detailed_metrics.json"),
        ("Simultaneous",       f"{METRICS}/07_simultaneous_escalation/ds5p3_tc242/detailed_metrics.json"),
    ],
    "+1$\\sigma$": [
        ("Fixed $\\Delta S$",  f"{METRICS}/06_fixed_deltaS_TC_sweep/ds10p5_tc335/detailed_metrics.json"),
        ("Fixed $T_C$",        f"{METRICS}/fixed_Tc_delta_S_sweep/deltaS_18.9_TC_300_detailed_metrics.json"),
        ("Simultaneous",       f"{METRICS}/07_simultaneous_escalation/ds18p9_tc335/detailed_metrics.json"),
    ],
    "+2$\\sigma$": [
        ("Fixed $\\Delta S$",  f"{METRICS}/06_fixed_deltaS_TC_sweep/ds10p5_tc428/detailed_metrics.json"),
        ("Fixed $T_C$",        f"{METRICS}/fixed_Tc_delta_S_sweep/deltaS_32.5_TC_300_detailed_metrics.json"),
        ("Simultaneous",       f"{METRICS}/07_simultaneous_escalation/ds32p5_tc428/detailed_metrics.json"),
    ],
}

# ============================================================
# EDITABLE: Strategy colours (consistent with fig16 / fig23)
# ============================================================
STRATEGY_COLOURS = {
    "Fixed $\\Delta S$": "#0072B2",
    "Fixed $T_C$":       "#E69F00",
    "Simultaneous":      "#009E73",
}

# ============================================================
# EDITABLE: Y-axis clip value (eV/atom) — avoids extreme outliers
# compressing the interesting region near 0
# ============================================================
Y_CLIP = 2.0

STABILITY_THRESHOLD = 0.1   # eV/atom

# ── Load data ─────────────────────────────────────────────────
# Positions: 3 groups of 3 violins, with gap between groups
positions = []
group_centres = []
group_labels  = []
group_data    = []   # list of (strategy, ehull_array, stable_mask, positions_list)

pos = 0
INNER_GAP  = 1.0   # between violins within a group
GROUP_GAP  = 2.2   # extra gap between conditioning groups

for level, runs in LEVEL_RUNS.items():
    centre_start = pos
    level_positions = []
    for strat, path in runs:
        d = json.load(open(path))
        eh    = np.array(d["energy_above_hull_per_atom"])
        eh_clipped = np.clip(eh, None, Y_CLIP)
        stable = np.array(d["stable"])
        group_data.append((strat, eh_clipped, stable, pos))
        level_positions.append(pos)
        pos += INNER_GAP
    group_centres.append(np.mean(level_positions))
    group_labels.append(level)
    pos += GROUP_GAP   # extra gap before next group

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(13.0, 5.2))

all_positions = [item[3] for item in group_data]

# Violin plot
vp = ax.violinplot(
    [item[1] for item in group_data],
    positions=all_positions,
    widths=0.75,
    showmedians=True,
    showextrema=False,
)

# Colour each violin by strategy
for body, (strat, _, _, _) in zip(vp["bodies"], group_data):
    body.set_facecolor(STRATEGY_COLOURS[strat])
    body.set_alpha(0.55)
    body.set_edgecolor("white")
    body.set_linewidth(0.5)
vp["cmedians"].set_color("white")
vp["cmedians"].set_linewidth(1.5)

# Overlay IQR box and stable-structure scatter per violin
rng = np.random.default_rng(42)
for strat, eh, stable, xpos in group_data:
    colour = STRATEGY_COLOURS[strat]

    # IQR box
    q25, q75 = np.percentile(eh, [25, 75])
    ax.plot([xpos - 0.08, xpos + 0.08], [q25, q25], color=colour, lw=1.2, zorder=4)
    ax.plot([xpos - 0.08, xpos + 0.08], [q75, q75], color=colour, lw=1.2, zorder=4)
    ax.plot([xpos, xpos], [q25, q75], color=colour, lw=1.5, zorder=4)

    # Scatter: stable structures (jittered)
    n_stable = stable.sum()
    if n_stable > 0:
        jitter = rng.uniform(-0.12, 0.12, n_stable)
        ax.scatter(xpos + jitter, eh[stable],
                   color=colour, s=18, alpha=0.85, zorder=5,
                   edgecolors="white", linewidth=0.3)
    ax.text(xpos, -0.14, f"n={n_stable}", ha="center", va="top",
            fontsize=7, color=colour)

# Stability threshold line
ax.axhline(STABILITY_THRESHOLD, color="#D55E00", lw=1.2, ls="--", alpha=0.75, zorder=3,
           label=f"Stability threshold ({STABILITY_THRESHOLD} eV/atom)")
ax.text(all_positions[-1] + 0.6, STABILITY_THRESHOLD + 0.03,
        "stable", fontsize=8, color="#D55E00", va="bottom")

# Group labels and dividers
for i, (centre, label) in enumerate(zip(group_centres, group_labels)):
    ax.text(centre, Y_CLIP + 0.04, label, ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    if i < len(group_centres) - 1:
        divider = (group_centres[i] + group_centres[i + 1]) / 2
        ax.axvline(divider, color="#cccccc", lw=0.8, ls="--", zorder=0)

# Clip line annotation
ax.axhline(Y_CLIP, color="#aaaaaa", lw=0.7, ls=":", zorder=0)
ax.text(all_positions[-1] + 0.6, Y_CLIP - 0.03,
        f"clip at {Y_CLIP}", fontsize=7, color="#aaaaaa", va="top")

# Legend
legend_handles = [
    mpatches.Patch(facecolor=c, alpha=0.7, label=s)
    for s, c in STRATEGY_COLOURS.items()
]
legend_handles.append(
    mlines.Line2D([], [], color="#D55E00", ls="--", lw=1.2, label=f"Stability threshold")
)
legend_handles.append(
    plt.scatter([], [], color="gray", s=18, label="Stable structures (dots)")
)
ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right", framealpha=0.92)

ax.set_xticks(all_positions)
ax.set_xticklabels(
    [s.replace("Fixed $\\Delta S$", "Fix.ΔS").replace("Fixed $T_C$", "Fix.$T_C$")
       .replace("Simultaneous", "Simul.")
     for s, *_ in group_data],
    fontsize=7.5
)
ax.set_ylabel("Energy above hull (eV/atom)", fontsize=11)
ax.set_title(
    "$E_{hull}$ Distribution Across Joint Conditioning Strategies\n"
    "(coloured dots = stable structures; violin = full distribution; white line = median; "
    f"y capped at {Y_CLIP} eV/atom)",
    fontsize=10, pad=8)
ax.set_ylim(-0.2, Y_CLIP + 0.15)
ax.set_xlim(all_positions[0] - 0.7, all_positions[-1] + 0.7)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout(pad=1.5)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{FIGURES}/fig24_ehull_violin.{ext}", dpi=300, bbox_inches="tight")
print("Saved fig24 — E_hull violin plot, 9 runs (3 strategies × 3 conditioning levels)")
plt.close()
