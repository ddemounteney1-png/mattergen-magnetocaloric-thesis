import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# EDITABLE: Conditioning step labels and values
# All three sweeps share 3 steps; the conditioning variable differs
# per series (see axis label and legend for interpretation).
# ============================================================
STEP_LABELS = ["Median\nΔS=5.3, $T_C$=242", "+1σ\nΔS=18.9, $T_C$=335", "+2σ\nΔS=32.5, $T_C$=428"]

# ============================================================
# EDITABLE: Actual SUN and stability values from mattergen-evaluate
# ============================================================
# 06_fixed_deltaS_TC_sweep — ΔS fixed at 10.5 J/kg/K, TC swept
# order: tc242 (median), tc335 (+1σ), tc428 (+2σ)
FIXED_DS = {
    "SUN":    [0.1250, 0.0982, 0.0804],
    "stable": [0.2500, 0.2054, 0.1964],
    "ehull":  [0.4326, 0.2708, 0.3870],  # note: tc335 is the outlier low
}

# fixed_Tc_delta_S_sweep — TC fixed at 300 K, ΔS swept
# order: ds5.3 (median), ds18.9 (+1σ), ds32.5 (+2σ)
FIXED_TC = {
    "SUN":    [0.0268, 0.0714, 0.0982],
    "stable": [0.1875, 0.1786, 0.1786],
    "ehull":  [0.3595, 0.4637, 0.4312],
}

# 07_simultaneous_escalation — ΔS and TC both escalate together
# order: (5.3,242) median, (18.9,335) +1σ, (32.5,428) +2σ
SIMUL = {
    "SUN":    [0.0893, 0.0714, 0.1071],
    "stable": [0.1786, 0.2054, 0.2054],
    "ehull":  [0.3516, 0.3534, 0.3408],
}

# ============================================================
# EDITABLE: Colours and bar width
# ============================================================
C_FIXED_DS = "#0072B2"
C_FIXED_TC = "#E69F00"
C_SIMUL    = "#009E73"
WIDTH      = 0.25

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
fig.subplots_adjust(wspace=0.38)
fig.suptitle("Joint ΔS + $T_C$ Conditioning: Strategy Comparison (Joint Fine-tune Model)",
             fontsize=12, y=1.01)

x = np.arange(3)
offsets = [-WIDTH, 0, WIDTH]
series = [
    ("Fixed ΔS=10.5, sweep $T_C$", C_FIXED_DS, FIXED_DS),
    ("Fixed $T_C$=300K, sweep ΔS",  C_FIXED_TC, FIXED_TC),
    ("Simultaneous escalation",      C_SIMUL,    SIMUL),
]

# ── Panel A: SUN fraction ─────────────────────────────────────
ax = axes[0]
for (label, colour, data), offset in zip(series, offsets):
    ax.bar(x + offset, data["SUN"], width=WIDTH, color=colour, alpha=0.85,
           edgecolor="white", linewidth=0.7, label=label)

ax.set_xticks(x)
ax.set_xticklabels(STEP_LABELS, fontsize=8)
ax.set_ylabel("S.U.N. fraction", fontsize=10)
ax.set_title("Stable · Unique · Novel (S.U.N.)", fontsize=11, pad=7)
ax.set_ylim(0, 0.18)
ax.legend(fontsize=8, loc="upper right")
ax.text(0.03, 0.97, "A", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# ── Panel B: Stability fraction + E_hull (twin axis) ─────────
ax2 = axes[1]
ax2b = ax2.twinx()
for (label, colour, data), offset in zip(series, offsets):
    ax2.bar(x + offset, data["stable"], width=WIDTH, color=colour, alpha=0.85,
            edgecolor="white", linewidth=0.7)
    ax2b.plot(x + offset, data["ehull"], marker="o", ms=5, lw=0,
              color=colour, alpha=0.9, zorder=5)

# Connect E_hull points per series with dashed lines
for (label, colour, data), offset in zip(series, offsets):
    ax2b.plot(x + offset, data["ehull"], lw=1.1, ls="--", color=colour, alpha=0.55,
              label=f"Avg. $E_{{hull}}$: {label}")

ax2.set_xticks(x)
ax2.set_xticklabels(STEP_LABELS, fontsize=8)
ax2.set_ylabel("Fraction stable (<0.1 eV/atom above hull)", fontsize=9)
ax2b.set_ylabel("Avg. $E_{hull}$ (eV/atom)  [dashed lines]", fontsize=8.5, color="#555555")
ax2b.tick_params(axis="y", labelcolor="#555555", labelsize=8)
ax2.set_title("Structural Stability & Energy Above Hull", fontsize=11, pad=7)
ax2.set_ylim(0, 0.35)
ax2b.set_ylim(0, 0.6)

# Flag the E_hull outlier (fixed ΔS, TC=+1σ step)
ax2b.annotate("Low $E_{hull}$\n(TC=335K sweet spot?)",
              xy=(1 + offsets[0], FIXED_DS["ehull"][1]),
              xytext=(25, 18), textcoords="offset points",
              fontsize=7.5, color=C_FIXED_DS,
              arrowprops=dict(arrowstyle="->", color=C_FIXED_DS, lw=0.9))

ax2b.legend(fontsize=7.5, loc="upper right", framealpha=0.92, title="── dashed lines", title_fontsize=7)
ax2.text(0.03, 0.97, "B", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")

plt.tight_layout(pad=1.5)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{FIGURES}/fig16_joint_conditioning_sweep.{ext}", dpi=300, bbox_inches="tight")
print("Saved fig16 — 3 joint conditioning strategies × 3 escalation steps")
plt.close()
