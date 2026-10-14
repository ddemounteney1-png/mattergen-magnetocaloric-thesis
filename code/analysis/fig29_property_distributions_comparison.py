import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES, FINETUNE_DATA, SUPERCONDUCTOR_DATA  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

# ============================================================
# EDITABLE: Data paths
# ============================================================
MCE_TRAIN = f"{FINETUNE_DATA}/train_repaired.csv"
MCE_VAL   = f"{FINETUNE_DATA}/val_repaired.csv"
SC_TRAIN  = f"{SUPERCONDUCTOR_DATA}/train.csv"

# ============================================================
# EDITABLE: Conditioning targets (Median, +1σ, +2σ, +3σ, +4σ)
# ============================================================
DS_TARGETS = [
    (5.300, "Med"),
    (18.9,  "+1σ"),
    (32.5,  "+2σ"),
    (46.1,  "+3σ"),
    (59.7,  "+4σ"),
]
TC_MCE_TARGETS = [
    (242.0, "Med"),
    (335.0, "+1σ"),
    (428.0, "+2σ"),
    (521.0, "+3σ"),
    (614.0, "+4σ"),
]
TC_SC_TARGETS = [
    (3.80,  "Med"),
    (40.3,  "+1σ"),
    (76.9,  "+2σ"),
    (113.4, "+3σ"),
    (149.9, "+4σ"),
]

# Interpolation boundary: shade x > +2σ value as extrapolation
DS_EXTRAP_THRESHOLD   = 32.5
TC_MCE_EXTRAP_THRESHOLD = 428.0
TC_SC_EXTRAP_THRESHOLD  = 76.9

# ============================================================
# EDITABLE: Colours
# ============================================================
C_DS  = "#0072B2"   # blue  — ΔS
C_TC  = "#009E73"   # green — magnetocaloric T_C
C_SC  = "#D55E00"   # orange — superconductor T_c
C_TARGET_LINE = "#CC79A7"   # pink — target lines
C_EXTRAP = "#DDDDDD"        # grey — extrapolation shading

# ============================================================
# EDITABLE: Histogram settings
# ============================================================
DS_BIN_WIDTH   = 2.0    # J/kg/K
TC_MCE_BIN_WIDTH = 20.0  # K
TC_SC_BIN_WIDTH  = 5.0   # K
TC_SC_XLIM = 200.0       # cap x-axis for SC (most data <200 K)

# ── Load data ─────────────────────────────────────────────────
train = pd.read_csv(MCE_TRAIN)
val   = pd.read_csv(MCE_VAL).dropna(subset=["n_atoms_primitive"])
mce   = pd.concat([train, val], ignore_index=True)

ds_vals = pd.to_numeric(mce["magnetocaloric_delta_s"], errors="coerce").dropna()
tc_vals = pd.to_numeric(mce["magnetocaloric_curie_temp"], errors="coerce").dropna()
sc_all  = pd.read_csv(SC_TRAIN)
sc_vals = pd.to_numeric(sc_all["sc_transition_temp"], errors="coerce").dropna()

# Computed stats (for annotation)
ds_med, ds_std  = ds_vals.median(), ds_vals.std()
tc_med, tc_std  = tc_vals.median(), tc_vals.std()
sc_med, sc_std  = sc_vals.median(), sc_vals.std()

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97, top=0.88, bottom=0.14)

TARGET_STYLES = [
    {"lw": 2.2, "ls": "-",   "alpha": 0.95},  # Median
    {"lw": 1.8, "ls": "--",  "alpha": 0.85},  # +1σ
    {"lw": 1.8, "ls": "-.",  "alpha": 0.85},  # +2σ
    {"lw": 1.4, "ls": ":",   "alpha": 0.75},  # +3σ
    {"lw": 1.4, "ls": (0,(3,1,1,1,1,1)), "alpha": 0.70},  # +4σ
]
TARGET_COLOURS = ["#222222", "#E69F00", "#D55E00", "#CC79A7", "#56B4E9"]


def draw_targets(ax, targets, extrap_thresh, xmax=None):
    ymax = ax.get_ylim()[1]
    # Extrapolation shading
    x_end = xmax if xmax else targets[-1][0] * 1.15
    ax.axvspan(extrap_thresh, x_end, color=C_EXTRAP, alpha=0.35, zorder=0,
               label="Extrapolation region (>+2σ)")
    for i, (val, lbl) in enumerate(targets):
        st = TARGET_STYLES[i]
        ax.axvline(val, color=TARGET_COLOURS[i],
                   lw=st["lw"], ls=st["ls"], alpha=st["alpha"],
                   zorder=3, label=f"{lbl}  ({val:g})")


# ── Panel A: ΔS ───────────────────────────────────────────────
ax = axes[0]
bins_ds = np.arange(0, ds_vals.max() + DS_BIN_WIDTH, DS_BIN_WIDTH)
ax.hist(ds_vals, bins=bins_ds, color=C_DS, edgecolor="white",
        linewidth=0.5, alpha=0.85, zorder=2)
ax.set_xlim(0, None)
ax.set_ylim(bottom=0)
draw_targets(ax, DS_TARGETS, DS_EXTRAP_THRESHOLD, xmax=ds_vals.max() + DS_BIN_WIDTH)
ax.set_xlabel("$|\\Delta S_{mag}|$ (J kg$^{-1}$ K$^{-1}$)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title(
    "Magnetocaloric $\\Delta S_{mag}$\n"
    f"(n={len(ds_vals)}, med={ds_med:.1f}, σ={ds_std:.1f} J kg$^{{-1}}$K$^{{-1}}$)",
    fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92,
          title="Conditioning targets", title_fontsize=7.5)
ax.text(0.03, 0.97, "A", transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top")

# ── Panel B: MCE T_C ──────────────────────────────────────────
ax = axes[1]
bins_tc = np.arange(0, tc_vals.max() + TC_MCE_BIN_WIDTH, TC_MCE_BIN_WIDTH)
ax.hist(tc_vals, bins=bins_tc, color=C_TC, edgecolor="white",
        linewidth=0.5, alpha=0.85, zorder=2)
ax.set_xlim(0, None)
ax.set_ylim(bottom=0)
draw_targets(ax, TC_MCE_TARGETS, TC_MCE_EXTRAP_THRESHOLD,
             xmax=tc_vals.max() + TC_MCE_BIN_WIDTH)
ax.set_xlabel("$T_C$ (K)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title(
    "Magnetocaloric Curie Temperature $T_C$\n"
    f"(n={len(tc_vals)}, med={tc_med:.0f} K, σ={tc_std:.0f} K)",
    fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92,
          title="Conditioning targets", title_fontsize=7.5)
ax.text(0.03, 0.97, "B", transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top")

# ── Panel C: SC T_c ───────────────────────────────────────────
ax = axes[2]
bins_sc = np.arange(0, TC_SC_XLIM + TC_SC_BIN_WIDTH, TC_SC_BIN_WIDTH)
sc_plot = sc_vals[sc_vals <= TC_SC_XLIM]
n_clipped = (sc_vals > TC_SC_XLIM).sum()
ax.hist(sc_plot, bins=bins_sc, color=C_SC, edgecolor="white",
        linewidth=0.5, alpha=0.85, zorder=2)
ax.set_xlim(0, TC_SC_XLIM)
ax.set_ylim(bottom=0)
draw_targets(ax, TC_SC_TARGETS, TC_SC_EXTRAP_THRESHOLD, xmax=TC_SC_XLIM)
ax.set_xlabel("$T_c$ (K)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
clip_note = f"\n(plot capped at {TC_SC_XLIM:.0f} K; {n_clipped} omitted)" if n_clipped else ""
ax.set_title(
    "Superconductor Transition Temperature $T_c$\n"
    f"(n={len(sc_vals)}, med={sc_med:.1f} K, σ={sc_std:.1f} K){clip_note}",
    fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92,
          title="Conditioning targets", title_fontsize=7.5)
ax.text(0.03, 0.97, "C", transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top")

fig.suptitle(
    "Training Dataset Property Distributions with Conditioning Targets\n"
    "(shaded region = extrapolation beyond +2σ training boundary)",
    fontsize=10.5, y=0.99)

plt.tight_layout(pad=1.2)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{FIGURES}/fig29_property_distributions_comparison.{ext}",
                dpi=300, bbox_inches="tight")
print(f"Saved fig29 — ΔS n={len(ds_vals)}, MCE T_C n={len(tc_vals)}, SC T_c n={len(sc_vals)}")
plt.close()
