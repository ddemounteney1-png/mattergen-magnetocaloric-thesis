import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES, FINETUNE_DATA  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import os

# ============================================================
# EDITABLE: Titles and labels
# ============================================================
TITLE_A   = "$\\Delta S_{mag}$ Distribution"
TITLE_B   = "$T_C$ Distribution"
TITLE_C   = "Applied Field Distribution"
SUPTITLE  = "Magnetocaloric Property Distributions — Fine-Tuning Dataset"
XLABEL_A  = "$\\Delta S_{mag}$ (J kg$^{-1}$ K$^{-1}$)"
XLABEL_B  = "$T_C$ (K)"
XLABEL_C  = "Applied field $H$ (T)"
YLABEL    = "Count"

# ============================================================
# EDITABLE: Conditioning target lines
# Values from computed statistics (median and std of train set)
# ============================================================
DS_TARGETS = {
    "Median (5.3)":   {"val": 5.3,   "ls": "-",  "lw": 1.8, "alpha": 0.9},
    "+1σ (18.9)":     {"val": 18.9,  "ls": "--", "lw": 1.4, "alpha": 0.8},
    "+2σ (32.5)":     {"val": 32.5,  "ls": "--", "lw": 1.4, "alpha": 0.8},
    "+3σ (46.1)":     {"val": 46.1,  "ls": ":",  "lw": 1.2, "alpha": 0.7},
    "+4σ (59.7)":     {"val": 59.7,  "ls": ":",  "lw": 1.2, "alpha": 0.7},
}
TC_TARGETS = {
    "Median (242 K)": {"val": 242.0, "ls": "-",  "lw": 1.8, "alpha": 0.9},
    "+1σ (335 K)":    {"val": 335.0, "ls": "--", "lw": 1.4, "alpha": 0.8},
    "+2σ (428 K)":    {"val": 428.0, "ls": "--", "lw": 1.4, "alpha": 0.8},
    "Room temp (300 K)": {"val": 300.0, "ls": "-.", "lw": 2.0, "alpha": 1.0},
}
TC_EXTRAPOLATION_START = 428.0   # shade above +2σ

# ============================================================
# EDITABLE: Colours
# ============================================================
C_HIST   = "#0072B2"
C_TARGET = "#D55E00"
C_ROOM   = "#CC79A7"
C_SHADE  = "#FFD580"

# ============================================================
# Load data
# ============================================================
train = pd.read_csv(f"{FINETUNE_DATA}/train_repaired.csv")
val   = pd.read_csv(f"{FINETUNE_DATA}/val_repaired.csv").dropna(subset=["n_atoms_primitive"])
both  = pd.concat([train, val], ignore_index=True)

ds_vals  = pd.to_numeric(both["magnetocaloric_delta_s"],    errors="coerce").dropna()
tc_vals  = pd.to_numeric(both["magnetocaloric_curie_temp"], errors="coerce").dropna()
h_vals   = pd.to_numeric(both["magnetocaloric_field"],      errors="coerce").dropna()

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(9.0, 4.0))
fig.subplots_adjust(wspace=0.45)

# ── Panel A: ΔS ────────────────────────────────────────────
ax = axes[0]
bins_ds = np.linspace(0, ds_vals.max() + 2, 30)
ax.hist(ds_vals, bins=bins_ds, color=C_HIST, alpha=0.80, edgecolor="white")
ymax = ax.get_ylim()[1]
ax.axvspan(DS_TARGETS["+2σ (32.5)"]["val"], bins_ds[-1],
           alpha=0.12, color=C_SHADE, label="Extrapolation region (>+2σ)")
for label, cfg in DS_TARGETS.items():
    clr = C_ROOM if "Median" in label else C_TARGET
    ax.axvline(cfg["val"], color=clr, ls=cfg["ls"], lw=cfg["lw"],
               alpha=cfg["alpha"])
ax.set_title(TITLE_A, fontsize=11, pad=6)
ax.set_xlabel(XLABEL_A, fontsize=10)
ax.set_ylabel(YLABEL, fontsize=10)
ax.tick_params(labelsize=9)
ax.text(0.04, 0.96, "A", transform=ax.transAxes, fontsize=13,
        fontweight="bold", va="top")

# ── Panel B: T_C ───────────────────────────────────────────
ax2 = axes[1]
bins_tc = np.linspace(tc_vals.min() - 10, tc_vals.max() + 10, 30)
ax2.hist(tc_vals, bins=bins_tc, color=C_HIST, alpha=0.80, edgecolor="white")
ax2.axvspan(TC_EXTRAPOLATION_START, bins_tc[-1],
            alpha=0.12, color=C_SHADE)
for label, cfg in TC_TARGETS.items():
    clr = C_ROOM if "Room" in label else C_TARGET
    ax2.axvline(cfg["val"], color=clr, ls=cfg["ls"], lw=cfg["lw"],
                alpha=cfg["alpha"])
ax2.set_title(TITLE_B, fontsize=11, pad=6)
ax2.set_xlabel(XLABEL_B, fontsize=10)
ax2.tick_params(labelsize=9)
ax2.text(0.04, 0.96, "B", transform=ax2.transAxes, fontsize=13,
         fontweight="bold", va="top")

# ── Panel C: Applied field ─────────────────────────────────
ax3 = axes[2]
bins_h = np.arange(0, h_vals.max() + 1.5, 0.5)
ax3.hist(h_vals, bins=bins_h, color=C_HIST, alpha=0.80, edgecolor="white")
ax3.set_title(TITLE_C, fontsize=11, pad=6)
ax3.set_xlabel(XLABEL_C, fontsize=10)
ax3.tick_params(labelsize=9)
ax3.text(0.04, 0.96, "C", transform=ax3.transAxes, fontsize=13,
         fontweight="bold", va="top")

# Shared legend for target lines
target_line  = mlines.Line2D([], [], color=C_TARGET, ls="-",  lw=1.8, label="Conditioning targets")
room_line    = mlines.Line2D([], [], color=C_ROOM,   ls="-.", lw=2.0, label="Room temp (300 K)")
extrap_patch = plt.Rectangle((0,0), 1, 1, fc=C_SHADE, alpha=0.4, label="Extrapolation zone")
fig.legend(handles=[target_line, room_line, extrap_patch],
           loc="lower center", ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

fig.suptitle(SUPTITLE, fontsize=12, y=1.01)
plt.tight_layout(pad=1.5)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIGURES}/fig08_property_distributions.{ext}", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIGURES}/fig08_property_distributions.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig08 — ΔS n={len(ds_vals)}, T_C n={len(tc_vals)}, H n={len(h_vals)}")
plt.close()
