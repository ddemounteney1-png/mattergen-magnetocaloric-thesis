import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES, FINETUNE_DATA  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os

# ============================================================
# EDITABLE: Metrics from mattergen-evaluate (ΔS-only model)
# ============================================================
DS_VALUES = [5.3, 18.9, 32.5, 46.1, 59.7]
STABLE    = [0.2452, 0.3798, 0.3654, 0.4471, 0.3510]
UNIQUE    = [0.9663, 0.9519, 0.9471, 0.9135, 0.9231]
NOVEL     = [0.7596, 0.7067, 0.7260, 0.6635, 0.7404]
SUN       = [0.0913, 0.1731, 0.1490, 0.1587, 0.1346]

# ============================================================
# EDITABLE: Sigma reference positions and labels
# ============================================================
SIGMA_LEVELS = {
    "Median": 5.3, "+1σ": 18.9, "+2σ": 32.5, "+3σ": 46.1, "+4σ": 59.7
}

# ============================================================
# EDITABLE: Colours
# ============================================================
C_STABLE = "#0072B2"
C_UNIQUE = "#009E73"
C_NOVEL  = "#E69F00"
C_SUN    = "#CC79A7"
C_HIST   = "#0072B2"
C_SIGMA  = "#D55E00"
C_SHADE  = "#FFD580"

# ── Load ΔS training distribution ────────────────────────────
train = pd.read_csv(f"{FINETUNE_DATA}/train_repaired.csv")
val   = pd.read_csv(f"{FINETUNE_DATA}/val_repaired.csv").dropna(subset=["n_atoms_primitive"])
ds_vals = pd.to_numeric(
    pd.concat([train, val], ignore_index=True)["magnetocaloric_delta_s"],
    errors="coerce"
).dropna()

X_MAX = 65

# ── Figure layout ─────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(7.5, 6.5))
gs  = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.08)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

pct = lambda x: [v * 100 for v in x]

# ── Top panel: generation quality lines ───────────────────────
ax_top.plot(DS_VALUES, pct(STABLE), marker="o", ms=7, lw=2.0,
            color=C_STABLE, label="Stable (<0.1 eV/atom above hull)", zorder=3)
ax_top.plot(DS_VALUES, pct(UNIQUE), marker="s", ms=7, lw=2.0,
            color=C_UNIQUE, label="Unique", zorder=3)
ax_top.plot(DS_VALUES, pct(NOVEL),  marker="^", ms=7, lw=2.0,
            color=C_NOVEL,  label="Novel",  zorder=3)
ax_top.plot(DS_VALUES, pct(SUN),    marker="D", ms=7, lw=2.0,
            color=C_SUN,    label="S.U.N. (Stable · Unique · Novel)", zorder=3)

for label, x in SIGMA_LEVELS.items():
    ax_top.axvline(x, color=C_SIGMA, lw=0.9, ls="--", alpha=0.45, zorder=0)
    ax_top.text(x, 101.5, label, ha="center", va="bottom",
                fontsize=8, color=C_SIGMA)

ax_top.set_ylabel("Percentage (%)", fontsize=11)
ax_top.set_title(
    "Generation Quality vs $\\Delta S_{mag}$ Conditioning (ΔS-only model)",
    fontsize=11, pad=10)
ax_top.set_ylim(0, 108)
ax_top.set_xlim(0, X_MAX)
ax_top.legend(fontsize=9, loc="center right")
ax_top.tick_params(labelbottom=False, labelsize=9)
ax_top.text(0.01, 0.97, "A", transform=ax_top.transAxes,
            fontsize=13, fontweight="bold", va="top")

# ── Bottom panel: ΔS training distribution ───────────────────
bins_ds = np.linspace(0, X_MAX, 35)
ax_bot.hist(ds_vals, bins=bins_ds, color=C_HIST, alpha=0.75, edgecolor="white")
ax_bot.axvspan(SIGMA_LEVELS["+2σ"], X_MAX, alpha=0.12, color=C_SHADE,
               label="Extrapolation region (>+2σ)")

for label, x in SIGMA_LEVELS.items():
    ax_bot.axvline(x, color=C_SIGMA, lw=0.9, ls="--", alpha=0.45, zorder=0)

ax_bot.set_xlabel("$\\Delta S_{mag}$ (J kg$^{-1}$ K$^{-1}$)", fontsize=11)
ax_bot.set_ylabel("Count", fontsize=11)
ax_bot.set_title("Training Data $\\Delta S_{mag}$ Distribution", fontsize=10, pad=5)
ax_bot.tick_params(labelsize=9)
ax_bot.text(0.01, 0.95, "B", transform=ax_bot.transAxes,
            fontsize=13, fontweight="bold", va="top")
ax_bot.legend(fontsize=8, loc="upper right")

fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.10)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{FIGURES}/fig21_ds_conditioning_combined.{ext}",
                dpi=300, bbox_inches="tight")
print(f"Saved fig21 — combined ΔS conditioning + training distribution, n_ds={len(ds_vals)}")
plt.close()
