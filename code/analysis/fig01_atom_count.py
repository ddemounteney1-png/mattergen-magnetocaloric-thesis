import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES, FINETUNE_DATA  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ============================================================
# EDITABLE: Titles and labels
# ============================================================
TITLE  = "Atom-Count Distribution of Magnetocaloric Fine-Tuning Dataset"
XLABEL = "$n_{atoms}$ (atoms in primitive cell)"
YLABEL = "Count"
VLINE_LABEL_20 = "Published MatterGen ceiling (20 atoms)"
VLINE_LABEL_51 = "Custom ceiling (51 atoms)"

# ============================================================
# EDITABLE: Annotation positions — (x_data, y_data, x_offset_pts, y_offset_pts)
# ============================================================
ANNOTATIONS = {
    "La(Fe,Si)$_{13}$ family\n(SG 226, 28 atoms)": {
        "xy_data": (28, 18), "offset": (-80, 48), "arrow": True},
    "Gd$_5$(Si,Ge)$_4$\n(SG 14, 36 atoms)": {
        "xy_data": (36, 10), "offset": (120, 28), "arrow": True},
    "RE$_2$Fe$_{17}$\n(SG 194, 38 atoms)": {
        "xy_data": (38, 6),  "offset": (100, 0), "arrow": True},
}

# ============================================================
# EDITABLE: Colours
# ============================================================
C_HIST  = "#0072B2"
C_V20   = "#D55E00"
C_V51   = "#009E73"
C_SHADE = "#81CACAE6"

# ============================================================
# Load data — combine train and val into one distribution
# ============================================================
train = pd.read_csv(f"{FINETUNE_DATA}/train_repaired.csv")
val   = pd.read_csv(f"{FINETUNE_DATA}/val_repaired.csv").dropna(subset=["n_atoms_primitive"])
all_n = pd.concat([train["n_atoms_primitive"], val["n_atoms_primitive"]]).reset_index(drop=True)
n_total = len(all_n)

bins = np.arange(0.5, 105.5, 2)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7.0, 4.5))

# Shade region above 20
ax.axvspan(20, 105, alpha=0.07, color=C_SHADE, zorder=0,
           label="Excluded by published model (>20 atoms)")

ax.hist(all_n, bins=bins, color=C_HIST, alpha=0.80,
        label=f"All structures", zorder=2)

ax.axvline(20, color=C_V20, lw=4.0, ls="--", label=VLINE_LABEL_20, zorder=4)
ax.axvline(50, color=C_V51, lw=4.0, ls="-",  label=VLINE_LABEL_51, zorder=4)

# Annotations
for text, cfg in ANNOTATIONS.items():
    x0, y0 = cfg["xy_data"]
    dx, dy  = cfg["offset"]
    arrowprops = dict(arrowstyle="->", color="#F25555", lw=1.0) if cfg["arrow"] else None
    ax.annotate(text, xy=(x0, y0), xytext=(dx, dy),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.5, color="#222222",
                arrowprops=arrowprops)

ax.set_title(TITLE, fontsize=12, pad=10)
ax.set_xlabel(XLABEL, fontsize=11)
ax.set_ylabel(YLABEL, fontsize=11)
ax.set_xlim(0, 102)
ax.tick_params(labelsize=10)
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout(pad=1.5)

os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIGURES}/fig01_atom_count_distribution.{ext}", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIGURES}/fig01_atom_count_distribution.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig01 — n_total={n_total}, structures >20 atoms: {(all_n > 20).sum()}")
plt.close()
