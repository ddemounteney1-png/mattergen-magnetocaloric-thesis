import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES, FINETUNE_DATA  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

TRAIN_CSV = f"{FINETUNE_DATA}/train_repaired.csv"
VAL_CSV   = f"{FINETUNE_DATA}/val_repaired.csv"
OUT_DIR   = str(FIGURES)

train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV)

# Drop val rows that are entirely empty (rows 16-30 are blank placeholder rows)
val_atoms = val["n_atoms_primitive"].dropna()
val_ds    = val["magnetocaloric_delta_s"].dropna()
train_atoms = train["n_atoms_primitive"]
train_ds    = train["magnetocaloric_delta_s"].dropna()

# Okabe-Ito colours
C_TRAIN = "#0072B2"   # blue
C_VAL   = "#E69F00"   # orange
C_LINE20 = "#009E73"  # green
C_LINE51 = "#D55E00"  # red-orange

plt.style.use("seaborn-v0_8-paper")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ═══════════════════════════════════════════════════════════════════════════
# Panel 1 — Atom count histogram
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[0]

bins = np.arange(0.5, 105.5, 2)   # 2-atom wide bins up to 104

ax.hist(train_atoms, bins=bins, color=C_TRAIN, alpha=0.75,
        label=f"Train (n={len(train_atoms)})", zorder=2)
ax.hist(val_atoms,   bins=bins, color=C_VAL,   alpha=0.85,
        label=f"Val (n={len(val_atoms)})", zorder=3)

# Ceiling lines
ax.axvline(20, color=C_LINE20, linewidth=2.0, linestyle="--",
           label="MatterGen published ceiling (20 atoms)", zorder=4)
ax.axvline(51, color=C_LINE51, linewidth=2.0, linestyle="-",
           label="This work ceiling (51 atoms)", zorder=4)

# Shade region above 51
ax.axvspan(51, bins[-1], alpha=0.08, color=C_LINE51, zorder=1,
           label=f"Above ceiling ({(train_atoms > 51).sum()} train structs)")

ax.set_xlabel("Number of atoms in primitive cell", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Atom count distribution\nMagnetocaloric fine-tune dataset", fontsize=12)
ax.legend(fontsize=8.5, loc="upper right")
ax.tick_params(labelsize=10)
ax.set_xlim(0, 105)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(which="major", linestyle="--", alpha=0.4)
ax.grid(which="minor", linestyle=":",  alpha=0.2)

# Annotate median
med = train_atoms.median()
ax.axvline(med, color=C_TRAIN, linewidth=1.2, linestyle=":",
           alpha=0.8, zorder=3)
ax.text(med + 1, ax.get_ylim()[1] * 0.92, f"Median\n{med:.0f} atoms",
        color=C_TRAIN, fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════
# Panel 2 — ΔS histogram
# ═══════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

# Log-spaced bins to handle the heavy right tail
ds_all   = pd.concat([train_ds, val_ds])
ds_max   = ds_all.max()
bins_ds  = np.linspace(0, ds_max + 2, 35)

ax2.hist(train_ds, bins=bins_ds, color=C_TRAIN, alpha=0.75,
         label=f"Train (n={len(train_ds)})", zorder=2)
ax2.hist(val_ds,   bins=bins_ds, color=C_VAL,   alpha=0.85,
         label=f"Val (n={len(val_ds)})", zorder=3)

# Median and mean lines
for ds, color, split in [(train_ds, C_TRAIN, "Train"), (val_ds, C_VAL, "Val")]:
    ax2.axvline(ds.median(), color=color, linewidth=1.5, linestyle="--",
                alpha=0.9, zorder=4)
    ax2.axvline(ds.mean(),   color=color, linewidth=1.5, linestyle=":",
                alpha=0.9, zorder=4)

# Custom legend entries for lines
from matplotlib.lines import Line2D
extra = [
    Line2D([0], [0], color="grey", lw=1.5, linestyle="--", label="Median"),
    Line2D([0], [0], color="grey", lw=1.5, linestyle=":",  label="Mean"),
]
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles + extra, fontsize=8.5, loc="upper right")

ax2.set_xlabel("ΔS (J kg⁻¹ K⁻¹)", fontsize=13)
ax2.set_ylabel("Count", fontsize=13)
ax2.set_title("Magnetocaloric entropy change (ΔS)\nMagnetocaloric fine-tune dataset", fontsize=12)
ax2.tick_params(labelsize=10)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax2.grid(which="major", linestyle="--", alpha=0.4)
ax2.grid(which="minor", linestyle=":",  alpha=0.2)

fig.tight_layout(pad=2.5)

for ext in ("png", "pdf"):
    path = f"{OUT_DIR}/finetune_dataset_histograms.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

# ── Print summary stats ────────────────────────────────────────────────────
print("\nATOM COUNT — train")
print(train_atoms.describe().round(2).to_string())
print(f"  > 51 atoms : {(train_atoms > 51).sum()} structures")
print(f"  > 20 atoms : {(train_atoms > 20).sum()} structures ({(train_atoms>20).mean()*100:.1f}%)")

print("\nATOM COUNT — val")
print(val_atoms.describe().round(2).to_string())

print("\nΔS — train")
print(train_ds.describe().round(3).to_string())
print(f"  Skewness   : {train_ds.skew():.3f}")

print("\nΔS — val")
print(val_ds.describe().round(3).to_string())

print(f"\nNOTE: Val CSV has {len(val)} rows; {val['n_atoms_primitive'].isna().sum()} "
      f"are entirely blank (padding rows). Effective val size = {len(val_atoms)}.")
