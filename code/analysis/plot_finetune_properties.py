import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

TRAIN_CSV = "/home/daniel/Data_analysis/train_repaired.csv"
VAL_CSV   = "/home/daniel/Data_analysis/val_repaired.csv"
OUT_DIR   = "/home/daniel/Data_analysis/figures"

train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV).dropna(subset=["n_atoms_primitive"])  # drop blank padding rows

train_ds = train["magnetocaloric_delta_s"].dropna()
val_ds   = val["magnetocaloric_delta_s"].dropna()
train_tc = train["magnetocaloric_curie_temp"].dropna()
val_tc   = val["magnetocaloric_curie_temp"].dropna()

C_TRAIN = "#0072B2"
C_VAL   = "#E69F00"

plt.style.use("seaborn-v0_8-paper")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

def add_stat_lines(ax, train_s, val_s, color_t, color_v):
    """Draw median (dashed) and mean (dotted) lines for train and val."""
    for series, color in [(train_s, color_t), (val_s, color_v)]:
        ax.axvline(series.median(), color=color, lw=1.6, ls="--", alpha=0.9, zorder=4)
        ax.axvline(series.mean(),   color=color, lw=1.6, ls=":",  alpha=0.9, zorder=4)

def stat_legend_extras():
    return [
        Line2D([0], [0], color="grey", lw=1.6, ls="--", label="Median"),
        Line2D([0], [0], color="grey", lw=1.6, ls=":",  label="Mean"),
    ]

# ── Panel 1: ΔS ───────────────────────────────────────────────────────────
ax = axes[0]
ds_max  = max(train_ds.max(), val_ds.max())
bins_ds = np.linspace(0, ds_max + 2, 36)

ax.hist(train_ds, bins=bins_ds, color=C_TRAIN, alpha=0.72,
        label=f"Train (n={len(train_ds)})", zorder=2)
ax.hist(val_ds,   bins=bins_ds, color=C_VAL,   alpha=0.85,
        label=f"Val (n={len(val_ds)})", zorder=3)

add_stat_lines(ax, train_ds, val_ds, C_TRAIN, C_VAL)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + stat_legend_extras(), fontsize=9, loc="upper right")

ax.set_xlabel("ΔS  (J kg⁻¹ K⁻¹)", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Magnetocaloric entropy change (ΔS)\nFine-tune dataset — train & val", fontsize=12)
ax.tick_params(labelsize=10)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(which="major", ls="--", alpha=0.4)
ax.grid(which="minor", ls=":",  alpha=0.2)

# ── Panel 2: Curie temperature ────────────────────────────────────────────
ax2 = axes[1]
tc_min = min(train_tc.min(), val_tc.min())
tc_max = max(train_tc.max(), val_tc.max())
bins_tc = np.linspace(tc_min - 10, tc_max + 10, 36)

ax2.hist(train_tc, bins=bins_tc, color=C_TRAIN, alpha=0.72,
         label=f"Train (n={len(train_tc)})", zorder=2)
ax2.hist(val_tc,   bins=bins_tc, color=C_VAL,   alpha=0.85,
         label=f"Val (n={len(val_tc)})", zorder=3)

add_stat_lines(ax2, train_tc, val_tc, C_TRAIN, C_VAL)

# Mark room temperature
ax2.axvline(293, color="#CC79A7", lw=1.8, ls="-.", zorder=4, label="Room temp (293 K)")

handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles=handles2 + stat_legend_extras(), fontsize=9, loc="upper left")

ax2.set_xlabel("Curie temperature T$_C$  (K)", fontsize=13)
ax2.set_ylabel("Count", fontsize=13)
ax2.set_title("Curie temperature (T$_C$)\nFine-tune dataset — train & val", fontsize=12)
ax2.tick_params(labelsize=10)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(25))
ax2.grid(which="major", ls="--", alpha=0.4)
ax2.grid(which="minor", ls=":",  alpha=0.2)

fig.tight_layout(pad=2.5)

for ext in ("png", "pdf"):
    path = f"{OUT_DIR}/finetune_property_distributions.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

# ── Summary stats ──────────────────────────────────────────────────────────
for name, t, v in [("ΔS (J/kg/K)", train_ds, val_ds),
                   ("T_C (K)",     train_tc, val_tc)]:
    print(f"\n{name}")
    print(f"  Train  n={len(t):3d}  min={t.min():.1f}  median={t.median():.1f}"
          f"  mean={t.mean():.1f}  max={t.max():.1f}  std={t.std():.1f}  skew={t.skew():.2f}")
    print(f"  Val    n={len(v):3d}  min={v.min():.1f}  median={v.median():.1f}"
          f"  mean={v.mean():.1f}  max={v.max():.1f}  std={v.std():.1f}  skew={v.skew():.2f}")
