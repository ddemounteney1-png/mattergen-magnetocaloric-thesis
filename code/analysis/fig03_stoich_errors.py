import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ============================================================
# EDITABLE: Titles and labels
# ============================================================
TITLE  = "Stoichiometric Error Distribution in Fine-Tuning Structures"
XLABEL = "Maximum composition error (%)"
YLABEL = "Count"
THRESHOLD_LABEL = "5% acceptance threshold"
LABEL_BEFORE = "Before SQS ordering"
LABEL_AFTER  = "After SQS ordering (final)"

# ============================================================
# EDITABLE: Threshold value
# ============================================================
STOICH_THRESHOLD_PCT = 5.0   # structures above this were reviewed/rejected

# ============================================================
# EDITABLE: Colours
# ============================================================
C_BEFORE    = "#56B4E9"
C_AFTER     = "#0072B2"
C_THRESHOLD = "#D55E00"

# ============================================================
# Load data
# ============================================================
cq = pd.read_csv("Processing_data/cif_quality_report.csv")
before = pd.to_numeric(cq["before_stoich_err_pct"], errors="coerce").dropna()
after  = pd.to_numeric(cq["after_stoich_err_pct"],  errors="coerce").dropna()

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
fig.subplots_adjust(wspace=0.4)

for ax, data, label, color in [
    (axes[0], before, LABEL_BEFORE, C_BEFORE),
    (axes[1], after,  LABEL_AFTER,  C_AFTER),
]:
    # Clip display at 30% (a few extreme outliers exist up to 57%)
    display = data.clip(upper=30)
    bins = np.linspace(0, 30, 25)
    ax.hist(display, bins=bins, color=color, alpha=0.80, edgecolor="white")
    ax.axvline(STOICH_THRESHOLD_PCT, color=C_THRESHOLD, lw=2.0, ls="--",
               label=THRESHOLD_LABEL)
    n_above = (data > STOICH_THRESHOLD_PCT).sum()
    ax.set_title(f"{label}\n(>{STOICH_THRESHOLD_PCT:.0f}%: {n_above})",
                 fontsize=10, pad=6)
    ax.set_xlabel(XLABEL, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="upper right")
    #ax.text(0.04, 0.96, panel, transform=ax.transAxes,
    #        fontsize=13, fontweight="bold", va="top")
    # Annotate stats
    ax.text(0.97, 0.80,
            f"Mean: {data.mean():.2f}%\nMedian: {data.median():.2f}%\nMax: {data.max():.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))

fig.suptitle(TITLE, fontsize=12, y=1.02)
plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig03_stoichiometry_errors.{ext}", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig03_stoichiometry_errors.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig03 — before n={len(before)}, after n={len(after)}, "
      f"above 5% threshold: {(after > 5).sum()}")
plt.close()
