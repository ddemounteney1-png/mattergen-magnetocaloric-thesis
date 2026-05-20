import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# EDITABLE: Metrics for each model
# MatterGen: MP2020correction hull | Fine-tuned: TRI2024 hull
# ============================================================
MODELS = {
    "MatterGen\n(full baseline)": {
        "stable": 0.7329, "unique": 0.9990, "novel": 0.6165, "rmsd": 0.0629,
        "colour": "#009E73",
    },
    "MatterGen\n(MP-20)": {
        "stable": 0.4705, "unique": 0.9978, "novel": 0.7598, "rmsd": 0.1472,
        "colour": "#56B4E9",
    },
    "My base model\n(unconditional,\nn=1008)": {
        "stable": 0.3859, "unique": 0.9980, "novel": 0.8095, "rmsd": 0.2068,
        "colour": "#E69F00",
    },
}

# ============================================================
# EDITABLE: Panel titles and y-axis limits
# ============================================================
PANELS = [
    ("stable", "Fraction stable\n(<0.1 eV/atom above hull)", (0, 0.85), "A"),
    ("unique", "Fraction unique",                            (0, 1.08), "B"),
    ("novel",  "Fraction novel",                             (0, 1.08), "C"),
    ("rmsd",   "Avg. RMSD from DFT relaxation (Å)",         (0, 0.20), "D"),
]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 4, figsize=(12.0, 4.2))
fig.subplots_adjust(wspace=0.48)
fig.suptitle("Base Model vs MatterGen Baseline: Generation Quality Metrics",
             fontsize=12, y=1.02)

model_names = list(MODELS.keys())
x = np.arange(len(model_names))
colours = [MODELS[m]["colour"] for m in model_names]

for ax, (metric, ylabel, ylim, panel_letter) in zip(axes, PANELS):
    vals = [MODELS[m][metric] for m in model_names]
    bars = ax.bar(x, vals, color=colours, edgecolor="white",
                  linewidth=0.7, alpha=0.88, width=0.55)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + ylim[1] * 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=8)
    ax.text(0.04, 0.97, panel_letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    # Grey dashed MatterGen reference line
    ref = MODELS[list(MODELS.keys())[0]][metric]
    ax.axhline(ref, color="#009E73", lw=1.0, ls="--", alpha=0.55, zorder=0)

# Footnote on evaluation framework difference
fig.text(0.01, -0.05,
         "Note: MatterGen stability evaluated using MP2020correction; "
         "base model uses TRI2024correction. RMSD comparison is framework-independent.",
         fontsize=7.5, color="#666666")

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"figures/fig18_finetune_vs_mattergen.{ext}", dpi=300, bbox_inches="tight")
print("Saved fig18 — 4-panel comparison: fine-tuned models vs MatterGen")
plt.close()
