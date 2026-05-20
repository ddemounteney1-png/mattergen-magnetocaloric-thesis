import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import os

# ============================================================
# EDITABLE: Loss milestone values for annotation
# ============================================================
MILESTONES = {
    "pretrain": {"start": 2.460, "end": 0.519, "min": 0.515, "min_ep": 974,
                 "title": "Base Pre-train\n274k structures · 4×A100"},
    "ds_only":  {"start": 1.590, "end": 1.080, "min": 0.864, "min_ep": 843,
                 "title": "$\\Delta S$-only Fine-tune\n284 structures · 1×A100"},
    "joint":    {"start": 1.590, "end": 1.110, "min": 0.912, "min_ep": 769,
                 "title": "Joint $\\Delta S$+$T_C$ Fine-tune\n291 structures · 1×A100"},
    "supercon": {"start": 0.359, "end": 0.275, "min": 0.255, "min_ep": 413,
                 "title": "Superconductor Fine-tune\n1000 structures · 1×A100"},
}

# ============================================================
# EDITABLE: CSV file paths
# ============================================================
CSV_PATHS = {
    "pretrain": "losses_csvs/pretrain_losses.csv",
    "ds_only":  "losses_csvs/finetune_1000_losses.csv",
    "joint":    "finetune_joint_losses.csv",
    "supercon": "finetune_sc_losses.csv",
}

# ============================================================
# EDITABLE: Colours
# ============================================================
COLOURS = {
    "pretrain": "#0072B2",
    "ds_only":  "#E69F00",
    "joint":    "#009E73",
    "supercon": "#CC79A7",
}
SUPTITLE = "Training Convergence Comparison"
MA_WINDOW = 50

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.8))
fig.subplots_adjust(wspace=0.35, top=0.80)
fig.suptitle(SUPTITLE, fontsize=13, y=0.98)

panel_labels = ["A", "B", "C", "D"]
keys = list(MILESTONES.keys())

# Compute shared y-limits per-group: pretrain uses its own scale,
# fine-tunes share a scale so they can be compared
all_losses = {}
for key in keys:
    df = pd.read_csv(CSV_PATHS[key])
    all_losses[key] = df["loss_train_epoch"].values

for idx, key in enumerate(keys):
    ax    = axes[idx]
    ms    = MILESTONES[key]
    loss  = all_losses[key]
    ep    = np.arange(len(loss))
    ma    = uniform_filter1d(loss.astype(float), size=MA_WINDOW, mode="nearest")
    color = COLOURS[key]

    ax.plot(ep, loss, color=color, alpha=0.18, linewidth=0.7)
    ax.plot(ep, ma,   color=color, linewidth=2.0)

    # Annotate minimum
    min_ep = np.argmin(loss)
    ax.scatter([min_ep], [loss[min_ep]], s=50, color=color,
               zorder=4, edgecolors="white", linewidth=0.8)
    ax.annotate(f"min={loss[min_ep]:.3f}\n(ep {min_ep})",
                xy=(min_ep, loss[min_ep]),
                xytext=(0, -28), textcoords="offset points",
                ha="center", fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=color))

    ax.set_title(ms["title"], fontsize=9, pad=6)
    ax.set_xlabel("Epoch", fontsize=9)
    if idx == 0:
        ax.set_ylabel("Total training loss", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.set_xlim(0, len(loss) - 1)

    # Panel label
    ax.text(0.96, 0.96, panel_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")

    # Stats box
    ax.text(0.97, 0.60,
            f"Start: {ms['start']:.3f}\nEnd: {ms['end']:.3f}\nBest: {ms['min']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85))

plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig11_convergence_comparison.{ext}", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig11_convergence_comparison.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print("Saved fig11 — 4-panel convergence comparison")
for key, ms in MILESTONES.items():
    print(f"  {key}: {ms['start']:.3f} → {ms['end']:.3f} (best {ms['min']:.3f} ep {ms['min_ep']})")
plt.close()
