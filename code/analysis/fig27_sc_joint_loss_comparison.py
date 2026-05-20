import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import os

# ============================================================
# EDITABLE: Loss CSV paths
# ============================================================
SC_CSV    = "finetune_sc_losses.csv"
JOINT_CSV = "finetune_joint_losses.csv"

# ============================================================
# EDITABLE: Smoothing window (epochs)
# ============================================================
SMOOTH_WINDOW = 50

# ============================================================
# EDITABLE: Colours
# ============================================================
C_SC    = "#009E73"   # green — superconductor
C_JOINT = "#CC79A7"   # pink  — joint ΔS+TC

# ── Load ──────────────────────────────────────────────────────
sc    = pd.read_csv(SC_CSV)
joint = pd.read_csv(JOINT_CSV)

sc_raw    = sc["loss_train_epoch"].values
joint_raw = joint["loss_train_epoch"].values
sc_smooth    = uniform_filter1d(sc_raw,    size=SMOOTH_WINDOW)
joint_smooth = uniform_filter1d(joint_raw, size=SMOOTH_WINDOW)

sc_ep    = sc["epoch"].values
joint_ep = joint["epoch"].values

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8.0, 4.5))

# Raw (faint)
ax.plot(sc_ep,    sc_raw,    color=C_SC,    alpha=0.15, lw=0.7)
ax.plot(joint_ep, joint_raw, color=C_JOINT, alpha=0.15, lw=0.7)

# Smoothed
ax.plot(sc_ep,    sc_smooth,    color=C_SC,    lw=2.0,
        label=f"Superconductor fine-tune  (start={sc_raw[0]:.3f}, end={sc_smooth[-1]:.3f}, "
              f"min={sc_raw.min():.3f} @ ep {sc_ep[sc_raw.argmin()]})")
ax.plot(joint_ep, joint_smooth, color=C_JOINT, lw=2.0,
        label=f"Joint $\\Delta S$+$T_C$ fine-tune  (start={joint_raw[0]:.3f}, end={joint_smooth[-1]:.3f}, "
              f"min={joint_raw.min():.3f} @ ep {joint_ep[joint_raw.argmin()]})")

# Min markers
ax.scatter(sc_ep[sc_raw.argmin()],    sc_raw.min(),    marker="v", s=80,
           color=C_SC,    zorder=5, edgecolors="white", linewidth=0.8)
ax.scatter(joint_ep[joint_raw.argmin()], joint_raw.min(), marker="v", s=80,
           color=C_JOINT, zorder=5, edgecolors="white", linewidth=0.8)
ax.text(sc_ep[sc_raw.argmin()] + 15,    sc_raw.min() - 0.015,
        f"min {sc_raw.min():.3f}", fontsize=7.5, color=C_SC)
ax.text(joint_ep[joint_raw.argmin()] + 15, joint_raw.min() - 0.03,
        f"min {joint_raw.min():.3f}", fontsize=7.5, color=C_JOINT)

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Total training loss", fontsize=11)
ax.set_title(
    "Training Loss: Superconductor vs Joint $\\Delta S$+$T_C$ Fine-Tune\n"
    f"(faint = raw per-epoch; solid = {SMOOTH_WINDOW}-epoch moving average; ▼ = minimum)",
    fontsize=10, pad=8)
ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
ax.tick_params(labelsize=9)
ax.set_xlim(0, max(sc_ep[-1], joint_ep[-1]))

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"figures/fig27_sc_joint_loss_comparison.{ext}", dpi=300, bbox_inches="tight")
print("Saved fig27 — SC vs joint fine-tune loss comparison")
plt.close()
