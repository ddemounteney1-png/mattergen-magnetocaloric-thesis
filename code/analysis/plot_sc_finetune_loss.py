import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES, LOSS_CURVES  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress

CSV_PATH = f"{LOSS_CURVES}/finetune_sc_losses.csv"
OUT_DIR  = str(FIGURES)

df   = pd.read_csv(CSV_PATH)
ep   = df["epoch"].values
loss = df["loss_train_epoch"].values

WINDOW = 50

def smooth(y, w=WINDOW):
    return uniform_filter1d(y.astype(float), size=w, mode="nearest")

ma = smooth(loss)

# Convergence: linear regression on last 200 smoothed epochs
slope, _, r, _, _ = linregress(ep[-200:], ma[-200:])
r2 = r**2

# 100-epoch window means
window_means = [(s, loss[s:s+100].mean()) for s in range(0, 1000, 100)]

C_TOTAL  = "#0072B2"
C_POS    = "#E69F00"
C_CELL   = "#009E73"
C_ATOMIC = "#CC79A7"
C_MA     = "#D55E00"

min_idx = np.argmin(loss)

plt.style.use("seaborn-v0_8-paper")
fig, axes = plt.subplots(3, 1, figsize=(9, 10),
                         gridspec_kw={"height_ratios": [2.5, 1, 1]})

# ── Panel 1: total loss ────────────────────────────────────────────────────
ax = axes[0]
ax.plot(ep, loss, color=C_TOTAL, alpha=0.20, linewidth=0.6, label="Loss (raw)")
ax.plot(ep, ma,   color=C_MA,   linewidth=2.0, label=f"Moving avg (w={WINDOW})")

ax.annotate(
    f"Min: {loss[min_idx]:.4f}\n(epoch {ep[min_idx]})",
    xy=(ep[min_idx], loss[min_idx]),
    xytext=(ep[min_idx] + 80, loss[min_idx] + 0.012),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=9,
)
ax.annotate(
    f"Final: {loss[-1]:.4f}\n(epoch 999)",
    xy=(ep[-1], loss[-1]),
    xytext=(ep[-1] - 220, loss[-1] + 0.012),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=9,
)

ax.set_ylabel("Total training loss", fontsize=13)
ax.set_title(
    "Superconductor fine-tune — 1000 epochs  |  1× A100\n"
    "1000 train structures  |  125 batches/epoch  |  initialised from base pre-train",
    fontsize=11, pad=8
)
ax.legend(fontsize=10)
ax.tick_params(labelsize=10)
ax.set_xlim(0, 999)

# ── Panel 2: component losses ──────────────────────────────────────────────
ax2 = axes[1]
components = {
    "pos":          ("pos_train_epoch",           C_POS),
    "cell":         ("cell_train_epoch",           C_CELL),
    "atomic num.":  ("atomic_numbers_train_epoch", C_ATOMIC),
}
for label, (col, color) in components.items():
    raw = df[col].values
    ax2.plot(ep, raw,         color=color, alpha=0.15, linewidth=0.6)
    ax2.plot(ep, smooth(raw), color=color, linewidth=1.8, label=label)

ax2.set_ylabel("Component loss", fontsize=12)
ax2.legend(fontsize=9)
ax2.tick_params(labelsize=10)
ax2.set_xlim(0, 999)

# ── Panel 3: 100-epoch window means ───────────────────────────────────────
ax3 = axes[2]
win_starts = [s for s, _ in window_means]
win_vals   = [v for _, v in window_means]
bars = ax3.bar(win_starts, win_vals, width=90, align="edge",
               color=C_TOTAL, alpha=0.75, edgecolor="white")
for bar, val in zip(bars, win_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.0008,
             f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)

ax3.set_xlabel("Epoch", fontsize=13)
ax3.set_ylabel("Mean loss\n(100-ep window)", fontsize=11)
ax3.tick_params(labelsize=10)
ax3.set_xlim(0, 999)

for a in axes:
    a.xaxis.set_major_locator(ticker.MultipleLocator(100))
    a.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    a.grid(which="major", linestyle="--", alpha=0.4)
    a.grid(which="minor", linestyle=":",  alpha=0.2)

fig.tight_layout(h_pad=3)

for ext in ("png", "pdf"):
    path = f"{OUT_DIR}/finetune_sc_loss_curve.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

# ── Convergence report ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("CONVERGENCE ANALYSIS — Superconductor fine-tune")
print("="*60)
print(f"Epoch 0 loss    : {loss[0]:.4f}")
print(f"Epoch 999 loss  : {loss[-1]:.4f}")
print(f"Global min      : {loss.min():.4f}  (epoch {ep[min_idx]})")
print(f"Total drop      : {loss[0] - loss[-1]:.4f} ({(loss[0]-loss[-1])/loss[0]*100:.1f}%)")
print()
print("100-epoch window means:")
for i, (s, v) in enumerate(window_means):
    delta = f"  Δ={v - window_means[i-1][1]:+.5f}" if i > 0 else ""
    print(f"  Epochs {s:3d}–{s+99:3d}: {v:.5f}{delta}")
print()
print(f"Last-200-ep smoothed slope : {slope:.7f} loss/epoch  (R²={r2:.4f})")
print(f"First-100 mean : {loss[:100].mean():.4f}")
print(f"Mid (450-550)  : {loss[450:550].mean():.4f}")
print(f"Last-100 mean  : {loss[-100:].mean():.4f}")
print(f"Mid→End Δ      : {loss[-100:].mean() - loss[450:550].mean():+.4f}")
print(f"\nNote: min at epoch {ep[min_idx]}, final loss {((loss[-1]-loss.min())/loss.min()*100):.1f}% above min")
