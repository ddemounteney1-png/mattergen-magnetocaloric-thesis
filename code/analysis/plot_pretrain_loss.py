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

CSV_PATH = f"{LOSS_CURVES}/pretrain_losses.csv"
OUT_DIR  = str(FIGURES)

df = pd.read_csv(CSV_PATH)
ep   = df["epoch"].values
loss = df["loss_train_epoch"].values

WINDOW = 50

def smooth(y, w=WINDOW):
    return uniform_filter1d(y.astype(float), size=w, mode="nearest")

# Identify spike(s): epochs where loss > 3× the overall median
median_loss = np.median(loss)
spike_mask  = loss > 3 * median_loss
spike_epochs = ep[spike_mask]

# Convergence metrics
# Rate of improvement: mean loss in each 100-epoch window
windows = [(i, i+100) for i in range(0, 1000, 100)]
window_means = [(s, loss[s:e].mean()) for s, e in windows]

# Linear regression on last 200 epochs (smoothed) to get slope
ma = smooth(loss)
last200_ep   = ep[-200:]
last200_ma   = ma[-200:]
slope, intercept, r2, _, _ = linregress(last200_ep, last200_ma)
r2 = r2**2  # linregress returns r, not r²

# Improvement between first and last 100 epochs
first100_mean = loss[:100].mean()
last100_mean  = loss[-100:].mean()
mid_mean      = loss[450:550].mean()  # epochs 450–550

# ── Okabe-Ito colours ─────────────────────────────────────────────────────
C_TOTAL  = "#0072B2"
C_POS    = "#E69F00"
C_CELL   = "#009E73"
C_ATOMIC = "#CC79A7"
C_MA     = "#D55E00"
C_SPIKE  = "#E62728"

plt.style.use("seaborn-v0_8-paper")
fig, axes = plt.subplots(3, 1, figsize=(9, 10),
                         gridspec_kw={"height_ratios": [2.5, 1, 1]})

# ── Panel 1: total loss (log scale) ───────────────────────────────────────
ax = axes[0]
ax.semilogy(ep, loss, color=C_TOTAL, alpha=0.20, linewidth=0.6, label="Loss (raw)")
ax.semilogy(ep, ma,   color=C_MA,   linewidth=2.0,
            label=f"Moving avg (w={WINDOW})")

# mark spikes
if len(spike_epochs):
    ax.semilogy(spike_epochs, loss[spike_mask], "x", color=C_SPIKE,
                markersize=7, markeredgewidth=1.5, label=f"Spike (>{3:.0f}× median)")

# annotate final value
ax.annotate(
    f"Final: {loss[-1]:.3f}\n(epoch 999)",
    xy=(ep[-1], loss[-1]),
    xytext=(ep[-1] - 200, loss[-1] * 1.6),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=9,
)
ax.annotate(
    f"Min: {loss.min():.3f}\n(epoch {ep[np.argmin(loss)]})",
    xy=(ep[np.argmin(loss)], loss.min()),
    xytext=(ep[np.argmin(loss)] - 200, loss.min() * 0.7),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=9,
)

ax.set_ylabel("Total training loss (log scale)", fontsize=13)
ax.set_title(
    "Base pre-training — 1000 epochs  |  4× A100  |  274 k structures\n"
    "GemNetT 44.6 M params, 1717 batches/epoch (batch size 32 per GPU)",
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
    ax2.plot(ep, raw,         color=color, alpha=0.15, linewidth=1.0)
    ax2.plot(ep, smooth(raw), color=color, linewidth=0.8, label=label)

ax2.set_ylabel("Component loss", fontsize=12)
ax2.legend(fontsize=9)
ax2.tick_params(labelsize=10)
ax2.set_xlim(0, 999)

# ── Panel 3: 100-epoch window means (convergence diagnostic) ──────────────
ax3 = axes[2]
win_starts = [s for s, _ in window_means]
win_vals   = [v for _, v in window_means]
bars = ax3.bar(win_starts, win_vals, width=90, align="edge",
               color=C_TOTAL, alpha=0.75, edgecolor="white")
for bar, val in zip(bars, win_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

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
    path = f"{OUT_DIR}/pretrain_loss_curve.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

# ── Convergence report ────────────────────────────────────────────────────
print("\n" + "="*60)
print("CONVERGENCE ANALYSIS")
print("="*60)
print(f"Epoch 0 loss          : {loss[0]:.4f}")
print(f"Epoch 999 loss        : {loss[-1]:.4f}")
print(f"Global min loss       : {loss.min():.4f}  (epoch {ep[np.argmin(loss)]})")
print(f"Spike epochs (>{3:.0f}×med) : {list(spike_epochs)}")
print()
print("100-epoch window means:")
for s, v in window_means:
    delta = ""
    if s > 0:
        prev = window_means[s//100 - 1][1]
        delta = f"  Δ={v-prev:+.4f}"
    print(f"  Epochs {s:3d}–{s+99:3d}: {v:.4f}{delta}")
print()
print(f"Last-200-epoch smoothed slope : {slope:.6f} loss/epoch")
print(f"  (R² = {r2:.4f})")
print(f"First-100 mean  : {first100_mean:.4f}")
print(f"Mid (450-550)   : {mid_mean:.4f}")
print(f"Last-100 mean   : {last100_mean:.4f}")
print(f"Mid→End Δ       : {last100_mean - mid_mean:+.4f}")
