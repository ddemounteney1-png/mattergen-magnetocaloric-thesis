import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress

CSV_PATH = "/home/daniel/Data_analysis/finetune_joint_losses.csv"
OUT_DIR  = "/home/daniel/Data_analysis/figures"

df   = pd.read_csv(CSV_PATH)
ep   = df["epoch"].values
loss = df["loss_train_epoch"].values

WINDOW = 50

def smooth(y, w=WINDOW):
    return uniform_filter1d(y.astype(float), size=w, mode="nearest")

ma = smooth(loss)
slope, _, r, _, _ = linregress(ep[-200:], ma[-200:])
r2 = r**2

window_means = [(s, loss[s:s+100].mean()) for s in range(0, 1000, 100)]
min_idx = np.argmin(loss)

C_TOTAL  = "#0072B2"
C_POS    = "#E69F00"
C_CELL   = "#009E73"
C_ATOMIC = "#CC79A7"
C_MA     = "#D55E00"
C_MIN    = "#D55E00"

plt.style.use("seaborn-v0_8-paper")
fig, axes = plt.subplots(3, 1, figsize=(9, 10),
                         gridspec_kw={"height_ratios": [2.5, 1, 1]})

# ── Panel 1: total loss ────────────────────────────────────────────────────
ax = axes[0]
ax.plot(ep, loss, color=C_TOTAL, alpha=0.20, linewidth=0.6, label="Loss (raw)")
ax.plot(ep, ma,   color=C_MA,   linewidth=2.0, label=f"Moving avg (w={WINDOW})")

# Shade the region beyond the minimum — highlight drift-up
ax.axvspan(ep[min_idx], ep[-1], alpha=0.06, color="red",
           label=f"Post-minimum drift (+{(loss[-1]-loss.min())/loss.min()*100:.0f}% above min)")

ax.annotate(
    f"Best: {loss[min_idx]:.4f}\n(epoch {ep[min_idx]})",
    xy=(ep[min_idx], loss[min_idx]),
    xytext=(ep[min_idx] - 200, loss[min_idx] - 0.04),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=9,
)
ax.annotate(
    f"Final: {loss[-1]:.4f}\n(epoch 999)",
    xy=(ep[-1], loss[-1]),
    xytext=(ep[-1] - 230, loss[-1] + 0.03),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=9,
)

ax.set_ylabel("Total training loss", fontsize=13)
ax.set_title(
    "Joint magnetocaloric fine-tune — 1000 epochs  |  1× A100\n"
    "Conditioning: ΔS + T$_C$  |  291 train structures  |  37 batches/epoch",
    fontsize=11, pad=8
)
ax.legend(fontsize=9.5, loc="upper right")
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
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.001,
             f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax3.set_xlabel("Epoch", fontsize=13)
ax3.set_ylabel("Mean loss\n(100-ep window)", fontsize=11)
ax3.tick_params(labelsize=10)
ax3.set_xlim(0, 999)

for a in axes:
    a.xaxis.set_major_locator(ticker.MultipleLocator(100))
    a.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    a.grid(which="major", ls="--", alpha=0.4)
    a.grid(which="minor", ls=":",  alpha=0.2)

fig.tight_layout(h_pad=3)

for ext in ("png", "pdf"):
    path = f"{OUT_DIR}/finetune_joint_loss_curve.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
