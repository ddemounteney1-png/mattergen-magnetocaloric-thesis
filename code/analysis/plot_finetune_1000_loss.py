import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d

CSV_PATH = "/home/daniel/Data_analysis/finetune_1000_losses.csv"
OUT_DIR  = "/home/daniel/Data_analysis/figures"

df = pd.read_csv(CSV_PATH)

WINDOW = 50  # moving-average window (epochs)

def smooth(y, w=WINDOW):
    return uniform_filter1d(y.astype(float), size=w, mode="nearest")

# Okabe-Ito colours
C_TOTAL  = "#0072B2"  # blue
C_POS    = "#E69F00"  # orange
C_CELL   = "#009E73"  # green
C_ATOMIC = "#CC79A7"  # pink
C_MA     = "#D55E00"  # red-orange for moving average

plt.style.use("seaborn-v0_8-paper")
fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                         gridspec_kw={"height_ratios": [2, 1]})

# ── Top panel: total loss + moving average ─────────────────────────────────
ax = axes[0]
ep = df["epoch"].values
loss = df["loss_train_epoch"].values
ma   = smooth(loss)

ax.plot(ep, loss, color=C_TOTAL, alpha=0.25, linewidth=0.7, label="Loss (raw)")
ax.plot(ep, ma,   color=C_MA,   linewidth=2.0,
        label=f"Moving avg (w={WINDOW})")

# annotate min
min_idx = np.argmin(loss)
ax.annotate(
    f"Min: {loss[min_idx]:.3f}\n(epoch {ep[min_idx]})",
    xy=(ep[min_idx], loss[min_idx]),
    xytext=(ep[min_idx] + 60, loss[min_idx] + 0.05),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    fontsize=9,
)

ax.set_ylabel("Total training loss", fontsize=13)
ax.set_title(
    "Magnetocaloric fine-tune — 1000 epochs\n"
    "(initialised from base pre-train checkpoint)",
    fontsize=12, pad=8
)
ax.legend(fontsize=10)
ax.tick_params(labelsize=10)
ax.set_xlim(0, 999)
ax.set_xlabel("")

# ── Bottom panel: component losses ────────────────────────────────────────
ax2 = axes[1]
components = {
    "pos (position)":     ("pos_train_epoch",             C_POS),
    "cell (lattice)":     ("cell_train_epoch",             C_CELL),
    "atomic numbers":     ("atomic_numbers_train_epoch",   C_ATOMIC),
}

for label, (col, color) in components.items():
    raw = df[col].values
    ax2.plot(ep, raw,        color=color, alpha=0.20, linewidth=0.6)
    ax2.plot(ep, smooth(raw), color=color, linewidth=1.8, label=label)

ax2.set_xlabel("Epoch", fontsize=13)
ax2.set_ylabel("Component loss", fontsize=13)
ax2.legend(fontsize=9)
ax2.tick_params(labelsize=10)
ax2.set_xlim(0, 999)

# shared x-axis formatting
for a in axes:
    a.xaxis.set_major_locator(ticker.MultipleLocator(100))
    a.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    a.grid(which="major", linestyle="--", alpha=0.4)
    a.grid(which="minor", linestyle=":",  alpha=0.2)

fig.tight_layout(h_pad=3)

for ext in ("png", "pdf"):
    path = f"{OUT_DIR}/finetune_1000_loss_curve.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

# ── Print summary stats ────────────────────────────────────────────────────
print(f"\nSummary statistics — loss_train_epoch")
print(f"  Epoch 0   : {loss[0]:.4f}")
print(f"  Epoch 999 : {loss[-1]:.4f}")
print(f"  Min       : {loss.min():.4f}  (epoch {ep[np.argmin(loss)]})")
print(f"  Max       : {loss.max():.4f}")
print(f"  Mean      : {loss.mean():.4f}")
print(f"  Std       : {loss.std():.4f}")
print(f"\nFirst-100-epoch avg  : {loss[:100].mean():.4f}")
print(f"Last-100-epoch avg   : {loss[-100:].mean():.4f}")
print(f"Improvement (mean)   : {loss[:100].mean() - loss[-100:].mean():.4f}")
