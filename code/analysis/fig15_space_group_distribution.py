import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES, FINETUNE_DATA, METRICS  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle, os

# ============================================================
# EDITABLE: Bin width for space group histogram
# ============================================================
BIN_WIDTH = 10   # group SGs into bands of this width

# ============================================================
# EDITABLE: Colours per series
# ============================================================
C_TRAIN = "#0072B2"
COLOURS_GEN = {
    "Delta_S_only":     "#E69F00",
    "Joint_Tc_delta_S": "#009E73",
    "Superconductor":   "#CC79A7",
}
MODEL_LABELS = {
    "Delta_S_only":     "ΔS-only",
    "Joint_Tc_delta_S": "Joint ΔS+$T_C$",
    "Superconductor":   "Superconductor",
}

# ── Training data space groups ────────────────────────────────
train = pd.read_csv(f"{FINETUNE_DATA}/train_repaired.csv")
val   = pd.read_csv(f"{FINETUNE_DATA}/val_repaired.csv").dropna(subset=["n_atoms_primitive"])
both  = pd.concat([train, val])
sg_train = pd.to_numeric(both["space_group_number"], errors="coerce").dropna().astype(int)

# ── Generated space groups (from parsed pkl) ──────────────────
with open(f"{METRICS}/parsed_elem_sg.pkl", "rb") as f:
    sg_counts = pickle.load(f)["sg_counts"]

# ── Bin edges 1–230 ───────────────────────────────────────────
bins   = np.arange(1, 231 + BIN_WIDTH, BIN_WIDTH)
labels = [f"{b}" for b in bins[:-1]]

def bin_sg(sg_series):
    return pd.cut(sg_series, bins=bins, right=False).value_counts().sort_index().values

train_binned = bin_sg(sg_train)

# Normalise to fractions
def to_frac(arr):
    s = arr.sum()
    return arr / s if s > 0 else arr

# ── 4-panel figure (training + 3 model comparisons) ──────────
fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.0), sharey=False)
fig.subplots_adjust(wspace=0.38)

xi = np.arange(len(labels))

# Panel 0: training data alone
ax = axes[0]
ax.bar(xi, to_frac(train_binned), color=C_TRAIN, alpha=0.82, edgecolor="white", width=0.8)
ax.set_title("Training data", fontsize=10, pad=6)
ax.set_ylabel("Relative frequency", fontsize=9)
_ticks = xi[::2]; ax.set_xticks(_ticks)
ax.set_xticklabels([labels[i] for i in _ticks], rotation=45, ha="right", fontsize=7)
ax.set_xlabel("Space group number", fontsize=9)
ax.text(0.05, 0.97, "A", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

# Panels 1-3: training vs each generated model
for idx, (model_key, model_label) in enumerate(MODEL_LABELS.items()):
    ax = axes[idx + 1]
    counter = sg_counts[model_key]
    gen_sg  = pd.Series(list(counter.elements()) if hasattr(counter, 'elements')
                        else sum([[k]*v for k,v in counter.items()], []))
    gen_binned = bin_sg(gen_sg)

    ax.bar(xi - 0.2, to_frac(train_binned), width=0.38, color=C_TRAIN, alpha=0.72,
           edgecolor="white", label="Training")
    ax.bar(xi + 0.2, to_frac(gen_binned),   width=0.38,
           color=COLOURS_GEN[model_key], alpha=0.82, edgecolor="white", label="Generated")
    ax.set_title(model_label, fontsize=10, pad=6)
    ax.set_xticks(_ticks)
    ax.set_xticklabels([labels[i] for i in _ticks], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Space group number", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    panel_letter = chr(66 + idx)  # B, C, D
    ax.text(0.05, 0.97, panel_letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

fig.suptitle("Space Group Distribution: Training Data vs Generated Structures",
             fontsize=12, y=1.02)
plt.tight_layout(pad=1.5)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIGURES}/fig15_space_group_distribution.{ext}", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIGURES}/fig15_space_group_distribution.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig15 — training ({len(sg_train)} structures) vs 3 generated models")
plt.close()
