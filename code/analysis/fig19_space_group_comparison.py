import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import ANALYSIS_WORK, FIGURES, METRICS  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json
import os
from collections import Counter
from pymatgen.core import Structure

# ============================================================
# EDITABLE: Colours
# ============================================================
C_TRAIN = "#0072B2"
C_GEN   = "#E69F00"

# Crystal system SG ranges (standard assignment)
SG_TO_SYSTEM = {
    "Triclinic":    range(1,   3),
    "Monoclinic":   range(3,  16),
    "Orthorhombic": range(16, 75),
    "Tetragonal":   range(75, 143),
    "Trigonal":     range(143, 168),
    "Hexagonal":    range(168, 195),
    "Cubic":        range(195, 231),
}

def sg_to_system(sg):
    for system, r in SG_TO_SYSTEM.items():
        if sg in r:
            return system
    return "Unknown"

# ── Load training data ────────────────────────────────────────
meta = pd.read_csv(f"{ANALYSIS_WORK}/Processing_data/full_metadata.csv",
                   usecols=["spacegroup_number", "crystal_system", "num_atoms"])
sg_train  = pd.to_numeric(meta["spacegroup_number"], errors="coerce").dropna().astype(int)
sys_train = meta["crystal_system"].value_counts()
n_atoms_train = meta["num_atoms"].dropna().astype(int)

# ── Load generated structure space groups + atom counts ───────
with open(f"{METRICS}/base_gen_sg.pkl", "rb") as f:
    sg_gen_counter = pickle.load(f)

sg_gen_series = pd.Series(
    sum([[k]*v for k,v in sg_gen_counter.items()], []), dtype=int
)
sys_gen = pd.Series([sg_to_system(sg) for sg in sg_gen_series]).value_counts()

det = json.load(open(f"{METRICS}/01_base_unconditional/detailed_metrics.json"))
n_atoms_gen = pd.Series(
    [len(Structure.from_dict(e["structure"])) for e in det["entry"]]
)

n_train = len(sg_train)
n_gen   = len(sg_gen_series)

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
fig.subplots_adjust(wspace=0.38)
fig.suptitle("Space Group Distribution: Base Model Training Data vs Unconditional Generation\n"
             f"(Training n={n_train:,}  |  Generated n={n_gen:,})",
             fontsize=12, y=1.02)

# ── Panel A: Crystal system ───────────────────────────────────
ax = axes[0]
systems = ["Triclinic", "Monoclinic", "Orthorhombic",
           "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
train_frac = np.array([sys_train.get(s, 0) / n_train for s in systems])
gen_frac   = np.array([sys_gen.get(s,   0) / n_gen   for s in systems])

xi = np.arange(len(systems))
ax.bar(xi - 0.2, train_frac, width=0.38, color=C_TRAIN, alpha=0.85,
       edgecolor="white", label="Training data")
ax.bar(xi + 0.2, gen_frac,   width=0.38, color=C_GEN,   alpha=0.85,
       edgecolor="white", label="Generated")

ax.set_xticks(xi)
ax.set_xticklabels(systems, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Relative frequency", fontsize=10)
ax.set_title("Crystal System Distribution", fontsize=11, pad=7)
ax.legend(fontsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.text(0.03, 0.97, "A", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# ── Panel B: Atom count distribution ─────────────────────────
ax2 = axes[1]
MAX_ATOMS = 51
bins = np.arange(0.5, MAX_ATOMS + 1.5, 2)

train_hist, edges = np.histogram(n_atoms_train, bins=bins)
gen_hist,   _     = np.histogram(n_atoms_gen,   bins=bins)

train_frac2 = train_hist / train_hist.sum()
gen_frac2   = gen_hist   / gen_hist.sum()

mids = (edges[:-1] + edges[1:]) / 2
w = 0.8  # single bar per bin, offset by half-width
ax2.bar(mids - 0.45, train_frac2, width=0.88, color=C_TRAIN, alpha=0.85,
        edgecolor="white", linewidth=0.4, label="Training data")
ax2.bar(mids + 0.45, gen_frac2,   width=0.88, color=C_GEN,   alpha=0.85,
        edgecolor="white", linewidth=0.4, label="Generated")

# Mark the 20-atom MatterGen ceiling
ax2.axvline(20, color="#D55E00", lw=1.4, ls="--", alpha=0.8,
            label="MatterGen ceiling (20 atoms)")

ax2.set_xlabel("$n_{atoms}$ per primitive cell", fontsize=10)
ax2.set_ylabel("Relative frequency", fontsize=10)
ax2.set_title("Atom Count Distribution", fontsize=11, pad=7)
ax2.set_xlim(0, MAX_ATOMS)
ax2.legend(fontsize=8.5)
ax2.tick_params(labelsize=9)
ax2.text(0.03, 0.97, "B", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")

plt.tight_layout(pad=1.5)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{FIGURES}/fig19_space_group_comparison.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved fig19 — training ({n_train:,} structures) vs generated ({n_gen:,} structures)")
plt.close()
