import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES, FINETUNE_DATA, METRICS  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle, re, os

# ============================================================
# EDITABLE: Number of top elements to show per panel
# ============================================================
TOP_N = 15

# ============================================================
# EDITABLE: Colours
# ============================================================
C_TRAIN = "#0072B2"
C_GEN   = "#E69F00"

# ── Training data element frequencies ────────────────────────
train = pd.read_csv(f"{FINETUNE_DATA}/train_repaired.csv")
val   = pd.read_csv(f"{FINETUNE_DATA}/val_repaired.csv").dropna(subset=["n_atoms_primitive"])
both  = pd.concat([train, val])

train_elem: dict = {}
for mid in both["material_id"].dropna():
    for e in re.findall(r"[A-Z][a-z]?", mid):
        train_elem[e] = train_elem.get(e, 0) + 1

# ── Generated structure element frequencies (from parsed pkl) ─
with open(f"{METRICS}/parsed_elem_sg.pkl", "rb") as f:
    parsed = pickle.load(f)
elem_counts = parsed["elem_counts"]

# ── 3-panel figure, one per model ────────────────────────────
MODEL_LABELS = {
    "Delta_S_only":      "ΔS-only fine-tune",
    "Joint_Tc_delta_S":  "Joint ΔS + $T_C$ fine-tune",
    "Superconductor":    "Superconductor fine-tune",
}

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
fig.subplots_adjust(wspace=0.45)

for ax, (model_key, model_label) in zip(axes, MODEL_LABELS.items()):
    gen_counter = elem_counts[model_key]
    gen_total   = sum(gen_counter.values())
    train_total = sum(train_elem.values())

    # Top N elements ranked by generated frequency
    top_elems = [e for e, _ in gen_counter.most_common(TOP_N)]

    gen_frac   = np.array([gen_counter.get(e, 0) / gen_total   for e in top_elems])
    train_frac = np.array([train_elem.get(e, 0)  / train_total for e in top_elems])

    xi = np.arange(TOP_N)
    ax.bar(xi - 0.2, train_frac, width=0.38, color=C_TRAIN, alpha=0.82,
           edgecolor="white", label="Training data")
    ax.bar(xi + 0.2, gen_frac,   width=0.38, color=C_GEN,   alpha=0.82,
           edgecolor="white", label="Generated")

    ax.set_xticks(xi)
    ax.set_xticklabels(top_elems, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Element", fontsize=9)
    ax.set_ylabel("Relative frequency", fontsize=9)
    ax.set_title(model_label, fontsize=10, pad=6)
    ax.tick_params(axis="y", labelsize=8)
    if ax is axes[0]:
        ax.legend(fontsize=8)

fig.suptitle("Element Frequency: Training Data vs Generated Structures", fontsize=12, y=1.02)
plt.tight_layout(pad=1.5)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIGURES}/fig14_element_frequency.{ext}", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIGURES}/fig14_element_frequency.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig14 — top {TOP_N} elements per model, training vs generated")
plt.close()
