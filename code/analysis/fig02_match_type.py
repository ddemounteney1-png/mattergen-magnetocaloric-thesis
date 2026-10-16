import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# EDITABLE: Titles and labels
# ============================================================
TITLE_A = "Structure Match Type\n(Materials Project API)"
TITLE_B = "Ordered vs Disordered Structures"
SUPTITLE = "Fine-Tuning Dataset: Structure Acquisition Summary"

# ============================================================
# EDITABLE: Data values
# From relaxation_results.csv (389 pipeline entries):
#   generation_method: prototype_wyckoff=388, direct_mp_cif=1
#   + 21 manual (ICSD + literature), giving 409 total compositions
# 3-tier breakdown from project methodology doc:
#   - Exact match (formula + SG): estimated minority
#   - Parent match (parent formula + SG): the majority
#   - Parent-only (parent formula, SG differs): smallest tier
# Exact per-tier counts not available in CSV — update when known.
# Ordered/disordered from cif_quality_report.csv: 128/277
# ============================================================
MATCH_DATA = {
    "Prototype\n(Wyckoff)": 388,   # prototype_wyckoff in generation_method
    "Direct MP\nCIF":          1,  # direct_mp_cif in generation_method
    "Manual\n(ICSD/lit.)":    21,  # not in MP, manual sourcing
}
ORDER_DATA = {
    "Ordered": 128,   # is_disordered=False from cif_quality_report.csv
    "SQS\n(disordered)": 277,  # is_disordered=True
}

# ============================================================
# EDITABLE: Colours
# ============================================================
C_MATCH  = ["#0072B2", "#56B4E9", "#009E73"]
C_ORDER  = ["#0072B2", "#E69F00"]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
fig.suptitle(SUPTITLE, fontsize=12, y=1.01)
fig.subplots_adjust(wspace=0.4)

# ── Panel A: Match type ────────────────────────────────────
ax = axes[0]
labels_a = list(MATCH_DATA.keys())
counts_a = list(MATCH_DATA.values())
bars = ax.bar(labels_a, counts_a, color=C_MATCH, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, counts_a):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title(TITLE_A, fontsize=11, pad=8)
ax.set_ylabel("Count", fontsize=10)
ax.set_ylim(0, max(counts_a) * 1.20)
ax.tick_params(labelsize=9)
# Panel label
ax.text(0.04, 0.96, "A", transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top")

# ── Panel B: Ordered vs SQS ───────────────────────────────
ax2 = axes[1]
labels_b = list(ORDER_DATA.keys())
counts_b = list(ORDER_DATA.values())
bars2 = ax2.bar(labels_b, counts_b, color=C_ORDER, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars2, counts_b):
    pct = val / sum(counts_b) * 100
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f"{val}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_title(TITLE_B, fontsize=11, pad=8)
ax2.set_ylabel("Count", fontsize=10)
ax2.set_ylim(0, max(counts_b) * 1.25)
ax2.tick_params(labelsize=9)
ax2.text(0.04, 0.96, "B", transform=ax2.transAxes,
         fontsize=13, fontweight="bold", va="top")

plt.tight_layout(pad=1.5)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIGURES}/fig02_match_type_breakdown.{ext}", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIGURES}/fig02_match_type_breakdown.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig02 — match types: {MATCH_DATA}, ordering: {ORDER_DATA}")
plt.close()
