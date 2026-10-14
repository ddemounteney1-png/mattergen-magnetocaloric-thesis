import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import FIGURES  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ============================================================
# EDITABLE: Model display names, SUN values, colours, grouping
# Mainstream models evaluated with MP2020correction convex hull.
# Fine-tuned models evaluated with TRI2024correction — not
# directly comparable; flagged with hatching and footnote.
# ============================================================

# --- Mainstream baselines (MP2020correction) -----------------
MAINSTREAM = [
    ("FTCP",                 0.000),
    ("G-SchNet",             0.0753),
    ("P-G-SchNet",           0.1104),
    ("CDVAE",                0.1833),
    ("DiffCSP (MP-20)",      0.2013),
    ("MatterGen (MP-20)",    0.2576),
    ("MatterGen (full)",     0.3886),
]

# --- My base model (TRI2024correction, unconditional, n=1008) ----
MY_MODELS = [
    ("My base model\n(unconditional)", 0.2312),
]

# ============================================================
# EDITABLE: Colours
# ============================================================
C_MAINSTREAM = "#0072B2"
C_MY         = "#E69F00"
C_MATTERGEN  = "#009E73"   # highlight MatterGen separately

# ── Build combined sorted list ────────────────────────────────
all_models = (
    [(name, val, "mainstream") for name, val in MAINSTREAM] +
    [(name, val, "mine")       for name, val in MY_MODELS]
)
all_models.sort(key=lambda x: x[1])

labels  = [m[0] for m in all_models]
values  = [m[1] for m in all_models]
groups  = [m[2] for m in all_models]

colours = []
hatches = []
for label, val, grp in all_models:
    if grp == "mine":
        colours.append(C_MY)
        hatches.append("//")
    elif "MatterGen (full)" in label:
        colours.append(C_MATTERGEN)
        hatches.append("")
    else:
        colours.append(C_MAINSTREAM)
        hatches.append("")

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8.0, 5.5))

y = np.arange(len(labels))
bars = ax.barh(y, values, height=0.62, color=colours,
               edgecolor="white", linewidth=0.7)

# Apply hatching to fine-tuned bars
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
    bar.set_edgecolor("#888888" if hatch else "white")
    bar.set_linewidth(0.5)

# Value labels
for bar, val in zip(bars, values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", ha="left", fontsize=8.5)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("S.U.N. fraction (Stable · Unique · Novel)", fontsize=11)
ax.set_title("S.U.N. Comparison: Fine-tuned Models vs Mainstream Baselines",
             fontsize=12, pad=10)
ax.set_xlim(0, 0.48)
ax.tick_params(axis="x", labelsize=9)

# Legend
patch_main = mpatches.Patch(color=C_MAINSTREAM, label="Mainstream baselines (MP2020 hull)")
patch_mg   = mpatches.Patch(color=C_MATTERGEN,  label="MatterGen full (MP2020 hull)")
patch_mine = mpatches.Patch(facecolor=C_MY, hatch="//", edgecolor="#888888",
                             label="This work — base model (TRI2024 hull) ‡")
ax.legend(handles=[patch_main, patch_mg, patch_mine], fontsize=8.5,
          loc="lower right")

# Framework caveat footnote
ax.text(0.01, -0.10,
        "‡ Base model uses TRI2024correction for convex hull stability; "
        "mainstream models use MP2020correction.\n"
        "  Direct numerical comparison of SUN/stability fractions should be treated with caution.",
        transform=ax.transAxes, fontsize=7.5, color="#666666", va="top")

plt.tight_layout(pad=1.5)
fig.subplots_adjust(bottom=0.14)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{FIGURES}/fig17_sun_comparison.{ext}", dpi=300, bbox_inches="tight")
print("Saved fig17 — SUN comparison, all models")
plt.close()
