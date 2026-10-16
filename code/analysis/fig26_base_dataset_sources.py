import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import ANALYSIS_WORK, FIGURES  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

# ============================================================
# EDITABLE: Source grouping — set EXPERIMENTAL_SOURCES to whichever
# sources should receive bright colours. All others get pastel.
# Actual source values in the data: MP, MP_Full, COD, Alexandria, OQMD
# ============================================================
EXPERIMENTAL_SOURCES = {"COD", "MP"}     # ← adjust if user clarifies the 53k boundary

# ============================================================
# EDITABLE: Bright colours (experimental sources)
# ============================================================
BRIGHT_COLOURS = {
    "COD": "#009E73",    # vivid green
    "MP":  "#0072B2",    # vivid blue
}

# ============================================================
# EDITABLE: Pastel colours (computational sources)
# ============================================================
PASTEL_COLOURS = {
    "MP_Full":    "#A8D0E6",   # pastel blue
    "OQMD":       "#FADADD",   # pastel rose
    "Alexandria": "#C8E6C9",   # pastel green
}

# ============================================================
# EDITABLE: Display labels for each source
# ============================================================
SOURCE_LABELS = {
    "COD":        "COD\n(experimental crystal structures)",
    "MP":         "Materials Project\n(curated, ICSD-linked)",
    "MP_Full":    "Materials Project Full\n(broader computational set)",
    "OQMD":       "OQMD\n(DFT on experimental prototypes)",
    "Alexandria": "Alexandria\n(high-throughput computational)",
}

# ============================================================
# EDITABLE: Slice order (clockwise from top — bright first)
# ============================================================
SLICE_ORDER = ["COD", "MP", "MP_Full", "OQMD", "Alexandria"]

# ── Load source counts ────────────────────────────────────────
df = pd.read_csv(f"{ANALYSIS_WORK}/Processing_data/full_metadata.csv",
                 usecols=["source"], low_memory=False)
counts = df["source"].value_counts()

sizes  = [counts.get(s, 0) for s in SLICE_ORDER]
total  = sum(sizes)
colours = [BRIGHT_COLOURS.get(s) or PASTEL_COLOURS.get(s, "#dddddd")
           for s in SLICE_ORDER]

# Explode experimental slices slightly
explode = [0.04 if s in EXPERIMENTAL_SOURCES else 0.0 for s in SLICE_ORDER]

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8.0, 6.5))

wedges, texts, autotexts = ax.pie(
    sizes,
    colors=colours,
    explode=explode,
    startangle=90,
    counterclock=False,
    autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100)):,})",
    pctdistance=0.72,
    wedgeprops=dict(edgecolor="white", linewidth=1.5),
    radius=1.0,
)

# Style percentage labels
for atext, s in zip(autotexts, SLICE_ORDER):
    atext.set_fontsize(8.5)
    atext.set_color("white" if s in EXPERIMENTAL_SOURCES else "#333333")
    atext.set_fontweight("bold" if s in EXPERIMENTAL_SOURCES else "normal")

# ── Legend ───────────────────────────────────────────────────
legend_handles = []
for s in SLICE_ORDER:
    c = BRIGHT_COLOURS.get(s) or PASTEL_COLOURS.get(s, "#dddddd")
    lbl = f"{SOURCE_LABELS[s].replace(chr(10), ' ')}  ({counts.get(s, 0):,})"
    patch = mpatches.Patch(facecolor=c, edgecolor="white", linewidth=1.2, label=lbl)
    legend_handles.append(patch)

ax.legend(handles=legend_handles, fontsize=8.5, loc="lower center",
          bbox_to_anchor=(0.5, -0.22), ncol=1, framealpha=0.95,
          title="Source  (brighter = experimentally derived)", title_fontsize=8.5)

# Experimental total annotation
exp_total = sum(counts.get(s, 0) for s in EXPERIMENTAL_SOURCES)
comp_total = total - exp_total
ax.text(0, 0, f"Total\n{total:,}", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#333333")

ax.set_title(
    "Base Pre-training Dataset: Structure Sources\n"
    f"(n = {total:,} total  |  {exp_total:,} experimentally derived  |  {comp_total:,} computational)",
    fontsize=11, pad=16)

plt.tight_layout(pad=1.5)
os.makedirs(FIGURES, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{FIGURES}/fig26_base_dataset_sources.{ext}",
                dpi=300, bbox_inches="tight")
print(f"Saved fig26 — {len(SLICE_ORDER)} sources, {total:,} total structures")
print(f"  Bright (experimental): {', '.join(EXPERIMENTAL_SOURCES)} = {exp_total:,}")
print(f"  Pastel (computational): {comp_total:,}")
plt.close()
