import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# EDITABLE: COD filtering stage data
# Source: project reference document / COD processing logs
# ============================================================
STAGES = [
    ("Starting CIF files",          533_464, "start"),
    ("Removed: partial occupancy", -166_447, "remove"),
    ("Removed: >51 atoms",         -317_673, "remove"),
    ("Removed: organic",             -8_318, "remove"),
    ("Removed: theoretical",           -165, "remove"),
    ("Removed: internal duplicates",  -5_435, "remove"),
    ("Removed: parse errors",         -1_635, "remove"),
    ("Final: retained structures",   33_791, "final"),
]

# ============================================================
# EDITABLE: Titles and labels
# ============================================================
TITLE  = "COD Database Filtering Pipeline"
XLABEL = "Number of structures"

# ============================================================
# EDITABLE: Colours
# ============================================================
C_START  = "#3498db"
C_REMOVE = "#e74c3c"
C_FINAL  = "#2ecc71"

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7.0, 4.5))

TOTAL_START = 533_464
labels   = [s[0] for s in STAGES]
values   = [s[1] for s in STAGES]
kinds    = [s[2] for s in STAGES]

# Build running totals for waterfall
running = 0
bar_starts = []
bar_widths = []
for i, (val, kind) in enumerate(zip(values, kinds)):
    if kind == "start":
        bar_starts.append(0)
        bar_widths.append(val)
        running = val
    elif kind == "remove":
        running += val          # val is negative
        bar_starts.append(running)
        bar_widths.append(-val)  # positive width
    else:  # final
        bar_starts.append(0)
        bar_widths.append(val)

y_pos = np.arange(len(STAGES))
colours = [C_START if k == "start" else C_FINAL if k == "final" else C_REMOVE
           for k in kinds]

bars = ax.barh(y_pos, bar_widths, left=bar_starts, color=colours,
               edgecolor="white", linewidth=0.8, height=0.65)

# Labels on bars
for i, (bar, val, kind) in enumerate(zip(bars, values, kinds)):
    pct = abs(val) / TOTAL_START * 100
    count_str = f"{abs(val):,}"
    pct_str   = f" ({pct:.1f}%)" if kind != "start" else ""
    label_x   = bar.get_x() + bar.get_width() + TOTAL_START * 0.012
    ax.text(label_x, bar.get_y() + bar.get_height() / 2,
            count_str + pct_str, va="center", ha="left", fontsize=8.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel(XLABEL, fontsize=11)
ax.set_title(TITLE, fontsize=12, pad=10)
ax.set_xlim(0, TOTAL_START * 1.30)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.tick_params(axis="x", labelsize=9)

# Colour legend
from matplotlib.patches import Patch
legend_items = [Patch(color=C_START,  label="Starting total"),
                Patch(color=C_REMOVE, label="Removed"),
                Patch(color=C_FINAL,  label="Retained")]
ax.legend(handles=legend_items, fontsize=9, loc="lower right")

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig07_cod_filtering_waterfall.{ext}", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig07_cod_filtering_waterfall.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig07 — {TOTAL_START:,} → 33,791 structures "
      f"({33791/TOTAL_START*100:.1f}% retained)")
plt.close()
