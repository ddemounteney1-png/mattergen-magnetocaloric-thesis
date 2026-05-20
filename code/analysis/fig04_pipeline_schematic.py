import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ============================================================
# EDITABLE: Pipeline stages — (label, type, colour_key)
# Types: 'data' (parallelogram), 'process' (rect), 'check' (rect)
# ============================================================
STAGES = [
    ("Franco et al. (2012)\n641 measurements · 409 compositions", "data",    "input"),
    ("PDF parsing (pdfplumber)\n→ 537 measurement rows",           "process", "process"),
    ("Remove amorphous materials\n− 103 compositions",             "check",   "remove"),
    ("Prototype → space group\nlookup (AFLOW library)",            "process", "process"),
    ("Parent formula extraction\n(strip dopants < 10%)",           "process", "process"),
    ("Materials Project API\n3-tier matching (exact / parent / parent-only)\n"
     "+ 21 manual (14 ICSD Gd₅SiGe₄ · 3 ICSD YFe₂H₄ · 4 literature)",
                                                                   "process", "process"),
    ("Wyckoff site assignment\n& fractional occupancy",            "process", "process"),
    ("SQS ordering\n(274 disordered structures)",                   "process", "process"),
    ("CHGNet relaxation\n(313 converged · 1 partial · 75 skipped no CIF)",
                                                                   "process", "process"),
    ("Property repair\n(composition-based matching)",              "check",   "check"),
    ("Deduplication\n(remove ~60 unresolvable duplicates)",        "check",   "remove"),
    ("Final quality checks\n(≤51 atoms · no partial occ. · <5% stoich. error)",
                                                                   "check",   "check"),
    ("315 structures\n284 train  /  31 val",                       "data",    "output"),
]

# ============================================================
# EDITABLE: Colours
# ============================================================
COLOURS = {
    "input":   "#2ecc71",   # green
    "process": "#3498db",   # blue
    "check":   "#f39c12",   # orange
    "remove":  "#e74c3c",   # red
    "output":  "#27ae60",   # dark green
}
TEXT_COLOUR = "white"

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7.0, 13.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

n = len(STAGES)
box_h   = 0.052
gap     = 0.018
total_h = n * box_h + (n - 1) * gap
y_start = (1 - total_h) / 2 + total_h

box_x, box_w = 0.08, 0.84

for i, (label, btype, ckey) in enumerate(STAGES):
    y_top = y_start - i * (box_h + gap)
    y_cen = y_top - box_h / 2
    colour = COLOURS[ckey]

    rect = FancyBboxPatch((box_x, y_top - box_h), box_w, box_h,
                          boxstyle="round,pad=0.008", linewidth=0,
                          facecolor=colour, zorder=2)
    ax.add_patch(rect)
    ax.text(box_x + box_w / 2, y_cen, label,
            ha="center", va="center", fontsize=8.2,
            color=TEXT_COLOUR, fontweight="bold", zorder=3,
            multialignment="center")

    # Arrow to next box
    if i < n - 1:
        ax.annotate("", xy=(0.5, y_top - box_h - gap),
                    xytext=(0.5, y_top - box_h),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3))

# Legend
legend_items = [
    mpatches.Patch(color=COLOURS["input"],   label="Source data"),
    mpatches.Patch(color=COLOURS["process"], label="Processing step"),
    mpatches.Patch(color=COLOURS["check"],   label="Quality check"),
    mpatches.Patch(color=COLOURS["remove"],  label="Removal / filtering"),
    mpatches.Patch(color=COLOURS["output"],  label="Output dataset"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8.5,
          framealpha=0.9, bbox_to_anchor=(0.99, 0.01))

ax.set_title("Magnetocaloric Fine-Tuning Dataset Construction Pipeline",
             fontsize=12, pad=12, fontweight="bold")

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig04_pipeline_schematic.{ext}", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig04_pipeline_schematic.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig04 — {len(STAGES)} pipeline stages")
plt.close()
