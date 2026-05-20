import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# EDITABLE: Source contributions (unique, after deduplication)
# Priority order: MP > COD > Alexandria > OQMD
# Exact overlap counts not available — showing unique contributions only
# TODO: replace with exact counts from deduplication logs if available
# ============================================================
SOURCES = {
    "Materials Project": {"unique": 154_000, "colour": "#0072B2"},
    "Alexandria":        {"unique":  83_500, "colour": "#009E73"},
    "COD (filtered)":    {"unique":  33_800, "colour": "#E69F00"},
    "OQMD":              {"unique":   2_900, "colour": "#CC79A7"},
}
TOTAL = sum(v["unique"] for v in SOURCES.values())

# ============================================================
# EDITABLE: Titles
# ============================================================
TITLE  = "Base Dataset Source Contributions After Deduplication"
XLABEL = "Unique structures retained"
NOTE   = "⚠ Deduplication priority: MP > COD > Alexandria > OQMD\n" \
         "Structures present in multiple sources assigned to higher-priority source.\n" \
         "Exact overlap counts unavailable — showing unique contributions only."

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7.0, 3.5))

names   = list(SOURCES.keys())
uniques = [SOURCES[n]["unique"] for n in names]
colours = [SOURCES[n]["colour"] for n in names]

# Sort largest first (top)
order   = np.argsort(uniques)[::-1]
names_s = [names[i] for i in order]
vals_s  = [uniques[i] for i in order]
cols_s  = [colours[i] for i in order]

bars = ax.barh(names_s, vals_s, color=cols_s, edgecolor="white", linewidth=0.8, height=0.6)
for bar, val in zip(bars, vals_s):
    pct = val / TOTAL * 100
    ax.text(bar.get_width() + TOTAL * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val/1000:.0f}k  ({pct:.0f}%)",
            va="center", ha="left", fontsize=10)

ax.set_title(TITLE, fontsize=12, pad=10)
ax.set_xlabel(XLABEL, fontsize=11)
ax.set_xlim(0, TOTAL * 1.25)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.tick_params(labelsize=10)
ax.text(0.02, -0.22, NOTE, transform=ax.transAxes, fontsize=7.5,
        color="#777777", va="top", style="italic")

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig12_source_overlap.{ext}", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig12_source_overlap.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig12 — total {TOTAL:,} structures, {len(SOURCES)} sources")
plt.close()
