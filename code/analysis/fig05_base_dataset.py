import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# EDITABLE: Base dataset source counts
# Source: project reference document
# ============================================================
SOURCE_COUNTS = {
    "Materials Project": 154_000,
    "COD (filtered)":     33_800,
    "Alexandria":         83_500,
    "OQMD":                2_900,   # remainder to ~274,200 total
}
TOTAL = sum(SOURCE_COUNTS.values())

# ============================================================
# EDITABLE: Phase system distribution — exact counts (N=274,602)
# ============================================================
PHASE_LABELS  = ["Unary", "Binary", "Ternary", "Quaternary", "Quinary", "6+"]
PHASE_COUNTS  = [1_192, 30_167, 138_784, 87_711, 14_689, 2_059]  # 6+ = senary+7+8

# ============================================================
# EDITABLE: Atom count distribution — exact binned counts (N=274,602)
# Bins are unequal width; bar widths are scaled accordingly.
# bin_mid: display position, bin_width: actual atom range, count: structures
# ============================================================
ATOM_BIN_DATA = [
    # (label,   mid,   width, count)
    ("1",         1,     1,     293),
    ("2",         2,     1,   3_222),
    ("3–4",       3.5,   2,  38_303),
    ("5–8",       6.5,   4,  51_104),
    ("9–12",     10.5,   4,  58_791),
    ("13–16",    14.5,   4,  28_213),
    ("17–20",    18.5,   4,  30_338),
    ("21–30",    25.5,  10,  31_771),
    ("31–40",    35.5,  10,  21_612),
    ("41–50",    45.5,  10,  10_955),
]
ATOM_MEDIAN = 12
ATOM_MEAN   = 15.4

# ============================================================
# EDITABLE: Colours
# ============================================================
C_SOURCES = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
C_PHASE   = "#56B4E9"
C_ATOM    = "#0072B2"

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.5))
fig.subplots_adjust(wspace=0.45)
fig.suptitle(f"Base Pre-Training Dataset Composition (N = {TOTAL:,} structures)", fontsize=12, y=1.02)

# ── Panel A: Source breakdown ──────────────────────────────
ax = axes[0]
names  = list(SOURCE_COUNTS.keys())
counts = list(SOURCE_COUNTS.values())
bars = ax.bar(names, counts, color=C_SOURCES, edgecolor="white", linewidth=0.7)
for bar, val in zip(bars, counts):
    pct = val / TOTAL * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800,
            f"{val/1000:.0f}k\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8)
ax.set_title("Contributions by Source", fontsize=10, pad=6)
ax.set_ylabel("Structures", fontsize=10)
ax.tick_params(axis="x", labelsize=8, rotation=15)
ax.tick_params(axis="y", labelsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.set_ylim(0, max(counts) * 1.30)
ax.text(0.04, 0.96, "A", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# ── Panel B: Atom count (exact binned data, bars scaled by bin width) ─────
ax2 = axes[1]
for label, mid, bw, count in ATOM_BIN_DATA:
    ax2.bar(mid, count, width=bw * 0.88, color=C_ATOM, alpha=0.82, edgecolor="white", linewidth=0.5)
ax2.axvline(ATOM_MEDIAN, color="#D55E00", lw=1.4, ls="--",
            label=f"Median ({ATOM_MEDIAN})")
ax2.axvline(ATOM_MEAN,   color="#CC79A7", lw=1.4, ls=":",
            label=f"Mean ({ATOM_MEAN})")
ax2.set_title("Atom Count Distribution", fontsize=10, pad=6)
ax2.set_xlabel("$n_{atoms}$ per primitive cell", fontsize=10)
ax2.set_ylabel("Structures", fontsize=10)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax2.tick_params(labelsize=8)
ax2.legend(fontsize=7.5, loc="upper right")
ax2.text(0.04, 0.96, "B", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")

# ── Panel C: Phase system (exact counts) ──────────────────
ax3 = axes[2]
bars3 = ax3.bar(PHASE_LABELS, PHASE_COUNTS, color=C_PHASE, edgecolor="white", linewidth=0.7)
for bar, val in zip(bars3, PHASE_COUNTS):
    pct = val / TOTAL * 100
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800,
             f"{val/1000:.0f}k\n({pct:.0f}%)", ha="center", va="bottom", fontsize=7.5)
ax3.set_title("Phase System Distribution", fontsize=10, pad=6)
ax3.set_ylabel("Structures", fontsize=10)
ax3.tick_params(axis="x", labelsize=8)
ax3.tick_params(axis="y", labelsize=9)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax3.set_ylim(0, max(PHASE_COUNTS) * 1.30)
ax3.text(0.04, 0.96, "C", transform=ax3.transAxes, fontsize=13, fontweight="bold", va="top")

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig05_base_dataset_composition.{ext}", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig05_base_dataset_composition.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig05 — total ~{TOTAL:,} structures across {len(SOURCE_COUNTS)} sources")
plt.close()
