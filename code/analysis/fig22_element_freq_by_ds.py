import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json
import os
from collections import Counter

# ============================================================
# EDITABLE: ΔS-only run paths and labels
# ============================================================
RUNS = [
    ("5.3  (Median)", "Metrics/Delta_S_only/median_5.3_deltaS_only_detailed_metrics.json"),
    ("18.9 (+1σ)",    "Metrics/Delta_S_only/deltaS_only_plus1sigma_18.9_detailed_metrics.json"),
    ("32.5 (+2σ)",    "Metrics/Delta_S_only/plus2sigma_32.5_deltaS_only_detailed_metrics.json"),
    ("46.1 (+3σ)",    "Metrics/Delta_S_only/deltaS_only_plus3sigma_46.1_detailed_metrics.json"),
    ("59.7 (+4σ)",    "Metrics/Delta_S_only/Delta_S_only_plus4sigma_59.7_detailed_metrics.json"),
]

# ============================================================
# EDITABLE: Number of top elements to show
# ============================================================
TOP_N = 18

# ============================================================
# EDITABLE: Highlight known MCE-relevant elements
# ============================================================
MCE_ELEMENTS = {"Gd", "La", "Nd", "Dy", "Ho", "Er", "Fe", "Mn", "Co", "In"}

# ── Extract element frequencies per run ──────────────────────
run_counters = {}
for label, path in RUNS:
    d = json.load(open(path))
    n = len(d["entry"])
    counts = Counter()
    for e in d["entry"]:
        for el in e["composition"]:
            counts[el] += 1
    # Normalise to fraction of structures containing element
    run_counters[label] = {el: c / n for el, c in counts.items()}

# ── Select top N elements by total occurrence across all runs ─
total = Counter()
for label, path in RUNS:
    d = json.load(open(path))
    n = len(d["entry"])
    for e in d["entry"]:
        for el in e["composition"]:
            total[el] += 1
top_elements = [el for el, _ in total.most_common(TOP_N)]

# ── Colour ramp: blue→red across conditioning levels ─────────
colours = ["#2166AC", "#74ADD1", "#8E8E02", "#F46D43", "#A50026"]

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12.0, 5.0))

n_runs = len(RUNS)
width  = 0.14
x      = np.arange(len(top_elements))
offsets = np.linspace(-(n_runs - 1) / 2 * width,
                       (n_runs - 1) / 2 * width, n_runs)

for (label, _), colour, offset in zip(RUNS, colours, offsets):
    freqs = [run_counters[label].get(el, 0) * 100 for el in top_elements]
    ax.bar(x + offset, freqs, width=width, color=colour,
           edgecolor="white", linewidth=0.5, label=f"ΔS = {label}")

# Highlight MCE-relevant elements on x-axis
tick_labels = []
for el in top_elements:
    tick_labels.append(f"$\\bf{{{el}}}$*" if el in MCE_ELEMENTS else el)

ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=10)
ax.set_ylabel("% of generated structures containing element", fontsize=10)
ax.set_xlabel("Element  (bold* = known MCE-relevant)", fontsize=10)
ax.set_title(
    "Element Frequency vs $\\Delta S_{mag}$ Conditioning Level\n"
    "(ΔS-only fine-tuned model — darker bars = higher conditioning)",
    fontsize=12, pad=8)
ax.tick_params(axis="y", labelsize=9)
ax.legend(fontsize=9, loc="upper right", title="ΔS conditioning (J kg⁻¹ K⁻¹)",
          title_fontsize=8.5)

# Shade every other element column for readability
for i in range(0, len(top_elements), 2):
    ax.axvspan(i - 0.5, i + 0.5, color="#f0f0f0", alpha=0.5, zorder=0)

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"figures/fig22_element_freq_by_ds.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved fig22 — top {TOP_N} elements across {len(RUNS)} ΔS conditioning levels")
plt.close()
