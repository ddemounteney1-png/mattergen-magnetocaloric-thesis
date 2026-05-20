import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# EDITABLE: ΔS conditioning values and metrics (ΔS-only model)
# Source: mattergen-evaluate, frac_* fields from metrics.json
# ============================================================
DS_VALUES = [5.3, 18.9, 32.5, 46.1, 59.7]   # J kg⁻¹ K⁻¹

STABLE  = [0.2452, 0.3798, 0.3654, 0.4471, 0.3510]
UNIQUE  = [0.9663, 0.9519, 0.9471, 0.9135, 0.9231]
NOVEL   = [0.7596, 0.7067, 0.7260, 0.6635, 0.7404]

# ============================================================
# EDITABLE: Colours and styling
# ============================================================
C_STABLE = "#0072B2"
C_UNIQUE = "#009E73"
C_NOVEL  = "#E69F00"

# Sigma level positions for reference lines
SIGMA_LABELS = {
    "Median": 5.3, "+1σ": 18.9, "+2σ": 32.5, "+3σ": 46.1, "+4σ": 59.7
}

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7.0, 4.2))

pct = lambda x: [v * 100 for v in x]

ax.plot(DS_VALUES, pct(STABLE), marker="o", ms=7, lw=2.0,
        color=C_STABLE, label="Stable (<0.1 eV/atom above hull)")
ax.plot(DS_VALUES, pct(UNIQUE), marker="s", ms=7, lw=2.0,
        color=C_UNIQUE, label="Unique")
ax.plot(DS_VALUES, pct(NOVEL),  marker="^", ms=7, lw=2.0,
        color=C_NOVEL,  label="Novel")

# Sigma reference lines
for label, x in SIGMA_LABELS.items():
    ax.axvline(x, color="#cccccc", lw=0.8, ls="--", zorder=0)
    ax.text(x, 97, label, ha="center", va="top", fontsize=7.5, color="#888888")

ax.set_xlabel("$\\Delta S_{mag}$ conditioning value (J kg$^{-1}$ K$^{-1}$)", fontsize=11)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_title("Generation Quality vs $\\Delta S_{mag}$ Conditioning Level\n(ΔS-only fine-tuned model)",
             fontsize=11, pad=8)
ax.set_xlim(0, 65)
ax.set_ylim(0, 105)
ax.legend(fontsize=9, loc="center right")
ax.tick_params(labelsize=10)
ax.set_xticks(DS_VALUES)

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"figures/fig20_ds_conditioning_lines.{ext}", dpi=300, bbox_inches="tight")
print("Saved fig20 — stability/uniqueness/novelty vs ΔS conditioning (ΔS-only model)")
plt.close()
