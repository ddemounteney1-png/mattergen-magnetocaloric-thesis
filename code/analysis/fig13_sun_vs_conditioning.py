import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# EDITABLE: Conditioning levels (sigma labels, shared x-axis)
# ============================================================
COND_LEVELS = ["Median", "+1σ", "+2σ", "+3σ", "+4σ"]

# ============================================================
# EDITABLE: Actual metrics from mattergen-evaluate
# None = run not performed at that level (bar omitted)
# ============================================================
# S.U.N. = frac_novel_unique_stable_structures
SUN = {
    "ΔS-only":        [0.0913, 0.1731, 0.1490, 0.1587, 0.1346],
    "Joint ΔS+$T_C$": [0.0268, 0.0714, 0.0982, None,   None  ],
    "Superconductor":  [0.1394, 0.1779, 0.1538, 0.1731, 0.2115],
}
# frac_stable_structures (within 0.1 eV/atom of hull)
STABLE = {
    "ΔS-only":        [0.2452, 0.3798, 0.3654, 0.4471, 0.3510],
    "Joint ΔS+$T_C$": [0.1875, 0.1786, 0.1786, None,   None  ],
    "Superconductor":  [0.3750, 0.3510, 0.4904, 0.5817, 0.5962],
}
# avg_energy_above_hull_per_atom (eV/atom)
E_HULL = {
    "ΔS-only":        [0.4092, 0.3414, 0.3578, 0.3038, 0.3455],
    "Joint ΔS+$T_C$": [0.3595, 0.4637, 0.4312, None,   None  ],
    "Superconductor":  [0.1778, 0.1979, 0.1762, 0.1269, 0.1174],
}

# ============================================================
# EDITABLE: Colours
# ============================================================
COLOURS = {
    "ΔS-only":        "#0072B2",
    "Joint ΔS+$T_C$": "#009E73",
    "Superconductor":  "#CC79A7",
}

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
fig.subplots_adjust(wspace=0.35)

models = list(SUN.keys())
x = np.arange(len(COND_LEVELS))
width = 0.25
offsets = np.linspace(-(len(models)-1)/2 * width, (len(models)-1)/2 * width, len(models))

# Panel A — S.U.N.
ax = axes[0]
for model, offset in zip(models, offsets):
    vals = [v if v is not None else 0 for v in SUN[model]]
    mask = [v is not None for v in SUN[model]]
    bars = ax.bar(x[mask] + offset, [v for v, m in zip(vals, mask) if m],
                  width=width, color=COLOURS[model], alpha=0.82,
                  edgecolor="white", linewidth=0.7, label=model)
ax.set_xticks(x)
ax.set_xticklabels(COND_LEVELS, fontsize=9)
ax.set_xlabel("Conditioning level (σ above training median)", fontsize=9)
ax.set_ylabel("S.U.N. fraction", fontsize=10)
ax.set_title("Stable · Unique · Novel (S.U.N.)", fontsize=11, pad=8)
ax.set_ylim(0, 0.28)
ax.legend(fontsize=8, loc="upper left")
ax.text(0.04, 0.97, "A", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# Panel B — Stability fraction + energy above hull (twin axis)
ax2 = axes[1]
ax2b = ax2.twinx()
for model, offset in zip(models, offsets):
    s_vals = [v if v is not None else 0 for v in STABLE[model]]
    e_vals = [v if v is not None else np.nan for v in E_HULL[model]]
    mask = [v is not None for v in STABLE[model]]
    ax2.bar(x[mask] + offset, [v for v, m in zip(s_vals, mask) if m],
            width=width, color=COLOURS[model], alpha=0.82,
            edgecolor="white", linewidth=0.7)
    ax2b.plot(x + offset, e_vals, marker="o", ms=4, lw=1.2,
              color=COLOURS[model], linestyle="--", alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(COND_LEVELS, fontsize=9)
ax2.set_xlabel("Conditioning level (σ above training median)", fontsize=9)
ax2.set_ylabel("Fraction stable (<0.1 eV/atom above hull)", fontsize=9)
ax2b.set_ylabel("Avg. $E_{hull}$ (eV/atom)", fontsize=9, color="#555555")
ax2b.tick_params(axis="y", labelcolor="#555555", labelsize=8)
ax2.set_title("Structural Stability", fontsize=11, pad=8)
ax2.set_ylim(0, 0.75)
ax2b.set_ylim(0, 0.6)
ax2.text(0.04, 0.97, "B", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")

fig.suptitle("Generation Quality vs Conditioning Level", fontsize=12, y=1.01)
plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig13_sun_vs_conditioning.{ext}", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig13_sun_vs_conditioning.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print("Saved fig13 — actual mattergen-evaluate data, 3 models × up to 5 conditioning levels")
plt.close()
