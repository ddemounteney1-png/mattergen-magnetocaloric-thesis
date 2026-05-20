import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os

# ============================================================
# EDITABLE: Titles and labels
# ============================================================
TITLE  = "Magnetocaloric Property Landscape of Fine-Tuning Dataset\nwith Conditioning Targets"
XLABEL = "$T_C$ (K)"
YLABEL = "$|\\Delta S_{mag}|$ (J kg$^{-1}$ K$^{-1}$)"

# ============================================================
# EDITABLE: Ideal refrigeration region
# ============================================================
IDEAL_TC_MIN = 275
IDEAL_TC_MAX = 325
IDEAL_DS_MIN = 10

# ============================================================
# EDITABLE: Field colormap
# ============================================================
CMAP_NAME   = "plasma"
FIELD_LABEL = "Applied field (T)"

# ============================================================
# EDITABLE: ΔS-only conditioning targets → horizontal dotted lines
# ============================================================
DS_ONLY_TARGETS = [
    (5.3,  "Med"),
    (18.9, "+1σ"),
    (32.5, "+2σ"),
    (46.1, "+3σ"),
    (59.7, "+4σ"),
]
C_DS_LINE = "#D55E00"   # burnt orange

# ============================================================
# EDITABLE: TC-only conditioning targets → vertical dotted lines
# ============================================================
TC_ONLY_TARGETS = [
    (242, "Med"),
    (335, "+1σ"),
    (428, "+2σ"),
    (521, "+3σ"),
    (614, "+4σ"),
]
C_TC_LINE = "#0072B2"   # blue

# ============================================================
# EDITABLE: Simultaneous escalation targets → stars
# (TC K, DS J/kg/K, label)
# ============================================================
SIMUL_TARGETS = [
    (242, 5.3,  "Med/Med"),
    (335, 18.9, "+1σ/+1σ"),
    (428, 32.5, "+2σ/+2σ"),
]
C_SIMUL = "#62FF00"   # green

# ============================================================
# EDITABLE: Fixed DS=10.5, sweep TC → diamond markers
# ============================================================
FIXED_DS_TARGETS = [
    (242, 10.5),
    (335, 10.5),
    (428, 10.5),
]
C_FIXED_DS = "#00A2FF"  # sky blue

# ============================================================
# EDITABLE: Fixed TC=300, sweep DS → triangle markers
# ============================================================
FIXED_TC_TARGETS = [
    (300, 5.3),
    (300, 18.9),
    (300, 32.5),
]
C_FIXED_TC = "#FF0000"  # pink

# ============================================================
# EDITABLE: x-axis limit — extend past training max to show TC-only lines
# ============================================================
X_MAX = 650   # K

# ============================================================
# Load data
# ============================================================
train = pd.read_csv("train_repaired.csv")
val   = pd.read_csv("val_repaired.csv").dropna(subset=["n_atoms_primitive"])
both  = pd.concat([train, val], ignore_index=True)
both["ds"]    = pd.to_numeric(both["magnetocaloric_delta_s"],    errors="coerce")
both["tc"]    = pd.to_numeric(both["magnetocaloric_curie_temp"], errors="coerce")
both["field"] = pd.to_numeric(both["magnetocaloric_field"],      errors="coerce")
plot_df = both.dropna(subset=["ds", "tc", "field"]).copy()

Y_MAX = plot_df["ds"].max() * 1.08

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9.0, 5.5))

# Ideal refrigeration region
rect = mpatches.FancyBboxPatch(
    (IDEAL_TC_MIN, IDEAL_DS_MIN),
    IDEAL_TC_MAX - IDEAL_TC_MIN,
    Y_MAX - IDEAL_DS_MIN,
    boxstyle="square,pad=0",
    linewidth=1.0, edgecolor="#2ecc71",
    facecolor="#2ecc71", alpha=0.10, zorder=1,
)
ax.add_patch(rect)
ax.text(IDEAL_TC_MAX - 3, Y_MAX * 0.98,
        "Ideal\nregion",
        ha="right", va="top", fontsize=7, color="#27ae60", style="italic")

# ── Training data scatter (coloured by applied field) ────────
norm = mcolors.Normalize(vmin=plot_df["field"].min(), vmax=plot_df["field"].max())
cmap = matplotlib.colormaps[CMAP_NAME]
sc = ax.scatter(plot_df["tc"], plot_df["ds"],
                c=plot_df["field"], cmap=cmap, norm=norm,
                s=35, alpha=0.75, edgecolors="white", linewidth=0.3, zorder=3)
cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.90)
cbar.set_label(FIELD_LABEL, fontsize=9)
cbar.ax.tick_params(labelsize=8)

# ── ΔS-only: horizontal dotted lines ─────────────────────────
for ds_val, lbl in DS_ONLY_TARGETS:
    ax.axhline(ds_val, color=C_DS_LINE, lw=1.5, ls=(0, (3, 4)),
               alpha=0.55, zorder=2)
    ax.text(X_MAX - 2, ds_val + 0.7, lbl,
            ha="right", va="bottom", fontsize=7, color=C_DS_LINE)

# ── TC-only: vertical dotted lines ───────────────────────────
for tc_val, lbl in TC_ONLY_TARGETS:
    ax.axvline(tc_val, color=C_TC_LINE, lw=1.5, ls=(0, (3, 4)),
               alpha=0.55, zorder=2)
    ax.text(tc_val + 3, Y_MAX * 0.985, lbl,
            ha="left", va="top", fontsize=7, color=C_TC_LINE, rotation=0)

# ── Simultaneous escalation: stars ───────────────────────────
for tc, ds, lbl in SIMUL_TARGETS:
    ax.scatter(tc, ds, marker="*", s=260, color=C_SIMUL, zorder=7,
               edgecolors="black", linewidth=0.5)

# ── Fixed DS=10.5, sweep TC: diamond markers ──────────────────
for tc, ds in FIXED_DS_TARGETS:
    ax.scatter(tc, ds, marker="D", s=60, color=C_FIXED_DS, zorder=7,
               edgecolors="black", linewidth=0.5)

# ── Fixed TC=300, sweep DS: triangle markers ─────────────────
for tc, ds in FIXED_TC_TARGETS:
    ax.scatter(tc, ds, marker="^", s=70, color=C_FIXED_TC, zorder=7,
               edgecolors="black", linewidth=0.5)

# ── Legend ───────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_DS_LINE, alpha=0.7,
                   label="$\\Delta S$-only targets (horizontal lines)"),
    mpatches.Patch(color=C_TC_LINE, alpha=0.7,
                   label="$T_C$-only targets (vertical lines)"),
    plt.scatter([], [], marker="*", s=130, color=C_SIMUL, label="Simultaneous escalation"),
    plt.scatter([], [], marker="D", s=50,  color=C_FIXED_DS, label="Fixed $\\Delta S$=10.5, sweep $T_C$"),
    plt.scatter([], [], marker="^", s=60,  color=C_FIXED_TC, label="Fixed $T_C$=300 K, sweep $\\Delta S$"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
          framealpha=0.92, borderaxespad=0.5)

ax.set_title(TITLE, fontsize=11, pad=10)
ax.set_xlabel(XLABEL, fontsize=11)
ax.set_ylabel(YLABEL, fontsize=11)
ax.set_xlim(0, X_MAX)
ax.set_ylim(0, Y_MAX)
ax.tick_params(labelsize=10)

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"figures/fig10_delta_s_vs_tc_scatter.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved fig10 — {len(plot_df)} structures, TC {plot_df['tc'].min():.0f}–{plot_df['tc'].max():.0f} K, "
      f"DS {plot_df['ds'].min():.1f}–{plot_df['ds'].max():.1f} J/kg/K, "
      f"x-axis extended to {X_MAX} K to show TC-only extrapolation targets")
plt.close()
