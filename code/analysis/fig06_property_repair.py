import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ============================================================
# EDITABLE: Titles and labels
# ============================================================
SUPTITLE = "Property Assignment Repair: $\\Delta S_{mag}$ Values"
XLABEL   = "Correct $\\Delta S_{mag}$ (J kg$^{-1}$ K$^{-1}$)"
YLABEL_A = "Assigned $\\Delta S_{mag}$ before repair (J kg$^{-1}$ K$^{-1}$)"

# ============================================================
# EDITABLE: Colours
# ============================================================
C_OK      = "#0072B2"   # already-correct assignments
C_ERROR   = "#E74C3C"   # incorrect assignments (off 1:1 line)
C_REFLINE = "#222222"

# ── Load repair reports (train + val) ────────────────────────
both = pd.concat([
    pd.read_csv("Processing_data/train_repair_report.csv"),
    pd.read_csv("Processing_data/val_repair_report.csv"),
])
df = both.dropna(subset=["old_delta_s", "new_delta_s"]).copy()
df["changed"] = df["old_delta_s"] != df["new_delta_s"]

n_total   = len(df)
n_changed = df["changed"].sum()
n_ok      = n_total - n_changed

AXIS_MAX = 130

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
fig.subplots_adjust(wspace=0.42)
fig.suptitle(SUPTITLE, fontsize=12, y=1.02)

# ── Panel A: Before repair ────────────────────────────────────
ax = axes[0]
ax.set_aspect("equal", adjustable="box")
ax.plot([0, AXIS_MAX], [0, AXIS_MAX], color=C_REFLINE, ls="--", lw=1.1,
        label="1:1 reference", zorder=2)

correct = df[~df["changed"]]
wrong   = df[df["changed"]]

ax.scatter(correct["new_delta_s"], correct["old_delta_s"],
           color=C_OK, s=14, alpha=0.55, label=f"Unchanged ({n_ok})", zorder=3)
ax.scatter(wrong["new_delta_s"], wrong["old_delta_s"],
           color=C_ERROR, s=14, alpha=0.65, label=f"Reassigned ({n_changed})", zorder=4)

ax.set_xlim(0, AXIS_MAX); ax.set_ylim(0, AXIS_MAX)
ax.set_xlabel(XLABEL, fontsize=9)
ax.set_ylabel(YLABEL_A, fontsize=8.5)
ax.set_title("Before Repair", fontsize=11, pad=6)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8, loc="lower right")
ax.text(0.04, 0.97, "A", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
pct = n_changed / n_total * 100
ax.text(0.04, 0.88, f"{pct:.0f}% of assignments\nwere incorrect",
        transform=ax.transAxes, fontsize=8, va="top", color=C_ERROR)

# ── Panel B: After repair (all on 1:1) ───────────────────────
ax2 = axes[1]
ax2.set_aspect("equal", adjustable="box")
ax2.plot([0, AXIS_MAX], [0, AXIS_MAX], color=C_REFLINE, ls="--", lw=1.1,
         label="1:1 reference", zorder=2)

ax2.scatter(df["new_delta_s"], df["new_delta_s"],
            color=C_OK, s=14, alpha=0.55, label=f"All repaired ({n_total})", zorder=3)

ax2.set_xlim(0, AXIS_MAX); ax2.set_ylim(0, AXIS_MAX)
ax2.set_xlabel(XLABEL, fontsize=9)
ax2.set_ylabel("")
ax2.set_title("After Repair", fontsize=11, pad=6)
ax2.tick_params(labelsize=8)
ax2.legend(fontsize=8, loc="lower right")
ax2.text(0.04, 0.97, "B", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top")

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig06_property_repair.{ext}", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig06_property_repair.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig06 — {n_total} structures, {n_changed} reassigned ({pct:.0f}%), {n_ok} unchanged")
plt.close()
