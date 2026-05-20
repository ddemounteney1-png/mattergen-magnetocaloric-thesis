import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================================================
# EDITABLE: Titles and labels
# ============================================================
TITLE  = "Unique Compositions by Material Class (Franco et al. 2012)"
XLABEL = "Number of unique compositions"
YLABEL = ""
EXCLUDED_CLASS  = "Amorphous_alloys"
EXCLUDED_LABEL  = "(excluded from training)"

# ============================================================
# EDITABLE: Display name mapping (underscore → readable)
# ============================================================
CLASS_NAMES = {
    "Manganites":              "Manganites",
    "Rare_earth_free_alloys":  "Rare-earth-free alloys",
    "La(Fe,Si)13_family":      "La(Fe,Si)₁₃ family",
    "Laves_phases":            "Laves phases",
    "Fe3C_and_other_RE":       "Fe₃C & other RE",
    "R2Fe17_stoichiometries":  "R₂Fe₁₇ stoichiometries",
    "Elemental":               "Elemental",
    "Gd5(Si,Ge)4_family":     "Gd₅(Si,Ge)₄ family",
    "Amorphous_alloys":        "Amorphous alloys (excluded)",
    "Heusler_alloys":          "Heusler alloys",
    "124_family":              "124 family",
}

# ============================================================
# EDITABLE: Colours
# ============================================================
C_BAR      = "#0072B2"
C_EXCLUDED = "#AAAAAA"

# ============================================================
# Load data — count unique formulas per class
# ============================================================
rel = pd.read_csv("Processing_data/relaxation_results.csv")
counts = (rel.dropna(subset=["material_class"])
             .drop_duplicates(subset=["material_formula"])
             .groupby("material_class")["material_formula"]
             .count()
             .reset_index()
             .rename(columns={"material_formula": "count"}))

excl = counts[counts["material_class"] == EXCLUDED_CLASS]
main = counts[counts["material_class"] != EXCLUDED_CLASS].sort_values("count")

# Map names
def display(cls):
    return CLASS_NAMES.get(cls, cls.replace("_", " "))

main["display"] = main["material_class"].map(display)
excl_count = excl["count"].sum() if len(excl) else 0

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(4.0, max(3.5, len(main) * 0.42 + 1.0)))

bars = ax.barh(main["display"], main["count"],
               color=C_BAR, edgecolor="white", linewidth=0.6)

# Count labels
for bar in bars:
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", ha="left", fontsize=9)

ax.set_title(TITLE, fontsize=11, pad=8)
ax.set_xlabel(XLABEL, fontsize=10)
ax.tick_params(labelsize=9)
ax.set_xlim(0, main["count"].max() * 1.20)

plt.tight_layout(pad=1.5)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig09_material_class_breakdown.{ext}", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig09_material_class_breakdown.jpg", dpi=150, bbox_inches="tight",
            pil_kwargs={"quality": 90})
print(f"Saved fig09 — {len(main)} classes plotted, {excl_count} excluded")
plt.close()
