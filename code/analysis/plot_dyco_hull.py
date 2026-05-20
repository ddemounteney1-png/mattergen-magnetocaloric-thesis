import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.image import imread
import numpy as np
import csv, os

from ase.io import read
from ase.visualize.plot import plot_atoms

DATA_CSV   = "/home/daniel/Data_analysis/convex_hull/dyco_hull_data.csv"
STRUCT_DIR = "/home/daniel/Data_analysis/convex_hull/structures"
IMG_DIR    = "/home/daniel/Data_analysis/convex_hull/structure_images"
OUT_DIR    = "/home/daniel/Data_analysis/figures"
os.makedirs(IMG_DIR, exist_ok=True)

# ── Load hull data ─────────────────────────────────────────────────────────
stable = []
with open(DATA_CSV) as f:
    for row in csv.DictReader(f):
        stable.append({
            "formula":  row["formula"],
            "co_frac":  float(row["co_fraction"]),
            "form_e":   float(row["formation_energy_per_atom"]),
            "mp_id":    row["mp_id"],
        })
stable.sort(key=lambda x: x["co_frac"])

# ── Element colours (consistent across all renders) ───────────────────────
# ASE uses its own colour table; override via ase.data.colors
from ase.data.colors import jmol_colors
from ase.data import atomic_numbers
DY_COLOR = np.array([0.13, 0.40, 0.73])   # steel blue
CO_COLOR = np.array([0.75, 0.18, 0.18])   # dark red
jmol_colors[atomic_numbers["Dy"]] = DY_COLOR
jmol_colors[atomic_numbers["Co"]] = CO_COLOR

# ── Render structure images ────────────────────────────────────────────────
# Rotation angles tuned to give clear views for each structure type
rotations = {
    "Dy":      "0x,0y,0z",
    "Dy3Co":   "10x,20y,0z",
    "Dy12Co7": "10x,15y,0z",
    "DyCo2":   "15x,10y,0z",
    "DyCo3":   "10x,20y,0z",
    "Dy2Co17": "10x,15y,0z",
    "Co":      "0x,0y,0z",
}

print("Rendering structure images...")
for d in stable:
    img_path = f"{IMG_DIR}/{d['formula']}.png"
    mp_id_raw = d["mp_id"]
    cif_path  = f"{STRUCT_DIR}/{d['formula']}_{mp_id_raw}.cif"

    atoms = read(cif_path)
    rot   = rotations.get(d["formula"], "10x,10y,0z")

    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    plot_atoms(atoms, ax, radii=0.45, rotation=rot)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.savefig(img_path, dpi=180, transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"  {d['formula']}: {len(atoms)} atoms")

# ── DyCo5 (off-hull, GGA artefact) ────────────────────────────────────────
dyco5 = {
    "formula": "DyCo5",
    "co_frac": 5 / 6,           # 0.8333
    "form_e": -0.1119,          # eV/atom from MP (mp-2551)
    "e_above_hull": 0.0196,     # eV/atom above hull
}

# ── Compose the convex hull figure ────────────────────────────────────────
co_fracs = [d["co_frac"] for d in stable]
energies = [d["form_e"]  for d in stable]
formulas = [d["formula"] for d in stable]

# ── Image + label offsets (edit here to reposition) ──────────────────────────
# All values in points. Positive x = right, positive y = up.
# x_off / y_off : centre of the structure image box relative to the hull point
# label is placed at (x_off, y_off - LABEL_GAP) — always below the image
# energy label is fixed LEFT of every hull point (see ENERGY_X_OFF)
LABEL_GAP    = 50   # pts below image centre where formula text sits
ENERGY_X_OFF = -8   # pts left of hull point for "X.XXX eV/at" text

#               Dy    Dy3Co  Dy12Co7  DyCo2  DyCo3  Dy2Co17   Co
y_offsets = [ -110,    100,     -90,    70,    -90,    -100,  -120]
x_offsets = [    0,     15,     -30,   -80,    -20,       0,     0]

plt.style.use("seaborn-v0_8-paper")
fig, ax = plt.subplots(figsize=(14, 7))

# Hull line
ax.plot(co_fracs, energies, color="#2C3E50", linewidth=1.8, zorder=2)

# Stable hull points (filled)
ax.scatter(co_fracs, energies, s=100, color="#2C3E50", zorder=4,
           edgecolors="white", linewidth=1.2)

# DyCo5 off-hull point (hollow diamond)
ax.scatter(dyco5["co_frac"], dyco5["form_e"], s=120, marker="D",
           facecolors="white", edgecolors="#2C3E50", linewidth=1.8, zorder=4)

# Reference line at 0
ax.axhline(0, color="grey", linestyle="--", linewidth=0.9, alpha=0.5, zorder=1)

# Dashed vertical drop from DyCo5 to hull line to show e_above_hull
# Find the hull energy at Co=5/6 by linear interpolation between DyCo3 and Dy2Co17
dyco3  = next(d for d in stable if d["formula"] == "DyCo3")
dy2co17 = next(d for d in stable if d["formula"] == "Dy2Co17")
x0, y0 = dyco3["co_frac"],  dyco3["form_e"]
x1, y1 = dy2co17["co_frac"], dy2co17["form_e"]
hull_e_at_dyco5 = y0 + (y1 - y0) * (dyco5["co_frac"] - x0) / (x1 - x0)
ax.annotate("", xy=(dyco5["co_frac"], hull_e_at_dyco5),
            xytext=(dyco5["co_frac"], dyco5["form_e"]),
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=1.0, linestyle="dashed"))
ax.annotate(f"+{dyco5['e_above_hull']*1000:.0f} meV/atom\nabove hull",
            xy=(dyco5["co_frac"], (dyco5["form_e"] + hull_e_at_dyco5) / 2),
            xytext=(18, 0), textcoords="offset points",
            fontsize=8, color="#888888", va="center")

# ── Structure images + labels for stable phases ───────────────────────────
ZOOM = 0.38

def place_point(ax, x, y, formula, y_off, color="#1a1a1a", label_suffix="", x_off=0):
    sign = 1 if y_off > 0 else -1
    img_path = f"{IMG_DIR}/{formula}.png"
    if os.path.exists(img_path):
        img = imread(img_path)
        imagebox = OffsetImage(img, zoom=ZOOM)
        ab = AnnotationBbox(
            imagebox, (x, y),
            xybox=(x_off, y_off),
            xycoords="data",
            boxcoords="offset points",
            frameon=True,
            bboxprops=dict(boxstyle="round,pad=0.15", fc="white",
                           ec="#cccccc", lw=0.8, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.9),
            zorder=5,
        )
        ax.add_artist(ab)
    # Formula label — always below image (x_off keeps it aligned with image)
    ax.annotate(
        formula + label_suffix,
        xy=(x, y), xytext=(x_off, y_off - LABEL_GAP),
        textcoords="offset points",
        ha="center", va="top",
        fontsize=11, fontweight="bold", color=color, zorder=6,
    )
    # Energy label — always left of hull point, independent of image position
    ax.annotate(
        f"{y:.3f} eV/at",
        xy=(x, y), xytext=(ENERGY_X_OFF, 0),
        textcoords="offset points",
        ha="right", va="center",
        fontsize=7.5, color="#555555", zorder=6,
    )

for i, d in enumerate(stable):
    place_point(ax, d["co_frac"], d["form_e"], d["formula"], y_offsets[i], x_off=x_offsets[i])

# DyCo5 — offset up and to the left to avoid overlap with Dy2Co17
place_point(ax, dyco5["co_frac"], dyco5["form_e"], "DyCo5",
            y_off=110, color="#777777", label_suffix="*", x_off=-110)

# Axis formatting
ax.set_xlabel("Co content (atomic fraction)", fontsize=13)
ax.set_ylabel("Formation energy (eV / atom)", fontsize=13)
ax.set_xlim(-0.08, 1.08)
e_min = min(energies)
ax.set_ylim(e_min * 1.60, 0.08)
ax.tick_params(labelsize=11)
ax.set_title("Dy–Co Binary Convex Hull  (Materials Project GGA)", fontsize=14, pad=10)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

# Legend
dy_patch   = mpatches.Patch(color=DY_COLOR, label="Dy")
co_patch   = mpatches.Patch(color=CO_COLOR, label="Co")
hull_pt    = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2C3E50",
                        markersize=9, label="Stable (on hull)")
offhull_pt = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                        markeredgecolor="#2C3E50", markeredgewidth=1.8,
                        markersize=9, label="Off-hull (GGA artefact)*")
ax.legend(handles=[dy_patch, co_patch, hull_pt, offhull_pt],
          fontsize=10, loc="lower center", title="Elements / phases",
          title_fontsize=9, framealpha=0.9, ncol=2)

fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    path = f"{OUT_DIR}/dyco_convex_hull.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
