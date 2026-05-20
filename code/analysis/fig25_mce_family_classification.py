import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from collections import Counter

# ============================================================
# EDITABLE: Run paths, grouped by conditioning level
# ============================================================
LEVEL_RUNS = {
    "Median":      [
        "Metrics/06_fixed_deltaS_TC_sweep/ds10p5_tc242/detailed_metrics.json",
        "Metrics/fixed_Tc_delta_S_sweep/deltaS_5.3_TC_300_detailed_metrics.json",
        "Metrics/07_simultaneous_escalation/ds5p3_tc242/detailed_metrics.json",
    ],
    "+1$\\sigma$": [
        "Metrics/06_fixed_deltaS_TC_sweep/ds10p5_tc335/detailed_metrics.json",
        "Metrics/fixed_Tc_delta_S_sweep/deltaS_18.9_TC_300_detailed_metrics.json",
        "Metrics/07_simultaneous_escalation/ds18p9_tc335/detailed_metrics.json",
    ],
    "+2$\\sigma$": [
        "Metrics/06_fixed_deltaS_TC_sweep/ds10p5_tc428/detailed_metrics.json",
        "Metrics/fixed_Tc_delta_S_sweep/deltaS_32.5_TC_300_detailed_metrics.json",
        "Metrics/07_simultaneous_escalation/ds32p5_tc428/detailed_metrics.json",
    ],
}

# ============================================================
# EDITABLE: MCE family classification rules
# Applied in order — first match wins
# ============================================================
RE_ELEMENTS = {'La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Y','Sc'}
LA_TYPE_RE  = {'La','Ce','Pr','Nd'}      # La(Fe,Si)13 host RE elements
HEUSLER_B   = {'Ni','Co','Cu','Pd','Pt','Rh','Fe'}  # Heusler B-site metals
HEUSLER_Z   = {'Ga','In','Al','Sn','Sb','Ge','Zn','Bi','Si'}  # Heusler Z-site elements

def classify(comp_dict, is_novel):
    els = set(comp_dict.keys())

    # Manganite perovskite: Mn + O (oxide-based)
    if 'Mn' in els and 'O' in els:
        return 'Manganite'

    # La(Fe,Si)13 family: La/Ce/Pr/Nd + Fe + (Si/Ge/Al)
    if (els & LA_TYPE_RE) and 'Fe' in els and (els & {'Si', 'Ge', 'Al'}):
        return 'La(Fe,Si)$_{13}$'

    # MnFe(P,Si): Mn + Fe + P (phosphorus is the key marker)
    if 'Mn' in els and 'Fe' in els and 'P' in els:
        return 'MnFe(P,Si)'

    # Heusler alloy: Mn + (Ni/Co/Cu/Fe/Rh/Pd) + (Ga/In/Al/Sn/Ge/Si/...), ≥3 elements
    if 'Mn' in els and (els & HEUSLER_B) and (els & HEUSLER_Z) and len(els) >= 3:
        return 'Heusler'

    # RE2Fe17: RE + Fe (Fe-rich, no Mn/O/P/Si/Ge)
    if (els & RE_ELEMENTS) and 'Fe' in els and not (els & {'Mn','O','P','C','Si','Ge'}):
        re_n = sum(v for k, v in comp_dict.items() if k in RE_ELEMENTS)
        fe_n = comp_dict.get('Fe', 0)
        if re_n > 0 and fe_n / re_n > 3:
            return 'RE$_2$Fe$_{17}$'

    # Gd-based (not already classified)
    if 'Gd' in els:
        return 'Gd-based'

    # Other RE compound
    if els & RE_ELEMENTS:
        return 'Other RE'

    # Other Mn compound (no RE, no O, no P, just Mn-bearing)
    if 'Mn' in els:
        return 'Other Mn'

    # Stable + novel + none of the above → genuinely novel discovery
    if is_novel:
        return 'Novel (other)'

    return 'Other'

# ============================================================
# EDITABLE: Family display order and colours
# ============================================================
FAMILY_ORDER = [
    'Heusler',
    'MnFe(P,Si)',
    'La(Fe,Si)$_{13}$',
    'Gd-based',
    'RE$_2$Fe$_{17}$',
    'Manganite',
    'Other RE',
    'Other Mn',
    'Novel (other)',
    'Other',
]

FAMILY_COLOURS = {
    'Heusler':            '#CC79A7',
    'MnFe(P,Si)':         '#009E73',
    'La(Fe,Si)$_{13}$':   '#0072B2',
    'Gd-based':           '#D55E00',
    'RE$_2$Fe$_{17}$':    '#56B4E9',
    'Manganite':          '#E69F00',
    'Other RE':           '#F0E442',
    'Other Mn':           '#bbbbbb',
    'Novel (other)':      '#2d2d2d',
    'Other':              '#eeeeee',
}

# ── Classify all stable structures per level ──────────────────
family_counts   = {}   # level → Counter
novel_stable    = {}   # level → {'novel': n, 'non_novel': n}

for level, paths in LEVEL_RUNS.items():
    fc = Counter()
    nv = {'novel': 0, 'non_novel': 0}
    for path in paths:
        d = json.load(open(path))
        for i, entry in enumerate(d['entry']):
            if d['stable'][i]:
                fam = classify(entry['composition'], d['novel'][i])
                fc[fam] += 1
                if d['novel'][i]:
                    nv['novel'] += 1
                else:
                    nv['non_novel'] += 1
    family_counts[level] = fc
    novel_stable[level]  = nv

levels = list(LEVEL_RUNS.keys())

# ── Print classification summary ──────────────────────────────
print("Family classification summary:")
for level in levels:
    total = sum(family_counts[level].values())
    print(f"\n{level} (n_stable={total}):")
    for fam in FAMILY_ORDER:
        n = family_counts[level].get(fam, 0)
        if n:
            print(f"  {fam}: {n}")

# ── Plot ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(10.0, 9.0))
gs = gridspec.GridSpec(2, 3, height_ratios=[3.5, 2.0],
                       hspace=0.35, wspace=0.25,
                       left=0.09, right=0.97, top=0.93, bottom=0.06)

ax_bar = fig.add_subplot(gs[0, :])   # full-width stacked bar
ax_pie = [fig.add_subplot(gs[1, j]) for j in range(3)]  # 3 pie charts

# ── Stacked bar ───────────────────────────────────────────────
x = np.arange(len(levels))
BAR_WIDTH = 0.52
bottoms = np.zeros(len(levels))

for fam in FAMILY_ORDER:
    vals = np.array([family_counts[lv].get(fam, 0) for lv in levels], dtype=float)
    if vals.sum() == 0:
        continue
    bars = ax_bar.bar(x, vals, BAR_WIDTH, bottom=bottoms,
                      color=FAMILY_COLOURS[fam], edgecolor="white",
                      linewidth=0.6, label=fam, zorder=3)
    # Label slices with count if large enough
    for xi, val, bot in zip(x, vals, bottoms):
        if val >= 3:
            ax_bar.text(xi, bot + val / 2, str(int(val)),
                        ha="center", va="center", fontsize=8,
                        color="white" if FAMILY_COLOURS[fam] not in ('#F0E442','#eeeeee') else "#444444",
                        fontweight="bold")
    bottoms += vals

# Total count above each bar
for xi, total in zip(x, bottoms):
    ax_bar.text(xi, total + 0.4, f"n={int(total)}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(levels, fontsize=11)
ax_bar.set_ylabel("Number of stable structures", fontsize=10)
ax_bar.set_title(
    "MCE Material Family Classification of Stable Generated Structures\n"
    "(combined across Fixed $\\Delta S$, Fixed $T_C$, and Simultaneous strategies)",
    fontsize=10.5, pad=8)
ax_bar.set_ylim(0, bottoms.max() * 1.12)
ax_bar.tick_params(axis="y", labelsize=9)
ax_bar.legend(fontsize=8, loc="upper right", ncol=2,
              framealpha=0.92, title="Material family", title_fontsize=8)
ax_bar.text(0.01, 0.98, "A", transform=ax_bar.transAxes,
            fontsize=13, fontweight="bold", va="top")

# ── Pie charts: novel vs non-novel stable ─────────────────────
C_NOVEL     = "#009E73"
C_NON_NOVEL = "#E69F00"

for j, (ax_p, level) in enumerate(zip(ax_pie, levels)):
    nv = novel_stable[level]
    n_nov  = nv['novel']
    n_known = nv['non_novel']
    total  = n_nov + n_known
    sizes  = [n_nov, n_known]
    colours = [C_NOVEL, C_NON_NOVEL]
    labels  = [f"Novel\nn={n_nov}\n({100*n_nov/total:.0f}%)",
               f"Non-novel\nn={n_known}\n({100*n_known/total:.0f}%)"]

    wedges, texts = ax_p.pie(
        sizes, colors=colours, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
        radius=0.9,
    )
    ax_p.set_title(level, fontsize=10, pad=4, fontweight="bold")

    # Manual annotation of slices
    for wedge, lbl in zip(wedges, labels):
        angle = (wedge.theta1 + wedge.theta2) / 2
        x_txt = 0.62 * np.cos(np.deg2rad(angle))
        y_txt = 0.62 * np.sin(np.deg2rad(angle))
        ax_p.text(x_txt, y_txt, lbl, ha="center", va="center",
                  fontsize=8, color="white" if wedge.get_facecolor()[0] < 0.7 else "#333333",
                  fontweight="bold")

    ax_p.text(-1.1, 1.1, "BCD"[j], fontsize=12, fontweight="bold", va="top")

# Shared pie legend
pie_handles = [
    mpatches.Patch(color=C_NOVEL,     label="Stable + Novel (genuine discovery)"),
    mpatches.Patch(color=C_NON_NOVEL, label="Stable + Non-novel (reproduces known structure)"),
]
fig.legend(handles=pie_handles, fontsize=8.5, loc="lower center",
           ncol=2, bbox_to_anchor=(0.5, 0.0), framealpha=0.95)

os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"figures/fig25_mce_family_classification.{ext}",
                dpi=300, bbox_inches="tight")
print("\nSaved fig25 — MCE family classification stacked bar + novelty pie charts")
plt.close()
