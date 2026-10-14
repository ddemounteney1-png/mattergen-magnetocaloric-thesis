"""
process_icsd_failed.py
======================
Generates ordered CIFs for previously-failed materials using ICSD template
structures found in the 'Online Cifs' folder.

Two families are addressed:

  1. Gd5(Si,Ge,X)4 family  — monoclinic P1121/a (SG 14)
     Template: Online Cifs/Gd5Ge2Si2.cif  (ICSD 84084)
     36 atoms per unit cell → icsd_derived_cifs/ for review

  2. YFe2(H/D)4.2 family   — monoclinic P1c1 (SG 7)
     Template: Online Cifs/YFe2D4.2.cif   (ICSD 260534)
     ~58 atoms per unit cell → icsd_derived_cifs/ for review

IMPORTANT — Minority dopant policy:
  For any dopant whose round(frac × n_sites) == 0, force n_dopant = 1 atom.
  The majority species is scaled down proportionally. The model must learn
  which atoms are present (chemical identity) above exact stoichiometry.

Outputs:
  icsd_derived_cifs/          — new CIFs for user review before adopting
  icsd_derived_cifs/icsd_recovery_log.csv

After reviewing, copy approved CIFs to:
  cif_files/   (≤50 atoms — Gd5 family, 36 atoms each)
  cif_>50/     (51-60 atoms — YFe2 family, ~58 atoms)

NOTE: The Gd5(Si,Ge)4 CIFs currently in clean_cifs/ use SG 62 Pnma
(orthorhombic, from Materials Project). The correct phase for the giant MCE
is SG 14 P1121/a (monoclinic). These new CIFs replace those — please confirm
before copying.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PREPROCESSING_WORK  # noqa: E402


import os, re, csv, warnings, logging
import numpy as np
from collections import defaultdict

from pymatgen.core import Structure, Element, Site
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Species
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import (
    OrderDisorderedStructureTransformation,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── PATHS ──────────────────────────────────────────

BASE_DIR      = str(PREPROCESSING_WORK)
ONLINE_CIFS   = os.path.join(BASE_DIR, "Online Cifs")
OUT_DIR       = os.path.join(BASE_DIR, "icsd_derived_cifs")
LOG_CSV       = os.path.join(OUT_DIR, "icsd_recovery_log.csv")

GD5_TEMPLATE  = os.path.join(ONLINE_CIFS, "Gd5Ge2Si2.cif")
YFE2D_TEMPLATE= os.path.join(ONLINE_CIFS, "YFe2D4.2.cif")

MAX_ATOMS       = 50
MAX_ATOMS_LARGE = 60

# ─────────────────────── MATERIALS TO PROCESS ───────────────────────────────

# Gd5(Si,Ge,X)4 — currently have Pnma CIFs (wrong phase); need monoclinic.
GD5_TARGETS = {
    "Gd5Ge2Si2":                   "M (P1121/a)",
    "Gd5Si2Ge2":                   "M (P1121/a)",
    "Gd5Ge2.28Si1.72":             "M (P1121/a)",
    "Gd5Si1.72Ge2.28":             "M (P1121/a)",
    "Gd5Si1Ge3":                   "M (P1121/a)",
    "Gd5Si0.9Ge3.1":               "M (P1121/a)",
    "Gd5Si1.95Ge2.05":             "M (P1121/a)",
    "Gd5Si2.05Ge1.95":             "M (P1121/a)",
    "Gd5Si2.035Ge1.935Mn0.03":     "M (P1121/a)",
    "Gd5Si2.03Ge1.93Mn0.08":       "M (P1121/a)",
    "Gd5Si1.925Ge2.025In0.05":     "M (P1121/a)",
    "Gd5Si1.975Ge1.975Nb0.05":     "Mixed",
    "Gd5Si1.95Ge1.95Nb0.1":        "Mixed",
    "Gd5Si1.925Ge1.925Nb0.15":     "Mixed",
    "Gd5Si1.9Ge1.9Nb0.2":          "Mixed",
    "Gd5Si0.935Ge2.035Sn0.03":     "M (P1121/a)",
}

# YFe2(H/D)4.2 — genuinely absent (no CIF exists)
YFE2_TARGETS = {
    "YFe2D4.2":         "monoclinic",
    "YFe2H4.2":         "monoclinic",
    "Y0.5Tb0.5Fe2D4.2": "monoclinic",
}

# ─────────────────────────── UTILITIES ──────────────────────────────────────

def safe_cif_name(formula: str) -> str:
    n = formula.replace("(", "_").replace(")", "_").replace(".", "p")
    n = re.sub(r"[^A-Za-z0-9_]", "_", n)
    return re.sub(r"_+", "_", n).strip("_") + ".cif"


def elem_symbol(sp) -> str:
    """
    Return the chemical symbol of a pymatgen Species or Element.
    Handles deuterium (D): pymatgen stores D as Species('D', 0) with
    element=Element('H'), so we use str(sp)[0] for the actual symbol.
    """
    s = str(sp)  # e.g. 'D0+', 'Si0+', 'Gd0+'
    # Strip charge suffix and return leading element symbol
    # Regex: match one or two uppercase+lowercase letters at start
    m = re.match(r"([A-Z][a-z]?)", s)
    return m.group(1) if m else s


def force_min_atoms(raw_fracs: dict, n_total_sites: int) -> dict:
    """
    Ensure every species with frac > 0 gets ≥ 1 atom across n_total_sites.
    The majority species absorbs any reduction.
    Returns adjusted fracs summing to 1.0.
    """
    if not raw_fracs or n_total_sites == 0:
        return raw_fracs

    sorted_sp = sorted(raw_fracs.items(), key=lambda x: x[1])   # ascending
    majority_sp = sorted_sp[-1][0]

    n_assigned = {}
    used = 0
    for sp, frac in sorted_sp[:-1]:
        if frac <= 0:
            n_assigned[sp] = 0
            continue
        n_raw = round(frac * n_total_sites)
        if n_raw == 0:
            n_raw = 1
            log.info(f"    Forced 1 atom of {sp} "
                     f"(target frac={frac:.4f} on {n_total_sites} sites)")
        n_assigned[sp] = n_raw
        used += n_raw

    n_assigned[majority_sp] = max(1, n_total_sites - used)
    total = sum(n_assigned.values())
    return {sp: n / total for sp, n in n_assigned.items()}


def odst_ordered(struct: Structure) -> tuple:
    """
    Order a disordered structure via ODST (Ewald minimisation).
    Falls back to even-distribution if ODST fails.
    Returns (ordered_struct, method_str).
    """
    if struct.is_ordered:
        return struct, "already_ordered"
    try:
        odst    = OrderDisorderedStructureTransformation(algo=0)
        ordered = odst.apply_transformation(struct)
        return ordered, "ODST"
    except Exception as e:
        log.warning(f"    ODST failed ({e}); using even-distribution fallback")
        return _distribute_ordered(struct), "distribute_fallback"


def _distribute_ordered(struct: Structure) -> Structure:
    """
    Even-distribution ordering: places minority atoms at evenly-spaced
    positions within each group of identical mixed sites, guaranteeing
    ≥ 1 atom per species with frac > 0.
    """
    result = struct.copy()
    groups = defaultdict(list)
    for i, site in enumerate(result):
        if not site.is_ordered:
            key = tuple(sorted((str(k), float(v))
                               for k, v in site.species.items()))
            groups[key].append(i)

    for key, site_indices in groups.items():
        n    = len(site_indices)
        raw  = dict(key)
        srt  = sorted(raw.items(), key=lambda x: x[1])   # ascending
        maj  = srt[-1][0]

        n_asgn = {}
        used   = 0
        for sp, frac in srt[:-1]:
            if frac <= 0:
                n_asgn[sp] = 0; continue
            cnt = max(1, round(frac * n))
            cnt = min(cnt, n - used - 1)
            n_asgn[sp] = cnt
            used += cnt
        n_asgn[maj] = max(1, n - used)

        assignment = [maj] * n
        offset = 0.0
        for sp, frac in srt[:-1]:
            cnt = n_asgn.get(sp, 0)
            if not cnt: continue
            step = n / cnt
            for k in range(cnt):
                idx = int((offset + k * step)) % n
                while assignment[idx] != maj:
                    idx = (idx + 1) % n
                assignment[idx] = sp
            offset += step / 2

        for site_idx, sp_str in zip(site_indices, assignment):
            result.replace(site_idx, sp_str)
    return result


def write_cif(struct: Structure, path: str) -> None:
    try:
        CifWriter(struct, symprec=0.1).write_file(path)
    except Exception:
        CifWriter(struct, symprec=None).write_file(path)


def report_sg(struct: Structure) -> tuple:
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.2)
        return sga.get_space_group_number(), sga.get_space_group_symbol()
    except Exception:
        return "", ""


# ═══════════════════════════ Gd5(Si,Ge,X)4 ═══════════════════════════════════

def process_gd5_variant(formula: str, template: Structure) -> dict:
    """
    Generate one ordered monoclinic P1121/a CIF for a Gd5(Si,Ge,X)4 variant.

    Steps:
      1. Parse target composition (Si, Ge, optional dopant).
      2. Find all mixed Si/Ge Wyckoff sites in the template (16 sites, 4e × 4).
      3. Compute fractional occupancies; force ≥ 1 atom for every minority dopant.
      4. Replace site occupancies in template; call ODST (Ewald) to order.
    """
    log.info(f"  → {formula}")
    res = {"formula": formula, "family": "Gd5(Si,Ge,X)4",
           "prototype": "", "status": "", "method": "", "n_atoms": "",
           "sg_number": "", "sg_symbol": "", "stoich_error_pct": "",
           "cif_path": "", "notes": ""}

    # Parse composition
    try:
        comp = Composition(formula)
    except Exception as e:
        res.update(status="FAILED", notes=f"formula parse error: {e}")
        return res

    gd    = float(comp.get(Element("Gd"), 0))
    si    = float(comp.get(Element("Si"), 0))
    ge    = float(comp.get(Element("Ge"), 0))
    dop   = {str(el): float(v)
             for el, v in comp.items() if str(el) not in ("Gd", "Si", "Ge")}
    total_anion = si + ge + sum(dop.values())
    if total_anion <= 0:
        res.update(status="FAILED", notes="no anion elements"); return res

    raw_fracs = {"Si": si/total_anion, "Ge": ge/total_anion}
    raw_fracs.update({el: cnt/total_anion for el, cnt in dop.items()})

    # Locate mixed Si/Ge sites (identified by element symbol)
    mixed_idx = [i for i, s in enumerate(template)
                 if len(s.species) > 1
                 and any(elem_symbol(sp) in ("Si", "Ge")
                         for sp in s.species)]
    n_mixed = len(mixed_idx)
    if n_mixed == 0:
        res.update(status="FAILED", notes="no mixed Si/Ge sites in template")
        return res

    # Adjust fracs: force ≥ 1 atom per minority species
    adj_fracs = force_min_atoms(raw_fracs, n_mixed)

    # Replace site occupancies using Species objects (required by ODST)
    # All species get oxi_state=0 to match the CIF convention
    working = template.copy()
    sp_dict  = {el: Species(el, 0) for el in adj_fracs}
    occ_dict = {sp_dict[el]: frac for el, frac in adj_fracs.items()}

    for i in mixed_idx:
        working.replace(i, occ_dict)

    # Order
    ordered, method = odst_ordered(working)

    # Stoichiometry error: max |actual_global_frac - target_global_frac|
    total_target_atoms = gd + total_anion
    actual_comp = ordered.composition
    worst_err = 0.0
    for el_sym, raw_frac in raw_fracs.items():
        try:
            actual_count = float(actual_comp.get(Element(el_sym), 0))
        except Exception:
            actual_count = 0.0
        actual_gfrac = actual_count / len(ordered)
        target_gfrac = raw_frac * (total_anion / total_target_atoms)
        worst_err = max(worst_err, abs(actual_gfrac - target_gfrac))

    sg_num, sg_sym = report_sg(ordered)

    notes = []
    for el, frac in adj_fracs.items():
        if el not in ("Si", "Ge") and frac > 0:
            notes.append(
                f"{el}_1atom_in_{n_mixed}sites"
                f"(target_frac={raw_fracs[el]:.4f})"
            )

    cif_path = os.path.join(OUT_DIR, safe_cif_name(formula))
    write_cif(ordered, cif_path)
    log.info(f"    {len(ordered)} atoms | SG {sg_num} {sg_sym} | "
             f"err={worst_err*100:.2f}% → {os.path.basename(cif_path)}")

    res.update(
        status           = "OK" if worst_err < 0.10 else "WARN",
        method           = method,
        n_atoms          = len(ordered),
        sg_number        = sg_num,
        sg_symbol        = sg_sym,
        stoich_error_pct = f"{worst_err*100:.2f}",
        cif_path         = cif_path,
        notes            = "; ".join(notes),
    )
    return res


# ═══════════════════════════ YFe2(H/D)4.2 ═════════════════════════════════════

def _resolve_d_vacancies(struct: Structure, target_d_total: float,
                         d_sym: str = "D") -> Structure:
    """
    Resolve partial D/H site occupancies by choosing the top `target_d_total`
    (rounded to nearest integer) positions to keep, ranked by original occupancy.
    Sites not kept become genuine vacancies (removed from the structure).

    Returns a new Structure with only the kept sites (fully occupied).
    """
    n_keep = round(target_d_total)

    # Collect (occ, site_index) for all D-type positions
    d_positions = []
    for i, site in enumerate(struct):
        for sp, occ in site.species.items():
            if elem_symbol(sp) == d_sym or str(sp).startswith(d_sym):
                d_positions.append((float(occ), i))
                break

    if not d_positions:
        log.warning(f"    No {d_sym} sites found in structure.")
        return struct

    # Sort descending by occupancy; keep top n_keep
    d_positions.sort(key=lambda x: -x[0])
    keep_indices = {idx for _, idx in d_positions[:n_keep]}
    drop_indices = {idx for _, idx in d_positions[n_keep:]}

    if drop_indices:
        log.info(f"    Dropping {len(drop_indices)} {d_sym} vacancies "
                 f"(keeping {n_keep}/{len(d_positions)}, "
                 f"target {target_d_total:.1f})")

    # Build new structure: keep all non-D sites + selected D sites.
    # Use Species objects with oxi_state=0 to preserve element labels
    # (including deuterium 'D', which pymatgen would otherwise collapse to H).
    new_species = []
    new_coords  = []
    for i, site in enumerate(struct):
        is_d_site = any(str(sp).startswith(d_sym) for sp in site.species)
        if not is_d_site:
            # Y, Fe etc. — always keep as pure ordered element
            sp_clean = max(site.species.items(), key=lambda x: float(x[1]))[0]
            new_species.append(Species(elem_symbol(sp_clean), 0))
            new_coords.append(site.frac_coords)
        elif i in keep_indices:
            # Occupied D/H site → fully occupied
            new_species.append(Species(d_sym, 0))
            new_coords.append(site.frac_coords)
        # else: vacancy — skip entirely

    return Structure(struct.lattice, new_species, new_coords)


def process_yfe2_variant(formula: str, template: Structure) -> dict:
    """
    Generate one ordered CIF for the YFe2(H/D)4.2 family.

    YFe2D4.2:         D vacancy disorder resolved by keeping the 34 highest-
                       occupancy D positions (target 4.2 × 8 = 33.6 → 34 D).
    YFe2H4.2:         Same logic with H instead of D.
    Y0.5Tb0.5Fe2D4.2: D sites resolved first, then Y sites set to {Y:0.5, Tb:0.5}
                       and ODST used to place Tb optimally.

    Structure has ~58 atoms → goes to cif_>50/.
    """
    log.info(f"  → {formula}")
    res = {"formula": formula, "family": "YFe2(H/D)4.2",
           "prototype": "monoclinic", "status": "", "method": "", "n_atoms": "",
           "sg_number": "", "sg_symbol": "", "stoich_error_pct": "",
           "cif_path": "", "notes": ""}

    working = template.copy()
    d_sym   = "H" if formula == "YFe2H4.2" else "D"
    n_form  = 8    # Z = 8 formula units in this cell
    target_d = 4.2 * n_form   # 33.6

    notes = []
    method = "manual_D_vacancy_resolution"

    # --- Step 1: resolve D/H vacancy disorder ---
    ordered = _resolve_d_vacancies(working, target_d, d_sym="D")

    # For YFe2H4.2: relabel D → H in the ordered structure
    if formula == "YFe2H4.2":
        new_species = []
        new_coords  = []
        for site in ordered:
            if site.species_string == "D":
                new_species.append("H")
            else:
                new_species.append(site.species_string)
            new_coords.append(site.frac_coords)
        ordered = Structure(ordered.lattice, new_species, new_coords)
        notes.append("D→H relabelling")

    # --- Step 2: for Y0.5Tb0.5, mix Y and Tb sites via ODST ---
    if formula == "Y0.5Tb0.5Fe2D4.2":
        # Set Y sites to 50/50 {Y: 0.5, Tb: 0.5}
        for i, site in enumerate(ordered):
            if any(elem_symbol(sp) == "Y" for sp in site.species):
                ordered.replace(i, {Species("Y", 0): 0.5,
                                    Species("Tb", 0): 0.5})
        mixed_after, method2 = odst_ordered(ordered)
        ordered = mixed_after
        method  = f"manual_D_vacancy + {method2}_for_YTb"
        notes.append("Y sites 50/50 Y/Tb via ODST")

    # --- Validate ---
    sg_num, sg_sym = report_sg(ordered)
    n = len(ordered)

    # Stoich error on D/H content
    try:
        d_actual = float(ordered.composition.get(Element(d_sym), 0))
        y_like   = float(ordered.composition.get(Element("Y"), 0)) + \
                   float(ordered.composition.get(Element("Tb"), 0))
        d_per_f  = d_actual / y_like if y_like > 0 else 0.0
        err_pct  = abs(d_per_f - 4.2) / (n / n_form) * 100 if n > 0 else 0.0
    except Exception:
        d_per_f = 0.0; err_pct = 0.0

    cif_path = os.path.join(OUT_DIR, safe_cif_name(formula))
    write_cif(ordered, cif_path)
    log.info(f"    {n} atoms | SG {sg_num} {sg_sym} | "
             f"{d_sym}/YFe2={d_per_f:.2f} (target 4.2) | "
             f"err={err_pct:.2f}% → {os.path.basename(cif_path)}")

    status = "OK" if n <= MAX_ATOMS_LARGE and ordered.is_ordered else "WARN"
    if n > MAX_ATOMS_LARGE:
        notes.append(f"WARNING: {n} atoms > {MAX_ATOMS_LARGE} limit")
        status = "WARN_SIZE"

    res.update(
        status           = status,
        method           = method,
        n_atoms          = n,
        sg_number        = sg_num,
        sg_symbol        = sg_sym,
        stoich_error_pct = f"{err_pct:.2f}",
        cif_path         = cif_path,
        notes            = "; ".join(notes),
    )
    return res


# ═════════════════════════════════ MAIN ═════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    # ── Load templates ───────────────────────────────────────────────────────
    log.info("Loading ICSD templates...")
    try:
        gd5_template = Structure.from_file(GD5_TEMPLATE)
        sg, sym = report_sg(gd5_template)
        log.info(f"  Gd5 template: {len(gd5_template)} atoms, SG {sg} {sym}")
    except Exception as e:
        log.error(f"Cannot load Gd5 template: {e}")
        gd5_template = None

    try:
        yfe2_template = Structure.from_file(YFE2D_TEMPLATE)
        log.info(f"  YFe2D template: {len(yfe2_template)} atoms (disordered)")
    except Exception as e:
        log.error(f"Cannot load YFe2D4.2 template: {e}")
        yfe2_template = None

    # ── Gd5(Si,Ge,X)4 variants ──────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("Gd5(Si,Ge,X)4 — monoclinic P1121/a (SG 14)")
    log.info("="*60)

    seen_comps = set()
    if gd5_template is not None:
        for formula, proto in GD5_TARGETS.items():
            try:
                # Deduplicate by reduced composition
                ck = frozenset((str(el), round(float(v), 3))
                               for el, v in Composition(formula).items())
            except Exception:
                ck = formula
            if ck in seen_comps:
                log.info(f"  Skipping {formula} (duplicate composition)")
                continue
            seen_comps.add(ck)
            res = process_gd5_variant(formula, gd5_template)
            res["prototype"] = proto
            results.append(res)

    # ── YFe2(H/D)4.2 variants ───────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("YFe2(H/D)4.2 — monoclinic P1c1 (SG 7)")
    log.info("="*60)

    if yfe2_template is not None:
        for formula, proto in YFE2_TARGETS.items():
            res = process_yfe2_variant(formula, yfe2_template)
            res["prototype"] = proto
            results.append(res)

    # ── Write log ────────────────────────────────────────────────────────────
    log_cols = ["formula", "family", "prototype", "status", "method",
                "n_atoms", "sg_number", "sg_symbol",
                "stoich_error_pct", "cif_path", "notes"]
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=log_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    log.info(f"\nLog → {LOG_CSV}")

    # ── Summary ──────────────────────────────────────────────────────────────
    ok   = sum(1 for r in results if r["status"] == "OK")
    warn = sum(1 for r in results if "WARN" in str(r["status"]))
    fail = sum(1 for r in results if r["status"] == "FAILED")
    log.info("─"*60)
    log.info(f"  Processed : {len(results)}  OK:{ok}  WARN:{warn}  FAIL:{fail}")
    log.info(f"  Review CIFs in: {OUT_DIR}")
    log.info("\n  NEXT STEPS:")
    log.info("  1. Review CIFs in icsd_derived_cifs/")
    log.info("  2. For Gd5 variants: copy to cif_files/ "
             "(replacing existing Pnma CIFs) and clean_cifs/")
    log.info("  3. For YFe2 variants: copy to cif_>50/")
    log.info("  4. Update magnetocaloric_materials_with_wyckoff.csv")
    log.info("  5. Run step1_generate_sqs_v2.py")
    log.info("\n  NOTE: Gd6Co2Si3 (Ce6Ni2Si3) still needs a structural paper/CIF.")


if __name__ == "__main__":
    main()
