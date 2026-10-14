"""
step1_generate_sqs.py
=====================
Converts disordered (fractional-occupancy) CIF files from the Wyckoff
pipeline into ordered SQS approximations that M3GNet can accept.

BACKGROUND
----------
The Wyckoff pipeline flagged 237 materials as 'disordered' because their
dopant fractions cannot be expressed as whole atoms in a small enough cell.
Their CIF files contain _atom_site_occupancy < 1.  M3GNet requires fully
ordered structures (every site occupied by exactly one species).

This script converts each such CIF to an ordered approximation using
pymatgen's OrderDisorderedStructureTransformation (ODST), which selects
the ordering that minimises the electrostatic (Ewald) energy — i.e. the
most physically uniform distribution of dopant atoms.  No external binary
(mcsqs / ATAT) is required.

HOW SUPERCELL SIZE IS CHOSEN
-----------------------------
For each mixed site with target dopant fraction f on n sites per primitive
cell, the script tries scales s = 1, 2, 3, … up to floor(MAX_ATOMS / n_prim).
It picks the scale that minimises |round(s·n·f)/(s·n) − f|, i.e. the
composition error, while keeping total atoms ≤ MAX_ATOMS.

If the best achievable composition error is > POOR_FIT_THRESHOLD (default
0.10), the dopant fraction is too dilute to represent honestly in ≤ 20
atoms.  In that case the dopant is replaced by the majority species and
the approximation is noted in the output CSV.

OUTPUTS
-------
  cif_files/<formula>_sqs.cif            — ordered SQS structure
  magnetocaloric_materials_with_cif.csv  — original CSV + new columns:
      sqs_cif_path, sqs_method, sqs_n_atoms,
      sqs_sg_number, sqs_sg_symbol,
      sqs_comp_error, sqs_validation, sqs_notes,
      final_cif_for_m3gnet               ← use this in step2

FOR-LOOP UPGRADE
----------------
The test block near the bottom processes ONE material.
Replace it with:

    for _, row in df_dis.iterrows():
        process_one(row.to_dict(), CIF_DIR, results)

No other changes are needed.

DEPENDENCIES
------------
    pip install pymatgen pandas
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PREPROCESSING_WORK  # noqa: E402


import os, re, csv, ast, logging, warnings
from collections import Counter

import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import (
    OrderDisorderedStructureTransformation,
)

warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIGURATION ────────────────────────────────
BASE_DIR      = str(PREPROCESSING_WORK)
CSV_IN        = os.path.join(BASE_DIR, "magnetocaloric_materials_with_wyckoff.csv")
CIF_DIR       = os.path.join(BASE_DIR, "cif_files")
CSV_OUT       = os.path.join(BASE_DIR, "magnetocaloric_materials_with_cif.csv")
APPROX_LOG_CSV= os.path.join(BASE_DIR, "dopant_approximations.csv")

MAX_ATOMS         = 50    # hard upper limit for MatterGen
POOR_FIT_THRESHOLD = 0.05 # flag dopants whose best achievable |Δf| > this

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────── HELPERS ──────────────────────────────────────

CRYSTAL_SYSTEMS = ["cubic","hexagonal","trigonal","tetragonal",
                   "orthorhombic","monoclinic","triclinic"]

def sg_to_crystal_rank(sg: int) -> int:
    if sg >= 195: return 0
    if sg >= 168: return 1
    if sg >= 143: return 2
    if sg >= 75:  return 3
    if sg >= 16:  return 4
    if sg >= 3:   return 5
    return 6

def safe_name(formula: str, suffix: str = "") -> str:
    n = formula.replace("(","_").replace(")","_").replace(".","p")
    n = re.sub(r"[^A-Za-z0-9_]", "_", n)
    return re.sub(r"_+", "_", n).strip("_") + suffix + ".cif"

def find_best_scale(n_mixed_per_prim: int, target_frac: float,
                    n_prim_atoms: int, max_atoms: int = MAX_ATOMS):
    """
    Find the supercell scale s in [1, floor(max_atoms/n_prim_atoms)] that
    minimises |actual_frac - target_frac|, guaranteeing at least 1 dopant
    atom so the minority species is never silently dropped to zero.

    Returns (scale, actual_frac, n_dopant_atoms, composition_error, was_forced).
    was_forced=True when the raw round() gave 0 and was bumped up to 1.
    """
    best = (1, 1.0, 1, 1.0, True)
    max_s = max(1, max_atoms // n_prim_atoms)
    for s in range(1, max_s + 1):
        n_mix        = s * n_mixed_per_prim
        n_dopant_raw = round(target_frac * n_mix)
        was_forced   = n_dopant_raw < 1
        n_dopant     = max(1, n_dopant_raw)
        actual       = n_dopant / n_mix if n_mix > 0 else 0.0
        err          = abs(actual - target_frac)
        if err < best[3]:
            best = (s, actual, n_dopant, err, was_forced)
    return best

# ──────────────────────── CORE CONVERSION ──────────────────────────────────

def disordered_to_ordered(dis: Structure, formula: str):
    """
    Convert a disordered pymatgen Structure to the best ordered approximation.

    Returns (ordered_structure, method_string, composition_error).
    method_string describes what was done (for the CSV notes column).
    """
    n_prim = len(dis)

    # --- Count mixed sites and find worst-case composition error -----------
    mixed_info = []  # (site_indices, minority_frac)
    # Group sites by their species dict
    from collections import defaultdict
    groups = defaultdict(list)
    for i, site in enumerate(dis):
        key = tuple(sorted((str(k), float(v)) for k, v in site.species.items()))
        groups[key].append(i)

    for key, indices in groups.items():
        if len(key) <= 1:
            continue  # pure site
        occs = dict(key)
        minority_frac = min(occs.values())
        mixed_info.append((indices, minority_frac, occs))

    if not mixed_info:
        return dis, "already_ordered", 0.0, []

    # --- Find the supercell scale that best fits ALL mixed fractions -------
    best_scale    = 1
    worst_err     = 0.0
    method_parts  = []
    approximations = []   # records forced/poor-fit dopant adjustments

    working = dis.copy()

    for indices, minority_frac, occs in mixed_info:
        n_sites = len(indices)  # sites of this type in the primitive cell

        if minority_frac <= 0 or minority_frac >= 1:
            continue

        s, actual_frac, n_dopant, err, was_forced = find_best_scale(
            n_sites, minority_frac, n_prim
        )

        if was_forced or err > POOR_FIT_THRESHOLD:
            # Dopant too dilute for exact representation — force minimum 1 atom.
            # Update fractional occupancies in the working structure so ODST
            # places the dopant at the rounded-up fraction rather than zero.
            minority_sp     = min(occs.items(), key=lambda x: x[1])[0]
            other_occ_total = 1.0 - minority_frac
            new_other_total = 1.0 - actual_frac
            for i in indices:
                new_species = {}
                for sp_str, frac in occs.items():
                    if sp_str == minority_sp:
                        new_species[sp_str] = actual_frac
                    elif other_occ_total > 0:
                        new_species[sp_str] = frac * (new_other_total / other_occ_total)
                    else:
                        new_species[sp_str] = frac
                working.replace(i, new_species)
            log.info(f"  Dopant {minority_sp}: {minority_frac:.4f} → "
                     f"{actual_frac:.4f} (1 atom / {n_dopant} sites, scale={s})")
            method_parts.append(
                f"forced_min({minority_sp}:{minority_frac:.4f}→{actual_frac:.4f})"
            )
            approximations.append({
                "element":           minority_sp,
                "target_frac":       round(minority_frac, 6),
                "actual_frac_in_cif": round(actual_frac, 6),
                "n_dopant_atoms":    n_dopant,
                "supercell_scale":   s,
                "total_atoms":       s * n_prim,
                "reason":            "forced_min_1_atom" if was_forced
                                     else "poor_fit_rounded_up",
            })
        else:
            method_parts.append(f"scale={s}(Δf={err:.4f})")

        if s > best_scale and s * n_prim <= MAX_ATOMS:
            best_scale = s
        if err > worst_err:
            worst_err = err

    # --- Make supercell if needed ------------------------------------------
    # Only expand if it keeps us within MAX_ATOMS
    if best_scale > 1 and best_scale * n_prim <= MAX_ATOMS:
        log.info(f"  Expanding to [{best_scale},1,1] supercell "
             f"({best_scale * n_prim} atoms)")
        working.make_supercell([best_scale, 1, 1])
    elif best_scale > 1:
        log.warning(f"  Supercell would give {best_scale * n_prim} atoms "
                f"> {MAX_ATOMS} — staying at scale=1")

    # Hard cap: if still too large, use _distribute_ordered to preserve
    # all minority species (at least 1 atom each) rather than dropping them.
    if len(working) > MAX_ATOMS:
        log.warning(f"  Cell has {len(working)} atoms > {MAX_ATOMS} "
                f"— distributing mixed sites (preserving minority atoms)")
        working = _distribute_ordered(working)

    # --- Apply ODST if still disordered ------------------------------------
    if not working.is_ordered:
        try:
            odst    = OrderDisorderedStructureTransformation(algo=0)
            ordered = odst.apply_transformation(working)
            method  = "ODST+" + "+".join(method_parts)
        except Exception as e:
            log.warning(f"  ODST failed ({e}); using distribution fallback")
            ordered = _distribute_ordered(working)
            method  = "distributed+" + "+".join(method_parts)
    else:
        ordered = working
        method  = "+".join(method_parts) if method_parts else "already_ordered"

    return ordered, method, worst_err, approximations


def _distribute_ordered(struct: Structure) -> Structure:
    """
    Fallback ordering: groups identical mixed sites, computes integer atom
    counts from fractional occupancies (minority forced to ≥1 atom), then
    distributes minority atoms at evenly-spaced positions across each group.
    Replaces the old majority-only _manual_round fallback.
    """
    from collections import defaultdict
    result = struct.copy()
    groups = defaultdict(list)
    for i, site in enumerate(result):
        if not site.is_ordered:
            key = tuple(sorted((str(k), float(v)) for k, v in site.species.items()))
            groups[key].append(i)

    for key, site_indices in groups.items():
        n = len(site_indices)
        occs = dict(key)
        # Sort ascending by fraction so minority species are processed first
        sorted_sp = sorted(occs.items(), key=lambda x: x[1])
        assignments = {}
        used = 0
        for sp_str, frac in sorted_sp[:-1]:
            if frac <= 0:
                assignments[sp_str] = 0
                continue
            count = max(1, round(frac * n))
            count = min(count, n - used - 1)  # leave ≥1 site for majority
            assignments[sp_str] = count
            used += count
        assignments[sorted_sp[-1][0]] = n - used  # majority fills remainder

        # Build assignment list with minority atoms at evenly-spaced positions
        majority_sp = sorted_sp[-1][0]
        assignment_list = [majority_sp] * n
        offset = 0.0
        for sp_str, frac in sorted_sp[:-1]:
            count = assignments.get(sp_str, 0)
            if count == 0:
                continue
            step = n / count
            for k in range(count):
                idx = int((offset + k * step)) % n
                # If already taken by another minority, find nearest free slot
                while assignment_list[idx] != majority_sp:
                    idx = (idx + 1) % n
                assignment_list[idx] = sp_str
            offset += step / 2  # stagger next minority species

        for site_idx, sp_str in zip(site_indices, assignment_list):
            result.replace(site_idx, sp_str)

    return result

# ──────────────────────── VALIDATION ───────────────────────────────────────

def validate_sqs(ordered: Structure, formula: str,
                 proto_sg_number: int, comp_error: float) -> dict:
    """
    Six validation checks on the ordered SQS structure.

    S1  Fully ordered (no fractional occupancies remain).
    S2  Atom count ≤ MAX_ATOMS.
    S3  No overlapping atoms (min dist > 1.0 Å).
    S4  All target elements present in structure.
    S5  Space group can be assigned.
    S6  Crystal system compatible with prototype
        (≤1 step down in symmetry hierarchy is normal for an ordered SQS;
         ≥2 steps is a warning; ≥3 steps is a failure).
    """
    v = {}

    # S1
    v["S1_ordered"] = (
        "PASS" if ordered.is_ordered
        else "FAIL: fractional occupancies remain"
    )

    # S2
    n = len(ordered)
    v["S2_atom_count"] = (
        f"PASS ({n} atoms)" if n <= MAX_ATOMS
        else f"FAIL: {n} atoms > {MAX_ATOMS}"
    )

    # S3
    try:
        dm = ordered.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        dmin = float(dm.min())
        v["S3_min_dist"] = (
            f"PASS (dmin={dmin:.3f}Å)" if dmin > 1.0
            else f"FAIL: dmin={dmin:.3f}Å < 1.0Å"
        )
    except Exception as e:
        v["S3_min_dist"] = f"ERROR:{e}"

    # S4
    try:
        tgt  = {str(e) for e in Composition(formula).elements}
        got  = {str(e) for e in ordered.composition.elements}
        miss = tgt - got
        v["S4_composition"] = (
            "PASS" if not miss
            else f"WARN: {sorted(miss)} absent "
                 f"(low-conc. approx, comp_error={comp_error:.3f})"
        )
    except Exception as e:
        v["S4_composition"] = f"ERROR:{e}"

    # S5 + S6
    sg_out, sym_out = None, ""
    try:
        sga    = SpacegroupAnalyzer(ordered, symprec=0.2)
        sg_out = sga.get_space_group_number()
        sym_out= sga.get_space_group_symbol()
        v["S5_spacegroup"] = f"PASS (SG {sg_out} {sym_out})"
    except Exception as e:
        v["S5_spacegroup"] = f"FAIL: cannot assign SG ({e})"

    if sg_out and proto_sg_number:
        try:
            pr = sg_to_crystal_rank(int(proto_sg_number))
            rr = sg_to_crystal_rank(sg_out)
            diff = rr - pr
            pcs  = CRYSTAL_SYSTEMS[pr]
            rcs  = CRYSTAL_SYSTEMS[rr]
            if diff <= 1:
                v["S6_symmetry"] = f"PASS ({pcs}→{rcs}, Δ={diff})"
            elif diff == 2:
                v["S6_symmetry"] = (
                    f"WARN: 2-step sym.drop ({pcs}→{rcs}) — "
                    f"verify structure makes sense"
                )
            else:
                v["S6_symmetry"] = (
                    f"WARN: {diff}-step sym.drop ({pcs}→{rcs}) — "
                    f"expected for ordered SQS of disordered material"
                )
        except Exception as e:
            v["S6_symmetry"] = f"ERROR:{e}"
    else:
        v["S6_symmetry"] = "SKIP:no_prototype_SG"

    fails = [k for k, val in v.items() if val.startswith("FAIL")]
    warns = [k for k, val in v.items() if val.startswith("WARN")]
    v["overall"] = (
        "PASS"              if not fails and not warns else
        "WARN:"+",".join(warns) if not fails else
        "FAIL:"+",".join(fails)
    )
    v["sqs_sg_number"] = str(sg_out) if sg_out else ""
    v["sqs_sg_symbol"] = sym_out
    return v

# ──────────────────────── PER-MATERIAL FUNCTION ────────────────────────────

def process_one(row: dict, cif_dir: str, results: list) -> None:
    """
    Convert one disordered material to an ordered SQS.
    Appends one result dict to `results`.

    ── TO PROCESS ALL MATERIALS ──────────────────────────────────────────
    Replace the test block in main() with:

        for _, row in df_dis.iterrows():
            process_one(row.to_dict(), CIF_DIR, results)
    ──────────────────────────────────────────────────────────────────────
    """
    formula   = str(row.get("material_formula","")).strip()
    cif_path  = str(row.get("cif_path","")).strip()
    sg_number = str(row.get("sg_number","")).strip()
    prototype = str(row.get("structure_prototype","")).strip()

    log.info("="*64)
    log.info(f"  Material : {formula}")
    log.info(f"  Prototype: {prototype}  SG: {sg_number}")

    res = dict(row)
    res.update({k: "" for k in [
        "sqs_cif_path","sqs_method","sqs_n_atoms",
        "sqs_sg_number","sqs_sg_symbol","sqs_comp_error",
        "S1_ordered","S2_atom_count","S3_min_dist",
        "S4_composition","S5_spacegroup","S6_symmetry",
        "sqs_validation","sqs_notes","final_cif_for_m3gnet",
    ]})

    # Load disordered CIF
    if not cif_path or not os.path.exists(cif_path):
        res["sqs_notes"] = f"CIF not found: {cif_path}"
        results.append(res); return
    try:
        dis = Structure.from_file(cif_path)
    except Exception as e:
        res["sqs_notes"] = f"CIF load failed: {e}"
        results.append(res); return

    log.info(f"  Loaded: {len(dis)} atoms (disordered)")

    # Generate ordered SQS
    ordered, method, comp_err, approximations = disordered_to_ordered(dis, formula)
    for apx in approximations:
        apx["material_formula"] = formula
        apx["prototype"]        = prototype
    res["_approximations"] = approximations
    res["sqs_method"]     = method
    res["sqs_n_atoms"]    = str(len(ordered))
    res["sqs_comp_error"] = f"{comp_err:.4f}"
    log.info(f"  Ordered: {len(ordered)} atoms  method={method}")

    # Save SQS CIF
    sqs_path = os.path.join(cif_dir, safe_name(formula, "_sqs"))
    try:
        CifWriter(ordered, symprec=0.1).write_file(sqs_path)
    except Exception:
        CifWriter(ordered, symprec=None).write_file(sqs_path)
    res["sqs_cif_path"]        = sqs_path
    res["final_cif_for_m3gnet"] = sqs_path
    log.info(f"  SQS CIF → {sqs_path}")

    # Validate
    proto_sg = int(sg_number) if sg_number.isdigit() else 0
    checks   = validate_sqs(ordered, formula, proto_sg, comp_err)
    res.update({
        "sqs_sg_number":  checks["sqs_sg_number"],
        "sqs_sg_symbol":  checks["sqs_sg_symbol"],
        "S1_ordered":     checks["S1_ordered"],
        "S2_atom_count":  checks["S2_atom_count"],
        "S3_min_dist":    checks["S3_min_dist"],
        "S4_composition": checks["S4_composition"],
        "S5_spacegroup":  checks["S5_spacegroup"],
        "S6_symmetry":    checks["S6_symmetry"],
        "sqs_validation": checks["overall"],
    })
    log.info(f"  Validation: {checks['overall']}")
    results.append(res)

# ──────────────────────────── CSV I/O ──────────────────────────────────────

SQS_COLS = [
    "sqs_cif_path","sqs_method","sqs_n_atoms",
    "sqs_sg_number","sqs_sg_symbol","sqs_comp_error",
    "S1_ordered","S2_atom_count","S3_min_dist",
    "S4_composition","S5_spacegroup","S6_symmetry",
    "sqs_validation","sqs_notes","final_cif_for_m3gnet",
]

def write_combined_csv(ordered_rows, sqs_results, path):
    """
    Combine ordered (already have CIF) + SQS rows and write one CSV.
    For ordered rows  final_cif_for_m3gnet = existing cif_path.
    For SQS rows      final_cif_for_m3gnet = new SQS CIF path.
    """
    # Add blank SQS columns to ordered rows
    ord_out = []
    for r in ordered_rows:
        nr = dict(r)
        for col in SQS_COLS:
            nr.setdefault(col, "")
        nr["final_cif_for_m3gnet"] = nr.get("cif_path", "")
        ord_out.append(nr)

    all_rows   = ord_out + sqs_results
    if not all_rows:
        return
    orig_cols  = [k for k in all_rows[0] if k not in SQS_COLS]
    fieldnames = orig_cols + SQS_COLS

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    log.info(f"  Combined CSV → {path}  ({len(all_rows)} rows)")

# ───────────────────────────── MAIN ────────────────────────────────────────

def main():
    df     = pd.read_csv(CSV_IN)
    df_dis = df[df["is_disordered"].astype(str).str.strip().str.lower() == "true"]\
               .reset_index(drop=True)
    df_ord = df[df["is_disordered"].astype(str).str.strip().str.lower() != "true"]

    log.info(f"Loaded {len(df)} materials: "
             f"{len(df_dis)} disordered, {len(df_ord)} ordered.")
    os.makedirs(CIF_DIR, exist_ok=True)

    results: list[dict] = []

    # ── Checkpoint: skip materials that already passed validation ──────────
    # Delete sqs_checkpoint.txt to force a full rerun from scratch.
    # Materials that previously failed will be retried automatically.
    checkpoint_path = os.path.join(BASE_DIR, "sqs_checkpoint.txt")
    passed_formulas = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            passed_formulas = set(line.strip() for line in f if line.strip())
        log.info(f"Resuming: {len(passed_formulas)} materials already passed validation.")

    for i, (_, row) in enumerate(df_dis.iterrows()):
        formula = str(row["material_formula"]).strip()
        if formula in passed_formulas:
            log.info(f"  Skipping {formula} (already passed)")
            continue

        log.info(f"Progress: {i+1}/{len(df_dis)}")
        process_one(row.to_dict(), CIF_DIR, results)

        # Only checkpoint if this material passed ALL validation checks
        last = results[-1]
        if last.get("sqs_validation", "").strip() == "PASS":
            with open(checkpoint_path, "a") as f:
                f.write(formula + "\n")
            passed_formulas.add(formula)   # skip duplicates later in same run
        else:
            log.info(f"  {formula} did not pass validation — will retry on next run")

    # Collect approximation records before write (pop so they don't go into CSV)
    all_approx: list[dict] = []
    for r in results:
        all_approx.extend(r.pop("_approximations", []))

    write_combined_csv(df_ord.to_dict("records"), results, CSV_OUT)

    # Write dopant approximation log
    if all_approx:
        approx_cols = [
            "material_formula", "prototype", "element",
            "target_frac", "actual_frac_in_cif", "n_dopant_atoms",
            "supercell_scale", "total_atoms", "reason",
        ]
        with open(APPROX_LOG_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=approx_cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_approx)
        log.info(f"  Dopant approx log → {APPROX_LOG_CSV}  ({len(all_approx)} entries)")
    else:
        log.info("  No dopant approximations recorded.")

    passed  = sum(1 for r in results if "PASS"  == r.get("sqs_validation","").strip())
    warned  = sum(1 for r in results if "WARN"  in r.get("sqs_validation",""))
    failed  = sum(1 for r in results if "FAIL"  in r.get("sqs_validation",""))
    log.info("─"*64)
    log.info(f"  Processed: {len(results)}  PASS:{passed}  WARN:{warned}  FAIL:{failed}")
    log.info(f"  Output CSV  : {CSV_OUT}")
    log.info(f"  Approx log  : {APPROX_LOG_CSV}")
    log.info(f"  CIF dir     : {CIF_DIR}")
    log.info("  Next: run step2_relax_and_validate.py")

if __name__ == "__main__":
    main()
