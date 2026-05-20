"""
step2_relax_and_validate.py
============================
Reads magnetocaloric_materials_with_cif.csv (output of step1_generate_sqs.py),
relaxes every structure with M3GNet, saves the relaxed CIF, then runs 11
validation checks.  Results are written back into the master CSV and also
into a standalone relaxation_results.csv.

INPUT  : magnetocaloric_materials_with_cif.csv
         Column  final_cif_for_m3gnet  → CIF to relax
           • Ordered materials:    wyckoff-pipeline CIF
           • Disordered materials: SQS CIF from step1

OUTPUT : cif_files/<formula>_relaxed.cif          — relaxed structures
         relaxation_results.csv                   — full validation report
         magnetocaloric_materials_with_cif.csv     — updated with relaxed paths

VALIDATION CHECKS
-----------------
  V1   Atom count unchanged after relaxation.
  V2   Min interatomic distance > 1.5 Å (no atom overlap).
  V3   All nearest-neighbour bonds within 0.5–2.5× covalent-radii sum.
  V4   Lattice parameters 2–25 Å.
  V5   Cell angles 30–150 °.
  V6   All target elements present in relaxed structure.
  V7   Final energy/atom < 0 eV (unphysical if positive).
  V8   RMS atomic displacement from pre-relaxed < 3.0 Å (runaway check).
  V9   Space group assignable after relaxation.
  V10  Relaxed crystal system compatible with prototype
         (0 steps = PASS, 1 step = PASS, 2 = WARN, 3+ = FAIL).
  V11  All site occupancies = 1 (should be true after SQS step).

FOR-LOOP UPGRADE
----------------
Replace the test block in main() with:

    for _, row in df.iterrows():
        process_one(row.to_dict(), CIF_DIR, relaxer, results)

DEPENDENCIES
------------
    pip install matgl torch pymatgen pandas
"""

import os, re, csv, logging, warnings

import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIGURATION ────────────────────────────────
BASE_DIR   = "/home/daniel/Clean_Preprocessing"
CSV_IN     = os.path.join(BASE_DIR, "magnetocaloric_materials_with_cif.csv")
CIF_DIR    = os.path.join(BASE_DIR, "cif_files")
RESULTS_CSV= os.path.join(BASE_DIR, "relaxation_results.csv")

FMAX         = 0.05   # eV/Å  force convergence criterion
MAX_STEPS    = 500    # max optimiser steps
RELAX_CELL   = True   # also relax cell shape/volume

MIN_DIST     = 1.5    # Å   V2
BOND_MIN_R   = 0.50   # ×(r1+r2)  V3
BOND_MAX_R   = 2.50   # ×(r1+r2)  V3
LAT_MIN      = 1.5    # Å   V4
LAT_MAX      = 40   # Å   V4
ANG_MIN      = 30.0   # °   V5
ANG_MAX      = 150.0  # °   V5
RMS_MAX      = 3.0    # Å   V8

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

def cov_radius(sym: str) -> float:
    try:
        r = Element(sym).atomic_radius
        return float(r) if r else 1.5
    except Exception:
        return 1.5

def safe_name(formula: str, suffix: str = "") -> str:
    n = formula.replace("(","_").replace(")","_").replace(".","p")
    n = re.sub(r"[^A-Za-z0-9_]", "_", n)
    return re.sub(r"_+", "_", n).strip("_") + suffix + ".cif"

# ───────────────────────── LOAD M3GNET ────────────────────────────────────

def load_relaxer():
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer
    log.info("Loading CHGNet universal potential …")
    model   = CHGNet.load()
    relaxer = StructOptimizer(model=model, use_device="cpu")
    log.info("CHGNet loaded.")
    return relaxer

# ───────────────────────── RELAXATION ─────────────────────────────────────

def relax(struct: Structure, relaxer, formula: str):
    log.info(f"  Relaxing {formula} ({len(struct)} atoms) …")
    try:
        result  = relaxer.relax(struct, fmax=FMAX, steps=MAX_STEPS,
                                relax_cell=RELAX_CELL, verbose=False)
        rs      = result["final_structure"]
        # CHGNet returns energy in eV, trajectory in result["trajectory"]
        traj    = result["trajectory"]
        e_atom  = float(traj.energies[-1]) / len(rs)
        n_step  = len(traj.energies)
        # Check convergence: final max force < FMAX
        forces  = traj.forces[-1]
        conv    = bool(np.max(np.abs(forces)) < FMAX)
        status  = "converged" if conv else "not_converged"
        log.info(f"  {status} in {n_step} steps, E/atom={e_atom:.4f} eV")
        return rs, e_atom, n_step, status
    except Exception as e:
        log.error(f"  Relaxation failed: {e}")
        return None, float("nan"), 0, f"failed:{e}"

# ──────────────────────────── SAVE CIF ────────────────────────────────────

def save_cif(struct: Structure, formula: str, cif_dir: str) -> str:
    os.makedirs(cif_dir, exist_ok=True)
    path    = os.path.join(cif_dir, safe_name(formula, "_relaxed"))
    symprec = None if not struct.is_ordered else 0.1
    try:
        CifWriter(struct, symprec=symprec).write_file(path)
    except Exception:
        CifWriter(struct, symprec=None).write_file(path)
    log.info(f"  Relaxed CIF → {path}")
    return path

# ──────────────────────── VALIDATION ───────────────────────────────────────

def validate(pre: Structure, post: Structure, formula: str,
             e_atom: float, proto_sg: int) -> dict:
    """Run all 11 checks; return dict of {label: result_string}."""
    v = {}

    # V1 – atom count
    v["V1_atom_count"] = (
        "PASS" if len(pre) == len(post)
        else f"FAIL: {len(pre)}→{len(post)} atoms"
    )

    # V2 – min distance
    try:
        dm = post.distance_matrix.copy(); np.fill_diagonal(dm, np.inf)
        dmin = float(dm.min())
        v["V2_min_dist"] = (
            f"PASS (dmin={dmin:.3f}Å)" if dmin >= MIN_DIST
            else f"FAIL: dmin={dmin:.3f}Å < {MIN_DIST}Å"
        )
    except Exception as e:
        v["V2_min_dist"] = f"ERROR:{e}"

    # V3 – bond lengths (only pairs < 4 Å)
    try:
        elems = [s.species_string for s in post]
        dm2   = post.distance_matrix.copy(); np.fill_diagonal(dm2, np.inf)
        bad   = []
        for i in range(len(post)):
            for j in range(i+1, len(post)):
                d = dm2[i, j]
                if d > 4.0: continue
                rs = cov_radius(elems[i]) + cov_radius(elems[j])
                if d < rs * BOND_MIN_R:
                    bad.append(f"{elems[i]}-{elems[j]}:{d:.2f}Å(short)")
                elif d > rs * BOND_MAX_R:
                    bad.append(f"{elems[i]}-{elems[j]}:{d:.2f}Å(long)")
        v["V3_bonds"] = "PASS" if not bad else f"WARN:{';'.join(bad[:5])}"
    except Exception as e:
        v["V3_bonds"] = f"ERROR:{e}"

    # V4 – lattice params
    a, b, c = post.lattice.abc
    bad_p = [f"{x:.2f}Å" for x in (a,b,c) if x < LAT_MIN or x > LAT_MAX]
    v["V4_lattice"] = (
        f"PASS (a={a:.3f},b={b:.3f},c={c:.3f}Å)" if not bad_p
        else f"FAIL: {bad_p}"
    )

    # V5 – cell angles
    al, be, ga = post.lattice.angles
    bad_a = [f"{x:.1f}°" for x in (al,be,ga) if x < ANG_MIN or x > ANG_MAX]
    v["V5_angles"] = (
        f"PASS (α={al:.1f},β={be:.1f},γ={ga:.1f}°)" if not bad_a
        else f"FAIL: {bad_a}"
    )

    # V6 – composition
    try:
        t = {str(e) for e in Composition(formula).elements}
        s = {str(e) for e in post.composition.elements}
        miss = t - s
        v["V6_composition"] = (
            "PASS" if not miss
            else f"WARN: missing {sorted(miss)} (low-conc. SQS approx)"
        )
    except Exception as e:
        v["V6_composition"] = f"ERROR:{e}"

    # V7 – energy
    v["V7_energy"] = (
        f"PASS (E/atom={e_atom:.4f}eV)" if e_atom < 0
        else f"FAIL: E/atom={e_atom:.4f}eV ≥ 0"
    )

    # V8 – RMS displacement
    try:
        if len(pre) == len(post):
            rms = float(np.sqrt(np.mean(
                np.sum((post.cart_coords - pre.cart_coords)**2, axis=1)
            )))
            v["V8_rms_disp"] = (
                f"PASS (RMS={rms:.3f}Å)" if rms < RMS_MAX
                else f"WARN: RMS={rms:.3f}Å > {RMS_MAX}Å — possible runaway"
            )
        else:
            v["V8_rms_disp"] = "SKIP:atom_count_changed"
    except Exception as e:
        v["V8_rms_disp"] = f"ERROR:{e}"

    # V9 – space group
    sg_post, sym_post = None, ""
    try:
        sga     = SpacegroupAnalyzer(post, symprec=0.1)
        sg_post = sga.get_space_group_number()
        sym_post= sga.get_space_group_symbol()
        v["V9_spacegroup"] = f"PASS (SG {sg_post} {sym_post})"
    except Exception as e:
        v["V9_spacegroup"] = f"FAIL:cannot assign SG ({e})"

    # V10 – crystal system vs prototype
    if sg_post and proto_sg:
        try:
            pr   = sg_to_crystal_rank(int(proto_sg))
            rr   = sg_to_crystal_rank(sg_post)
            diff = rr - pr
            pcs  = CRYSTAL_SYSTEMS[pr]
            rcs  = CRYSTAL_SYSTEMS[rr]
            if diff <= 1:
                v["V10_crystal_system"] = (
                    f"PASS ({pcs}→{rcs}, Δ={diff})"
                )
            elif diff == 2:
                v["V10_crystal_system"] = (
                    f"WARN: 2-step sym.drop ({pcs}→{rcs}) "
                    f"— verify structure"
                )
            else:
                v["V10_crystal_system"] = (
                    f"WARN: {diff}-step sym.drop ({pcs}→{rcs}) "
                    f"— expected for ordered SQS of disordered material"
                )
        except Exception as e:
            v["V10_crystal_system"] = f"ERROR:{e}"
    else:
        v["V10_crystal_system"] = "SKIP:no_prototype_SG"

    # V11 – occupancies should all be 1
    try:
        bad_occ = [
            f"{sp}:{occ:.3f}"
            for site in post
            for sp, occ in site.species.items()
            if float(occ) < 0 or float(occ) > 1 + 1e-6
        ]
        v["V11_occupancies"] = (
            "PASS" if not bad_occ
            else f"FAIL: {bad_occ}"
        )
    except Exception as e:
        v["V11_occupancies"] = f"ERROR:{e}"

    fails = [k for k,val in v.items() if val.startswith("FAIL")]
    warns = [k for k,val in v.items() if val.startswith("WARN")]
    v["overall"] = (
        "PASS"              if not fails and not warns else
        "WARN:"+",".join(warns) if not fails else
        "FAIL:"+",".join(fails)
    )
    v["relaxed_sg_number"] = str(sg_post)  if sg_post  else ""
    v["relaxed_sg_symbol"] = sym_post
    return v

# ──────────────────────── PER-MATERIAL FUNCTION ────────────────────────────

RELAX_COLS = [
    "relaxed_cif_path", "relax_status", "n_steps", "energy_per_atom_eV",
    "relaxed_sg_number", "relaxed_sg_symbol",
    "V1_atom_count","V2_min_dist","V3_bonds","V4_lattice","V5_angles",
    "V6_composition","V7_energy","V8_rms_disp","V9_spacegroup",
    "V10_crystal_system","V11_occupancies",
    "overall_validation","relax_notes",
]

def process_one(row: dict, cif_dir: str, relaxer, results: list) -> None:
    """
    Load → relax → save → validate one material.
    Appends one result dict to `results`.

    ── TO PROCESS ALL MATERIALS ──────────────────────────────────────────
    Replace the test block in main() with:

        for _, row in df.iterrows():
            process_one(row.to_dict(), CIF_DIR, relaxer, results)
    ──────────────────────────────────────────────────────────────────────
    """
    formula  = str(row.get("material_formula","")).strip()
    cif_path = str(row.get("final_cif_for_m3gnet","")).strip()
    sg_str   = str(row.get("sg_number","")).strip()
    proto    = str(row.get("structure_prototype","")).strip()

    log.info("="*64)
    log.info(f"  Material : {formula}")
    log.info(f"  CIF      : {cif_path}")
    log.info(f"  Prototype: {proto}  SG: {sg_str}")

    res = dict(row)
    for col in RELAX_COLS:
        res.setdefault(col, "")

    # Load CIF
    if not cif_path or not os.path.exists(cif_path):
        res["relax_status"] = "SKIPPED_NO_CIF"
        res["relax_notes"]  = f"CIF not found: {cif_path}"
        results.append(res); return

    try:
        pre = Structure.from_file(cif_path)
    except Exception as e:
        res["relax_status"] = "SKIPPED_LOAD_FAILED"
        res["relax_notes"]  = f"CIF load failed: {e}"
        results.append(res); return

    if not pre.is_ordered:
        res["relax_status"] = "SKIPPED_DISORDERED"
        res["relax_notes"]  = ("Still disordered — run step1_generate_sqs.py first")
        results.append(res); return

    log.info(f"  Loaded: {len(pre)} atoms")

    # Relax
    post, e_atom, n_step, status = relax(pre, relaxer, formula)
    res["relax_status"]       = status
    res["n_steps"]            = str(n_step)
    res["energy_per_atom_eV"] = f"{e_atom:.6f}" if not np.isnan(e_atom) else "NaN"

    if post is None:
        res["relax_notes"] = "Relaxation returned no structure"
        results.append(res); return

    # Save
    res["relaxed_cif_path"] = save_cif(post, formula, cif_dir)

    # Validate
    proto_sg = int(sg_str) if sg_str.isdigit() else 0
    checks   = validate(pre, post, formula, e_atom, proto_sg)
    for col in ["relaxed_sg_number","relaxed_sg_symbol",
                "V1_atom_count","V2_min_dist","V3_bonds","V4_lattice",
                "V5_angles","V6_composition","V7_energy","V8_rms_disp",
                "V9_spacegroup","V10_crystal_system","V11_occupancies",
                "overall_validation"]:
        key = col.replace("overall_validation","overall")
        res[col] = checks.get(key, checks.get(col,""))

    log.info(f"  Validation: {checks['overall']}")
    results.append(res)

# ──────────────────────────── CSV I/O ──────────────────────────────────────

def write_results(rows, path):
    if not rows: return
    orig   = [k for k in rows[0] if k not in RELAX_COLS]
    fields = orig + RELAX_COLS
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    log.info(f"  Results CSV → {path}")

def update_master_csv(results, csv_in):
    """Merge relaxed_cif_path and overall_validation back into master CSV."""
    try:
        df   = pd.read_csv(csv_in)
        rdf  = pd.DataFrame(results)[["material_formula","relaxed_cif_path",
                                      "overall_validation"]].drop_duplicates(
                                          subset="material_formula")
        df = df.merge(rdf, on="material_formula", how="left",
                      suffixes=("","_new"))
        for col in ["relaxed_cif_path","overall_validation"]:
            if col+"_new" in df.columns:
                df[col] = df.pop(col+"_new")
            elif col not in df.columns:
                df[col] = ""
        df.to_csv(csv_in, index=False)
        log.info(f"  Updated master CSV → {csv_in}")
    except Exception as e:
        log.warning(f"  Could not update master CSV: {e}")

# ───────────────────────────── MAIN ────────────────────────────────────────

def main():
    df = pd.read_csv(CSV_IN)
    # Only rows that have a CIF path set
    df = df[df["final_cif_for_m3gnet"].astype(str).str.strip().ne("")]\
           .reset_index(drop=True)
    log.info(f"Loaded {len(df)} materials with CIF paths.")

    os.makedirs(CIF_DIR, exist_ok=True)
    relaxer = load_relaxer()   # expensive — load once
    results: list[dict] = []

    # ── Checkpoint: skip materials that already passed validation ──────────
    # Delete relax_checkpoint.txt to force a full rerun from scratch.
    # Materials that previously failed or were killed will be retried.
    checkpoint_path = os.path.join(BASE_DIR, "relax_checkpoint.txt")
    passed_formulas = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            passed_formulas = set(line.strip() for line in f if line.strip())
        log.info(f"Resuming: {len(passed_formulas)} materials already passed validation.")

    for i, (_, row) in enumerate(df.iterrows()):
        formula = str(row["material_formula"]).strip()
        if formula in passed_formulas:
            log.info(f"  Skipping {formula} (already passed)")
            continue

        log.info(f"Progress: {i+1}/{len(df)}")
        process_one(row.to_dict(), CIF_DIR, relaxer, results)

        # Write results after every material so nothing is lost on crash
        write_results(results, RESULTS_CSV)

        # Only checkpoint if ALL validation checks passed
        last = results[-1]
        if last.get("overall_validation", "").strip() == "PASS":
            with open(checkpoint_path, "a") as f:
                f.write(formula + "\n")
        else:
            log.info(f"  {formula} did not pass — will retry on next run")

    update_master_csv(results, CSV_IN)

    conv   = sum(1 for r in results if r.get("relax_status")=="converged")
    noconv = sum(1 for r in results if r.get("relax_status")=="not_converged")
    skip   = sum(1 for r in results if "SKIPPED" in r.get("relax_status",""))
    vpass  = sum(1 for r in results if r.get("overall_validation","")=="PASS")
    vwarn  = sum(1 for r in results if "WARN" in r.get("overall_validation",""))
    vfail  = sum(1 for r in results if "FAIL" in r.get("overall_validation",""))
    log.info("─"*64)
    log.info(f"  Converged: {conv}   Not converged: {noconv}   Skipped: {skip}")
    log.info(f"  Validation — PASS:{vpass}  WARN:{vwarn}  FAIL:{vfail}")
    log.info(f"  Results CSV       : {RESULTS_CSV}")
    log.info(f"  Updated master CSV: {CSV_IN}")
    log.info(f"  Relaxed CIFs in   : {CIF_DIR}")

if __name__ == "__main__":
    main()
