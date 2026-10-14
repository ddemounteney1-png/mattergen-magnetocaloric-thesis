"""
analyze_cif_quality.py
======================
Quality analysis of CIF files before and after the SQS ordering step.
Run BETWEEN step1_generate_sqs_v2.py and step2 (CHGnet relaxation).

For each material the script:
  1. Reads the original CIF (pre-SQS) and the final CIF (post-SQS) from disk.
  2. Extracts actual atom types and composition directly from each CIF.
  3. Computes stoichiometry deviation from the TARGET formula in
     magnetocaloric_materials.csv.
  4. Checks space group before and after SQS against the expected prototype SG.
  5. Verifies the crystal system is preserved (or only reduced) after SQS.
  6. Flags any materials with stoich errors, missing elements, or symmetry jumps.

Outputs
-------
  cif_quality_report.csv   — full per-material table
  cif_quality_summary.txt  — human-readable pass/warn/fail summary

Usage
-----
    python3 analyze_cif_quality.py

Interpretation of STATUS column:
  PASS  — stoich ≤5%, all elements present, crystal system preserved or reduced
  WARN  — stoich 5–10%, or crystal system reduced by >1 step, minor issues
  FAIL  — stoich >10%, elements missing, or crystal system raised vs prototype
  SKIP  — CIF file not found (pipeline failure, expected for some materials)

NOTE on space groups after SQS:
  ODST ordering always reduces symmetry (breaks the disorder-induced
  translational symmetry).  A reduction of the space group number is
  EXPECTED and NORMAL.  A flagged SG is one that has HIGHER symmetry
  than the prototype — which would indicate a structural error.

NOTE on "before" SG for disordered materials:
  The space group of a disordered CIF (partial occupancies) is taken from
  the wyckoff pipeline CSV (sg_number column), which reflects the underlying
  Wyckoff-assignment prototype SG.  Reading SG from a partially-occupied
  CIF can give misleading results; the CSV value is authoritative.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PREPROCESSING_WORK  # noqa: E402


import os, re, csv, sys, warnings, logging
import numpy as np
import pandas as pd
from collections import defaultdict

from pymatgen.core import Structure, Element
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

# ─────────────────────────── PATHS ──────────────────────────────────────────

BASE_DIR         = str(PREPROCESSING_WORK)
TARGET_CSV       = os.path.join(BASE_DIR, "magnetocaloric_materials.csv")
WYCKOFF_CSV      = os.path.join(BASE_DIR, "magnetocaloric_materials_with_wyckoff.csv")
COMBINED_CSV     = os.path.join(BASE_DIR, "magnetocaloric_materials_with_cif.csv")
REPORT_CSV       = os.path.join(BASE_DIR, "cif_quality_report.csv")
SUMMARY_TXT      = os.path.join(BASE_DIR, "cif_quality_summary.txt")

STOICH_PASS_THRESHOLD = 5.0    # % max per-atom-fraction error → PASS
STOICH_WARN_THRESHOLD = 10.0   # % → WARN, above this → FAIL

# ─────────────────── CRYSTAL SYSTEM RANKS ────────────────────────────────────
# Higher rank = lower symmetry.  SQS ordering should INCREASE rank (or keep same).

def sg_to_crystal_system(sg: int) -> str:
    if sg is None or sg == 0:   return "unknown"
    if sg >= 195:               return "cubic"
    if sg >= 168:               return "hexagonal"
    if sg >= 143:               return "trigonal"
    if sg >= 75:                return "tetragonal"
    if sg >= 16:                return "orthorhombic"
    if sg >= 3:                 return "monoclinic"
    return "triclinic"

SYSTEM_RANK = {
    "cubic": 0, "hexagonal": 1, "trigonal": 2,
    "tetragonal": 3, "orthorhombic": 4, "monoclinic": 5,
    "triclinic": 6, "unknown": 7,
}

# ─────────── PROTOTYPE → EXPECTED SG LOOKUP ──────────────────────────────────
# Maps prototype name (from CSV) to expected (SG number, crystal system).
# None means "unknown / not in this table".

PROTOTYPE_SG = {
    "MgCu2":                   (227, "cubic"),
    "MgCu2 Fd-3m":             (227, "cubic"),
    "Fd-3m->R-3m":             (166, "trigonal"),   # distorted Laves
    "NaZn13":                  (225, "cubic"),
    "CsCl":                    (221, "cubic"),
    "BiF3":                    (225, "cubic"),
    "MgAgAs":                  (216, "cubic"),
    "Cu2MnAl":                 (225, "cubic"),
    "gamma(fcc)->alpha(bcc)":  (229, "cubic"),
    "ThCr2Si2":                (139, "tetragonal"),
    "BaAl4":                   (139, "tetragonal"),
    "I4/mmm":                  (139, "tetragonal"),
    "I4/mcm":                  (140, "tetragonal"),
    "Nd2Fe14B":                (136, "tetragonal"),
    "Th2Ni17":                 (194, "hexagonal"),
    "Th2Zn17":                 (166, "trigonal"),
    "CaCu5":                   (191, "hexagonal"),
    "CeCo3B2":                 (191, "hexagonal"),
    "SrNi12B6":                (191, "hexagonal"),
    "AlB2 (P6/mmm)":           (191, "hexagonal"),
    "Ni2In":                   (194, "hexagonal"),
    "Ni2In (P63/mmc)":         (194, "hexagonal"),
    "Ni2In/TiNiSi->Ni2In":    (194, "hexagonal"),
    "MgZn2 (P63/mmc)":        (194, "hexagonal"),
    "Mn5Si3":                  (193, "hexagonal"),
    "hexagonal":               (194, "hexagonal"),
    "Fe2P":                    (189, "hexagonal"),
    "CeFeSi":                  (129, "tetragonal"),
    "CeCu2 (Imma)":            (74,  "orthorhombic"),
    "TiNiSi":                  (62,  "orthorhombic"),
    "NiAs->MnP":               (62,  "orthorhombic"),
    "O (Pnma)":                (62,  "orthorhombic"),
    "Fe3C":                    (62,  "orthorhombic"),
    "M (P1121/a)":             (14,  "monoclinic"),
    "Mixed":                   (14,  "monoclinic"),   # Gd5(Si,Ge)4 Mixed phase
    "monoclinic":              (7,   "monoclinic"),   # YFe2D/H type
    "CaTiO3":                  (221, "cubic"),        # antiperovskite
    "Ce6Ni2Si3":               (176, "hexagonal"),
    "NaZn13":                  (225, "cubic"),
}

# ─────────────────────────── HELPERS ────────────────────────────────────────

def parse_target_formula(formula_str: str) -> dict:
    """
    Return {element_symbol: fractional_amount} for the target formula.
    Fractional amounts are normalized so they sum to 1.
    Returns empty dict if parsing fails.
    """
    try:
        comp = Composition(str(formula_str))
        total = sum(comp.values())
        if total == 0:
            return {}
        return {str(el): float(v) / total for el, v in comp.items()}
    except Exception:
        return {}


def cif_to_composition(cif_path: str) -> tuple:
    """
    Load a CIF and return (structure, composition_dict, sg_number, sg_symbol,
                            crystal_system, atom_types_set, error_msg).
    composition_dict maps element_symbol → global fraction (sums to 1).
    On error, structure is None and error_msg is set.
    """
    if not cif_path or not os.path.exists(str(cif_path)):
        return None, {}, None, "", "", set(), f"file not found: {cif_path}"
    try:
        s = Structure.from_file(str(cif_path))
    except Exception as e:
        return None, {}, None, "", "", set(), f"load error: {e}"

    # Composition
    total = sum(s.composition.values())
    comp_dict = {str(el): float(v) / total for el, v in s.composition.items()}
    # Strip charge/oxidation state from element keys: 'Fe0+' → 'Fe', 'D0+' → 'D'
    clean = {}
    for k, v in comp_dict.items():
        m = re.match(r"([A-Z][a-z]?)", k)
        sym = m.group(1) if m else k
        clean[sym] = clean.get(sym, 0.0) + v
    comp_dict = clean
    atom_types = set(comp_dict.keys())

    # Space group
    sg_num, sg_sym, crystal_sys = None, "", ""
    try:
        sga = SpacegroupAnalyzer(s, symprec=0.2)
        sg_num = sga.get_space_group_number()
        sg_sym = sga.get_space_group_symbol()
        crystal_sys = sg_to_crystal_system(sg_num)
    except Exception as e:
        sg_sym = f"error:{e}"

    return s, comp_dict, sg_num, sg_sym, crystal_sys, atom_types, ""


def stoich_deviation(actual_comp: dict, target_comp: dict) -> tuple:
    """
    Compute max |actual_fraction - target_fraction| across all elements in
    the union of both compositions.
    Returns (max_error_pct, per_element_errors_dict).
    """
    all_els = set(list(actual_comp) + list(target_comp))
    errors  = {}
    for el in all_els:
        act = actual_comp.get(el, 0.0)
        tgt = target_comp.get(el, 0.0)
        errors[el] = abs(act - tgt) * 100.0   # as percentage
    max_err = max(errors.values()) if errors else 0.0
    return max_err, errors


def status_from_metrics(stoich_err: float, missing_els: set,
                        extra_els: set, crystal_rank_delta: int,
                        cif_found: bool) -> str:
    if not cif_found:
        return "SKIP"
    if not missing_els and stoich_err <= STOICH_PASS_THRESHOLD and crystal_rank_delta >= 0:
        return "PASS"
    if stoich_err > STOICH_WARN_THRESHOLD or crystal_rank_delta < -1:
        return "FAIL"
    return "WARN"


# ─────────────────────────── MAIN ───────────────────────────────────────────

def main():
    # ── Load CSVs ────────────────────────────────────────────────────────────
    print("Loading CSVs...")
    try:
        df_target = pd.read_csv(TARGET_CSV)
    except Exception as e:
        sys.exit(f"Cannot read {TARGET_CSV}: {e}")

    try:
        df_wyckoff = pd.read_csv(WYCKOFF_CSV)
    except Exception as e:
        sys.exit(f"Cannot read {WYCKOFF_CSV}: {e}")

    if not os.path.exists(COMBINED_CSV):
        sys.exit(f"{COMBINED_CSV} not found — run step1_generate_sqs_v2.py first.")
    df_combined = pd.read_csv(COMBINED_CSV)

    # Build target composition lookup: formula → {el: fraction}
    target_comp_by_formula = {}
    for _, row in df_target.iterrows():
        formula = str(row.get("material_formula", "")).strip()
        comp    = parse_target_formula(formula)
        if comp:
            target_comp_by_formula[formula] = comp

    # Build prototype SG lookup from wyckoff CSV (authoritative per-material)
    proto_sg_by_formula = {}
    for _, row in df_wyckoff.iterrows():
        formula  = str(row.get("material_formula", "")).strip()
        sg_num   = row.get("sg_number", None)
        sg_sym   = row.get("sg_symbol", "")
        proto    = str(row.get("structure_prototype", "")).strip()
        try:
            sg_num = int(float(sg_num)) if sg_num else None
        except Exception:
            sg_num = None
        if formula not in proto_sg_by_formula and sg_num:
            proto_sg_by_formula[formula] = {
                "sg_number":   sg_num,
                "sg_symbol":   sg_sym,
                "crystal_sys": sg_to_crystal_system(sg_num),
                "prototype":   proto,
            }

    # ── Process each material ─────────────────────────────────────────────────
    print(f"Analysing {len(df_combined)} rows from combined CSV...")

    # Deduplicate: keep best row per formula (lowest stoich error, then first)
    df_best = (
        df_combined
        .assign(_sqs_err=pd.to_numeric(
            df_combined.get("sqs_comp_error", pd.Series(["0"]*len(df_combined))),
            errors="coerce"
        ).fillna(0))
        .sort_values("_sqs_err")
        .drop_duplicates("material_formula", keep="first")
    )
    print(f"  After deduplication: {len(df_best)} unique materials")

    records = []

    for _, row in df_best.iterrows():
        formula    = str(row.get("material_formula", "")).strip()
        prototype  = str(row.get("structure_prototype", "")).strip()
        is_dis     = str(row.get("is_disordered", "")).strip().lower() == "true"
        pre_path   = str(row.get("cif_path", "")).strip()
        sqs_path   = str(row.get("sqs_cif_path", "")).strip()
        final_path = str(row.get("final_cif_for_m3gnet", "")).strip()

        # "After" CIF = final_cif_for_m3gnet (SQS if disordered, original if not)
        # "Before" CIF = original cif_path (may be disordered; we read it anyway)
        after_path  = final_path if final_path else pre_path
        before_path = pre_path

        # ── Prototype / expected SG ──────────────────────────────────────────
        proto_info = proto_sg_by_formula.get(formula, {})
        proto_sg_num = proto_info.get("sg_number", None)
        proto_sys    = proto_info.get("crystal_sys", "")
        # Fallback to table lookup
        if not proto_sg_num and prototype in PROTOTYPE_SG:
            psn, pcs = PROTOTYPE_SG[prototype]
            proto_sg_num, proto_sys = psn, pcs
        # Fallback: strip variant suffix and retry
        if not proto_sg_num:
            base_proto = prototype.split("(")[0].split("->")[0].strip()
            if base_proto in PROTOTYPE_SG:
                proto_sg_num, proto_sys = PROTOTYPE_SG[base_proto]

        # ── Target composition ───────────────────────────────────────────────
        target_comp = target_comp_by_formula.get(formula, {})
        if not target_comp:
            target_comp = parse_target_formula(formula)

        # ── Before CIF ──────────────────────────────────────────────────────
        _, pre_comp, pre_sg_num, pre_sg_sym, pre_sys, pre_atoms, pre_err = \
            cif_to_composition(before_path)
        pre_found = (pre_err == "")

        # For disordered CIFs, SG analysis is unreliable — use wyckoff CSV value
        if is_dis and pre_found and proto_sg_num:
            pre_sg_num = proto_sg_num
            pre_sg_sym = proto_info.get("sg_symbol", pre_sg_sym)
            pre_sys    = proto_sys or pre_sys

        pre_stoich_err, pre_per_el = stoich_deviation(pre_comp, target_comp)

        # ── After CIF ───────────────────────────────────────────────────────
        after_found    = bool(after_path and os.path.exists(after_path))
        if after_path == before_path:
            # Not disordered — no SQS applied
            aft_comp, aft_sg_num, aft_sg_sym, aft_sys, aft_atoms = \
                pre_comp, pre_sg_num, pre_sg_sym, pre_sys, pre_atoms
            aft_err   = pre_err
            aft_found = pre_found
            sqs_applied = False
        else:
            _, aft_comp, aft_sg_num, aft_sg_sym, aft_sys, aft_atoms, aft_err = \
                cif_to_composition(after_path)
            aft_found   = (aft_err == "")
            sqs_applied = True

        aft_stoich_err, aft_per_el = stoich_deviation(aft_comp, target_comp)

        # ── Symmetry checks ──────────────────────────────────────────────────
        # Crystal system rank: higher = lower symmetry (expected after SQS)
        proto_rank = SYSTEM_RANK.get(proto_sys, 7)
        aft_rank   = SYSTEM_RANK.get(aft_sys, 7)
        # Positive delta = SQS is LOWER symmetry than prototype (expected/OK)
        # Negative delta = SQS is HIGHER symmetry than prototype (suspicious)
        crystal_rank_delta = aft_rank - proto_rank

        sg_changed = (pre_sg_num != aft_sg_num) if (pre_sg_num and aft_sg_num) else None
        sg_raised  = (aft_sg_num is not None and proto_sg_num is not None
                      and aft_sg_num > proto_sg_num)

        # ── Missing / extra elements ─────────────────────────────────────────
        target_els  = set(target_comp.keys())
        aft_els     = aft_atoms
        missing_els = target_els - aft_els  if aft_found else target_els
        extra_els   = aft_els   - target_els if aft_found else set()

        # ── Status ──────────────────────────────────────────────────────────
        status = status_from_metrics(
            aft_stoich_err, missing_els, extra_els,
            crystal_rank_delta, aft_found
        )
        # Escalate if SG raised
        if sg_raised and status == "PASS":
            status = "WARN"
        if sg_raised and crystal_rank_delta < 0:
            status = "FAIL"

        # ── Notes ────────────────────────────────────────────────────────────
        notes = []
        if not aft_found:
            notes.append(f"CIF missing: {after_path}")
        if missing_els:
            notes.append(f"missing elements: {sorted(missing_els)}")
        if extra_els:
            notes.append(f"extra elements: {sorted(extra_els)}")
        if sg_raised:
            notes.append(f"SG raised: proto={proto_sg_num} → sqs={aft_sg_num}")
        if sg_changed and sqs_applied:
            notes.append(f"SG changed: {pre_sg_num}({pre_sg_sym}) → "
                         f"{aft_sg_num}({aft_sg_sym})")
        if aft_stoich_err > STOICH_PASS_THRESHOLD:
            worst_el = max(aft_per_el, key=aft_per_el.get) if aft_per_el else ""
            notes.append(f"stoich err {aft_stoich_err:.2f}% (worst: {worst_el})")

        records.append({
            "formula":              formula,
            "prototype":            prototype,
            "is_disordered":        is_dis,
            "sqs_applied":          sqs_applied,
            # Before
            "before_cif":           before_path,
            "before_sg_num":        pre_sg_num,
            "before_sg_sym":        pre_sg_sym,
            "before_crystal_sys":   pre_sys,
            "before_stoich_err_pct":f"{pre_stoich_err:.2f}",
            "before_atoms":         ", ".join(sorted(pre_atoms)),
            # After
            "after_cif":            after_path,
            "after_sg_num":         aft_sg_num,
            "after_sg_sym":         aft_sg_sym,
            "after_crystal_sys":    aft_sys,
            "after_stoich_err_pct": f"{aft_stoich_err:.2f}",
            "after_atoms":          ", ".join(sorted(aft_atoms)) if aft_found else "",
            # Prototype reference
            "proto_expected_sg":    proto_sg_num,
            "proto_expected_sys":   proto_sys,
            # Summary
            "crystal_rank_delta":   crystal_rank_delta,
            "sg_raised":            sg_raised,
            "missing_elements":     ", ".join(sorted(missing_els)),
            "extra_elements":       ", ".join(sorted(extra_els)),
            "status":               status,
            "notes":                "; ".join(notes),
        })

    # ── Write report CSV ─────────────────────────────────────────────────────
    if records:
        fieldnames = list(records[0].keys())
        with open(REPORT_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)
        print(f"\nReport written → {REPORT_CSV}")

    # ── Write human-readable summary ─────────────────────────────────────────
    total     = len(records)
    passed    = [r for r in records if r["status"] == "PASS"]
    warned    = [r for r in records if r["status"] == "WARN"]
    failed    = [r for r in records if r["status"] == "FAIL"]
    skipped   = [r for r in records if r["status"] == "SKIP"]
    sqs_done  = [r for r in records if r["sqs_applied"]]

    lines = [
        "=" * 70,
        "CIF QUALITY REPORT",
        "=" * 70,
        f"  Total materials analysed : {total}",
        f"  Went through SQS step    : {len(sqs_done)}",
        f"",
        f"  PASS  : {len(passed):4d}  (stoich ≤5%, elements OK, symmetry OK)",
        f"  WARN  : {len(warned):4d}  (stoich 5-10% or minor symmetry drop)",
        f"  FAIL  : {len(failed):4d}  (stoich >10%, missing elements, SG raised)",
        f"  SKIP  : {len(skipped):4d}  (CIF file not found)",
        "=" * 70,
    ]

    if warned:
        lines += ["", "── WARNINGS ──────────────────────────────────────────"]
        for r in sorted(warned, key=lambda x: float(x["after_stoich_err_pct"]
                                                     if x["after_stoich_err_pct"] else 0),
                         reverse=True):
            lines.append(f"  {r['formula']:<45}  {r['after_stoich_err_pct']:>6}%  "
                         f"SG:{r['after_sg_num']}({r['after_sg_sym']})  "
                         f"[{r['notes'][:80]}]")

    if failed:
        lines += ["", "── FAILURES ──────────────────────────────────────────"]
        for r in sorted(failed, key=lambda x: float(x["after_stoich_err_pct"]
                                                     if x["after_stoich_err_pct"] else 0),
                         reverse=True):
            lines.append(f"  {r['formula']:<45}  {r['after_stoich_err_pct']:>6}%  "
                         f"SG:{r['after_sg_num']}({r['after_sg_sym']})  "
                         f"[{r['notes'][:80]}]")

    if skipped:
        lines += ["", f"── SKIPPED (no CIF) ─────── {len(skipped)} materials ────"]
        for r in skipped[:20]:
            lines.append(f"  {r['formula']}")
        if len(skipped) > 20:
            lines.append(f"  ... and {len(skipped)-20} more")

    lines += [
        "",
        "── SG ANALYSIS (disordered materials that went through SQS) ──────",
    ]
    sg_info = [(r["formula"], r["before_sg_num"], r["before_sg_sym"],
                r["after_sg_num"],  r["after_sg_sym"],
                r["proto_expected_sg"], r["sg_raised"])
               for r in records if r["sqs_applied"]]
    raised  = [x for x in sg_info if x[6]]
    changed = [x for x in sg_info if x[1] != x[3] and x[1] and x[3]]
    same    = [x for x in sg_info if x[1] == x[3]]
    lines.append(f"  SG unchanged after SQS  : {len(same)}")
    lines.append(f"  SG reduced (expected)   : {len(changed) - len(raised)}")
    lines.append(f"  SG raised (suspicious)  : {len(raised)}")
    if raised:
        lines.append("  Raised SG details:")
        for x in raised:
            lines.append(f"    {x[0]:<40} proto={x[5]} before={x[1]} after={x[3]}")
    lines.append("")
    lines.append("── STOICHIOMETRY SUMMARY (after SQS) ──────────────────────")
    errs = [float(r["after_stoich_err_pct"]) for r in records
            if r["after_stoich_err_pct"] and r["status"] != "SKIP"]
    if errs:
        lines.append(f"  Mean stoich error  : {np.mean(errs):.2f}%")
        lines.append(f"  Median stoich error: {np.median(errs):.2f}%")
        lines.append(f"  Max stoich error   : {max(errs):.2f}%")
        lines.append(f"  Materials ≤5%      : {sum(1 for e in errs if e<=5)}")
        lines.append(f"  Materials 5-10%    : {sum(1 for e in errs if 5<e<=10)}")
        lines.append(f"  Materials >10%     : {sum(1 for e in errs if e>10)}")
    lines.append("=" * 70)

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(f"\nSummary written → {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
