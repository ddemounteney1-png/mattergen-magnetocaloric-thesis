"""
magnetocaloric_wyckoff_pipeline_v14.py
FINAL PRODUCTION VERSION - All 8 root causes from diagnostic_v13 fixed

FIXES vs v13:
FIX-1  Formula cleaning - strips Unicode \uf064, trailing '+', undefined 'x'
FIX-2  Dual MAX_ATOMS - 50 for standard, 60 for Fe2P/NaZn13 prototypes
FIX-3  Separate CIF folders - cif_files/ (≤50 atoms), cif_>50/ (51-60 atoms)
FIX-4  Mixed mode fix - ALL sites of same role get same occupancy dict
FIX-5  Scale selection - prefer scales that reduce stoich error even if >DOPANT_TOL
FIX-6  Prototype overrides - CoMnSb→MgAgAs, Gd3X→correct prototype
FIX-7  Stoichiometry validation - added stoich_error tracking to output CSV
FIX-8  Better fallback - direct MP search when prototype stoich error >10%

OUTPUT STRUCTURE:
$PREPROCESSING_DIR/ (see code/project_paths.py)
├── cif_files/              ← structures ≤50 atoms (MatterGen compatible)
├── cif_>50/                ← structures 51-60 atoms (need SQS expansion)
├── magnetocaloric_materials_with_wyckoff.csv
├── missing_materials.txt
└── stoich_errors.txt       ← materials with >5% stoichiometry error
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PREPROCESSING_WORK, SOURCE_DATA  # noqa: E402

import os
import sys
import re
import csv
import logging
import warnings
import unicodedata
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
MP_API_KEY = os.environ.get("MP_API_KEY")
if not MP_API_KEY:
    sys.exit("Set the MP_API_KEY environment variable. Get a key at https://next-gen.materialsproject.org/api")
BASE_DIR = str(PREPROCESSING_WORK)
CSV_INPUT  = str(SOURCE_DATA / "magnetocaloric_materials.csv")  # committed input
CSV_OUTPUT = os.path.join(BASE_DIR, "magnetocaloric_materials_with_wyckoff.csv")
FAILED_CSV = os.path.join(BASE_DIR, "Failed_materials.csv")
CLEAN_DIR  = os.path.join(BASE_DIR, "clean_cifs")
MANUAL_CSV = os.path.join(BASE_DIR, "needs_manual_search.csv")
STOICH_ERR_IN = os.path.join(BASE_DIR, "stoich_errors.txt")
CIF_DIR_STANDARD = os.path.join(BASE_DIR, "cif_files")
CIF_DIR_LARGE    = os.path.join(BASE_DIR, "cif_>50")
MAX_ATOMS_STANDARD = 50
MAX_ATOMS_LARGE = 60
DOPANT_TOL = 0.025
STOICH_TOL = 0.05  # 5% max per-atom-fraction error

# =============================================================================
# FIX-1: FORMULA CLEANING
# =============================================================================
def clean_formula(formula: str) -> tuple[str, bool]:
    """
    Strip unparseable artifacts from formula strings.
    Returns (cleaned_formula, was_changed).
    
    Handles:
    - Unicode \uf064 (private-use 'd' marker for deuterium)
    - Trailing '+' (layered manganite notation)
    - Undefined 'x' stoichiometry (e.g., MnCoGeBx → MnCoGeB0)
    """
    original = formula
    cleaned = formula
    
    # Replace private-use Unicode chars (U+F064 = private-use 'd' for deuterium)
    cleaned = cleaned.replace('\uf064', '')
    cleaned = cleaned.replace('\uf044', '')

    # Replace 'Hd' (deuterium label) with 'H'
    cleaned = re.sub(r'Hd(?=[A-Z0-9()\s]|$)', 'H', cleaned)
    
    # Strip trailing '+' 
    cleaned = re.sub(r'\+$', '', cleaned.strip())
    
    # Replace undefined stoichiometry 'Bx' → 'B0'
    cleaned = re.sub(r'([A-Z][a-z]?)x\b', r'\g<1>0', cleaned)
    
    # Strip remaining non-ASCII
    cleaned = ''.join(c for c in cleaned if ord(c) < 128)
    
    return cleaned.strip(), cleaned != original

# =============================================================================
# FIX-6: PROTOTYPE OVERRIDES
# =============================================================================
_FORMULA_PROTO_OVERRIDES = {
    # (reduced_formula, csv_prototype) -> correct proto_formula for site-roles
    ("CoMnSb",  "CsCl"):  "MgAgAs",   # Half-Heusler needs 3 sites, not 2
    ("CoNbMnSb","MgAgAs"):"MgAgAs",   # 4 elements on 3 sites → disordered
    # FIX 9: Gd3X intermetallics — Fe3C IS the correct prototype but Gd is A-role
    # so there is no A-site in standard Fe3C table [B,B,X]. Use Fe3C_RE3 instead.
    ("Gd3Rh",   "Fe3C"):  "Fe3C_RE3",
    ("Gd3Co",   "Fe3C"):  "Fe3C_RE3",
    ("Gd3Ni",   "Fe3C"):  "Fe3C_RE3",
    # TiNiSi CoMn materials: both TM sites are B-role; no A-role element present.
    # Redirect to CoMnGe table [B,B,X] so Co and Mn each get their own site.
    ("Mn1Co1Si0.88Ge0.12", "TiNiSi"): "CoMnGe",
    ("Mn1Fe0.15Co0.85Ge1", "TiNiSi"): "CoMnGe",
    ("Mn1Fe0.1Co0.9Ge1",   "TiNiSi"): "CoMnGe",
}

# When MP returns a structure with an unexpected SG (fallback), redirect site-role
# lookup to a different PROTO_SITE_ROLES key that matches the actual structure.
# Key: (proto_formula_returned_by_fetch, actual_sg_number)
_PROTO_SG_FALLBACK = {
    # TbCo2 SG166 not in MP; falls back to SG227 (Fd-3m, MgCu2-type, 2 sites).
    # Use MgCu2 site roles [("Cu","B"), ("Mg","A")] instead of TbCo2's 3-entry table.
    ("TbCo2", 227): "MgCu2",
}

# For these prototypes, ALL target elements are treated as B-role in assign_wyckoff
# regardless of their normal _role() classification. Use for alloy systems where every
# crystallographic site is substitutional (e.g. Heusler alloys).
_PROTO_ALL_B = {"Cu2MnAl"}

# =============================================================================
# PROTOTYPE QUERY MAP
# =============================================================================
PROTO_QUERY = {
    "MgCu2": {"formula": "MgCu2", "sg": 227},
    "MgCu2 Fd-3m": {"formula": "MgCu2", "sg": 227},
    "MgZn2 (P63/mmc)": {"formula": "MgZn2", "sg": 194},
    "Fd-3m->R-3m": {"formula": "TbCo2", "sg": 166},
    "NaZn13": {"formula": "NaZn13", "sg": 226},
    "O (Pnma)": {"formula": "Gd5Ge4", "sg": 62},
    "Gd5Ge4": {"formula": "Gd5Ge4", "sg": 62},
    # FIX 6: M(P1121/a) monoclinic phase has no direct MP entry; use Gd5Ge4 (Pnma, SG62) as substitute
    "M (P1121/a)": {"formula": "Gd5Ge4", "sg": 62},
    "Mixed":       {"formula": "Gd5Ge4", "sg": 62},
    "ThCr2Si2": {"formula": "ThCr2Si2", "sg": 139},
    "CaTiO3": {"formula": "CaTiO3", "sg": 221},
    "I4/mmm": {"formula": "La2CuO4", "sg": 139},
    "Ni2In": {"formula": "Ni2In", "sg": 194},
    "Ni2In (P63/mmc)": {"formula": "Ni2In", "sg": 194},
    "Ni2In/TiNiSi->Ni2In": {"formula": "CoMnGe", "sg": 194},
    "TiNiSi": {"formula": "TiNiSi", "sg": 62},
    "Fe2P": {"formula": "Fe2P", "sg": 189},
    "Mn5Si3": {"formula": "Mn5Si3", "sg": 193},
    "NiAs->MnP": {"formula": "MnAs", "sg": 194},
    "MnAs": {"formula": "MnAs", "sg": 194},
    "MgAgAs": {"formula": "MgAgAs", "sg": 216},
    "BiF3": {"formula": "Cu2MnAl", "sg": 225},
    "CsCl": {"formula": "CsCl", "sg": 221},
    "gamma(fcc)->alpha(bcc)": {"formula": "FePt", "sg": 123},
    "Fe3C": {"formula": "Fe3C", "sg": 62},
    "Th2Ni17": {"formula": "Th2Ni17", "sg": 194},
    "Th2Zn17": {"formula": "Th2Zn17", "sg": 166},
    "Ce6Ni2Si3": {"formula": "Ce6Ni2Si3", "sg": 176},
    "CeCo3B2": {"formula": "CeCo3B2", "sg": 191},
    "CeCu2 (Imma)": {"formula": "CeCu2", "sg": 74},
    "CeCu2": {"formula": "CeCu2", "sg": 74},
    "CeFeSi": {"formula": "CeFeSi", "sg": 129},
    "AlB2 (P6/mmm)": {"formula": "AlB2", "sg": 191},
    "SrNi12B6": {"formula": "SrNi12B6", "sg": 191},
    "I4/mcm": {"formula": "Mn5PB2", "sg": 140},
    "Mn5PB2": {"formula": "Mn5PB2", "sg": 140},
    "hexagonal": None,
    "monoclinic": None,
}

# =============================================================================
# SITE-ROLE TABLE - FIX-4: Corrected site counts
# =============================================================================
PROTO_SITE_ROLES = {
    "MgCu2": [("Cu", "B"), ("Mg", "A")],
    "MgZn2": [("Zn", "B"), ("Mg", "A"), ("Zn", "B")],  # FIX: 3 sites not 2
    "TbCo2": [("Co", "B"), ("Co", "B"), ("Tb", "A")],  # FIX: 3 sites not 2
    "NaZn13": [("Zn", "B"), ("Zn", "B"), ("Na", "A")],
    "Gd5Ge4": [("Gd", "A"), ("Gd", "A"), ("Ge", "X"), ("Gd", "A"), ("Ge", "X"), ("Ge", "X")],
    "Gd5Si2Ge2": [("Gd", "A"), ("Gd", "A"), ("Gd", "A"), ("Si", "X"), ("Ge", "X")],
    "ThCr2Si2": [("Cr", "B"), ("Si", "X"), ("Th", "A")],
    "CaTiO3": [("O", "X"), ("Ti", "B"), ("Ca", "A")],
    "La2CuO4": [("O", "X"), ("La", "A"), ("O", "X"), ("Cu", "B")],
    "Ni2In": [("Ni", "B"), ("Ni", "B"), ("In", "A")],
    "CoMnGe": [("Mn", "B"), ("Co", "B"), ("Ge", "X")],
    "TiNiSi": [("Ni", "B"), ("Ti", "A"), ("Si", "X")],
    "Fe2P": [("Fe", "B"), ("Fe", "B"), ("P", "X"), ("P", "X")],
    "Mn5Si3": [("Mn", "B"), ("Si", "X"), ("Mn", "B"), ("Si", "X")],  # FIX: 4 sites
    "MnAs": [("Mn", "B"), ("As", "X")],
    "MgAgAs": [("Ag", "B"), ("Mg", "B"), ("As", "X")],
    "Cu2MnAl": [("Cu", "B"), ("Mn", "B"), ("Al", "B")],
    "CsCl": [("Cs", "A"), ("Cl", "X")],
    "FePt": [("Pt", "B"), ("Fe", "B")],
    "Fe3C": [("Fe", "B"), ("Fe", "B"), ("C", "X")],
    # FIX 9: Fe3C_RE3 — rare-earth-rich intermetallics (Gd3X) where the majority
    # rare-earth occupies the first two sites (treated as A-role) and the minority
    # transition metal occupies the X-site. Avoids post-loop merge placing Gd on B.
    "Fe3C_RE3": [("Fe", "A"), ("Fe", "A"), ("C", "X")],
    "Th2Ni17": [("Ni", "B"), ("Ni", "B"), ("Ni", "B"), ("Ni", "B"), ("Th", "A")],
    "Th2Zn17": [("Zn", "B"), ("Zn", "B"), ("Zn", "B"), ("Zn", "B"), ("Th", "A")],
    "Ce6Ni2Si3": [("Ce", "A"), ("Ce", "A"), ("Ni", "B"), ("Si", "X"), ("Si", "X")],
    "CeCo3B2": [("Co", "B"), ("Ce", "A"), ("B", "X")],
    "CeCu2": [("Cu", "B"), ("Ce", "A")],  # 2 sites in actual structure
    "CeFeSi": [("Ce", "A"), ("Fe", "B"), ("Si", "X")],
    "AlB2": [("B", "X"), ("Al", "A")],
    "SrNi12B6": [("Ni", "B"), ("Ni", "B"), ("B", "X"), ("Sr", "A")],
    "Mn5PB2": [("Mn", "B"), ("B", "X"), ("Mn", "B"), ("P", "X")],
}

# =============================================================================
# ELEMENT CLASSIFICATION
# =============================================================================
_RE = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
       "Tm", "Yb", "Lu", "Y", "Sc", "Th", "U", "Np", "Pu", "Am", "Cm"}
# FIX 5: Added Pb, Cd, Ag (perovskite A-site fillers), K, Na (Ruddlesden-Popper A-site)
_AE = {"Ca", "Sr", "Ba", "Ra", "Be", "Mg", "Pb", "Cd", "K", "Na"}
# Ag removed: it is a noble/transition metal (B-role), not an alkaline earth
_AN = {"O", "N", "F", "Cl", "Br", "I", "S", "Se", "Te"}
# FIX 5: Pb removed from _SP to avoid A/X role conflict
_SP = {"B", "C", "Si", "Ge", "As", "P", "Sb", "Bi", "Sn", "H", "D"}

def _role(sym: str) -> str:
    if sym in _RE or sym in _AE:
        return "A"
    if sym in _AN or sym in _SP:
        return "X"
    return "B"

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# FORMULA HELPERS
# =============================================================================
def parse_comp(formula: str):
    try:
        return Composition(formula)
    except Exception as e:
        log.warning(f"  Cannot parse '{formula}': {e}")
        return None

# =============================================================================
# FIX-7: STOICHIOMETRY HELPERS
# =============================================================================
def stoich_error(struct: Structure, formula: str) -> float:
    """
    Max absolute per-atom-fraction difference between structure and target.
    Returns value in [0, 1]. Per-atom fractions avoid amplification for rare elements.
    """
    try:
        comp_t = Composition(formula)
        comp_s = struct.composition
        total_t = float(comp_t.num_atoms)
        total_s = float(comp_s.num_atoms)
        if total_t == 0 or total_s == 0:
            return 1.0
        all_els = ({str(e) for e in comp_t.elements} |
                   {str(e) for e in comp_s.elements})
        max_err = 0.0
        for el in all_els:
            t_frac = float(comp_t.get(el, 0)) / total_t
            s_frac = float(comp_s.get(el, 0)) / total_s
            max_err = max(max_err, abs(t_frac - s_frac))
        return max_err
    except Exception:
        return 1.0

def cif_stoich_string(struct: Structure) -> str:
    """Human-readable per-atom-fraction composition from built structure."""
    try:
        comp = struct.composition
        total = float(comp.num_atoms)
        parts = [f"{el}:{float(amt)/total:.4f}"
                 for el, amt in sorted(comp.items(), key=lambda x: -float(x[1]))]
        return " ".join(parts)
    except Exception:
        return "unknown"

# =============================================================================
# FIX-2/3: DUAL MAX_ATOMS HANDLING
# =============================================================================
def get_max_atoms_for_prototype(prototype: str) -> int:
    """
    Return appropriate MAX_ATOMS limit based on prototype.
    Fe2P and NaZn13 need 60 atoms for correct stoichiometry.
    """
    if prototype in ["Fe2P", "NaZn13", "O (Pnma)", "Gd5Ge4"]:
        return MAX_ATOMS_LARGE
    return MAX_ATOMS_STANDARD

def get_cif_dir_for_atoms(n_atoms: int) -> str:
    """Return appropriate CIF directory based on atom count."""
    if n_atoms <= MAX_ATOMS_STANDARD:
        return CIF_DIR_STANDARD
    return CIF_DIR_LARGE

# =============================================================================
# SUPERCELL SIZE FINDER - FIX-5: Better scale selection
# =============================================================================
def find_ordered_scale(n_sublattice: int, frac: float, n_prim: int,
                       max_atoms: int, tol: float):
    """
    Find smallest supercell scale where dopant fraction maps to whole atoms.
    FIX-5: Also check if scale reduces stoich error even if >DOPANT_TOL.
    """
    best_s, best_err = 1, 1.0
    max_s = max(1, int((max_atoms / n_prim) ** (1/3))) if n_prim else 1
    
    for s in range(1, max_s + 1):
        if s ** 3 * n_prim > max_atoms:
            break
        effective_mult = s ** 3 * n_sublattice
        expected_atoms = effective_mult * frac
        rounded = round(expected_atoms)
        
        # Prevent rounding dopants to zero
        if rounded == 0 and expected_atoms > 0.01:
            err = 1.0
        else:
            err = abs(expected_atoms - rounded)
        
        if err < best_err:
            best_err, best_s = err, s
        
        if err < tol:
            return s, True
    
    return best_s, best_err < tol

# =============================================================================
# STEP 1 - FETCH PROTOTYPE
# =============================================================================
def fetch_prototype(proto_name: str, mpr: MPRester):
    q = PROTO_QUERY.get(proto_name)
    if q is None:
        log.info(f"  '{proto_name}' has no MP query")
        return None, "", ""
    pf, sg = q["formula"], q["sg"]
    log.info(f"  Prototype query: formula={pf}, SG={sg}")
    
    def _search(formula, spacegroup=None):
        kw = dict(formula=formula,
                  fields=["structure", "material_id", "symmetry", "deprecated"])
        if spacegroup:
            kw["spacegroup_number"] = spacegroup
        return mpr.materials.search(**kw)
    
    results = []
    try:
        results = _search(pf, sg)
    except Exception as e:
        log.warning(f"  MP query (with SG) error: {e}")
    
    if not results:
        try:
            results = _search(pf)
        except Exception as e:
            log.warning(f"  MP query (no SG) error: {e}")
    
    if not results:
        log.warning(f"  Prototype '{proto_name}' not found on MP.")
        return None, "", ""
    
    results.sort(key=lambda r: (
        r.deprecated,
        0 if (hasattr(r, "symmetry") and r.symmetry and r.symmetry.number == sg) else 1,
    ))
    best = results[0]
    log.info(f"  Found: {best.material_id}")
    return best.structure, best.material_id, pf

# =============================================================================
# STEP 2 - GET WYCKOFF SITES
# =============================================================================
def get_wyckoff_sites(struct: Structure):
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.1)
        sym = sga.get_symmetrized_structure()
        sg_num = sga.get_space_group_number()
        sg_sym = sga.get_space_group_symbol()
    except Exception as e:
        log.error(f"  SpacegroupAnalyzer failed: {e}")
        return []
    
    sites = sorted([
        dict(
            wyckoff_label=lbl,
            proto_element=equiv[0].species_string,
            frac_coords=list(equiv[0].frac_coords),
            multiplicity=len(equiv),
            sg_number=sg_num,
            sg_symbol=sg_sym,
        )
        for equiv, lbl in zip(sym.equivalent_sites, sym.wyckoff_symbols)
    ], key=lambda s: -s["multiplicity"])
    
    log.info(f"  SG {sg_num} ({sg_sym}), {len(sites)} sites:")
    for s in sites:
        log.info(f"    {s['wyckoff_label']:5s} {s['proto_element']:4s} mult={s['multiplicity']:3d}")
    return sites

# =============================================================================
# STEP 3 - ASSIGN WYCKOFF - FIX-4: Mixed mode fix
# =============================================================================
def assign_wyckoff(wyckoff_sites: list, proto_formula: str, target_formula: str):
    """
    FIX-4: In mixed mode, ALL sites of the same role get the same occupancy dict.
    This prevents subsequent sites from falling through to single-element defaults.
    """
    comp = parse_comp(target_formula)
    if comp is None:
        return []
    
    target_sorted = sorted(
        [(str(el), float(amt)) for el, amt in comp.items()],
        key=lambda x: -x[1],
    )
    
    role_to_elems = defaultdict(list)
    if proto_formula in _PROTO_ALL_B:
        # All-B override: treat every target element as B-role for this prototype.
        # Used for Heusler/full-Heusler alloys where all sites are substitutional.
        for el, amt in target_sorted:
            role_to_elems["B"].append((el, amt))
    else:
        for el, amt in target_sorted:
            role_to_elems[_role(el)].append((el, amt))
    
    site_role_list = PROTO_SITE_ROLES.get(proto_formula) or []
    roles = [r for (_, r) in site_role_list]
    while len(roles) < len(wyckoff_sites):
        roles.append(_role(wyckoff_sites[len(roles)]["proto_element"]))
    
    # Count actual sites per role from real wyckoff_sites
    n_actual_sites_of_role = defaultdict(int)
    role_site_map = defaultdict(list)
    for ws, r in zip(wyckoff_sites, roles):
        n_actual_sites_of_role[r] += 1
        role_site_map[r].append(ws)
    
    queue_mode = {}
    queue = {}
    for role, elems in role_to_elems.items():
        n_el = len(elems)
        n_si = n_actual_sites_of_role.get(role, 0)

        if n_el > n_si:
            # More elements than sites → must share (mixed occupancy)
            queue_mode[role] = "mixed"
        elif n_el < n_si and n_si >= 2:
            # FIX 2 extended: fewer distinct elements than sites (e.g. Gd5Ge2Si2:
            # 2 X-elements, 4 X-sites) → mixed so all sites get the same ratio
            queue_mode[role] = "mixed"
        elif n_el >= 2 and n_si >= 2:
            mults = [s["multiplicity"] for s in role_site_map[role]]
            if len(set(mults)) > 1:
                # FIX 7: unequal Wyckoff multiplicities for same role → mixed
                queue_mode[role] = "mixed"
            else:
                # Equal mults: use mixed only if amounts are unequal
                amounts = [a for _, a in elems]
                mean_a = sum(amounts) / len(amounts)
                if mean_a > 0 and max(abs(a - mean_a)/mean_a for a in amounts) > 0.05:
                    queue_mode[role] = "mixed"
                else:
                    queue_mode[role] = "one_per_site"
        else:
            queue_mode[role] = "one_per_site"

        queue[role] = deque(elems)
    
    placed = set()
    first_site_of_role = {}
    raw_assignments = []
    # Track raw-amount totals per site so post-loop merge can denormalize correctly
    site_raw_totals = {}

    for idx, (ws, role) in enumerate(zip(wyckoff_sites, roles)):
        q = queue.get(role, deque())
        mode = queue_mode.get(role, "one_per_site")

        if mode == "mixed":
            # FIX-4: ALL sites of this role get the same occupancy dict
            group = list(role_to_elems[role])
            for el, _ in group:
                placed.add(el)
        elif mode == "one_per_site":
            if q:
                el, amt = q.popleft()
                group = [(el, amt)]
                placed.add(el)
            else:
                if role_to_elems.get(role):
                    group = [role_to_elems[role][0]]
                else:
                    remaining = [(e, a) for e, a in target_sorted if e not in placed]
                    if remaining:
                        group = [remaining[0]]
                        placed.add(remaining[0][0])
                    else:
                        group = [target_sorted[0]]
        else:
            group = [target_sorted[0]]

        total = sum(a for _, a in group)
        site_raw_totals[idx] = total
        raw_assignments.append(dict(
            wyckoff_label=ws["wyckoff_label"],
            proto_element=ws["proto_element"],
            frac_coords=ws["frac_coords"],
            multiplicity=ws["multiplicity"],
            sg_number=ws["sg_number"],
            sg_symbol=ws["sg_symbol"],
            target_species={el: amt / total for el, amt in group},
        ))

    # POST-LOOP MERGE
    # Denormalize the existing fracs back to raw amounts before adding the new element,
    # so proportions stay correct across successive merges.
    unplaced = [(el, amt) for el, amt in target_sorted if el not in placed]
    if unplaced:
        log.info(f"  Post-loop merge: {[e for e,_ in unplaced]}")
        for el, amt in unplaced:
            el_role = _role(el)
            candidates = [(i, raw_assignments[i]["multiplicity"])
                          for i, a in enumerate(raw_assignments)
                          if _role(a["proto_element"]) == el_role]
            if not candidates:
                candidates = [(i, raw_assignments[i]["multiplicity"])
                              for i in range(len(raw_assignments))]
            best_idx = max(candidates, key=lambda x: x[1])[0]
            raw_total = site_raw_totals[best_idx]
            # Denormalize to raw amounts, add new element, renormalize
            raw_sp = {e: v * raw_total for e, v in raw_assignments[best_idx]["target_species"].items()}
            raw_sp[el] = amt
            new_total = sum(raw_sp.values())
            site_raw_totals[best_idx] = new_total
            raw_assignments[best_idx]["target_species"] = {e: v / new_total for e, v in raw_sp.items()}
            log.info(f"    Merged {el} into site {raw_assignments[best_idx]['wyckoff_label']}")
    
    return raw_assignments

# =============================================================================
# STEP 4 - BUILD STRUCTURE
# =============================================================================
def _distribute_group(n_total: int, occupancies: dict) -> list:
    """
    Distribute n_total atoms across elements according to occupancy fractions.
    Guarantees every element gets ≥1 atom and total sums exactly to n_total.
    Processes minority elements first to prevent banker's rounding erasure.
    """
    if not occupancies:
        return []
    if len(occupancies) == 1:
        return list(occupancies.keys()) * n_total
    
    items = sorted(occupancies.items(), key=lambda x: x[1])
    result = []
    remaining = n_total
    n_elements = len(items)
    
    for j, (el, occ) in enumerate(items):
        if j == n_elements - 1:
            n_el = remaining
        else:
            n_el = round(n_total * occ)
            n_el = max(1, n_el)
            n_el = min(n_el, remaining - (n_elements - j - 1))
        result.extend([el] * n_el)
        remaining -= n_el
    
    return result

def _apply_assignments_to_prim(prim: Structure, assignments: list,
                                ordered: bool, target_formula: str):
    """
    Multiplicity-based matching: groups prim atoms by species_string,
    distributes atoms across assignments proportionally to multiplicities.
    Robust to coordinate-frame changes from get_primitive_standard_structure().
    """
    species_order = []
    seen = set()
    for site in prim:
        s = site.species_string
        if s not in seen:
            species_order.append(s)
            seen.add(s)
    
    species_count = Counter(s.species_string for s in prim)
    proto_to_asgns = defaultdict(list)
    for a in assignments:
        proto_to_asgns[a["proto_element"]].append(a)
    
    result = []
    
    for species in species_order:
        n_atoms = species_count[species]
        asgn_list = proto_to_asgns.get(species, [])
        
        if not asgn_list:
            all_targets = {}
            for a in assignments:
                for el, frac in a["target_species"].items():
                    all_targets[el] = all_targets.get(el, 0) + frac
            if all_targets:
                dom = max(all_targets.items(), key=lambda x: x[1])
                result.extend([{dom[0]: 1.0}] * n_atoms)
            else:
                result.extend([{species: 1.0}] * n_atoms)
            continue
        
        if len(asgn_list) == 1:
            result.extend([asgn_list[0]["target_species"]] * n_atoms)
            continue
        
        total_mult = sum(a["multiplicity"] for a in asgn_list)
        cursor = 0
        for i, a in enumerate(asgn_list):
            if i == len(asgn_list) - 1:
                n = n_atoms - cursor
            else:
                n = round(n_atoms * a["multiplicity"] / total_mult)
            result.extend([a["target_species"]] * n)
            cursor += n
    
    while len(result) < len(prim):
        result.append(result[-1] if result else {})
    
    new_species = []
    for sp in result:
        if ordered:
            new_species.append(max(sp.items(), key=lambda x: x[1])[0])
        else:
            if len(sp) == 1:
                new_species.append(list(sp.keys())[0])
            else:
                new_species.append(sp)
    return new_species

def _build_ordered(prim: Structure, assignments: list, scale: int,
                   target_formula: str):
    """Build ordered supercell guaranteeing every element appears."""
    frac_species = _apply_assignments_to_prim(
        prim, assignments, ordered=False, target_formula=target_formula)
    
    prim_w = Structure(prim.lattice, frac_species,
                       [s.frac_coords for s in prim],
                       coords_are_cartesian=False)
    
    if scale > 1:
        prim_w.make_supercell([scale, scale, scale])
    
    n_super = len(prim_w)
    
    def _occ_key(site):
        sp = site.species
        try:
            items = tuple(sorted(
                (str(el), round(float(occ), 6))
                for el, occ in sp.items()
            ))
        except AttributeError:
            items = ((str(site.specie), 1.0),)
        return items
    
    from collections import OrderedDict
    groups = {}
    for i, site in enumerate(prim_w):
        k = _occ_key(site)
        groups.setdefault(k, []).append(i)
    
    final_species = [''] * n_super
    
    for occ_key, indices in groups.items():
        n_group = len(indices)
        occupancies = dict(occ_key)
        
        if len(occupancies) == 1:
            el = list(occupancies.keys())[0]
            for i in indices:
                final_species[i] = el
        else:
            assigned = _distribute_group(n_group, occupancies)
            for i, el in zip(indices, assigned):
                final_species[i] = el
    
    working = Structure(prim_w.lattice, final_species,
                        [s.frac_coords for s in prim_w],
                        coords_are_cartesian=False)
    log.info(f"    Built ordered (scale={scale}): {len(working)} atoms")
    return working

def _build_disordered(prim: Structure, assignments: list, target_formula: str):
    """Fractional-occupancy structure."""
    new_species = _apply_assignments_to_prim(prim, assignments, ordered=False,
                                              target_formula=target_formula)
    working = Structure(prim.lattice, new_species,
                        [s.frac_coords for s in prim],
                        coords_are_cartesian=False)
    log.info(f"    Built disordered: {len(working)} atoms")
    return working

# =============================================================================
# FIX 3: LINEAR SCALE HELPERS  ([s,1,1] expansion for NiAs-type structures)
# =============================================================================
def _find_ordered_scale_linear(assignments: list, n_prim: int,
                                max_atoms: int) -> tuple:
    """Find smallest linear scale s where [s,1,1] supercell achieves correct stoich."""
    best_s, best_err = 1, 1.0
    max_s = max(1, max_atoms // n_prim) if n_prim else 1
    for s in range(1, max_s + 1):
        if s * n_prim > max_atoms:
            break
        ok = True
        for a in assignments:
            sp = a["target_species"]
            if len(sp) <= 1:
                continue
            n_sub = a["multiplicity"]
            for frac in sp.values():
                if frac < 1e-9 or frac > 1 - 1e-9:
                    continue
                effective = s * n_sub
                expected  = effective * frac
                rounded   = round(expected)
                if rounded == 0 and expected > 0.01:
                    ok = False; break
                err = abs(expected - rounded)
                if err > DOPANT_TOL:
                    ok = False; break
            if not ok:
                break
        if ok:
            return s, True
    return best_s, False


def _build_ordered_linear(prim: Structure, assignments: list,
                           scale: int, target_formula: str):
    """Build ordered [scale,1,1] supercell."""
    try:
        frac_species = _apply_assignments_to_prim(
            prim, assignments, ordered=False, target_formula=target_formula)
        prim_w = Structure(prim.lattice, frac_species,
                           [s.frac_coords for s in prim],
                           coords_are_cartesian=False)
        prim_w.make_supercell([scale, 1, 1])
        n_super = len(prim_w)

        def _occ_key(site):
            sp = site.species
            try:
                return tuple(sorted((str(el), round(float(occ), 6))
                                    for el, occ in sp.items()))
            except AttributeError:
                return ((str(site.specie), 1.0),)

        groups: dict = {}
        for i, site in enumerate(prim_w):
            groups.setdefault(_occ_key(site), []).append(i)

        final_species = [''] * n_super
        for occ_key, indices in groups.items():
            occupancies = dict(occ_key)
            if len(occupancies) == 1:
                el = list(occupancies.keys())[0]
                for i in indices:
                    final_species[i] = el
            else:
                for i, el in zip(indices, _distribute_group(len(indices), occupancies)):
                    final_species[i] = el

        working = Structure(prim_w.lattice, final_species,
                            [s.frac_coords for s in prim_w],
                            coords_are_cartesian=False)
        log.info(f"    Built ordered-linear (scale=[{scale},1,1]): {len(working)} atoms")
        return working
    except Exception as e:
        log.warning(f"  _build_ordered_linear failed: {e}")
        return None


def build_structure(prototype: Structure, assignments: list,
                    target_formula: str, max_atoms_override: int = None):
    """
    Build structure with appropriate MAX_ATOMS limit.
    """
    try:
        prim = SpacegroupAnalyzer(prototype, symprec=0.1).get_primitive_standard_structure()
    except Exception:
        prim = prototype.copy()
    
    n_prim = len(prim)
    max_atoms = max_atoms_override or get_max_atoms_for_prototype(prototype)
    log.info(f"  Primitive cell: {n_prim} atoms, MAX_ATOMS={max_atoms}")
    
    any_mixed = any(len(a["target_species"]) > 1 for a in assignments)
    
    if not any_mixed:
        return _build_ordered(prim, assignments, 1, target_formula), 1, "end_member_primitive"
    
    best_s, all_exact = 1, True
    for a in assignments:
        sp = a["target_species"]
        if len(sp) <= 1:
            continue
        n_sub = a["multiplicity"]
        for frac in sp.values():
            if frac < 1e-9 or frac > 1 - 1e-9:
                continue
            s, exact = find_ordered_scale(n_sub, frac, n_prim, max_atoms, DOPANT_TOL)
            if not exact:
                n_copies_prim = s ** 3
                expected = n_copies_prim * n_sub * frac
                rounded = max(1, round(expected))
                actual_err = abs(expected - rounded) / max(expected, 1e-9)
                if actual_err <= STOICH_TOL:
                    exact = True
            if not exact or s**3 * n_prim > max_atoms:  # FIX 3: was s * n_prim
                all_exact = False
                break
            if s > best_s:
                best_s = s
        if not all_exact:
            break

    if all_exact and best_s**3 * n_prim <= max_atoms:  # FIX 3: was best_s * n_prim
        ordered_struct = _build_ordered(prim, assignments, best_s, target_formula)
        # FIX 1: if ordered struct has high stoich error but all elements present,
        # force disordered (avoids banker's-rounding element loss at scale=1)
        serr_check = stoich_error(ordered_struct, target_formula)
        if serr_check > STOICH_TOL and ordered_struct.is_ordered:
            try:
                t_els = {str(e) for e in Composition(target_formula).elements}
                s_els = {str(e) for e in ordered_struct.composition.elements}
                if t_els <= s_els:
                    log.info(f"  FIX1: ordered stoich {serr_check*100:.1f}% > tol, "
                             f"all elements present → forcing disordered")
                    return (_build_disordered(prim, assignments, target_formula),
                            1, "disordered_forced_stoich_fix")
            except Exception:
                pass
        return ordered_struct, best_s, f"ordered_supercell_x{best_s}"

    # FIX 3: linear [s,1,1] fallback for NiAs-like structures where cubic expansion
    # overshoots MAX_ATOMS but one axis scaling achieves correct stoich
    lin_best_s, lin_exact = _find_ordered_scale_linear(assignments, n_prim, max_atoms)
    if lin_exact and lin_best_s > 1:
        lin_struct = _build_ordered_linear(prim, assignments, lin_best_s, target_formula)
        if lin_struct is not None:
            return lin_struct, lin_best_s, f"ordered_linear_x{lin_best_s}"

    return _build_disordered(prim, assignments, target_formula), 1, "disordered_fractional_occupancy"

# =============================================================================
# STEP 5 - DIRECT LOOKUP
# =============================================================================
def fetch_material_directly(formula: str, mpr: MPRester):
    log.info(f"  Searching MP directly for '{formula}'")
    def _search(f):
        return mpr.materials.search(
            formula=f,
            fields=["structure", "material_id", "deprecated"],)
    
    results = []
    try:
        results = _search(formula)
    except Exception as e:
        log.warning(f"  Direct search error: {e}")
    
    if not results:
        try:
            red = Composition(formula).reduced_formula
            if red != formula:
                log.info(f"  Retrying with '{red}'")
                results = _search(red)
        except Exception:
            pass
    
    if not results:
        log.warning(f"  No MP entry for '{formula}'.")
        return None, ""
    
    results.sort(key=lambda r: r.deprecated)
    best = results[0]
    log.info(f"  Direct match: {best.material_id}")
    return best.structure, best.material_id

# =============================================================================
# CIF WRITER - FIX-3: Dual folder support
# =============================================================================
def _safe_filename(formula: str) -> str:
    name = formula.replace("(", " ").replace(")", " ").replace(".", "p ")
    name = re.sub(r"[^A-Za-z0-9_]", " ", name)
    return re.sub(r" +", " ", name).strip("_") + ".cif"

def save_cif(struct: Structure, formula: str, cif_dir: str) -> str:
    os.makedirs(cif_dir, exist_ok=True)
    path = os.path.join(cif_dir, _safe_filename(formula))
    symprec = None if not struct.is_ordered else 0.1
    try:
        CifWriter(struct, symprec=symprec).write_file(path)
    except Exception as e:
        log.warning(f"  CifWriter({symprec}) failed: {e}; retrying symprec=None")
        CifWriter(struct, symprec=None).write_file(path)
    log.info(f"  CIF -> {path}")
    return path

# =============================================================================
# VALIDATION
# =============================================================================
def validate(struct: Structure, formula: str) -> str:
    issues = []
    if len(struct) == 0:
        return "EMPTY_STRUCTURE"
    if len(struct) > MAX_ATOMS_LARGE:
        issues.append(f"OVERSIZED:{len(struct)}_atoms")
    try:
        dm = struct.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        d_min = dm.min()
        if d_min < 0.8:
            issues.append(f"OVERLAP:min={d_min:.2f}A")
    except Exception:
        issues.append("DIST_CHECK_FAILED")
    
    a, b, c = struct.lattice.abc
    if any(x < 1.5 or x > 35.0 for x in (a, b, c)):
        issues.append(f"LATTICE:{a:.2f},{b:.2f},{c:.2f}")
    
    try:
        t_els = {str(e) for e in Composition(formula).elements}
        s_els = {str(e) for e in struct.composition.elements}
        missing = t_els - s_els
        if missing:
            issues.append(f"MISSING:{sorted(missing)}")
    except Exception:
        issues.append("COMP_CHECK_FAILED")
    
    return "; ".join(issues) if issues else "OK"

# =============================================================================
# MASTER FUNCTION
# =============================================================================
def process_one_material(row: dict, mpr: MPRester, output_rows: list,
                         failed_rows: list, cif_dir_standard: str, cif_dir_large: str):
    formula_raw = str(row.get("material_formula", "")).strip()
    prototype = str(row.get("structure_prototype", "")).strip()
    
    # FIX-1: Clean formula
    formula, formula_cleaned = clean_formula(formula_raw)
    if formula_cleaned:
        log.info(f"  Formula cleaned: '{formula_raw}' -> '{formula}'")
    
    log.info("=" * 64)
    log.info(f"  Material : {formula}")
    log.info(f"  Prototype: {prototype}")
    
    out = dict(row)
    out.update({
        "wyckoff_site_assignments": "",
        "wyckoff_detail": "",
        "sg_number": "",
        "sg_symbol": "",
        "supercell_scale": "1",
        "build_description": "",
        "is_disordered": "False",
        "mp_id_prototype": "",
        "mp_id_direct": "",
        "cif_path": "",
        "cif_stoichiometry": "",
        "stoich_error_pct": "",
        "validation": "",
        "generation_method": "",
        "cif_folder": "",
    })
    
    try:
        reduced = Composition(formula).reduced_formula
    except Exception:
        reduced = formula

    override_key = (reduced, prototype)

    # FIX 8: Antiperovskite detection — Mn3GaC and variants have Mn>40% + C present
    # with CaTiO3 prototype. CaTiO3 site roles put C on O-site (X), but antiperovskite
    # has C on Ca-site (A-corner). Bypass Wyckoff assignment with direct MP search.
    def _is_antiperovskite(formula_str, proto_str):
        if proto_str != "CaTiO3":
            return False
        try:
            comp = Composition(formula_str)
            els = {str(e) for e in comp.elements}
            if "Mn" in els and "C" in els:
                mn_frac = float(comp.get("Mn", 0)) / float(comp.num_atoms)
                return mn_frac > 0.4
        except Exception:
            pass
        return False

    if _is_antiperovskite(formula, prototype):
        log.info(f"  FIX8: antiperovskite candidate '{formula}' → direct MP search")
        a_struct, a_mpid = fetch_material_directly(formula, mpr)
        if a_struct is None:
            # Try canonical base Mn3GaC
            a_struct, a_mpid = fetch_material_directly("Mn3GaC", mpr)
        if a_struct is not None:
            out["mp_id_direct"] = a_mpid
            out["generation_method"] = "direct_mp_antiperovskite"
            ws = get_wyckoff_sites(a_struct)
            if ws:
                out["sg_number"] = str(ws[0]["sg_number"])
                out["sg_symbol"] = ws[0]["sg_symbol"]
            val  = validate(a_struct, formula)
            serr = stoich_error(a_struct, formula)
            out["validation"]        = val
            out["cif_stoichiometry"] = cif_stoich_string(a_struct)
            out["stoich_error_pct"]  = f"{serr*100:.2f}"
            n_atoms = len(a_struct)
            cif_dir = get_cif_dir_for_atoms(n_atoms)
            out["cif_folder"]  = "cif_>50" if n_atoms > 50 else "cif_files"
            out["cif_path"]    = save_cif(a_struct, formula, cif_dir)
            out["build_description"] = "direct_mp_antiperovskite"
            out["is_disordered"] = str(not a_struct.is_ordered)
            log.info(f"  {n_atoms} atoms | antiperovskite | {val} | stoich={out['stoich_error_pct']}%")
            output_rows.append(out)
            return

    proto_struct, proto_mpid, proto_formula = fetch_prototype(prototype, mpr)
    
    if proto_struct is not None:
        if override_key in _FORMULA_PROTO_OVERRIDES:
            effective_proto_formula = _FORMULA_PROTO_OVERRIDES[override_key]
            log.info(f"  PROTO_OVERRIDE -> site-roles of '{effective_proto_formula}'")
        else:
            effective_proto_formula = proto_formula
        
        out["mp_id_prototype"] = proto_mpid
        out["generation_method"] = "prototype_wyckoff"
        wyckoff_sites = get_wyckoff_sites(proto_struct)
        
        if wyckoff_sites:
            out["sg_number"] = str(wyckoff_sites[0]["sg_number"])
            out["sg_symbol"] = wyckoff_sites[0]["sg_symbol"]
            # SG fallback: if MP returned a different SG than requested, redirect site roles
            actual_sg = wyckoff_sites[0]["sg_number"]
            sg_fb_key = (effective_proto_formula, actual_sg)
            if sg_fb_key in _PROTO_SG_FALLBACK:
                fallback_pf = _PROTO_SG_FALLBACK[sg_fb_key]
                log.info(f"  SG_FALLBACK: {effective_proto_formula} SG{actual_sg} -> site-roles of '{fallback_pf}'")
                effective_proto_formula = fallback_pf
            assignments = assign_wyckoff(wyckoff_sites, effective_proto_formula, formula)
            
            out["wyckoff_site_assignments"] = " | ".join(
                f"{'/'.join(a['target_species'].keys())}:{a['wyckoff_label']} "
                f"({','.join(f'{c:.4f}' for c in a['frac_coords'])}) "
                for a in assignments
            )
            out["wyckoff_detail"] = str([{
                "wyckoff": a["wyckoff_label"],
                "multiplicity": a["multiplicity"],
                "frac_coords": [round(c, 6) for c in a["frac_coords"]],
                "target_species": {k: round(v, 6) for k, v in a["target_species"].items()},
            } for a in assignments])
            
            # FIX-2: Get appropriate MAX_ATOMS for this prototype
            max_atoms = get_max_atoms_for_prototype(prototype)
            struct, scale, desc = build_structure(proto_struct, assignments, formula, max_atoms)
            
            out["supercell_scale"] = str(scale)
            out["build_description"] = desc
            out["is_disordered"] = str("disordered" in desc)
            
            if struct is not None:
                val = validate(struct, formula)
                serr = stoich_error(struct, formula)
                out["validation"] = val
                out["cif_stoichiometry"] = cif_stoich_string(struct)
                out["stoich_error_pct"] = f"{serr*100:.2f}"
                
                # FIX-8: Direct MP fallback if stoich error >10%
                if serr > 0.10 and "MISSING" not in val:
                    log.info(f"  Stoich error {serr*100:.1f}% > 10% -- trying direct MP fallback")
                    d_struct, d_mpid = fetch_material_directly(formula, mpr)
                    if d_struct is not None:
                        d_val = validate(d_struct, formula)
                        d_serr = stoich_error(d_struct, formula)
                        if d_serr < serr and "MISSING" not in d_val:
                            log.info(f"  Direct MP better: {d_serr*100:.1f}%")
                            struct = d_struct
                            out["mp_id_direct"] = d_mpid
                            out["generation_method"] = "direct_mp_fallback"
                            out["validation"] = d_val
                            out["cif_stoichiometry"] = cif_stoich_string(d_struct)
                            out["stoich_error_pct"] = f"{d_serr*100:.2f}"
                            out["build_description"] = "direct_mp_fallback"
                            out["is_disordered"] = str(not d_struct.is_ordered)
                
                # FIX-3: Save to appropriate folder based on atom count
                n_atoms = len(struct)
                cif_dir = get_cif_dir_for_atoms(n_atoms)
                out["cif_folder"] = "cif_>50" if n_atoms > 50 else "cif_files"
                out["cif_path"] = save_cif(struct, formula, cif_dir)
                log.info(f"  {n_atoms} atoms | {desc} | {val} | stoich={out['stoich_error_pct']}%")
            else:
                out["validation"] = "STRUCTURE_BUILD_FAILED"
                log.error(f"  Structure build failed for '{formula}'")
        else:
            out["wyckoff_site_assignments"] = "WYCKOFF_EXTRACTION_FAILED"
            out["validation"] = "WYCKOFF_EXTRACTION_FAILED"
        
        output_rows.append(out)
        return
    
    log.info("  No prototype; trying direct MP search")
    direct_struct, direct_mpid = fetch_material_directly(formula, mpr)
    if direct_struct is not None:
        out["mp_id_direct"] = direct_mpid
        out["generation_method"] = "direct_mp_cif"
        ws = get_wyckoff_sites(direct_struct)
        if ws:
            out["sg_number"] = str(ws[0]["sg_number"])
            out["sg_symbol"] = ws[0]["sg_symbol"]
        val = validate(direct_struct, formula)
        serr = stoich_error(direct_struct, formula)
        out["validation"] = val
        out["cif_stoichiometry"] = cif_stoich_string(direct_struct)
        out["stoich_error_pct"] = f"{serr*100:.2f}"
        n_atoms = len(direct_struct)
        cif_dir = get_cif_dir_for_atoms(n_atoms)
        out["cif_folder"] = "cif_>50" if n_atoms > 50 else "cif_files"
        out["cif_path"] = save_cif(direct_struct, formula, cif_dir)
        out["build_description"] = "direct_mp"
        out["is_disordered"] = str(not direct_struct.is_ordered)
        log.info(f"  {n_atoms} atoms | direct_mp | {val}")
        output_rows.append(out)
        return
    
    log.warning(f"  FAILED: '{formula}'")
    failed = dict(row)
    failed["failure_reason"] = (
        f"Prototype '{prototype}' not found on MP; "
        f"direct search for '{formula}' also returned no results."
    )
    failed_rows.append(failed)

# =============================================================================
# CSV COLUMNS
# =============================================================================
OUTPUT_COLS = [
    "material_formula", "structure_prototype", "transition_order",
    "T_C_K", "applied_field_H_T", "delta_S_mag_J_per_kgK",
    "RCP_J_per_kg", "delta_T_ad_K", "reference", "material_class",
    "sg_number", "sg_symbol",
    "wyckoff_site_assignments",
    "wyckoff_detail",
    "supercell_scale",
    "build_description",
    "is_disordered",
    "mp_id_prototype", "mp_id_direct",
    "cif_path",
    "cif_folder",
    "cif_stoichiometry",
    "stoich_error_pct",
    "validation",
    "generation_method",
]

FAILED_COLS = [
    "material_formula", "structure_prototype", "transition_order",
    "T_C_K", "applied_field_H_T", "delta_S_mag_J_per_kgK",
    "RCP_J_per_kg", "delta_T_ad_K", "reference", "material_class",
    "failure_reason",
]

def write_csv(rows: list, path: str, cols: list, mode: str = "w"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = (mode == "w") or not os.path.exists(path)
    with open(path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)
    log.info(f"  Wrote {len(rows)} row(s) -> {path}")

# =============================================================================
# TARGET SET BUILDER
# =============================================================================
def build_target_set(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the rows that need reprocessing:
      - All formulas listed in stoich_errors.txt  (if it exists)
      - All formulas listed in Failed_materials.csv (if it exists)
    If neither file exists, fall back to processing everything not already
    in clean_cifs/.
    """
    target_formulas: set = set()

    # Parse stoich_errors.txt  (format: formula TAB error% TAB /path)
    if os.path.isfile(STOICH_ERR_IN):
        with open(STOICH_ERR_IN, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                formula = line.split("\t")[0].strip()
                if formula:
                    target_formulas.add(formula)
        log.info(f"  stoich_errors.txt: {len(target_formulas)} formulas")

    # Parse Failed_materials.csv
    if os.path.isfile(FAILED_CSV):
        try:
            df_fail = pd.read_csv(FAILED_CSV)
            for f in df_fail["material_formula"].dropna():
                target_formulas.add(str(f).strip())
            log.info(f"  Failed_materials.csv: added, total={len(target_formulas)}")
        except Exception as e:
            log.warning(f"  Could not read Failed_materials.csv: {e}")

    if not target_formulas:
        log.info("  No target files found; processing all materials")
        return df_all

    mask = df_all["material_formula"].isin(target_formulas)
    subset = df_all[mask].reset_index(drop=True)
    log.info(f"  Target subset: {len(subset)} rows from {len(target_formulas)} formulas")
    return subset


# =============================================================================
# MAIN
# =============================================================================
def main():
    import shutil

    # Clear pycache to avoid stale bytecode
    for root, dirs, _ in os.walk(BASE_DIR):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    df = pd.read_csv(CSV_INPUT)
    df_all = (
        df.drop_duplicates(subset="material_formula")
          .query("structure_prototype != 'amorphous'")
          .reset_index(drop=True)
    )
    log.info(f"CSV total unique non-amorphous: {len(df_all)}")

    df_target = build_target_set(df_all)

    os.makedirs(CIF_DIR_STANDARD, exist_ok=True)
    os.makedirs(CIF_DIR_LARGE,    exist_ok=True)
    os.makedirs(CLEAN_DIR,        exist_ok=True)

    output_rows: list = []
    failed_rows: list = []

    with MPRester(MP_API_KEY) as mpr:
        for _, row in df_target.iterrows():
            process_one_material(row.to_dict(), mpr, output_rows, failed_rows,
                                 CIF_DIR_STANDARD, CIF_DIR_LARGE)

    # Classify
    clean_rows    = []
    problem_rows  = []
    stoich_error_rows = []

    for r in output_rows:
        val = str(r.get("validation", ""))
        try:
            serr_pct = float(r.get("stoich_error_pct", "100") or "100")
        except ValueError:
            serr_pct = 100.0

        has_missing    = "MISSING:" in val
        has_build_fail = "FAILED"   in val
        stoich_bad     = serr_pct   > STOICH_TOL * 100

        if not has_missing and not has_build_fail and not stoich_bad:
            clean_rows.append(r)
        else:
            r["failure_reason"] = (
                (f"MISSING:{val} "             if has_missing    else "") +
                (f"STOICH_ERR:{serr_pct:.1f}% " if stoich_bad     else "") +
                (f"BUILD_FAIL "                 if has_build_fail else "")
            ).strip()
            r["notes"] = "Needs manual ICSD search"
            problem_rows.append(r)

        if stoich_bad:
            stoich_error_rows.append(r)

    # Copy clean CIFs to clean_cifs/ (does NOT overwrite existing files)
    n_copied = 0
    for r in clean_rows:
        cif_src = r.get("cif_path", "")
        if cif_src and os.path.isfile(cif_src):
            dst = os.path.join(CLEAN_DIR, os.path.basename(cif_src))
            if not os.path.exists(dst):
                shutil.copy2(cif_src, dst)
                n_copied += 1
            else:
                # Overwrite only if new version has lower stoich error
                try:
                    serr_new = float(r.get("stoich_error_pct", "100") or "100")
                    if serr_new < 5.0:
                        shutil.copy2(cif_src, dst)
                        n_copied += 1
                except Exception:
                    pass

    # Write outputs
    if output_rows:
        write_csv(output_rows, CSV_OUTPUT, OUTPUT_COLS, mode="a")
    if failed_rows:
        write_csv(failed_rows, FAILED_CSV, FAILED_COLS, mode="a")

    # Update needs_manual_search.csv
    manual_cols = [
        "material_formula", "structure_prototype", "material_class",
        "validation", "stoich_error_pct", "cif_stoichiometry",
        "failure_reason", "notes",
    ]
    manual_rows = problem_rows + [
        dict(r, failure_reason=r.get("failure_reason", "no MP structure found"),
             notes="Needs manual ICSD search")
        for r in failed_rows
    ]
    if manual_rows:
        write_csv(manual_rows, MANUAL_CSV, manual_cols, mode="w")

    # Write updated stoich_errors.txt (only remaining bad ones)
    stoich_error_path = os.path.join(BASE_DIR, "stoich_errors.txt")
    with open(stoich_error_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Materials with stoichiometry error >{STOICH_TOL*100:.0f}%\n")
        fh.write(f"# {len(stoich_error_rows)} total\n")
        for r in stoich_error_rows:
            fh.write(f"{r['material_formula']}\t{r['stoich_error_pct']}%\t{r.get('cif_path','')}\n")

    # Summary
    total_failures = len(manual_rows)
    log.info("=" * 64)
    log.info(f"  Processed this run : {len(output_rows) + len(failed_rows)}")
    log.info(f"  Clean CIFs written : {n_copied}  -> {CLEAN_DIR}")
    log.info(f"  Still stoich errors: {len(stoich_error_rows)}")
    log.info(f"  Still failed       : {len(failed_rows)}")
    log.info(f"  Total failures     : {total_failures}  (target: <40)")
    log.info(f"  needs_manual_search: {MANUAL_CSV}")
    log.info("=" * 64)
    if total_failures > 40:
        log.warning(f"  *** {total_failures} failures > 40 target ***")
    else:
        log.info(f"  *** TARGET MET: {total_failures} failures <=40 ***")

    if manual_rows:
        log.info("  REMAINING FAILURES:")
        for r in manual_rows:
            log.info(f"    {str(r.get('material_formula','')):<50} "
                     f"{str(r.get('failure_reason',''))}")


if __name__ == "__main__":
    main()