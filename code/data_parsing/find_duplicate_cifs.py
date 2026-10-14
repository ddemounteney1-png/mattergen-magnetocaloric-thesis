#!/usr/bin/env python3
"""
find_duplicate_cifs.py — Detect structurally identical CIFs in the training set.

Two CIFs are considered duplicates if they represent the same crystal structure
(same composition, same space group, same lattice parameters within tolerance,
same atom count). Uses a fast hash-first approach, then pymatgen StructureMatcher
for confirmation.

Usage:
    python3 find_duplicate_cifs.py                     # check cif_files/ + clean_cifs/
    python3 find_duplicate_cifs.py --folder cif_files  # one folder only
    python3 find_duplicate_cifs.py --csv duplicates.csv

Output:
    duplicates.csv  — groups of identical CIFs with training values for ranking
    Prints recommended action: which CIF to keep (higher training value) and which to drop.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PREPROCESSING_WORK  # noqa: E402


import csv
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

BASE = PREPROCESSING_WORK


def fast_hash(cif_path: Path) -> str:
    """Quick structural fingerprint: formula + SG + sorted lattice params (rounded)."""
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        s = Structure.from_file(str(cif_path), occupancy_tolerance=2.0)
        sg = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
        latt = s.lattice
        # Round lattice params to 3 decimal places
        abc = tuple(round(x, 3) for x in sorted([latt.a, latt.b, latt.c]))
        ang = tuple(round(x, 2) for x in sorted([latt.alpha, latt.beta, latt.gamma]))
        formula = s.composition.reduced_formula
        n = len(s)
        key = f"{formula}|{sg}|{n}|{abc}|{ang}"
        return key
    except Exception as e:
        return f"ERROR:{cif_path.name}:{e}"


def confirm_duplicate(cif_a: Path, cif_b: Path) -> bool:
    """Use pymatgen StructureMatcher to confirm two CIFs are structurally equivalent."""
    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.structure_matcher import StructureMatcher
        s1 = Structure.from_file(str(cif_a), occupancy_tolerance=2.0)
        s2 = Structure.from_file(str(cif_b), occupancy_tolerance=2.0)
        sm = StructureMatcher(ltol=0.1, stol=0.2, angle_tol=5.0,
                               primitive_cell=True, scale=True,
                               attempt_supercell=False)
        return sm.fit(s1, s2)
    except Exception:
        return False


def load_training_values(csv_path: Path) -> dict[str, float]:
    """Load training-value metrics from magnetocaloric_materials.csv.

    Returns dict: formula → composite_value (higher = more important to keep).
    Uses |dS| (magnetocaloric entropy change) as primary metric, RCP as secondary.
    """
    values = {}
    if not csv_path.exists():
        return values
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            formula = row.get("formula", row.get("material_formula", ""))
            try:
                ds = abs(float(row.get("delta_S_mag_J_per_kgK", 0) or 0))
                rcp = abs(float(row.get("RCP_J_per_kg", 0) or 0))
                val = ds * 10 + rcp  # weighted composite: dS more important
                values[formula] = max(values.get(formula, 0), val)
            except (ValueError, TypeError):
                pass
    return values


def find_duplicates(cif_paths: list[Path]) -> list[list[Path]]:
    """Return groups of structurally identical CIFs."""
    print(f"Computing structural fingerprints for {len(cif_paths)} CIFs...")
    by_hash = defaultdict(list)
    for i, cf in enumerate(cif_paths):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cif_paths)}...", end="\r")
        h = fast_hash(cf)
        by_hash[h].append(cf)

    # Groups with >1 CIF — candidates for duplicates
    candidates = [group for group in by_hash.values() if len(group) > 1]
    print(f"\n  Fast hash: {len(candidates)} candidate groups")

    # Confirm with StructureMatcher
    confirmed_groups = []
    for group in candidates:
        # Build confirmed clusters within each hash group
        confirmed = [group[0]]
        for cf in group[1:]:
            is_dup = False
            for ref in confirmed:
                if confirm_duplicate(ref, cf):
                    is_dup = True
                    break
            if is_dup:
                confirmed.append(cf)
        if len(confirmed) > 1:
            confirmed_groups.append(confirmed)

    return confirmed_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", nargs="*",
                        default=["cif_files", "cif_>50"],
                        help="Folders to scan (default: cif_files + cif_>50)")
    parser.add_argument("--csv", default="duplicates.csv")
    parser.add_argument("--confirm", action="store_true",
                        help="Use StructureMatcher for confirmation (slower but more accurate)")
    args = parser.parse_args()

    # Collect CIF files (deduplicate by name to avoid double-counting)
    cif_paths = []
    seen_names = set()
    for folder in args.folder:
        d = BASE / folder
        if d.exists():
            for cf in sorted(d.glob("*.cif")):
                if cf.name not in seen_names and not cf.name.endswith(".Identifier"):
                    seen_names.add(cf.name)
                    cif_paths.append(cf)

    print(f"Scanning {len(cif_paths)} unique CIF files from: {args.folder}")

    # Load training values
    training_vals = load_training_values(BASE / "magnetocaloric_materials.csv")
    print(f"Loaded training values for {len(training_vals)} materials")

    # Find duplicates
    dup_groups = find_duplicates(cif_paths)
    print(f"\nFound {len(dup_groups)} duplicate groups ({sum(len(g) for g in dup_groups)} CIFs total)")

    if not dup_groups:
        print("No duplicates found. Dataset is clean.")
        return

    # Report and write CSV
    csv_rows = []
    print("\n" + "="*80)
    print("DUPLICATE GROUPS (keep highest training value, drop others):")
    print("="*80)

    for i, group in enumerate(dup_groups):
        print(f"\nGroup {i+1} ({len(group)} files):")
        group_data = []
        for cf in group:
            # Try to find matching target formula
            stem = cf.stem.replace("_sqs", "").replace("p", ".").replace("_", " ")
            # Find training value by trying different formula variations
            val = 0.0
            matched_formula = ""
            for formula, tv in training_vals.items():
                f_norm = formula.lower().replace(" ", "").replace("_", "")
                s_norm = stem.lower().replace(" ", "")
                if f_norm == s_norm or formula in cf.name:
                    val = tv
                    matched_formula = formula
                    break
            group_data.append((cf, val, matched_formula))
            print(f"  {'→KEEP' if False else '      '} {cf.name:55s}  val={val:.1f}")

        # Rank by training value (highest = keep)
        group_data.sort(key=lambda x: -x[1])
        keep_cf, keep_val, keep_formula = group_data[0]
        print(f"\n  Recommendation:")
        print(f"    KEEP: {keep_cf.name} (val={keep_val:.1f})")
        for cf, val, formula in group_data[1:]:
            print(f"    DROP: {cf.name} (val={val:.1f})")

        for rank, (cf, val, formula) in enumerate(group_data):
            csv_rows.append({
                "group": i + 1,
                "rank": rank + 1,
                "action": "KEEP" if rank == 0 else "DROP",
                "filename": cf.name,
                "folder": cf.parent.name,
                "path": str(cf),
                "formula": formula,
                "training_value": val,
            })

    # Save CSV
    csv_path = BASE / args.csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group","rank","action","filename",
                                                "folder","formula","training_value","path"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nReport saved → {csv_path}")

    # Count drops
    drops = [r for r in csv_rows if r["action"] == "DROP"]
    print(f"\nTotal CIFs to drop: {len(drops)}")


if __name__ == "__main__":
    main()
