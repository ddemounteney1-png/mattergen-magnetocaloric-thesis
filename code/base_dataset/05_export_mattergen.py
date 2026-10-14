#!/usr/bin/env python3
"""Stage 5: Export dataset in MatterGen format."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import BASE_DATASET_WORK  # noqa: E402


import csv
import pickle
import logging
import random
import datetime
from pathlib import Path
from io import StringIO
import os

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

BASE = BASE_DATASET_WORK
CHECKPOINTS = BASE / "checkpoints"
LOGS = BASE / "logs"
OUTPUT = BASE / "merged_dataset"

LOG_FILE = LOGS / "export.log"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def notify(message, title="MatterGen Pipeline - Export"):
    _NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
    if not _NTFY_TOPIC:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{_NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": title},
            timeout=10,
        )
    except Exception:
        pass


def niggli_reduce_structure(structure):
    """Apply Niggli reduction via ASE. Returns reduced structure or original on failure."""
    try:
        import ase.build
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(structure)
        ase.build.niggli_reduce(atoms)
        return AseAtomsAdaptor.get_structure(atoms), True
    except Exception as e:
        logger.warning(f"Niggli reduction failed: {e}")
        return structure, False


def get_primitive_structure(structure):
    """Get primitive standard structure."""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure)
        prim = sga.get_primitive_standard_structure()
        sg_num = sga.get_space_group_number()
        crystal_system = sga.get_crystal_system()
        return prim, sg_num, crystal_system
    except Exception as e:
        logger.warning(f"SpacegroupAnalyzer failed: {e}")
        return structure, None, None


def structure_to_cif_string(structure):
    """Convert structure to CIF string."""
    from pymatgen.io.cif import CifWriter
    return str(CifWriter(structure))


def get_crystal_system(sg_num):
    """Get crystal system from space group number."""
    if sg_num is None:
        return "Unknown"
    if sg_num <= 2:
        return "Triclinic"
    if sg_num <= 15:
        return "Monoclinic"
    if sg_num <= 74:
        return "Orthorhombic"
    if sg_num <= 142:
        return "Tetragonal"
    if sg_num <= 167:
        return "Trigonal"
    if sg_num <= 194:
        return "Hexagonal"
    return "Cubic"


def process_entry(entry):
    """Standardise one entry for export. Returns enriched dict."""
    structure = entry["structure"]

    # Get primitive standard structure with spacegroup info
    prim, sg_num, crystal_system = get_primitive_structure(structure)

    # Use spacegroup from entry if not available from SpacegroupAnalyzer
    if sg_num is None:
        sg_num = entry.get("spacegroup_number")
    if crystal_system is None:
        crystal_system = get_crystal_system(sg_num)

    # Niggli reduce
    prim_reduced, niggli_ok = niggli_reduce_structure(prim)

    # Generate CIF string
    try:
        cif_string = structure_to_cif_string(prim_reduced)
    except Exception as e:
        logger.error(f"CIF generation failed for {entry['source_id']}: {e}")
        return None

    lat = prim_reduced.lattice
    elements_str = str(sorted([str(el) for el in prim_reduced.composition.elements]))

    return {
        "source_id": entry["source_id"],
        "source": entry["source"],
        "structure": prim_reduced,
        "material_id": entry["source_id"],
        "pretty_formula": prim_reduced.composition.reduced_formula,
        "elements": elements_str,
        "spacegroup_number": sg_num,
        "crystal_system": crystal_system,
        "num_atoms": len(prim_reduced),
        "cif": cif_string,
        "formation_energy_per_atom": entry.get("formation_energy_per_atom"),
        "dft_band_gap": entry.get("band_gap"),
        "e_above_hull": entry.get("energy_above_hull"),
        "is_stable": entry.get("is_stable"),
        "volume": lat.volume,
        "density": entry.get("density"),
        "cod_id": entry.get("cod_id"),
        "journal": entry.get("journal"),
        "year": entry.get("year"),
        "doi": entry.get("doi"),
        "authors": entry.get("authors"),
        "r_factor": entry.get("r_factor"),
        "cell_a": lat.a, "cell_b": lat.b, "cell_c": lat.c,
        "cell_alpha": lat.alpha, "cell_beta": lat.beta, "cell_gamma": lat.gamma,
        "cell_volume": lat.volume,
        # Pass through any NEMAD properties
        **{k: v for k, v in entry.items() if k.startswith("nemad_")},
    }


def stratified_split(entries, train_frac=0.80, val_frac=0.10, seed=42):
    """Split entries with stratification by crystal system."""
    random.seed(seed)

    # Group by crystal system
    by_system = {}
    for e in entries:
        cs = e.get("crystal_system", "Unknown")
        if cs not in by_system:
            by_system[cs] = []
        by_system[cs].append(e)

    train, val, test = [], [], []

    for cs, group in by_system.items():
        random.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_frac))
        n_val = max(0, int(n * val_frac))
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    # Shuffle splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_mattergen_csv(entries, path, global_index_offset=0):
    """Write entries as a MatterGen-compatible CSV."""
    rows = []
    for i, e in enumerate(entries):
        rows.append({
            "": i + global_index_offset,
            "material_id": e["material_id"],
            "formation_energy_per_atom": e.get("formation_energy_per_atom", ""),
            "dft_band_gap": e.get("dft_band_gap", ""),
            "pretty_formula": e["pretty_formula"],
            "e_above_hull": e.get("e_above_hull", ""),
            "elements": e["elements"],
            "cif": e["cif"],
            "spacegroup_number": e.get("spacegroup_number", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    return df


def verify_csv(path, n_sample=5):
    """Re-read the CSV and verify CIF integrity."""
    from pymatgen.io.cif import CifParser
    df = pd.read_csv(path)
    assert len(df) > 0, f"Empty CSV: {path}"

    sample_indices = random.sample(range(len(df)), min(n_sample, len(df)))
    for idx in sample_indices:
        cif_str = df.iloc[idx]["cif"]
        try:
            parser = CifParser(StringIO(cif_str))
            structures = parser.parse_structures()
            assert len(structures) > 0
            assert len(structures[0]) <= 51
        except Exception as e:
            logger.error(f"CIF verification failed in {path} row {idx}: {e}")
            return False, str(e)

    return True, f"{len(df)} rows, {n_sample} CIFs verified"


def main():
    # Load merged structures
    merged_pkl = CHECKPOINTS / "merged_structures.pkl"
    if not merged_pkl.exists():
        print("ERROR: merged_structures.pkl not found. Run 04_merge_dedup.py first.")
        raise SystemExit(1)

    print("Loading merged structures...")
    with open(merged_pkl, "rb") as f:
        merged = pickle.load(f)
    print(f"Loaded {len(merged)} structures")

    # Process all entries
    print("Processing entries (primitive + Niggli + CIF)...")
    processed = []
    n_failed = 0
    for entry in tqdm(merged, desc="Processing", unit="struct"):
        result = process_entry(entry)
        if result is not None:
            processed.append(result)
        else:
            n_failed += 1
            logger.error(f"process_entry failed for {entry.get('source_id','?')}")

    print(f"Processed: {len(processed)} | Failed: {n_failed}")

    # Split
    print("Splitting into train/val/test (80/10/10)...")
    train, val, test = stratified_split(processed)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Write MatterGen CSVs
    print("Writing train.csv...")
    write_mattergen_csv(train, OUTPUT / "train.csv")

    print("Writing val.csv...")
    write_mattergen_csv(val, OUTPUT / "val.csv")

    print("Writing test.csv...")
    write_mattergen_csv(test, OUTPUT / "test.csv")

    # Write full metadata CSV
    print("Writing full_metadata.csv...")
    all_rows = []
    for e in processed:
        row = {k: v for k, v in e.items() if k != "structure"}
        all_rows.append(row)
    df_meta = pd.DataFrame(all_rows)
    df_meta.to_csv(OUTPUT / "full_metadata.csv", index=False, quoting=csv.QUOTE_ALL)

    # Write extxyz
    print("Writing combined_dataset.extxyz...")
    from ase.io import write as ase_write
    from pymatgen.io.ase import AseAtomsAdaptor
    all_atoms = []
    for e in tqdm(processed, desc="Building extxyz", unit="struct"):
        try:
            atoms = AseAtomsAdaptor.get_atoms(e["structure"])
            atoms.info["material_id"] = e["material_id"]
            atoms.info["formula"] = e["pretty_formula"]
            atoms.info["spacegroup"] = e.get("spacegroup_number", "")
            atoms.info["source"] = e["source"]
            atoms.pbc = [True, True, True]
            all_atoms.append(atoms)
        except Exception as ex:
            logger.warning(f"ASE conversion failed for {e['material_id']}: {ex}")

    ase_write(str(OUTPUT / "combined_dataset.extxyz"), all_atoms, format="extxyz")

    # Verify outputs
    print("\nVerifying outputs...")
    checks = {}

    ok, msg = verify_csv(OUTPUT / "train.csv")
    checks["train.csv parseable"] = "PASS" if ok else f"FAIL: {msg}"
    print(f"  train.csv: {checks['train.csv parseable']} ({msg})")

    ok, msg = verify_csv(OUTPUT / "val.csv")
    checks["val.csv parseable"] = "PASS" if ok else f"FAIL: {msg}"
    print(f"  val.csv: {checks['val.csv parseable']} ({msg})")

    ok, msg = verify_csv(OUTPUT / "test.csv")
    checks["test.csv parseable"] = "PASS" if ok else f"FAIL: {msg}"
    print(f"  test.csv: {checks['test.csv parseable']} ({msg})")

    # Verify extxyz
    try:
        from ase.io import read as ase_read
        frames = ase_read(str(OUTPUT / "combined_dataset.extxyz"), index=":", format="extxyz")
        n_frames = len(frames)
        assert n_frames == len(processed), f"Frame count mismatch: {n_frames} vs {len(processed)}"
        checks["extxyz readable"] = "PASS"
        checks["Frame count matches"] = "PASS"
        print(f"  extxyz: PASS ({n_frames} frames)")
    except Exception as e:
        checks["extxyz readable"] = f"FAIL: {e}"
        checks["Frame count matches"] = "FAIL"
        print(f"  extxyz: FAIL: {e}")

    # Compute statistics
    atom_dist = {
        "1-10": sum(1 for e in processed if 1 <= e["num_atoms"] <= 10),
        "11-20": sum(1 for e in processed if 11 <= e["num_atoms"] <= 20),
        "21-30": sum(1 for e in processed if 21 <= e["num_atoms"] <= 30),
        "31-40": sum(1 for e in processed if 31 <= e["num_atoms"] <= 40),
        "41-51": sum(1 for e in processed if 41 <= e["num_atoms"] <= 51),
    }
    crystal_dist = {}
    for e in processed:
        cs = e.get("crystal_system", "Unknown")
        crystal_dist[cs] = crystal_dist.get(cs, 0) + 1
    source_dist = {}
    for e in processed:
        src = e["source"]
        source_dist[src] = source_dist.get(src, 0) + 1

    n_total = len(processed)

    # Write final report
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""=== MATTERGEN BASE MODEL TRAINING DATASET BUILD REPORT ===

Date: {now}
Environment: base_model_env conda env
Working directory: {BASE}/

--- DATASET SPLITS ---
  Training set:    {len(train)} structures (train.csv)
  Validation set:  {len(val)} structures (val.csv)
  Test set:        {len(test)} structures (test.csv)
  Total:           {n_total} structures

--- ATOM COUNT DISTRIBUTION ---
  1-10 atoms:  {atom_dist['1-10']} ({100*atom_dist['1-10']/n_total:.1f}%)
  11-20 atoms: {atom_dist['11-20']} ({100*atom_dist['11-20']/n_total:.1f}%)
  21-30 atoms: {atom_dist['21-30']} ({100*atom_dist['21-30']/n_total:.1f}%)
  31-40 atoms: {atom_dist['31-40']} ({100*atom_dist['31-40']/n_total:.1f}%)
  41-51 atoms: {atom_dist['41-51']} ({100*atom_dist['41-51']/n_total:.1f}%)

--- SOURCE DISTRIBUTION ---
  MP only:         {source_dist.get('MP', 0)} ({100*source_dist.get('MP',0)/n_total:.1f}%)
  COD only:        {source_dist.get('COD', 0)} ({100*source_dist.get('COD',0)/n_total:.1f}%)
  Alexandria only: {source_dist.get('Alexandria', 0)} ({100*source_dist.get('Alexandria',0)/n_total:.1f}%)
  NEMAD only:      {source_dist.get('NEMAD', 0)} ({100*source_dist.get('NEMAD',0)/n_total:.1f}%)

--- CRYSTAL SYSTEM DISTRIBUTION ---
  Cubic:        {crystal_dist.get('Cubic', 0)}
  Hexagonal:    {crystal_dist.get('Hexagonal', 0)}
  Tetragonal:   {crystal_dist.get('Tetragonal', 0)}
  Orthorhombic: {crystal_dist.get('Orthorhombic', 0)}
  Monoclinic:   {crystal_dist.get('Monoclinic', 0)}
  Triclinic:    {crystal_dist.get('Triclinic', 0)}
  Trigonal:     {crystal_dist.get('Trigonal', 0)}

--- OUTPUT FILES ---
  train.csv:               {OUTPUT}/train.csv
  val.csv:                 {OUTPUT}/val.csv
  test.csv:                {OUTPUT}/test.csv
  full_metadata.csv:       {OUTPUT}/full_metadata.csv
  combined_dataset.extxyz: {OUTPUT}/combined_dataset.extxyz

--- VALIDATION CHECKS ---
  train.csv parseable:           {checks.get('train.csv parseable', 'NOT CHECKED')}
  val.csv parseable:             {checks.get('val.csv parseable', 'NOT CHECKED')}
  test.csv parseable:            {checks.get('test.csv parseable', 'NOT CHECKED')}
  extxyz readable by ASE:        {checks.get('extxyz readable', 'NOT CHECKED')}
  Frame count matches total:     {checks.get('Frame count matches', 'NOT CHECKED')}
"""

    print("\n" + report)
    (LOGS / "final_report.txt").write_text(report)
    print(f"Report saved to {LOGS}/final_report.txt")

    logger.info("Export complete")
    notify(f"Export complete! {n_total} structures | Train:{len(train)} Val:{len(val)} Test:{len(test)}")


if __name__ == "__main__":
    main()
