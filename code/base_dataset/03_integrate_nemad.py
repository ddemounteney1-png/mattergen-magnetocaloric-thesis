#!/usr/bin/env python3
"""Stage 3: Integrate NEMAD supplementary CSVs.

NEMAD CSVs serve as a whitelist confirming experimental verification. This stage:
  - Annotates existing MP/COD/Alexandria structures with NEMAD property metadata
  - Generates enriched labelled NEMAD CSVs with matched CIF strings
  - Does NOT add new NEMAD-only structures to the merge (no valid structural data)

NOTE: The cod_rejected.pkl whitelist-override feature (re-including COD structures
rejected as 'theoretical' if confirmed by NEMAD) was skipped because Stage 1 did
not save a rejected-structures file. Prior testing showed NEMAD adds zero new
structures beyond MP/COD anyway, so the practical impact is nil.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import BASE_DATASET_WORK  # noqa: E402


import pickle
import logging
from pathlib import Path
import os

import pandas as pd
import requests
from tqdm import tqdm

BASE = BASE_DATASET_WORK
NEMAD_DIR = BASE / "NEMAD_csvs"
CHECKPOINTS = BASE / "checkpoints"
LOGS = BASE / "logs"
NEMAD_LABELLED_DIR = BASE / "merged_dataset" / "nemad_labelled"

LOG_FILE = LOGS / "nemad_integration.log"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
NEMAD_LABELLED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def notify(message, title="MatterGen Pipeline - NEMAD"):
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


def normalise_formula(formula_str):
    if not formula_str or str(formula_str).strip() in ("", "nan", "NaN"):
        return None
    try:
        from pymatgen.core import Composition
        return Composition(str(formula_str).strip()).reduced_formula
    except Exception:
        return None


def get_formula_from_row(row, csv_name):
    for col in ["Material_Name", "Chemical_Composition", "material_name", "formula"]:
        val = row.get(col)
        if val and str(val).strip() not in ("", "nan", "NaN"):
            norm = normalise_formula(str(val))
            if norm:
                return norm
    val = row.get("New_Column_Concatenated")
    if val and str(val).strip() not in ("", "nan", "NaN"):
        norm = normalise_formula(str(val))
        if norm:
            return norm
    return None


def extract_nemad_properties(row, csv_name):
    props = {}
    csv_stem = csv_name.replace(".csv", "")

    if "magnetic_anisotropy" in csv_stem:
        for col in ["Anisotropy Constant K1", "Anisotropy Constant K2",
                    "Anisotropy_Energy", "Anisotropy_Field", "Curie(TC)", "Coercivity"]:
            val = row.get(col)
            if val and str(val).strip() not in ("", "nan", "NaN"):
                props[f"nemad_mag_anisotropy_{col.replace(' ', '_')}"] = str(val)
    elif "magnetic_materials" in csv_stem:
        for col in ["Curie(Tc)", "Neel(Tn)", "Magnetic_Moment", "Magnetization",
                    "Coercivity", "Remanence"]:
            val = row.get(col)
            if val and str(val).strip() not in ("", "nan", "NaN"):
                props[f"nemad_mag_{col}"] = str(val)
    elif "superconductor" in csv_stem:
        for col in ["Superconducting_Transition_Temperature", "Median_Tc_By_Composition_K",
                    "Superconductor_Type", "Pressure"]:
            val = row.get(col)
            if val and str(val).strip() not in ("", "nan", "NaN"):
                props[f"nemad_sc_{col}"] = str(val)
    elif "thermoelectric" in csv_stem:
        for col in ["Clean_Figure_of_Merit", "Clean Seebeck Coefficient",
                    "Clean Electrical Conductivity", "Clean Thermal Conductivity"]:
            val = row.get(col)
            if val and str(val).strip() not in ("", "nan", "NaN"):
                props[f"nemad_te_{col.replace(' ', '_')}"] = str(val)

    props["nemad_source_csv"] = csv_stem
    return props


def structure_to_cif(structure):
    try:
        return structure.to(fmt="cif")
    except Exception:
        return None


def main():
    # Load all three source checkpoints
    mp_pkl = CHECKPOINTS / "mp_structures.pkl"
    cod_pkl = CHECKPOINTS / "cod_structures.pkl"
    alex_pkl = CHECKPOINTS / "alex_structures.pkl"

    print("Loading MP structures...")
    if not mp_pkl.exists():
        raise SystemExit("ERROR: mp_structures.pkl not found.")
    with open(mp_pkl, "rb") as f:
        mp_structures = pickle.load(f)
    print(f"  {len(mp_structures)} MP structures")

    print("Loading COD structures...")
    cod_structures = []
    if cod_pkl.exists():
        with open(cod_pkl, "rb") as f:
            cod_structures = pickle.load(f)
        print(f"  {len(cod_structures)} COD structures")
    else:
        print("  WARNING: cod_structures.pkl not found")

    print("Loading Alexandria structures...")
    alex_structures = []
    if alex_pkl.exists():
        with open(alex_pkl, "rb") as f:
            alex_structures = pickle.load(f)
        print(f"  {len(alex_structures)} Alexandria structures")
    else:
        print("  WARNING: alex_structures.pkl not found. Run 02b first.")

    # Build formula → list-of-indices lookups (priority: MP > COD > Alexandria)
    print("Building formula lookup tables...")
    mp_formula_idx = {}
    for i, e in enumerate(mp_structures):
        mp_formula_idx.setdefault(e["formula"], []).append(i)

    cod_formula_idx = {}
    for i, e in enumerate(cod_structures):
        cod_formula_idx.setdefault(e["formula"], []).append(i)

    alex_formula_idx = {}
    for i, e in enumerate(alex_structures):
        alex_formula_idx.setdefault(e["formula"], []).append(i)

    print(f"  MP formulas:         {len(mp_formula_idx)}")
    print(f"  COD formulas:        {len(cod_formula_idx)}")
    print(f"  Alexandria formulas: {len(alex_formula_idx)}")

    csv_files = sorted([f for f in NEMAD_DIR.iterdir()
                        if f.suffix == ".csv" and "Zone" not in f.name])
    print(f"\nFound {len(csv_files)} NEMAD CSV files")

    stats = {
        "total_rows": 0,
        "no_formula": 0,
        "matched_mp": 0,
        "matched_cod": 0,
        "matched_alex": 0,
        "unmatched": 0,
    }

    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file, dtype=str, low_memory=False)
        except Exception as e:
            logger.error(f"Failed to read {csv_file}: {e}")
            print(f"  ERROR reading: {e}")
            continue

        print(f"  Rows: {len(df)}, Columns: {list(df.columns)[:8]}...")

        # Build per-row labelled output
        labelled_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_file.stem, unit="row"):
            stats["total_rows"] += 1
            row_dict = row.to_dict()

            formula = get_formula_from_row(row_dict, csv_file.name)
            if formula is None:
                stats["no_formula"] += 1
                labelled_rows.append({**row_dict,
                                       "cif": None, "structure_source": None,
                                       "structure_source_id": None,
                                       "spacegroup_number": None, "num_atoms": None})
                continue

            nemad_props = extract_nemad_properties(row_dict, csv_file.name)

            matched_source = None
            matched_id = None
            matched_sg = None
            matched_natoms = None
            matched_cif = None

            # Match against MP (highest priority)
            if formula in mp_formula_idx:
                idxs = mp_formula_idx[formula]
                for idx in idxs:
                    mp_structures[idx].update(nemad_props)
                stats["matched_mp"] += 1
                matched_source = "MP"
                rep = mp_structures[idxs[0]]
                matched_id = rep["source_id"]
                matched_sg = rep.get("spacegroup_number")
                matched_natoms = rep.get("num_atoms")
                matched_cif = structure_to_cif(rep["structure"])
                logger.debug(f"NEMAD matched MP: {formula}")

            # Match COD (annotate even if already matched MP)
            if formula in cod_formula_idx:
                idxs = cod_formula_idx[formula]
                for idx in idxs:
                    cod_structures[idx].update(nemad_props)
                if matched_source is None:
                    stats["matched_cod"] += 1
                    matched_source = "COD"
                    rep = cod_structures[idxs[0]]
                    matched_id = rep["source_id"]
                    matched_sg = rep.get("spacegroup_number")
                    matched_natoms = rep.get("num_atoms")
                    matched_cif = structure_to_cif(rep["structure"])
                    logger.debug(f"NEMAD matched COD: {formula}")

            # Match Alexandria (annotate even if already matched above)
            if formula in alex_formula_idx:
                idxs = alex_formula_idx[formula]
                for idx in idxs:
                    alex_structures[idx].update(nemad_props)
                if matched_source is None:
                    stats["matched_alex"] += 1
                    matched_source = "Alexandria"
                    rep = alex_structures[idxs[0]]
                    matched_id = rep["source_id"]
                    matched_sg = rep.get("spacegroup_number")
                    matched_natoms = rep.get("num_atoms")
                    matched_cif = structure_to_cif(rep["structure"])
                    logger.debug(f"NEMAD matched Alexandria: {formula}")

            if matched_source is None:
                stats["unmatched"] += 1
                logger.debug(f"NEMAD unmatched: {formula} from {csv_file.name}")

            labelled_rows.append({
                **row_dict,
                "cif": matched_cif,
                "structure_source": matched_source,
                "structure_source_id": matched_id,
                "spacegroup_number": matched_sg,
                "num_atoms": matched_natoms,
            })

        # Save enriched labelled CSV
        out_path = NEMAD_LABELLED_DIR / csv_file.name
        pd.DataFrame(labelled_rows).to_csv(out_path, index=False)
        n_with_cif = sum(1 for r in labelled_rows if r.get("cif"))
        print(f"  Saved labelled CSV: {out_path.name} ({n_with_cif}/{len(labelled_rows)} rows have CIF)")

    # Summary
    print(f"\nNEMAD Integration Complete!")
    print(f"Total rows processed:   {stats['total_rows']}")
    print(f"No formula found:       {stats['no_formula']}")
    print(f"Matched to MP:          {stats['matched_mp']}")
    print(f"Matched to COD:         {stats['matched_cod']}")
    print(f"Matched to Alexandria:  {stats['matched_alex']}")
    print(f"Unmatched (no struct):  {stats['unmatched']}")
    print(f"\nNote: NEMAD does not add new structures to the merge/dedup stage.")
    print(f"      cod_rejected.pkl override feature was skipped (Stage 1 did not")
    print(f"      save rejected structures; prior testing showed zero impact).")
    logger.info(f"NEMAD integration complete: {stats}")

    # Save updated checkpoints with NEMAD annotations
    print("\nSaving updated checkpoints...")
    with open(mp_pkl, "wb") as f:
        pickle.dump(mp_structures, f)
    print(f"  MP updated ({len(mp_structures)} structures)")

    if cod_structures:
        with open(cod_pkl, "wb") as f:
            pickle.dump(cod_structures, f)
        print(f"  COD updated ({len(cod_structures)} structures)")

    if alex_structures:
        with open(alex_pkl, "wb") as f:
            pickle.dump(alex_structures, f)
        print(f"  Alexandria updated ({len(alex_structures)} structures)")

    # Save empty nemad_new_structures for compatibility
    nemad_pkl = CHECKPOINTS / "nemad_new_structures.pkl"
    with open(nemad_pkl, "wb") as f:
        pickle.dump([], f)
    print(f"  nemad_new_structures.pkl saved (empty — no new structures from NEMAD)")

    msg = (
        f"NEMAD integration complete!\n"
        f"Total rows: {stats['total_rows']}\n"
        f"Matched MP: {stats['matched_mp']}\n"
        f"Matched COD: {stats['matched_cod']}\n"
        f"Matched Alexandria: {stats['matched_alex']}\n"
        f"Unmatched: {stats['unmatched']}\n"
        f"Labelled CSVs written to merged_dataset/nemad_labelled/"
    )
    notify(msg)


if __name__ == "__main__":
    main()
