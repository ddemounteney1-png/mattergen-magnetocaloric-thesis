#!/usr/bin/env python3
"""Stage 2: Load and filter Materials Project local cache."""

import json
import os
import pickle
import logging
from pathlib import Path

import requests

BASE = Path("/home/daniel/base_model_dataset")
MP_CACHE = BASE / "MP_Local_Cache"
CHECKPOINTS = BASE / "checkpoints"
LOGS = BASE / "logs"

FINAL_PKL = CHECKPOINTS / "mp_structures.pkl"
LOG_FILE = LOGS / "mp_filter.log"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def notify(message, title="MatterGen Pipeline - MP"):
    try:
        requests.post(
            "https://ntfy.sh/daniel_sheffield_claude",
            data=message.encode("utf-8"),
            headers={"Title": title},
            timeout=10,
        )
    except Exception:
        pass


def load_structure_from_dict(s_dict):
    """Reconstruct a pymatgen Structure from its JSON dict representation."""
    from pymatgen.core import Structure
    return Structure.from_dict(s_dict)


def filter_structure(structure, material_id):
    """
    Apply all critical filters. Returns (prim_structure, reject_reason) where
    reject_reason is None if the structure passes.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    # --- Partial occupancy check ---
    for site in structure.sites:
        species = site.species
        total_occ = sum(v for v in species.values())
        if abs(total_occ - 1.0) > 1e-6:
            return None, f"partial_occupancy:{total_occ:.4f}"
        if len(species) > 1:
            return None, "mixed_occupancy"

    # --- Get primitive standard structure ---
    try:
        sga = SpacegroupAnalyzer(structure)
        prim = sga.get_primitive_standard_structure()
    except Exception as e:
        logger.warning(f"{material_id}: SpacegroupAnalyzer failed ({e}), using as-is")
        prim = structure

    # --- Atom count check ---
    if len(prim) > 51:
        return None, f"too_many_atoms:{len(prim)}"

    # --- Inorganic check ---
    elems = {str(sp) for sp in prim.composition.elements}
    if "C" in elems and "H" in elems:
        return None, "organic_CH"

    return prim, None


def main():
    chunks = sorted(
        [f for f in MP_CACHE.iterdir() if f.suffix == ".json" and "Zone" not in f.name]
    )
    print(f"Found {len(chunks)} JSON chunk files in {MP_CACHE}")
    logger.info(f"Found {len(chunks)} MP JSON chunks")

    passed = []
    n_total = 0
    counters = {
        "partial_occupancy": 0,
        "too_many_atoms": 0,
        "organic_CH": 0,
        "parse_fail": 0,
    }

    for chunk_path in chunks:
        print(f"Loading {chunk_path.name}...", end="", flush=True)
        try:
            with open(chunk_path, "r") as f:
                chunk_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {chunk_path}: {e}")
            print(f" ERROR: {e}")
            continue

        chunk_passed = 0
        for entry in chunk_data:
            n_total += 1
            material_id = entry.get("material_id", "unknown")

            try:
                structure = load_structure_from_dict(entry["structure"])
            except Exception as e:
                logger.error(f"PARSE_FAIL {material_id}: {e}")
                counters["parse_fail"] += 1
                continue

            prim, reason = filter_structure(structure, material_id)

            if reason is not None:
                if "partial_occupancy" in reason or "mixed_occupancy" in reason:
                    counters["partial_occupancy"] += 1
                elif "too_many_atoms" in reason:
                    counters["too_many_atoms"] += 1
                elif "organic_CH" in reason:
                    counters["organic_CH"] += 1
                logger.debug(f"REJECT {material_id}: {reason}")
                continue

            # All checks passed — extract metadata
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(prim)
                sg_num = sga.get_space_group_number()
            except Exception:
                sg_num = None

            lat = prim.lattice
            result = {
                "source_id": material_id,
                "source": "MP",
                "structure": prim,
                "formula": prim.composition.reduced_formula,
                "elements": sorted([str(el) for el in prim.composition.elements]),
                "spacegroup_number": sg_num,
                "num_atoms": len(prim),
                "formation_energy_per_atom": None,
                "band_gap": None,
                "energy_above_hull": None,
                "is_stable": None,
                "volume": lat.volume,
                "density": None,
                "cod_id": None,
                "journal": None,
                "year": None,
                "doi": None,
                "authors": None,
                "r_factor": None,
                "cell_a": lat.a, "cell_b": lat.b, "cell_c": lat.c,
                "cell_alpha": lat.alpha, "cell_beta": lat.beta, "cell_gamma": lat.gamma,
            }
            passed.append(result)
            chunk_passed += 1

        print(f" {chunk_passed} passed")
        logger.info(f"{chunk_path.name}: {chunk_passed}/{len(chunk_data)} passed")

    print(f"\nMP Filtering Complete!")
    print(f"Total structures loaded: {n_total}")
    print(f"Passed all filters: {len(passed)}")
    print(f"Rejected (partial occupancy): {counters['partial_occupancy']}")
    print(f"Rejected (>51 atoms): {counters['too_many_atoms']}")
    print(f"Rejected (organic C+H): {counters['organic_CH']}")
    print(f"Parse failures: {counters['parse_fail']}")

    with open(FINAL_PKL, "wb") as f:
        pickle.dump(passed, f)
    print(f"Saved {len(passed)} structures to {FINAL_PKL}")

    logger.info(
        f"MP filtering complete: total={n_total} pass={len(passed)} "
        f"partial_occ={counters['partial_occupancy']} "
        f"too_many={counters['too_many_atoms']} organic={counters['organic_CH']}"
    )

    msg = (
        f"MP loading/filtering complete!\n"
        f"Total loaded: {n_total}\n"
        f"Passed: {len(passed)}\n"
        f"Rejected (partial occ): {counters['partial_occupancy']}\n"
        f"Rejected (>51 atoms): {counters['too_many_atoms']}\n"
        f"Rejected (organic C+H): {counters['organic_CH']}\n"
        f"Parse failures: {counters['parse_fail']}"
    )
    notify(msg)


if __name__ == "__main__":
    main()
