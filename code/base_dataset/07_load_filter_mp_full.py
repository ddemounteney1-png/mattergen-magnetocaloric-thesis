#!/usr/bin/env python3
"""Stage 7: Download and filter full Materials Project dataset via API.

Downloads ALL MP structures (including theoretical) with ≤51 atoms in batches
of 500. Excludes IDs already present in mp_structures.pkl (the experimental
set from Stage 2). Applies standard filters: no partial occupancy, ≤51 atoms
in primitive cell, inorganic only.

Output: checkpoints/mp_full_structures.pkl  (only NEW IDs not already in Stage 2)
Flag:   logs/mp_full_complete.flag
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import BASE_DATASET_WORK  # noqa: E402


import os
import sys
import pickle
import logging
import signal
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

BASE = BASE_DATASET_WORK
CHECKPOINTS = BASE / "checkpoints"
LOGS = BASE / "logs"

MP_FULL_PKL = CHECKPOINTS / "mp_full_structures.pkl"
FLAG_FILE = LOGS / "mp_full_complete.flag"
LOG_FILE = LOGS / "mp_full_filter.log"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MP_API_KEY = os.environ.get("MP_API_KEY")
if not MP_API_KEY:
    sys.exit("Set the MP_API_KEY environment variable. Get a key at https://next-gen.materialsproject.org/api")
BATCH_SIZE = 500


def notify(message, title="MatterGen Pipeline - MP Full"):
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


class _SgaTimeout(Exception):
    pass


def _sga_alarm_handler(signum, frame):
    raise _SgaTimeout()


def _run_sga(structure, timeout_sec=30):
    old_handler = signal.signal(signal.SIGALRM, _sga_alarm_handler)
    signal.alarm(timeout_sec)
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure)
        prim = sga.get_primitive_standard_structure()
        sg_num = sga.get_space_group_number()
        return prim, sg_num
    except _SgaTimeout:
        return None, None
    except Exception:
        return None, None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def filter_mp_doc(doc, existing_ids):
    """Filter one MPRester summary doc. Returns (result_dict|None, status, reason)."""
    mat_id = str(doc.material_id)
    source_id = mat_id if mat_id.startswith("mp-") else f"mp-{mat_id}"

    if source_id in existing_ids:
        return None, "skip", "already_in_experimental_set"

    try:
        structure = doc.structure
    except Exception as e:
        return None, "error", f"no_structure: {e}"

    for site in structure.sites:
        sp = site.species
        total_occ = sum(v for v in sp.values())
        if abs(total_occ - 1.0) > 1e-4:
            return None, "reject", f"partial_occupancy:{total_occ:.4f}"
        if len(sp) > 1:
            return None, "reject", "mixed_occupancy"

    if len(structure) > 300:
        return None, "reject", f"too_many_atoms_raw:{len(structure)}"

    prim, sg_num = _run_sga(structure)
    if prim is None:
        prim = structure
        sg_num = None

    if len(prim) > 51:
        return None, "reject", f"too_many_atoms:{len(prim)}"

    elems = {str(sp) for sp in prim.composition.elements}
    if "C" in elems and "H" in elems:
        return None, "reject", "organic_CH"

    lat = prim.lattice
    return {
        "source_id": source_id,
        "source": "MP_Full",
        "structure": prim,
        "formula": prim.composition.reduced_formula,
        "elements": sorted([str(el) for el in prim.composition.elements]),
        "spacegroup_number": sg_num,
        "num_atoms": len(prim),
        "formation_energy_per_atom": getattr(doc, "formation_energy_per_atom", None),
        "band_gap": getattr(doc, "band_gap", None),
        "energy_above_hull": getattr(doc, "energy_above_hull", None),
        "is_stable": getattr(doc, "is_stable", None),
        "volume": lat.volume,
        "density": getattr(doc, "density", None),
        "cod_id": None,
        "journal": None,
        "year": None,
        "doi": None,
        "authors": None,
        "r_factor": None,
        "cell_a": lat.a, "cell_b": lat.b, "cell_c": lat.c,
        "cell_alpha": lat.alpha, "cell_beta": lat.beta, "cell_gamma": lat.gamma,
    }, "pass", None


def main():
    notify("MP Full download starting via API")
    logger.info("MP Full stage starting")

    # Load existing experimental MP IDs to skip
    existing_ids = set()
    mp_pkl = CHECKPOINTS / "mp_structures.pkl"
    if mp_pkl.exists():
        with open(mp_pkl, "rb") as f:
            mp_exp = pickle.load(f)
        existing_ids = {e["source_id"] for e in mp_exp}
        print(f"Loaded {len(existing_ids)} existing MP IDs to skip")
    logger.info(f"Excluding {len(existing_ids)} already-downloaded IDs")

    from mp_api.client import MPRester

    passed = []
    n_skip = n_reject = n_error = 0
    counters = {"partial_occupancy": 0, "too_many_atoms": 0,
                "organic_CH": 0, "other_reject": 0}

    SAVE_EVERY = 5000
    total_processed = 0

    print(f"Querying MP API for all structures with ≤51 atoms (batch size {BATCH_SIZE})...")

    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            num_sites=(1, 51),
            fields=["material_id", "structure", "formation_energy_per_atom",
                    "band_gap", "energy_above_hull", "is_stable", "density"],
            chunk_size=BATCH_SIZE,
        )

        total_estimate = None
        pbar = tqdm(desc="Downloading & filtering MP Full", unit="structures")

        for doc in docs:
            total_processed += 1
            pbar.update(1)

            result, status, reason = filter_mp_doc(doc, existing_ids)

            if status == "pass":
                passed.append(result)
                existing_ids.add(result["source_id"])
            elif status == "skip":
                n_skip += 1
            elif status == "reject":
                n_reject += 1
                if "partial_occupancy" in reason or "mixed_occupancy" in reason:
                    counters["partial_occupancy"] += 1
                elif "too_many_atoms" in reason:
                    counters["too_many_atoms"] += 1
                elif "organic_CH" in reason:
                    counters["organic_CH"] += 1
                else:
                    counters["other_reject"] += 1
            else:
                n_error += 1
                logger.error(f"MP_FULL_ERROR {doc.material_id}: {reason}")

            if total_processed % SAVE_EVERY == 0:
                tmp = MP_FULL_PKL.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(passed, f)
                tmp.replace(MP_FULL_PKL)
                logger.info(
                    f"Checkpoint at {total_processed}: {len(passed)} new passed, "
                    f"{n_skip} skipped, {n_reject} rejected"
                )
                notify(
                    f"MP Full checkpoint: {total_processed} processed\n"
                    f"{len(passed)} new structures so far"
                )

        pbar.close()

    # Final save
    with open(MP_FULL_PKL, "wb") as f:
        pickle.dump(passed, f)

    print(f"\nMP Full Download & Filtering Complete!")
    print(f"Total processed:       {total_processed}")
    print(f"New structures passed: {len(passed)}")
    print(f"Skipped (in Stage 2):  {n_skip}")
    print(f"Rejected:              {n_reject}")
    print(f"  - Partial occupancy: {counters['partial_occupancy']}")
    print(f"  - Too many atoms:    {counters['too_many_atoms']}")
    print(f"  - Organic (C+H):     {counters['organic_CH']}")
    print(f"  - Other:             {counters['other_reject']}")
    print(f"Errors:                {n_error}")
    print(f"Saved {len(passed)} structures to {MP_FULL_PKL}")

    logger.info(
        f"Complete: total={total_processed} pass={len(passed)} "
        f"skip={n_skip} reject={n_reject} error={n_error} | {counters}"
    )

    with open(FLAG_FILE, "w") as f:
        f.write(f"Complete: {len(passed)} new structures at {datetime.now()}")

    notify(
        f"MP Full complete!\n"
        f"Total processed: {total_processed}\n"
        f"New structures: {len(passed)}\n"
        f"Skipped (already in Stage 2): {n_skip}\n"
        f"Rejected: {n_reject}"
    )


if __name__ == "__main__":
    main()
