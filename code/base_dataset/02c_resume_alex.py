#!/usr/bin/env python3
"""Stage 2c: Resume Alexandria filter from the last checkpoint.

Loads existing checkpoint (alex_structures.pkl), determines the last
processed index from the log, then continues filtering from that point.
Saves the complete result to alex_structures_complete.pkl.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import BASE_DATASET_WORK  # noqa: E402


import json
import pickle
import re
import signal
import logging
from datetime import datetime
from pathlib import Path
import os

import requests
from tqdm import tqdm

BASE = BASE_DATASET_WORK
ALEX_JSON = BASE / "alexandria_convex_hull.json"
CHECKPOINTS = BASE / "checkpoints"
LOGS = BASE / "logs"

ALEX_PKL = CHECKPOINTS / "alex_structures.pkl"
ALEX_COMPLETE_PKL = CHECKPOINTS / "alex_structures_complete.pkl"
LOG_FILE = LOGS / "alex_filter.log"
FLAG_FILE = LOGS / "alex_filter_complete.flag"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def notify(message, title="MatterGen Pipeline - Alexandria"):
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


def process_entry(idx, entry_dict):
    """Filter one Alexandria entry. Returns (result_dict | None, status, reason)."""
    try:
        from pymatgen.entries.computed_entries import ComputedStructureEntry
        entry = ComputedStructureEntry.from_dict(entry_dict)
        structure = entry.structure
    except Exception as e:
        return None, "error", f"load_error: {e}"

    try:
        mat_id = entry.data.get("mat_id") if entry.data else None
    except Exception:
        mat_id = None
    source_id = f"alex-{mat_id}" if mat_id else f"alex-{idx}"

    for site in structure.sites:
        species = site.species
        total_occ = sum(v for v in species.values())
        if abs(total_occ - 1.0) > 1e-4:
            return None, "reject", f"partial_occupancy:{total_occ:.4f}"
        if len(species) > 1:
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

    try:
        e_per_atom = entry.energy_per_atom
    except Exception:
        e_per_atom = None

    lat = prim.lattice
    return {
        "source_id": source_id,
        "source": "Alexandria",
        "structure": prim,
        "formula": prim.composition.reduced_formula,
        "elements": sorted([str(el) for el in prim.composition.elements]),
        "spacegroup_number": sg_num,
        "num_atoms": len(prim),
        "formation_energy_per_atom": e_per_atom,
        "band_gap": None,
        "energy_above_hull": 0.0,
        "is_stable": True,
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
    }, "pass", None


def find_last_checkpoint_idx():
    """Parse the log to find the last checkpoint index processed."""
    last = 0
    try:
        with open(LOG_FILE) as f:
            for line in f:
                m = re.search(r"Checkpoint at (\d+):", line)
                if m:
                    last = int(m.group(1))
    except Exception:
        pass
    return last


def main():
    notify("Alexandria filter resuming from checkpoint")

    start_idx = find_last_checkpoint_idx()
    print(f"Last checkpoint: entry {start_idx}")

    if ALEX_PKL.exists() and start_idx > 0:
        print(f"Loading structures from checkpoint...")
        with open(ALEX_PKL, "rb") as f:
            passed = pickle.load(f)
        n_pass = len(passed)
        print(f"  Loaded {n_pass} structures (from {start_idx} entries processed)")
    else:
        print("No valid checkpoint found — starting from scratch")
        passed = []
        n_pass = 0
        start_idx = 0

    print(f"Loading {ALEX_JSON} (~846 MB — may take a minute)...")
    with open(ALEX_JSON) as f:
        data = json.load(f)

    entries_raw = data["entries"] if (isinstance(data, dict) and "entries" in data) else data
    total = len(entries_raw)
    remaining = total - start_idx
    print(f"Total entries: {total}, resuming from {start_idx}, {remaining} remaining")
    logger.info(f"Resume: total={total} start_idx={start_idx} already_passed={n_pass}")

    n_reject = n_error = 0
    counters = {"partial_occupancy": 0, "too_many_atoms": 0,
                "organic_CH": 0, "other_reject": 0, "parse_fail": 0}

    SAVE_EVERY = 10000

    for idx in tqdm(range(start_idx, total), desc="Filtering Alexandria (resume)", unit="entry"):
        entry_dict = entries_raw[idx]
        result, status, reason = process_entry(idx, entry_dict)

        if status == "pass":
            n_pass += 1
            passed.append(result)
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
            counters["parse_fail"] += 1
            logger.error(f"ALEX_ERROR idx={idx}: {reason}")

        if (idx + 1) % SAVE_EVERY == 0:
            tmp = ALEX_PKL.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(passed, f)
            tmp.replace(ALEX_PKL)
            logger.info(f"Checkpoint at {idx+1}: {n_pass} passed so far")

    with open(ALEX_COMPLETE_PKL, "wb") as f:
        pickle.dump(passed, f)

    print(f"\nAlexandria Filtering Complete!")
    print(f"Total entries:         {total}")
    print(f"Total passed:          {n_pass}")
    print(f"Rejected (this run):   {n_reject}")
    print(f"Errors (this run):     {n_error}")
    print(f"Saved {len(passed)} structures to {ALEX_COMPLETE_PKL}")

    logger.info(
        f"Complete: total={total} total_pass={n_pass} "
        f"reject_this_run={n_reject} error_this_run={n_error} | {counters}"
    )

    with open(FLAG_FILE, "w") as f:
        f.write(f"Complete: {n_pass} structures from {total} entries at {datetime.now()}")

    notify(
        f"Alexandria filtering complete!\n"
        f"Total entries: {total}\n"
        f"Total passed: {n_pass}\n"
        f"Rejected (this run): {n_reject}\n"
        f"Errors (this run): {n_error}"
    )


if __name__ == "__main__":
    main()
