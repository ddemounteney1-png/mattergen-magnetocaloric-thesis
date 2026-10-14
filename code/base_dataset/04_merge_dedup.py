#!/usr/bin/env python3
"""Stage 4: Merge MP + COD + Alexandria and deduplicate.

NEMAD is not included in this stage — all NEMAD-matched structures originate
from MP/COD/Alexandria and would be removed by deduplication anyway.
Priority order: MP > COD > Alexandria.
"""

import pickle
import logging
from pathlib import Path
from collections import defaultdict
import os

import requests
from tqdm import tqdm

BASE = Path("/home/daniel/base_model_dataset")
CHECKPOINTS = BASE / "checkpoints"
LOGS = BASE / "logs"

LOG_FILE = LOGS / "merge_dedup.log"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def notify(message, title="MatterGen Pipeline - Merge"):
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


def get_memory_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        return None


SOURCE_PRIORITY = {"MP": 0, "COD": 1, "Alexandria": 2}


def merge_metadata(keep_entry, drop_entry):
    """Merge non-None metadata from drop_entry into keep_entry."""
    for key, val in drop_entry.items():
        if key in ("source_id", "source", "structure", "formula", "elements",
                   "spacegroup_number", "num_atoms"):
            continue
        if keep_entry.get(key) is None and val is not None:
            keep_entry[key] = val
    return keep_entry


def dedup_formula_group(entries):
    """
    Deduplicate structures in one formula group.
    Returns (kept_entries, n_duplicates_removed).
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    if len(entries) == 1:
        return entries, 0

    # Sort by priority so the first occurrence of each unique structure is
    # the highest-priority source
    entries = sorted(entries, key=lambda e: SOURCE_PRIORITY.get(e["source"], 99))

    kept = []
    n_dups = 0

    for entry in entries:
        is_dup = False
        s1 = entry["structure"]
        v1 = s1.lattice.volume

        for kept_entry in kept:
            s2 = kept_entry["structure"]
            v2 = s2.lattice.volume

            if abs(v1 - v2) / max(v1, v2, 1e-6) > 0.20:
                continue

            try:
                if matcher.fit(s1, s2):
                    merge_metadata(kept_entry, entry)
                    logger.info(
                        f"DUP: {entry['source_id']} matches {kept_entry['source_id']} "
                        f"— kept {kept_entry['source_id']}"
                    )
                    n_dups += 1
                    is_dup = True
                    break
            except Exception as e:
                logger.warning(
                    f"StructureMatcher error {entry['source_id']} vs "
                    f"{kept_entry['source_id']}: {e}"
                )

        if not is_dup:
            kept.append(entry)

    return kept, n_dups


def final_validation(entry):
    """Re-validate one entry. Returns (is_valid, reason)."""
    structure = entry["structure"]

    if len(structure) > 51:
        return False, f"too_many_atoms:{len(structure)}"

    for site in structure.sites:
        species = site.species
        total_occ = sum(v for v in species.values())
        if abs(total_occ - 1.0) > 1e-6:
            return False, f"partial_occupancy:{total_occ:.4f}"
        if len(species) > 1:
            return False, "mixed_occupancy"

    elems = {str(sp) for sp in structure.composition.elements}
    if "C" in elems and "H" in elems:
        return False, "organic_CH"

    try:
        from pymatgen.io.cif import CifWriter
        _ = str(CifWriter(structure))
    except Exception as e:
        return False, f"cif_error:{e}"

    return True, None


def main():
    mp_pkl = CHECKPOINTS / "mp_structures.pkl"
    cod_pkl = CHECKPOINTS / "cod_structures.pkl"
    alex_pkl = CHECKPOINTS / "alex_structures.pkl"

    all_entries = []

    print("Loading MP structures...")
    with open(mp_pkl, "rb") as f:
        mp_structs = pickle.load(f)
    print(f"  {len(mp_structs)} MP structures")
    all_entries.extend(mp_structs)

    print("Loading COD structures...")
    cod_structs = []
    if cod_pkl.exists():
        with open(cod_pkl, "rb") as f:
            cod_structs = pickle.load(f)
        print(f"  {len(cod_structs)} COD structures")
        all_entries.extend(cod_structs)
    else:
        print("  WARNING: cod_structures.pkl not found")

    print("Loading Alexandria structures...")
    alex_structs = []
    if alex_pkl.exists():
        with open(alex_pkl, "rb") as f:
            alex_structs = pickle.load(f)
        print(f"  {len(alex_structs)} Alexandria structures")
        all_entries.extend(alex_structs)
    else:
        print("  WARNING: alex_structures.pkl not found. Run 02b first.")

    print(f"\nTotal before deduplication: {len(all_entries)}")

    mem = get_memory_gb()
    if mem is not None:
        print(f"Memory usage after loading all sources: {mem:.2f} GB (RSS)")
    logger.info(
        f"Pre-dedup: MP={len(mp_structs)} COD={len(cod_structs)} "
        f"Alexandria={len(alex_structs)} total={len(all_entries)} "
        f"memory={mem:.2f}GB" if mem else ""
    )

    # Group by reduced formula
    print("\nGrouping by formula...")
    formula_groups = defaultdict(list)
    for entry in all_entries:
        formula_groups[entry["formula"]].append(entry)
    print(f"Unique formulas: {len(formula_groups)}")

    # Deduplicate within each formula group
    print("Deduplicating (this may take a while for large formula groups)...")
    final_entries = []
    total_dups = 0

    for formula, group in tqdm(formula_groups.items(), desc="Deduplicating", unit="formula"):
        kept, n_dups = dedup_formula_group(group)
        final_entries.extend(kept)
        total_dups += n_dups

    print(f"\nDeduplication complete!")
    print(f"Duplicates removed:             {total_dups}")
    print(f"Unique structures after dedup:  {len(final_entries)}")
    logger.info(f"After dedup: {len(final_entries)} unique, {total_dups} duplicates removed")

    # Final validation sweep
    print("\nRunning final validation sweep...")
    valid_entries = []
    n_failed = 0
    fail_reasons = {}

    for entry in tqdm(final_entries, desc="Validating", unit="struct"):
        is_valid, reason = final_validation(entry)
        if is_valid:
            valid_entries.append(entry)
        else:
            n_failed += 1
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            logger.warning(f"FINAL_VALIDATION_FAIL {entry['source_id']}: {reason}")

    print(f"Final validation: {len(valid_entries)} passed, {n_failed} failed")
    if fail_reasons:
        for r, c in sorted(fail_reasons.items()):
            print(f"  {r}: {c}")

    logger.info(f"Final validation: passed={len(valid_entries)} failed={n_failed}")

    # Source distribution
    source_counts = {}
    for entry in valid_entries:
        s = entry["source"]
        source_counts[s] = source_counts.get(s, 0) + 1
    print(f"\nSource distribution: {source_counts}")

    # Save merged checkpoint
    merged_pkl = CHECKPOINTS / "merged_structures.pkl"
    with open(merged_pkl, "wb") as f:
        pickle.dump(valid_entries, f)
    print(f"Saved {len(valid_entries)} merged structures to {merged_pkl}")

    stats_msg = (
        f"Merge & dedup complete!\n"
        f"MP: {len(mp_structs)} | COD: {len(cod_structs)} | Alexandria: {len(alex_structs)}\n"
        f"Total before dedup: {len(all_entries)}\n"
        f"Duplicates removed: {total_dups}\n"
        f"Failed final validation: {n_failed}\n"
        f"Final unique structures: {len(valid_entries)}\n"
        f"Sources: {source_counts}"
    )
    print(f"\n{stats_msg}")
    logger.info(stats_msg)
    notify(stats_msg)


if __name__ == "__main__":
    main()
