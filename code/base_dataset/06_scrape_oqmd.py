#!/usr/bin/env python3
"""Stage 6: Scrape OQMD structures via REST API.

Downloads stable (stability < 0.1 eV/atom) inorganic structures with ≤51 atoms.
Applies standard filters: no partial occupancy, ≤51 atoms in primitive cell, inorganic only.

Saves to: checkpoints/oqmd_structures.pkl
Flag:     logs/oqmd_complete.flag
"""

import json
import pickle
import logging
import time
import signal
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

BASE = Path("/home/daniel/base_model_dataset")
CHECKPOINTS = BASE / "checkpoints"
LOGS = BASE / "logs"
OQMD_CACHE = BASE / "OQMD_cache"

OQMD_PKL = CHECKPOINTS / "oqmd_structures.pkl"
LOG_FILE = LOGS / "oqmd_filter.log"
FLAG_FILE = LOGS / "oqmd_complete.flag"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
OQMD_CACHE.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

OQMD_BASE_URL = "http://oqmd.org/oqmdapi/formationenergy"
OQMD_FIELDS = "name,entry_id,icsd_id,spacegroup,natoms,volume,stability,delta_e,band_gap,sites,unit_cell"
OQMD_FILTER = "stability<0.1 AND natoms<52"


def notify(message, title="MatterGen Pipeline - OQMD"):
    try:
        requests.post(
            "https://ntfy.sh/daniel_sheffield_claude",
            data=message.encode("utf-8"),
            headers={"Title": title},
            timeout=10,
        )
    except Exception:
        pass


def get_oqmd_page(offset, limit=100, retries=3):
    """Fetch one page from OQMD API with retry + exponential backoff."""
    params = {
        "fields": OQMD_FIELDS,
        "filter": OQMD_FILTER,
        "noduplicate": "True",
        "limit": limit,
        "offset": offset,
        "format": "json",
    }
    delays = [10, 30, 60]
    for attempt in range(retries):
        try:
            resp = requests.get(OQMD_BASE_URL, params=params, timeout=90)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                delay = delays[attempt]
                logger.warning(f"offset={offset} attempt={attempt+1} failed: {e}. Retry in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"offset={offset} failed after {retries} attempts: {e}")
                raise


def parse_unit_cell(raw):
    """Parse unit cell into 3x3 list of lists."""
    if isinstance(raw, list):
        if len(raw) == 3 and isinstance(raw[0], list):
            return raw
        if len(raw) == 9:
            return [raw[i*3:(i+1)*3] for i in range(3)]
    return raw


def parse_sites(raw):
    """Parse OQMD sites into (species_list, frac_coords_list). Handles string and list formats."""
    species = []
    coords = []

    if isinstance(raw, str):
        # e.g. "Fe1 @ 0.0 0.0 0.0, Ni2 @ 0.5 0.5 0.5"
        for token in raw.split(","):
            token = token.strip()
            if not token or "@" not in token:
                continue
            elem_part, xyz = token.split("@", 1)
            elem = "".join(c for c in elem_part.strip() if c.isalpha())
            xyz_vals = [float(v) for v in xyz.strip().split()]
            species.append(elem)
            coords.append(xyz_vals[:3])

    elif isinstance(raw, list):
        for site in raw:
            if isinstance(site, str) and "@" in site:
                elem_part, xyz = site.split("@", 1)
                elem = "".join(c for c in elem_part.strip() if c.isalpha())
                xyz_vals = [float(v) for v in xyz.strip().split()]
                species.append(elem)
                coords.append(xyz_vals[:3])
            elif isinstance(site, dict):
                elem = site.get("species") or site.get("element") or site.get("type_symbol", "")
                elem = "".join(c for c in str(elem).strip() if c.isalpha())
                frac = site.get("fractional_coordinates") or [
                    site.get("x", 0.0), site.get("y", 0.0), site.get("z", 0.0)
                ]
                species.append(elem)
                coords.append(list(frac)[:3])

    return species, coords


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


def oqmd_entry_to_structure(entry):
    """Convert raw OQMD API entry to pymatgen Structure. Returns (Structure|None, error_str|None)."""
    try:
        from pymatgen.core import Structure, Lattice

        unit_cell_raw = entry.get("unit_cell")
        if unit_cell_raw is None:
            return None, "missing unit_cell"

        sites_raw = entry.get("sites")
        if sites_raw is None:
            return None, "missing sites"

        unit_cell = parse_unit_cell(unit_cell_raw)
        sp, frac = parse_sites(sites_raw)

        if not sp:
            return None, f"no sites parsed from: {str(sites_raw)[:100]}"

        lattice = Lattice(unit_cell)
        structure = Structure(lattice, sp, frac)
        return structure, None
    except Exception as e:
        return None, str(e)


def filter_oqmd_entry(entry):
    """Parse and filter one OQMD entry. Returns (result_dict|None, status, reason)."""
    entry_id = entry.get("entry_id", "?")
    source_id = f"oqmd-{entry_id}"

    structure, err = oqmd_entry_to_structure(entry)
    if structure is None:
        return None, "error", f"parse_error: {err}"

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
        "source": "OQMD",
        "structure": prim,
        "formula": prim.composition.reduced_formula,
        "elements": sorted([str(el) for el in prim.composition.elements]),
        "spacegroup_number": sg_num,
        "num_atoms": len(prim),
        "formation_energy_per_atom": entry.get("delta_e"),
        "band_gap": entry.get("band_gap"),
        "energy_above_hull": entry.get("stability"),
        "is_stable": entry.get("stability") == 0.0,
        "volume": lat.volume,
        "density": None,
        "icsd_id": entry.get("icsd_id"),
        "oqmd_entry_id": entry_id,
        "cod_id": None,
        "journal": None,
        "year": None,
        "doi": None,
        "authors": None,
        "r_factor": None,
        "cell_a": lat.a, "cell_b": lat.b, "cell_c": lat.c,
        "cell_alpha": lat.alpha, "cell_beta": lat.beta, "cell_gamma": lat.gamma,
    }, "pass", None


def download_all_oqmd(page_limit=100):
    """Download all OQMD pages matching filter. Returns list of raw entry dicts."""
    # Get total count
    print("Querying OQMD for total entry count...")
    data = get_oqmd_page(offset=0, limit=1)
    total_available = data.get("meta", {}).get("data_available", 0)
    print(f"Total structures matching query: {total_available}")
    logger.info(f"OQMD total available: {total_available}")
    notify(f"OQMD download starting\n{total_available} structures to download")

    # Resume: load already-downloaded batches
    existing_batches = sorted(OQMD_CACHE.glob("batch_?????.json"))
    all_entries = []
    for bf in existing_batches:
        with open(bf) as f:
            all_entries.extend(json.load(f))
    total_retrieved = len(all_entries)
    batch_num = len(existing_batches)
    if batch_num > 0:
        print(f"Resuming from batch {batch_num} ({total_retrieved} entries already downloaded)")

    pbar = tqdm(total=total_available, initial=total_retrieved,
                desc="Downloading OQMD", unit="structures")

    while total_retrieved < total_available:
        try:
            data = get_oqmd_page(offset=total_retrieved, limit=page_limit)
        except Exception as e:
            logger.error(f"Unrecoverable error at batch {batch_num}: {e}")
            break

        entries = data.get("data", [])
        if not entries:
            break

        batch_file = OQMD_CACHE / f"batch_{batch_num:05d}.json"
        with open(batch_file, "w") as f:
            json.dump(entries, f)

        all_entries.extend(entries)
        total_retrieved += len(entries)
        pbar.update(len(entries))

        if batch_num % 50 == 0 and batch_num > 0:
            ckpt = OQMD_CACHE / "all_entries_checkpoint.json"
            with open(ckpt, "w") as f:
                json.dump(all_entries, f)
            logger.info(f"Download checkpoint: batch={batch_num} total={total_retrieved}")

        if len(entries) < page_limit or not data.get("links", {}).get("next"):
            break

        batch_num += 1
        time.sleep(0.5)

    pbar.close()

    final_file = OQMD_CACHE / "all_entries_final.json"
    with open(final_file, "w") as f:
        json.dump(all_entries, f)

    print(f"Download complete: {total_retrieved} entries in {batch_num + 1} batches")
    logger.info(f"OQMD download complete: {total_retrieved} entries")
    notify(f"OQMD download complete!\n{total_retrieved} entries retrieved")
    return all_entries


def main():
    notify("OQMD scrape starting")
    logger.info("OQMD scrape starting")

    all_entries = download_all_oqmd()

    if not all_entries:
        logger.error("No entries downloaded from OQMD — writing empty pkl")
        with open(OQMD_PKL, "wb") as f:
            pickle.dump([], f)
        with open(FLAG_FILE, "w") as f:
            f.write(f"Failed: no entries downloaded at {datetime.now()}")
        notify("OQMD scrape failed: no entries downloaded")
        return

    # Inspect format of first entry (logged for debugging)
    first = all_entries[0]
    logger.info(
        f"Format inspection — unit_cell type={type(first.get('unit_cell')).__name__} "
        f"value={str(first.get('unit_cell'))[:200]}"
    )
    logger.info(
        f"Format inspection — sites type={type(first.get('sites')).__name__} "
        f"value={str(first.get('sites'))[:200]}"
    )
    print("Format inspection (first entry):")
    print(f"  unit_cell ({type(first.get('unit_cell')).__name__}): {str(first.get('unit_cell'))[:100]}")
    print(f"  sites ({type(first.get('sites')).__name__}): {str(first.get('sites'))[:200]}")

    # Filter all entries
    print(f"\nFiltering {len(all_entries)} OQMD entries...")
    passed = []
    n_pass = n_reject = n_error = 0
    counters = {"partial_occupancy": 0, "too_many_atoms": 0,
                "organic_CH": 0, "other_reject": 0, "parse_fail": 0}

    SAVE_EVERY = 5000

    for i, entry in tqdm(enumerate(all_entries), total=len(all_entries),
                         desc="Filtering OQMD", unit="entry"):
        result, status, reason = filter_oqmd_entry(entry)

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
            logger.error(f"OQMD_ERROR entry_id={entry.get('entry_id', '?')}: {reason}")

        if (i + 1) % SAVE_EVERY == 0:
            tmp = OQMD_PKL.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(passed, f)
            tmp.replace(OQMD_PKL)
            logger.info(f"Filter checkpoint at {i+1}: {n_pass} passed")

    with open(OQMD_PKL, "wb") as f:
        pickle.dump(passed, f)

    print(f"\nOQMD Filtering Complete!")
    print(f"Total downloaded:      {len(all_entries)}")
    print(f"Passed:                {n_pass}")
    print(f"Rejected:              {n_reject}")
    print(f"  - Partial occupancy: {counters['partial_occupancy']}")
    print(f"  - Too many atoms:    {counters['too_many_atoms']}")
    print(f"  - Organic (C+H):     {counters['organic_CH']}")
    print(f"  - Other:             {counters['other_reject']}")
    print(f"Errors (parse fail):   {n_error}")
    print(f"Saved {n_pass} structures to {OQMD_PKL}")

    logger.info(
        f"OQMD complete: total={len(all_entries)} pass={n_pass} "
        f"reject={n_reject} error={n_error} | {counters}"
    )

    with open(FLAG_FILE, "w") as f:
        f.write(f"Complete: {n_pass} structures from {len(all_entries)} entries at {datetime.now()}")

    notify(
        f"OQMD filtering complete!\n"
        f"Total: {len(all_entries)}\n"
        f"Passed: {n_pass}\n"
        f"Rejected (>51 atoms): {counters['too_many_atoms']}\n"
        f"Rejected (organic): {counters['organic_CH']}\n"
        f"Rejected (partial occ): {counters['partial_occupancy']}\n"
        f"Parse errors: {n_error}"
    )


if __name__ == "__main__":
    main()
