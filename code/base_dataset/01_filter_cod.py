#!/usr/bin/env python3
"""Stage 1: Filter COD CIF files and extract valid structures."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import BASE_DATASET_WORK  # noqa: E402


import os
import re
import shutil
import logging
import pickle
import signal
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait as cf_wait, FIRST_COMPLETED
import multiprocessing

import requests

BASE = BASE_DATASET_WORK
COD_RAW = BASE / "COD_raw"
COD_FILTERED = BASE / "COD_filtered"
CHECKPOINTS = BASE / "checkpoints"
LOGS = BASE / "logs"

PARTIAL_PKL = CHECKPOINTS / "cod_structures_partial.pkl"
FINAL_PKL = CHECKPOINTS / "cod_structures.pkl"
LOG_FILE = LOGS / "cod_filter.log"

COD_FILTERED.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def notify(message, title="MatterGen Pipeline - COD"):
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


def _run_sga(structure, timeout_sec=45):
    """Run SpacegroupAnalyzer with a SIGALRM hard-kill inside the worker process.

    This is the real timeout — it fires inside the subprocess, so the worker
    process is not blocked and the executor slot stays available. The outer
    future.result() timeout was a no-op because as_completed() only yields
    already-completed futures.
    """
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


def extract_cif_tag(cif_text, tag):
    pattern = rf"{re.escape(tag)}\s+(.+)"
    m = re.search(pattern, cif_text)
    if m:
        val = m.group(1).strip().strip("'\"")
        return val if val not in (".", "?", "") else None
    return None


def extract_cif_tag_multi(cif_text, tag):
    pattern = rf"{re.escape(tag)}\s+(.+)"
    return [m.group(1).strip().strip("'\"") for m in re.finditer(pattern, cif_text)]


def process_cif(cif_path):
    """
    Process a single CIF file. Returns (result_dict, status, reason).
    status: 'pass', 'reject', or 'error'.
    """
    cif_path = str(cif_path)
    cod_id = None
    try:
        fname = os.path.basename(cif_path)
        cod_id = int(os.path.splitext(fname)[0])
    except ValueError:
        pass

    try:
        with open(cif_path, "r", encoding="utf-8", errors="replace") as f:
            cif_text = f.read()
    except Exception as e:
        return None, "error", f"read_error: {e}"

    # CHECK 6: Duplicate entry (fast text check)
    if "_cod_duplicate_entry" in cif_text:
        return None, "reject", "duplicate_entry"

    # CHECK 5: Theoretical/predicted (fast text check)
    lower_cif = cif_text.lower()
    if "theoretical" in lower_cif or "predicted" in lower_cif:
        if re.search(r"_cod_suboptimal_structure|_theoretical|predicted", cif_text, re.IGNORECASE):
            return None, "reject", "theoretical"

    # CHECK 2: Partial occupancy (raw text scan — fast pre-filter)
    raw_occu_vals = re.findall(r"_atom_site_occupancy\s+([\d.]+)", cif_text)
    for v in raw_occu_vals:
        try:
            if abs(float(v) - 1.0) > 1e-4:
                return None, "reject", f"partial_occupancy_raw:{v}"
        except ValueError:
            pass

    # PARSE: try pymatgen first, then ASE fallback
    structure = None
    try:
        from pymatgen.io.cif import CifParser
        parser = CifParser(cif_path, occupancy_tolerance=1.0)
        structures = parser.parse_structures(primitive=False)
        if structures:
            structure = structures[0]
    except Exception as e1:
        try:
            import ase.io
            from pymatgen.io.ase import AseAtomsAdaptor
            atoms = ase.io.read(cif_path, format="cif")
            structure = AseAtomsAdaptor.get_structure(atoms)
        except Exception as e2:
            logger.error(f"PARSE_FAIL cod-{cod_id} | pymatgen: {e1} | ase: {e2}")
            return None, "error", f"parse_fail: {e1}"

    if structure is None:
        return None, "error", "parse_none"

    # CHECK 2 (pymatgen level): verify occupancies on all sites
    for site in structure.sites:
        species = site.species
        total_occ = sum(v for v in species.values())
        if abs(total_occ - 1.0) > 1e-4:
            return None, "reject", f"partial_occupancy_site:{total_occ:.4f}"
        if len(species) > 1:
            return None, "reject", "mixed_occupancy"

    # CHECK 3: Atom count — fast reject >300, else run SGA with SIGALRM timeout
    if len(structure) > 300:
        return None, "reject", f"too_many_atoms:{len(structure)}"

    prim, sg_num = _run_sga(structure)
    if prim is None:
        prim = structure
        sg_num = None

    if len(prim) > 51:
        return None, "reject", f"too_many_atoms:{len(prim)}"

    # CHECK 4: Inorganic (no C+H)
    elems = {str(sp) for sp in prim.composition.elements}
    if "C" in elems and "H" in elems:
        return None, "reject", "organic_CH"

    def safe_float(val):
        if val is None:
            return None
        try:
            val = re.sub(r"\(\d+\)", "", str(val))
            return float(val)
        except Exception:
            return None

    journal = extract_cif_tag(cif_text, "_journal_name_full")
    year = extract_cif_tag(cif_text, "_journal_year")
    doi = extract_cif_tag(cif_text, "_journal_paper_doi")
    authors_list = extract_cif_tag_multi(cif_text, "_publ_author_name")
    authors = "; ".join(authors_list) if authors_list else None
    r_factor = extract_cif_tag(cif_text, "_refine_ls_R_factor_gt")
    if r_factor is None:
        r_factor = extract_cif_tag(cif_text, "_refine_ls_R_factor_all")

    lat = prim.lattice
    result = {
        "source_id": f"cod-{cod_id}",
        "source": "COD",
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
        "cod_id": cod_id,
        "journal": journal,
        "year": safe_float(year),
        "doi": doi,
        "authors": authors,
        "r_factor": safe_float(r_factor),
        "cell_a": lat.a, "cell_b": lat.b, "cell_c": lat.c,
        "cell_alpha": lat.alpha, "cell_beta": lat.beta, "cell_gamma": lat.gamma,
    }

    # Copy to COD_filtered
    try:
        dest = str(COD_FILTERED / f"{cod_id}.cif")
        if not os.path.exists(dest):
            shutil.copy2(cif_path, dest)
    except Exception as e:
        logger.warning(f"copy_fail cod-{cod_id}: {e}")

    return result, "pass", None


def categorize_reason(reason):
    if not reason:
        return "other_reject"
    if "partial_occupancy" in reason or "mixed_occupancy" in reason:
        return "partial_occupancy"
    if "too_many_atoms" in reason:
        return "too_many_atoms"
    if "organic_CH" in reason:
        return "organic_CH"
    if "theoretical" in reason:
        return "theoretical"
    if "duplicate_entry" in reason:
        return "duplicate_entry"
    return "other_reject"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--resume", action="store_true", help="Resume from partial checkpoint")
    args = parser.parse_args()

    notify("COD filtering started")

    print("Scanning for CIF files...")
    all_cifs = list(COD_RAW.rglob("*.cif"))
    total = len(all_cifs)
    print(f"Found {total} CIF files")
    logger.info(f"Found {total} CIF files")

    passed = []
    processed_paths = set()
    if args.resume and PARTIAL_PKL.exists():
        with open(PARTIAL_PKL, "rb") as f:
            checkpoint = pickle.load(f)
        passed = checkpoint.get("passed", [])
        processed_paths = set(checkpoint.get("processed_paths", []))
        print(f"Resuming: {len(passed)} passed, {len(processed_paths)} already processed")
        all_cifs = [c for c in all_cifs if str(c) not in processed_paths]
        print(f"Remaining: {len(all_cifs)} CIF files")

    n_pass = len(passed)
    n_reject = 0
    n_error = 0
    counters = {
        "partial_occupancy": 0,
        "too_many_atoms": 0,
        "organic_CH": 0,
        "theoretical": 0,
        "duplicate_entry": 0,
        "parse_fail": 0,
        "other_reject": 0,
    }

    start = time.time()
    SAVE_EVERY = 5000
    PRINT_EVERY = 5000
    BATCH_SIZE = 2000
    processed_count = len(processed_paths)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for batch_start in range(0, len(all_cifs), BATCH_SIZE):
            batch = all_cifs[batch_start:batch_start + BATCH_SIZE]
            future_to_path = {executor.submit(process_cif, cif): cif for cif in batch}
            pending = set(future_to_path.keys())

            while pending:
                # Wait up to 90s for at least one future to finish.
                # SIGALRM inside each worker limits SGA to 45s, so futures should
                # always complete well within 90s. The 90s here is only a backstop
                # for truly pathological cases (e.g. C extension ignoring signals).
                done, pending = cf_wait(pending, timeout=90, return_when=FIRST_COMPLETED)

                for future in done:
                    cif_path = future_to_path[future]
                    processed_paths.add(str(cif_path))
                    processed_count += 1
                    try:
                        result, status, reason = future.result()
                    except Exception as e:
                        result, status, reason = None, "error", str(e)
                        logger.error(f"FUTURE_ERROR {cif_path}: {e}")

                    if status == "pass":
                        n_pass += 1
                        passed.append(result)
                    elif status == "reject":
                        n_reject += 1
                        counters[categorize_reason(reason)] += 1
                        logger.debug(f"REJECT {cif_path}: {reason}")
                    else:
                        n_error += 1
                        counters["parse_fail"] += 1

                    if processed_count % PRINT_EVERY == 0:
                        elapsed = time.time() - start
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        eta = (total - processed_count) / rate if rate > 0 else 0
                        print(
                            f"Processed {processed_count}/{total} | "
                            f"Passed: {n_pass} | Rejected: {n_reject} | Errors: {n_error} | "
                            f"Rate: {rate:.0f}/s | ETA: {eta/60:.0f}min"
                        )

                    if processed_count % SAVE_EVERY == 0:
                        print(f"Saving checkpoint at {processed_count}...")
                        tmp = PARTIAL_PKL.with_suffix(".tmp")
                        with open(tmp, "wb") as f:
                            pickle.dump({"passed": passed, "processed_paths": list(processed_paths)}, f)
                        os.replace(tmp, PARTIAL_PKL)  # atomic rename — safe if killed mid-write
                        logger.info(f"Checkpoint saved at {processed_count}")

                if not done and pending:
                    # 90s elapsed with zero completions — SIGALRM didn't work on these.
                    # Abandon them and continue.
                    for future in list(pending):
                        cif_path = future_to_path[future]
                        logger.error(f"HUNG_FUTURE {cif_path}: no response after 90s, skipping")
                        future.cancel()
                        processed_paths.add(str(cif_path))
                        processed_count += 1
                        n_error += 1
                        counters["parse_fail"] += 1
                    pending = set()

    print(f"\nCOD Filtering Complete!")
    print(f"Total files: {total}")
    print(f"Passed: {n_pass}")
    print(f"Rejected: {n_reject}")
    print(f"  - Partial occupancy:      {counters['partial_occupancy']}")
    print(f"  - Too many atoms (>51):   {counters['too_many_atoms']}")
    print(f"  - Organic (C+H):          {counters['organic_CH']}")
    print(f"  - Theoretical/predicted:  {counters['theoretical']}")
    print(f"  - Duplicate COD entry:    {counters['duplicate_entry']}")
    print(f"  - Other:                  {counters['other_reject']}")
    print(f"Parse/read errors: {n_error} (counted in parse_fail: {counters['parse_fail']})")

    with open(FINAL_PKL, "wb") as f:
        pickle.dump(passed, f)
    print(f"Saved {len(passed)} structures to {FINAL_PKL}")

    logger.info(
        f"COD filtering complete: total={total} pass={n_pass} reject={n_reject} error={n_error} | {counters}"
    )

    msg = (
        f"COD filtering complete!\n"
        f"Total CIFs: {total}\n"
        f"Passed: {n_pass}\n"
        f"Rejected (partial occ): {counters['partial_occupancy']}\n"
        f"Rejected (>51 atoms): {counters['too_many_atoms']}\n"
        f"Rejected (organic): {counters['organic_CH']}\n"
        f"Rejected (theoretical): {counters['theoretical']}\n"
        f"Rejected (duplicate): {counters['duplicate_entry']}\n"
        f"Parse errors: {n_error}"
    )
    notify(msg)


if __name__ == "__main__":
    main()
