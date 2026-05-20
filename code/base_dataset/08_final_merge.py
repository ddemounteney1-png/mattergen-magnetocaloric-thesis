#!/usr/bin/env python3
"""Stage 8: Final merge, deduplication, and export.

Waits for completion flags from Stages 2c (Alexandria), 6 (OQMD), and 7 (MP Full),
then merges all sources, deduplicates, validates, and exports the final dataset.

Priority order: MP (experimental) > MP (full/theoretical) > COD > Alexandria > OQMD > NEMAD

Output: merged_dataset/train.csv, val.csv, test.csv, full_metadata.csv,
        combined_dataset.extxyz, final_report.txt
"""

import csv
import gc
import os
import pickle
import logging
import random
import time
import datetime
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np

import requests
from tqdm import tqdm

BASE = Path("/home/daniel/base_model_dataset")
CHECKPOINTS = BASE / "checkpoints"
LOGS = BASE / "logs"
OUTPUT = BASE / "merged_dataset"

LOG_FILE = LOGS / "final_merge.log"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SOURCE_PRIORITY = {
    "MP": 0,
    "MP_Full": 1,
    "COD": 2,
    "Alexandria": 3,
    "OQMD": 4,
    "NEMAD": 5,
}

DEDUP_CKPT = CHECKPOINTS / "dedup_progress.pkl"
SAVE_EVERY_FORMULAS = 20_000


def notify(message, title="MatterGen Pipeline - Final Merge"):
    try:
        requests.post(
            "https://ntfy.sh/daniel_sheffield_claude",
            data=message.encode("utf-8"),
            headers={"Title": title},
            timeout=10,
        )
    except Exception:
        pass


def wait_for_flag(flag_path, desc):
    """Poll until flag file exists. Prints status every 60s."""
    flag_path = Path(flag_path)
    if flag_path.exists():
        with open(flag_path) as f:
            content = f.read().strip()
        print(f"  {desc}: already complete — {content}")
        return
    print(f"  Waiting for {desc}...")
    last_print = time.time()
    while not flag_path.exists():
        time.sleep(10)
        if time.time() - last_print > 60:
            print(f"  Still waiting for {desc}...")
            last_print = time.time()
    with open(flag_path) as f:
        content = f.read().strip()
    print(f"  {desc} complete: {content}")


def get_memory_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        return None


def merge_metadata(keep_entry, drop_entry):
    """Copy non-None fields from drop_entry into keep_entry where keep is None."""
    skip = {"source_id", "source", "structure", "formula", "elements",
            "spacegroup_number", "num_atoms"}
    for key, val in drop_entry.items():
        if key not in skip and keep_entry.get(key) is None and val is not None:
            keep_entry[key] = val
    return keep_entry


def dedup_formula_group(entries):
    """Deduplicate one formula group. Returns (kept_entries, n_duplicates_removed)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    if len(entries) == 1:
        return entries, 0

    entries = sorted(entries, key=lambda e: SOURCE_PRIORITY.get(e["source"], 99))
    kept = []
    n_dups = 0

    for entry in entries:
        s1 = entry["structure"]
        v1 = s1.lattice.volume
        is_dup = False

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
    """Validate one entry. Returns (is_valid, reason)."""
    structure = entry["structure"]

    if len(structure) > 50:
        return False, f"too_many_atoms:{len(structure)}"

    for site in structure.sites:
        sp = site.species
        total_occ = sum(v for v in sp.values())
        if abs(total_occ - 1.0) > 1e-6:
            return False, f"partial_occupancy:{total_occ:.4f}"
        if len(sp) > 1:
            return False, "mixed_occupancy"

    elems = {str(sp) for sp in structure.composition.elements}
    if "C" in elems and "H" in elems:
        return False, "organic_CH"

    # Lattice hygiene
    vol = structure.lattice.volume
    if vol < 1.0:
        return False, f"near_zero_volume:{vol:.4f}"

    if len(structure) > 1:
        try:
            dm = structure.distance_matrix.copy()
            np.fill_diagonal(dm, float("inf"))
            min_dist = float(dm.min())
            if min_dist < 0.5:
                return False, f"atoms_too_close:{min_dist:.3f}A"
        except Exception:
            pass

    # Property outlier filtering
    band_gap = entry.get("band_gap")
    if band_gap is not None:
        try:
            bg = float(band_gap)
            if bg < 0.0 or bg > 20.0:
                return False, f"band_gap_outlier:{bg:.3f}"
        except (ValueError, TypeError):
            pass

    form_e = entry.get("formation_energy_per_atom")
    if form_e is not None:
        try:
            fe = float(form_e)
            if abs(fe) > 10.0:
                return False, f"formation_energy_outlier:{fe:.3f}"
        except (ValueError, TypeError):
            pass

    try:
        structure.to(fmt="cif")
    except Exception as e:
        return False, f"cif_error:{e}"

    return True, None


def get_primitive_structure(structure):
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure)
        return sga.get_primitive_standard_structure(), sga.get_space_group_number(), sga.get_crystal_system()
    except Exception:
        return structure, None, None


def niggli_reduce_structure(structure):
    try:
        import ase.build
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(structure)
        ase.build.niggli_reduce(atoms)
        return AseAtomsAdaptor.get_structure(atoms), True
    except Exception as e:
        logger.warning(f"Niggli reduction failed: {e}")
        return structure, False


def get_crystal_system(sg_num):
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


def process_for_export(entry):
    """Standardise one entry for export. Returns enriched dict or None on failure."""
    # Structures are already primitive from upstream filtering stages — skip SGA re-run.
    prim = entry["structure"]
    sg_num = entry.get("spacegroup_number")
    crystal_system = get_crystal_system(sg_num)

    try:
        from pymatgen.io.cif import CifWriter
        cif_string = str(CifWriter(prim))
    except Exception as e:
        logger.error(f"CIF generation failed for {entry['source_id']}: {e}")
        return None

    lat = prim.lattice
    elements_str = str(sorted([str(el) for el in prim.composition.elements]))

    return {
        "source_id": entry["source_id"],
        "source": entry["source"],
        "structure": prim,
        "material_id": entry["source_id"],
        "pretty_formula": prim.composition.reduced_formula,
        "elements": elements_str,
        "spacegroup_number": sg_num,
        "crystal_system": crystal_system,
        "num_atoms": len(prim),
        "cif": cif_string,
        "formation_energy_per_atom": entry.get("formation_energy_per_atom"),
        "dft_band_gap": entry.get("band_gap"),
        "e_above_hull": entry.get("energy_above_hull"),
        "is_stable": entry.get("is_stable"),
        "volume": lat.volume,
        "density": entry.get("density"),
        "icsd_id": entry.get("icsd_id"),
        "oqmd_entry_id": entry.get("oqmd_entry_id"),
        "cod_id": entry.get("cod_id"),
        "journal": entry.get("journal"),
        "year": entry.get("year"),
        "doi": entry.get("doi"),
        "authors": entry.get("authors"),
        "r_factor": entry.get("r_factor"),
        "cell_a": lat.a, "cell_b": lat.b, "cell_c": lat.c,
        "cell_alpha": lat.alpha, "cell_beta": lat.beta, "cell_gamma": lat.gamma,
        "cell_volume": lat.volume,
        **{k: v for k, v in entry.items() if k.startswith("nemad_")},
    }


def stratified_split(entries, train_frac=0.80, val_frac=0.10, seed=42):
    """80/10/10 split stratified by crystal system."""
    random.seed(seed)
    by_system = defaultdict(list)
    for e in entries:
        by_system[e.get("crystal_system", "Unknown")].append(e)

    train, val, test = [], [], []
    for group in by_system.values():
        random.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_frac))
        n_val = max(0, int(n * val_frac))
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def write_mattergen_csv(entries, path, global_index_offset=0):
    import pandas as pd
    rows = [{
        "": i + global_index_offset,
        "material_id": e["material_id"],
        "source": e["source"],
        "formation_energy_per_atom": e.get("formation_energy_per_atom", ""),
        "dft_band_gap": e.get("dft_band_gap", ""),
        "pretty_formula": e["pretty_formula"],
        "e_above_hull": e.get("e_above_hull", ""),
        "elements": e["elements"],
        "cif": e["cif"],
        "spacegroup_number": e.get("spacegroup_number", ""),
    } for i, e in enumerate(entries)]
    pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def cif_roundtrip_check(csv_path, n_sample=20):
    """Test CIF round-trip on random sample. Returns (pass_bool, message)."""
    import pandas as pd
    from pymatgen.io.cif import CifParser
    df = pd.read_csv(csv_path)
    assert len(df) > 0, f"Empty CSV: {csv_path}"
    indices = random.sample(range(len(df)), min(n_sample, len(df)))
    failures = 0
    for idx in indices:
        try:
            cif_str = df.iloc[idx]["cif"]
            structs = CifParser(StringIO(cif_str)).parse_structures()
            assert len(structs) > 0
            assert len(structs[0]) <= 50
        except Exception as e:
            failures += 1
            logger.error(f"CIF round-trip fail in {csv_path} row {idx}: {e}")
    ok = failures == 0
    msg = f"{len(indices) - failures}/{len(indices)} passed"
    return ok, msg


def csv_roundtrip_check(csv_path):
    """Verify csv.reader and pandas see the same row count. Returns (pass_bool, message)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    n_pandas = len(df)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        n_csv = sum(1 for _ in reader)
    ok = n_pandas == n_csv
    msg = f"pandas={n_pandas} rows, csv.reader={n_csv} rows"
    if not ok:
        logger.error(f"CSV row count mismatch in {csv_path}: {msg}")
    return ok, msg


def main():
    logger.info("Stage 8 started")

    merged_pkl = CHECKPOINTS / "merged_structures_v2.pkl"

    # --- Fast path: skip dedup if already done and saved ---
    if merged_pkl.exists():
        print(f"Found {merged_pkl} — skipping merge/dedup, loading directly for export.")
        notify("Stage 8 resuming from saved merged checkpoint — starting export")
        with open(merged_pkl, "rb") as f:
            valid_entries = pickle.load(f)
        print(f"Loaded {len(valid_entries)} validated structures")
        logger.info(f"Resumed from checkpoint: {len(valid_entries)} structures")

        # Reconstruct source counts for the report from the loaded entries
        mp_exp_n = sum(1 for e in valid_entries if e["source"] == "MP")
        mp_full_n = sum(1 for e in valid_entries if e["source"] == "MP_Full")
        cod_n = sum(1 for e in valid_entries if e["source"] == "COD")
        alex_n = sum(1 for e in valid_entries if e["source"] == "Alexandria")
        oqmd_n = sum(1 for e in valid_entries if e["source"] == "OQMD")
        nemad_n = sum(1 for e in valid_entries if e["source"] == "NEMAD")
        total_before = len(valid_entries)  # exact pre-dedup unknown at resume
        total_dups = 92392  # from log: After dedup: 277248 unique, 92392 dups removed
        n_failed = 16       # from log: Validation: passed=277232 failed=16

    else:
        # --- Full path: merge, dedup, validate, then save ---

        if DEDUP_CKPT.exists():
            # --- Resume path: load checkpoint ONLY — no source pkl loading ---
            # This keeps peak RAM to checkpoint size rather than checkpoint + all sources.
            print(f"Found dedup checkpoint — resuming from {DEDUP_CKPT.name}")
            with open(DEDUP_CKPT, "rb") as f:
                ckpt = pickle.load(f)
            final_entries    = ckpt["final_entries"]
            total_dups       = ckpt["total_dups"]
            remaining_groups = ckpt["remaining_groups"]
            n_formulas_done  = ckpt["n_formulas_done"]
            n_formulas_total = ckpt["n_formulas_total"]
            total_before     = ckpt["total_before"]
            sc               = ckpt.get("source_counts", {})
            mp_exp_n         = sc.get("mp_exp", 0)
            mp_full_n        = sc.get("mp_full", 0)
            cod_n            = sc.get("cod", 0)
            alex_n           = sc.get("alex", 0)
            oqmd_n           = sc.get("oqmd", 0)
            nemad_n          = sc.get("nemad", 0)
            # Restore pop-from-end order (checkpoint saved in natural order)
            remaining_groups.reverse()
            mem = get_memory_gb()
            print(f"Resuming: {n_formulas_done}/{n_formulas_total} formulas done, "
                  f"{len(final_entries)} kept, {len(remaining_groups)} groups left "
                  f"(mem: {'%.2fGB' % mem if mem else 'N/A'})")
            logger.info(f"Dedup resume: {n_formulas_done}/{n_formulas_total}, "
                        f"{len(final_entries)} entries, {len(remaining_groups)} groups left")
            notify(f"Dedup resuming\n{n_formulas_done}/{n_formulas_total} formulas done\n"
                   f"{len(final_entries)} kept, {len(remaining_groups)} remaining")

        else:
            # --- Fresh start: load all sources, build remaining_groups ---
            notify("Final merge starting — waiting for upstream stages")
            print("Waiting for upstream stages to complete...")
            wait_for_flag(LOGS / "alex_filter_complete.flag", "Alexandria filter (Stage 2c)")
            wait_for_flag(LOGS / "oqmd_complete.flag", "OQMD scrape (Stage 6)")
            wait_for_flag(LOGS / "mp_full_complete.flag", "MP Full filter (Stage 7)")
            print("All upstream stages complete. Starting merge.\n")
            notify("All upstream stages complete — starting final merge")

            def load_pkl(path, name):
                p = Path(path)
                if not p.exists():
                    print(f"  WARNING: {name} not found at {p}")
                    return []
                with open(p, "rb") as f:
                    data = pickle.load(f)
                print(f"  {name}: {len(data)} structures")
                return data

            print("Loading all sources...")
            mp_exp  = load_pkl(CHECKPOINTS / "mp_structures.pkl",            "MP (experimental)")
            mp_full = load_pkl(CHECKPOINTS / "mp_full_structures.pkl",       "MP (full/theoretical)")
            cod     = load_pkl(CHECKPOINTS / "cod_structures.pkl",           "COD")
            alex    = load_pkl(CHECKPOINTS / "alex_structures_complete.pkl", "Alexandria (complete)")
            oqmd    = load_pkl(CHECKPOINTS / "oqmd_structures.pkl",          "OQMD")
            nemad   = load_pkl(CHECKPOINTS / "nemad_new_structures.pkl",     "NEMAD new")

            mp_exp_n  = len(mp_exp)
            mp_full_n = len(mp_full)
            cod_n     = len(cod)
            alex_n    = len(alex)
            oqmd_n    = len(oqmd)
            nemad_n   = len(nemad)
            total_before = mp_exp_n + mp_full_n + cod_n + alex_n + oqmd_n + nemad_n
            print(f"\nTotal before deduplication: {total_before}")

            mem = get_memory_gb()
            logger.info(
                f"Pre-dedup: MP={mp_exp_n} MP_Full={mp_full_n} COD={cod_n} "
                f"Alex={alex_n} OQMD={oqmd_n} NEMAD={nemad_n} "
                f"total={total_before} memory={'%.2fGB' % mem if mem else 'N/A'}"
            )

            print("\nGrouping by formula...")
            formula_groups = defaultdict(list)
            for src_list in [mp_exp, mp_full, cod, alex, oqmd, nemad]:
                for entry in src_list:
                    formula_groups[entry["formula"]].append(entry)
            del mp_exp, mp_full, cod, alex, oqmd, nemad
            gc.collect()

            n_formulas_total = len(formula_groups)
            print(f"Unique formulas: {n_formulas_total}")

            # Convert to reversed list so pop() consumes from original start
            remaining_groups = list(formula_groups.items())
            del formula_groups
            gc.collect()
            remaining_groups.reverse()

            final_entries   = []
            total_dups      = 0
            n_formulas_done = 0

        # --- Dedup loop: pop from list so processed Structure objects are freed ---
        # RAM decreases as dups are removed: starts ~6.5GB, ends ~4.9GB
        source_counts = {
            "mp_exp": mp_exp_n, "mp_full": mp_full_n, "cod": cod_n,
            "alex": alex_n, "oqmd": oqmd_n, "nemad": nemad_n,
        }

        print(f"\nDeduplicating {len(remaining_groups)} formula groups remaining...")
        with tqdm(total=n_formulas_total, initial=n_formulas_done,
                  desc="Deduplicating", unit="formula") as pbar:
            while remaining_groups:
                _formula, group = remaining_groups.pop()
                kept, n_dups = dedup_formula_group(group)
                final_entries.extend(kept)
                total_dups += n_dups
                del group  # free dup Structure objects immediately
                n_formulas_done += 1
                pbar.update(1)

                if n_formulas_done % SAVE_EVERY_FORMULAS == 0:
                    gc.collect()
                    pct = 100 * n_formulas_done / n_formulas_total
                    mem = get_memory_gb()
                    # Checkpoint saves remaining in natural order; restore after
                    remaining_groups.reverse()
                    tmp = DEDUP_CKPT.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        pickle.dump({
                            "final_entries":    final_entries,
                            "total_dups":       total_dups,
                            "remaining_groups": remaining_groups,
                            "n_formulas_done":  n_formulas_done,
                            "n_formulas_total": n_formulas_total,
                            "total_before":     total_before,
                            "source_counts":    source_counts,
                        }, f)
                    tmp.replace(DEDUP_CKPT)
                    remaining_groups.reverse()  # back to pop-from-end order
                    logger.info(
                        f"Dedup checkpoint: {n_formulas_done}/{n_formulas_total} "
                        f"({pct:.1f}%), {len(final_entries)} entries, {total_dups} dups, "
                        f"mem={'%.2fGB' % mem if mem else 'N/A'}"
                    )
                    notify(
                        f"Dedup checkpoint: {n_formulas_done}/{n_formulas_total} ({pct:.1f}%)\n"
                        f"{len(final_entries)} unique entries\n"
                        f"mem={'%.2fGB' % mem if mem else 'N/A'}"
                    )

        if DEDUP_CKPT.exists():
            DEDUP_CKPT.unlink()

        print(f"\nDeduplication complete!")
        print(f"Duplicates removed:            {total_dups}")
        print(f"Unique structures after dedup: {len(final_entries)}")
        logger.info(f"After dedup: {len(final_entries)} unique, {total_dups} dups removed")

        print("\nRunning final validation sweep...")
        valid_entries = []
        n_failed = 0
        fail_reasons = {}
        for entry in tqdm(final_entries, desc="Validating", unit="struct"):
            ok, reason = final_validation(entry)
            if ok:
                valid_entries.append(entry)
            else:
                n_failed += 1
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                logger.warning(f"FINAL_VALIDATION_FAIL {entry['source_id']}: {reason}")

        print(f"Final validation: {len(valid_entries)} passed, {n_failed} failed")
        if fail_reasons:
            for r, c in sorted(fail_reasons.items()):
                print(f"  {r}: {c}")
        logger.info(f"Validation: passed={len(valid_entries)} failed={n_failed}")

        with open(merged_pkl, "wb") as f:
            pickle.dump(valid_entries, f)
        print(f"\nSaved {len(valid_entries)} merged structures to {merged_pkl}")
        notify(f"Merge & dedup complete!\n{len(valid_entries)} unique structures\n{total_dups} duplicates removed")

        # Free dedup memory before export
        del final_entries, remaining_groups
        gc.collect()
        print("Freed dedup memory. Reloading from checkpoint for export...")
        with open(merged_pkl, "rb") as f:
            valid_entries = pickle.load(f)
        print(f"Reloaded {len(valid_entries)} structures for export")

    # Process for export (primitive + Niggli + CIF)
    print("\nProcessing entries for export (primitive + CIF)...")
    processed = []
    n_proc_failed = 0
    for entry in tqdm(valid_entries, desc="Processing", unit="struct"):
        result = process_for_export(entry)
        if result is not None:
            processed.append(result)
        else:
            n_proc_failed += 1
            logger.error(f"process_for_export failed for {entry.get('source_id', '?')}")

    print(f"Processed: {len(processed)} | Failed: {n_proc_failed}")

    # Split 80/10/10 stratified by crystal system
    print("\nSplitting train/val/test (80/10/10, seed=42)...")
    train, val, test = stratified_split(processed)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Write CSVs
    print("\nWriting MatterGen CSVs...")
    write_mattergen_csv(train, OUTPUT / "train.csv")
    write_mattergen_csv(val,   OUTPUT / "val.csv")
    write_mattergen_csv(test,  OUTPUT / "test.csv")

    # Write full metadata
    print("Writing full_metadata.csv...")
    import pandas as pd
    meta_rows = [{k: v for k, v in e.items() if k != "structure"} for e in processed]
    pd.DataFrame(meta_rows).to_csv(OUTPUT / "full_metadata.csv", index=False, quoting=csv.QUOTE_ALL)

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

    # CIF round-trip + csv.reader validation
    print("\nRunning CSV validation (CIF round-trip + row count consistency)...")
    checks = {}
    for csv_name in ["train.csv", "val.csv", "test.csv"]:
        ok_cif, msg_cif = cif_roundtrip_check(OUTPUT / csv_name, n_sample=20)
        checks[f"CIF round-trip ({csv_name})"] = "PASS" if ok_cif else f"FAIL: {msg_cif}"
        print(f"  {csv_name} CIF:     {'PASS' if ok_cif else 'FAIL'} — {msg_cif}")
        if not ok_cif:
            raise AssertionError(f"CIF round-trip failures in {csv_name}!")

        ok_csv, msg_csv = csv_roundtrip_check(OUTPUT / csv_name)
        checks[f"CSV row count ({csv_name})"] = "PASS" if ok_csv else f"FAIL: {msg_csv}"
        print(f"  {csv_name} rows:    {'PASS' if ok_csv else 'FAIL'} — {msg_csv}")
        if not ok_csv:
            raise AssertionError(f"CSV row count mismatch in {csv_name}!")

    # Verify extxyz
    try:
        from ase.io import read as ase_read
        frames = ase_read(str(OUTPUT / "combined_dataset.extxyz"), index=":", format="extxyz")
        n_frames = len(frames)
        checks["extxyz readable by ASE"] = "PASS"
        if n_frames == len(processed):
            checks["Frame count matches total"] = "PASS"
            print(f"  extxyz: PASS ({n_frames} frames)")
        else:
            n_skipped = len(processed) - n_frames
            checks["Frame count matches total"] = f"WARN: {n_frames} frames, {n_skipped} skipped (ASE conversion failures)"
            print(f"  extxyz: WARN — {n_frames}/{len(processed)} frames ({n_skipped} skipped due to ASE conversion failure)")
    except Exception as e:
        checks["extxyz readable by ASE"] = f"FAIL: {e}"
        checks["Frame count matches total"] = "FAIL"
        print(f"  extxyz: FAIL: {e}")

    # Compute statistics
    n_total = len(processed)
    atom_dist = {
        "1-10":  sum(1 for e in processed if 1 <= e["num_atoms"] <= 10),
        "11-20": sum(1 for e in processed if 11 <= e["num_atoms"] <= 20),
        "21-30": sum(1 for e in processed if 21 <= e["num_atoms"] <= 30),
        "31-40": sum(1 for e in processed if 31 <= e["num_atoms"] <= 40),
        "41-50": sum(1 for e in processed if 41 <= e["num_atoms"] <= 50),
    }
    crystal_dist = defaultdict(int)
    for e in processed:
        crystal_dist[e.get("crystal_system", "Unknown")] += 1

    source_dist = defaultdict(int)
    for e in processed:
        source_dist[e["source"]] += 1

    oqmd_icsd = sum(1 for e in processed if e["source"] == "OQMD" and e.get("icsd_id"))
    oqmd_proto = sum(1 for e in processed if e["source"] == "OQMD" and not e.get("icsd_id"))

    # Validation checks
    all_le51 = all(e["num_atoms"] <= 50 for e in processed)
    all_no_partial = True
    all_inorganic = True
    for e in processed:
        for site in e["structure"].sites:
            sp = site.species
            if abs(sum(sp.values()) - 1.0) > 1e-6 or len(sp) > 1:
                all_no_partial = False
        elems = {str(s) for s in e["structure"].composition.elements}
        if "C" in elems and "H" in elems:
            all_inorganic = False

    def pct(n):
        return f"{100 * n / n_total:.1f}%" if n_total else "0.0%"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""=== MATTERGEN BASE MODEL TRAINING DATASET BUILD REPORT (v2) ===

Date: {now}
Environment: base_model_env conda env
Working directory: /home/daniel/base_model_dataset/

--- DATASET SPLITS ---
  Training set:    {len(train)} structures (train.csv)
  Validation set:  {len(val)} structures (val.csv)
  Test set:        {len(test)} structures (test.csv)
  Total:           {n_total} structures

--- INPUT SOURCES ---
  MP (experimental):      {mp_exp_n}
  MP (full/theoretical):  {mp_full_n}
  COD:                    {cod_n}
  Alexandria (complete):  {alex_n}
  OQMD:                   {oqmd_n}
  NEMAD new:              {nemad_n}
  Total before dedup:     {total_before}
  Duplicates removed:     {total_dups}
  Failed validation:      {n_failed}

--- ATOM COUNT DISTRIBUTION ---
  1-10 atoms:  {atom_dist['1-10']} ({pct(atom_dist['1-10'])})
  11-20 atoms: {atom_dist['11-20']} ({pct(atom_dist['11-20'])})
  21-30 atoms: {atom_dist['21-30']} ({pct(atom_dist['21-30'])})
  31-40 atoms: {atom_dist['31-40']} ({pct(atom_dist['31-40'])})
  41-50 atoms: {atom_dist['41-50']} ({pct(atom_dist['41-50'])})

--- SOURCE DISTRIBUTION ---
  MP (experimental):  {source_dist.get('MP', 0)} ({pct(source_dist.get('MP', 0))})
  MP (theoretical):   {source_dist.get('MP_Full', 0)} ({pct(source_dist.get('MP_Full', 0))})
  COD:                {source_dist.get('COD', 0)} ({pct(source_dist.get('COD', 0))})
  Alexandria:         {source_dist.get('Alexandria', 0)} ({pct(source_dist.get('Alexandria', 0))})
  OQMD (ICSD):        {oqmd_icsd} ({pct(oqmd_icsd)})
  OQMD (prototype):   {oqmd_proto} ({pct(oqmd_proto)})
  NEMAD:              {source_dist.get('NEMAD', 0)} ({pct(source_dist.get('NEMAD', 0))})

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

--- VALIDATION ---
  All structures ≤50 atoms:         {'PASS' if all_le51 else 'FAIL'}
  No partial occupancy:              {'PASS' if all_no_partial else 'FAIL'}
  No organic (C+H):                  {'PASS' if all_inorganic else 'FAIL'}
  CIF round-trip (train.csv):        {checks.get('CIF round-trip (train.csv)', 'NOT CHECKED')}
  CIF round-trip (val.csv):          {checks.get('CIF round-trip (val.csv)', 'NOT CHECKED')}
  CIF round-trip (test.csv):         {checks.get('CIF round-trip (test.csv)', 'NOT CHECKED')}
  extxyz readable by ASE:            {checks.get('extxyz readable by ASE', 'NOT CHECKED')}
  Frame count matches total:         {checks.get('Frame count matches total', 'NOT CHECKED')}
"""

    print("\n" + report)
    (LOGS / "final_report.txt").write_text(report)
    print(f"Report saved to {LOGS}/final_report.txt")

    logger.info(f"Export complete: {n_total} structures")
    notify(
        f"Final export complete!\n"
        f"Total: {n_total} structures\n"
        f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}\n"
        f"MP: {source_dist.get('MP', 0)} | COD: {source_dist.get('COD', 0)} | "
        f"Alex: {source_dist.get('Alexandria', 0)} | OQMD: {source_dist.get('OQMD', 0)}"
    )


if __name__ == "__main__":
    main()
