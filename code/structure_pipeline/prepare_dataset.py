#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py (v2) — Magnetocaloric MatterGen Dataset Preparation
Phases:
  0 — MatterGen source modifications (globals.py + yaml configs)
  1 — Populate best_attempt_cifs/ (SQS preference, order check, size, stoich, dedup)
  2 — CHGNet relaxation + CSV construction
  3 — MatterGen CSV validation

v2 fixes over v1:
  FIX 1  parse_target_formula: strip spaces before p→. regex
  FIX 2  Phase 2: skip CHGNet if already in relaxed_best_attempt_cifs/
  FIX 3  Eu_Rh0p6Zn0p4_2_relaxed.cif treated as already-relaxed source
  FIX 4  Strip 'Hd' from material_id (not filename); log HD_STRIPPED
  FIX 5  Load CSV with encoding_errors='replace'; PROP_OVERRIDES for 4 stems;
         log UNRESOLVED_PROPERTY_NOT_FOUND for any remaining misses
  NEW    MANUALLY_EXCLUDED set (7 confirmed-bad files)
  NEW    MANUALLY_RECOVERED post-loop (2 files force-added despite dedup)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import CLEAN_PREPROCESSING_WORK, MATTERGEN_DIR  # noqa: E402


import csv
import math
import os
import re
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
WORK_DIR         = CLEAN_PREPROCESSING_WORK
MATTERGEN_DIR    = MATTERGEN_DIR
CIF_SOURCE       = WORK_DIR / "cif_files"
BEST_ATTEMPT_DIR = WORK_DIR / "best_attempt_cifs"
RELAXED_DIR      = WORK_DIR / "relaxed_best_attempt_cifs"
TRAINING_DIR     = WORK_DIR / "training_data"
PROPS_CSV        = WORK_DIR / "magnetocaloric_materials.csv"
LOG_FILE         = WORK_DIR / "dataset_log.txt"

GLOBALS_PY     = MATTERGEN_DIR / "mattergen/common/utils/globals.py"
PROP_EMBED_DIR = (MATTERGEN_DIR /
    "mattergen/conf/lightning_module/diffusion_module/model/property_embeddings")
TEMPLATE_YAML  = PROP_EMBED_DIR / "dft_mag_density.yaml"
NEW_PROP_IDS   = [
    "magnetocaloric_delta_s",
    "magnetocaloric_field",
    "magnetocaloric_curie_temp",
]

# ──────────────────────────────────────────────────────────────────────────────
# Manual review decisions (v2)
# ──────────────────────────────────────────────────────────────────────────────

# Files confirmed bad — excluded regardless of Phase 1 filters
MANUALLY_EXCLUDED = {
    "La0p7Ag0p3MnO3_sqs.cif":            "Too much La and O",
    "La1p4_Sr0p4Ba0p6_1p6Mn2O7_sqs.cif": "1 too many La, 1 too few Mn",
    "La1p4_Sr0p8Ba0p2_1p6Mn2O7_sqs.cif": "Same problem as above",
    "La_Fe0p92Co0p08_11p9Si1p1_sqs.cif":  "1 too many Co, 1 too few Fe",
    "MnAs0p97Si0p03N_sqs.cif":            "Bad stoichiometry",
    "Tb_Co0p95Fe0p5_2_sqs.cif":           "Bad stoichiometry",
    "LaFe10p98Co0p22Si1p8_sqs.cif":       "1 too many Co, 1 too few Fe",
}

# Files confirmed good — force-added to best_attempt_cifs/ after dedup loop
MANUALLY_RECOVERED = [
    "Gd5_Ge0p9175Si0p0825_4_sqs.cif",
    "La_Fe0p96Co0p04_11p9Si1p1_sqs.cif",
]

# CIF stem (no extension) → (T_C_K, field_T, delta_S) for files where
# composition matching fails (garbled CSV formula or SQS dev > 5%)
PROP_OVERRIDES: dict[str, tuple] = {
    "La2p05K0p95Mn2O7_sqs":            (254.0, 1.0,  1.1),
    "La2p15K0p85Mn2O7_sqs":            (247.0, 1.0,  1.4),
    "Gd5Si0p935Ge2p035Sn0p03_sqs":     (277.0, 2.0, 28.9),
    "La_Fe0p97Mn0p03_11p7Si1p3Hd_sqs": (287.0, 5.0, 15.9),
}

# Files in cif_files/ that are already relaxed outputs (not raw SQS)
# Maps source_filename → dest_filename (stored in both best_attempt_cifs/ and
# relaxed_best_attempt_cifs/ during Phase 1 so FIX 2 skips CHGNet for them)
ALREADY_RELAXED_SOURCES: dict[str, str] = {
    "Eu_Rh0p6Zn0p4_2_relaxed.cif": "Eu_Rh0p6Zn0p4_2.cif",
}
_ALREADY_RELAXED_DEST_NAMES = set(ALREADY_RELAXED_SOURCES.values())

# ──────────────────────────────────────────────────────────────────────────────
# Logging — append mode, never overwrite
# ──────────────────────────────────────────────────────────────────────────────
_log_fh = open(LOG_FILE, "a", encoding="utf-8")

def log(msg: str = "") -> None:
    print(msg, flush=True)
    _log_fh.write(msg + "\n")
    _log_fh.flush()

log()
log("=" * 60)
log(f"Dataset preparation run (v2): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

from dataset_helpers import (  # noqa: E402  (moved so they can be tested)
    fmt_val,
    nan_or_float,
    normalise_stem,
    parse_target_formula,
    prop_sort_key,
)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — MATTERGEN SOURCE MODIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

log()
log("─" * 60)
log("PHASE 0 — MatterGen source modifications")
log("─" * 60)

# ── 0a: globals.py — append missing property source IDs ──────────────────────
log("0a: Editing globals.py ...")
content = GLOBALS_PY.read_text(encoding="utf-8")
for prop_id in NEW_PROP_IDS:
    if f'"{prop_id}"' in content or f"'{prop_id}'" in content:
        log(f"  '{prop_id}' already present — skipping")
        continue
    content = re.sub(
        r'(PROPERTY_SOURCE_IDS\s*=\s*\[.*?)\n(\s*\])',
        lambda m: m.group(1) + f'\n    "{prop_id}",\n' + m.group(2),
        content,
        flags=re.DOTALL,
        count=1,
    )
    log(f"  Added '{prop_id}' to PROPERTY_SOURCE_IDS")
GLOBALS_PY.write_text(content, encoding="utf-8")
log("  globals.py saved.")

# ── 0b: Create property embedding yaml configs ────────────────────────────────
log("0b: Creating property embedding yaml configs ...")
template_content = TEMPLATE_YAML.read_text(encoding="utf-8")
for prop_name in NEW_PROP_IDS:
    dest = PROP_EMBED_DIR / f"{prop_name}.yaml"
    if dest.exists():
        log(f"  {dest.name} already exists — skipping")
        continue
    new_content = template_content.replace("name: dft_mag_density", f"name: {prop_name}", 1)
    dest.write_text(new_content, encoding="utf-8")
    log(f"  Created {dest.name}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — POPULATE best_attempt_cifs/
# ══════════════════════════════════════════════════════════════════════════════

log()
log("─" * 60)
log("PHASE 1 — Populate best_attempt_cifs/")
log("─" * 60)

# Ensure persistent output dirs exist before any clearing
RELAXED_DIR.mkdir(exist_ok=True)
TRAINING_DIR.mkdir(exist_ok=True)

# ── Load previous convergence records (before clearing train/val CSVs) ────────
prev_convergence: dict[str, bool] = {}
for csv_path in [TRAINING_DIR / "train.csv", TRAINING_DIR / "val.csv"]:
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid  = row.get("material_id", "")
                conv = row.get("converged", "").strip().lower()
                if mid:
                    prev_convergence[mid] = (conv == "true")
log(f"Loaded {len(prev_convergence)} previous convergence records from train/val CSVs.")

# ── Clear best_attempt_cifs/ and recreate fresh ────────────────────────────────
if BEST_ATTEMPT_DIR.exists():
    shutil.rmtree(BEST_ATTEMPT_DIR)
BEST_ATTEMPT_DIR.mkdir()
log(f"Cleared and recreated {BEST_ATTEMPT_DIR}")

# Counters
cnt_examined           = 0
cnt_passed             = 0
cnt_excl_non_sqs       = 0
cnt_excl_disordered    = 0
cnt_excl_too_large     = 0
cnt_excl_duplicate     = 0
cnt_excl_manual        = 0
cnt_manually_recovered = 0
cnt_parse_fail         = 0
cnt_stoich_warn        = 0
stoich_warn_files: list[str] = []

# ── Group all CIFs by normalised stem (exclude ALREADY_RELAXED_SOURCES) ───────
_ars_src_names = set(ALREADY_RELAXED_SOURCES.keys())
all_cifs = [
    p for p in sorted(CIF_SOURCE.glob("*.cif"))
    if p.name not in _ars_src_names
]
cnt_examined = len(all_cifs)
log(f"Found {cnt_examined} CIF files in {CIF_SOURCE} (excluding {len(_ars_src_names)} already-relaxed sources)")

stem_groups: dict[str, list[Path]] = defaultdict(list)
for p in all_cifs:
    stem_groups[normalise_stem(p.stem)].append(p)

# ── RULE 1: SQS preference — within each stem group ──────────────────────────
candidates: list[Path] = []
for norm_stem, paths in stem_groups.items():
    sqs_paths     = [p for p in paths if '_sqs' in p.stem.lower()]
    non_sqs_paths = [p for p in paths if '_sqs' not in p.stem.lower()]
    if sqs_paths:
        for p in non_sqs_paths:
            sqs_names = ", ".join(s.name for s in sqs_paths)
            log(f"EXCLUDED_NON_SQS: {p.name} (superseded by {sqs_names})")
            cnt_excl_non_sqs += 1
        candidates.extend(sqs_paths)
    else:
        candidates.extend(non_sqs_paths)

log(f"Candidates after SQS preference: {len(candidates)}")

# ── RULES 2-4: Ordered check, primitive size, stoich warning ─────────────────
# Each passing entry: (path, prim_structure, target_comp, max_dev, dup_key)
passing: list[tuple] = []

for cif_path in sorted(candidates):
    filename = cif_path.name

    # MANUALLY_EXCLUDED check (applied before any parsing)
    if filename in MANUALLY_EXCLUDED:
        log(f"MANUALLY_EXCLUDED: {filename} ({MANUALLY_EXCLUDED[filename]})")
        cnt_excl_manual += 1
        continue

    # Parse target formula from filename (FIX 1 applied in parse_target_formula)
    target_comp, err = parse_target_formula(filename)
    if target_comp is None:
        log(f"PARSE_FAIL: {filename}: {err}")
        cnt_parse_fail += 1
        continue

    # RULE 2: Parse CIF and check all occupancies == 1
    try:
        parser     = CifParser(str(cif_path))
        structures = parser.parse_structures(primitive=False, on_error="ignore")
        if not structures:
            raise ValueError("CifParser returned no structures")
        structure = structures[0]
    except Exception as e:
        log(f"PARSE_ERROR: {filename}: {e}")
        cnt_parse_fail += 1
        continue

    ordered = all(
        abs(occ - 1.0) < 1e-3
        for site in structure
        for occ in site.species.values()
    )
    if not ordered:
        log(f"EXCLUDED_DISORDERED: {filename}")
        cnt_excl_disordered += 1
        continue

    # RULE 3: Primitive cell atom count strictly <= 64
    try:
        prim = structure.get_primitive_structure()
    except Exception as e:
        log(f"PARSE_ERROR: {filename}: get_primitive_structure failed: {e}")
        cnt_parse_fail += 1
        continue

    if len(prim) > 64:
        log(f"EXCLUDED_TOO_LARGE: {filename} ({len(prim)} atoms in primitive)")
        cnt_excl_too_large += 1
        continue

    # RULE 4: Stoichiometry check — warn but do not exclude
    actual_fc  = Composition(structure.composition).fractional_composition
    target_fc  = target_comp.fractional_composition
    all_elems  = set(actual_fc.keys()) | set(target_fc.keys())
    max_dev    = max(abs(actual_fc.get(el, 0) - target_fc.get(el, 0)) for el in all_elems)
    if max_dev > 0.05:
        log(f"STOICH_WARNING: {filename} | max_deviation={max_dev:.4f} | "
            f"actual={actual_fc} | target={target_fc}")
        cnt_stoich_warn += 1
        stoich_warn_files.append(filename)

    # Duplicate key: frozenset of (element, count) pairs in primitive cell
    dup_key = frozenset(Counter(site.specie.symbol for site in prim).items())
    passing.append((cif_path, prim, target_comp, max_dev, dup_key))

# ── RULE 5: Duplicate detection ───────────────────────────────────────────────
dup_groups: dict = defaultdict(list)
for item in passing:
    dup_groups[item[4]].append(item)

final_items: list[tuple] = []
for dup_key, items in dup_groups.items():
    if len(items) == 1:
        final_items.append(items[0])
        continue
    # Keep the entry with smallest max_dev; ties broken alphabetically
    items_sorted = sorted(items, key=lambda x: (round(x[3], 4), x[0].name))
    kept = items_sorted[0]
    final_items.append(kept)
    for item in items_sorted[1:]:
        log(f"EXCLUDED_DUPLICATE: {item[0].name} | "
            f"kept={kept[0].name} | "
            f"actual_composition={item[1].composition} | "
            f"note=unrepresentable at this SQS resolution")
        cnt_excl_duplicate += 1

# ── Copy passing files to best_attempt_cifs/ ─────────────────────────────────
for item in final_items:
    shutil.copy2(str(item[0]), str(BEST_ATTEMPT_DIR / item[0].name))
    cnt_passed += 1

# ── Post-loop: handle ALREADY_RELAXED_SOURCES ─────────────────────────────────
for src_name, dest_name in ALREADY_RELAXED_SOURCES.items():
    src = CIF_SOURCE / src_name
    if src.exists():
        shutil.copy2(str(src), str(BEST_ATTEMPT_DIR / dest_name))
        shutil.copy2(str(src), str(RELAXED_DIR / dest_name))
        log(f"ALREADY_RELAXED_SOURCE: {src_name} → {dest_name} "
            f"(added to best_attempt_cifs/ and relaxed_best_attempt_cifs/)")
        cnt_passed += 1
    else:
        log(f"WARNING: ALREADY_RELAXED_SOURCE {src_name} not found in {CIF_SOURCE}")

# ── Post-loop: MANUALLY_RECOVERED — force-add despite dedup ──────────────────
for recover_name in MANUALLY_RECOVERED:
    src = CIF_SOURCE / recover_name
    dest = BEST_ATTEMPT_DIR / recover_name
    if src.exists():
        shutil.copy2(str(src), str(dest))
        log(f"MANUALLY_RECOVERED: {recover_name} — force-added to best_attempt_cifs/")
        cnt_passed += 1
        cnt_manually_recovered += 1
    else:
        log(f"WARNING: MANUALLY_RECOVERED {recover_name} not found in {CIF_SOURCE}")

# ── Phase 1 summary ───────────────────────────────────────────────────────────
log()
log("─" * 40)
log("Phase 1 Summary")
log("─" * 40)
log(f"Total CIFs examined:              {cnt_examined}")
log(f"Passed to best_attempt_cifs/:     {cnt_passed}")
log(f"EXCLUDED_NON_SQS:                 {cnt_excl_non_sqs}")
log(f"EXCLUDED_DISORDERED:              {cnt_excl_disordered}")
log(f"EXCLUDED_TOO_LARGE:               {cnt_excl_too_large}")
log(f"EXCLUDED_DUPLICATE:               {cnt_excl_duplicate}")
log(f"MANUALLY_EXCLUDED:                {cnt_excl_manual}")
log(f"MANUALLY_RECOVERED:               {cnt_manually_recovered}")
log(f"PARSE_FAIL / PARSE_ERROR:         {cnt_parse_fail}")
log(f"STOICH_WARNING (manual review):   {cnt_stoich_warn}")
if stoich_warn_files:
    for f in stoich_warn_files:
        log(f"  {f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHGNET RELAXATION AND CSV CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

log()
log("─" * 60)
log("PHASE 2 — CHGNet relaxation and CSV construction")
log("─" * 60)

# ── Load and validate property CSV (FIX 5: encoding_errors='replace') ─────────
props_df = pd.read_csv(PROPS_CSV, encoding="utf-8", encoding_errors="replace")  # FIX 5
required_cols = {"material_formula", "T_C_K", "applied_field_H_T", "delta_S_mag_J_per_kgK"}
missing_cols  = required_cols - set(props_df.columns)
if missing_cols:
    _log_fh.close()
    raise RuntimeError(f"Property CSV missing required columns: {missing_cols}")

prop_entries: list[tuple] = []   # (fractional_Composition, T_C_K, field, delta_s)
prop_parse_failures = 0
for _, row in props_df.iterrows():
    formula_str = str(row["material_formula"])
    try:
        fc = Composition(formula_str).fractional_composition
    except Exception:
        prop_parse_failures += 1
        continue
    prop_entries.append((fc, row["T_C_K"], row["applied_field_H_T"], row["delta_S_mag_J_per_kgK"]))

log(f"Loaded {len(prop_entries)} valid property rows from {PROPS_CSV.name} "
    f"({prop_parse_failures} rows skipped due to unparseable formula)")

# ── Sort best_attempt_cifs/ by ascending primitive atom count ─────────────────
def get_prim_size(p: Path) -> int:
    try:
        s = CifParser(str(p)).parse_structures(primitive=False, on_error="ignore")[0]
        return len(s.get_primitive_structure())
    except Exception:
        return 9999

log("Sorting best_attempt_cifs/ by primitive atom count (smallest first) ...")
best_cifs = sorted(BEST_ATTEMPT_DIR.glob("*.cif"), key=get_prim_size)
log(f"Found {len(best_cifs)} CIFs to process.")

# ── Load CHGNet — FIRE optimizer by default ───────────────────────────────────
log("Loading CHGNet model ...")
from chgnet.model import CHGNet
from chgnet.model.dynamics import StructOptimizer

CHGNet.load()
relaxer = StructOptimizer()         # optimizer_class="FIRE" is the default
log("CHGNet model loaded.")

# ── Relaxation loop ───────────────────────────────────────────────────────────
cnt_relaxed              = 0
cnt_converged            = 0
cnt_not_converged        = 0
cnt_skipped_relax        = 0
cnt_relax_error          = 0
cnt_prop_not_found       = 0
cnt_prop_override        = 0
cnt_fully_labelled       = 0
cnt_partly_labelled      = 0
cnt_unlabelled           = 0
prop_not_found_files: list[str] = []
rows: list[dict] = []

for cif_path in best_cifs:
    filename = cif_path.name
    stem     = cif_path.stem
    log(f"\nProcessing: {filename}")

    # ── Step 1: Parse CIF and pre-check primitive size ────────────────────────
    try:
        structure = CifParser(str(cif_path)).parse_structures(
            primitive=False, on_error="ignore")[0]
        prim_pre  = structure.get_primitive_structure()
        if len(prim_pre) > 64:
            log(f"WARNING_POST_PRIM: {filename} ({len(prim_pre)} atoms) skipped")
            continue
    except Exception as e:
        log(f"PARSE_ERROR: {filename}: {e}")
        cnt_relax_error += 1
        continue

    # ── Compute material_id (FIX 4: strip _sqs suffix and Hd) ────────────────
    material_id = re.sub(r'_sqs$', '', stem, flags=re.IGNORECASE)
    if 'Hd' in material_id:
        original_mid = material_id
        material_id  = material_id.replace('Hd', '')
        log(f"HD_STRIPPED: {original_mid} → {material_id}")

    # ── Step 2: CHGNet relaxation — skip if already done (FIX 2 + FIX 3) ─────
    out_path    = RELAXED_DIR / filename
    skip_chgnet = False
    relaxed     = None

    if out_path.exists():
        try:
            relaxed = CifParser(str(out_path)).parse_structures(
                primitive=False, on_error="ignore")[0]
            skip_chgnet = True
        except Exception as e:
            log(f"RELOAD_ERROR: {filename}: {e} — will re-relax")

    if skip_chgnet:
        assert relaxed is not None
        converged = prev_convergence.get(stem, prev_convergence.get(material_id, False))
        max_f     = float('nan')
        cnt_relaxed += 1
        cnt_skipped_relax += 1
        if converged:
            cnt_converged += 1
        else:
            cnt_not_converged += 1
        # FIX 3: distinguish already-relaxed source from normal skip
        if filename in _ALREADY_RELAXED_DEST_NAMES:
            log(f"ALREADY_RELAXED_SOURCE: {filename} — loaded from pre-relaxed source, skipping CHGNet")
        else:
            log(f"SKIPPED_RELAXATION: {filename} — loaded from existing relaxed CIF "
                f"(converged={converged})")
    else:
        # ── CHGNet relaxation ─────────────────────────────────────────────────
        try:
            result    = relaxer.relax(structure, fmax=0.05, steps=500, verbose=False)
            relaxed   = result["final_structure"]
            traj      = result["trajectory"]
            if len(traj.forces) > 0:
                max_f     = float(max(np.linalg.norm(f) for f in traj.forces[-1]))
                converged = max_f <= 0.05
            else:
                max_f     = 0.0
                converged = True
        except Exception as e:
            log(f"RELAX_ERROR: {filename}: {e}")
            cnt_relax_error += 1
            continue

        # ── Save relaxed CIF immediately ──────────────────────────────────────
        try:
            CifWriter(relaxed).write_file(str(out_path))
        except Exception as e:
            log(f"WRITE_ERROR: {filename}: {e} — trying fallback")
            try:
                relaxed.to(filename=str(out_path), fmt="cif")
            except Exception as e2:
                log(f"WRITE_ERROR_FALLBACK: {filename}: {e2} — skipping")
                cnt_relax_error += 1
                continue

        cnt_relaxed += 1
        if converged:
            cnt_converged += 1
            log(f"  Relaxed OK (converged), max_force={max_f:.4f} eV/Å")
        else:
            cnt_not_converged += 1
            log(f"  Relaxed (NOT converged), max_force={max_f:.4f} eV/Å")

    # ── Step 3: Post-relaxation primitive reduction ───────────────────────────
    assert relaxed is not None
    try:
        spa        = SpacegroupAnalyzer(relaxed, symprec=0.1)
        final_prim = spa.get_primitive_standard_structure()
        sg_number  = spa.get_space_group_number()
    except Exception:
        final_prim = relaxed.get_primitive_structure()
        sg_number  = 1
        log(f"WARNING_SGA_FAILED: {filename} — fell back to get_primitive_structure()")

    # ── Step 4: Property matching (FIX 5: PROP_OVERRIDES + UNRESOLVED log) ────
    if stem in PROP_OVERRIDES:
        # Hard-coded override for known problematic stems
        t_c_k_val, field_raw, ds_raw = PROP_OVERRIDES[stem]
        curie_temp = nan_or_float(t_c_k_val)
        field_val  = nan_or_float(field_raw)
        delta_s    = nan_or_float(ds_raw)
        log(f"PROP_OVERRIDE: {filename} → T_C={curie_temp}, field={field_val}, dS={delta_s}")
        cnt_prop_override += 1
    else:
        actual_fc = Composition(relaxed.composition).fractional_composition
        matching: list[tuple] = []
        for csv_fc, t_c_k, field_raw, ds_raw in prop_entries:
            all_elems = set(actual_fc.keys()) | set(csv_fc.keys())
            dev = max(abs(actual_fc.get(el, 0) - csv_fc.get(el, 0)) for el in all_elems)
            if dev <= 0.05:
                matching.append((csv_fc, t_c_k, field_raw, ds_raw))

        if not matching:
            delta_s    = float('nan')
            field_val  = float('nan')
            curie_temp = float('nan')
            log(f"UNRESOLVED_PROPERTY_NOT_FOUND: {filename}")  # FIX 5
            cnt_prop_not_found += 1
            prop_not_found_files.append(filename)
        else:
            matching.sort(key=prop_sort_key)
            _, t_c_k_val, field_raw, ds_raw = matching[0]
            delta_s    = nan_or_float(ds_raw)
            field_val  = nan_or_float(field_raw)
            curie_temp = nan_or_float(t_c_k_val)

    # ── Step 5: Build CSV row ─────────────────────────────────────────────────
    cif_string = out_path.read_text(encoding="utf-8")

    nan_count = sum(
        1 for v in (delta_s, field_val, curie_temp)
        if isinstance(v, float) and math.isnan(v)
    )
    if nan_count == 0:
        cnt_fully_labelled += 1
    elif nan_count == 3:
        cnt_unlabelled += 1
    else:
        cnt_partly_labelled += 1

    rows.append({
        "material_id":               material_id,      # FIX 4: cleaned (no _sqs, no Hd)
        "cif":                       cif_string,
        "n_atoms_primitive":         len(final_prim),
        "converged":                 converged,
        "space_group_number":        sg_number,
        "magnetocaloric_delta_s":    fmt_val(delta_s),
        "magnetocaloric_field":      fmt_val(field_val),
        "magnetocaloric_curie_temp": fmt_val(curie_temp),
    })

log(f"\nRelaxation loop complete. {cnt_relaxed} structures successfully processed.")

# ── Train / val split (90 / 10) ───────────────────────────────────────────────
random.seed(42)
random.shuffle(rows)
n_val      = len(rows) // 10
n_train    = len(rows) - n_val
train_rows = rows[:n_train]
val_rows   = rows[n_train:]

# ── Write CSVs ────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "material_id", "cif", "n_atoms_primitive", "converged", "space_group_number",
    "magnetocaloric_delta_s", "magnetocaloric_field", "magnetocaloric_curie_temp",
]

def write_dataset_csv(path: Path, data: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDNAMES,
            quoting=csv.QUOTE_ALL, lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(data)

write_dataset_csv(TRAINING_DIR / "train.csv", train_rows)
write_dataset_csv(TRAINING_DIR / "val.csv",   val_rows)
log(f"Wrote train.csv ({len(train_rows)} rows) and val.csv ({len(val_rows)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — MATTERGEN CSV VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

log()
log("─" * 60)
log("PHASE 3 — MatterGen CSV validation")
log("─" * 60)

validation_result = "FAILED"
try:
    sys.path.insert(0, str(MATTERGEN_DIR))
    from mattergen.common.data.dataset import CrystalDatasetBuilder
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        CrystalDatasetBuilder.from_csv(
            csv_path=str(TRAINING_DIR / "train.csv"),
            cache_path=tmp,
        )
    print("✓ MatterGen CSV validation passed")
    log("VALIDATION: PASSED")
    validation_result = "PASSED"
except Exception as e:
    print(f"✗ MatterGen CSV validation FAILED: {e}")
    log(f"VALIDATION_FAILED: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

stoich_block  = "\n".join(f"  {f}" for f in stoich_warn_files)   or "  (none)"
prop_nf_block = "\n".join(f"  {f}" for f in prop_not_found_files) or "  (none)"

summary = f"""
══════════════════════════════════════════
DATASET PREPARATION COMPLETE (v2)
══════════════════════════════════════════
Phase 1 — CIF filtering:
  Input CIFs examined:       {cnt_examined}
  Passed to best_attempt:    {cnt_passed}
  Excluded (non-SQS):        {cnt_excl_non_sqs}
  Excluded (disordered):     {cnt_excl_disordered}
  Excluded (too large):      {cnt_excl_too_large}
  Excluded (duplicate):      {cnt_excl_duplicate}
  Manually excluded:         {cnt_excl_manual}
  Manually recovered:        {cnt_manually_recovered}
  Parse failures:            {cnt_parse_fail}
  Stoich warnings (review!): {cnt_stoich_warn}

Phase 2 — Relaxation:
  Successfully processed:    {cnt_relaxed}
  — CHGNet relaxed (new):    {cnt_relaxed - cnt_skipped_relax}
  — Loaded from cache:       {cnt_skipped_relax}
  Converged:                 {cnt_converged}
  Not converged (included):  {cnt_not_converged}
  Relax/parse errors:        {cnt_relax_error}
  Property overrides used:   {cnt_prop_override}
  Unresolved PROP_NOT_FOUND: {cnt_prop_not_found}
  Fully labelled rows:       {cnt_fully_labelled}
  Partially labelled:        {cnt_partly_labelled}
  Unlabelled:                {cnt_unlabelled}

Output:
  train.csv rows:            {len(train_rows)}
  val.csv rows:              {len(val_rows)}
  MatterGen validation:      {validation_result}

Files requiring manual review (STOICH_WARNING):
{stoich_block}

Files with UNRESOLVED_PROPERTY_NOT_FOUND:
{prop_nf_block}
══════════════════════════════════════════
"""

print(summary)
log(summary)
_log_fh.close()
