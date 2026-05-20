# Data

All training data and source files for the project.

Note: The full base model training CSVs (`merged_dataset/train.csv` etc.) are **not included** in this repository — they are ~400 MB each and contain full CIF strings. Only metadata-only versions (without CIF columns) are provided in `base_metadata/`.

---

## data/source/

Source and intermediate pipeline data for the magnetocaloric fine-tuning dataset.

| File | Rows | Description |
|------|------|-------------|
| `magnetocaloric_materials.csv` | 641 rows (409 unique compositions) | Franco et al. source data — formula, prototype, T_C_K, ΔS (J/kg·K), RCP (J/kg), applied field, material class. Primary input to the CIF pipeline. |
| `magnetocaloric_materials_with_wyckoff.csv` | 743 rows | Pipeline output linking formula to CIF path and Wyckoff assignments. |
| `duplicates.csv` | ~74 groups | Duplicate CIF groups identified by `find_duplicate_cifs.py`. Used to drop 121 redundant structures, leaving 374 unique CIFs. |
| `dopant_approximations.csv` | ~300 rows | Records stoichiometric approximations made when assigning minority dopants to ordered SQS cells. |
| `Failed_materials.csv` | ~11 rows | Materials where neither prototype lookup nor direct MP structure was found. |
| `cif_quality_report.csv` | ~629 rows | Per-CIF quality metrics: atom count, stoichiometry error, parse status, space group. |

---

## data/finetune/

Final fine-tuning dataset for magnetocaloric conditioning. These are the **repaired** versions — use these, not the raw pipeline outputs.

| File | Rows | Description |
|------|------|-------------|
| `train_repaired.csv` | 299 | Training set. 292 rows have ΔS, 298 have T_C. 6 structures exceed 51 atoms and are silently filtered by MatterGen's dataloader. Property-mismapping bug in the raw pipeline output was fixed by composition-based matching. |
| `val_repaired.csv` | 31 | Validation set. Only 16 rows have data; 15 are empty padding rows (drop with `dropna(subset=['n_atoms_primitive'])`). |

Columns: `material_id, cif, n_atoms_primitive, converged, space_group_number, magnetocaloric_delta_s, magnetocaloric_field, magnetocaloric_curie_temp`

Some `magnetocaloric_curie_temp` values are empty strings (not NaN) — use `pd.to_numeric(..., errors='coerce')`.

---

## data/superconductor/

Superconductor fine-tuning dataset (1,088 structures total).

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 1,000 | SC training set. Column: `sc_transition_temp` (K). Median = 3.80 K, std = 36.53 K. Source: NEMAD superconductor database + Materials Project structure matching. |
| `val.csv` | 88 | SC validation set. |
| `superconductor_materials_unique_nemad.csv` | 19,058 | Full NEMAD superconductor database before MP matching. |

---

## data/base_metadata/

Metadata-only views of the base pre-training dataset splits. CIF strings are excluded to keep file sizes manageable.

| File | Rows | Description |
|------|------|-------------|
| `base_train_metadata.csv` | 219,679 | Training split metadata (no CIF column). |
| `base_val_metadata.csv` | 27,457 | Validation split metadata. |
| `base_test_metadata.csv` | 27,466 | Test split metadata. |

Total base dataset: **274,602 structures.**

Columns: `material_id, pretty_formula, elements, spacegroup_number, source, formation_energy_per_atom, dft_band_gap, e_above_hull` (plus any NEMAD property annotations).

The full CSVs with embedded CIF strings live at `/home/daniel/base_model_dataset/merged_dataset/` on the local machine and are also available on Stanage HPC. They are not committed to this repository due to size (~400 MB per split).
