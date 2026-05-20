# Docs

Reference data and pipeline documentation files.

---

## Files

| File | Description |
|------|-------------|
| `mattergen_project_reference.md` | Comprehensive project reference document covering the full MatterGen pipeline architecture, training hyperparameters, data file conventions, figure specifications, and known issues. Maintained in the `Data_analysis/` session. |
| `relaxation_results.csv` | Per-structure CHGNet relaxation summary: material_id, formula, convergence status, energy, forces. 547 KB, ~300 rows covering all CIFs that passed through the relaxation pipeline. |
| `cif_quality_report.csv` | Per-CIF quality assessment from `analyze_cif_quality.py`: atom count, stoichiometry error (%), space group number, parse status (pass/warn/fail), CIF path. 122 KB, ~629 rows. |
| `dopant_approximations.csv` | Records the stoichiometric approximations made when assigning minority dopants to ordered SQS/ODST cells. Shows target vs actual per-atom fractions and approximation error. 5.7 KB. |
| `duplicates.csv` | Full duplicate group analysis from `find_duplicate_cifs.py`. Each group lists the duplicate CIFs detected by StructureMatcher, with the recommended CIF to keep and the others to drop. 28 KB. |

---

## Notes

The `Processing_data/` directory in `Data_analysis/` also contains `full_metadata.csv` (486 MB) — a complete structural metadata file covering all base dataset structures. This is too large to commit and is not included here.
