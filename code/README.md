# Code

All Python scripts for the project, organised by pipeline stage.

## data_parsing/

Scripts for building the magnetocaloric fine-tuning dataset from the Franco et al. source CSV.

| Script | Description |
|--------|-------------|
| `magnetocaloric_wyckoff_pipeline_v14.py` | **Final version.** Reads `magnetocaloric_materials.csv`, fetches prototype structures from Materials Project, assigns Wyckoff sites, builds ordered CIFs, validates stoichiometry. Outputs ~374 CIFs to `clean_cifs/`. |
| `step1_generate_sqs_v2.py` | SQS ordering for disordered structures (alloys with fractional occupancy). Uses sqsgenerator. Writes ordered CIFs to `cif_files/`. |
| `find_duplicate_cifs.py` | Finds duplicate CIF groups using StructureMatcher. Outputs `duplicates.csv`. |
| `extract_approved_cifs.py` | Filters CIFs to approved deduplicated set. |
| `process_icsd_failed.py` | Processes ICSD template CIFs for materials that failed the standard pipeline (e.g. Gd5(Si,Ge)4 monoclinic phase). |
| `analyze_cif_quality.py` | Reports CIF quality metrics (stoichiometry errors, parse failures). Outputs `cif_quality_report.csv`. |
| `validate_franco_dataset.py` | Validates the Franco et al. CSV against known material properties. |

## structure_pipeline/

Scripts for CIF relaxation and training dataset assembly.

| Script | Description |
|--------|-------------|
| `step1_generate_sqs_v2.py` | SQS ordering for disordered CIFs (copy from data_parsing for clarity). |
| `step2_relax_and_validate_v2.py` | CHGNet relaxation and structure validation step. |
| `prepare_dataset.py` | **Final pipeline runner.** Filters CIFs to `best_attempt_cifs/`, runs CHGNet relaxation (277 structures), attaches magnetocaloric properties (ΔS, T_C, field), exports `training_data/train.csv` and `val.csv`. Most recent run: April 16 2026. |

## base_dataset/

8-stage pipeline to assemble the 274,602-structure base pre-training dataset from MP, COD, Alexandria, and NEMAD.

| Script | Description |
|--------|-------------|
| `01_filter_cod.py` | Filters 533,464 COD CIF files: removes partial occupancy, >51 atoms, organic, theoretical. Outputs ~33,791 structures. Uses per-file 60s timeout for SpacegroupAnalyzer. |
| `02_load_filter_mp.py` | Loads Materials Project JSON chunks, applies ≤51-atom / no partial occupancy / inorganic filters. Outputs ~36,827 structures. |
| `02b_load_filter_alex.py` | Loads and filters Alexandria database. |
| `02c_resume_alex.py` | Resume-capable version of the Alexandria loader. |
| `03_integrate_nemad.py` | Annotates MP structures with NEMAD property labels (magnetic, SC, thermoelectric). |
| `04_merge_dedup.py` | Merges all sources, deduplicates with StructureMatcher (priority: MP > COD > Alexandria > OQMD). |
| `05_export_mattergen.py` | Exports final MatterGen-format CSVs (material_id, pretty_formula, elements, cif, spacegroup_number, formation_energy_per_atom, dft_band_gap, e_above_hull). |
| `06_scrape_oqmd.py` | Scrapes OQMD structures as supplementary source. |
| `07_load_filter_mp_full.py` | Extended MP load for the full dataset merge. |
| `08_final_merge.py` | Final merge and deduplication across all sources to produce `merged_dataset/train.csv`, `val.csv`, `test.csv`. |

## analysis/

Figure generation scripts and loss-curve parsers. All scripts use `matplotlib.use("Agg")` for WSL compatibility.

### Loss parsers / curve plotters (existing)

| Script | Description |
|--------|-------------|
| `parse_pretrain.py` | Parses `pretrain_4.out` SLURM log → `losses_csvs/pretrain_losses.csv`. |
| `parse_finetune_1000.py` | Parses finetune SLURM log → `losses_csvs/finetune_1000_losses.csv`. |
| `plot_pretrain_loss.py` | 3-panel pretrain loss curve + convergence analysis. |
| `plot_finetune_1000_loss.py` | 2-panel ΔS fine-tune loss curve. |
| `plot_finetune_dataset.py` | Atom-count + ΔS histograms (superseded by fig01/fig08). |
| `plot_finetune_properties.py` | ΔS + T_C property distribution histograms (superseded by fig08). |
| `plot_joint_finetune_loss.py` | 3-panel joint fine-tune loss curve. |
| `plot_sc_finetune_loss.py` | 3-panel superconductor fine-tune loss curve. |
| `plot_dyco_hull.py` | Dy-Co binary convex hull with structure insets (appendix figure). |

### Figure scripts (fig01–fig29)

| Script | Output | Description |
|--------|--------|-------------|
| `fig01_atom_count.py` | `fig01_atom_count_distribution` | Atom-count histogram of fine-tuning dataset |
| `fig02_match_type.py` | `fig02_match_type_breakdown` | Structure match type and ordered/SQS split |
| `fig03_stoich_errors.py` | `fig03_stoichiometry_errors` | Distribution of stoichiometric errors |
| `fig04_pipeline_schematic.py` | `fig04_pipeline_schematic` | Full dataset construction pipeline flowchart |
| `fig05_base_dataset.py` | `fig05_base_dataset_composition` | Base dataset atom count, sources, phase systems |
| `fig06_property_repair.py` | `fig06_property_repair` | Before/after scatter for property repair |
| `fig07_cod_waterfall.py` | `fig07_cod_filtering_waterfall` | COD filtering waterfall (533k → 33.8k) |
| `fig08_property_distributions.py` | `fig08_property_distributions` | ΔS, T_C, and field violin plots |
| `fig09_material_class.py` | `fig09_material_class_breakdown` | Unique compositions by material class |
| `fig10_delta_s_vs_tc.py` | `fig10_delta_s_vs_tc_scatter` | ΔS vs T_C scatter coloured by space group |
| `fig11_convergence_comparison.py` | `fig11_convergence_comparison` | 1×4 loss curve comparison (all 4 training runs) |
| `fig12_source_overlap.py` | `fig12_source_overlap` | Base dataset source contributions after dedup |
| `fig13_sun_vs_conditioning.py` | `fig13_sun_vs_conditioning` | S.U.N. metrics vs conditioning level |
| `fig14_element_frequency.py` | `fig14_element_frequency` | Element frequency in generated structures |
| `fig15_space_group_distribution.py` | `fig15_space_group_distribution` | Space group distribution comparison |
| `fig16_joint_conditioning_sweep.py` | `fig16_joint_conditioning_sweep` | Joint conditioning sweep results |
| `fig17_sun_comparison.py` | `fig17_sun_comparison` | S.U.N. comparison across model variants |
| `fig18_finetune_vs_mattergen.py` | `fig18_finetune_vs_mattergen` | Fine-tuned vs baseline MatterGen metrics |
| `fig19_space_group_comparison.py` | `fig19_space_group_comparison` | Space group comparison across models |
| `fig20_ds_conditioning_lines.py` | `fig20_ds_conditioning_lines` | ΔS conditioning line plots |
| `fig21_ds_conditioning_combined.py` | `fig21_ds_conditioning_combined` | Combined ΔS conditioning summary |
| `fig22_element_freq_by_ds.py` | `fig22_element_freq_by_ds` | Element frequency broken down by ΔS level |
| `fig23_joint_conditioning_components.py` | `fig23_joint_conditioning_components` | Joint conditioning component analysis |
| `fig24_ehull_violin.py` | `fig24_ehull_violin` | Energy-above-hull violin by conditioning level |
| `fig25_mce_family_classification.py` | `fig25_mce_family_classification` | MCE material family classification of generated structures |
| `fig26_base_dataset_sources.py` | `fig26_base_dataset_sources` | Base dataset source breakdown bar chart |
| `fig27_sc_joint_loss_comparison.py` | `fig27_sc_joint_loss_comparison` | SC vs joint fine-tune loss comparison |
| `fig28_tc_vs_sc_sun_comparison.py` | `fig28_tc_vs_sc_sun_comparison` | T_C vs SC transition temperature S.U.N. comparison |
| `fig29_property_distributions_comparison.py` | `fig29_property_distributions_comparison` | Property distribution comparison across models |

## slurm_scripts/

No SLURM `.sbatch` files were found locally — all training was launched directly on Stanage HPC and scripts are not available in the local working directory. Equivalent training commands are documented in `configs/pretrain_resolved_config.yaml` and the finetune resolved configs.
