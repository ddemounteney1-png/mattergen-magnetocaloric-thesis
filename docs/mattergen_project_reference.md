# MatterGen Project Reference Document

## Purpose

This document reconstructs the complete methodology for building, training, and fine-tuning a custom MatterGen generative diffusion model for magnetocaloric crystal structure generation. It serves as a reference for writing the Materials and Methods section of the thesis. The document is structured in two halves: first, the methodology as it should be presented (the logical, cleaned-up narrative); second, a candid account of what actually happened, including mistakes, dead ends, and lessons learned.

---

# PART A — THE METHODOLOGY (AS IT SHOULD HAVE BEEN DONE)

This section presents the work in the order a reader would expect: problem definition, data acquisition, model preparation, training, and evaluation. It strips away the iterative debugging and presents the final, working approach.

---

## 1. Problem Definition and Motivation

The objective was to train a generative diffusion model capable of proposing novel magnetocaloric crystal structures conditioned on target properties (magnetic entropy change ΔS, Curie temperature T_C, and applied field H). Microsoft's MatterGen was selected as the base architecture — a score-based diffusion model that jointly denoises lattice parameters, atomic coordinates, and chemical species over 1000 diffusion timesteps, using a GemNetT graph neural network as the denoiser (44.6M parameters).

The published MatterGen base model (`mp_20_base`) was trained exclusively on Materials Project structures with ≤20 atoms per primitive unit cell. This created a fundamental incompatibility: many of the most scientifically important magnetocaloric material families have significantly larger unit cells. The La(Fe,Si)₁₃ family (NaZn₁₃-type, Fm-3c) has 28 atoms in the primitive cell. The Gd₅(Si,Ge)₄ family (monoclinic P112₁/a) has 36 atoms. RE₂Fe₁₇ hexagonal phases reach 38 atoms. Using the published base model would exclude these families entirely from generation, defeating the purpose of the project.

The solution required two stages of work: (1) retraining the base model from scratch on a much larger and more diverse dataset with an atom ceiling of 51, and (2) fine-tuning the retrained base model on a curated magnetocaloric dataset with property conditioning.

> **THESIS SUGGESTION:** A figure showing the atom-count distribution of the magnetocaloric fine-tuning dataset, with vertical lines at 20 atoms (published model ceiling) and 51 atoms (custom ceiling), would immediately communicate why retraining was necessary. Annotate the bars above 20 atoms with the material families they represent (LaFeSi, Gd₅SiGe, etc.). This is a very strong visual argument.

---

## 2. Magnetocaloric Fine-Tuning Dataset Construction

### 2.1 Source Data Extraction

The source dataset was extracted from Supplemental Table 1 of Franco et al. (2012), "The magnetocaloric effect and magnetic refrigeration near room temperature: materials and models" (Annual Review of Materials Research, 42). This table contains 641 measurements across 11 material family groups (Laves phases, manganites, La(Fe,Si)₁₃ family, Heusler alloys, Gd₅(Si,Ge)₄ family, and others), with properties including peak magnetic entropy change |ΔS_mag|, refrigerant capacity (RCP), adiabatic temperature change ΔT_ad, Curie/Néel temperature, applied field, and transition order.

The PDF table was parsed programmatically using pdfplumber. This was non-trivial because many materials had measurements at multiple applied fields encoded as compact notation (e.g., "a)2T b)5T" with corresponding "a)value b)value" entries), chemical formula subscripts were split across lines by the PDF renderer, and section header rows were interspersed with data rows. The parser expanded multi-field measurements so that each output row represented one material at one field value, yielding 641 measurement rows from 409 unique non-amorphous compositions.

Structure prototype names from the table were mapped to space group symbols using a manually compiled lookup table (e.g., MgCu₂ → Fd-3m, NaZn₁₃ → Fm-3c, CaTiO₃ → Pnma). Amorphous materials were excluded. A parent formula extraction algorithm was applied to identify undoped stoichiometric parents for lightly doped compositions — elements contributing less than 10% of total atoms were classified as dopants and stripped, yielding the parent reduced formula (e.g., Gd(Co₀.₉₇Al₀.₀₃)₂ → GdCo₂).

> **THESIS SUGGESTION:** Include the prototype-to-space-group mapping table in an appendix. It demonstrates domain knowledge and is a genuinely useful reference artefact.

### 2.2 Crystal Structure Acquisition from Materials Project

Structures were obtained from the Materials Project (MP) API using a three-tier matching strategy:

1. **Exact match** — the reduced formula and space group both match an MP entry.
2. **Parent match** — the undoped parent formula and space group match (used for lightly doped materials that do not exist in MP as distinct entries).
3. **Parent-only match** — the parent formula matches but the space group does not (lowest confidence, flagged for review).

All rows sharing the same parent reduced formula and space group reused a single MP structure, avoiding duplicate CIF entries and redundant API calls. This yielded structures for 388 of the 409 target compositions. The remaining 21 required manual intervention (see Section 2.4).

### 2.3 Wyckoff Site Assignment and SQS Ordering

For doped materials, the prototype crystal structure from MP provides only the parent composition. Target elements must be assigned to crystallographic sites (Wyckoff positions) and then ordered into a finite supercell. This was handled by a custom pipeline (`magnetocaloric_wyckoff_pipeline_v14.py`) that:

1. Retrieved the prototype structure from MP.
2. Identified site roles (A-site, B-site, X-site) based on element classification and Wyckoff position multiplicities.
3. Assigned target element occupancies to each site based on stoichiometry.
4. Built ordered supercells, or flagged structures for Special Quasirandom Structure (SQS) treatment when ordered approximations could not achieve acceptable stoichiometry.

For disordered structures (where multiple elements share crystallographic sites at non-integer ratios), SQS generation was performed using `step1_generate_sqs_v2.py`. SQS structures approximate the pair correlation functions of a random alloy in a finite supercell, preserving the statistical character of disorder while providing a fully ordered structure compatible with MatterGen's requirement that every site have exactly 1.0 occupancy.

A critical design rule was enforced throughout: **minority dopants must never be dropped**. Even when a dopant concentration was too low to be represented at the target stoichiometry within a given supercell size, a minimum of 1 atom was forced and the majority species rescaled downward proportionally. The rationale: the model must learn which elements are present (chemical identity) more than exact stoichiometry. Dropping a dopant entirely removes chemically critical information, whereas a small stoichiometry error from forcing 1 atom is absolutely small even if relatively large as a percentage.

However, this rule has a knock-on consequence: when multiple minority dopants compete for a limited number of sites, forcing one minority dopant may require dropping another. For example, Gd₅Si₁.₉₇₅Ge₁.₉₇₅Nb₀.₀₅ becomes Gd₅Si₂Ge₁.₉₅Nb₀.₀₅ in the ordered structure — the Nb is preserved but one of the majority species (Ge) absorbs the rounding error. This is an unavoidable consequence of finite supercell size and is accepted as a minor stoichiometric approximation.

Symmetry consistency was verified through the Wyckoff pipeline and SQS generation steps using automated checking to confirm that space group assignments remained consistent through each transformation stage.

> **THESIS SUGGESTION:** This minority-dopant rule is a design decision worth emphasising. An examiner might ask "why not just drop elements below the representable threshold?" The answer — that element identity carries more information than exact stoichiometry for a generative model — shows good physical intuition applied to an ML problem. Consider a small table showing example materials where this rule was triggered, with the target vs actual stoichiometry and the absolute error.

### 2.4 Handling Structures Not in Materials Project

Twenty-one materials had no MP entry. These fell into three categories:

- **14 × Gd₅(Si,Ge)₄ monoclinic variants** — the giant MCE phase (P112₁/a, SG 14) is not in MP, which only has the orthorhombic Pnma phase. The correct structure was obtained from ICSD entry 84084 (Pecharsky & Gschneidner 1997). This is significant: the giant magnetocaloric effect in this family occurs at the orthorhombic→monoclinic phase transition, so using the wrong phase would undermine the physical relevance of the training data.
- **3 × YFe₂H₄.₂/D₄.₂ hydrides** — obtained from ICSD entry 260534 (Ropka et al. 2009), with deuterium sites relabelled as hydrogen where appropriate.
- **4 × other prototypes** — resolved through manual literature search and alternative MP queries.

The ICSD-derived structures were used as end-member templates, with dopant elements substituted onto the appropriate sites using the same Wyckoff assignment logic as the MP-derived structures.

> **THESIS SUGGESTION:** The Gd₅(Si,Ge)₄ phase issue is a strong example of where domain knowledge prevented a subtle but serious error. The MP structure is crystallographically valid but physically wrong for magnetocaloric applications. This is worth a paragraph in the thesis — it demonstrates that automated pipelines require expert oversight. Additionally, recommend that future researchers working on magnetocaloric materials use ICSD rather than Materials Project as their primary structural source — ICSD has better coverage of experimentally determined phases, particularly polymorphs and metastable phases that are critical for functional materials.

### 2.5 CHGNet Relaxation

All 277 structures passing the filtering criteria (≤50 atoms in primitive cell, no partial occupancy, all target elements present, stoichiometry error <5%) were relaxed using CHGNet, a universal neural network potential for crystal structure relaxation. CHGNet was chosen over full DFT relaxation for practical reasons — relaxing 277 structures with DFT would require significant HPC allocation, whereas CHGNet provides comparable structural relaxation on a laptop CPU in seconds per structure.

Results: 258 structures converged, 19 did not converge but were retained (the unconverged structures had small residual forces and were deemed acceptable for training purposes). Zero structures failed to parse or relax entirely.

### 2.6 Property Value Assignment and Repair

The relaxed structures were exported as MatterGen-compatible CSVs with three conditioning properties: `magnetocaloric_delta_s` (peak |ΔS| in J/kg·K), `magnetocaloric_curie_temp` (ordering temperature in K), and `magnetocaloric_field` (applied field in T).

A systematic cross-check against the source Franco et al. data revealed that approximately 30% of the property values in the initial v14 pipeline output were incorrect. The root cause was brittle string-based matching between sanitised material IDs (which had replaced dots with 'p', parentheses with underscores, and introduced stray spaces) and the source CSV formulas. Fifty-two different perovskite materials had been assigned identical property values (ΔS=10.5, T_C=275K, field=5T) due to fallback-to-default behaviour — a physically impossible situation.

A repair script (`repair_properties.py`) was developed using composition-based matching rather than string matching. Each CIF's chemical formula sum was parsed into a molar-fraction signature and compared against source formulas using tolerance-based L1-distance matching (tolerance=0.05) to handle SQS supercell rounding errors. Material IDs were reverse-parsed as a tie-breaker. Six edge cases required manual overrides (dilute substitutions where composition matching was ambiguous, cementite-type formula parsing, Unicode delta-suffix in source formulas). An additional ~6 rows could not be matched at all because the source spreadsheet contained delta (δ) symbols in chemical formulas that the parser could not handle. This achieved a 100% match rate across all 284 train + 31 val rows after manual intervention for edge cases.

> **THESIS SUGGESTION:** The 30% property error rate in the v14 output is actually a valuable finding for the thesis. It demonstrates that data quality assurance is not optional — a generative model trained on 30% incorrect labels would learn spurious correlations. Present this as a data quality case study: show the distribution of errors before repair, give the perovskite example (52 materials with identical impossible properties), and explain the composition-matching solution. This could be a figure: a scatter plot of assigned vs correct ΔS values, with the 1:1 line, before and after repair.

### 2.7 Final Fine-Tuning Dataset

After repair and filtering (removing 7 rows with NaN ΔS values and 2 duplicate CIF pairs):

- **Training set**: 250 structures (later expanded to 284 after including >50-atom structures when the base model ceiling was raised)
- **Validation set**: 27 structures (later 31)
- Format: MatterGen CSV with columns `material_id`, `cif` (embedded CIF string), `n_atoms_primitive`, `converged`, `space_group_number`, `magnetocaloric_delta_s`, `magnetocaloric_field`, `magnetocaloric_curie_temp`

---

## 3. Base Model Retraining Dataset Construction

### 3.1 Rationale and Scale Target

The published MatterGen base model was trained on ~608,000 structures. The custom base model needed a dataset of comparable scale but with a higher atom-count ceiling (51 atoms instead of 20) to support the larger magnetocaloric structures. An initial Materials Project-only scrape (experimental structures, ≤64 atoms) yielded only ~39,000 structures — roughly 6% of the target, and clearly insufficient.

### 3.2 Data Sources

Five sources were combined, in order of priority for deduplication:

1. **Materials Project (MP)** — ~154,000 structures (both experimental and DFT-computed). Highest priority because MP structures are widely used, well-validated, and include provenance metadata. Downloaded as 40 JSON chunks via the MP API.

2. **Crystallography Open Database (COD)** — 533,464 experimentally determined CIF files, bulk-downloaded via rsync (117 GB uncompressed). Open access under CC0 license.

3. **Alexandria Materials Database** — convex hull subset containing ~175,000 thermodynamically stable DFT-computed structures. Open access under CC-BY 4.0. This follows the same approach used by MatterGen's authors for their `alex_mp_20` base model.

4. **Open Quantum Materials Database (OQMD)** — ~1.4 million DFT-calculated structures, filtered to those within 0.1 eV/atom of the convex hull. Accessed via REST API with pagination and checkpointing.

5. **NEMAD (National Energy Materials Advancement Database)** — supplementary CSVs containing property annotations for magnetic, thermoelectric, and superconductor materials. Used as a whitelist/union source (materials appearing in NEMAD were considered experimentally verified by definition), but NEMAD itself does not contain atomic positions or crystal structures, so structures had to be matched from other sources (MP, COD). In hindsight, cross-referencing NEMAD with structural databases from the outset would have been a more efficient approach to building a labelled dataset, as NEMAD provides a large quantity of experimentally validated property labels.

The decision to include DFT-computed structures (Alexandria, OQMD, theoretical MP entries) was a pragmatic compromise. Initially, only experimentally verified structures were considered. However, the combined experimental-only yield from MP + COD was only ~53,000 structures after deduplication — 9% of the MatterGen training set size. Including thermodynamically stable DFT structures follows the precedent set by MatterGen's own authors and is justified by the observation that convex-hull structures have, by definition, been computationally validated as energetically favourable.

> **THESIS SUGGESTION:** The experimental-vs-DFT decision is worth a paragraph of critical discussion. Acknowledge the trade-off: DFT structures may contain systematic errors (GGA underbinding, incorrect magnetic ground states for some transition metal compounds), but excluding them would produce a base model trained on too few examples to learn the structural grammar of inorganic crystals. Reference the MatterGen paper's own use of Alexandria data to justify the approach.

### 3.3 Filtering Criteria

Every structure from every source was subjected to the same five filters:

1. **Atom count**: ≤51 atoms in the primitive unit cell (strictly <52). The primitive cell was computed using pymatgen's SpacegroupAnalyzer. The ceiling of 51 was chosen to accommodate the largest magnetocaloric fine-tuning structures (50 atoms) with one atom of headroom.
2. **Occupancy**: Exactly 1.0 on every crystallographic site. MatterGen cannot represent partial or mixed occupancy — it generates fully ordered structures.
3. **Composition**: Inorganic only. Structures containing both carbon and hydrogen were rejected as likely organic/organometallic.
4. **Uniqueness**: Deduplicated using pymatgen's StructureMatcher with source priority MP > COD > Alexandria > OQMD > NEMAD. When duplicates were found across sources, the highest-priority source's version was retained.
5. **Thermodynamic stability**: For DFT sources (Alexandria, OQMD), only structures on or near the convex hull were accepted (OQMD: stability < 0.1 eV/atom; Alexandria: convex hull membership). An additional outlier filter rejected structures with formation energy > 10 eV/atom (2,646 structures removed).

### 3.4 Pipeline Architecture

The filtering pipeline was implemented as five sequential stages, each with checkpoint saves and completion flags:

- **Stage 1 (COD filter)**: Parallel processing of 533,464 CIF files using Python's multiprocessing with 7 workers. Each CIF was parsed, checked for partial occupancy, reduced to its primitive cell, and filtered by atom count and composition. A 60-second per-file timeout was implemented to handle pathological CIFs that caused pymatgen's SpacegroupAnalyzer to hang indefinitely. Checkpoints were saved every 5,000 files. Result: 33,791 structures passed.
- **Stage 2 (MP filter)**: Sequential processing of 40 JSON chunks. Result: 36,827 structures passed.
- **Stage 3 (Alexandria + OQMD filter)**: Alexandria JSON parsed and filtered; OQMD accessed via paginated REST API with checkpointing.
- **Stage 4 (Merge and deduplication)**: All passing structures merged and deduplicated using StructureMatcher with the priority ordering described above.
- **Stage 5 (Export)**: Exported as MatterGen-format train/val/test CSVs (80/10/10 split) with embedded CIF strings, plus .extxyz format for compatibility with other tools.

### 3.5 Final Base Dataset

- **274,602 unique structures** passed all filters and validation
- **Train**: 219,679 structures (331 MB CSV)
- **Validation**: 27,457 structures (41 MB CSV)
- **Test**: 27,466 structures (42 MB CSV)
- Atom count distribution: 1–10: ~33%, 11–20: ~29%, 21–30: ~18%, 31–40: ~13%, 41–51: ~8%
- Source breakdown (after deduplication): MP ~37%, COD ~13%, Alexandria + OQMD ~50%

> **THESIS SUGGESTION:** Two key figures here. (1) A stacked histogram of atom counts coloured by source (MP, COD, Alexandria, OQMD), showing how the dataset covers the full 1–51 range and which sources contribute most at each size. (2) A Venn-style or Sankey diagram showing how many structures each source contributed before and after deduplication — this visualises why multiple sources were needed and how much overlap existed. Also include a table comparing this dataset to the original MatterGen training set (mp_20: 608k structures, ≤20 atoms, MP-only; this work: 274k structures, ≤51 atoms, 4 sources).

---

## 4. HPC Deployment and Model Training

### 4.1 Environment and Infrastructure

All training was performed on the University of Sheffield Stanage HPC cluster using NVIDIA A100-SXM4-80GB GPUs. The software environment required careful version pinning:

- Python 3.10, PyTorch 2.2.1 compiled against CUDA 11.8
- PyTorch Geometric extensions (torch-sparse, torch-scatter, torch-cluster) compiled from source against CUDA 11.8 and GCC 11.3
- MatterGen installed in editable mode from a local fork with modifications for custom property conditioning

The environment manager was uv (MatterGen's recommended tool), not conda. All caches (uv, pip) and training outputs were directed to scratch storage (`/mnt/parscratch/`) to avoid home directory quota issues.

### 4.2 MatterGen Code Modifications

Three categories of modification were made to the MatterGen codebase:

1. **Property registration**: `magnetocaloric_delta_s`, `magnetocaloric_field`, and `magnetocaloric_curie_temp` were added to `PROPERTY_SOURCE_IDS` in `globals.py`. Without this registration, `csv-to-dataset` silently discards property columns.
2. **Property embedding configs**: Three YAML files were created (one per property), following the template of the existing `dft_mag_density.yaml` config.
3. **GemNet scale file path**: The hardcoded Microsoft-internal Azure path for GemNet scale factors was replaced with a relative path within the repository (`scales/gemnet_scales.json`). The existing OC20-derived scale factors were retained — they produced correct training loss in smoke testing and recomputing them would have required a standalone scale-computation pipeline that does not exist in the MatterGen codebase.

### 4.3 Data Preprocessing

MatterGen's `csv-to-dataset` command preprocesses CSVs into NumPy arrays (`.npy` files for atomic numbers, cell parameters, positions, atom counts, and structure IDs; `.json` files for scalar properties). This preprocessing also automatically applies Niggli reduction to standardise cell representations. The preprocessed cache was created on the HPC in an interactive GPU session.

Key finding: `csv-to-dataset` applies `primitive=True` (hardcoded in `dataset.py`), so all structures are automatically reduced to their primitive cells during preprocessing. The Niggli reduction applied in the export pipeline was therefore redundant but harmless.

### 4.4 Base Model Pre-Training

**Configuration:**
- 4 × A100 GPUs with DDP (Distributed Data Parallel) strategy
- Batch size: 32 per GPU, gradient accumulation over 4 steps → effective batch size 512 (matching Microsoft's published setup)
- Optimiser: Adam, learning rate 1×10⁻⁴
- LR scheduler: ReduceLROnPlateau (factor 0.6, patience 100 epochs, minimum LR 1×10⁻⁶)
- Architecture: GemNetT denoiser, 44.6M parameters, hidden_dim 512, 4 interaction blocks
- Diffusion: 1000 timesteps, variance-adjusted noise schedule scaled by structure size (`NumAtomsVarianceAdjustedWrappedVESDE`), mask diffusion for atomic number denoising
- Training budget: 1000 epochs, wall time ~50–60 hours on 4 GPUs

**Results:**
- Total loss decreased from 2.46 (epoch 1) to 0.52 (epoch 1000)
- Cell loss converged near zero (~0.022)
- Position and atomic number losses showed diminishing returns in the final ~200 epochs
- The ReduceLROnPlateau scheduler was confirmed active from the config, providing implicit convergence detection

> **THESIS SUGGESTION:** Plot the per-epoch training loss curves (total, cell, position, atomic number components) on separate axes or as a multi-panel figure. The cell loss converging near zero while position/species losses plateau tells an interesting story about what the model finds easy vs hard to learn. Also note that MatterGen has no explicit convergence criterion or early stopping — it uses a fixed epoch budget with the LR scheduler as an implicit mechanism. This is worth a sentence in the methods.

### 4.5 Magnetocaloric Fine-Tuning

Fine-tuning added a property embedding adapter on top of the pretrained base model, increasing parameters from 44.6M to 48.8M. The adapter learns to condition the denoising process on the target property value.

**Configuration:**
- Checkpoint: custom pretrained base model (not Microsoft's published checkpoint)
- Property: `magnetocaloric_delta_s` as primary conditioning property
- Dataset: 284 train / 31 val structures
- Training: 1000 epochs (after an initial 200-epoch exploratory run)
- ModelCheckpoint with `save_top_k=1` preserves the best validation checkpoint regardless of overfitting, making extended training safe

**Results:**
- Loss decreased from 1.63 to 1.14 over 200 epochs
- Mean loss across 50-epoch blocks showed continued improvement: 1.376 → 1.264 → 1.225 → 1.185
- High per-epoch variance due to the small dataset (37 batches per epoch)
- The 1000-epoch run restarted from the base pretrain checkpoint rather than continuing from epoch 200, because `mattergen-finetune` has `ckpt_path=None` hardcoded — it cannot resume from a fine-tuning checkpoint

**Note on resume behaviour:** The `finetune.py` entry point does not support checkpoint resumption. Each fine-tuning run starts fresh from the base pretrained weights. This means the 1000-epoch run repeated the first 200 epochs of learning. Given the short per-epoch training time, this was accepted rather than modifying the codebase.

> **THESIS SUGGESTION:** The inability to resume fine-tuning is a genuine limitation of the MatterGen codebase worth noting. It means that if a fine-tuning run is interrupted, all progress is lost. For a 1000-epoch run on a small dataset this was not practically problematic, but it would be for larger fine-tuning datasets or longer training budgets.

### 4.6 Superconductor Fine-Tuning (Secondary Application)

A second fine-tuning dataset was prepared to demonstrate the generality of the approach. The NEMAD superconductor database was cross-referenced with the Materials Project, yielding 1,088 structures with superconducting transition temperature (T_c) labels after regex parsing of free-text temperature strings (the initial extraction yielded only 58 rows because non-numeric values like "90 K" were silently dropped). This was split 1000/88 (train/val) and configured for fine-tuning with a `sc_transition_temp` property adapter.

> **THESIS SUGGESTION:** The superconductor dataset shows the pipeline is not specific to magnetocaloric materials — it generalises to any property where labelled structures exist. Even if the superconductor results are preliminary, mentioning the setup demonstrates methodological breadth. An examiner will notice this favourably.

---

## 5. Key Design Decisions and Their Justifications

| Decision | Justification | Potential Limitation |
|----------|--------------|---------------------|
| Atom ceiling of 51 (not 20, not 64) | Accommodates all magnetocaloric families while keeping base dataset tractable. Largest fine-tune structure is 50 atoms. | Structures with 52–64 atoms are excluded from generation. Some complex phases (e.g., certain σ-phases) cannot be generated. |
| SQS for disordered structures | Provides fully ordered structures compatible with MatterGen while preserving statistical disorder character. Standard approach in computational materials science. | SQS is an approximation. Finite supercell size limits the accuracy of pair correlation matching. Some short-range order effects are lost. |
| CHGNet relaxation instead of DFT | Enables rapid relaxation of 277 structures on a laptop. CHGNet is a universal potential trained on MP data. | CHGNet may not accurately relax structures far from its training distribution. Magnetic ground states may be incorrect for some rare-earth compounds. |
| Including DFT-computed structures in base dataset | Necessary to reach adequate dataset size (~274k vs ~53k experimental-only). Follows MatterGen authors' own precedent. | DFT structures may contain systematic errors. The model may learn GGA-specific artefacts. Base dataset is ~50% non-experimental. |
| Composition-based property matching (not string-based) | Robust to formatting inconsistencies between pipeline-generated material IDs and source CSV formulas. Achieved 100% match rate. | Relies on stoichiometry tolerance (0.05). Very similar compositions could theoretically be mismatched. Six edge cases required manual overrides. |
| ΔS as primary conditioning property (not RCP or T_C) | ΔS is the most direct measure of magnetocaloric performance and has the most complete coverage in the Franco et al. dataset. | Single-property conditioning cannot capture the full performance landscape. Multi-property conditioning would be more powerful but requires all properties present in every training row. |

---

# PART B — WHAT ACTUALLY HAPPENED

This section documents the real chronology of the project, including mistakes, debugging sessions, and iterative problem-solving. It is organised chronologically rather than thematically.

---

## 6. Phase 0: Learning Curve (~Late 2025 — Early 2026)

The project began with no prior experience in Linux command-line tools, Python environment management, HPC job submission, or generative AI model training. The first working session covered WSL/Ubuntu setup, Git and Git LFS installation, conda environment creation, and a first successful MatterGen generation test on CPU. Key conceptual foundations were established: the difference between system packages (apt), environment managers (conda), and Python package installers (pip/uv); how diffusion models denoise from random noise to generate crystal structures; and the distinction between inference (generation) and training (weight optimisation).

An early test on Sheffield Stanage HPC established the SLURM workflow: module loading, conda activation, sbatch submission, and squeue monitoring. Git LFS was not available as a module and had to be installed manually to `~/bin/`.

> **THESIS SUGGESTION:** The learning curve itself is not thesis material, but the HPC setup procedure (module loading sequence, Git LFS workaround, SLURM script structure) belongs in an appendix or supplementary materials for reproducibility.

---

## 7. Phase 1: Source Data and the Wyckoff Pipeline (~Nov 2024 — Apr 2026)

### 7.1 Franco et al. PDF Parsing

The Franco et al. (2012) supplemental table was parsed programmatically using pdfplumber across three script versions (v1, v2, v3). The main challenges were multi-field measurement encoding, split subscript lines, and interspersed section headers. v2 added the prototype-to-space-group mapping and parent formula extraction. v3 preserved the RCP column for potential third-adapter training.

### 7.2 The Wyckoff Pipeline: v9 → v14

The Wyckoff site assignment pipeline went through at least six major versions (v9–v14), each addressing bugs discovered through diagnostic scripts and test cases. This was the most iteratively debugged component of the entire project.

**v9–v11** (prior to the documented sessions): Fixed missing Wyckoff sites for several prototypes (Mn₅Si₃, CeCu₂, MgZn₂, TbCo₂), fixed the majority-wins bug in `_build_ordered` that dropped minority dopants, established the 5% stoichiometry tolerance, and added prototype overrides (e.g., CoMnSb: CsCl → MgAgAs).

**v12**: Discovered that the diagnostic script had been importing v8 instead of v11 — meaning none of the v9–v11 fixes had actually been tested. This explained why "fixed" issues kept reappearing. v12 added `stoich_error()` using per-atom-fraction comparison, the `clean_cifs/` output folder, and `needs_manual_search.csv` for failed materials.

**v13**: The critical `_build_ordered` rewrite. Two distinct bugs were identified:
- **Bug A (scale=1)**: Python's banker's rounding (`round(0.5) = 0`) caused minority elements to disappear when processing atoms individually. For {La:0.5, Pr:0.5} on a 2-atom site, each atom independently ran `max()` and both picked the same winner.
- **Bug B (scale>1)**: pymatgen's `make_supercell` reorders atoms by electronegativity, not by primitive-site-major order. The old code assumed the original ordering, causing species-coordinate mismatches in supercells.

The fix introduced `_distribute_group()` — a function that groups all supercell sites by their fractional-occupancy composition key (ordering-agnostic) and distributes minority elements across the entire group at once, guaranteeing every element with occupancy >0 gets ≥1 atom.

**v14**: Run by the user on 409 materials. Results: 120 stoichiometry errors >5% and 21 complete failures. These were classified into 12 root-cause groups, and 9 specific fixes were designed for a planned v15 (force-disordered treatment for several families, A-role classification expansion for Pb/Cd/Ag, mixed-mode switching for unequal site multiplicities, antiperovskite prototype correction, and direct ICSD CIF use for the monoclinic Gd₅(Si,Ge)₄ family).

> **THESIS SUGGESTION:** The Wyckoff pipeline debugging is a double-edged sword for the thesis. On one hand, the sheer number of bugs suggests the approach was fragile. On the other hand, the systematic diagnostic methodology (41 test cases covering all root causes, per-element error breakdowns, recommended fixes) demonstrates rigorous engineering. Frame it as: "Assigning arbitrary element combinations to prototype Wyckoff sites is an inherently complex combinatorial problem. Systematic testing revealed 12 distinct failure modes, each requiring a targeted fix." The diagnostic approach is the strength, not the bug count.

### 7.3 The SQS Generation Step

SQS structures were generated for disordered materials using `step1_generate_sqs_v2.py`. Of 221 materials processed, 11 passed without warnings, 210 generated S6 symmetry warnings, and 0 failed entirely. The 210 warnings were investigated and found to be a consequence of the SQS method itself — breaking the parent structure's symmetry to represent disorder necessarily lowers the space group. Since MatterGen learns symmetry from training data rather than having it imposed as a constraint, these lower-symmetry SQS structures were accepted.

### 7.4 The Identical CIF Problem and Deduplication

A potentially serious issue was identified: due to the finite resolution achievable within the ≤50 atom constraint, many nominally different compositions produced identical CIF files. For example, Gd₅Si₁.₉₅Ge₁.₉₅Nb₀.₁, Gd₅Si₁.₉₇₅Ge₁.₉₇₅Nb₀.₀₅, and Gd₅Si₁.₉Ge₁.₉Nb₀.₂ all rounded to the same ordered cell: Gd₁₉Ge₈NbSi₈. This is a fundamental limitation of representing continuously-variable compositions in discrete atomic structures at small supercell sizes.

The solution adopted was to eliminate identical CIFs and assign training property values from the target composition that most closely matched the actual atomic makeup of the retained CIF file. This ensured that the model was not trained on duplicate structures with conflicting property labels, which would introduce noise into the property conditioning.

A deduplication step identified 74 duplicate groups (121 proposed drops) among the generated CIFs. After deduplication, 374 unique CIFs were transferred to `/home/daniel/Clean_Preprocessing/mattergen_cifs/`. Of these, 199 had ≤20 atoms (compatible with the published MatterGen base model), 95 had 21–30 atoms, and 80 had >30 atoms.

Two versions of the training CSV were initially created: one with applied field as a conditioning property (for potential field-conditioned generation) and one without (collapsed, so that materials measured at different fields but otherwise identical were merged). Ultimately, applied field was retained as a column but its use as a conditioning property was reconsidered (see Section 13, Outstanding Issues).

The atom count breakdown was the trigger for the base model retraining decision: the most scientifically valuable material families (LaFe₁₁Si, Gd₅(Si,Ge)₄) all fell above the 20-atom ceiling. This realisation, described as arriving "after lots of stress," led to the decision to retrain the base model with a 51-atom ceiling rather than attempting to work within the 20-atom constraint.

> **THESIS SUGGESTION:** The identical-CIF problem is worth a paragraph in the thesis. It illustrates a fundamental tension in representing continuously-variable alloy compositions as discrete crystal structures. A small table showing 3–4 nominal compositions that map to the same ordered cell would communicate this clearly. Frame it as a limitation of the SQS approach at small supercell sizes, not as a pipeline bug.

---

## 8. Phase 2: Fine-Tuning Dataset Assembly (~13 Apr — 22 Apr)

### 8.1 CHGNet Relaxation

Relaxation was performed in `/home/daniel/Clean_Preprocessing/` using a Claude Code agent. 277 CIFs passed filtering (≤50 atoms, no partial occupancy, all elements present, stoichiometry error <5%). All 277 were successfully relaxed: 258 converged, 19 did not converge but were retained.

Structures were relaxed in ascending order of atom count (smallest first, largest last). This was a deliberate strategy to avoid crashing problems — if CHGNet encountered memory or convergence issues, the smaller (and more numerous) structures would already be complete.

The atom ceiling was initially 50 but was later raised. On 20 April, the question was asked: "if we raise the ceiling to 64 how many additional CIF files pass?" The excluded CIFs (58 structures with >50 atoms) were relaxed and added to the training data, expanding the dataset. This decision was made after confirming that the base model retraining would use a 51-atom ceiling anyway.

Note: the working CIF directory at one point contained 724 CIF files — this included both pre-SQS and post-SQS versions of the same materials. The prepare_dataset.py script handled the selection of the correct version (SQS where available, ordered otherwise) and excluded the superseded copies.

The train/val split was adjusted during this phase: 15 materials were moved from training to validation to achieve a rounder training set size of ~300 rows. This was acknowledged as a small dataset for fine-tuning purposes.

### 8.2 The v14 Property Mismapping Bug

The most consequential data quality issue was discovered during the HPC deployment phase (see Section 10.2), but it originated here. The v14 pipeline's property-lookup logic used string matching between sanitised material IDs and source CSV formulas. The sanitisation process (replacing `.` with `p`, `(` with `_`, introducing stray spaces) created material IDs that could not be reliably matched back to source formulas. When matching failed, the code fell back to a default row — resulting in 91 incorrect ΔS values, 88 incorrect Curie temperatures, and 76 fabricated field values across the 315-row dataset.

This bug was not caught until a systematic cross-check was performed as part of the HPC deployment preparation, roughly two weeks after the CSVs were first generated. The repair (composition-based matching) was developed and validated before any training was performed on the fine-tuning data.

> **THESIS SUGGESTION:** The time gap between generating the data and catching the bug is worth reflecting on. It argues for building validation checks into the pipeline itself rather than relying on post-hoc audits. A simple sanity check — "are there any property values that appear more than N times?" — would have immediately flagged the 52 identical perovskite entries. This is a genuine lesson learned.

---

## 9. Phase 3: Base Model Dataset Construction (~21 Apr — 27 Apr)

### 9.1 The Scale Problem

The initial plan was to use only experimentally verified structures from MP and COD. When this yielded only ~53,000 structures after deduplication (9% of MatterGen's training set), the constraint was relaxed to include DFT-computed structures from Alexandria and OQMD.

ICSD was investigated as an alternative experimental source. Sheffield has institutional access via PSDS, but the web interface lacks a direct atom-count filter, the annual export limit is 20,000 CIFs (which would not only be insufficient but would consume the university's entire annual allowance for a single project), and the API requires project-level purchasing approval. ICSD was abandoned as too cumbersome for the available timeline.

### 9.2 COD Filtering: Timeouts and Crashes

The COD filtering stage (533,464 CIF files) was the most operationally challenging component. Key problems encountered:

- **SpacegroupAnalyzer hangs**: Some pathological CIF files caused pymatgen's SpacegroupAnalyzer to hang indefinitely. The initial pipeline had no timeout mechanism. After the first run got stuck on batch 26 (~50,000 files), a 60-second per-file timeout was added using Python's `concurrent.futures` with `future.result(timeout=60)`. An optimisation was also applied: skip SpacegroupAnalyzer entirely for structures >300 atoms (fast reject by atom count alone), and call SpacegroupAnalyzer only once for structures in the 52–300 atom range, reusing the result for both primitive cell computation and space group determination (the original code called it twice).

- **Checkpoint corruption**: The initial checkpoint interval was set to every 50,000 files. A WSL crash during checkpointing produced a corrupt pickle file (2.2 MB, truncated by EOFError). The entire COD filtering had to be restarted from scratch because the checkpoint was unrecoverable. The interval was subsequently reduced to every 5,000 files, and atomic (write-to-temp-then-rename) saves were implemented.

- **WSL crashes**: WSL2 crashed repeatedly during long-running pipeline jobs. The root cause was traced via Windows Event Viewer to the laptop going to sleep/hibernate while the pipeline was running, corrupting WSL2's virtual machine state. The fix was to disable all sleep and hibernate timeouts via PowerShell (`powercfg /change standby-timeout-ac 0` etc.).

Final COD result: 33,791 structures passed from 533,464 inputs. Rejections: 166,447 (partial occupancy), 317,673 (>51 atoms), 8,318 (organic), 165 (theoretical), 5,435 (COD internal duplicates), 1,635 (parse errors).

> **THESIS SUGGESTION:** The COD filtering statistics make a compelling table or figure. The dominance of partial-occupancy rejections (31% of all CIFs) and >51-atom rejections (60%) tells a story about the character of the COD database that contextualises the filtering decisions. A waterfall/funnel chart showing 533k → 33k through each filter stage would be visually effective.

### 9.3 The Alexandria Race Condition

The most insidious pipeline bug was a race condition in the shell orchestration script (`run_stages_3_to_5.sh`). The script polled for the existence of `alex_structures.pkl` to trigger the merge-and-dedup stage (Stage 4). However, the Alexandria filter saved checkpoint pickles after every 10,000 entries, creating the file long before filtering was complete. Stage 4 fired after the first checkpoint (9,325 structures) while filtering was still running — and the filter itself crashed around entry 100,000, leaving 54,651 entries unprocessed.

The result: the initial "complete" dataset of 53,291 structures contained only 9,325 Alexandria structures instead of the expected ~83,540+. This was not immediately obvious because the dataset appeared reasonable in size (it was dominated by MP and COD entries, which had completed correctly).

The fix was straightforward: use completion flag files (empty sentinel files written only after a stage finishes successfully) instead of polling for output file existence. This is a standard pattern in pipeline orchestration but was missed in the initial implementation.

> **THESIS SUGGESTION:** The race condition is an excellent example of a subtle bug that produces plausible-looking but incorrect results. It is worth describing briefly in the thesis (perhaps in a "challenges encountered" subsection) because it illustrates a general principle: polling for file existence is not equivalent to polling for process completion. The fix (completion flags) is simple but the diagnosis was not.

### 9.4 Dataset Expansion

After identifying the Alexandria race condition and the insufficient dataset size, the pipeline was expanded:

- Alexandria filtering was fixed and rerun, recovering ~83,540 structures.
- OQMD was added as a new source via its REST API, with pagination (200 entries per page), checkpointing, and a stability filter (< 0.1 eV/atom from convex hull).
- The full Materials Project (including theoretical/DFT-computed structures) was processed, roughly tripling the MP contribution.

The expansion pipeline was designed for parallel execution (Alexandria, OQMD, and full MP filtering could run concurrently) with Stage D (merge/dedup) gated on all three completion flags.

Additional WSL stability problems required further intervention: increasing the WSL memory allocation to 10–12 GB (via `.wslconfig`), and implementing crash-resilient checkpointing so that a WSL crash would not lose more than the most recent batch of work.

Final result: 274,602 structures passed all filters and validation, with 2,646 rejected as formation energy outliers (>10 eV/atom).

---

## 10. Phase 4: HPC Deployment (~26 Apr — 6 May)

### 10.1 Local Smoke Testing

Before transferring to the HPC, a comprehensive smoke test was run on the local laptop (Intel i7-1195G7, 16 GB RAM, no GPU). Eight structured tests verified: environment integrity, codebase reconnaissance (confirming `primitive=True` in `csv-to-dataset`), property registration in `globals.py`, preprocessing, dataloader shapes, and a single training step (loss=2.140, finite — confirming the pipeline works end-to-end). Peak RAM usage reached 3,040 MB during the training step. Two tests were skipped due to RAM constraints.

### 10.2 The Property Mismapping Discovery

During smoke test preparation, a systematic cross-check of the fine-tuning CSVs against the source Franco et al. data revealed the 30% property error rate described in Section 2.6. The timing was fortunate — this was caught before any training was performed on the incorrect data. The repair script was developed and validated, producing `train_repaired.csv` and `val_repaired.csv`.

### 10.3 Transfer and Environment Nightmares

The project bundle was transferred to Stanage via two-stage rsync (code/configs first, then the 414 MB base dataset with `--partial --append-verify` for resumability). Then began the most time-consuming phase of the entire project: getting the Python environment to work on the HPC.

**Problem 1 — CPU architecture mismatch**: The venv was initially built on a CPU worker node (Intel). GPU nodes use AMD EPYC 7413 (Zen 3) which lacks AVX-512 support. This caused "Illegal instruction" crashes for numpy and torch. Resolution: delete and rebuild the entire venv on a GPU node.

**Problem 2 — GCC version conflict**: CUDA 11.8's nvcc rejected GCC 12.2. Resolution: `module swap GCCcore/12.2.0 GCCcore/11.3.0`.

**Problem 3 — PyG glibc incompatibility**: Pre-built wheels for torch-sparse, torch-scatter, and torch-cluster required glibc 2.27+, but Stanage runs CentOS 7 with glibc 2.17. Resolution: compile all three from source with `--no-build-isolation` on the GPU node. The optional `pyg-lib` acceleration library was removed entirely (not required by MatterGen).

**Problem 4 — numpy version conflict**: The from-source PyG build pulled in numpy 2.x, incompatible with torch 2.2.1. Resolution: pin `numpy<2`.

**Problem 5 — wandb build failure**: wandb 0.26.0 required the Go compiler (not available on Stanage). Resolution: downgrade to wandb 0.18.7 (ships a pre-built wheel), which required removing the `protobuf==7.34.1` pin.

**Problem 6 — Home directory quota**: The uv cache filled the home directory quota (338k files vs 300k soft limit). Resolution: redirect `UV_CACHE_DIR` and `PIP_CACHE_DIR` to scratch storage, then run `conda clean --all -y` to recover space.

> **THESIS SUGGESTION:** The environment setup section of the thesis should be concise but honest. Do not enumerate every problem — instead, summarise: "The HPC environment required compilation of PyTorch Geometric extensions from source due to glibc version constraints, GCC version pinning for CUDA compatibility, and careful cache management to stay within filesystem quotas. A detailed reproducibility guide is provided in Appendix X." Then put the full details in an appendix. This shows you understand the problems without boring the examiner.

### 10.4 Training Failures and Debugging

**Failure 1 — Wrong dataset**: The first pre-training job completed in ~70 minutes, which was suspiciously fast for 1000 epochs on 219k structures. Investigation revealed that `base_pretrain.yaml` had `root_dir` pointing to the fine-tune data cache (299 structures, 10 batches/epoch) instead of the base dataset cache (219k structures, ~6,865 batches/epoch). The model trained on the wrong dataset for the full 1000 epochs, wasting the GPU allocation.

**Failure 2 — Stale checkpoint**: A test job set `max_epochs=100`, but `auto_resume: true` found a stale checkpoint at `epoch=999-step=10000.ckpt` from the wrong-dataset run. The trainer refused to resume because `current_epoch=999 > max_epochs=100`. A second attempt with `max_epochs=1000` resumed from epoch 999 and immediately stopped after one epoch.

Resolution: all checkpoints and lightning_logs were cleared. The `max_epochs` was set to 5000 to provide headroom for future resumes. The `root_dir` was corrected.

**Failure 3 — SLURM ntasks**: A job failed within 30 seconds because PyTorch Lightning rejected `--ntasks=4`. Lightning manages multi-GPU parallelism internally via DDP; SLURM should launch 1 task per node (`--ntasks-per-node=1`) and let Lightning's `trainer.devices=4` handle GPU worker spawning.

> **THESIS SUGGESTION:** The wrong-dataset failure is a cautionary tale worth a brief mention. A single incorrect path in a YAML config caused a full training run to complete silently on the wrong data, producing plausible-looking loss curves. The diagnostic was the batch count: 10 batches/epoch (wrong) vs ~6,865 batches/epoch (correct). A one-sentence recommendation: "Always verify the first epoch's batch count matches expectations before committing to long training runs."

### 10.5 Successful Training and Fine-Tuning

With all issues resolved, the base pre-training completed 1000 epochs on 4 × A100 GPUs in approximately 50–60 hours. Total loss dropped from 2.46 to 0.52. The checkpoint was saved as `epoch=999-step=430000.ckpt`, with a symlink created to `last.ckpt` (expected by the fine-tuning entry point).

Fine-tuning was performed in two runs: an exploratory 200-epoch run (loss: 1.63 → 1.14) followed by a 1000-epoch run. Because `mattergen-finetune` cannot resume from fine-tuning checkpoints (`ckpt_path=None` hardcoded), the 1000-epoch run restarted from the base pretrain checkpoint. ModelCheckpoint with `save_top_k=1` ensured the best validation model was always preserved.

### 10.6 Hydra Override Syntax

The correct override syntax for MatterGen's Hydra-based configuration was established through trial and error:

- Existing keys: `++key=value` (override, may or may not exist)
- New keys: `+key=value` (add)
- Property embedding path: `+lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY`

This is fragile and poorly documented in the MatterGen repository. Incorrect syntax produces cryptic Hydra errors.

---

## 11. Phase 5: Extension to Superconductors (6 May)

A superconductor fine-tuning dataset was prepared from the NEMAD superconductor database. The initial CIF-labelled extraction yielded only 58 rows because non-numeric T_c values (e.g., "90 K", "< 4.2 K") were silently dropped. A second pass using regex parsing to extract numeric temperatures from free-text strings recovered 1,088 usable rows. The dataset was split 1000/88 (train/val), a `sc_transition_temp` property was registered in `globals.py`, and configuration files were created. The SLURM batch script (`finetune_sc.sbatch`) was prepared for single-GPU training.

---

## 12. Chronological Summary of Agent Usage

| Period | Agent/Tool | Working Directory | What Was Done |
|--------|-----------|-------------------|---------------|
| Late 2025 | Claude.ai chat | — | MatterGen installation, basic concepts, first HPC test |
| ~Nov 2024 → early 2026 | Claude.ai chat | — | Franco PDF parsing, MP cross-referencing, property conditioning design |
| ~Mar 2026 | Claude.ai chat | — | Wyckoff pipeline v12–v13, diagnostic scripts, ICSD CIF acquisition |
| 24 Mar → 10 Apr | Claude Code | `/home/daniel/Pre-mattergen_dataprocessing` | v14 pipeline execution, SQS generation, deduplication → 374 CIFs |
| 13 Apr → 22 Apr | Claude Code | `/home/daniel/Clean_Preprocessing` | CHGNet relaxation, MatterGen CSV export, atom ceiling expansion |
| 21 Apr → 27 Apr | Claude Code | `/home/daniel/base_model_dataset` | COD download/filter, MP filter, Alexandria, OQMD, WSL debugging, 274k dataset |
| 26 Apr → 27 Apr | Claude Code | `/home/daniel/mattergen_project` | Local smoke testing, GemNet path patch, fine-tune data staging |
| ~27 Apr → 30 Apr | Claude.ai chat | — | HPC deployment, environment setup, pretrain debugging, batch submission |
| 30 Apr → 6 May | Claude.ai chat | — | Pretrain verification, fine-tune runs, superconductor dataset, quota management |
| 6 May | Claude Code | `/home/daniel/.../nemad_labelled` | NEMAD CIF enrichment |

---

## 13. Outstanding Issues and Future Work

1. **Multi-property conditioning**: Only single-property (ΔS) conditioning was implemented for magnetocaloric fine-tuning. The original plan included T_C and RCP as additional adapters, with composition at inference time. The MatterGen codebase supports this but it was not executed within the project timeline.

2. **H (applied field) as a conditioning property**: H was initially included as a conditioning property but should be reconsidered. Applied magnetic field is an experimental condition, not an intrinsic material property. Conditioning a generative model on H is not physically meaningful — the model cannot generate a structure that "has" a particular applied field. ΔS values should ideally be normalised to a single reference field (e.g., 5T) before training.

3. **v15 pipeline fixes**: Nine specific fixes were designed for a v15 of the Wyckoff pipeline but were not implemented. The v14 output was used for training after property repair. A v15 run would likely recover additional materials (particularly the force-disordered treatment for Fe₂P, BiF₃, and R₂Fe₁₇ families) and could expand the fine-tuning dataset.

4. **GemNet scale factors**: The existing OC20-derived scale factors were used without recomputation. MatterGen's codebase does not include a standalone scale-computation entry point. The smoke test confirmed these produce finite training loss, but they were originally fitted on OC20 data (molecular adsorbates on surfaces) rather than bulk crystals. The impact on model quality is unknown.

5. **Evaluation**: No evaluation of generated structures (stability, novelty, validity, property accuracy) was documented in the available session records. This is the natural next step after training.

> **THESIS SUGGESTION:** The evaluation gap is the biggest risk for a strong thesis mark. Examiners will want to see: (a) what structures the model generates, (b) whether they are chemically valid, (c) whether they are novel (not in training data), (d) whether DFT validation confirms the predicted properties. Even preliminary results — "50 structures were generated and 38 passed pymatgen validity checks" — would strengthen the thesis significantly. If time permits, DFT relaxation of a handful of generated structures would be the gold standard. If DFT is not feasible, CHGNet relaxation with energy-above-hull estimation is a reasonable proxy.

> **THESIS SUGGESTION (GRAPHICS SUMMARY):** Across the full thesis, the following figures would form a strong visual narrative:
> 1. Atom-count histogram of fine-tune dataset with 20-atom and 51-atom ceiling lines
> 2. Stacked atom-count histogram of base dataset coloured by source
> 3. Funnel/waterfall chart of COD filtering stages
> 4. Property repair scatter plot (assigned vs correct ΔS, before and after)
> 5. Training loss curves (base pretrain: total, cell, position, species components)
> 6. Fine-tune loss curve with 50-epoch moving average overlaid
> 7. Comparison table: this dataset vs MatterGen's published datasets
> 8. (If available) Example generated structures visualised with VESTA or pymatgen

---

## 14. Phase 8: Superconductor Dataset Refinement & Additional Fine-Tuning (~6 May onwards)

### 14.1 Superconductor T_c Parsing Fix

The initial superconductor dataset extraction from NEMAD yielded only 58 rows because the T_c column contained free-text strings (e.g., "90 K", "up to 2.8 K", "< 4.2 K") that were silently dropped by the initial numeric parser. Refined Claude Code instructions were written to re-parse the `Superconducting_Transition_Temperature` column using regex extraction of numeric values from free-text strings. This recovered **1,088 usable rows** — an 18× improvement. The dataset was split 1000/88 (train/val).

### 14.2 Superconductor Fine-Tuning Setup

The complete pipeline for superconductor fine-tuning was configured:

- Property embedding config (`sc_transition_temp.yaml`) created following the template of existing property configs
- Data module config (`finetune_superconductor.yaml`) created
- `sc_transition_temp` registered in `globals.py` alongside the magnetocaloric properties
- `csv-to-dataset` run to preprocess into NumPy cache
- A `root_dir` path issue in the data module config was identified and corrected (an initial suggestion to change `../../` to `../../../` was wrong — the original `../../` was correct)
- SLURM batch script `finetune_sc.sbatch` created for 1000 epochs on a single GPU

### 14.3 HPC Disk Space Crisis

The file count quota was hit (338k files vs 300k soft limit). Diagnosis identified `.conda` as the primary culprit (184k files, 14 GB). Resolved with `conda clean --all -y`. This is a recurring operational issue on Stanage — the uv and pip caches also contribute and were redirected to scratch storage earlier in the project.

### 14.4 Joint Magnetocaloric Fine-Tuning (ΔS + Curie Temperature)

A joint fine-tuning model was trained to condition on both `magnetocaloric_delta_s` and `magnetocaloric_curie_temp` simultaneously. Key design decisions:

- MatterGen supports multi-property conditioning but requires training a **single adapter on both properties simultaneously** — separately trained single-property adapters cannot be combined at inference time.
- The dataset already contained `magnetocaloric_curie_temp` in the CSVs and preprocessed cache, and the property was already registered in `globals.py`, so no data pipeline changes were needed.
- A `magnetocaloric_curie_temp.yaml` property embedding config was created.
- `finetune_joint.sbatch` was created, identical to the working ΔS-only script except for: adding the second property embedding override, listing both properties in `data_module.properties`, and critically setting **`dropout_fields_iid=true`**.

The `dropout_fields_iid=true` setting is significant: it causes the model to randomly drop individual property conditions during training, so the model learns all four conditioning modes: p(x), p(x|ΔS), p(x|T_C), and p(x|ΔS, T_C). Without this, the model would only learn two modes: fully conditioned p(x|ΔS, T_C) and fully unconditioned p(x), and single-property conditioning at inference would be unreliable.

### 14.5 Checkpoint Backup

Before overwriting the fine-tune output directory with the joint model, the single-property ΔS fine-tune was backed up. The essential files for model recovery are the `.hydra/config.yaml` plus the best checkpoint `.ckpt` file. The ΔS-only fine-tune checkpoint (`epoch=739-loss_val=0.60.ckpt`) was saved to Google Drive.

---

## 15. Phase 9: Generation and Evaluation Plan

### 15.1 Generation Strategy

A systematic generation plan was designed to evaluate all trained models across their conditioning distributions using a standard-deviation sampling strategy. For each conditioned model, structures are generated at the median training value and at +1σ, +2σ, +3σ, and +4σ above the median. This probes performance from interpolation (well-represented training data) through to deep extrapolation (far beyond training distribution), providing a complete picture of model reliability.

The conditioning values were computed from the training data distributions:

**Magnetocaloric ΔS** (n=292, median=5.3, σ=13.6 J/kg·K): 5.3, 18.9, 32.5, 46.1, 59.7

**Magnetocaloric T_C** (n=298, median=242, σ=93 K): 242, 335, 428, 521, 614

**Superconductor T_c** (n=1000, median=3.8, σ=36.5 K): 3.8, 40.3, 76.9, 113.4, 149.9

An additional joint-conditioned run generates structures at three ΔS levels (median, +1σ, +2σ) all at T_C = 300 K (room temperature), targeting the scientifically relevant regime for magnetic refrigeration applications.

### 15.2 Generation Runs

| Run | Model | Structures | Conditioning |
|-----|-------|-----------|--------------|
| 01 | Base pretrain | 1,008 | Unconditional |
| 02a–e | ΔS-only fine-tune | 5 × 208 = 1,040 | ΔS at 5 levels |
| 03a–c | Joint fine-tune | 3 × 112 = 336 | ΔS + T_C=300K at 3 ΔS levels |
| 04a–e | Joint fine-tune | 5 × 208 = 1,040 | T_C only at 5 levels |
| 05a–e | SC fine-tune | 5 × 208 = 1,040 | T_c at 5 levels |
| **Total** | | **4,464** | |

### 15.3 Evaluation Pipeline

Each sub-run is evaluated immediately after generation using `mattergen-evaluate` with:
- MatterSim relaxation (`--relax=True`)
- TRI2024 energy correction scheme and reference dataset (recommended for non-benchmarking evaluation)
- Per-structure detailed metrics saved via `--save_detailed_as`
- Disordered structure matcher

The TRI2024 reference dataset (860 MB) was obtained by cloning the MatterGen repo into scratch and pulling via Git LFS — the existing file in the project repo was a 134-byte LFS pointer stub, not the actual data.

### 15.4 Script Design

The generation+evaluation script was designed to:
- Generate then evaluate each sub-run sequentially, so partial results are preserved if the job times out
- Continue on failure rather than aborting (a failed conditioning level should not prevent subsequent runs)
- Log timestamps and exit codes for every step to `GENERATION_LOG.txt`
- Use `--checkpoint_epoch=best` to automatically select the lowest validation loss checkpoint
- Disable trajectory recording to save disk space

### 15.5 Verified Checkpoints

All four model checkpoints were confirmed to exist and will be correctly selected:

| Model | Checkpoint | Val Loss |
|-------|-----------|----------|
| Base pretrain | `epoch=959-loss_val=0.52.ckpt` | 0.52 |
| ΔS-only fine-tune | `epoch=739-loss_val=0.60.ckpt` | 0.60 |
| Joint ΔS+T_C fine-tune | `epoch=969-loss_val=0.59.ckpt` | 0.59 |
| Superconductor T_c fine-tune | `epoch=159-loss_val=0.23.ckpt` | 0.23 |

> **THESIS SUGGESTION:** The checkpoint table above belongs in the methods section. Note the SC model's low validation loss (0.23 vs 0.59–0.60 for magnetocaloric models) — this is likely because the SC dataset is much larger (1000 train vs 284 train) rather than because SC structures are intrinsically easier to model. Worth a sentence of discussion.

> **THESIS SUGGESTION (ADDITIONAL GRAPHICS):** The generation results will enable several additional figures:
> 9. S.U.N. vs conditioning level (σ) for each model — expect declining performance at higher σ
> 10. Element frequency histograms of generated structures at different conditioning levels — do high-ΔS structures preferentially contain rare-earth elements?
> 11. Space group distribution of generated structures vs training data
> 12. Comparison table: base model metrics vs published MatterGen benchmarks (mp_20_base, mattergen_base, CDVAE, G-SchNet)
> 13. Cherry-picked VESTA visualisations of the most promising magnetocaloric candidates (valid, novel, stable, MCE-relevant chemistry)

---

## 16. Updated Chronological Summary of Agent Usage

| Period | Agent/Tool | What Was Done |
|--------|-----------|---------------|
| Late 2025 | Claude.ai chat | MatterGen installation, basic concepts, first HPC test |
| ~Nov 2024 → early 2026 | Claude.ai chat | Franco PDF parsing, MP cross-referencing, property conditioning design |
| ~Mar 2026 | Claude.ai chat | Wyckoff pipeline v12–v13, diagnostic scripts, ICSD CIF acquisition |
| 24 Mar → 10 Apr | Claude Code (`Pre-mattergen_dataprocessing`) | v14 pipeline execution, SQS generation, deduplication → 374 CIFs |
| 13 Apr → 22 Apr | Claude Code (`Clean_Preprocessing`) | CHGNet relaxation, MatterGen CSV export, atom ceiling expansion |
| 21 Apr → 27 Apr | Claude Code (`base_model_dataset`) | COD/MP/Alexandria/OQMD filtering, WSL debugging, 274k dataset |
| 26 Apr → 27 Apr | Claude Code (`mattergen_project`) | Local smoke testing, GemNet path patch, property repair |
| ~27 Apr → 30 Apr | Claude.ai chat (HPC session) | HPC deployment, environment setup, pretrain debugging |
| 30 Apr → 6 May | Claude.ai chat | Pretrain verification, ΔS fine-tune (200 then 1000 epochs) |
| 6 May | Claude Code (`nemad_labelled`) | NEMAD CIF enrichment |
| 6 May → 9 May | Claude.ai chat | SC T_c parsing fix, SC fine-tune setup, joint ΔS+T_C fine-tune, disk quota crisis, generation plan |
| 9 May | Claude.ai chat (this session) | Project reference document, data analysis CLAUDE.md, generation+evaluation instructions |
