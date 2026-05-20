# Training MatterGen for Magnetocaloric Crystal Structure Generation

This repository contains the full code, data pipeline, training configurations, evaluation metrics, and thesis figures for a masters project that adapts Microsoft's MatterGen generative diffusion model to conditionally generate magnetocaloric crystal structures. The work trains a base model from scratch on a custom 274,602-structure dataset assembled from Materials Project, the Crystallography Open Database (COD), and Alexandria, then fine-tunes the model to condition on the magnetocaloric entropy change (ΔS) and Curie temperature (T_C) of target materials. A parallel superconductor fine-tune on 1,088 structures validates the transfer-learning pipeline. All training ran on the University of Sheffield Stanage HPC cluster (4× NVIDIA A100 GPUs for pre-training; 1× A100 for fine-tuning).

## Repository Structure

```
mattergen-magnetocaloric-thesis/
├── code/
│   ├── data_parsing/          # Magnetocaloric CIF pipeline (v14 final) + utilities
│   ├── structure_pipeline/    # SQS ordering, CHGNet relaxation, dataset preparation
│   ├── base_dataset/          # 8-stage pipeline to build 274k-structure base dataset
│   ├── analysis/              # Figure generation scripts (fig01–fig29) + loss parsers
│   └── slurm_scripts/         # SLURM .sbatch files for HPC training runs
├── data/
│   ├── source/                # Franco et al. magnetocaloric source CSV + pipeline logs
│   ├── finetune/              # train_repaired.csv + val_repaired.csv (315 structures)
│   ├── superconductor/        # Superconductor train/val CSVs (1,088 structures)
│   └── base_metadata/         # Metadata-only views of base training splits (no CIF strings)
├── configs/
│   ├── property_embeddings/   # YAML configs for ΔS, T_C, and field property embeddings
│   ├── pretrain_resolved_config.yaml
│   ├── finetune_mc_1000_resolved_config.yaml
│   └── finetune_sc_resolved_config.yaml
├── results/
│   ├── metrics/               # All mattergen-evaluate JSON output (50 files, multiple runs)
│   ├── loss_curves/           # Epoch-by-epoch training loss CSVs (4 runs)
│   ├── generated_structures/  # (empty — generated CIFs live on HPC)
│   └── figures/               # All 29 thesis figures as PNG + PDF
├── checkpoints/               # Checkpoint documentation (weights stored on HPC/Drive)
└── docs/                      # Reference data: relaxation logs, repair reports, quality CSVs
```

## Key Numbers

| Item | Count |
|------|-------|
| Base model training structures | 274,602 |
| Magnetocaloric fine-tune structures | 315 (284 train / 31 val) |
| Superconductor fine-tune structures | 1,088 (1,000 train / 88 val) |
| Source magnetocaloric compositions (Franco et al.) | 409 unique |
| Thesis figures generated | 29 |
| Evaluation metric runs (JSON files) | 50 |

## Reproduction Instructions

### Environment

```bash
# MatterGen venv (for training and generation)
source ~/mattergen_project/mattergen/.venv/bin/activate

# Data pipeline env (pymatgen, matminer)
source ~/Pre-mattergen_dataprocessing/matminer_env/bin/activate

# Base dataset env (pymatgen, ASE)
conda activate base_model_env
```

### Data pipeline (magnetocaloric)

```bash
# 1. Build ordered CIFs from Materials Project prototypes
python code/data_parsing/magnetocaloric_wyckoff_pipeline_v14.py

# 2. SQS ordering for disordered structures
python code/structure_pipeline/step1_generate_sqs_v2.py

# 3. CHGNet relaxation + dataset export
python code/structure_pipeline/prepare_dataset.py
```

### Base model dataset

```bash
# Run stages 01–08 in sequence (see code/base_dataset/)
python code/base_dataset/01_filter_cod.py --workers 7
python code/base_dataset/02_load_filter_mp.py
# ... through 08_final_merge.py
```

### Training on HPC

Pre-training and fine-tuning were launched via SLURM (see `code/slurm_scripts/`). Resolved Hydra configs are in `configs/`.

### Generating figures

```bash
cd Data_analysis/  # scripts expect relative data paths
python code/analysis/fig11_convergence_comparison.py
# etc. for fig01–fig29
```

## Checkpoints

Model weights are stored on Stanage HPC and Google Drive (not in this repo due to size). See `checkpoints/README.md` for details.

## Author

**Daniel De Mounteney**
University of Sheffield — MEng General Engineering, 2026
Supervisor: Prof Nicola Morley
