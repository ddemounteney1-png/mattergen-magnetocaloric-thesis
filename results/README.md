# Results

Evaluation metrics, training loss curves, and thesis figures from all model runs.

---

## results/metrics/

All `mattergen-evaluate` output JSON files. Each evaluation run generates two files:
- `metrics.json` — summary metrics (stability rate, uniqueness, novelty, S.U.N.)
- `detailed_metrics.json` — per-structure breakdown

### Directory structure

```
metrics/
├── 01_base_unconditional/         # Base model, no conditioning (2 files)
├── Delta_S_only/                  # ΔS-only fine-tune sweeps (10 files)
│   ├── median_5.3_deltaS_only_*
│   ├── deltaS_only_plus1sigma_18.9_*
│   ├── plus2sigma_32.5_deltaS_only_*
│   ├── deltaS_only_plus3sigma_46.1_*
│   └── Delta_S_only_plus4sigma_59.7_*
├── 04_joint_TC_only/              # Joint model, T_C sweep (10 files)
│   ├── median_242/, plus1sigma_335/, plus2sigma_428/
│   ├── plus3sigma_521/, plus4sigma_614/
├── fixed_Tc_delta_S_sweep/        # Fixed T_C=300K, ΔS sweep (6 files)
│   ├── deltaS_5.3_TC_300_*, deltaS_18.9_TC_300_*, deltaS_32.5_TC_300_*
├── 06_fixed_deltaS_TC_sweep/      # Fixed ΔS, T_C sweep (6 files)
│   ├── ds10p5_tc242_*, ds10p5_tc335_*, ds10p5_tc428_*
├── 07_simultaneous_escalation/    # Joint escalation (6 files)
│   ├── ds5p3_tc242_*, ds18p9_tc335_*, ds32p5_tc428_*
├── Superconductor/                # SC fine-tune, T_c sweep (10 files)
│   ├── superconductor_median_3.8_*, superconductor_plus1sigma_40.3_*
│   ├── superconductor_plus2sigma_76.9_*, superconductor_plus3sigma_113.4_*
│   └── superconductor_plus4sigma_149.9_*
└── other_mainstream_model_metrics/  # Comparison baselines (7 files)
    ├── cdvae.json, diffcsp_mp_20.json, ftcp.json
    ├── g_schnet.json, mattergen.json, mattergen_mp_20.json
    └── p_g_schnet.json
```

Total: 50 evaluation JSON files.

---

## results/loss_curves/

Epoch-by-epoch training loss extracted from SLURM output logs.

| File | Epochs | Run | Notes |
|------|--------|-----|-------|
| `pretrain_losses.csv` | 1,000 | Base pretrain | 1,717 batches/epoch, 4× A100. Final val loss: 0.52. |
| `finetune_1000_losses.csv` | 1,000 | ΔS-only fine-tune | 37 batches/epoch, 1× A100. Best val loss: 0.60 at epoch 739. |
| `finetune_joint_losses.csv` | 1,000 | Joint ΔS+T_C fine-tune | 37 batches/epoch. Best val loss: 0.59 at epoch 969. |
| `finetune_sc_losses.csv` | 1,000 | Superconductor fine-tune | 125 batches/epoch. Best val loss: 0.23 at epoch 159. |

Columns: `epoch, loss_train_epoch, pos_train_epoch, cell_train_epoch, atomic_numbers_train_epoch`

---

## results/figures/

All 29 thesis figures as PNG (300 dpi) and PDF (vector). Figures are numbered sequentially: fig01–fig29. See `code/analysis/` for the generating scripts.

| Figure | Description |
|--------|-------------|
| fig01 | Atom-count distribution of fine-tuning dataset |
| fig02 | Structure match type and ordered/SQS split |
| fig03 | Stoichiometric error distribution |
| fig04 | Dataset construction pipeline schematic |
| fig05 | Base dataset composition (atom count, sources, phase systems) |
| fig06 | Property repair before/after scatter |
| fig07 | COD filtering waterfall (533k → 33.8k) |
| fig08 | ΔS, T_C, and applied field violin plots |
| fig09 | Unique compositions by material class |
| fig10 | ΔS vs T_C scatter coloured by space group |
| fig11 | Convergence comparison (all 4 training runs) |
| fig12 | Base dataset source contributions after deduplication |
| fig13 | S.U.N. metrics vs conditioning level (ΔS-only) |
| fig14 | Element frequency in generated structures |
| fig15 | Space group distribution: training vs generated |
| fig16 | Joint conditioning sweep results |
| fig17 | S.U.N. comparison across model variants |
| fig18 | Fine-tuned vs baseline MatterGen S.U.N. metrics |
| fig19 | Space group comparison across models |
| fig20 | ΔS conditioning effect on generation (line plots) |
| fig21 | ΔS conditioning combined summary |
| fig22 | Element frequency broken down by ΔS conditioning level |
| fig23 | Joint conditioning component analysis |
| fig24 | Energy-above-hull violin by conditioning level |
| fig25 | MCE material family classification of generated structures |
| fig26 | Base dataset source breakdown bar chart |
| fig27 | Superconductor vs joint fine-tune loss comparison |
| fig28 | T_C vs SC transition temperature S.U.N. comparison |
| fig29 | Property distribution comparison across models |
| dyco_convex_hull | Dy-Co binary convex hull (appendix/context figure) |

---

## results/generated_structures/

This directory is empty in the repository. Generated CIF files from model inference runs are stored on Stanage HPC. Not available locally due to size.
