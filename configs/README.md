# Configs

Training configurations for all MatterGen runs.

---

## Resolved Hydra Configs

These are the fully-resolved Hydra configurations actually used for training runs on Stanage HPC. They capture all hyperparameters, dataset paths, and model architecture settings.

| File | Run | Description |
|------|-----|-------------|
| `pretrain_resolved_config.yaml` | Base pretrain | Configuration for pre-training on 274,602 structures. 4× A100, ~1000 epochs. Final checkpoint: `epoch=959-loss_val=0.52.ckpt`. |
| `finetune_mc_1000_resolved_config.yaml` | Magnetocaloric fine-tune (ΔS-only) | Configuration for the 1000-epoch ΔS-only fine-tune on 284 training structures. 1× A100. Conditions on `magnetocaloric_delta_s`. Final checkpoint: `epoch=739-loss_val=0.60.ckpt`. |
| `finetune_sc_resolved_config.yaml` | Superconductor fine-tune | Configuration for superconductor fine-tune on 1,000 training structures. Conditions on `sc_transition_temp`. Final checkpoint: `epoch=159-loss_val=0.23.ckpt`. |

Note: A joint ΔS + T_C fine-tune was also run; its config is not available locally but the training loss CSV is at `results/loss_curves/finetune_joint_losses.csv`.

---

## configs/property_embeddings/

YAML files defining the property embedding architecture for each conditioning property. These are loaded by MatterGen's Hydra config system.

| File | Property | Description |
|------|----------|-------------|
| `magnetocaloric_delta_s.yaml` | `magnetocaloric_delta_s` | Continuous scalar embedding for magnetocaloric entropy change (J/kg·K). Used in ΔS-only and joint fine-tunes. |
| `magnetocaloric_curie_temp.yaml` | `magnetocaloric_curie_temp` | Continuous scalar embedding for Curie temperature (K). Used in joint ΔS+T_C fine-tune. |
| `magnetocaloric_field.yaml` | `magnetocaloric_field` | Continuous scalar embedding for applied magnetic field (T). Present in dataset but not used as a conditioning property in these runs. |

These YAML files follow the same schema as the upstream MatterGen property embeddings (e.g. `dft_band_gap.yaml`). To add a new conditioning property, create a matching YAML in this directory, add the property name to `PROPERTY_SOURCE_IDS` in `mattergen/common/utils/globals.py`, and reference it in the finetune config.
