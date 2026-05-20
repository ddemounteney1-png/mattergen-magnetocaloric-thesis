# Checkpoints

Model checkpoint documentation. The `.ckpt` files themselves are **not committed to this repository** due to size (each is several GB). They are stored on Stanage HPC (`/mnt/parscratch/users/<username>/`) and backed up to Google Drive.

---

## Checkpoint Inventory

### Base pre-train checkpoint

**File:** `epoch=959-loss_val=0.52.ckpt`
**Training run:** Pre-train from scratch on 274,602 custom base dataset structures
**Hardware:** 4× NVIDIA A100 (80 GB), Sheffield Stanage HPC
**Epochs:** 1,000 (best checkpoint at epoch 959)
**Final val loss:** 0.52
**Architecture:** GemNetT denoiser, no property conditioning
**Config:** `configs/pretrain_resolved_config.yaml`

---

### ΔS-only magnetocaloric fine-tune checkpoint

**File:** `epoch=739-loss_val=0.60.ckpt`
**Training run:** Fine-tune on 284 magnetocaloric training structures, conditioning on `magnetocaloric_delta_s` only
**Hardware:** 1× NVIDIA A100 (80 GB), Sheffield Stanage HPC
**Epochs:** 1,000 (best checkpoint at epoch 739)
**Final val loss:** 0.60
**Architecture:** GemNetTAdapter with `magnetocaloric_delta_s` property embedding
**Config:** `configs/finetune_mc_1000_resolved_config.yaml`

---

### Joint ΔS + T_C magnetocaloric fine-tune checkpoint

**File:** `epoch=969-loss_val=0.59.ckpt`
**Training run:** Fine-tune on ~291 magnetocaloric training structures, conditioning on both `magnetocaloric_delta_s` and `magnetocaloric_curie_temp`
**Hardware:** 1× NVIDIA A100 (80 GB), Sheffield Stanage HPC
**Epochs:** 1,000 (best checkpoint at epoch 969)
**Final val loss:** 0.59
**Architecture:** GemNetTAdapter with both ΔS and T_C property embeddings
**Note:** Config not available locally; loss CSV at `results/loss_curves/finetune_joint_losses.csv`

---

### Superconductor T_c fine-tune checkpoint

**File:** `epoch=159-loss_val=0.23.ckpt`
**Training run:** Fine-tune on 1,000 superconductor training structures, conditioning on `sc_transition_temp`
**Hardware:** 1× NVIDIA A100 (80 GB), Sheffield Stanage HPC
**Epochs:** 1,000 (best checkpoint at epoch 159)
**Final val loss:** 0.23
**Architecture:** GemNetTAdapter with `sc_transition_temp` property embedding
**Config:** `configs/finetune_sc_resolved_config.yaml`

---

## How to use checkpoints

To generate structures from a checkpoint, activate the MatterGen venv and run:

```bash
mattergen-generate \
  --checkpoint_path /path/to/epoch=739-loss_val=0.60.ckpt \
  --properties_to_condition_on "magnetocaloric_delta_s=18.9" \
  --num_samples 100 \
  --output_dir generated/
```

For the joint model, provide both properties:

```bash
mattergen-generate \
  --checkpoint_path /path/to/epoch=969-loss_val=0.59.ckpt \
  --properties_to_condition_on "magnetocaloric_delta_s=18.9,magnetocaloric_curie_temp=300" \
  --num_samples 100 \
  --output_dir generated/
```

To evaluate generated structures:

```bash
mattergen-evaluate \
  --structures_path generated/ \
  --output_path results/ \
  --reference_dataset_path data-release/alex-mp/reference_TRI2024correction.gz
```
