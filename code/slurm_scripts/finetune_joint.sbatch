#!/bin/bash
#SBATCH --job-name=mg_finetune_joint
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --time=08:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.8.0
module swap GCCcore/12.2.0 GCCcore/11.3.0

export UV_CACHE_DIR=/mnt/parscratch/users/$USER/uv_cache
export PIP_CACHE_DIR=/mnt/parscratch/users/$USER/pip_cache
export OUTPUT_DIR=/mnt/parscratch/users/$USER/mattergen_outputs/finetune_joint

cd $HOME/mattergen_project/mattergen
source .venv/bin/activate

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

export PROPERTY1=magnetocaloric_delta_s
export PROPERTY2=magnetocaloric_curie_temp
mattergen-finetune \
    adapter.pretrained_name=null \
    adapter.model_path=/mnt/parscratch/users/$USER/mattergen_outputs/pretrain \
    data_module=finetune_magnetocaloric \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
    ~trainer.logger \
    'data_module.properties=["magnetocaloric_delta_s","magnetocaloric_curie_temp"]' \
    ++lightning_module.diffusion_module.pre_corruption_fn.dropout_fields_iid=true \
    trainer.accelerator=gpu \
    ++trainer.devices=1 \
    ++data_module.batch_size.train=8 \
    ++data_module.batch_size.val=8 \
    ++data_module.num_workers.train=2 \
    ++data_module.num_workers.val=2 \
    ++trainer.max_epochs=1000

echo "Joint fine-tuning completed."
