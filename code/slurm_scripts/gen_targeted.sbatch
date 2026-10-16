#!/bin/bash
#SBATCH --job-name=mg_gen_targeted
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_targeted_%j.out
#SBATCH --error=logs/gen_targeted_%j.err
#SBATCH --mail-type=ALL

set -uo pipefail

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.8.0
module swap GCCcore/12.2.0 GCCcore/11.3.0

export UV_CACHE_DIR=/mnt/parscratch/users/$USER/uv_cache
export PIP_CACHE_DIR=/mnt/parscratch/users/$USER/pip_cache

cd $HOME/mattergen_project/mattergen
source .venv/bin/activate

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

RESULTS_ROOT="/mnt/parscratch/users/$USER/mattergen_outputs/generation_results/08_targeted_useful"
JOINT_MODEL="/mnt/parscratch/users/$USER/mattergen_outputs/finetune_joint"
REF_DATASET="$HOME/mattergen_project/mattergen/data-release/alex-mp/reference_TRI2024correction.gz"

TC_VALUES=("3.15" "270" "300")
TC_LABELS=("supercooling_3K" "freezer_270K" "aircon_300K")
DS_VALUE="32.5"

for i in "${!TC_VALUES[@]}"; do
    TC="${TC_VALUES[$i]}"
    LABEL="${TC_LABELS[$i]}"
    OUTPUT_DIR="$RESULTS_ROOT/$LABEL"
    mkdir -p "$OUTPUT_DIR"

    echo "=== Generating $LABEL (dS=$DS_VALUE, Tc=$TC) at $(date) ==="

    mattergen-generate "$OUTPUT_DIR/" \
        --model_path="$JOINT_MODEL" \
        --batch_size=16 \
        --num_batches=13 \
        --checkpoint_epoch=best \
        --properties_to_condition_on="{'magnetocaloric_delta_s': $DS_VALUE, 'magnetocaloric_curie_temp': $TC}" \
        --diffusion_guidance_factor=2.0 \
        --record_trajectories=False

    GEN_EXIT=$?
    echo "=== Generation $LABEL finished with exit code $GEN_EXIT at $(date) ==="

    if [ $GEN_EXIT -eq 0 ]; then
        echo "=== Evaluating $LABEL at $(date) ==="

        mattergen-evaluate \
            --structures_path="$OUTPUT_DIR/generated_crystals.extxyz" \
            --relax=True \
            --structure_matcher='ordered' \
            --save_as="$OUTPUT_DIR/metrics.json" \
            --save_detailed_as="$OUTPUT_DIR/detailed_metrics.json" \
            --reference_dataset_path="$REF_DATASET" \
            --energy_correction_scheme=TRI2024

        EVAL_EXIT=$?
        echo "=== Evaluation $LABEL finished with exit code $EVAL_EXIT at $(date) ==="
    else
        echo "=== SKIPPING evaluation for $LABEL — generation failed ==="
    fi
done

echo "=== ALL TARGETED RUNS COMPLETE at $(date) ==="
