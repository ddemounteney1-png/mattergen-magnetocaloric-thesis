#!/bin/bash
#SBATCH --job-name=mg_evaluate
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --time=24:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
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

RESULTS_ROOT="/mnt/parscratch/users/$USER/mattergen_outputs/generation_results"
REF_DATASET="$HOME/mattergen_project/mattergen/data-release/alex-mp/reference_TRI2024correction.gz"
LOG_FILE="$RESULTS_ROOT/EVALUATION_LOG.txt"

log_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_evaluate() {
    local OUTPUT_DIR="$1"
    local LABEL="$2"

    # Create a unique temp directory for each evaluation to avoid LMDB conflicts
    local EVAL_TMPDIR=$(mktemp -d /tmp/mattergen_eval_XXXXXX)
    export TMPDIR="$EVAL_TMPDIR"

    log_step "EVALUATE $LABEL — starting"

    local START_TIME=$(date +%s)
    mattergen-evaluate \
        --structures_path="$OUTPUT_DIR/" \
        --relax=True \
        --structure_matcher='disordered' \
        --save_as="$OUTPUT_DIR/metrics.json" \
        --save_detailed_as="$OUTPUT_DIR/detailed_metrics.json" \
        --reference_dataset_path="$REF_DATASET" \
        --energy_correction_scheme="TRI2024"
    local EXIT_CODE=$?
    local ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))

    # Clean up temp directory
    rm -rf "$EVAL_TMPDIR"
    unset TMPDIR

    log_step "EVALUATE $LABEL — EXIT CODE $EXIT_CODE — ${ELAPSED} min"
    return $EXIT_CODE
}

log_step "========== EVALUATION STARTING =========="

# Run 1: Base unconditional
run_evaluate "$RESULTS_ROOT/01_base_unconditional" "01_base_unconditional"

# Run 2: Delta-S only
for dir in median_5.3 plus1sigma_18.9 plus2sigma_32.5 plus3sigma_46.1 plus4sigma_59.7; do
    run_evaluate "$RESULTS_ROOT/02_deltaS_only/$dir" "02_deltaS_only/$dir"
done

# Run 3: Joint Delta-S + T_C at room temp
for dir in deltaS_5.3_TC_300 deltaS_18.9_TC_300 deltaS_32.5_TC_300; do
    if [ -d "$RESULTS_ROOT/03_joint_deltaS_TC_roomtemp/$dir" ]; then
        run_evaluate "$RESULTS_ROOT/03_joint_deltaS_TC_roomtemp/$dir" "03_joint_deltaS_TC_roomtemp/$dir"
    fi
done

# Run 4: Joint T_C only
for dir in median_242 plus1sigma_335 plus2sigma_428 plus3sigma_521 plus4sigma_614; do
    if [ -d "$RESULTS_ROOT/04_joint_TC_only/$dir" ]; then
        run_evaluate "$RESULTS_ROOT/04_joint_TC_only/$dir" "04_joint_TC_only/$dir"
    fi
done

# Run 5: Superconductor
for dir in median_3.8 plus1sigma_40.3 plus2sigma_76.9 plus3sigma_113.4 plus4sigma_149.9; do
    if [ -d "$RESULTS_ROOT/05_superconductor/$dir" ]; then
        run_evaluate "$RESULTS_ROOT/05_superconductor/$dir" "05_superconductor/$dir"
    fi
done

log_step "========== ALL EVALUATION COMPLETE =========="
