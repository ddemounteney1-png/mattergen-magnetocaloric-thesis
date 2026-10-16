#!/bin/bash
#SBATCH --job-name=mg_generate
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --time=24:00:00
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --mail-type=ALL

set -uo pipefail
# NOTE: not using -e because we want to continue if individual runs fail

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.8.0
module swap GCCcore/12.2.0 GCCcore/11.3.0

export UV_CACHE_DIR=/mnt/parscratch/users/$USER/uv_cache
export PIP_CACHE_DIR=/mnt/parscratch/users/$USER/pip_cache

cd $HOME/mattergen_project/mattergen
source .venv/bin/activate

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

# ============================================================
# CHECKPOINT PATHS — edit these if locations change
# ============================================================
PRETRAIN_MODEL="/mnt/parscratch/users/$USER/mattergen_outputs/pretrain"
FINETUNE_DELTAS="/mnt/parscratch/users/$USER/mattergen_outputs/finetune"
FINETUNE_JOINT="/mnt/parscratch/users/$USER/mattergen_outputs/finetune_joint"
FINETUNE_SC="/mnt/parscratch/users/$USER/mattergen_outputs/finetune_sc"

RESULTS_ROOT="/mnt/parscratch/users/$USER/mattergen_outputs/generation_results"
MATTERGEN_ROOT="$HOME/mattergen_project/mattergen"
REF_DATASET="$MATTERGEN_ROOT/data-release/alex-mp/reference_TRI2024correction.gz"

LOG_FILE="$RESULTS_ROOT/GENERATION_LOG.txt"

mkdir -p "$RESULTS_ROOT"

# ============================================================
# HELPER FUNCTIONS
# ============================================================
log_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_generate() {
    local OUTPUT_DIR="$1"
    local MODEL_PATH="$2"
    local BATCH_SIZE="$3"
    local NUM_BATCHES="$4"
    local LABEL="$5"
    shift 5
    # Remaining args are optional (--properties_to_condition_on, --diffusion_guidance_factor etc.)

    mkdir -p "$OUTPUT_DIR"
    log_step "GENERATE $LABEL — starting (${BATCH_SIZE}x${NUM_BATCHES}=$((BATCH_SIZE * NUM_BATCHES)) structures)"

    local START_TIME=$(date +%s)
    mattergen-generate "$OUTPUT_DIR/" \
        --model_path="$MODEL_PATH" \
        --batch_size="$BATCH_SIZE" \
        --num_batches="$NUM_BATCHES" \
        --checkpoint_epoch=best \
        --record_trajectories=False \
        "$@"
    local EXIT_CODE=$?
    local ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))

    log_step "GENERATE $LABEL — EXIT CODE $EXIT_CODE — ${ELAPSED} min"
    return $EXIT_CODE
}

run_evaluate() {
    local OUTPUT_DIR="$1"
    local LABEL="$2"

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

    log_step "EVALUATE $LABEL — EXIT CODE $EXIT_CODE — ${ELAPSED} min"
    return $EXIT_CODE
}

run_generate_and_evaluate() {
    local OUTPUT_DIR="$1"
    local MODEL_PATH="$2"
    local BATCH_SIZE="$3"
    local NUM_BATCHES="$4"
    local LABEL="$5"
    shift 5

    run_generate "$OUTPUT_DIR" "$MODEL_PATH" "$BATCH_SIZE" "$NUM_BATCHES" "$LABEL" "$@"
    if [ $? -eq 0 ]; then
        run_evaluate "$OUTPUT_DIR" "$LABEL"
    else
        log_step "SKIPPING EVALUATE $LABEL — generation failed"
    fi
}

# ============================================================
# PRE-FLIGHT: Check reference dataset
# ============================================================
if [ ! -f "$REF_DATASET" ]; then
    log_step "ERROR: Reference dataset not found at $REF_DATASET"
    log_step "Run: cd $MATTERGEN_ROOT && git lfs pull -I data-release/alex-mp/reference_TRI2024correction.gz --exclude=\"\""
    log_step "Continuing without evaluation — generation only"
    SKIP_EVAL=true
else
    SKIP_EVAL=false
fi

# Override run_evaluate if ref dataset missing
if [ "$SKIP_EVAL" = true ]; then
    run_generate_and_evaluate() {
        local OUTPUT_DIR="$1"
        local MODEL_PATH="$2"
        local BATCH_SIZE="$3"
        local NUM_BATCHES="$4"
        local LABEL="$5"
        shift 5
        run_generate "$OUTPUT_DIR" "$MODEL_PATH" "$BATCH_SIZE" "$NUM_BATCHES" "$LABEL" "$@"
        log_step "SKIPPING EVALUATE $LABEL — reference dataset not available"
    }
fi

log_step "========== GENERATION AND EVALUATION STARTING =========="

# ============================================================
# RUN 1: Base Model — Unconditional (1008 structures)
# ============================================================
log_step "===== RUN 1: Base Unconditional ====="
run_generate_and_evaluate \
    "$RESULTS_ROOT/01_base_unconditional" \
    "$PRETRAIN_MODEL" \
    16 63 \
    "01_base_unconditional"

# ============================================================
# RUN 2: Delta-S Only Fine-Tune — 5 conditioning levels
# ============================================================
log_step "===== RUN 2: Delta-S Only ====="

DELTAS_VALUES=("5.3" "18.9" "32.5" "46.1" "59.7")
DELTAS_LABELS=("median_5.3" "plus1sigma_18.9" "plus2sigma_32.5" "plus3sigma_46.1" "plus4sigma_59.7")

for i in "${!DELTAS_VALUES[@]}"; do
    VALUE="${DELTAS_VALUES[$i]}"
    LABEL="${DELTAS_LABELS[$i]}"

    run_generate_and_evaluate \
        "$RESULTS_ROOT/02_deltaS_only/$LABEL" \
        "$FINETUNE_DELTAS" \
        16 13 \
        "02_deltaS_only/$LABEL" \
        --properties_to_condition_on="{'magnetocaloric_delta_s': $VALUE}" \
        --diffusion_guidance_factor=2.0
done

# ============================================================
# RUN 3: Joint Model — Delta-S + T_C at Room Temperature
# ============================================================
if [ -d "$FINETUNE_JOINT" ] && [ -f "$FINETUNE_JOINT/config.yaml" ]; then
    log_step "===== RUN 3: Joint Delta-S + T_C at Room Temp ====="

    JOINT_DELTAS=("5.3" "18.9" "32.5")
    JOINT_LABELS=("deltaS_5.3_TC_300" "deltaS_18.9_TC_300" "deltaS_32.5_TC_300")

    for i in "${!JOINT_DELTAS[@]}"; do
        DS="${JOINT_DELTAS[$i]}"
        LABEL="${JOINT_LABELS[$i]}"

        run_generate_and_evaluate \
            "$RESULTS_ROOT/03_joint_deltaS_TC_roomtemp/$LABEL" \
            "$FINETUNE_JOINT" \
            16 7 \
            "03_joint_deltaS_TC_roomtemp/$LABEL" \
            --properties_to_condition_on="{'magnetocaloric_delta_s': $DS, 'magnetocaloric_curie_temp': 300}" \
            --diffusion_guidance_factor=2.0
    done

    # ============================================================
    # RUN 4: Joint Model — T_C Only
    # ============================================================
    log_step "===== RUN 4: Joint T_C Only ====="

    # Check dropout_fields_iid setting
    DROPOUT_IID=$(grep "dropout_fields_iid" "$FINETUNE_JOINT/config.yaml" | head -1 | awk '{print $2}')
    if [ "$DROPOUT_IID" != "true" ]; then
        log_step "WARNING: dropout_fields_iid=$DROPOUT_IID (not true). T_C-only conditioning may be unreliable."
    fi

    TC_VALUES=("242" "335" "428" "521" "614")
    TC_LABELS=("median_242" "plus1sigma_335" "plus2sigma_428" "plus3sigma_521" "plus4sigma_614")

    for i in "${!TC_VALUES[@]}"; do
        VALUE="${TC_VALUES[$i]}"
        LABEL="${TC_LABELS[$i]}"

        run_generate_and_evaluate \
            "$RESULTS_ROOT/04_joint_TC_only/$LABEL" \
            "$FINETUNE_JOINT" \
            16 13 \
            "04_joint_TC_only/$LABEL" \
            --properties_to_condition_on="{'magnetocaloric_curie_temp': $VALUE}" \
            --diffusion_guidance_factor=2.0
    done
else
    log_step "SKIPPING Runs 3 and 4 — joint model not found at $FINETUNE_JOINT"
fi

# ============================================================
# RUN 5: Superconductor T_c conditioning
# ============================================================
if [ -d "$FINETUNE_SC" ] && [ -f "$FINETUNE_SC/config.yaml" ]; then
    log_step "===== RUN 5: Superconductor T_c ====="

    SC_VALUES=("3.8" "40.3" "76.9" "113.4" "149.9")
    SC_LABELS=("median_3.8" "plus1sigma_40.3" "plus2sigma_76.9" "plus3sigma_113.4" "plus4sigma_149.9")

    for i in "${!SC_VALUES[@]}"; do
        VALUE="${SC_VALUES[$i]}"
        LABEL="${SC_LABELS[$i]}"

        run_generate_and_evaluate \
            "$RESULTS_ROOT/05_superconductor/$LABEL" \
            "$FINETUNE_SC" \
            16 13 \
            "05_superconductor/$LABEL" \
            --properties_to_condition_on="{'sc_transition_temp': $VALUE}" \
            --diffusion_guidance_factor=2.0
    done
else
    log_step "SKIPPING Run 5 — superconductor model not found at $FINETUNE_SC"
fi

log_step "========== ALL GENERATION AND EVALUATION COMPLETE =========="
