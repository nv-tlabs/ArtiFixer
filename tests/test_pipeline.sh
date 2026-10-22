#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# End-to-end pipeline test: train.py -> diffusion_forcing.py -> distillation.py
# Each stage loads the checkpoint from the previous stage.
#
# Configurable via environment variables:
#   NUM_GPUS         - number of GPUs (default: 1). Full S1/S2/S3 validation
#                      expects at least FULL_PIPELINE_MIN_GPUS GPUs.
#   TEST_ROOT        - output directory for checkpoints/logs
#   REPO_DIR         - path to the artifixer repo
#   DL3DV_DIR        - path to DL3DV dataset
#   SPLIT_PATH       - path to trainval_test_split.json
#   PROMPT_DIR       - path to prompt HDF5 root
#   HUGGINGFACE_HUB_CACHE - optional HF model cache
#   DATA_ARGS        - extra data-loader args
#   EXTRA_ARGS       - extra training args appended to every stage
#   MODEL_ID         - Diffusers model id to smoke test
#   RUN_SUFFIX       - suffix for wandb run names
#   NUM_FRAMES       - raw Wan frames for S1/S2 training. Defaults to 81 for
#                      S1-only smoke, or 125 when stages 2/3 are enabled.
#   FRAMES_PER_BLOCK - latent frames per AR block (default: 8)
#   LOCAL_ATTN_SIZE - Stage 3 KV-cache local attention size (default: 32)
#   SINK_SIZE       - Stage 3 KV-cache sink size (default: 8)
#   MAX_VALIDATION_TASKS - max validation tasks per validation step (default: 1)
#   CONTEXT_PARALLEL_SIZE - CP shard size for S3 and the S1 default (default: 8).
#                      The short stage-2 smoke uses CP=1.
#   SHARD_SIZE     - FSDP shard size for S2/S3. Defaults to NUM_GPUS when
#                      CONTEXT_PARALLEL_SIZE=1, otherwise CONTEXT_PARALLEL_SIZE.
#   S1_CONTEXT_PARALLEL_SIZE - CP shard size for S1. Defaults to CONTEXT_PARALLEL_SIZE
#                      when NUM_GPUS is large enough, otherwise 1 for S1-only smoke.
#   S1_SHARD_SIZE  - FSDP shard size for S1. Defaults to NUM_GPUS when
#                      S1_CONTEXT_PARALLEL_SIZE=1, otherwise S1_CONTEXT_PARALLEL_SIZE.
#   PIPELINE_STAGES  - comma-separated stages to run: 1, 2, 3. Defaults to
#                      1 on a single GPU, and 1,2,3 on multi-GPU runs.
#   FULL_PIPELINE_MIN_GPUS - minimum GPU count required for the default full
#                      3-stage smoke (default: 8)
#   ALLOW_SLOW_FULL_PIPELINE - set to 1 to bypass the minimum-GPU guardrail
set -euo pipefail

export PATH=/usr/local/bin:/usr/bin:/bin
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true
export HOME="${HOME:-/root}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${NUM_GPUS:=1}"
if [ -z "${REPO_DIR:-}" ]; then
    REPO_DIR="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || true)"
fi
: "${REPO_DIR:?Set REPO_DIR to the artifixer repo root}"
: "${DL3DV_DIR:?Set DL3DV_DIR to the DL3DV archive root}"
: "${SPLIT_PATH:?Set SPLIT_PATH to trainval_test_split.json}"
: "${PROMPT_DIR:?Set PROMPT_DIR to the prompt HDF5 root}"
: "${DATA_ARGS:=}"
: "${EXTRA_ARGS:=}"
: "${TEST_ROOT:=/tmp/artifixer-test-pipeline}"
: "${RUN_SUFFIX:=}"
if [ -n "${HUGGINGFACE_HUB_CACHE:-}" ]; then
    export HUGGINGFACE_HUB_CACHE
fi

: "${NUM_MACHINES:=1}"
: "${MASTER_ADDR:=localhost}"
: "${MASTER_PORT:=29501}"
: "${MACHINE_RANK:=${SLURM_PROCID:-0}}"
: "${FULL_PIPELINE_MIN_GPUS:=8}"
: "${ALLOW_SLOW_FULL_PIPELINE:=0}"

if [ -z "${PIPELINE_STAGES:-}" ]; then
    if [ "$NUM_GPUS" -gt 1 ]; then
        PIPELINE_STAGES="1,2,3"
    else
        PIPELINE_STAGES="1"
    fi
fi

stage_enabled() {
    case ",${PIPELINE_STAGES}," in
        *,"$1",*) return 0 ;;
        *) return 1 ;;
    esac
}

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
if [ -z "${NUM_FRAMES:-}" ]; then
    if stage_enabled 2 || stage_enabled 3; then
        NUM_FRAMES=125
    else
        NUM_FRAMES=81
    fi
fi
FRAMES_PER_BLOCK="${FRAMES_PER_BLOCK:-8}"
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-32}"
SINK_SIZE="${SINK_SIZE:-8}"
MAX_VALIDATION_TASKS="${MAX_VALIDATION_TASKS:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-8}"
if [ -z "${S1_CONTEXT_PARALLEL_SIZE:-}" ]; then
    if [ "$NUM_GPUS" -ge "$CONTEXT_PARALLEL_SIZE" ]; then
        S1_CONTEXT_PARALLEL_SIZE="$CONTEXT_PARALLEL_SIZE"
    else
        S1_CONTEXT_PARALLEL_SIZE=1
    fi
fi
if [ -z "${SHARD_SIZE:-}" ]; then
    if [ "$CONTEXT_PARALLEL_SIZE" -gt 1 ]; then
        SHARD_SIZE="$CONTEXT_PARALLEL_SIZE"
    else
        SHARD_SIZE="$NUM_GPUS"
    fi
fi
if [ -z "${S1_SHARD_SIZE:-}" ]; then
    if [ "$S1_CONTEXT_PARALLEL_SIZE" -gt 1 ]; then
        S1_SHARD_SIZE="$S1_CONTEXT_PARALLEL_SIZE"
    else
        S1_SHARD_SIZE="$NUM_GPUS"
    fi
fi

require_dir() {
    local path="$1"
    local description="$2"
    if [ ! -d "$path" ]; then
        echo "ERROR: Missing ${description}: $path"
        exit 1
    fi
}

require_file() {
    local path="$1"
    local description="$2"
    if [ ! -f "$path" ]; then
        echo "ERROR: Missing ${description}: $path"
        exit 1
    fi
}

require_nonempty_glob() {
    local pattern="$1"
    local description="$2"
    local match
    match=$(compgen -G "$pattern" | head -1 || true)
    if [ -z "$match" ]; then
        echo "ERROR: Missing ${description}: $pattern"
        exit 1
    fi
}

require_dir "$DL3DV_DIR" "DL3DV archive root"
require_dir "$PROMPT_DIR" "prompt HDF5 root"
require_file "$SPLIT_PATH" "train/validation split JSON"

run_stage() {
    local stage_id="$1"
    local stage_name="$2"
    local project_dir="$3"
    local checkpoint_var="$4"
    shift 4

    echo ""
    echo "================================================================"
    echo "STAGE ${stage_id}: ${stage_name} [${NUM_GPUS} GPUs]${RUN_SUFFIX:+ [$RUN_SUFFIX]}"
    echo "================================================================"
    local rc=0
    "$@" || rc=$?
    echo "STAGE ${stage_id} EXIT CODE: ${rc}"
    if [ "${rc}" -ne 0 ]; then
        exit "${rc}"
    fi

    require_dir "${project_dir}/checkpoints" "stage ${stage_id} checkpoints directory"
    require_nonempty_glob "${project_dir}/checkpoints/checkpoint_*/pytorch_model_fsdp_0" "stage ${stage_id} checkpoint shard"
    require_nonempty_glob "${project_dir}/wandb/wandb/*run-*/logs" "stage ${stage_id} wandb logs"

    local checkpoint_path
    checkpoint_path=$(ls -d "${project_dir}"/checkpoints/checkpoint_*/pytorch_model_fsdp_0 2>/dev/null | tail -1)
    local checkpoint_dir
    checkpoint_dir=$(dirname "${checkpoint_path}")
    if [ ! -f "${checkpoint_dir}/checkpoint_complete" ]; then
        echo "ERROR: Missing stage ${stage_id} checkpoint completion marker: ${checkpoint_dir}/checkpoint_complete"
        exit 1
    fi
    printf -v "${checkpoint_var}" '%s' "${checkpoint_path}"
    echo "Stage ${stage_id} checkpoint: ${checkpoint_path}"
}

if { stage_enabled 2 || stage_enabled 3; } && [ "$NUM_GPUS" -lt "$FULL_PIPELINE_MIN_GPUS" ] && [ "$ALLOW_SLOW_FULL_PIPELINE" != "1" ]; then
    echo "ERROR: Refusing to run stages ${PIPELINE_STAGES} with NUM_GPUS=$NUM_GPUS."
    echo "The full 3-stage pipeline smoke is not a credible validation path below ${FULL_PIPELINE_MIN_GPUS} GPUs."
    echo "Use PIPELINE_STAGES=1 for a single-GPU smoke, raise NUM_GPUS, or set ALLOW_SLOW_FULL_PIPELINE=1 to override."
    exit 1
fi
if stage_enabled 1 && [ "$NUM_GPUS" -lt "$S1_CONTEXT_PARALLEL_SIZE" ] && [ "$ALLOW_SLOW_FULL_PIPELINE" != "1" ]; then
    echo "ERROR: Refusing to run Stage 1 CP smoke with NUM_GPUS=$NUM_GPUS and S1_CONTEXT_PARALLEL_SIZE=$S1_CONTEXT_PARALLEL_SIZE."
    echo "Raise NUM_GPUS, lower S1_CONTEXT_PARALLEL_SIZE explicitly, or set ALLOW_SLOW_FULL_PIPELINE=1 to override."
    exit 1
fi
if stage_enabled 1 && [ $((NUM_GPUS % S1_CONTEXT_PARALLEL_SIZE)) -ne 0 ] && [ "$ALLOW_SLOW_FULL_PIPELINE" != "1" ]; then
    echo "ERROR: Refusing to run Stage 1 CP smoke with non-divisible NUM_GPUS=$NUM_GPUS and S1_CONTEXT_PARALLEL_SIZE=$S1_CONTEXT_PARALLEL_SIZE."
    echo "Use a GPU count divisible by S1_CONTEXT_PARALLEL_SIZE, lower S1_CONTEXT_PARALLEL_SIZE explicitly, or set ALLOW_SLOW_FULL_PIPELINE=1 to override."
    exit 1
fi
if [ "$NUM_GPUS" -gt 1 ] && [ $((NUM_GPUS % SHARD_SIZE)) -ne 0 ]; then
    echo "ERROR: Refusing to run with NUM_GPUS=$NUM_GPUS and SHARD_SIZE=$SHARD_SIZE."
    echo "Use an FSDP shard size that divides NUM_GPUS."
    exit 1
fi
if [ "$NUM_GPUS" -gt 1 ] && [ $((NUM_GPUS % S1_SHARD_SIZE)) -ne 0 ]; then
    echo "ERROR: Refusing to run with NUM_GPUS=$NUM_GPUS and S1_SHARD_SIZE=$S1_SHARD_SIZE."
    echo "Use a Stage 1 FSDP shard size that divides NUM_GPUS."
    exit 1
fi
if [ "$CONTEXT_PARALLEL_SIZE" -gt 1 ] && [ "$SHARD_SIZE" -ne "$CONTEXT_PARALLEL_SIZE" ]; then
    echo "ERROR: SHARD_SIZE must equal CONTEXT_PARALLEL_SIZE when context parallelism is enabled."
    exit 1
fi
if [ "$S1_CONTEXT_PARALLEL_SIZE" -gt 1 ] && [ "$S1_SHARD_SIZE" -ne "$S1_CONTEXT_PARALLEL_SIZE" ]; then
    echo "ERROR: S1_SHARD_SIZE must equal S1_CONTEXT_PARALLEL_SIZE when Stage 1 context parallelism is enabled."
    exit 1
fi

echo "================================================================"
echo "LAUNCH CONFIG ($(hostname) @ $(date))"
echo "  NUM_GPUS=$NUM_GPUS"
echo "  NUM_MACHINES=$NUM_MACHINES"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  MACHINE_RANK=$MACHINE_RANK"
echo "  SLURM_PROCID=${SLURM_PROCID:-UNSET}"
echo "  SLURM_NODEID=${SLURM_NODEID:-UNSET}"
echo "  SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-UNSET}"
echo "  REPO_DIR=$REPO_DIR"
echo "  DL3DV_DIR=$DL3DV_DIR"
echo "  SPLIT_PATH=$SPLIT_PATH"
echo "  PROMPT_DIR=$PROMPT_DIR"
echo "  TEST_ROOT=$TEST_ROOT"
echo "  HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-UNSET}"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "  NUM_FRAMES=$NUM_FRAMES"
echo "  FRAMES_PER_BLOCK=$FRAMES_PER_BLOCK"
echo "  LOCAL_ATTN_SIZE=$LOCAL_ATTN_SIZE"
echo "  SINK_SIZE=$SINK_SIZE"
echo "  MAX_VALIDATION_TASKS=$MAX_VALIDATION_TASKS"
echo "  CONTEXT_PARALLEL_SIZE=$CONTEXT_PARALLEL_SIZE"
echo "  SHARD_SIZE=$SHARD_SIZE"
echo "  S1_CONTEXT_PARALLEL_SIZE=$S1_CONTEXT_PARALLEL_SIZE"
echo "  S1_SHARD_SIZE=$S1_SHARD_SIZE"
echo "  DATA_ARGS=$DATA_ARGS"
echo "  EXTRA_ARGS=$EXTRA_ARGS"
echo "  PIPELINE_STAGES=$PIPELINE_STAGES"
echo "  FULL_PIPELINE_MIN_GPUS=$FULL_PIPELINE_MIN_GPUS"
echo "  ALLOW_SLOW_FULL_PIPELINE=$ALLOW_SLOW_FULL_PIPELINE"
echo "  NCCL_DEBUG=${NCCL_DEBUG:-UNSET}"
echo "================================================================"

cd "${REPO_DIR}"

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}, cap: {torch.cuda.get_device_capability()}')"
python -c "import torch; from model_training.net.attention_config import configure_attention; device = torch.device('cuda:0'); torch.cuda.set_device(device); backend = configure_attention(device); print(f'Default attention backend: {backend}')"

VALIDATION_ARGS=""
if [ -n "$MAX_VALIDATION_TASKS" ]; then
    VALIDATION_ARGS="--max_validation_tasks $MAX_VALIDATION_TASKS"
fi
COMMON="--model_id $MODEL_ID --num_frames $NUM_FRAMES --dataloader_num_workers 0 --split_path $SPLIT_PATH --dl3dv_dir $DL3DV_DIR --prompt_dir $PROMPT_DIR $VALIDATION_ARGS $DATA_ARGS $EXTRA_ARGS"
S1_CP_ARGS=""
if [ "$S1_CONTEXT_PARALLEL_SIZE" -gt 1 ]; then
    S1_CP_ARGS="--context_parallel_size $S1_CONTEXT_PARALLEL_SIZE --shard_size $S1_SHARD_SIZE"
elif [ "$S1_SHARD_SIZE" -gt 1 ]; then
    S1_CP_ARGS="--shard_size $S1_SHARD_SIZE"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH="accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes $NUM_GPUS --rdzv_backend c10d --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --num_machines $NUM_MACHINES --machine_rank $MACHINE_RANK --module"
else
    LAUNCH="python -m"
fi
echo "LAUNCH=$LAUNCH"

python tests/validate_pipeline_prompts.py \
    --split-path "$SPLIT_PATH" \
    --prompt-dir "$PROMPT_DIR" \
    --extra-args "$DATA_ARGS $EXTRA_ARGS"

rm -rf "$TEST_ROOT"

S1_CKPT=""
S2_CKPT=""

# =========================================================================
# STAGE 1: train.py
#   - 4 iters, validate at steps 2 and 4 (2 rounds of validation)
#   - save checkpoint at step 2
# =========================================================================
if stage_enabled 1; then
    run_stage 1 "train.py (4 iters, validate@2,4, save@2)" "$TEST_ROOT/s1" S1_CKPT \
        $LAUNCH model_training.train \
        --project_dir $TEST_ROOT/s1 \
        --max_iterations 4 --save_steps 2 --validation_steps 2 \
        $S1_CP_ARGS \
        --log_with wandb --tracker_run_name "test-s1${RUN_SUFFIX}" --tracker_project_name artifixer \
        $COMMON
fi

# =========================================================================
# STAGE 2: diffusion_forcing.py (CP=1; diffusion forcing uses the plain trainer)
#   - Same pattern: 4 iters, 2 validations, checkpoint in between
# =========================================================================
if stage_enabled 2; then
    [ -z "$S1_CKPT" ] && echo "ERROR: Stage 2 requested but no stage 1 checkpoint is available" && exit 1

    run_stage 2 "diffusion_forcing.py (4 iters)" "$TEST_ROOT/s2" S2_CKPT \
        $LAUNCH model_training.diffusion_forcing \
        --project_dir $TEST_ROOT/s2 \
        --base_checkpoint_dir $S1_CKPT \
        --frames_per_block $FRAMES_PER_BLOCK \
        --context_parallel_size 1 --shard_size $SHARD_SIZE \
        --max_iterations 4 --save_steps 2 --validation_steps 2 \
        --log_with wandb --tracker_run_name "test-s2${RUN_SUFFIX}" --tracker_project_name artifixer \
        $COMMON
fi

# =========================================================================
# STAGE 3: distillation.py
#   - Same pattern
# =========================================================================
if stage_enabled 3; then
    [ -z "$S2_CKPT" ] && echo "ERROR: Stage 3 requested but no stage 2 checkpoint is available" && exit 1

    run_stage 3 "distillation.py (4 iters)" "$TEST_ROOT/s3" _unused_stage3_ckpt \
        $LAUNCH model_training.distillation \
        --project_dir $TEST_ROOT/s3 \
        --base_checkpoint_dir $S2_CKPT \
        --base_checkpoint_dir_critic $S2_CKPT \
        --model_id_critic $MODEL_ID \
        --frames_per_block $FRAMES_PER_BLOCK --local_attn_size $LOCAL_ATTN_SIZE --sink_size $SINK_SIZE \
        --context_parallel_size $CONTEXT_PARALLEL_SIZE --shard_size $SHARD_SIZE \
        --max_iterations 4 --save_steps 2 --validation_steps 2 \
        --ema_weight 0 \
        --log_with wandb --tracker_run_name "test-s3${RUN_SUFFIX}" --tracker_project_name artifixer \
        $COMMON
fi

echo ""
echo "================================================================"
echo "REQUESTED STAGES COMPLETE ($PIPELINE_STAGES) [${NUM_GPUS} GPUs]${RUN_SUFFIX:+ [$RUN_SUFFIX]}"
echo "================================================================"
