#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generic multi-node Slurm template for ArtiFixer training.
#
# Uses model_training/slurm/lib/requeue_protocol.sh. The pre-timeout SIGUSR1
# handler requests a checkpoint with a unique token. The trainer acknowledges
# the completed checkpoint with that token before the launcher can requeue.
#
# Usage:
#   mkdir -p slurm-logs
#   SPLIT_PATH=/path/to/trainval_test_split.json \
#   DL3DV_DIR=/path/to/DL3DV-ALL-960P \
#   PROMPT_DIR=/path/to/artifixer-data/DL3DV-ALL-960P-captions \
#   RUN_ID=$(tr -dc a-z0-9 </dev/urandom | head -c 8) \
#   sbatch model_training/slurm/sample-slurm-submit.sh
#
# RUN_ID names the project directory and, when logging to W&B, pins the run
# lineage across requeues (resume=never on the first episode, resume=must on
# authorized requeues). Keep the generated value with your run records.
#
# Site-specific wrappers can source this script after setting paths and, optionally:
#   EXTRA_ENV              extra shell lines run on each node before launch
#   EXTRA_ACCELERATE_ARGS  extra `accelerate launch` flags
#   EXTRA_TRAIN_ARGS       extra model_training.train flags
#
# Set your cluster's account/partition with sbatch flags, or uncomment and edit:
##SBATCH --account=<account>
##SBATCH --partition=<partition>

#SBATCH --job-name=artifixer-train
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/artifixer-train-%j.out
#SBATCH --error=slurm-logs/artifixer-train-%j.err
#SBATCH --signal=B:SIGUSR1@600
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

: "${SPLIT_PATH:?Set SPLIT_PATH to trainval_test_split.json}"
: "${DL3DV_DIR:?Set DL3DV_DIR to the DL3DV archive root}"
: "${PROMPT_DIR:?Set PROMPT_DIR to the prompt HDF5 root}"
: "${RUN_ID:?Set RUN_ID to an 8-char [a-z0-9] run id (see usage header)}"

REPO_DIR="${REPO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
if [[ -z "${REPO_DIR}" ]]; then
    echo "Could not determine REPO_DIR; set REPO_DIR or submit from inside the repo." >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-artifixer-s1-14b}"
PROJECT_DIR="${PROJECT_DIR:-${REPO_DIR}/runs/${RUN_NAME}-${RUN_ID}}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
LOG_WITH="${LOG_WITH:-wandb}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
GPUS_PER_NODE="${GPUS_PER_NODE%%(*}"
NUM_MACHINES="${SLURM_NNODES:-1}"
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
TARGET_EFFECTIVE_GPUS="${TARGET_EFFECTIVE_GPUS:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(((TARGET_EFFECTIVE_GPUS + NUM_PROCESSES - 1) / NUM_PROCESSES))}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"

# shellcheck source=model_training/slurm/lib/requeue_protocol.sh
source "${REPO_DIR}/model_training/slurm/lib/requeue_protocol.sh"
rq_protocol_init
rq_scrub_leaked_shell_env
mkdir -p "$(dirname "${PROJECT_DIR}")"
# Fresh submissions take the atomic duplicate guard; authorized scheduler
# requeues validate the previous episode's requeue_authorization. The opt-in
# RESUME_EXISTING=1 branch (sbatch --export=ALL,RESUME_EXISTING=1 with the
# run's original RUN_ID) continues an existing project dir after a
# crash-without-handshake; it applies only to the first episode of a new
# sbatch — later requeues flow through the authorization machine even though
# the preserved submission environment still carries RESUME_EXISTING=1.
if [[ "${RESUME_EXISTING:-0}" == 1 && "$RESTART_COUNT" -eq 0 ]]; then
    rq_begin_episode_resume_existing
else
    rq_begin_episode
fi

export REPO_DIR RUN_ID RUN_NAME PROJECT_DIR SHOULD_CHECKPOINT_FLAG
export SPLIT_PATH DL3DV_DIR PROMPT_DIR MODEL_ID LOG_WITH GRADIENT_ACCUMULATION_STEPS
export NUM_MACHINES NUM_PROCESSES MASTER_ADDR MASTER_PORT WANDB_RESUME_MODE
# Site-wrapper hooks: evaluated/expanded inside the per-node launch shell.
export EXTRA_ENV="${EXTRA_ENV:-}"
export EXTRA_ACCELERATE_ARGS="${EXTRA_ACCELERATE_ARGS:-}"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"

read -r -d '' COMMAND <<'INNER' || true
set -euo pipefail
cd "${REPO_DIR}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Site-wrapper extra environment (runs before the HF cache guard so a
# wrapper-set HUGGINGFACE_HUB_CACHE is pinned by it).
eval "${EXTRA_ENV:-}"
if [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
    # Pin the highest-precedence HF cache variable so a leaked HF_HUB_CACHE or
    # HF_HOME cannot silently retarget the cache.
    unset HF_HOME
    export HUGGINGFACE_HUB_CACHE
    export HF_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
fi
# W&B identity: pin the run id + resume mode chosen by the episode gate
# (ignored when LOG_WITH != wandb).
export WANDB_RUN_ID="${RUN_ID}"
export WANDB_RESUME="${WANDB_RESUME_MODE}"
accelerate launch \
  --rdzv_backend c10d \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --multi_gpu \
  ${EXTRA_ACCELERATE_ARGS:-} \
  --num_machines "${NUM_MACHINES}" \
  --num_processes "${NUM_PROCESSES}" \
  --machine_rank "${SLURM_PROCID}" \
  --module model_training.train \
  --project_dir "${PROJECT_DIR}" \
  --split_path "${SPLIT_PATH}" \
  --dl3dv_dir "${DL3DV_DIR}" \
  --prompt_dir "${PROMPT_DIR}" \
  --model_id "${MODEL_ID}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --log_with "${LOG_WITH}" \
  --tracker_run_name "${RUN_NAME}" \
  --resume_from_checkpoint auto \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR:-2}" \
  --should_checkpoint_flag "${SHOULD_CHECKPOINT_FLAG}" \
  ${EXTRA_TRAIN_ARGS:-}
INNER

rq_install_signal_handlers

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm-logs"

srun_args=(
    --nodes="${NUM_MACHINES}"
    --ntasks="${NUM_MACHINES}"
    --ntasks-per-node=1
)

# Optional Pyxis/Enroot-style container support. Leave CONTAINER_IMAGE unset if
# your Slurm environment launches directly into an activated Python environment.
if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    srun_args+=(
        --container-image="${CONTAINER_IMAGE}"
        --container-mounts="${CONTAINER_MOUNTS:-${HOME}:${HOME}}"
        --container-workdir="${REPO_DIR}"
    )
fi

srun "${srun_args[@]}" bash --noprofile --norc -c "${COMMAND}" &
rq_wait_for_child $!
rq_requeue_or_exit
