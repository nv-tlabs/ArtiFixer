# shellcheck shell=bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The single in-repo source of the launcher-side checkpoint/requeue protocol.
#
# Protocol state machine:
#   episode 0: provenance gates -> atomic-mkdir duplicate guard -> flock on
#     $PROJECT_DIR/.training.lock (held all episode) -> atomic run_id.txt ->
#     rm stale flag/ack.
#   preemption: SIGUSR1 trap writes a ns-unique request token into the flag;
#     the trainer saves a commit-marked checkpoint and writes a token-bound
#     atomic ack; the launcher validates the ack + markers + no TERM veto,
#     writes requeue_authorization {restart_count, request, step} atomically,
#     and only then calls `scontrol requeue`.
#   episode N>0: requires the authorization file to match SLURM_RESTART_COUNT
#     and cross-match the ack; anything else (e.g. a crash that never received
#     a USR1 checkpoint handshake, which the scheduler would otherwise requeue
#     silently) exits 3 as an unauthorized scheduler requeue.
#   RESUME_EXISTING=1 (opt-in, episode 0 only): the human sbatch acts as the
#     authorization for continuing an existing PROJECT_DIR; see
#     rq_begin_episode_resume_existing.
#
# Usage (batch script, under `set -euo pipefail`):
#   source "$REPO_DIR/model_training/slurm/lib/requeue_protocol.sh"
#   rq_protocol_init            # requires PROJECT_DIR, RUN_ID, SLURM_JOB_ID
#   rq_begin_episode            # or rq_begin_episode_resume_existing
#   rq_install_signal_handlers  # before launching the payload
#   srun ... &
#   rq_wait_for_child $!
#   rq_requeue_or_exit          # never returns
#
# The script the templates run inside the container may source this file
# too, for the rq_gate_* helpers.
#
# Exit codes:
#   1   provenance gate failure
#   2   missing required path
#   3   episode-begin refusal (duplicate project, unauthorized requeue, veto,
#       lock contention, run-id mismatch)
#   125 post-preemption checkpoint ack missing/invalid, or requeue vetoed
#   126 scontrol requeue itself failed with the job not preempted (a
#       preempted job defers to the controller's own requeue and exits 0)

# ---------------------------------------------------------------------------
# Small helpers

rq_fail() {
    local code=$1
    shift
    printf '%s\n' "$*" >&2
    exit "$code"
}

rq_require_paths() {
    local required_path
    for required_path in "$@"; do
        [[ -e "$required_path" ]] || rq_fail 2 "Missing required path: $required_path"
    done
}

# ---------------------------------------------------------------------------
# Provenance gate helpers (shared by the outer batch script and the
# in-container script, which re-checks the pins from inside the container).

# rq_scrub_leaked_shell_env
# Remove inherited Conda variables and shell hooks before srun. Paths from
# the submission shell may not exist inside the container, and BASH_ENV would
# run in every non-interactive shell the job starts.
rq_scrub_leaked_shell_env() {
    unset BASH_ENV ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE \
        CONDA_SHLVL _CE_CONDA _CE_M 2>/dev/null || true
    unset -f conda __conda_activate __conda_hashr __conda_exe __add_sys_prefix_to_path 2>/dev/null || true
    return 0
}

# rq_git [git args...]
# Compare files without LFS filters because the training images lack git-lfs.
# Checkouts prepared with GIT_LFS_SKIP_SMUDGE=1 retain pointer files, so the
# comparison still detects modified or materialized files.
rq_git() {
    git -c filter.lfs.process= -c filter.lfs.clean= -c filter.lfs.smudge= \
        -c filter.lfs.required=false "$@"
}

# rq_gate_repo_clean_at_sha <repo_dir> <expected_sha> <label>
rq_gate_repo_clean_at_sha() {
    local repo_dir=$1 expected_sha=$2 label=$3
    local actual_sha status
    actual_sha=$(rq_git -C "$repo_dir" rev-parse HEAD)
    [[ "$actual_sha" == "$expected_sha" ]] \
        || rq_fail 1 "TRAIN_GATE_ERROR=${label}_sha:$actual_sha (expected $expected_sha)"
    status=$(rq_git -C "$repo_dir" status --porcelain --untracked-files=normal)
    [[ -z "$status" ]] || {
        printf 'TRAIN_GATE_%s_DIRTY_BEGIN\n%s\nTRAIN_GATE_%s_DIRTY_END\n' "$label" "$status" "$label" >&2
        exit 1
    }
}

# rq_gate_model_refs_byte_exact <model_cache_dir> <revision> <expected_symlink_count>
# cmp, not a string compare: command substitution strips a trailing newline,
# while huggingface_hub reads refs/main verbatim, so a newline-terminated
# refs file (which breaks offline loading) would falsely pass `[[ == ]]`. Also probes
# snapshot completeness: the transformer weights index must exist, no
# snapshot symlink may dangle, and the symlink count must match the
# revision-specific count the caller supplies.
rq_gate_model_refs_byte_exact() {
    local model_cache=$1 revision=$2 expected_symlink_count=$3
    local snapshot="$model_cache/snapshots/$revision"
    [[ -f "$model_cache/refs/main" ]] || rq_fail 1 "TRAIN_GATE_ERROR=model_refs_missing:$model_cache/refs/main"
    cmp -s <(printf '%s' "$revision") "$model_cache/refs/main" \
        || rq_fail 1 "TRAIN_GATE_ERROR=model_refs_not_byte_exact:$model_cache/refs/main (expected exactly '$revision' with no trailing newline)"
    # Probe the exact offline artifacts the trainer resolves.
    [[ -f "$snapshot/transformer/config.json" ]] \
        || rq_fail 1 "TRAIN_GATE_ERROR=model_snapshot_config_missing:$snapshot/transformer/config.json"
    [[ -f "$snapshot/transformer/diffusion_pytorch_model.safetensors.index.json" ]] \
        || rq_fail 1 "TRAIN_GATE_ERROR=model_snapshot_weights_index_missing:$snapshot/transformer/diffusion_pytorch_model.safetensors.index.json"
    local dangling_symlink
    dangling_symlink=$(find -L "$snapshot" -type l -print -quit)
    [[ -z "$dangling_symlink" ]] \
        || rq_fail 1 "TRAIN_GATE_ERROR=model_snapshot_dangling_symlink:$dangling_symlink"
    local symlink_count
    symlink_count=$(find "$snapshot" -type l | wc -l)
    [[ "$symlink_count" -eq "$expected_symlink_count" ]] \
        || rq_fail 1 "TRAIN_GATE_ERROR=model_snapshot_symlink_count:$symlink_count (expected $expected_symlink_count)"
}

# rq_gate_file_sha256 <path> <expected_sha256> <label>
rq_gate_file_sha256() {
    local path=$1 expected=$2 label=$3
    local actual
    actual=$(sha256sum "$path" | cut -d' ' -f1)
    [[ "$actual" == "$expected" ]] || rq_fail 1 "TRAIN_GATE_ERROR=${label}_sha256:$actual (expected $expected)"
}

# rq_gate_marker_line <marker_path> <exact_line>
rq_gate_marker_line() {
    local marker=$1 line=$2
    grep -Fxq -- "$line" "$marker" || rq_fail 1 "TRAIN_GATE_ERROR=marker_line_missing:'$line' in $marker"
}

# rq_emit_preflight_marker <marker_path> <key=value>...
# Writes the PREFLIGHT_PASS marker atomically (tmp + mv) with `result=PASS`
# as the LAST line, so a partially-written marker never carries the verdict.
# `result=` is reserved: callers supply only the key=value lines recording
# what was validated.
rq_emit_preflight_marker() {
    local marker_path=$1
    shift
    local tmp_path="$marker_path.tmp.$$"
    local line
    [[ $# -ge 1 ]] || rq_fail 1 "PREFLIGHT_MARKER_ERROR=no_key_value_lines"
    for line in "$@"; do
        [[ "$line" == *=* ]] || rq_fail 1 "PREFLIGHT_MARKER_ERROR=not_key_value:'$line'"
        [[ "$line" != result=* ]] || rq_fail 1 "PREFLIGHT_MARKER_ERROR=result_is_reserved:'$line'"
    done
    { printf '%s\n' "$@"; printf 'result=PASS\n'; } > "$tmp_path"
    mv "$tmp_path" "$marker_path"
}

# rq_gate_run_id_matches — run_id.txt must exist and equal RUN_ID.
rq_gate_run_id_matches() {
    [[ -f "$PROJECT_DIR/run_id.txt" ]] || rq_fail 3 "Missing $PROJECT_DIR/run_id.txt"
    [[ "$(<"$PROJECT_DIR/run_id.txt")" == "$RUN_ID" ]] \
        || rq_fail 3 "run_id.txt mismatch: $(<"$PROJECT_DIR/run_id.txt") != $RUN_ID"
}

# ---------------------------------------------------------------------------
# Protocol state

# Requires PROJECT_DIR, RUN_ID and SLURM_JOB_ID; derives the protocol paths
# and initializes the per-episode state the signal handlers and the requeue
# logic share.
rq_protocol_init() {
    : "${PROJECT_DIR:?rq_protocol_init requires PROJECT_DIR}"
    : "${RUN_ID:?rq_protocol_init requires RUN_ID}"
    : "${SLURM_JOB_ID:?rq_protocol_init requires SLURM_JOB_ID}"
    [[ "$RUN_ID" =~ ^[a-z0-9]{8}$ ]] || rq_fail 1 "RUN_ID must be an 8-char [a-z0-9] W&B run id, got: $RUN_ID"

    SHOULD_CHECKPOINT_FLAG="$PROJECT_DIR/should_checkpoint"
    CHECKPOINT_ACK_PATH="$SHOULD_CHECKPOINT_FLAG.complete"
    REQUEUE_AUTH_PATH="$PROJECT_DIR/requeue_authorization"
    TERMINATION_VETO_PATH="$PROJECT_DIR/termination_veto"
    RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
    # shellcheck disable=SC2034  # episode-state reset; read only by this lib's rq_wait_for_child, not by any batch script
    child=
    child_status=1
    preempt_signal=0
    termination_signal=0
    checkpoint_request_token=
}

# Take the single-writer lock for this run. Held (fd 9) for the whole episode.
rq_acquire_project_lock() {
    exec 9>"$PROJECT_DIR/.training.lock"
    flock -n 9 || rq_fail 3 "Another writer holds $PROJECT_DIR/.training.lock"
}

# Highest checkpoint step under $PROJECT_DIR/checkpoints whose directory
# contains the checkpoint_complete commit marker — exactly the trainer's
# auto-resume rule (train_utils), NOT the max step overall.
rq_latest_complete_checkpoint_step() {
    local best=-1 dir name step
    for dir in "$PROJECT_DIR"/checkpoints/checkpoint_*/; do
        [[ -d "$dir" ]] || continue
        name=${dir%/}
        step=${name##*checkpoint_}
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        [[ -f "$dir/checkpoint_complete" ]] || continue
        # Force base-10: a zero-padded name like checkpoint_010 would
        # otherwise be parsed as octal (or error on digits 8/9).
        step=$((10#$step))
        if (( step > best )); then
            best=$step
        fi
    done
    if (( best < 0 )); then
        return 1
    fi
    printf '%s\n' "$best"
}

# ---------------------------------------------------------------------------
# Episode begin

# Fresh submissions create PROJECT_DIR atomically (duplicate guard); requeued
# episodes are admitted only with a matching requeue authorization written by
# the previous episode after a validated checkpoint ack.
rq_begin_episode() {
    if [[ "$RESTART_COUNT" -eq 0 ]]; then
        mkdir "$PROJECT_DIR" || rq_fail 3 "Fresh project already exists: $PROJECT_DIR"
        WANDB_RESUME_MODE=never
    else
        [[ -d "$PROJECT_DIR" ]] || rq_fail 3 "Requeued project is missing: $PROJECT_DIR"
        [[ ! -e "$TERMINATION_VETO_PATH" ]] || rq_fail 3 "Requeue vetoed by TERM"
        WANDB_RESUME_MODE=must
    fi
    rq_acquire_project_lock
    if [[ "$RESTART_COUNT" -eq 0 ]]; then
        [[ ! -e "$PROJECT_DIR/run_id.txt" ]] || rq_fail 3 "Fresh project already has a run_id.txt: $PROJECT_DIR"
        printf '%s\n' "$RUN_ID" > "$PROJECT_DIR/.run_id.txt.$SLURM_JOB_ID.tmp"
        mv "$PROJECT_DIR/.run_id.txt.$SLURM_JOB_ID.tmp" "$PROJECT_DIR/run_id.txt"
    else
        rq_validate_requeue_authorization
    fi
    rm -f "$SHOULD_CHECKPOINT_FLAG" "$CHECKPOINT_ACK_PATH"
}

rq_validate_requeue_authorization() {
    local authorized_restart authorized_request authorized_step checkpoint_dir
    [[ -f "$REQUEUE_AUTH_PATH" ]] \
        || rq_fail 3 "Unauthorized scheduler requeue: $REQUEUE_AUTH_PATH is missing"
    authorized_restart=$(sed -n 's/^restart_count=//p' "$REQUEUE_AUTH_PATH")
    authorized_request=$(sed -n 's/^request=//p' "$REQUEUE_AUTH_PATH")
    authorized_step=$(sed -n 's/^step=//p' "$REQUEUE_AUTH_PATH")
    [[ "$authorized_restart" == "$RESTART_COUNT" ]] \
        || rq_fail 3 "Requeue authorization restart_count=$authorized_restart != SLURM_RESTART_COUNT=$RESTART_COUNT"
    [[ -n "$authorized_request" ]] || rq_fail 3 "Requeue authorization carries no request token"
    [[ "$authorized_step" =~ ^[0-9]+$ ]] || rq_fail 3 "Requeue authorization step is not numeric: $authorized_step"
    checkpoint_dir="$PROJECT_DIR/checkpoints/checkpoint_$authorized_step"
    [[ -f "$checkpoint_dir/checkpoint_complete" ]] \
        || rq_fail 3 "Authorized checkpoint is incomplete: $checkpoint_dir/checkpoint_complete is missing"
    [[ -f "$checkpoint_dir/pytorch_model_fsdp_0/.metadata" ]] \
        || rq_fail 3 "Authorized checkpoint is missing DCP metadata: $checkpoint_dir/pytorch_model_fsdp_0/.metadata"
    grep -Fxq -- "request=$authorized_request" "$CHECKPOINT_ACK_PATH" \
        || rq_fail 3 "Checkpoint ack does not match the authorized request token"
    grep -Fxq -- "step=$authorized_step" "$CHECKPOINT_ACK_PATH" \
        || rq_fail 3 "Checkpoint ack does not match the authorized step"
    rq_gate_run_id_matches
    rm -f "$REQUEUE_AUTH_PATH"
}

# RESUME_EXISTING=1 continues an existing PROJECT_DIR from a fresh submission,
# including after a crash or cluster transfer. Only episode 0 can use it;
# later requeues require the normal checkpoint authorization. The new
# submission authorizes the resume, and run_id.txt stays unchanged.
# Before submitting, verify that no job for this run is pending or running:
# flock cannot detect a queued requeue. If the old job was cancelled, remove
# its termination veto only when deliberately resuming that run.
rq_begin_episode_resume_existing() {
    [[ "$RESTART_COUNT" -eq 0 ]] \
        || rq_fail 3 "RESUME_EXISTING is only valid on a fresh submission (got SLURM_RESTART_COUNT=$RESTART_COUNT)"
    [[ -d "$PROJECT_DIR" ]] || rq_fail 3 "RESUME_EXISTING project is missing: $PROJECT_DIR"
    [[ ! -e "$TERMINATION_VETO_PATH" ]] \
        || rq_fail 3 "Resume vetoed by TERM: remove $TERMINATION_VETO_PATH deliberately iff the old job was scancel'd"
    # shellcheck disable=SC2034  # consumed by the sourcing batch script's W&B identity block
    WANDB_RESUME_MODE=must
    rq_acquire_project_lock
    [[ ! -e "$TERMINATION_VETO_PATH" ]] || rq_fail 3 "Resume vetoed by TERM"
    rq_gate_run_id_matches
    local resume_step
    resume_step=$(rq_latest_complete_checkpoint_step) \
        || rq_fail 3 "No complete checkpoint (checkpoint_complete marker) under $PROJECT_DIR/checkpoints"
    [[ -f "$PROJECT_DIR/checkpoints/checkpoint_$resume_step/pytorch_model_fsdp_0/.metadata" ]] \
        || rq_fail 3 "Latest complete checkpoint_$resume_step is missing pytorch_model_fsdp_0/.metadata"
    echo "RESUME_EXISTING_CHECKPOINT_STEP=$resume_step"
    # A stale authorization's restart_count could collide with the new
    # lineage's first legitimate requeue; clear it together with the stale
    # request flag and ack, under the lock.
    rm -f "$REQUEUE_AUTH_PATH" "$SHOULD_CHECKPOINT_FLAG" "$CHECKPOINT_ACK_PATH"
}

# ---------------------------------------------------------------------------
# Signals

# Atomically publish a ns-unique checkpoint request token. The trainer copies
# the token into its ack, which rq_requeue_or_exit later cross-checks, binding
# the eventual requeue to exactly this request.
rq_write_checkpoint_request() {
    local kind=$1
    local tmp_path="$PROJECT_DIR/.should_checkpoint.$SLURM_JOB_ID.tmp"
    checkpoint_request_token="${kind}:${SLURM_JOB_ID}:${RESTART_COUNT}:$(date -u +%s%N)"
    rm -f "$CHECKPOINT_ACK_PATH"
    printf '%s\n' "$checkpoint_request_token" > "$tmp_path"
    mv "$tmp_path" "$SHOULD_CHECKPOINT_FLAG"
}

# SIGTERM = deliberate cancellation: veto any future requeue (the veto file is
# never auto-removed), still ask for a final checkpoint if none is pending.
rq_term_handler() {
    termination_signal=1
    printf 'job_id=%s\nrestart_count=%s\n' "$SLURM_JOB_ID" "$RESTART_COUNT" > "$TERMINATION_VETO_PATH"
    if [[ ! -f "$SHOULD_CHECKPOINT_FLAG" ]]; then
        rq_write_checkpoint_request term
    fi
}

# SIGUSR1 requests one checkpoint before the job exits. Slurm sends it for
# the configured pre-timeout warning and can also send it on preemption with
# PreemptParameters=send_user_signal and PreemptMode=REQUEUE. Configure the
# warning and partition grace period to allow the checkpoint to finish.
rq_preempt_handler() {
    if [[ "$preempt_signal" -eq 0 ]]; then
        preempt_signal=1
        rq_write_checkpoint_request usr1
    fi
}

rq_install_signal_handlers() {
    trap rq_preempt_handler SIGUSR1
    trap rq_term_handler SIGTERM
}

# ---------------------------------------------------------------------------
# Child supervision + requeue decision

# Wait for the payload while remaining interruptible: `wait` returns >128 when
# a trapped signal arrives, so keep waiting as long as the child is alive.
rq_wait_for_child() {
    child=$1
    set +e
    while true; do
        wait "$child"
        child_status=$?
        if [[ "$child_status" -gt 128 ]] && kill -0 "$child" 2>/dev/null; then
            continue
        fi
        break
    done
    set -e
}

# True iff the controller has marked this job preempted (a timestamp in the
# job record's PreemptTime). The always-present PreemptEligibleTime cannot
# collide ("PreemptTime" is not a substring of it); the boundary is only
# defensive. Positive timestamp match: any scontrol failure, purged job,
# junk, or multi-record output falls on the NOT-preempted side, which takes
# the loud terminal path. Call only as an if/&&/|| condition — a bare call
# would abort at the assignment under set -e.
rq_job_is_preempted() {
    local preempt_time
    preempt_time=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | grep -oE '(^| )PreemptTime=[^ ]+' | head -n 1 | cut -d= -f2)
    [[ "$preempt_time" =~ ^[0-9]{4}- ]]
}

# Requeue iff: we requested a checkpoint for a preemption (not a TERM), the
# child exited with a checkpoint-and-exit status, and the trainer's ack binds
# our exact request token to a commit-marked checkpoint. Writes the
# authorization for the next episode before `scontrol requeue`. Never returns.
rq_requeue_or_exit() {
    local acknowledged_step next_restart auth_tmp checkpoint_dir
    if [[ "$preempt_signal" -eq 1 && "$termination_signal" -eq 0 ]]; then
        case "$child_status" in
            1|15|124|143)
                [[ -f "$CHECKPOINT_ACK_PATH" ]] \
                    || rq_fail 125 "Missing checkpoint acknowledgment: $CHECKPOINT_ACK_PATH"
                grep -Fxq -- "request=$checkpoint_request_token" "$CHECKPOINT_ACK_PATH" \
                    || rq_fail 125 "Checkpoint ack does not match request token $checkpoint_request_token"
                acknowledged_step=$(sed -n 's/^step=//p' "$CHECKPOINT_ACK_PATH")
                [[ "$acknowledged_step" =~ ^[0-9]+$ ]] \
                    || rq_fail 125 "Checkpoint ack step is not numeric: $acknowledged_step"
                checkpoint_dir="$PROJECT_DIR/checkpoints/checkpoint_$acknowledged_step"
                [[ -f "$checkpoint_dir/checkpoint_complete" ]] \
                    || rq_fail 125 "Acked checkpoint is incomplete: $checkpoint_dir/checkpoint_complete is missing"
                [[ -f "$checkpoint_dir/pytorch_model_fsdp_0/.metadata" ]] \
                    || rq_fail 125 "Acked checkpoint is missing DCP metadata: $checkpoint_dir/pytorch_model_fsdp_0/.metadata"
                [[ ! -e "$TERMINATION_VETO_PATH" ]] || rq_fail 125 "Requeue vetoed by TERM"
                next_restart=$((RESTART_COUNT + 1))
                auth_tmp="$PROJECT_DIR/.requeue_authorization.$SLURM_JOB_ID.tmp"
                printf 'restart_count=%s\nrequest=%s\nstep=%s\n' \
                    "$next_restart" "$checkpoint_request_token" "$acknowledged_step" > "$auth_tmp"
                mv "$auth_tmp" "$REQUEUE_AUTH_PATH"
                printf 'REQUEUE_CHECKPOINT_STEP=%s CHILD_STATUS=%s\n' "$acknowledged_step" "$child_status"
                [[ ! -e "$TERMINATION_VETO_PATH" ]] || {
                    rm -f "$REQUEUE_AUTH_PATH"
                    rq_fail 125 "Requeue vetoed by TERM"
                }
                if ! scontrol requeue "$SLURM_JOB_ID"; then
                    # Under PreemptMode=REQUEUE the controller requeues a
                    # preempted job itself, and can reject our explicit
                    # requeue while that is in flight. Deleting the
                    # authorization then would make the controller's episode
                    # die unauthorized (exit 3) — so when the job shows a
                    # PreemptTime, keep the authorization and defer to the
                    # scheduler's requeue.
                    if rq_job_is_preempted; then
                        echo "scontrol requeue failed but the job is preempted; deferring to the scheduler requeue" >&2
                        exit 0
                    fi
                    rm -f "$REQUEUE_AUTH_PATH"
                    exit 126
                fi
                exit 0
                ;;
        esac
    fi

    if [[ "$preempt_signal" -eq 1 && "$child_status" -ne 0 ]]; then
        printf 'Refusing requeue: preempt=%s term=%s child_status=%s\n' \
            "$preempt_signal" "$termination_signal" "$child_status" >&2
    fi
    exit "$child_status"
}
