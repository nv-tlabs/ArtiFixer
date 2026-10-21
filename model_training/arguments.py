# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from model_training.model_profiles import CAMERA_POSE_ALIGNMENTS, default_camera_pose_alignment

LOG_WITH_WANDB = "wandb"
LOG_WITH_CHOICES = (LOG_WITH_WANDB, "none")


def add_required_data_path_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split_path", type=Path, required=True)
    parser.add_argument("--dl3dv_dir", type=Path, required=True)
    parser.add_argument("--prompt_dir", type=Path, required=True)


def add_common_runtime_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_scaling_factor", default=0.01, type=float)

    # Available models include Wan-AI/Wan2.1-T2V-1.3B-Diffusers,
    # Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.2-TI2V-5B-Diffusers,
    # and nvidia/Cosmos3-Nano.
    parser.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-14B-Diffusers", type=str)
    parser.add_argument(
        "--default_negative_prompt_path",
        default=Path(__file__).resolve().parents[1] / "default_negative_prompt.pt",
        type=Path,
    )
    parser.add_argument(
        "--camera_pose_alignment",
        default=None,
        choices=CAMERA_POSE_ALIGNMENTS,
        help="Camera pose per latent frame: 'mean' averages each latent's raw-frame group; "
        "'last' uses the group's final frame. Training defaults to 'last' for Cosmos3 and 'mean' for Wan. "
        "Checkpoint evaluation uses the saved record and rejects conflicting values. "
        "Wan2.1 checkpoints without a record use 'mean'; other models require a record.",
    )
    parser.add_argument(
        "--attention_backend",
        default=None,
        type=str,
        help="Diffusers attention backend override. When omitted, auto-selects based on "
        "GPU arch: cuDNN SDPA on A100, native flash (FA3) on H100, native flash "
        "(FA4) on GB200. Valid values include '_native_flash', 'native', 'flash', 'flex'.",
    )
    parser.add_argument("--context_parallel_size", default=1, type=int)
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Base seed for per-rank training randomness and per-item inference noise. "
        "Inference seeds are independent of rank, item order, and skipped outputs. "
        "Training validation datasets retain their fixed sampling seed of 42.",
    )


class _TrainingArgumentParser(argparse.ArgumentParser):
    """Resolve model-specific camera defaults after parsing."""

    def parse_args(self, *args, **kwargs):
        parsed = super().parse_args(*args, **kwargs)
        if parsed.camera_pose_alignment is None and not parsed.validate_only:
            parsed.camera_pose_alignment = default_camera_pose_alignment(parsed.model_id)
        if parsed.test_panel_scenes < 0 or parsed.validation_chunk_frames <= 0:
            self.error("test_panel_scenes must be nonnegative and validation_chunk_frames must be positive")
        if parsed.validation_num_frames is not None and parsed.validation_num_frames <= 0:
            self.error("validation_num_frames must be positive")
        if parsed.max_checkpoints <= 0:
            self.error("max_checkpoints must be positive")
        if parsed.validate_only:
            parsed.gradient_checkpointing = False
        elif hasattr(parsed, "base_checkpoint_dir_critic") and parsed.base_checkpoint_dir_critic is None:
            self.error("DMD training requires --base_checkpoint_dir_critic")
        if (parsed.validate_only or parsed.validation_strict) and not parsed.validation_metrics:
            self.error("strict evaluation requires validation metrics")
        return parsed


def get_common_opts() -> argparse.ArgumentParser:
    parser = _TrainingArgumentParser()
    add_required_data_path_opts(parser)
    add_common_runtime_opts(parser)
    # Expose training controls only in the training parser.
    parser.add_argument("--log_with", default=LOG_WITH_WANDB, choices=LOG_WITH_CHOICES, type=str)
    parser.add_argument("--tracker_project_name", default="artifixer", type=str)
    parser.add_argument("--tracker_run_name", default=None, type=str)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument(
        "--dcp_cache_dir",
        default=Path.home() / ".cache" / "artifixer" / "dcp_weights",
        type=Path,
        help="Directory to cache DCP-format transformer weights (converted from HF pretrained on first use)",
    )
    parser.add_argument("--num_frames", default=81, type=int)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument(
        "--no_validation_metrics",
        dest="validation_metrics",
        action="store_false",
        help="Skip PSNR/SSIM and camera-consistency scoring of the validation clips.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run one validation pass on the loaded weights and exit without training. With "
        "--base_checkpoint_dir this evaluates a finished checkpoint: the clips, val/* scalars "
        "and test/* panel are written as JSON and optionally logged to W&B at step 0. "
        "Use --validation_manifest to replay a panel with compatible effective conditioning. See model_training/VALIDATION.md for capture/replay through the same pipeline.",
    )
    parser.add_argument(
        "--validation_num_frames",
        type=int,
        default=None,
        help="Override validation length in every stage, including DMD.",
    )
    parser.add_argument(
        "--validation_manifest",
        type=Path,
        default=None,
        help="Replay a panel manifest.json written by an earlier validation.",
    )
    parser.add_argument(
        "--validation_strict",
        action="store_true",
        help="Fail incomplete/nonfinite metric evaluation after writing results.json. Always on with --validate_only.",
    )
    parser.add_argument(
        "--validation_chunk_frames",
        type=int,
        default=32,
        help="Fixed raw-frame grid for chunk fidelity and boundary motion error.",
    )
    parser.add_argument(
        "--test_panel_scenes",
        default=0,
        type=int,
        help="Score this many fixed held-out samples at every validation under test/*, in the full "
        "and the zero-rendered configuration. val/* covers one sample per step, so its curve moves "
        "with the scene drawn; the panel's does not. Costs 2 extra generations per scene; 0 disables.",
    )
    parser.add_argument("--max_checkpoints", default=5, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--shard_size", default=None, type=int)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--checkpoint_every_n_blocks", default=1, type=int)
    parser.add_argument("--dataloader_num_workers", default=8, type=int)
    parser.add_argument("--dataloader_prefetch_factor", default=4, type=int)

    # Used to signal to the training process that it should save a checkpoint and exit with timeout signal for requeue
    parser.add_argument("--should_checkpoint_flag", default=None, type=Path)
    parser.add_argument("--check_state_finite", action="store_true")
    parser.add_argument(
        "--process_group_timeout_minutes",
        default=60,
        type=int,
        help=(
            "Accelerate process-group timeout used by FSDP, dataloader RNG-sync broadcasts, "
            "barriers, checkpointing, and distributed setup. Validation uses an additional "
            "dedicated sync group for long save/logging gaps."
        ),
    )
    parser.add_argument("--vae_chunk_halo_latent_frames", default=28, type=int)
    parser.add_argument("--max_validation_tasks", default=None, type=int)
    return parser


def get_eval_common_opts() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Leave pose alignment unset for the checkpoint resolver.
    add_common_runtime_opts(parser)
    # Shared model factories need these attributes; inference does not checkpoint gradients.
    parser.set_defaults(gradient_checkpointing=False, checkpoint_every_n_blocks=1)
    return parser
