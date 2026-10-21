# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from model_eval.checkpoint_loading import add_checkpoint_args
from model_eval.dl3dv_reconstruction_evalsets import (
    DL3DV_RECONSTRUCTION_EVALSETS,
    add_dl3dv_reconstruction_args,
    default_dl3dv_num_views,
    is_dl3dv_reconstruction_evalset,
)
from model_eval.reconstructed_colmap_evalsets import (
    DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS,
    RECONSTRUCTED_COLMAP_EVALSETS,
    is_reconstructed_colmap_evalset,
)
from model_training.arguments import get_eval_common_opts
from model_training.checkpoint_metadata import resolve_camera_pose_alignment

EVALSETS = (
    "3dgrut_dl3dv_ours",
    *DL3DV_RECONSTRUCTION_EVALSETS,
    *RECONSTRUCTED_COLMAP_EVALSETS,
    "nerfbusters",
)


def validate_evalset_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.bidirectional_chunk_size <= 0:
        parser.error("--bidirectional_chunk_size must be positive")
    if args.output_fps <= 0:
        parser.error("--output_fps must be positive")
    if args.num_views is not None and args.num_views <= 0:
        parser.error("--num_views must be positive")
    if args.scene_ids and not is_dl3dv_reconstruction_evalset(args.evalset):
        parser.error("--scene_ids is supported only for DL3DV reconstruction evalsets")

    if is_dl3dv_reconstruction_evalset(args.evalset):
        required_args = ("split_path", "dl3dv_dir", "prompt_dir", "recon_results_dir")
    else:
        required_args_by_evalset = {
            "nerfbusters": (
                "nerfbusters_dir",
                "nerfbusters_recon_results_dir",
                "nerfbusters_recon_experiment_name",
                "nerfbusters_captions_dir",
            ),
            "3dgrut_dl3dv_ours": ("split_path", "dl3dv_dir", "prompt_dir"),
            "reconstructed_colmap": ("split_path",),
        }
        required_args = required_args_by_evalset[args.evalset]

    for name in required_args:
        if getattr(args, name) is None or getattr(args, name) == "":
            parser.error(f"--{name} is required for --evalset {args.evalset}")

    if args.render_trajectory == "trajectory" and not is_reconstructed_colmap_evalset(args.evalset):
        parser.error("--render_trajectory=trajectory is currently supported only for --evalset reconstructed_colmap")

    if args.render_diagnostics and args.context_parallel_size > 1:
        # Only the CP group leader runs the diagnostics forwards; the other ranks return after
        # the main denoise, so the ranks' NCCL collectives would stop matching and hang.
        parser.error("--render_diagnostics is not supported with --context_parallel_size > 1")

    if args.num_views is None and not is_reconstructed_colmap_evalset(args.evalset):
        args.num_views = default_dl3dv_num_views(args.evalset)


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    add_checkpoint_args(parser)
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="Immutable Hugging Face revision for model config, VAE, and scheduler loading.",
    )
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument(
        "--render_name",
        type=str,
        required=True,
        help="Directory name for this render run under --save_dir.",
    )
    parser.add_argument(
        "--inference_pipeline",
        type=str,
        default="kv_cache",
        choices=["kv_cache", "bidirectional"],
        help="Inference pipeline to use. 'bidirectional' runs the full model over the whole clip; "
        "'kv_cache' runs chunked causal inference of a distilled model with a windowed KV cache "
        "(--frames_per_block, --local_attn_size, --sink_size). Both support context parallelism.",
    )
    parser.add_argument(
        "--save_frame_outputs_only",
        action="store_true",
        help="Only save pred/gt/rendered PNG frames needed for metrics; skip comparison videos and diagnostic videos.",
    )
    parser.add_argument(
        "--distributed_timeout_minutes",
        default=60,
        type=int,
        help="NCCL process-group timeout used by distributed eval barriers.",
    )
    parser.add_argument("--num_inference_steps", default=4, type=int)
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=None,
        help="Classifier-free text guidance for bidirectional sampling. 1 disables CFG; "
        "omit to keep the pipeline default (5). Only supported with --inference_pipeline bidirectional; "
        "the distilled KV-cache rollout does not apply CFG.",
    )
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="Spatially tiled VAE encode/decode to bound the VAE memory peak. Slower and near- but not "
        "bit-identical, so off by default.",
    )
    parser.add_argument(
        "--frames_per_block",
        default=7,
        type=int,
        help="Latent frames per causal chunk for --inference_pipeline=kv_cache; clips are padded "
        "so the latent frame count is divisible by it. Must match the distilled checkpoint's "
        "training value: released Wan2.1 checkpoints use 7/21/7 "
        "(frames_per_block/local_attn_size/sink_size, the defaults here); the Wan2.2 stage-3 "
        "training defaults are 8/32/8.",
    )
    parser.add_argument(
        "--local_attn_size",
        default=21,
        type=int,
        help="KV-cache attention window in latent frames; must match the distilled checkpoint's "
        "training value (see --frames_per_block).",
    )
    parser.add_argument(
        "--sink_size",
        default=7,
        type=int,
        help="Attention-sink size in latent frames; must match the distilled checkpoint's "
        "training value (see --frames_per_block).",
    )
    parser.add_argument("--render_diagnostics", action="store_true")
    parser.add_argument("--replace_if_exists", action="store_true")
    parser.add_argument(
        "--render_trajectory",
        default="val_frames",
        choices=["val_frames", "all_frames", "trajectory"],
        help=(
            "Camera trajectory to render: held-out validation frames, the full source clip, "
            "or the prepared trajectory in the split."
        ),
    )
    parser.add_argument(
        "--bidirectional_chunk_size",
        default=81,
        type=int,
        help="Frames per eval chunk when --inference_pipeline=bidirectional; ignored by kv_cache inference.",
    )
    parser.add_argument(
        "--max_neighbors_per_encode",
        default=None,
        type=int,
        help=(
            "Maximum neighbor RGB frames to VAE-encode at once. Defaults to encoding all neighbors "
            "together for speed - set to 1 on memory-constrained runs."
        ),
    )
    parser.add_argument(
        "--output_suffix",
        default="",
        type=str,
        help="Optional suffix appended to eval output directory names.",
    )
    parser.add_argument("--output_fps", default=15, type=int, help="FPS for final pred/gt/rendered MP4 outputs.")


def add_dl3dv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split_path", type=Path, default=None)
    parser.add_argument("--dl3dv_dir", type=Path, default=None)
    parser.add_argument("--prompt_dir", type=Path, default=None)
    parser.add_argument("--recon_results_dir", type=Path, default=None)
    parser.add_argument(
        "--num_views",
        default=None,
        type=int,
        help="Number of reconstruction views to condition on. Defaults to 6; the reconstruction evalsets "
        "(reconstructed_colmap and the DL3DV reconstruction evalsets) default to up to "
        f"{DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS} selected views.",
    )
    scene_filter = parser.add_mutually_exclusive_group()
    scene_filter.add_argument("--scene_id", type=str, default=None)
    scene_filter.add_argument(
        "--scene_ids",
        type=str,
        nargs="+",
        default=None,
        help="Evaluate only these DL3DV reconstruction scene IDs.",
    )
    parser.add_argument(
        "--neighbor_selection_mode",
        type=str,
        default="evenly_spaced",
        choices=["consecutive", "evenly_spaced", "covisibility"],
    )


def add_nerfbusters_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nerfbusters_dir", type=Path, default=None, help="Path to nerfbusters dataset directory")
    parser.add_argument(
        "--nerfbusters_recon_results_dir",
        type=Path,
        default=None,
        help="Path to nerfbusters reconstruction results directory",
    )
    parser.add_argument(
        "--nerfbusters_checkpoint",
        type=str,
        default="30000",
        help="Checkpoint step to use for nerfbusters reconstruction renders",
    )
    parser.add_argument(
        "--nerfbusters_recon_experiment_name",
        type=str,
        default=None,
        help="Experiment name for nerfbusters reconstruction (3DGRUT output structure)",
    )
    parser.add_argument(
        "--nerfbusters_image_folder",
        type=str,
        default=None,
        help="Image folder for nerfbusters (if None, uses per-scene mapping for height=960)",
    )
    parser.add_argument(
        "--nerfbusters_captions_dir",
        type=Path,
        default=None,
        help="Root of the shared nerfbusters caption tree; the prompt format picks the filename.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = get_eval_common_opts()
    add_inference_args(parser)
    parser.add_argument(
        "--evalset",
        default="3dgrut_dl3dv_ours",
        type=str,
        choices=EVALSETS,
        help="Evaluation dataset.",
    )
    add_dl3dv_args(parser)
    add_dl3dv_reconstruction_args(parser)
    add_nerfbusters_args(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_evalset_args(parser, args)
    if args.text_guidance_scale is not None:
        if args.inference_pipeline != "bidirectional":
            parser.error("--text_guidance_scale is only supported with --inference_pipeline bidirectional")
        if not math.isfinite(args.text_guidance_scale) or args.text_guidance_scale < 1:
            parser.error("--text_guidance_scale must be finite and at least 1 (1 disables CFG)")
    if args.max_neighbors_per_encode is not None and args.max_neighbors_per_encode <= 0:
        parser.error("--max_neighbors_per_encode must be positive when set")
    args.camera_pose_alignment = resolve_camera_pose_alignment(
        args.checkpoint, args.model_id, camera_pose_alignment=args.camera_pose_alignment
    )
    return args
