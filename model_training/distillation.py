# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from model_training.arguments import get_common_opts


def main(args: argparse.Namespace):
    from model_training.stages import run_distillation

    return run_distillation(args)


def build_parser() -> argparse.ArgumentParser:
    parser = get_common_opts()
    # Keep all ten scheduled checkpoints from the default 1,000-iteration run.
    # Timeout saves also count toward this limit; longer runs can override it.
    parser.set_defaults(max_checkpoints=10)

    parser.add_argument("--project_dir", required=True, type=Path)
    parser.add_argument("--base_checkpoint_dir", required=True, type=Path)
    parser.add_argument(
        "--base_checkpoint_dir_critic",
        default=None,
        type=Path,
        help="Required for DMD training; unused with --validate_only.",
    )

    parser.add_argument(
        "--model_id_critic",
        default=None,
        type=str,
        help="Model config for the DMD real/fake score critics. Defaults to --model_id; must "
        "share the student's model family.",
    )

    parser.add_argument("--num_inference_steps", default=4, type=int)
    parser.add_argument("--frames_per_block", default=7, type=int)
    parser.add_argument("--local_attn_size", default=21, type=int)
    parser.add_argument("--sink_size", default=7, type=int)
    parser.add_argument(
        "--truncate_rollout_gradients",
        action="store_true",
        help="Store rollout K/V detached so student-phase gradients stop at chunk boundaries. Off "
        "by default: gradients flow through cached K/V across chunks.",
    )
    parser.add_argument("--text_guidance_scale", default=3.0, type=float)
    parser.add_argument("--dfake_gen_update_ratio", default=5, type=int)

    parser.add_argument("--max_iterations", default=1000, type=int)
    parser.add_argument("--learning_rate", default=2e-6, type=float)
    parser.add_argument("--learning_rate_critic", default=4e-7, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=10.0, type=float)
    parser.add_argument("--ema_weight", default=0.99, type=float)
    parser.add_argument("--ema_start_step", default=200, type=int)

    # Preserve intermediate candidates for validation and checkpoint selection.
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--validation_steps", default=100, type=int)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
