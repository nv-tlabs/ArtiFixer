# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from model_training.arguments import get_common_opts


def main(args: argparse.Namespace):
    from model_training.stages import run_diffusion_forcing

    return run_diffusion_forcing(args)


def build_parser() -> argparse.ArgumentParser:
    parser = get_common_opts()

    parser.add_argument("--project_dir", required=True, type=Path)
    parser.add_argument("--base_checkpoint_dir", required=True, type=Path)
    parser.add_argument("--frames_per_block", default=7, type=int)

    parser.add_argument("--max_iterations", default=10000, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--validation_steps", default=1000, type=int)

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
