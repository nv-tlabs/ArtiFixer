# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from model_eval.inference_args import parse_args


def main(args):
    from model_eval.inference import main as run

    return run(args)


if __name__ == "__main__":
    main(parse_args())
