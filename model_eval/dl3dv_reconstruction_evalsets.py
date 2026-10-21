# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

DL3DV_RECONSTRUCTION_EVALSETS = ("3dgrut_dl3dv_difix", "artifixer3d_dl3dv_difix", "artifixer3d_dl3dv_ours")

# Evalset naming follows the reconstruction stage and DL3DV split policy:
# - 3dgrut_dl3dv_ours uses the HDF5 outputs from our sparse-reconstruction pipeline.
# - The entries below adapt existing reconstruction-render directories into the reconstructed-COLMAP loader.
RECON_SUBDIR_BY_EVALSET = {
    "3dgrut_dl3dv_difix": None,
    "artifixer3d_dl3dv_difix": "distill_difix_eval",
    "artifixer3d_dl3dv_ours": "distill_ours_eval",
}


def is_dl3dv_reconstruction_evalset(evalset: str) -> bool:
    return evalset in DL3DV_RECONSTRUCTION_EVALSETS


def default_dl3dv_num_views(evalset: str) -> int | None:
    """Directory reconstructions select references per scene; HDF5 evaluation uses six."""
    return None if is_dl3dv_reconstruction_evalset(evalset) else 6


def add_dl3dv_reconstruction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dl3dv_reconstruction_checkpoint",
        type=str,
        default="30000",
        help="Reconstruction output checkpoint iteration to read from DL3DV reconstruction render directories.",
    )
    parser.add_argument(
        "--dl3dv_recon_subdir",
        type=str,
        default=None,
        help="Override the reconstruction experiment directory under each DL3DV scene.",
    )
