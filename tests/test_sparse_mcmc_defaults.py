# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tests._helpers import argparse_default

SPARSE_MCMC_CONFIG = "apps/colmap_3dgut_sparse_mcmc.yaml"


def test_sparse_reconstruction_cli_defaults_to_mcmc_config():
    assert (
        argparse_default("data_processing/run_sparse_reconstruction.py", "--sparse_recon_config") == SPARSE_MCMC_CONFIG
    )
