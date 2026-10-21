# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS = 12
RECONSTRUCTED_COLMAP_EVALSETS = ("reconstructed_colmap",)


def is_reconstructed_colmap_evalset(evalset: str) -> bool:
    return evalset in RECONSTRUCTED_COLMAP_EVALSETS


def num_views_output_label(num_views: int | None) -> str:
    """View-count part of the output directory name; 'autoN' when num_views is chosen per scene."""
    if num_views is None:
        return f"auto{DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS}"
    return str(num_views)
