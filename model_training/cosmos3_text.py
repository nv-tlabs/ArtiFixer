# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 tokenizer mechanics: the token cap and input-id padding.

The family-neutral prompt-format registry and the shared text-dropout
policy live in model_training/prompt_formats.py.
"""

from __future__ import annotations

import numpy as np
import torch

COSMOS3_DEFAULT_MAX_TEXT_TOKENS = 1024


def pad_cosmos3_input_ids(
    input_ids: torch.Tensor | np.ndarray | list[int],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Padded to the same COSMOS3_DEFAULT_MAX_TEXT_TOKENS cap the prompt writer
    # clips captions against (prompt_hdf5.py), so the length assert below
    # holds for every corpus file by construction.
    max_length = COSMOS3_DEFAULT_MAX_TEXT_TOKENS
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    assert ids.ndim == 1, f"Cosmos3 input ids must be 1D, got shape {tuple(ids.shape)}"
    assert ids.numel() <= max_length, f"Cosmos3 input ids length {ids.numel()} exceeds max length {max_length}"
    padded = torch.full((max_length,), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros((max_length,), dtype=torch.bool)
    padded[: ids.numel()] = ids
    mask[: ids.numel()] = True
    return padded, mask
