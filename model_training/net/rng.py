# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-invariant initialization for modules the pretrained checkpoint does not supply.

train_utils.py seeds rank r with (seed + r) -- right for dropout and noise, wrong for
parameter init: FSDP2 shards each rank's OWN copy and never broadcasts rank 0, so any
module built here that the hub checkpoint does not ship (Wan's neighbor cross-attention
``add_k_proj``, SCoPE, the opacity embeddings, prope) becomes a patchwork of per-rank
draws, and the same --seed builds a different model at a different GPUs-per-node count.
"""

from __future__ import annotations

import functools
import random
from contextlib import contextmanager

import numpy as np
import torch

# Deliberately not --seed: holding init fixed is what makes a seed sweep interpretable.
MODULE_INIT_SEED = 0


@contextmanager
def pinned_init_rng(seed: int = MODULE_INIT_SEED):
    """Pin every RNG stream construction can draw from, then restore the caller's."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    # Modules constructed with device=... draw from the CUDA generator directly.
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def rank_invariant_init(init_fn):
    """Decorate ``__init__`` so everything it constructs is identical on every rank.

    On the constructor rather than the call sites (get_pipe, the KV-cache pipeline,
    the DMD critic in distillation.py): a missed call site fails silently, as a
    differently trained model rather than an error.
    """

    @functools.wraps(init_fn)
    def wrapper(*args, **kwargs):
        with pinned_init_rng():
            return init_fn(*args, **kwargs)

    return wrapper
