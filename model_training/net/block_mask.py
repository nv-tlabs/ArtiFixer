# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers that build the prefixed block-causal (diffusion-forcing) BlockMask
used by ``ArtifixerCosmos3Transformer`` in cosmos3_transformer.py.

Kept out of cosmos3_transformer.py so they can be imported and unit tested on
CPU-only torch builds; cosmos3_transformer.py needs the torch >= 2.11
flash-attention API at import time.
"""

from collections.abc import Callable

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def build_block_causal_ends(
    total_tokens: int,
    frames_per_block: int,
    frame_seqlen: int,
    device: torch.device | str,
) -> torch.Tensor:
    """``ends[i]`` is the (exclusive) end index of token ``i``'s own temporal
    block: the block-causal mask attends to all tokens before the end of the
    current block."""
    block_token_size = frames_per_block * frame_seqlen
    ends = torch.zeros(total_tokens, device=device, dtype=torch.long)
    block_starts = torch.arange(0, end=total_tokens, step=block_token_size, device=device)
    for block_start in block_starts:
        ends[block_start : block_start + block_token_size] = block_start + block_token_size
    return ends


def build_prefix_block_causal_mask_mod(prefix_len: int, ends: torch.Tensor) -> Callable:
    """flex_attention mask_mod for packed ``[prefix ⊕ block-causal]`` keys:
    the first ``prefix_len`` keys are visible to every query, the keys after
    them follow the block-causal rule (visible up to the end of the query's
    own temporal block, per ``ends``). Queries index only the block-causal
    tokens (``ends`` coordinates), not the prefix."""
    # prefix_len changes every training step (per-item caption token count), so
    # a captured Python int gets promoted to a SymInt on the second distinct
    # value and FLASH can't compile it. A 0-dim tensor capture avoids the
    # promotion but fails in the CuteDSL codegen ("Expected TensorSSA, but got
    # int" -- its load lowers with a literal index 0). A per-query vector
    # indexed by q_idx, matching the ends[q_idx] capture that already compiles
    # here, avoids both failure modes.
    prefix_vec = torch.full_like(ends, prefix_len)

    def attention_mask(b, h, q_idx, kv_idx):
        p = prefix_vec[q_idx]
        return (kv_idx < p) | (kv_idx - p < ends[q_idx])

    return attention_mask


def create_prefix_block_causal_block_mask(
    prefix_len: int,
    total_tokens: int,
    frames_per_block: int,
    frame_seqlen: int,
    device: torch.device | str,
    block_size: int | tuple[int, int],
    pad_query_tokens: int = 0,
) -> BlockMask:
    """BlockMask for block-causal queries over ``[prefix ⊕ block-causal]``
    keys: the query axis spans all ``total_tokens`` block-causal tokens and
    the key axis the full ``prefix_len + total_tokens`` sequence. This is one
    global mask — under context parallelism every rank reuses it as-is, with
    no per-rank query window.

    ``pad_query_tokens`` extends the QUERY axis with that many trailing CP
    pad rows (the wrapper pads the latent-frame axis so any head-compatible
    CP size shards evenly). Pad queries get ``ends = 0``, so they attend only
    to the text prefix — never a real gen key — and produce finite rows the
    wrapper cuts after the output gather. The KEY axis stays the real
    sequence: the attention processor drops pad keys right after the
    all-to-all exchange, so no real query can ever see one."""
    ends = build_block_causal_ends(total_tokens, frames_per_block, frame_seqlen, device)
    assert pad_query_tokens >= 0, f"pad_query_tokens must be non-negative, got {pad_query_tokens}"
    if pad_query_tokens:
        ends = torch.cat([ends, ends.new_zeros(pad_query_tokens)])
    prefix_mask_mod = build_prefix_block_causal_mask_mod(prefix_len, ends)
    return create_block_mask(
        prefix_mask_mod,
        B=None,
        H=None,
        Q_LEN=total_tokens + pad_query_tokens,
        KV_LEN=prefix_len + total_tokens,
        device=device,
        BLOCK_SIZE=block_size,
    )
