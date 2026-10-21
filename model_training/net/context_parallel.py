# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist


def transformer_cp_head_counts(transformer: torch.nn.Module, required: bool = True) -> tuple[int, int] | None:
    """Return the (query, key/value) attention head counts that constrain
    context-parallel group sizes on the head-exchange paths.

    A head-exchange (Ulysses) CP forward leaves every rank with the full
    token sequence but only ``heads / cp_size`` attention heads, so a CP
    group size is only valid when it divides BOTH counts. Cosmos3 has real
    grouped-query attention, so the (smaller) KV-head count is the tighter
    of the two constraints; Wan is multi-head attention and the two counts
    are equal. Not every path exchanges heads (the Wan rollout cache keeps
    all heads per rank); the pipeline classes' ``cp_head_divisor`` /
    ``cp_size_candidates`` apply this constraint only where a path does.

    Works for both family wrappers without isinstance forks: the attention
    modules that participate in CP are exactly the ones whose processor
    carries the ``_cp_mesh`` slot that ``enable_context_parallel`` sets, and
    those modules expose their head counts as plain attributes —
    ``num_attention_heads``/``num_key_value_heads`` on Cosmos3 attention,
    ``heads`` on Wan attention.  Layers could in principle disagree, so each
    result is the gcd across layers: a CP size must divide every layer's
    counts.

    A transformer with no CP-capable attention module (test stubs, foreign
    backbones — including plain objects that are not ``nn.Module`` at all)
    raises ``RuntimeError`` with ``required=True`` (the default) and returns
    ``None`` with ``required=False`` — tolerant callers treat ``None`` as
    "no head constraint".
    """
    modules = getattr(transformer, "modules", None)
    if modules is None:
        # Not an nn.Module (trainer test stubs, foreign wrappers): same
        # treatment as a transformer with no CP processors.
        if not required:
            return None
        raise RuntimeError(
            f"{type(transformer).__name__} is not an nn.Module, so CP head divisibility cannot be validated."
        )
    query_heads_gcd = 0
    kv_heads_gcd = 0
    for module in modules():
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "_cp_mesh"):
            continue
        num_query_heads = getattr(module, "num_attention_heads", None)
        if num_query_heads is None:
            num_query_heads = getattr(module, "heads", None)
        if num_query_heads is None:
            raise RuntimeError(
                f"{type(module).__name__} carries a context-parallel attention processor "
                f"({type(processor).__name__}) but exposes neither num_attention_heads nor heads, "
                "so CP head divisibility cannot be validated."
            )
        # Multi-head attention modules (Wan) have one K/V head per query head.
        num_kv_heads = getattr(module, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_query_heads
        query_heads_gcd = math.gcd(query_heads_gcd, int(num_query_heads))
        kv_heads_gcd = math.gcd(kv_heads_gcd, int(num_kv_heads))
    if query_heads_gcd == 0:
        if not required:
            # Test stubs and foreign transformers carry no CP processors;
            # tolerant callers treat that as "no head constraint".
            return None
        raise RuntimeError(
            f"{type(transformer).__name__} has no attention module with a context-parallel "
            "processor, so CP head divisibility cannot be validated."
        )
    return query_heads_gcd, kv_heads_gcd


def set_context_parallel_state(module: Any, mesh: Any) -> None:
    module._cp_mesh = mesh
    module._cp_rank = mesh.get_local_rank()
    module._cp_world_size = mesh.size()


def clear_context_parallel_state(module: Any) -> None:
    module._cp_mesh = None
    module._cp_rank = 0
    module._cp_world_size = 1


def cp_frame_slice(total_frames: int, rank: int, world_size: int) -> tuple[int, int]:
    """Frame range ``[start, end)`` owned by ``rank`` when ``total_frames``
    are split evenly across the CP group — the single sharding rule both
    wrappers derive their sequence and camera shards from. Callers whose
    frame count does not divide evenly pad it first (see
    ``cp_padded_frame_count``); this helper stays strict so a missing pad
    fails loudly instead of dropping frames."""
    if total_frames % world_size != 0:
        raise ValueError(
            f"Cannot evenly split {total_frames} frames across {world_size} CP ranks. "
            "Adjust the frame count or CP degree so that frames divide evenly."
        )
    frames_per_rank = total_frames // world_size
    start = rank * frames_per_rank
    return start, start + frames_per_rank


def cp_padded_frame_count(total_frames: int, world_size: int) -> int:
    """Smallest frame count >= ``total_frames`` that splits evenly across the
    CP group. The pad unit is a whole frame because the transformers accept
    nothing smaller; the Cosmos3 wrapper pads its latent-frame axis up to
    this count before sharding (and keeps pad frames out of real outputs —
    see its forward)."""
    assert total_frames > 0, f"total_frames must be positive, got {total_frames}"
    assert world_size > 0, f"world_size must be positive, got {world_size}"
    return ((total_frames + world_size - 1) // world_size) * world_size


def all_gather_cat(tensor: torch.Tensor, mesh: Any, dim: int) -> torch.Tensor:
    """All-gather ``tensor`` across the CP mesh and concatenate along ``dim``.

    This is the NO-GRAD gather. ``dist.all_gather``'s outputs are
    disconnected from the graph (including the local rank's own slice), so
    gradient through the gathered result would be silently zero — a
    grad-enabled call whose ``tensor`` carries grad raises here, and a fully
    frozen grad-enabled forward (``tensor`` carries no grad) stays valid.
    Grad-carrying CP paths exist and go through the autograd-aware
    ``_with_grad`` twins in ``common/context_parallel.py``
    (``all_gather_cat_with_grad`` / ``all_to_all_with_grad``) instead.
    """
    if torch.is_grad_enabled() and tensor.requires_grad:
        raise RuntimeError("CP all-gather is not autograd-aware and is only valid during inference.")
    gathered = [torch.empty_like(tensor) for _ in range(mesh.size())]
    dist.all_gather(gathered, tensor.contiguous(), group=mesh.get_group())
    return torch.cat(gathered, dim=dim)
