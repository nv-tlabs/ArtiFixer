# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the shared RolloutKvCache: the gradient storage flag on
gen chunks, the always-detached write-once context slots, and both families'
CP cache coordinates — Wan maps frame records to rank-LOCAL coordinates,
Cosmos3 keeps records GLOBAL over rank-local rows. Rollout-level behavior
(coverage, eviction, windowing) is pinned through the family test files that
drive the cache end-to-end."""

import pytest
import torch
from torch import nn

from model_training.net.rollout_kv_cache import RolloutKvCache


def _projected_chunk(projection: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    tokens = torch.randn(1, 4, 8)
    return projection(tokens), projection(tokens * 2)


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_gen_chunk_storage_honors_the_truncation_flag(truncate_rollout_gradients):
    """The flag's whole contract at the storage seam: by default a read-back
    gen chunk still connects to the parameters that produced it (backward
    through the cache reaches the projection weight); with truncation the
    stored entries carry no autograd history and the parameter gets no
    gradient through the cache."""
    projection = nn.Linear(8, 8)
    cache = RolloutKvCache(1, truncate_rollout_gradients=truncate_rollout_gradients)

    with torch.enable_grad():
        key, value = _projected_chunk(projection)
        cache.store_gen_chunk(0, 0, 0, 2, key, value)
        k_history, v_history = cache.read_gen_history(0, 1)
        if truncate_rollout_gradients:
            for tensor in (*k_history, *v_history):
                assert tensor.grad_fn is None and not tensor.requires_grad
        else:
            (k_history[0].sum() + v_history[0].sum()).backward()
            assert projection.weight.grad is not None and projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("slot", ["text", "neighbor"])
def test_context_slots_store_detached_and_write_once(slot):
    """Text and neighbor context K/V detach on store regardless of the gen
    flag (no trainable parameter sits upstream of a context write in either
    family), and a second store on the same layer is misuse."""
    projection = nn.Linear(8, 8)
    cache = RolloutKvCache(2, truncate_rollout_gradients=False)
    has_kv = getattr(cache, f"has_{slot}_kv")
    store_kv = getattr(cache, f"store_{slot}_kv")
    read_kv = getattr(cache, f"read_{slot}_kv")

    assert not has_kv(0) and not has_kv(1)
    with torch.enable_grad():
        key, value = _projected_chunk(projection)
        store_kv(0, key, value)
    assert has_kv(0) and not has_kv(1)
    for tensor in read_kv(0):
        assert tensor.grad_fn is None and not tensor.requires_grad
    # Reads return the stored objects (cached exactly once per rollout).
    assert read_kv(0)[0] is read_kv(0)[0]
    torch.testing.assert_close(read_kv(0)[0], key.detach())

    with pytest.raises(AssertionError, match="write-once"):
        store_kv(0, key, value)


def test_wan_cp_local_cache_coordinates_stay_contiguous_on_every_rank():
    """Under CP each Wan rank caches only its own frame shard, so the
    transformer maps the shard's global start (chunk_start + rank *
    frames_per_rank, the pipeline's frame_offset) to the rank's local
    stored-frame count before touching the cache (see
    ArtifixerTransformer.forward). A two-rank, two-chunk trace of that
    mapping must satisfy the contiguity contract on both ranks."""
    cp_size, frames_per_rank = 2, 4
    chunk_frames = cp_size * frames_per_rank
    key = value = torch.zeros(1, frames_per_rank, 8)
    for rank in range(cp_size):
        cache = RolloutKvCache(1, truncate_rollout_gradients=True)
        for chunk_index in range(2):
            frame_offset = chunk_index * chunk_frames + rank * frames_per_rank
            cache_offset = (frame_offset - rank * frames_per_rank) // cp_size
            # Every denoise pass of the chunk asserts; the final one commits.
            cache.assert_pass_starts_at(cache_offset)
            cache.commit_staged_chunk({0: (key, value)}, cache_offset, frames_per_rank)


def test_cosmos3_cp_global_frame_records_over_rank_local_rows():
    """Cosmos3's CP convention — the DELIBERATE divergence from Wan's local
    records pinned above: every rank stores only its own frame shard's ROWS
    (1/CP cache memory, all KV heads) but commits the GLOBAL frame record,
    because the pass contract (kv_cache_start_frame) is the global chunk
    start and also feeds the mRoPE/opacity seams. assert_pass_starts_at
    therefore consumes the transformer's global start UNCHANGED on every
    rank — no per-rank coordinate mapping — and coverage/eviction
    bookkeeping advances in cross-rank lockstep by construction (the
    records-only coverage arithmetic never looks at row counts). This test
    drives the cache directly, so it pins the CACHE side of the convention
    beside its Wan counterpart above: RolloutKvCache never converts
    coordinates. The transformer's commit seam — where a refactor to local
    records would actually land — is pinned through the real processor by
    the CP tests in tests/test_cosmos3_kv_cache_rollout.py."""
    cp_size, frames_per_chunk, spatial_rows = 2, 2, 3
    local_rows = (frames_per_chunk // cp_size) * spatial_rows
    caches = [RolloutKvCache(1, truncate_rollout_gradients=True) for _ in range(cp_size)]

    for chunk_index in range(3):
        global_start = chunk_index * frames_per_chunk  # the SAME start on every rank
        for rank, cache in enumerate(caches):
            cache.assert_pass_starts_at(global_start)
            shard = torch.full((local_rows, 2, 4), float(rank * 100 + chunk_index))
            cache.commit_staged_chunk({0: (shard, shard.clone())}, global_start, frames_per_chunk)

    for cache in caches:
        # Records are GLOBAL and identical across ranks; coverage counts
        # global frames even though each rank holds 1/CP of the rows.
        assert cache._gen_chunk_frames[0] == [(0, 2), (2, 2), (4, 2)]
        assert cache.total_stored_frames == 3 * frames_per_chunk
    rank0_rows, _ = caches[0].read_gen_history(0, 3)
    rank1_rows, _ = caches[1].read_gen_history(0, 3)
    for rank0_chunk, rank1_chunk in zip(rank0_rows, rank1_rows, strict=True):
        assert rank0_chunk.shape[0] == local_rows
        assert not torch.equal(rank0_chunk, rank1_chunk), "each rank stores its own shard's rows"

    # Window eviction driven by the same global records stays in lockstep:
    # coverage never rewinds, and the next pass start stays global.
    for cache in caches:
        cache.evict_oldest_gen_chunk(0)
        assert cache.total_stored_frames == 3 * frames_per_chunk
        cache.assert_pass_starts_at(3 * frames_per_chunk)
