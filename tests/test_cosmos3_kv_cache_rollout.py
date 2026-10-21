# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached-rollout outputs, gradients, camera inputs, and distributed parity.

Live teacher-forcing comparisons check correctness. Stored outputs detect drift
between changes to the model or rollout algorithm.
"""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from diffusers import Cosmos3OmniTransformer
from torch.distributed import DeviceMesh
from torch.nn.attention import SDPBackend, sdpa_kernel

import model_training.common.context_parallel as common_cp_module
import model_training.net.cosmos3_transformer as cosmos3_transformer_module
from model_training.net.cosmos3_transformer import (
    ArtifixerCosmos3AttnProcessor,
    ArtifixerCosmos3Transformer,
    get_3d_mrope_ids_vae_tokens,
)
from model_training.net.rollout_kv_cache import RolloutKvCache
from model_training.pipeline.cosmos3_kv_cache_pipeline import ArtifixerCosmos3KvCachePipeline
from tests._helpers import (
    REDUCED_CONFIG,
    VAE_SCALE_FACTOR_TEMPORAL,
    _activate_conditioning_adapters,
    _assert_cosmos3_cp_gradient_mean,
    _capture_cosmos3_forward,
    _cosmos3_noisy_sequence_indexes,
    _GroupedFakeCpMesh,
    _identity_rotary,
    _joint_reference_kwargs,
    _ProcessorAttention,
    _reduced_real_transformer,
    _run_cp_workers,
    _select_cosmos3_references,
    _slice_cosmos3_target_kwargs,
    _TinyCosmosVae,
    cosmos3_projected_gen_qkv,
    cosmos3_rankwise_all_to_all,
)
from tests._teacher_forcing_reference import cosmos3_teacher_forcing_reference

# Explicit rollout-window option values for every pipeline construction in
# this file: the production defaults. At the 2-frame test chunks the derived
# window is 10 chunks, so training-length rollouts below never evict.
LOCAL_ATTN_SIZE = 21
SINK_SIZE = 7

pytestmark = pytest.mark.slow

FRAMES_PER_CHUNK = 2
NUM_CHUNKS = 3
TOTAL_FRAMES = FRAMES_PER_CHUNK * NUM_CHUNKS

EXPECTED_OUTPUTS_PATH = Path(__file__).resolve().parent / "data" / "cosmos3_rollout_expected_outputs.pt"
# Allow small CPU-kernel differences in the stored-output checks.
EXPECTED_RTOL = 2e-3
EXPECTED_ATOL = 1e-5


@pytest.fixture(scope="module")
def expected_outputs() -> dict:
    return torch.load(EXPECTED_OUTPUTS_PATH, weights_only=True)


def _rollout_kwargs(total_frames: int = TOTAL_FRAMES) -> dict:
    return _joint_reference_kwargs(total_frames)


def _chunk_slice(kwargs: dict, start: int, end: int) -> dict:
    """Cache-pass conditioning: ONLY the chunk's frames, absolute indexing
    (opacity = the chunk's real 4-frame pixel windows)."""
    sliced = _slice_cosmos3_target_kwargs(kwargs, start, end)
    del sliced["hidden_states"]  # each pass supplies its own
    return sliced


def _chunk_tensors(seed: int, num_chunks: int = NUM_CHUNKS) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    torch.manual_seed(seed)
    noisy = [torch.randn(1, 48, FRAMES_PER_CHUNK, 4, 4) for _ in range(num_chunks)]
    clean = [torch.randn(1, 48, FRAMES_PER_CHUNK, 4, 4) for _ in range(num_chunks)]
    return noisy, clean


def _cache_chunk_forward(
    transformer: ArtifixerCosmos3Transformer,
    kwargs: dict,
    cache: RolloutKvCache,
    noisy_chunk: torch.Tensor,
    chunk_index: int,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
) -> torch.Tensor:
    start = chunk_index * frames_per_chunk
    return transformer(
        **_chunk_slice(kwargs, start, start + frames_per_chunk),
        hidden_states=noisy_chunk,
        kv_cache=cache,
        kv_cache_start_frame=start,
    )[0]


def _seed_chunk(
    transformer: ArtifixerCosmos3Transformer,
    kwargs: dict,
    cache: RolloutKvCache,
    clean_chunk: torch.Tensor,
    chunk_index: int,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
) -> None:
    start = chunk_index * frames_per_chunk
    seed_kwargs = _chunk_slice(kwargs, start, start + frames_per_chunk)
    seed_kwargs["timestep"] = None
    transformer(
        **seed_kwargs,
        hidden_states=clean_chunk.detach(),
        kv_cache=cache,
        kv_cache_seed=True,
        kv_cache_start_frame=start,
    )


# --- value oracle at the WRAPPER level: expected-output-pinned chunk passes ---


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_cached_chunk_passes_match_the_stored_expected_outputs(expected_outputs, truncate_rollout_gradients):
    """Compare cached chunk outputs with the stored regression fixture. The
    expected_outputs were measured bit-identical to the truncated-BPTT full-prefix
    oracle, so this pins the cache's absolute-position mRoPE, chunk-only
    conditioning, text K/V reuse, and history reads in one shot. Both gradient
    storage modes reproduce the same expected outputs: detaching never changes
    values."""
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    kwargs = _rollout_kwargs()
    noisy_chunks, clean_chunks = _chunk_tensors(seed=7)

    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=truncate_rollout_gradients)
        for chunk_index in range(NUM_CHUNKS):
            cache_out = _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[chunk_index], chunk_index)
            assert cache_out.shape == (1, 48, FRAMES_PER_CHUNK, 4, 4)
            torch.testing.assert_close(
                cache_out,
                expected_outputs["wrapper_chunk_x0"][chunk_index],
                rtol=EXPECTED_RTOL,
                atol=EXPECTED_ATOL,
                msg=lambda base, i=chunk_index: f"chunk {i}: cache pass diverged from the expected-output oracle\n{base}",
            )
            if chunk_index < NUM_CHUNKS - 1:
                _seed_chunk(transformer, kwargs, cache, clean_chunks[chunk_index], chunk_index)


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_kv_cache_gradient_checkpointing_recompute_reads_snapshotted_history(truncate_rollout_gradients):
    """Backward of a gradient-checkpointed cache pass re-runs the layers after
    later chunks were seeded, so cache reads must be pinned by the
    context-snapshotted history count, not live cache state. Pin: seed chunk 1
    AFTER chunk 1's forward but BEFORE its backward, then assert checkpointed
    grads equal the non-checkpointed grads bitwise — in both gradient storage
    modes."""
    kwargs = _rollout_kwargs()
    noisy_chunks, clean_chunks = _chunk_tensors(seed=13)

    def run(gradient_checkpointing: bool) -> dict[str, torch.Tensor | None]:
        transformer = _reduced_real_transformer(gradient_checkpointing=gradient_checkpointing)
        _activate_conditioning_adapters(transformer)
        cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=truncate_rollout_gradients)
        with torch.no_grad():
            _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[0], 0)
            _seed_chunk(transformer, kwargs, cache, clean_chunks[0], 0)
        with torch.enable_grad():
            output = _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[1], 1)
        # Grow the cache past the pass's snapshot BEFORE backward, exactly the
        # rollout's timing (later chunks seed before the trainer's backward).
        with torch.no_grad():
            _seed_chunk(transformer, kwargs, cache, clean_chunks[1], 1)
        with torch.enable_grad():
            output.square().sum().backward()
        return {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad
        }

    plain_grads = run(gradient_checkpointing=False)
    checkpointed_grads = run(gradient_checkpointing=True)
    assert plain_grads.keys() == checkpointed_grads.keys()
    for name, plain_grad in plain_grads.items():
        checkpointed_grad = checkpointed_grads[name]
        if plain_grad is None:
            assert checkpointed_grad is None, name
            continue
        assert checkpointed_grad is not None and torch.equal(
            plain_grad, checkpointed_grad
        ), f"{name}: checkpointed backward diverged — the recompute did not read the snapshotted history"


# --- cache hygiene ---


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_kv_cache_hygiene_text_kv_once_write_once_and_mode_dependent_detach(truncate_rollout_gradients):
    """Cache hygiene on the real class, with grad ENABLED for every pass so
    the storage policy is proven to come from the store, not from an ambient
    no_grad: text K/V are always detached and cached exactly once per rollout
    (same objects on later passes); gen entries are detached exactly when
    ``truncate_rollout_gradients`` is set; a chunk's entries never change
    after seeding (write-once), across both later denoise and later seed
    passes. Write-once IS the growing-prefix invariance: a chunk's K/V are
    computed knowing only chunks <= itself, then frozen — frame 0's
    contribution to attention is literally immutable no matter how many later
    chunks arrive."""
    transformer = _reduced_real_transformer()
    kwargs = _rollout_kwargs()
    noisy_chunks, clean_chunks = _chunk_tensors(seed=17)
    cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=truncate_rollout_gradients)
    num_layers = len(transformer.base_transformer.layers)

    with torch.enable_grad():
        _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[0].requires_grad_(True), 0)
    assert cache.num_cached_chunks == 0, "denoise passes must not write gen history"
    assert all(cache.has_text_kv(layer_index) for layer_index in range(num_layers))
    text_kv_objects = [cache.read_text_kv(layer_index) for layer_index in range(num_layers)]

    with torch.enable_grad():
        _seed_chunk(transformer, kwargs, cache, clean_chunks[0], 0)
    assert cache.num_cached_chunks == 1
    for layer_index in range(num_layers):
        for tensor in cache.read_text_kv(layer_index):
            assert (
                tensor.grad_fn is None and not tensor.requires_grad
            ), "cached text K/V must be detached even when stored under grad"
        for segments in cache.read_gen_history(layer_index, 1):
            for tensor in segments:
                if truncate_rollout_gradients:
                    assert (
                        tensor.grad_fn is None and not tensor.requires_grad
                    ), "truncation mode must store gen K/V detached even under grad"
                else:
                    assert tensor.requires_grad, (
                        "default mode must keep the gen K/V graph so later chunks' losses "
                        "backpropagate through cached context"
                    )
    chunk0_snapshot = []
    for layer_index in range(num_layers):
        entry_lists = cache.read_gen_history(layer_index, 1)
        chunk0_snapshot.append(tuple(entries[0].clone() for entries in entry_lists))

    with torch.enable_grad():
        _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[1], 1)
        _seed_chunk(transformer, kwargs, cache, clean_chunks[1], 1)
        _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[2], 2)
    assert cache.num_cached_chunks == 2
    for layer_index in range(num_layers):
        # Text K/V cached exactly once: later passes return the SAME tensor objects.
        for cached_now, cached_before in zip(
            cache.read_text_kv(layer_index), text_kv_objects[layer_index], strict=True
        ):
            assert cached_now is cached_before, "text K/V must be cached exactly once per rollout"
        # write-once: chunk 0's entries are bitwise-frozen after seeding.
        for segments_now, segments_before in zip(
            cache.read_gen_history(layer_index, 1), chunk0_snapshot[layer_index], strict=True
        ):
            for tensor_now in segments_now:
                assert torch.equal(
                    tensor_now, segments_before
                ), "a seeded chunk's cache entries changed on a later pass (write-once violated)"

    # The write-once contract is also enforced loudly at the API.
    layer0_text_kv = cache.read_text_kv(0)
    with pytest.raises(AssertionError, match="write-once"):
        cache.store_text_kv(0, *layer0_text_kv)
    k_vision, v_vision = (segments[0] for segments in cache.read_gen_history(0, 1))
    with pytest.raises(AssertionError, match="write-once and sequential"):
        cache.store_gen_chunk(0, 0, 0, FRAMES_PER_CHUNK, k_vision, v_vision)
    # ... and so is frame coverage at the store itself (the
    # chunk_index guard alone is tautological when callers snapshot it from
    # num_cached_chunks — a duplicate chunk at the CORRECT next index must
    # still fail on its frames).
    with pytest.raises(AssertionError, match="contiguously in temporal order"):
        cache.store_gen_chunk(0, cache.num_cached_chunks, FRAMES_PER_CHUNK, FRAMES_PER_CHUNK, k_vision, v_vision)


# --- offset seams (the length contracts and absolute mRoPE positions) ---


def test_kv_cache_chunk_mrope_positions_are_absolute():
    """The chunk-only pass reuses the CP ``start_frame_offset`` seam: its
    vision mRoPE ids must equal the full-clip builder's rows for the same
    absolute positions, bitwise (fps modulation active)."""
    start, end = 2 * FRAMES_PER_CHUNK, 3 * FRAMES_PER_CHUNK
    common = dict(
        grid_h=2,
        grid_w=2,
        temporal_offset=15004.0,
        reset_spatial_indices=True,
        fps=60.0,
        base_fps=24.0,
        temporal_compression_factor=VAE_SCALE_FACTOR_TEMPORAL,
    )
    full_ids, _ = get_3d_mrope_ids_vae_tokens(grid_t=end, start_frame_offset=0, **common)
    chunk_ids, _ = get_3d_mrope_ids_vae_tokens(grid_t=end - start, start_frame_offset=start, **common)
    spatial = common["grid_h"] * common["grid_w"]
    assert torch.equal(chunk_ids, full_ids[:, start * spatial : end * spatial])


def test_kv_cache_chunk_opacity_packing_matches_full_clip_slice():
    """Offset-aware opacity contract: a mid-video chunk's packed opacity
    tokens equal the full-clip packing's frames bitwise (real 4-frame
    windows, no first-frame pad), and a wrong pixel length still fails
    loudly."""
    transformer = _reduced_real_transformer()
    kwargs = _rollout_kwargs()
    start, end = FRAMES_PER_CHUNK, 2 * FRAMES_PER_CHUNK
    full_grid = transformer._opacity_tokens(kwargs["opacity"], kwargs["hidden_states"])
    chunk_kwargs = _chunk_slice(kwargs, start, end)
    chunk_grid = transformer._opacity_tokens(
        chunk_kwargs["opacity"], kwargs["hidden_states"][:, :, start:end], start_frame_offset=start
    )
    assert torch.equal(chunk_grid, full_grid[:, :, start:end])

    with pytest.raises(ValueError, match="chunk-only pass"):
        transformer._opacity_tokens(
            kwargs["opacity"][:, : (end - start - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1],
            kwargs["hidden_states"][:, :, start:end],
            start_frame_offset=start,
        )


# --- pass validation: the cache contract fails loudly, never silently ---


def test_kv_cache_pass_validation_rejects_contract_violations():
    transformer = _reduced_real_transformer()
    kwargs = _rollout_kwargs()
    noisy_chunks, clean_chunks = _chunk_tensors(seed=19)
    chunk0_kwargs = _chunk_slice(kwargs, 0, FRAMES_PER_CHUNK)

    # Seed passes are conditioning-mode: a timestep is a contract violation.
    with pytest.raises(AssertionError, match="timestep=None"):
        transformer(
            **chunk0_kwargs,
            hidden_states=clean_chunks[0],
            kv_cache=transformer.new_rollout_kv_cache(truncate_rollout_gradients=False),
            kv_cache_seed=True,
        )

    # A mid-video chunk on an empty cache means a skipped/reordered seed —
    # the frame-coverage contract rejects it (frame 2 vs coverage [0, 0)).
    with pytest.raises(AssertionError, match="contiguously in temporal order"):
        transformer(
            **_chunk_slice(kwargs, FRAMES_PER_CHUNK, 2 * FRAMES_PER_CHUNK),
            hidden_states=noisy_chunks[1],
            kv_cache=transformer.new_rollout_kv_cache(truncate_rollout_gradients=False),
            kv_cache_start_frame=FRAMES_PER_CHUNK,
        )

    # Reusing a rollout's cache for a new rollout (frame-0 chunk with gen
    # history) is rejected: caches live for ONE generate call.
    used_cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
    with torch.no_grad():
        _cache_chunk_forward(transformer, kwargs, used_cache, noisy_chunks[0], 0)
        _seed_chunk(transformer, kwargs, used_cache, clean_chunks[0], 0)
    with pytest.raises(AssertionError, match="ONE rollout"):
        transformer(**chunk0_kwargs, hidden_states=noisy_chunks[0], kv_cache=used_cache)

    # kv kwargs without a cache are a caller bug, not a silent no-op.
    with pytest.raises(AssertionError, match="require kv_cache"):
        transformer(**chunk0_kwargs, hidden_states=noisy_chunks[0], kv_cache_seed=True)

    # Supervised diffusion forcing (causal_block_frames) never takes cache
    # passes: the cache is a rollout-only contract.
    df_transformer = _reduced_real_transformer(causal_block_frames=FRAMES_PER_CHUNK)
    with pytest.raises(AssertionError, match="rollout-only contract"):
        df_transformer(
            **chunk0_kwargs,
            hidden_states=noisy_chunks[0],
            kv_cache=df_transformer.new_rollout_kv_cache(truncate_rollout_gradients=False),
        )


def test_kv_cache_frame_coverage_rejects_previously_silent_misuse():
    """Frame-coverage bookkeeping makes cache misuse loud: a duplicate
    mid-video seed (which would otherwise become the next "chunk" and skew
    every later denoise by ~1.7e-1), a re-denoise of an already-seeded chunk,
    and a gapped start frame must all raise, naming the offending frames."""
    transformer = _reduced_real_transformer()
    kwargs = _rollout_kwargs()
    noisy_chunks, clean_chunks = _chunk_tensors(seed=29)
    cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)

    with torch.no_grad():
        _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[0], 0)
        _seed_chunk(transformer, kwargs, cache, clean_chunks[0], 0)
        _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[1], 1)
        _seed_chunk(transformer, kwargs, cache, clean_chunks[1], 1)

        # Duplicate mid-video seed.
        with pytest.raises(AssertionError, match=r"seed pass starts at latent frame 2.*stored frames \[0, 4\)"):
            _seed_chunk(transformer, kwargs, cache, clean_chunks[1], 1)
        # Re-denoise of an already-seeded chunk (stale pass).
        with pytest.raises(AssertionError, match=r"denoise pass starts at latent frame 2.*stored frames \[0, 4\)"):
            _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[1], 1)
        # None of the rejected passes may have advanced or mutated coverage.
        assert cache.num_cached_chunks == 2 and cache.total_stored_frames == 2 * FRAMES_PER_CHUNK

        # Gapped start frame: skipping a chunk is equally loud.
        gapped_cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
        _cache_chunk_forward(transformer, kwargs, gapped_cache, noisy_chunks[0], 0)
        _seed_chunk(transformer, kwargs, gapped_cache, clean_chunks[0], 0)
        with pytest.raises(AssertionError, match=r"starts at latent frame 4.*stored frames \[0, 2\)"):
            _cache_chunk_forward(transformer, kwargs, gapped_cache, noisy_chunks[2], 2)

        # The valid continuation still works after all the rejected probes.
        _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[2], 2)


def test_processor_state_disarms_after_every_pass():
    """The conditioning pre-hook arms per-call processor state;
    without a paired disarm the LAST pass's KV-cache context stays pinned on
    every layer's processor after the rollout returns — holding the whole
    cache alive (~1 GiB at production shape). The post-attention disarm hook
    must clear it after every pass, including gradient-checkpointing
    recomputes during backward (which re-arm via the pre-hook and must
    re-clear)."""

    def assert_disarmed(transformer: ArtifixerCosmos3Transformer) -> None:
        for layer in transformer.base_transformer.layers:
            processor = layer.self_attn.processor
            assert processor._kv_cache_context is None, "a processor retained a KV-cache reference"
            assert processor._gen_block_mask is None, "a processor retained an attention mask"

    kwargs = _rollout_kwargs()
    noisy_chunks, clean_chunks = _chunk_tensors(seed=37)

    for gradient_checkpointing in (False, True):
        transformer = _reduced_real_transformer(gradient_checkpointing=gradient_checkpointing)
        cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
        with torch.no_grad():
            _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[0], 0)
            _seed_chunk(transformer, kwargs, cache, clean_chunks[0], 0)
        with torch.enable_grad():
            output = _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[1], 1)
        assert_disarmed(transformer)
        with torch.enable_grad():
            output.square().sum().backward()
        assert_disarmed(transformer)


def test_processor_state_disarms_after_an_exception_mid_pass(monkeypatch):
    """Exception arm of the disarm contract: forward hooks are SKIPPED for a
    module that raises, so an exception between the pre-hook arm and the
    post-attention disarm hook would leave the failing layer's processor
    pinning the KV-cache context past the failed pass — the wrapper's forward
    ``finally`` must clear every layer even then. Inject a failure at layer
    1's cache read, mid-pass: layer 0 has already disarmed via its hook,
    layer 1 is ARMED when it raises."""
    transformer = _reduced_real_transformer()
    kwargs = _rollout_kwargs()
    noisy_chunks, clean_chunks = _chunk_tensors(seed=41)
    cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
    with torch.no_grad():
        _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[0], 0)
        _seed_chunk(transformer, kwargs, cache, clean_chunks[0], 0)

    original_read = RolloutKvCache.read_gen_history

    def failing_read(self, layer_index, num_chunks):
        if layer_index == 1:
            raise RuntimeError("injected mid-pass cache failure")
        return original_read(self, layer_index, num_chunks)

    monkeypatch.setattr(RolloutKvCache, "read_gen_history", failing_read)
    with torch.no_grad(), pytest.raises(RuntimeError, match="injected mid-pass cache failure"):
        _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[1], 1)

    assert transformer._conditioning_context is None, "the failed pass leaked the conditioning context"
    for layer in transformer.base_transformer.layers:
        processor = layer.self_attn.processor
        assert (
            processor._kv_cache_context is None
        ), "an exception mid-pass left a processor pinning the KV-cache context"
        assert processor._gen_block_mask is None
    # The failed denoise pass wrote nothing: the cache is intact and the
    # rollout can continue once the fault is gone.
    monkeypatch.setattr(RolloutKvCache, "read_gen_history", original_read)
    assert cache.num_cached_chunks == 1 and cache.total_stored_frames == FRAMES_PER_CHUNK
    with torch.no_grad():
        _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[1], 1)


# --- context parallelism: rank-local staging (1/CP cache memory), grad-gated
# --- gathered reads, GLOBAL frame records — Wan's shipped cache geometry ---


def test_kv_cache_seed_pass_stages_rank_local_rows_and_commits_global_records(monkeypatch):
    """Wan-parity cache geometry (replaces the deleted global-staging pin): a
    CP seed pass stages THIS RANK's post-RoPE chunk rows BEFORE any
    collective — that is the 1/CP cache-memory contract — while the commit
    keeps GLOBAL frame records (the pass contract kv_cache_start_frame is a
    global chunk start feeding the mRoPE/opacity seams), so records and
    coverage are identical on every rank while the stored rows are each
    rank's own shard. Only the gen rows travel through the gather; text K/V
    are cached full-width, prepended outside it, and never gathered."""
    torch.manual_seed(0)
    world_size, frames_per_chunk, spatial_rows = 2, 2, 2
    attention = _ProcessorAttention(kv_heads=2)  # cosmos3's real GQA ratio
    und_seq = torch.randn(3, attention.hidden_size)
    global_chunk_seq = torch.randn(frames_per_chunk * spatial_rows, attention.hidden_size)
    local_rows = global_chunk_seq.shape[0] // world_size
    gathered_gen_rows = []

    def fake_all_gather_cat(tensor, mesh, dim):
        assert dim == 1 and tensor.shape[0] == 2
        gathered_gen_rows.append(tensor.shape[1])
        return torch.cat([tensor] * mesh.size(), dim=dim)

    # gather_kv_cat resolves the collectives from its own module, so the
    # fakes patch common_cp_module (the grad-mode dispatch still runs).
    monkeypatch.setattr(common_cp_module, "all_gather_cat", fake_all_gather_cat)

    caches = []
    for rank in range(world_size):
        processor = ArtifixerCosmos3AttnProcessor()
        processor._cp_mesh = _GroupedFakeCpMesh(rank=rank, size=world_size)
        cache = RolloutKvCache(1, truncate_rollout_gradients=False)
        stage = {}
        # The wrapper's row contract is the LOCAL shard count post-migration.
        processor._kv_cache_context = (cache, 0, 0, stage, local_rows)
        local_seq = global_chunk_seq[rank * local_rows : (rank + 1) * local_rows]
        with torch.no_grad():
            processor(attention, und_seq, local_seq, _identity_rotary(und_seq.shape[0], local_rows, attention.head_dim))

        _, expected_k, expected_v = cosmos3_projected_gen_qkv(attention, local_seq)
        staged_k, staged_v = stage[0]
        assert staged_k.shape[0] == local_rows, "seed staging must hold the rank-LOCAL rows, not the gathered chunk"
        torch.testing.assert_close(staged_k, expected_k, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(staged_v, expected_v, atol=1e-6, rtol=1e-6)
        # The wrapper commits GLOBAL frame records: the chunk's global start
        # and its FULL latent frame count, identical on every rank.
        cache.commit_staged_chunk(stage, 0, frames_per_chunk)
        caches.append(cache)

    # One paired K/V gather per rank, over the rank-local [history ⊕ own] gen
    # rows only — own rows alone on a fresh cache; the text K/V never enter.
    assert gathered_gen_rows == [local_rows] * world_size
    for cache in caches:
        assert cache._gen_chunk_frames[0] == [(0, frames_per_chunk)]
        assert cache.total_stored_frames == frames_per_chunk
        # The next chunk's GLOBAL start passes the coverage contract on
        # every rank — no per-rank coordinate mapping anywhere.
        cache.assert_pass_starts_at(frames_per_chunk)
    rank0_rows = caches[0].read_gen_history(0, 1)[0][0]
    rank1_rows = caches[1].read_gen_history(0, 1)[0][0]
    assert not torch.equal(rank0_rows, rank1_rows), "each rank must store its own shard's rows"

    # The row contract is LOCAL now: handing the processor the pre-migration
    # GLOBAL row count is loud misuse, not a silent geometry drift.
    stale_processor = ArtifixerCosmos3AttnProcessor()
    stale_processor._cp_mesh = _GroupedFakeCpMesh(rank=0, size=world_size)
    stale_processor._kv_cache_context = (
        RolloutKvCache(1, truncate_rollout_gradients=False),
        0,
        0,
        {},
        global_chunk_seq.shape[0],
    )
    with torch.no_grad(), pytest.raises(AssertionError, match="row"):
        stale_processor(
            attention,
            und_seq,
            global_chunk_seq[:local_rows],
            _identity_rotary(und_seq.shape[0], local_rows, attention.head_dim),
        )


def test_kv_cache_cp_gather_carries_grad_to_cached_history_and_projections(monkeypatch):
    """Replaces the deleted grad+CP rejection pin with the capability that
    superseded it (the Wan MR made the identical swap): a grad-enabled CP
    cache read gathers through the autograd-aware all_gather_cat_with_grad
    on the mesh group, so gradient from the attention output reaches BOTH
    the grad-carrying cached history (truncate_rollout_gradients=False) and
    this pass's K/V projections — the DMD rollout training path (mirror of
    Wan's processor-level grad pin in tests/test_cp_attention.py and
    tests/test_kv_cache_pipeline.py)."""
    torch.manual_seed(0)
    attention = _ProcessorAttention(kv_heads=2)
    mesh = _GroupedFakeCpMesh(rank=0, size=2)
    processor = ArtifixerCosmos3AttnProcessor()
    processor._cp_mesh = mesh
    cache = RolloutKvCache(1, truncate_rollout_gradients=False)
    k_history = torch.randn(4, attention.num_key_value_heads, attention.head_dim, requires_grad=True)
    v_history = torch.randn(4, attention.num_key_value_heads, attention.head_dim, requires_grad=True)
    cache.store_gen_chunk(0, 0, 0, 2, k_history, v_history)  # GLOBAL frames [0, 2)
    gather_groups = []

    def fake_all_gather_cat_with_grad(tensor, dim, group=None):
        assert dim == 1 and tensor.shape[0] == 2
        gather_groups.append(group)
        # Every-rank-holds-these-rows stand-in: pure cat keeps the gather on
        # the autograd graph, exactly like the real collective.
        return torch.cat([tensor, tensor], dim=dim)

    def no_grad_gather_must_not_run(tensor, mesh_arg, dim):
        pytest.fail("a grad-enabled CP cache read must use the grad-carrying gather")

    monkeypatch.setattr(common_cp_module, "all_gather_cat_with_grad", fake_all_gather_cat_with_grad)
    monkeypatch.setattr(common_cp_module, "all_gather_cat", no_grad_gather_must_not_run)

    und_seq = torch.randn(3, attention.hidden_size)
    gen_seq = torch.randn(2, attention.hidden_size)
    processor._kv_cache_context = (cache, 0, 1, None, gen_seq.shape[0])
    with torch.enable_grad():
        _, gen_out = processor(
            attention, und_seq, gen_seq, _identity_rotary(und_seq.shape[0], gen_seq.shape[0], attention.head_dim)
        )
        gen_out.square().sum().backward()

    assert gather_groups == [mesh._group], "one grad-carrying paired K/V gather on the mesh group"
    for source in (k_history, v_history, attention.add_k_proj.weight, attention.add_v_proj.weight):
        assert source.grad is not None
        assert torch.count_nonzero(source.grad) > 0


def test_kv_cache_cp_no_grad_read_gathers_rank_order_history_plus_own(monkeypatch):
    """Inference cache reads stay on the no-grad gather (all_gather_cat,
    whose loud grad-raise keeps guarding that no-grad-only helper) and
    attend [text ⊕ rank0(history ⊕ own) ⊕ rank1(history ⊕ own)]: text is
    prepended ONCE outside the gather, and the gathered key order is
    rank-interleaved — a permutation of the global temporal order (softmax
    is permutation-invariant) and exactly Wan's shipped cache layout."""
    torch.manual_seed(0)
    attention = _ProcessorAttention(kv_heads=2)
    processor = ArtifixerCosmos3AttnProcessor()
    processor._cp_mesh = _GroupedFakeCpMesh(rank=1, size=2)
    cache = RolloutKvCache(1, truncate_rollout_gradients=False)
    k_history = torch.randn(2, attention.num_key_value_heads, attention.head_dim)
    v_history = torch.randn(2, attention.num_key_value_heads, attention.head_dim)
    cache.store_gen_chunk(0, 0, 0, 2, k_history, v_history)

    def fake_all_gather_cat(tensor, mesh, dim):
        # Rank-order concatenation with the peer's slot marked: this caller
        # is rank 1, so the (marked) peer block must land FIRST.
        assert dim == 1 and tensor.shape[0] == 2 and mesh is processor._cp_mesh
        return torch.cat([tensor + 1000.0, tensor], dim=dim)

    def grad_gather_must_not_run(tensor, dim, group=None):
        pytest.fail("no-grad cache reads must stay on the no-grad gather")

    monkeypatch.setattr(common_cp_module, "all_gather_cat", fake_all_gather_cat)
    monkeypatch.setattr(common_cp_module, "all_gather_cat_with_grad", grad_gather_must_not_run)

    seen_keys = []
    original_dispatch = cosmos3_transformer_module.dispatch_attention_fn

    def recording_dispatch(query, key, value, **kwargs):
        seen_keys.append(key.detach().clone())
        return original_dispatch(query, key, value, **kwargs)

    monkeypatch.setattr(cosmos3_transformer_module, "dispatch_attention_fn", recording_dispatch)

    und_seq = torch.randn(3, attention.hidden_size)
    gen_seq = torch.randn(2, attention.hidden_size)
    processor._kv_cache_context = (cache, 0, 1, None, gen_seq.shape[0])
    with torch.no_grad():
        processor(attention, und_seq, gen_seq, _identity_rotary(und_seq.shape[0], gen_seq.shape[0], attention.head_dim))

    # The und causal call dispatches first; the gen call's keys carry the
    # full CP layout.
    gen_keys = seen_keys[-1].squeeze(0)
    k_und = attention.norm_k(attention.to_k(und_seq).view(-1, attention.num_key_value_heads, attention.head_dim))
    _, k_own, _ = cosmos3_projected_gen_qkv(attention, gen_seq)
    local_gen_cat = torch.cat([k_history, k_own], dim=0)
    expected_keys = torch.cat([k_und, local_gen_cat + 1000.0, local_gen_cat], dim=0)
    assert gen_keys.shape[0] == und_seq.shape[0] + 2 * (k_history.shape[0] + gen_seq.shape[0])
    torch.testing.assert_close(gen_keys, expected_keys, atol=1e-6, rtol=1e-6)


def test_kv_cache_cp_grad_enabled_forward_with_grad_free_kv_stays_valid(monkeypatch):
    """Mirror of Wan's frozen grad-enabled CP pin: grad MODE alone is not
    misuse — a grad-enabled cache pass whose K/V carry no grad (frozen
    projections, truncation-mode detached history) runs the grad-gated
    gather and completes, producing a grad-free output instead of raising."""
    torch.manual_seed(0)
    attention = _ProcessorAttention(kv_heads=2)
    attention.requires_grad_(False)
    processor = ArtifixerCosmos3AttnProcessor()
    processor._cp_mesh = _GroupedFakeCpMesh(rank=0, size=2)
    cache = RolloutKvCache(1, truncate_rollout_gradients=True)
    k_history = torch.randn(2, attention.num_key_value_heads, attention.head_dim)
    cache.store_gen_chunk(0, 0, 0, 2, k_history, k_history.clone())

    monkeypatch.setattr(
        common_cp_module,
        "all_gather_cat_with_grad",
        lambda tensor, dim, group=None: torch.cat([tensor, tensor], dim=dim),
    )

    und_seq = torch.randn(3, attention.hidden_size)
    gen_seq = torch.randn(2, attention.hidden_size)
    processor._kv_cache_context = (cache, 0, 1, None, gen_seq.shape[0])
    with torch.enable_grad():
        _, gen_out = processor(
            attention, und_seq, gen_seq, _identity_rotary(und_seq.shape[0], gen_seq.shape[0], attention.head_dim)
        )

    assert gen_out.shape == gen_seq.shape
    assert not gen_out.requires_grad


def test_cp_windowed_cache_rollout_matches_single_rank_rollout(monkeypatch):
    """CP=2 vs CP=1 equivalence across a windowed 3-chunk cached rollout at
    the processor seam (cosmos3's mirror of the Wan CP window pins in
    tests/test_kv_cache_pipeline.py): with an EXACT rank-aware gather fake,
    the two ranks' per-chunk outputs concatenated must equal the single-rank
    rollout at rtol 1e-4 / atol 1e-6 — never torch.equal, because the CP
    read's key order is rank-interleaved [text ⊕ rank_i(history ⊕ own)], a
    permutation of the global temporal order that changes the softmax
    reduction order. Eviction runs in cross-rank lockstep through the GLOBAL
    coverage records (window 2 chunks, no anchor: chunk 2 attends only chunk
    1's history on every side)."""
    torch.manual_seed(0)
    world_size, frames_per_chunk, spatial_rows, num_chunks = 2, 2, 2, 3
    window_chunks, sink_chunks = 2, 0
    attention = _ProcessorAttention(kv_heads=2)
    und_seq = torch.randn(3, attention.hidden_size)
    chunk_rows = frames_per_chunk * spatial_rows
    local_rows = chunk_rows // world_size
    chunks = [torch.randn(chunk_rows, attention.hidden_size) for _ in range(num_chunks)]

    def local_slice(chunk_index: int, rank: int) -> torch.Tensor:
        # The full-input contract shards each pass's chunk by latent frame:
        # rank r holds frame r of the 2-frame chunk (its 2 spatial rows).
        return chunks[chunk_index][rank * local_rows : (rank + 1) * local_rows]

    def run_single_rank() -> list[torch.Tensor]:
        processor = ArtifixerCosmos3AttnProcessor()
        cache = RolloutKvCache(1, truncate_rollout_gradients=False)
        outputs = []
        with torch.no_grad():
            for chunk_index in range(num_chunks):
                cache.evict_to_window(window_chunks, sink_chunks)
                stage = {}
                processor._kv_cache_context = (cache, 0, cache.num_cached_chunks, stage, chunk_rows)
                _, gen_out = processor(
                    attention,
                    und_seq,
                    chunks[chunk_index],
                    _identity_rotary(und_seq.shape[0], chunk_rows, attention.head_dim),
                )
                outputs.append(gen_out)
                cache.commit_staged_chunk(stage, chunk_index * frames_per_chunk, frames_per_chunk)
        return outputs

    expected_outputs = run_single_rank()

    # CP=2: both ranks advance chunk by chunk in lockstep. The gather fake is
    # exact: the peer's [history ⊕ own] comes from the peer's cache plus the
    # shared projections, the caller's own slot is the actual gathered
    # tensor, concatenated in rank order like the real collective.
    caches = [RolloutKvCache(1, truncate_rollout_gradients=False) for _ in range(world_size)]
    state = {"chunk": 0}

    def peer_gen_cat(kind: str, peer: int) -> torch.Tensor:
        k_history, v_history = caches[peer].read_gen_history(0, caches[peer].num_cached_chunks)
        _, own_k, own_v = cosmos3_projected_gen_qkv(attention, local_slice(state["chunk"], peer))
        if kind == "k":
            return torch.cat([*k_history, own_k], dim=0)
        return torch.cat([*v_history, own_v], dim=0)

    def fake_all_gather_cat(tensor, mesh, dim):
        assert dim == 1 and tensor.shape[0] == 2
        rank = mesh.get_local_rank()
        slots = [
            tensor if peer == rank else torch.stack([peer_gen_cat("k", peer), peer_gen_cat("v", peer)])
            for peer in range(world_size)
        ]
        return torch.cat(slots, dim=dim)

    monkeypatch.setattr(common_cp_module, "all_gather_cat", fake_all_gather_cat)

    processors = []
    for rank in range(world_size):
        processor = ArtifixerCosmos3AttnProcessor()
        processor._cp_mesh = _GroupedFakeCpMesh(rank=rank, size=world_size)
        processors.append(processor)

    with torch.no_grad():
        for chunk_index in range(num_chunks):
            state["chunk"] = chunk_index
            for cache in caches:
                cache.evict_to_window(window_chunks, sink_chunks)
            num_history_chunks = caches[0].num_cached_chunks
            rank_outputs = []
            stages = []
            for rank in range(world_size):
                stage = {}
                processors[rank]._kv_cache_context = (caches[rank], 0, num_history_chunks, stage, local_rows)
                _, gen_out = processors[rank](
                    attention,
                    und_seq,
                    local_slice(chunk_index, rank),
                    _identity_rotary(und_seq.shape[0], local_rows, attention.head_dim),
                )
                rank_outputs.append(gen_out)
                stages.append(stage)
            for rank in range(world_size):
                # GLOBAL frame records on every rank; rows are the shard.
                caches[rank].commit_staged_chunk(stages[rank], chunk_index * frames_per_chunk, frames_per_chunk)
            torch.testing.assert_close(
                torch.cat(rank_outputs, dim=0),
                expected_outputs[chunk_index],
                rtol=1e-4,
                atol=1e-6,
                msg=lambda base, i=chunk_index: f"chunk {i}: CP=2 rollout diverged from CP=1\n{base}",
            )

    # Lockstep eviction happened on both sides: chunk 2 read only chunk 1,
    # and the retained records are identical (and global) on every rank.
    for cache in caches:
        assert cache._gen_chunk_frames[0] == [(2, 2), (4, 2)]


def test_cache_path_accepts_chunk_divisor_only_cp_sizes(monkeypatch):
    """Path scoping of the CP-size constraints, cache side: production CP=7
    rollouts shard 7-frame chunks one latent frame per rank, and 7 divides
    neither 4 query heads nor 2 KV heads — the Ulysses head-divisibility
    asserts are scoped to the NON-cache branch (their raise is pinned in
    tests/test_cosmos3_transformer.py), so the cache pass must run, stage its
    local rows, and read [text ⊕ 7 rank slots]."""
    torch.manual_seed(0)
    cp_size = 7
    attention = _ProcessorAttention(kv_heads=2)
    processor = ArtifixerCosmos3AttnProcessor()
    processor._cp_mesh = _GroupedFakeCpMesh(rank=3, size=cp_size)
    cache = RolloutKvCache(1, truncate_rollout_gradients=False)

    monkeypatch.setattr(
        common_cp_module,
        "all_gather_cat",
        lambda tensor, mesh, dim: torch.cat([tensor] * cp_size, dim=dim),
    )
    seen_key_lengths = []
    original_dispatch = cosmos3_transformer_module.dispatch_attention_fn

    def recording_dispatch(query, key, value, **kwargs):
        seen_key_lengths.append(key.shape[1])
        return original_dispatch(query, key, value, **kwargs)

    monkeypatch.setattr(cosmos3_transformer_module, "dispatch_attention_fn", recording_dispatch)

    und_seq = torch.randn(3, attention.hidden_size)
    local_rows = 2  # one latent frame per rank at 2 spatial tokens
    gen_seq = torch.randn(local_rows, attention.hidden_size)
    stage = {}
    processor._kv_cache_context = (cache, 0, 0, stage, local_rows)
    with torch.no_grad():
        _, gen_out = processor(
            attention, und_seq, gen_seq, _identity_rotary(und_seq.shape[0], local_rows, attention.head_dim)
        )

    assert gen_out.shape == gen_seq.shape
    assert seen_key_lengths[-1] == und_seq.shape[0] + cp_size * local_rows
    assert stage[0][0].shape[0] == local_rows


# --- padded chunks: chunk sizes that do NOT divide the CP size pad the frame
# --- axis, exchange heads like the dense path, and cache each chunk by head
# --- group (full chunk per rank, one share of the heads) with LOCAL reads ---


def test_padded_chunk_cache_rollout_matches_single_rank_rollout(monkeypatch):
    """CP=2 vs CP=1 equivalence for the padded cache branch: 3-frame chunks
    pad to 4 on a 2-rank mesh (3 does not divide 2), each pass trades its
    padded frame shard for the FULL chunk with a share of the heads, drops
    the pad tail from K/V, stages the real chunk rows for this rank's head
    group, and reads history LOCALLY — any gather is a bug, and keys stay in
    GLOBAL temporal order (no rank interleaving), so per-chunk outputs must
    match the CP=1 rollout tightly (splitting heads cannot change any one
    head's math). Pad slots in the exchange fakes are zeros on purpose: pad
    rows must not be able to influence any real row."""
    torch.manual_seed(0)
    world_size, frames_per_chunk, spatial_rows, num_chunks = 2, 3, 2, 2
    padded_frames = 4
    attention = _ProcessorAttention(kv_heads=2)
    und_seq = torch.randn(3, attention.hidden_size)
    chunk_rows = frames_per_chunk * spatial_rows
    padded_rows = padded_frames * spatial_rows
    pad_tokens = padded_rows - chunk_rows
    local_rows = padded_rows // world_size
    chunks = [torch.randn(chunk_rows, attention.hidden_size) for _ in range(num_chunks)]
    # The wrapper feeds each rank its shard of the frame-padded chunk; the
    # pad rows are zeros here (their values are irrelevant by contract).
    padded_chunks = [torch.cat([chunk, chunk.new_zeros(pad_tokens, attention.hidden_size)]) for chunk in chunks]

    def run_single_rank() -> list[torch.Tensor]:
        processor = ArtifixerCosmos3AttnProcessor()
        cache = RolloutKvCache(1, truncate_rollout_gradients=False)
        outputs = []
        with torch.no_grad():
            for chunk_index in range(num_chunks):
                stage = {}
                processor._kv_cache_context = (cache, 0, cache.num_cached_chunks, stage, chunk_rows)
                _, gen_out = processor(
                    attention,
                    und_seq,
                    chunks[chunk_index],
                    _identity_rotary(und_seq.shape[0], chunk_rows, attention.head_dim),
                )
                outputs.append(gen_out)
                cache.commit_staged_chunk(stage, chunk_index * frames_per_chunk, frames_per_chunk)
        return outputs

    expected_outputs = run_single_rank()
    # to_add_out is identity in the double, so each expected output IS the
    # full-head attention output — what the reverse-exchange peers hold
    # (extended with zero pad rows).
    padded_attention_outs = [
        torch.cat(
            [
                out.view(-1, attention.num_attention_heads, attention.head_dim),
                out.new_zeros(pad_tokens, attention.num_attention_heads, attention.head_dim),
            ]
        )
        for out in expected_outputs
    ]

    def gather_must_not_run(*_args, **_kwargs):
        pytest.fail("head-sharded cache reads are LOCAL: no gather may run on the padded branch")

    monkeypatch.setattr(common_cp_module, "all_gather_cat", gather_must_not_run)
    monkeypatch.setattr(common_cp_module, "all_gather_cat_with_grad", gather_must_not_run)

    state = {"chunk": 0}

    def all_to_all_from_reference(rank: int):
        calls = {"forward": 0, "reverse": 0}

        def fake_all_to_all(tensor, scatter_dim, gather_dim, group=None):
            if scatter_dim == 1:
                assert gather_dim == 0
                kind = calls["forward"] % 3
                calls["forward"] += 1
                peers = []
                for peer in range(world_size):
                    rows = padded_chunks[state["chunk"]][peer * local_rows : (peer + 1) * local_rows]
                    peers.append(cosmos3_projected_gen_qkv(attention, rows)[kind])
            else:
                assert (scatter_dim, gather_dim) == (0, 1)
                calls["reverse"] += 1
                peers = list(padded_attention_outs[state["chunk"]].chunk(world_size, dim=1))
            return cosmos3_rankwise_all_to_all(
                tensor, peers, rank=rank, world_size=world_size, scatter_dim=scatter_dim, gather_dim=gather_dim
            )

        return fake_all_to_all

    caches = [RolloutKvCache(1, truncate_rollout_gradients=False) for _ in range(world_size)]
    processors = []
    for rank in range(world_size):
        processor = ArtifixerCosmos3AttnProcessor()
        processor._cp_mesh = _GroupedFakeCpMesh(rank=rank, size=world_size)
        # A nonzero pad count IS the head-sharded switch on cache passes.
        processor._cp_pad_tokens = pad_tokens
        processors.append(processor)

    fakes = [all_to_all_from_reference(rank) for rank in range(world_size)]
    with torch.no_grad():
        for chunk_index in range(num_chunks):
            state["chunk"] = chunk_index
            rank_outputs = []
            for rank in range(world_size):
                monkeypatch.setattr(cosmos3_transformer_module, "all_to_all_with_grad", fakes[rank])
                stage = {}
                processors[rank]._kv_cache_context = (
                    caches[rank],
                    0,
                    caches[rank].num_cached_chunks,
                    stage,
                    local_rows,
                )
                _, gen_out = processors[rank](
                    attention,
                    und_seq,
                    padded_chunks[chunk_index][rank * local_rows : (rank + 1) * local_rows],
                    _identity_rotary(und_seq.shape[0], local_rows, attention.head_dim),
                )
                rank_outputs.append(gen_out)
                # Staged rows are the FULL chunk's REAL rows (pads dropped)
                # with this rank's KV-head group.
                staged_k, staged_v = stage[0]
                assert staged_k.shape == (chunk_rows, attention.num_key_value_heads // world_size, attention.head_dim)
                assert staged_v.shape == staged_k.shape
                # Commits keep REAL frame records, so coverage bookkeeping
                # is identical to CP=1.
                caches[rank].commit_staged_chunk(stage, chunk_index * frames_per_chunk, frames_per_chunk)
            # Concatenated rank shards, pad frames stripped, must equal CP=1.
            cp_rows = torch.cat(rank_outputs, dim=0)[:chunk_rows]
            torch.testing.assert_close(
                cp_rows,
                expected_outputs[chunk_index],
                rtol=1e-5,
                atol=1e-6,
                msg=lambda base, i=chunk_index: f"chunk {i}: padded CP rollout diverged from CP=1\n{base}",
            )

    for cache in caches:
        # REAL frame records, identical to CP=1: pad frames never enter the
        # coverage bookkeeping.
        assert cache._gen_chunk_frames[0] == [(0, frames_per_chunk), (frames_per_chunk, frames_per_chunk)]
        assert cache.total_stored_frames == num_chunks * frames_per_chunk
        # The text cache holds this rank's share of the heads, not the full
        # width.
        cached_text_k, _ = cache.read_text_kv(0)
        assert cached_text_k.shape == (
            und_seq.shape[0],
            attention.num_key_value_heads // world_size,
            attention.head_dim,
        )


def test_padded_chunk_cache_pass_rejects_head_counts_not_divisible_by_cp_size(monkeypatch):
    """The padded cache branch splits the attention heads across the CP
    group, so — unlike the frame-sharded branch, which has no head
    constraint — it must reject a CP size that divides either head count
    unevenly, before any collective runs."""

    def collective_must_not_run(*_args, **_kwargs):
        pytest.fail("the head-divisibility asserts must fire before any collective")

    monkeypatch.setattr(cosmos3_transformer_module, "all_to_all_with_grad", collective_must_not_run)
    attention = _ProcessorAttention(kv_heads=2)  # 4 query heads, 2 KV heads
    processor = ArtifixerCosmos3AttnProcessor()
    processor._cp_mesh = _GroupedFakeCpMesh(rank=0, size=3)
    processor._cp_pad_tokens = 2
    und_seq = torch.randn(3, attention.hidden_size)
    gen_seq = torch.randn(2, attention.hidden_size)
    processor._kv_cache_context = (RolloutKvCache(1, truncate_rollout_gradients=False), 0, 0, {}, gen_seq.shape[0])

    with torch.no_grad(), pytest.raises(AssertionError, match="divisible"):
        processor(attention, und_seq, gen_seq, _identity_rotary(und_seq.shape[0], gen_seq.shape[0], attention.head_dim))


def test_cache_pass_rejects_pad_tokens_without_cp_mesh():
    """Pad rows only exist on head-exchange CP paths; a cache pass that
    somehow carries a pad count with no active mesh is a wrapper bug and
    must fail loudly instead of caching rows that other passes would
    misindex."""
    attention = _ProcessorAttention(kv_heads=2)
    processor = ArtifixerCosmos3AttnProcessor()
    processor._cp_pad_tokens = 2
    und_seq = torch.randn(3, attention.hidden_size)
    gen_seq = torch.randn(2, attention.hidden_size)
    processor._kv_cache_context = (RolloutKvCache(1, truncate_rollout_gradients=False), 0, 0, {}, gen_seq.shape[0])

    with torch.no_grad(), pytest.raises(AssertionError, match="pad"):
        processor(attention, und_seq, gen_seq, _identity_rotary(und_seq.shape[0], gen_seq.shape[0], attention.head_dim))


# --- the sliding attention window: eviction is driven through the coverage
# --- bookkeeping, on the real reduced-config transformer, at the production
# --- chunk size and window options ---

WINDOW_CHUNK_FRAMES = 7
# Explicit option values at the production defaults: a 21-frame window with
# a 7-frame anchor at 7-frame chunks -> 3 window chunks, 1 anchor chunk.
WINDOW_LOCAL_ATTN_SIZE = 21
WINDOW_SINK_SIZE = 7
WINDOW_CHUNKS = WINDOW_LOCAL_ATTN_SIZE // WINDOW_CHUNK_FRAMES
ANCHOR_CHUNKS = WINDOW_SINK_SIZE // WINDOW_CHUNK_FRAMES
WINDOW_HISTORY_FRAMES = WINDOW_LOCAL_ATTN_SIZE - WINDOW_CHUNK_FRAMES
WINDOW_NUM_CHUNKS = 5  # 35-frame synthetic sequence, longer than the window
# 4x4 latent grid, patch 2 -> 2x2 patch tokens per latent frame.
SPATIAL_TOKENS_PER_FRAME = 4


def test_kv_cache_windowed_rollout_attends_anchor_plus_window_and_evicts_by_coverage(monkeypatch):
    """Windowed cached rollout over a 35-frame synthetic sequence in 7-frame
    chunks with a 21-frame attention window and a 7-frame anchor: before
    each denoise pass, seeded history is evicted down to 2 chunks — the
    anchor chunk plus the most recent one (the SAME rule
    generate_samples_from_batch applies: retain at most
    WINDOW_CHUNKS - 1 chunks, evicting after the ANCHOR_CHUNKS anchor) — so
    every layer's gen attention reads keys over EXACTLY [text + 14 cached
    frames + own 7] — asserted at the attention-dispatch level, per layer,
    per chunk (per-frame-distinct cameras keep the absolute-position seams
    load-bearing). Frame eviction is verified through the coverage
    bookkeeping: retained records keep the anchor plus the rolling tail
    while ``total_stored_frames`` keeps the ABSOLUTE coverage end, so the
    write-once/contiguity contracts still reject duplicate and gapped passes
    after eviction. At the training length (3 chunks = the window) the rule
    never evicts."""
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    total_frames = WINDOW_NUM_CHUNKS * WINDOW_CHUNK_FRAMES
    kwargs = _rollout_kwargs(total_frames)
    # Isolate target-history eviction from reference attention.
    kwargs["neighbor_hidden_states"] = None
    kwargs["neighbor_w2cs"] = None
    kwargs["neighbor_Ks"] = None
    torch.manual_seed(43)
    noisy = [torch.randn(1, 48, WINDOW_CHUNK_FRAMES, 4, 4) for _ in range(WINDOW_NUM_CHUNKS)]
    clean = [torch.randn(1, 48, WINDOW_CHUNK_FRAMES, 4, 4) for _ in range(WINDOW_NUM_CHUNKS)]
    cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
    num_layers = len(transformer.base_transformer.layers)
    chunk_gen_rows = WINDOW_CHUNK_FRAMES * SPATIAL_TOKENS_PER_FRAME

    dispatch_shapes: list[tuple[int, int]] = []
    original_dispatch = cosmos3_transformer_module.dispatch_attention_fn

    def recording_dispatch(query, key, value, **dispatch_kwargs):
        dispatch_shapes.append((query.shape[1], key.shape[1]))
        # Cache passes are mask-free (causality via cache content), keeping
        # every dispatch flash-eligible — a mask would silently force the
        # SDPA fallback on flash backends.
        assert dispatch_kwargs.get("attn_mask") is None, "cache passes must dispatch without an attention mask"
        return original_dispatch(query, key, value, **dispatch_kwargs)

    monkeypatch.setattr(cosmos3_transformer_module, "dispatch_attention_fn", recording_dispatch)

    # Retained records before each chunk's passes: chunks 0-2 fill the
    # window; chunks 3 and 4 see the anchor (0, 7) plus the most recent
    # seeded chunk — eviction drops the oldest chunk AFTER the anchor.
    expected_records_per_chunk = [
        (),
        ((0, 7),),
        ((0, 7), (7, 7)),
        ((0, 7), (14, 7)),
        ((0, 7), (21, 7)),
    ]

    with torch.no_grad():
        for chunk_index in range(WINDOW_NUM_CHUNKS):
            start = chunk_index * WINDOW_CHUNK_FRAMES
            # The pipeline's eviction rule, verbatim.
            cache.evict_to_window(WINDOW_CHUNKS, ANCHOR_CHUNKS)
            expected_history_frames = min(start, WINDOW_HISTORY_FRAMES)
            # Eviction verified via the coverage bookkeeping: the retained
            # records are the anchor plus the rolling tail of seeded chunks,
            # and absolute coverage never rewinds.
            for layer_records in cache._gen_chunk_frames:
                assert tuple(layer_records) == expected_records_per_chunk[chunk_index]
                assert sum(frames for _, frames in layer_records) == expected_history_frames
            assert cache.total_stored_frames == start

            marker = len(dispatch_shapes)
            out = _cache_chunk_forward(transformer, kwargs, cache, noisy[chunk_index], chunk_index, WINDOW_CHUNK_FRAMES)
            assert out.shape == (1, 48, WINDOW_CHUNK_FRAMES, 4, 4)
            # The trimmed-prompt text K/V (prompt tokens + specials) are
            # cached on the first pass; their row count is the text segment
            # every gen row attends.
            text_rows = cache.read_text_kv(0)[0].shape[0]
            assert text_rows != chunk_gen_rows  # keeps the q-row filter below unambiguous
            gen_key_rows = [k_rows for q_rows, k_rows in dispatch_shapes[marker:] if q_rows == chunk_gen_rows]
            expected_key_rows = text_rows + (expected_history_frames + WINDOW_CHUNK_FRAMES) * SPATIAL_TOKENS_PER_FRAME
            assert gen_key_rows == [expected_key_rows] * num_layers, (
                f"chunk {chunk_index}: gen attention must read exactly [text ({text_rows} rows) + last "
                f"{expected_history_frames} cached frames + own {WINDOW_CHUNK_FRAMES}] = "
                f"{expected_key_rows} key rows on every layer, saw {gen_key_rows}"
            )
            if chunk_index < WINDOW_NUM_CHUNKS - 1:
                _seed_chunk(transformer, kwargs, cache, clean[chunk_index], chunk_index, WINDOW_CHUNK_FRAMES)

    # Coverage still counts evicted frames, so post-eviction misuse stays
    # loud: re-seeding an already-covered (and partly evicted) region is
    # rejected against ABSOLUTE coverage, not the shrunken retained set.
    with (
        torch.no_grad(),
        pytest.raises(AssertionError, match=r"seed pass starts at latent frame 21.*stored frames \[0, 28\)"),
    ):
        _seed_chunk(transformer, kwargs, cache, clean[3], 3, WINDOW_CHUNK_FRAMES)
    # Evicting past the retained history is a driver bug, never a silent
    # no-op: once only the anchor remains, evicting after it raises.
    while cache.num_cached_chunks > ANCHOR_CHUNKS:
        cache.evict_oldest_gen_chunk(ANCHOR_CHUNKS)
    with pytest.raises(AssertionError, match="no retained gen chunk to evict"):
        cache.evict_oldest_gen_chunk(ANCHOR_CHUNKS)


# --- pipeline-level oracle: the full AR rollout through
# --- ArtifixerCosmos3KvCachePipeline.generate_samples_from_batch ---


def _real_kv_cache_pipe(truncate_rollout_gradients: bool = False) -> ArtifixerCosmos3KvCachePipeline:
    torch.manual_seed(0)
    pipe = ArtifixerCosmos3KvCachePipeline(
        vae=_TinyCosmosVae(),
        transformer=Cosmos3OmniTransformer(**REDUCED_CONFIG),
        gradient_checkpointing=False,
        # Pin the diffusers dispatch backend: the expected_outputs were generated
        # under native dispatch + MATH SDPA.
        attention_backend="native",
        ar_frames_per_chunk=FRAMES_PER_CHUNK,
        local_attn_size=LOCAL_ATTN_SIZE,
        sink_size=SINK_SIZE,
        truncate_rollout_gradients=truncate_rollout_gradients,
    )
    _activate_conditioning_adapters(pipe.transformer)
    return pipe


def _pipeline_inputs() -> dict:
    kwargs = _rollout_kwargs()
    torch.manual_seed(31)
    return dict(
        condition=torch.randn(1, 48, TOTAL_FRAMES, 4, 4),
        rendered_opacity=kwargs["opacity"],
        neighbors_condition=kwargs["neighbor_hidden_states"],
        camera_rays=kwargs["camera_rays"],
        neighbor_camera_rays=kwargs["neighbor_camera_rays"],
        w2cs=kwargs["w2cs"],
        neighbor_w2cs=kwargs["neighbor_w2cs"],
        Ks=kwargs["Ks"],
        neighbor_Ks=kwargs["neighbor_Ks"],
        encoded_prompt=kwargs["encoder_hidden_states"],
        prompt_attention_mask=kwargs["encoder_attention_mask"],
    )


def _run_pipeline_rollout(
    pipe: ArtifixerCosmos3KvCachePipeline,
    inputs: dict,
    num_inference_steps: int = 3,
    use_exit_flag: bool = False,
) -> torch.Tensor:
    # The seed the expected_outputs were generated under: the z_mix draw sequence (one
    # at chunk start + one per SDE re-noise; seed passes draw nothing) is
    # part of what the expected_outputs pin.
    torch.manual_seed(1234)
    return pipe.generate_samples_from_batch(
        inputs["condition"],
        inputs["rendered_opacity"],
        inputs["neighbors_condition"],
        inputs["camera_rays"],
        inputs["w2cs"],
        inputs["neighbor_w2cs"],
        inputs["Ks"],
        inputs["neighbor_Ks"],
        inputs["encoded_prompt"],
        num_inference_steps,
        use_exit_flag,
        prompt_attention_mask=inputs["prompt_attention_mask"],
        neighbor_camera_rays=inputs["neighbor_camera_rays"],
    )


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_pipeline_rollout_matches_the_stored_expected_outputs(expected_outputs, truncate_rollout_gradients):
    """THE oracle gate: a full multi-chunk rollout on the real reduced-config
    transformer — 3 chunks, camera_rays + neighbors active, deterministic
    seeds, pinned native + MATH SDPA backend — reproduces the checked-in
    expected-output videos, which were measured bit-identical (per chunk, full video,
    and RNG end state) to the truncated-BPTT full-prefix oracle. Verified
    under no_grad, in the ambient-grad regime the trainer uses, and with
    exit-flag truncation active (the exit draw and the truncated step list
    change the z_mix draw sequence; the exit-flag expected-output pins that too).
    Both gradient storage modes reproduce the same videos: the mode moves
    graphs and memory, never values or RNG draws (seed passes draw nothing)."""
    inputs = _pipeline_inputs()

    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        video = _run_pipeline_rollout(_real_kv_cache_pipe(truncate_rollout_gradients), inputs)
    assert video.shape == inputs["condition"].shape
    assert bool(torch.isfinite(video).all())
    # The final video is the cat of per-chunk x0s, so equality is asserted
    # chunk by chunk first (localizes a divergence to its chunk).
    expected_video = expected_outputs["pipeline_video_steps3"]
    for chunk_index in range(NUM_CHUNKS):
        chunk = slice(chunk_index * FRAMES_PER_CHUNK, (chunk_index + 1) * FRAMES_PER_CHUNK)
        torch.testing.assert_close(
            video[:, :, chunk],
            expected_video[:, :, chunk],
            rtol=EXPECTED_RTOL,
            atol=EXPECTED_ATOL,
            msg=lambda base, i=chunk_index: f"chunk {i}: x0 diverged from the expected-output oracle\n{base}",
        )
    torch.testing.assert_close(video, expected_video, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)

    # Ambient grad (the trainer's regime: exit steps build graphs).
    with torch.enable_grad(), sdpa_kernel([SDPBackend.MATH]):
        video = _run_pipeline_rollout(_real_kv_cache_pipe(truncate_rollout_gradients), inputs, num_inference_steps=2)
    assert video.requires_grad
    torch.testing.assert_close(
        video.detach(), expected_outputs["pipeline_video_steps2_grad"], rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL
    )

    # Exit-flag truncation (one rank-synced draw before the chunk loop).
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        video = _run_pipeline_rollout(_real_kv_cache_pipe(truncate_rollout_gradients), inputs, use_exit_flag=True)
    torch.testing.assert_close(
        video, expected_outputs["pipeline_video_steps3_exit_flag"], rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL
    )


def test_default_mode_rollout_grads_match_with_and_without_gradient_checkpointing():
    """Default-mode seed passes run under grad, so their layers re-run at
    backward-time recompute; staging keeps the re-run from re-storing (the
    cache's write-once contract would trip) and the snapshotted history count
    keeps its reads exact. Pin: parameter gradients of a full default-mode
    rollout are bitwise equal with and without checkpointing."""
    inputs = _pipeline_inputs()

    def run(gradient_checkpointing: bool) -> dict[str, torch.Tensor | None]:
        torch.manual_seed(0)
        pipe = ArtifixerCosmos3KvCachePipeline(
            vae=_TinyCosmosVae(),
            transformer=Cosmos3OmniTransformer(**REDUCED_CONFIG),
            gradient_checkpointing=gradient_checkpointing,
            attention_backend="native",
            ar_frames_per_chunk=FRAMES_PER_CHUNK,
            local_attn_size=LOCAL_ATTN_SIZE,
            sink_size=SINK_SIZE,
            truncate_rollout_gradients=False,
        )
        _activate_conditioning_adapters(pipe.transformer)
        with torch.enable_grad(), sdpa_kernel([SDPBackend.MATH]):
            video = _run_pipeline_rollout(pipe, inputs, num_inference_steps=2)
            video.square().sum().backward()
        return {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in pipe.transformer.named_parameters()
            if parameter.requires_grad
        }

    plain_grads = run(gradient_checkpointing=False)
    checkpointed_grads = run(gradient_checkpointing=True)
    assert plain_grads.keys() == checkpointed_grads.keys()
    for name, plain_grad in plain_grads.items():
        checkpointed_grad = checkpointed_grads[name]
        if plain_grad is None:
            assert checkpointed_grad is None, name
            continue
        assert checkpointed_grad is not None and torch.equal(
            plain_grad, checkpointed_grad
        ), f"{name}: checkpointed backward diverged on the default-mode rollout"


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_kv_cache_rollout_seeds_each_finished_chunk_once_except_the_last(monkeypatch, truncate_rollout_gradients):
    """Pipeline plumbing: per chunk, denoise passes carry only the
    chunk (cache kwargs at the chunk's absolute start); each finished chunk
    EXCEPT the global last is seeded exactly once — conditioning-mode
    (timestep None), chunk-only conditioning slices. The seed pass's grad
    mode follows the truncation flag: truncation seeds under no_grad on the
    detached x0; the default seeds in the ambient grad mode on the
    graph-carrying x0."""
    captured = []
    original_forward = ArtifixerCosmos3Transformer.forward

    def spy_forward(self, **kwargs):
        captured.append({**kwargs, "grad_enabled": torch.is_grad_enabled()})
        return original_forward(self, **kwargs)

    monkeypatch.setattr(ArtifixerCosmos3Transformer, "forward", spy_forward)
    pipe = _real_kv_cache_pipe(truncate_rollout_gradients)
    inputs = _pipeline_inputs()

    with torch.enable_grad():
        _run_pipeline_rollout(pipe, inputs, num_inference_steps=2)

    # [denoise, exit-denoise, seed] per chunk; the last chunk is not seeded.
    assert len(captured) == NUM_CHUNKS * 2 + (NUM_CHUNKS - 1)
    assert [call.get("kv_cache_seed", False) for call in captured] == [
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    seed_grad_enabled = not truncate_rollout_gradients
    assert [call["grad_enabled"] for call in captured] == [
        False,
        True,
        seed_grad_enabled,
        False,
        True,
        seed_grad_enabled,
        False,
        True,
    ]
    for call_index, call in enumerate(captured):
        chunk_index = [0, 0, 0, 1, 1, 1, 2, 2][call_index]
        chunk_start = chunk_index * FRAMES_PER_CHUNK
        chunk_end = chunk_start + FRAMES_PER_CHUNK
        assert call["kv_cache_start_frame"] == chunk_start
        assert call["hidden_states"].shape[2] == FRAMES_PER_CHUNK
        # Chunk-only conditioning at absolute positions.
        assert torch.equal(call["camera_rays"], inputs["camera_rays"][:, chunk_start:chunk_end])
        assert torch.equal(call["w2cs"], inputs["w2cs"][:, chunk_start:chunk_end])
        pixel_start = 0 if chunk_start == 0 else (chunk_start - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1
        assert torch.equal(
            call["opacity"],
            inputs["rendered_opacity"][:, pixel_start : (chunk_end - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1],
        )
        if call.get("kv_cache_seed", False):
            assert call["timestep"] is None
            if truncate_rollout_gradients:
                assert (
                    call["hidden_states"].grad_fn is None and not call["hidden_states"].requires_grad
                ), "truncation-mode seed passes must receive the finished chunk's x0 DETACHED"
            else:
                assert call[
                    "hidden_states"
                ].requires_grad, "default-mode seed passes must receive the chunk's graph-carrying x0"


def test_every_rollout_gets_one_fresh_cache_and_training_length_never_evicts(monkeypatch):
    """Cache lifetime and window inactivity at training length: each
    generate_samples_from_batch call constructs exactly ONE fresh
    RolloutKvCache (never reused across calls — positions, prompt,
    and conditioning are rollout-specific), and a 3-chunk rollout (inside
    the attention window) evicts nothing: the cache stays full-history
    write-once, so training semantics are windowless."""
    constructed = []
    original_new = ArtifixerCosmos3Transformer.new_rollout_kv_cache

    def counting_new(self, **kwargs):
        cache = original_new(self, **kwargs)
        constructed.append(cache)
        return cache

    evictions = []
    original_evict = RolloutKvCache.evict_oldest_gen_chunk

    def counting_evict(self, num_anchor_chunks=0):
        evictions.append(self)
        return original_evict(self, num_anchor_chunks)

    monkeypatch.setattr(ArtifixerCosmos3Transformer, "new_rollout_kv_cache", counting_new)
    monkeypatch.setattr(RolloutKvCache, "evict_oldest_gen_chunk", counting_evict)
    pipe = _real_kv_cache_pipe()
    inputs = _pipeline_inputs()

    with torch.no_grad():
        _run_pipeline_rollout(pipe, inputs, num_inference_steps=1)
        _run_pipeline_rollout(pipe, inputs, num_inference_steps=1)

    assert len(constructed) == 2, "each rollout must construct exactly one fresh cache"
    assert constructed[0] is not constructed[1]
    assert not evictions, "a training-length rollout (window >= clip) must never evict"
    # Every seeded chunk (all but the global last) is still retained.
    assert constructed[-1].num_cached_chunks == NUM_CHUNKS - 1


# SCoPE when saving and reusing earlier frames.


def test_scope_delta_reaches_the_cached_keys():
    """Camera strength must affect the keys saved for use by later video chunks."""
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    kwargs = _rollout_kwargs()
    _, clean_chunks = _chunk_tensors(seed=7)

    def cached_keys(alpha: float) -> torch.Tensor:
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
            for injection in transformer.scope:
                injection.alpha.fill_(alpha)
            cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
            _seed_chunk(transformer, kwargs, cache, clean_chunks[0], 0)
            keys, _ = (segments[0] for segments in cache.read_gen_history(0, 1))
            return keys.clone()

    scoped = cached_keys(0.5)
    unscoped = cached_keys(0.0)

    assert scoped.shape == unscoped.shape
    assert not torch.allclose(scoped, unscoped), "the ray delta never reached the cached keys"


def test_scope_completes_a_full_cached_rollout():
    """Generating three chunks with active camera inputs must finish with finite outputs."""
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    with torch.no_grad():
        for injection in transformer.scope:
            injection.alpha.fill_(0.5)
    kwargs = _rollout_kwargs()
    noisy_chunks, clean_chunks = _chunk_tensors(seed=7)

    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
        for chunk_index in range(NUM_CHUNKS):
            out = _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[chunk_index], chunk_index)
            assert out.shape == (1, 48, FRAMES_PER_CHUNK, 4, 4)
            assert bool(torch.isfinite(out).all()), f"chunk {chunk_index} produced non-finite output"
            if chunk_index < NUM_CHUNKS - 1:
                _seed_chunk(transformer, kwargs, cache, clean_chunks[chunk_index], chunk_index)


def test_real_teacher_forcing_with_references_matches_cached_rollout():
    """The block-causal oracle must use CLEAN history, not timestep-conditioned noisy history.

    Compare a real clean_x weave with real cache denoise/seed passes. This is
    the real-model counterpart of test_teacher_forcing_forward_matches_kv_cache_rollout;
    references and nonzero camera/opacity adapters stay active in both modes.
    """
    total_frames = 2 * FRAMES_PER_CHUNK
    spatial_tokens = (4 // REDUCED_CONFIG["latent_patch_size"]) ** 2
    kwargs = _rollout_kwargs(total_frames)
    noisy_chunks, clean_chunks = _chunk_tensors(seed=31, num_chunks=2)
    noisy, clean = torch.cat(noisy_chunks, dim=2), torch.cat(clean_chunks, dim=2)
    block_timesteps = torch.tensor([800.0, 400.0])
    kwargs["hidden_states"] = noisy
    kwargs["timestep"] = block_timesteps.repeat_interleave(FRAMES_PER_CHUNK * spatial_tokens).unsqueeze(0)
    teacher = _reduced_real_transformer(causal_block_frames=FRAMES_PER_CHUNK)
    _activate_conditioning_adapters(teacher)
    rollout = _reduced_real_transformer()
    rollout.load_state_dict(teacher.state_dict(), strict=True)
    cache = rollout.new_rollout_kv_cache(truncate_rollout_gradients=False)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        with _capture_cosmos3_forward(teacher) as trace:
            expected = cosmos3_teacher_forcing_reference(teacher, **kwargs, clean_x=clean)[0]
        assert expected.shape == noisy.shape
        call = trace.base_kwargs[0]
        noisy_tokens = total_frames * spatial_tokens
        assert call["vision_timesteps"].shape == (noisy_tokens,)
        torch.testing.assert_close(
            call["vision_mse_loss_indexes"], call["vision_sequence_indexes"][-noisy_tokens:], rtol=0, atol=0
        )
        torch.testing.assert_close(
            _cosmos3_noisy_sequence_indexes(call), call["vision_mse_loss_indexes"], rtol=0, atol=0
        )
        for chunk_index in range(2):
            start, end = chunk_index * FRAMES_PER_CHUNK, (chunk_index + 1) * FRAMES_PER_CHUNK
            actual = _cache_chunk_forward(rollout, kwargs, cache, noisy_chunks[chunk_index], chunk_index)
            assert actual.shape == noisy_chunks[chunk_index].shape
            torch.testing.assert_close(actual, expected[:, :, start:end], rtol=1e-4, atol=1e-5)
            if chunk_index == 0:
                _seed_chunk(rollout, kwargs, cache, clean_chunks[chunk_index], chunk_index)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_real_joint_teacher_forcing_and_untruncated_cache_gradients_match(gradient_checkpointing):
    """One block-causal clean/noisy weave and separate cache passes have the same live graph."""
    total_frames = 2 * FRAMES_PER_CHUNK
    spatial_tokens = (4 // REDUCED_CONFIG["latent_patch_size"]) ** 2
    kwargs = _joint_reference_kwargs(total_frames)
    kwargs["timestep"] = torch.tensor([[800.0, 400.0]]).repeat_interleave(FRAMES_PER_CHUNK * spatial_tokens, dim=1)
    clean = torch.randn(1, 48, total_frames, 4, 4, generator=torch.Generator().manual_seed(47))

    def run(cached):
        transformer = _reduced_real_transformer(
            gradient_checkpointing=gradient_checkpointing,
            causal_block_frames=None if cached else FRAMES_PER_CHUNK,
        )
        _activate_conditioning_adapters(transformer)
        inputs = dict(kwargs)
        for name in ("hidden_states", "neighbor_hidden_states"):
            inputs[name] = kwargs[name].detach().clone().requires_grad_(True)
        clean_input = clean.detach().clone().requires_grad_(True)
        with sdpa_kernel([SDPBackend.MATH]):
            if cached:
                cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
                outputs = []
                for chunk_index in range(2):
                    start, end = chunk_index * FRAMES_PER_CHUNK, (chunk_index + 1) * FRAMES_PER_CHUNK
                    outputs.append(
                        _cache_chunk_forward(
                            transformer, inputs, cache, inputs["hidden_states"][:, :, start:end], chunk_index
                        )
                    )
                    if chunk_index == 0:
                        seed_inputs = _chunk_slice(inputs, start, end)
                        seed_inputs["timestep"] = None
                        transformer(
                            **seed_inputs,
                            hidden_states=clean_input[:, :, start:end],
                            kv_cache=cache,
                            kv_cache_seed=True,
                            kv_cache_start_frame=start,
                        )
                prediction = torch.cat(outputs, dim=2)
            else:
                prediction = cosmos3_teacher_forcing_reference(transformer, **inputs, clean_x=clean_input)[0]
            prediction.square().mean().backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad
        }
        for name in ("hidden_states", "neighbor_hidden_states"):
            gradient = inputs[name].grad
            assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
            gradients[f"input.{name}"] = gradient.detach().clone()
        assert clean_input.grad is not None and torch.isfinite(clean_input.grad).all()
        assert torch.count_nonzero(clean_input.grad[:, :, :FRAMES_PER_CHUNK]) > 0
        assert torch.count_nonzero(clean_input.grad[:, :, FRAMES_PER_CHUNK:]) == 0
        gradients["input.clean_x"] = clean_input.grad.detach().clone()
        return prediction.detach(), gradients

    expected, teacher_gradients = run(False)
    actual, cached_gradients = run(True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert teacher_gradients.keys() == cached_gradients.keys()
    for name, expected_gradient in teacher_gradients.items():
        actual_gradient = cached_gradients[name]
        if expected_gradient is None:
            assert actual_gradient is None, name
            continue
        assert actual_gradient is not None and torch.isfinite(actual_gradient).all(), name
        scale = float(expected_gradient.abs().max().clamp_min(1e-3))
        deviation = float((actual_gradient - expected_gradient).abs().max())
        assert deviation <= 2e-3 * scale, f"{name}: cache/teacher-forcing gradient deviation {deviation:.3e}"


@pytest.mark.parametrize("reference_frames", [1, 3])
def test_joint_cache_keeps_reference_positions_states_and_target_only_history(reference_frames):
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    kwargs = _joint_reference_kwargs(TOTAL_FRAMES, reference_frames=reference_frames)
    noisy_chunks, clean_chunks = _chunk_tensors(seed=37)
    spatial_tokens = (4 // REDUCED_CONFIG["latent_patch_size"]) ** 2
    reference_tokens = reference_frames * spatial_tokens
    chunk_tokens = FRAMES_PER_CHUNK * spatial_tokens
    cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        with _capture_cosmos3_forward(transformer) as full_trace:
            transformer(**kwargs)
        full_call = full_trace.base_kwargs[0]
        full_positions = full_call["position_ids"][:, full_call["vision_sequence_indexes"]]
        for chunk_index in range(NUM_CHUNKS):
            start, end = chunk_index * FRAMES_PER_CHUNK, (chunk_index + 1) * FRAMES_PER_CHUNK
            with _capture_cosmos3_forward(transformer) as denoise_trace:
                prediction = _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[chunk_index], chunk_index)
            assert prediction.shape == noisy_chunks[chunk_index].shape
            assert cache.num_cached_chunks == chunk_index, "denoise must not append reference OR target rows"
            call = denoise_trace.base_kwargs[0]
            positions = call["position_ids"][:, call["vision_sequence_indexes"]]
            torch.testing.assert_close(
                positions[:, :reference_tokens], full_positions[:, :reference_tokens], rtol=0, atol=0
            )
            torch.testing.assert_close(
                positions[:, reference_tokens:],
                full_positions[:, reference_tokens + start * spatial_tokens : reference_tokens + end * spatial_tokens],
                rtol=0,
                atol=0,
            )
            assert call["vision_timesteps"].shape == (chunk_tokens,)
            torch.testing.assert_close(
                call["vision_mse_loss_indexes"], call["vision_sequence_indexes"][reference_tokens:], rtol=0, atol=0
            )
            for expected_rows, actual_rows in zip(full_trace.gen_outputs, denoise_trace.gen_outputs, strict=True):
                torch.testing.assert_close(
                    actual_rows[0][:reference_tokens], expected_rows[0][:reference_tokens], rtol=1e-5, atol=1e-6
                )
            if chunk_index == NUM_CHUNKS - 1:
                continue
            seed_kwargs = _chunk_slice(kwargs, start, end)
            seed_kwargs["timestep"] = None
            with _capture_cosmos3_forward(transformer) as seed_trace:
                seed_output = transformer(
                    **seed_kwargs,
                    hidden_states=clean_chunks[chunk_index],
                    kv_cache=cache,
                    kv_cache_seed=True,
                    kv_cache_start_frame=start,
                )[0]
            assert seed_output.shape == clean_chunks[chunk_index].shape
            assert torch.count_nonzero(seed_output) == 0
            seed_call = seed_trace.base_kwargs[0]
            assert seed_call["vision_timesteps"].numel() == 0
            assert seed_call["vision_mse_loss_indexes"].numel() == 0
            assert _cosmos3_noisy_sequence_indexes(seed_call).numel() == 0
            assert cache.total_stored_frames == end
            for layer_index in range(len(transformer.base_transformer.layers)):
                torch.testing.assert_close(
                    seed_trace.gen_outputs[layer_index][0][:reference_tokens],
                    full_trace.gen_outputs[layer_index][0][:reference_tokens],
                    rtol=1e-5,
                    atol=1e-6,
                )
                for text_tensor in cache.read_text_kv(layer_index):
                    assert text_tensor.shape[0] == int(kwargs["encoder_attention_mask"].sum())
                for chunks in cache.read_gen_history(layer_index, cache.num_cached_chunks):
                    assert len(chunks) == chunk_index + 1
                    for tensor in chunks:
                        assert tensor.shape == (
                            chunk_tokens,
                            REDUCED_CONFIG["num_key_value_heads"],
                            REDUCED_CONFIG["head_dim"],
                        )


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_joint_cached_pass_recomputes_live_reference_features(truncate_rollout_gradients, gradient_checkpointing):
    """Even truncated target history must not detach or cache the current reference branch."""
    transformer = _reduced_real_transformer(gradient_checkpointing=gradient_checkpointing)
    _activate_conditioning_adapters(transformer)
    kwargs = _joint_reference_kwargs(TOTAL_FRAMES)
    noisy_chunks, clean_chunks = _chunk_tensors(seed=41)
    cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=truncate_rollout_gradients)
    with torch.no_grad():
        _seed_chunk(transformer, kwargs, cache, clean_chunks[0], 0)
    kwargs["neighbor_hidden_states"] = kwargs["neighbor_hidden_states"].detach().clone().requires_grad_(True)
    prediction = _cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[1], 1)
    prediction.square().mean().backward()
    reference_grad = kwargs["neighbor_hidden_states"].grad
    assert reference_grad is not None and torch.isfinite(reference_grad).all()
    for reference_index in range(kwargs["neighbor_hidden_states"].shape[2]):
        assert torch.count_nonzero(reference_grad[:, :, reference_index]) > 0
    with torch.no_grad():
        permuted = _cache_chunk_forward(
            transformer, _select_cosmos3_references(kwargs, [2, 0, 1]), cache, noisy_chunks[1], 1
        )
        torch.testing.assert_close(permuted, prediction, rtol=1e-5, atol=1e-6)
        changed = dict(kwargs)
        changed["neighbor_hidden_states"] = kwargs["neighbor_hidden_states"].detach().clone()
        changed["neighbor_hidden_states"][:, :, 0] += torch.linspace(-1, 1, 48).view(1, 48, 1, 1)
        updated = _cache_chunk_forward(transformer, changed, cache, noisy_chunks[1], 1)
    assert not torch.equal(prediction, updated), "current references were reused from a stale cache"
    assert cache.num_cached_chunks == 1 and cache.total_stored_frames == FRAMES_PER_CHUNK


def _run_joint_cached_cp_worker(
    rank,
    world_size,
    init_file,
    frames_per_chunk,
    truncate_rollout_gradients,
    ignore_neighbors,
    backend,
):
    from model_training.utils.precision import disable_tf32_matmul

    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
        disable_tf32_matmul()
    dist.init_process_group(
        backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    try:
        total_frames = 3 * frames_per_chunk
        spatial_tokens = (4 // REDUCED_CONFIG["latent_patch_size"]) ** 2
        reference_tokens = 0 if ignore_neighbors else 3 * spatial_tokens
        inputs = _rollout_kwargs(total_frames)
        inputs.update(ignore_neighbors=ignore_neighbors, w2cs=None, Ks=None, neighbor_w2cs=None, neighbor_Ks=None)
        inputs = {name: value.to(device) if torch.is_tensor(value) else value for name, value in inputs.items()}
        clean = torch.randn(1, 48, 2 * frames_per_chunk, 4, 4, generator=torch.Generator().manual_seed(43)).to(device)

        def rollout(transformer):
            kwargs = dict(inputs)
            for name in ("hidden_states", "neighbor_hidden_states"):
                kwargs[name] = inputs[name].detach().clone().requires_grad_(True)
            clean_states = clean.detach().clone().requires_grad_(True)
            cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=truncate_rollout_gradients)
            with _capture_cosmos3_forward(transformer) as trace:
                for chunk_index in range(2):
                    start, end = chunk_index * frames_per_chunk, (chunk_index + 1) * frames_per_chunk
                    seed_kwargs = _chunk_slice(kwargs, start, end)
                    seed_kwargs["timestep"] = None
                    seed_output = transformer(
                        **seed_kwargs,
                        hidden_states=clean_states[:, :, start:end],
                        kv_cache=cache,
                        kv_cache_seed=True,
                        kv_cache_start_frame=start,
                    )[0]
                    assert seed_output.shape == (1, 48, frames_per_chunk, 4, 4)
                    assert torch.count_nonzero(seed_output) == 0
                prediction = _cache_chunk_forward(
                    transformer,
                    kwargs,
                    cache,
                    kwargs["hidden_states"][:, :, 2 * frames_per_chunk :],
                    2,
                    frames_per_chunk=frames_per_chunk,
                )
            prediction.square().sum().backward()
            return prediction.detach(), cache, trace, kwargs, clean_states

        reference = _reduced_real_transformer()
        _activate_conditioning_adapters(reference)
        reference.to(device)
        with sdpa_kernel([SDPBackend.MATH]):
            expected, reference_cache, reference_trace, reference_inputs, reference_clean = rollout(reference)
        reference_grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
            if parameter.requires_grad
        }
        cp_transformer = _reduced_real_transformer().to(device)
        cp_transformer.load_state_dict(reference.state_dict(), strict=True)
        # Preserve the requested transport even when CUDA is visible: some NGC
        # Torch builds replace a CPU mesh's Gloo group with a mixed backend.
        mesh = DeviceMesh.from_group(dist.group.WORLD, device_type=device.type)
        assert dist.get_backend(mesh.get_group()) == backend
        cp_transformer.enable_context_parallel(mesh)
        with sdpa_kernel([SDPBackend.MATH]):
            actual, cp_cache, cp_trace, cp_inputs, cp_clean = rollout(cp_transformer)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        assert actual.shape == (1, 48, frames_per_chunk, 4, 4)
        assert cp_cache.num_cached_chunks == reference_cache.num_cached_chunks == 2
        assert cp_cache.total_stored_frames == reference_cache.total_stored_frames == 2 * frames_per_chunk
        head_sharded = frames_per_chunk % world_size != 0
        local_frames = (frames_per_chunk + world_size - 1) // world_size
        cached_rows = (frames_per_chunk if head_sharded else local_frames) * spatial_tokens
        cached_heads = REDUCED_CONFIG["num_key_value_heads"] // (world_size if head_sharded else 1)
        for layer_index in range(len(reference.base_transformer.layers)):
            for text in cp_cache.read_text_kv(layer_index):
                assert text.shape == (4, cached_heads, REDUCED_CONFIG["head_dim"])
            for chunks in cp_cache.read_gen_history(layer_index, 2):
                assert len(chunks) == 2
                for tensor in chunks:
                    assert tensor.shape == (cached_rows, cached_heads, REDUCED_CONFIG["head_dim"])
                    assert tensor.requires_grad is not truncate_rollout_gradients
            for call_index in range(3):
                torch.testing.assert_close(
                    cp_trace.gen_outputs[layer_index][call_index][:reference_tokens],
                    reference_trace.gen_outputs[layer_index][0][:reference_tokens],
                    rtol=1e-5,
                    atol=1e-6,
                )
                assert cp_trace.opacity_tokens[layer_index][call_index].shape[0] == local_frames * spatial_tokens
        for call_index, call in enumerate(cp_trace.base_kwargs):
            assert call["vision_sequence_indexes"].numel() == reference_tokens + local_frames * spatial_tokens
            expected_rows = 0 if call_index < 2 else local_frames * spatial_tokens
            assert call["vision_timesteps"].numel() == expected_rows
            assert call["vision_mse_loss_indexes"].numel() == expected_rows
            positions = call["position_ids"][:, call["vision_sequence_indexes"]]
            reference_call = reference_trace.base_kwargs[call_index]
            reference_positions = reference_call["position_ids"][:, reference_call["vision_sequence_indexes"]]
            torch.testing.assert_close(
                positions[:, :reference_tokens], reference_positions[:, :reference_tokens], rtol=0, atol=0
            )
            valid_frames = max(0, min(local_frames, frames_per_chunk - rank * local_frames))
            offset = reference_tokens + rank * local_frames * spatial_tokens
            torch.testing.assert_close(
                positions[:, reference_tokens : reference_tokens + valid_frames * spatial_tokens],
                reference_positions[:, offset : offset + valid_frames * spatial_tokens],
                rtol=0,
                atol=0,
            )
        for name, parameter in cp_transformer.named_parameters():
            if parameter.requires_grad:
                _assert_cosmos3_cp_gradient_mean(
                    parameter.grad, reference_grads[name], world_size=world_size, name=f"cache/{name}"
                )
        for name in ("hidden_states", "neighbor_hidden_states"):
            expected_grad = reference_inputs[name].grad
            if ignore_neighbors and name == "neighbor_hidden_states":
                assert expected_grad is None
            else:
                assert expected_grad is not None and torch.count_nonzero(expected_grad) > 0, name
            _assert_cosmos3_cp_gradient_mean(
                cp_inputs[name].grad, expected_grad, world_size=world_size, name=f"cache/{name}"
            )
        if truncate_rollout_gradients:
            assert reference_clean.grad is None and cp_clean.grad is None
        else:
            assert reference_clean.grad is not None and torch.count_nonzero(reference_clean.grad) > 0
            _assert_cosmos3_cp_gradient_mean(
                cp_clean.grad, reference_clean.grad, world_size=world_size, name="cache/clean_target_history"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("ignore_neighbors", [False, True], ids=["references", "dropped_references"])
@pytest.mark.parametrize("frames_per_chunk", [1, 2], ids=["head_sharded_padding", "frame_sharded"])
@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_real_cached_cp1_cp2_forward_backward_and_target_only_rows_match(
    tmp_path, frames_per_chunk, truncate_rollout_gradients, ignore_neighbors, backend
):
    """Actual collectives: refs are prepended once, never gathered/stored as video history."""
    _run_cp_workers(
        _run_joint_cached_cp_worker,
        tmp_path,
        frames_per_chunk,
        truncate_rollout_gradients,
        ignore_neighbors,
        backend=backend,
    )
