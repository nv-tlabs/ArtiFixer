# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the KV-cache inference pipeline call chain.

Two past bugs these tests catch: a positional call bound ``show_progress``
to ``ignore_neighbors``, which dropped reference-view conditioning at
inference; and ``prepare_latents`` max-pooled the opacity map without a
channel dimension on non-first chunks, which was only correct for batch
size 1 (through broadcasting).  The CP KV-cache tests cover the per-rank
chunk storage and windowed eviction used during chunked validation.

No GPU required -- run with:
    pytest tests/test_kv_cache_pipeline.py -v
"""

import inspect
import types

import pytest
import torch
from torch import nn

import model_training.net.context_parallel as context_parallel_mod
import model_training.net.wan_transformer as transformer_mod
from model_training.net.rollout_kv_cache import RolloutKvCache
from model_training.net.wan_transformer import ArtifixerSelfAttnProcessor, KvCacheWanSelfAttnProcessor
from model_training.pipeline.cosmos3_kv_cache_pipeline import ArtifixerCosmos3KvCachePipeline
from model_training.pipeline.cosmos3_pipeline import ArtifixerCosmos3Pipeline
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline
from model_training.pipeline.wan_pipeline import ArtifixerPipeline
from tests._helpers import _FakeCpMesh, _GroupedFakeCpMesh

_VAE_TEMPORAL = 4
_VAE_SPATIAL = 8


def _vae_stub() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        config=types.SimpleNamespace(scale_factor_temporal=_VAE_TEMPORAL, scale_factor_spatial=_VAE_SPATIAL),
    )


class _CaptureKvCachePipeline(ArtifixerKvCachePipeline):
    """Bypasses the heavy constructor and captures generate_samples_from_batch calls."""

    def __init__(self):
        self.vae = _vae_stub()
        self.transformer = types.SimpleNamespace(cp_mesh=None, training=False)
        self.captured = None

    def encode_video_frames(self, video_frames):
        batch_size, num_frames = video_frames.shape[:2]
        latent_frames = (num_frames - 1) // _VAE_TEMPORAL + 1
        return torch.zeros(batch_size, 16, latent_frames, 4, 4)

    def encode_neighbors(self, neighbor_frames, max_neighbors_per_encode=None):
        return torch.zeros(neighbor_frames.shape[0], 16, neighbor_frames.shape[1], 4, 4)

    def generate_samples_from_batch(self, *args, **kwargs):
        self.captured = (args, kwargs)
        return torch.zeros(1)


class _FakeWanAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Identity()])


def _install_lightweight_attention(
    monkeypatch,
    *,
    captured_attention_lengths=None,
    captured_attention_keys=None,
    captured_attention_tensors=None,
    rank_offsets: bool = False,
    all_to_all_cp_size: int | None = None,
):
    def fake_get_qkv_projections(attn, hidden_states, encoder_hidden_states):
        return attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)

    def fake_dispatch_attention_fn(query, key, value, **kwargs):
        if captured_attention_lengths is not None:
            captured_attention_lengths.append(key.shape[1])
        if captured_attention_keys is not None:
            captured_attention_keys.append(key.detach().clone())
        if captured_attention_tensors is not None:
            captured_attention_tensors.append(key)
        return torch.zeros_like(query)

    def fake_all_gather(outputs, tensor, group=None):
        for rank, output in enumerate(outputs):
            offset = rank * 1000 if rank_offsets else 0
            output.copy_(tensor + offset)

    def fake_all_to_all_with_grad(tensor, scatter_dim, gather_dim, group=None):
        # Single-process stand-in for the Ulysses exchange: every rank holds
        # this same shard, so each rank contributes its scatter chunk along
        # the gather dim. Pure chunk/cat keeps the exchange autograd-aware,
        # like the real collective.
        chunks = tensor.chunk(all_to_all_cp_size, dim=scatter_dim)
        return torch.cat([chunks[0]] * all_to_all_cp_size, dim=gather_dim)

    monkeypatch.setattr(transformer_mod, "_get_qkv_projections", fake_get_qkv_projections)
    monkeypatch.setattr(transformer_mod, "dispatch_attention_fn", fake_dispatch_attention_fn)
    monkeypatch.setattr(context_parallel_mod.dist, "all_gather", fake_all_gather)
    if all_to_all_cp_size is not None:
        monkeypatch.setattr(transformer_mod, "all_to_all_with_grad", fake_all_to_all_with_grad)


def _identity_fake_attention(dim: int, heads: int):
    attn = _FakeWanAttention(dim, heads)
    for linear in (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]):
        with torch.no_grad():
            linear.weight.zero_()
            linear.weight.fill_diagonal_(1)
    return attn


@pytest.mark.parametrize(
    "pipeline_cls,method,flags",
    [
        (
            ArtifixerKvCachePipeline,
            "generate_samples_from_batch",
            ("ignore_neighbors", "show_progress", "progress_bar_leave"),
        ),
        (
            ArtifixerKvCachePipeline,
            "denoise_to_latents",
            ("show_progress", "progress_bar_leave", "max_neighbors_per_encode"),
        ),
        (
            ArtifixerKvCachePipeline,
            "forward_inference",
            ("show_progress", "progress_bar_leave", "max_neighbors_per_encode"),
        ),
        (
            ArtifixerPipeline,
            "forward_inference",
            ("show_progress", "progress_bar_leave", "max_neighbors_per_encode"),
        ),
        (
            ArtifixerCosmos3Pipeline,
            "forward_inference",
            ("show_progress", "progress_bar_leave", "max_neighbors_per_encode"),
        ),
        (
            ArtifixerCosmos3KvCachePipeline,
            "forward_inference",
            ("show_progress", "progress_bar_leave", "max_neighbors_per_encode"),
        ),
        (
            ArtifixerCosmos3KvCachePipeline,
            "denoise_to_latents",
            ("show_progress", "progress_bar_leave", "max_neighbors_per_encode"),
        ),
        (
            ArtifixerCosmos3KvCachePipeline,
            "generate_samples_from_batch",
            ("ignore_neighbors", "show_progress", "progress_bar_leave"),
        ),
    ],
)
def test_pipeline_trailing_flags_are_keyword_only(pipeline_cls, method, flags):
    signature = inspect.signature(getattr(pipeline_cls, method))
    for name in flags:
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{method}: {name} must be keyword-only: a positional call once bound show_progress to "
            "ignore_neighbors, silently dropping reference views at inference"
        )


def test_denoise_to_latents_forwards_neighbors_and_progress_flags():
    pipe = _CaptureKvCachePipeline()
    pipe.denoise_to_latents(
        rendered_rgb=torch.rand(1, 9, 3, 32, 32),
        rendered_opacity=torch.rand(1, 9, 32, 32),
        neighbors=torch.rand(1, 2, 3, 32, 32),
        camera_rays=None,
        w2cs=None,
        neighbor_w2cs=None,
        Ks=None,
        neighbor_Ks=None,
        prompt=torch.zeros(1, 3, 8),
        num_inference_steps=4,
        show_progress=True,
        progress_bar_leave=False,
    )

    args, kwargs = pipe.captured
    assert kwargs["use_exit_flag"] is False
    assert kwargs.get("ignore_neighbors", False) is False
    assert kwargs["show_progress"] is True
    assert kwargs["progress_bar_leave"] is False
    neighbors_condition = args[2]
    assert neighbors_condition is not None, "neighbors_condition must reach the rollout"


def _reference_prepare_latents(condition, opacity, is_first_chunk, noise):
    pooled = (
        torch.cat([opacity[:, :1].repeat_interleave(3, dim=1), opacity], dim=1) if is_first_chunk else opacity
    ).unsqueeze(1)
    pooled = torch.nn.functional.max_pool3d(pooled, (_VAE_TEMPORAL, _VAE_SPATIAL, _VAE_SPATIAL))
    return condition * pooled + noise * (1 - pooled)


def test_prepare_latents_pools_opacity_with_channel_dim_for_any_batch_size():
    pipe = _CaptureKvCachePipeline()
    for batch_size in (1, 2, 16):
        for is_first_chunk in (True, False):
            num_frames = 9 if is_first_chunk else 8
            latent_frames = (num_frames + 3) // _VAE_TEMPORAL if is_first_chunk else num_frames // _VAE_TEMPORAL
            opacity = torch.rand(batch_size, num_frames, 32, 32)
            condition = torch.randn(batch_size, 16, latent_frames, 4, 4)

            torch.manual_seed(99)
            out = ArtifixerPipelineBase.prepare_latents(pipe, condition, opacity, is_first_chunk)
            torch.manual_seed(99)
            reference = _reference_prepare_latents(condition, opacity, is_first_chunk, torch.randn_like(condition))

            assert torch.equal(
                out, reference
            ), f"prepare_latents mismatch for batch_size={batch_size} is_first_chunk={is_first_chunk}"


def test_kv_cache_rejects_negative_sink_size():
    with pytest.raises(ValueError, match="non-negative"):
        ArtifixerKvCachePipeline(
            vae=_vae_stub(),
            transformer=types.SimpleNamespace(),
            tokenizer=None,
            text_encoder=None,
            ar_frames_per_chunk=7,
            local_attn_size=21,
            sink_size=-7,
            gradient_checkpointing=False,
            checkpoint_every_n_blocks=1,
            expand_timesteps=False,
        )


def test_kv_cache_pipeline_rejects_non_positive_ar_frames_per_chunk():
    with pytest.raises(ValueError, match="positive"):
        ArtifixerKvCachePipeline(
            vae=_vae_stub(),
            transformer=types.SimpleNamespace(),
            tokenizer=None,
            text_encoder=None,
            ar_frames_per_chunk=0,
            local_attn_size=21,
            sink_size=7,
            gradient_checkpointing=False,
            checkpoint_every_n_blocks=1,
            expand_timesteps=False,
        )


def test_kv_cache_pipeline_rejects_non_positive_local_attn_size():
    # -1 means unlimited; 0 and other non-positive values are contract
    # violations, not "no window".
    with pytest.raises(ValueError, match="unlimited attention window"):
        ArtifixerKvCachePipeline(
            vae=_vae_stub(),
            transformer=types.SimpleNamespace(),
            tokenizer=None,
            text_encoder=None,
            ar_frames_per_chunk=7,
            local_attn_size=0,
            sink_size=0,
            gradient_checkpointing=False,
            checkpoint_every_n_blocks=1,
            expand_timesteps=False,
        )


@pytest.mark.parametrize(
    "local_attn_size,sink_size,message",
    [
        (0, 0, "unlimited attention window"),
        (-2, 0, "unlimited attention window"),
        (21, -7, "non-negative"),
        (21, 21, "smaller than"),
        (-1, -1, "non-negative"),
    ],
)
def test_both_families_reject_the_same_window_options_via_the_shared_validator(local_attn_size, sink_size, message):
    """Both family pipelines take the same --local_attn_size/--sink_size
    option pair and range-check it through the ONE shared validator in
    pipeline_base, so an out-of-range pair fails identically no matter the
    model family."""
    from model_training.pipeline import cosmos3_kv_cache_pipeline, pipeline_base, wan_kv_cache_pipeline

    assert wan_kv_cache_pipeline.validate_rollout_window_options is pipeline_base.validate_rollout_window_options
    assert cosmos3_kv_cache_pipeline.validate_rollout_window_options is pipeline_base.validate_rollout_window_options

    with pytest.raises(ValueError, match=message):
        ArtifixerKvCachePipeline(
            vae=_vae_stub(),
            transformer=types.SimpleNamespace(),
            tokenizer=None,
            text_encoder=None,
            ar_frames_per_chunk=7,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            gradient_checkpointing=False,
            checkpoint_every_n_blocks=1,
            expand_timesteps=False,
        )
    with pytest.raises(ValueError, match=message):
        ArtifixerCosmos3KvCachePipeline(
            vae=_vae_stub(),
            transformer=types.SimpleNamespace(),
            gradient_checkpointing=False,
            ar_frames_per_chunk=7,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )


def test_cosmos3_kv_cache_pipeline_rejects_a_window_with_no_room_after_the_anchor():
    """The chunk-granularity companion to the shared range checks: at
    ar_frames_per_chunk granularity the anchor chunks must leave at least
    one non-anchor chunk inside a bounded window, or eviction could never
    free space."""
    with pytest.raises(ValueError, match="non-anchor chunk"):
        ArtifixerCosmos3KvCachePipeline(
            vae=_vae_stub(),
            transformer=types.SimpleNamespace(),
            gradient_checkpointing=False,
            ar_frames_per_chunk=7,
            local_attn_size=8,
            sink_size=7,
        )


def test_denoise_to_latents_rejects_transformer_in_train_mode():
    # Train mode keys the KV cache to the full frame span instead of the
    # deployed windowed/sink attention; inference must fail loudly.
    pipe = _CaptureKvCachePipeline()
    pipe.transformer.training = True

    with pytest.raises(AssertionError, match="eval mode"):
        pipe.denoise_to_latents(
            rendered_rgb=torch.rand(1, 9, 3, 32, 32),
            rendered_opacity=torch.rand(1, 9, 32, 32),
            neighbors=torch.rand(1, 2, 3, 32, 32),
            camera_rays=None,
            w2cs=None,
            neighbor_w2cs=None,
            Ks=None,
            neighbor_Ks=None,
            prompt=torch.zeros(1, 3, 8),
            num_inference_steps=4,
        )


def _drive_windowed_rollout(
    processor,
    attn,
    *,
    num_blocks: int,
    chunk_frames: int,
    tokens_per_shard: int,
    num_denoising_calls: int,
    window_chunks: int,
    sink_chunks: int,
    dim: int,
) -> RolloutKvCache:
    """One rank's windowed rollout at the pipeline's seams: evict-after-anchor
    before each chunk, per-step denoise calls over [cached history + own
    rows], and a final-call stage/commit for every chunk except the last.
    Block b's step d carries the constant b * 10 + d, so the attended keys
    identify exactly which block/step (and, with rank offsets, which rank)
    each row came from."""
    cache = RolloutKvCache(1, truncate_rollout_gradients=False)
    with torch.no_grad():
        for block_idx in range(num_blocks):
            cache.evict_to_window(window_chunks, sink_chunks)
            num_history_chunks = cache.num_cached_chunks
            for denoise_idx in range(num_denoising_calls):
                is_final_call = denoise_idx == num_denoising_calls - 1 and block_idx < num_blocks - 1
                kv_cache_stage = {} if is_final_call else None
                hidden_states = torch.full(
                    (1, tokens_per_shard, dim),
                    block_idx * 10 + denoise_idx,
                    dtype=torch.float32,
                )
                output = processor(
                    attn,
                    hidden_states,
                    None,
                    None,
                    None,
                    kv_cache=cache,
                    layer_index=0,
                    num_history_chunks=num_history_chunks,
                    kv_cache_stage=kv_cache_stage,
                )
                assert output.shape == hidden_states.shape
                if kv_cache_stage is not None:
                    cache.store_gen_chunk(
                        0, num_history_chunks, block_idx * chunk_frames, chunk_frames, *kv_cache_stage[0]
                    )
    return cache


@pytest.mark.parametrize("sink_size", [0, 7])
def test_cp_kv_cache_window_attends_anchor_plus_rolling_chunks(monkeypatch, sink_size):
    """CP validation with the windowed rollout cache, at the configuration of
    the 105-frame validation runs: 7-frame chunks, CP size 7 (one latent
    frame per rank per chunk), a 21-frame window, and repeated denoising
    calls per chunk. Each rank stores its own shard of every kept chunk;
    reads gather [history + own] per rank in rank order. The last chunk's
    attention must see, per rank, exactly the anchor (when sink_size > 0)
    plus the most recent chunks — carrying each source chunk's FINAL
    denoising-step values — and the first chunk's first call attends only its
    own rows (nothing else is stored yet)."""
    captured_attention_keys = []
    _install_lightweight_attention(monkeypatch, captured_attention_keys=captured_attention_keys, rank_offsets=True)

    cp_size = 7
    frames_per_chunk = 7
    latent_frames = 105
    frame_seq_length = 5
    num_denoising_calls = 4
    dim = 1

    processor = KvCacheWanSelfAttnProcessor()
    processor._cp_mesh = _FakeCpMesh(size=cp_size, rank=3)
    attn = _identity_fake_attention(dim, heads=1)

    _drive_windowed_rollout(
        processor,
        attn,
        num_blocks=latent_frames // frames_per_chunk,
        chunk_frames=frames_per_chunk,
        tokens_per_shard=(frames_per_chunk // cp_size) * frame_seq_length,
        num_denoising_calls=num_denoising_calls,
        window_chunks=21 // frames_per_chunk,
        sink_chunks=sink_size // frames_per_chunk,
        dim=dim,
    )

    assert captured_attention_keys[0].shape[1] == cp_size * frame_seq_length
    kept_blocks = [0, 13, 14] if sink_size else [12, 13, 14]
    expected_values = []
    for rank in range(cp_size):
        for block_idx in kept_blocks:
            expected_values.extend([block_idx * 10 + num_denoising_calls - 1 + rank * 1000] * frame_seq_length)
    assert torch.equal(captured_attention_keys[-1].flatten(), torch.tensor(expected_values, dtype=torch.float32))


def test_cp_kv_cache_window_handles_multi_frame_per_rank_chunks(monkeypatch):
    captured_attention_keys = []
    _install_lightweight_attention(monkeypatch, captured_attention_keys=captured_attention_keys, rank_offsets=True)

    cp_size = 4
    frames_per_chunk = 8
    latent_frames = 120
    frame_seq_length = 3
    dim = 1
    num_denoising_calls = 3

    processor = KvCacheWanSelfAttnProcessor()
    processor._cp_mesh = _FakeCpMesh(size=cp_size, rank=3)
    attn = _identity_fake_attention(dim, heads=1)

    _drive_windowed_rollout(
        processor,
        attn,
        num_blocks=latent_frames // frames_per_chunk,
        chunk_frames=frames_per_chunk,
        tokens_per_shard=(frames_per_chunk // cp_size) * frame_seq_length,
        num_denoising_calls=num_denoising_calls,
        window_chunks=24 // frames_per_chunk,
        sink_chunks=8 // frames_per_chunk,
        dim=dim,
    )

    expected_values = []
    for rank in range(cp_size):
        for block_idx in (0, 13, 14):
            expected_values.extend([block_idx * 10 + num_denoising_calls - 1 + rank * 1000] * 2 * frame_seq_length)
    assert torch.equal(captured_attention_keys[-1].flatten(), torch.tensor(expected_values, dtype=torch.float32))


def test_kv_cache_pipeline_rejects_rolling_window_smaller_than_rollout_chunk():
    with pytest.raises(ValueError, match="align to `ar_frames_per_chunk`"):
        ArtifixerKvCachePipeline(
            vae=_vae_stub(),
            transformer=types.SimpleNamespace(),
            tokenizer=None,
            text_encoder=None,
            ar_frames_per_chunk=7,
            local_attn_size=10,
            sink_size=7,
            gradient_checkpointing=False,
            checkpoint_every_n_blocks=1,
            expand_timesteps=False,
        )


def test_cp_self_attention_grad_enabled_forward_carries_kv_grad_without_block_mask(monkeypatch):
    """The invariant behind the old inference-only gather guard: gradient
    through CP self-attention keys/values must never be silently dropped.
    The mask-free CP path now exchanges Q/K/V through the autograd-aware
    Ulysses all-to-all, so a grad-enabled CP forward is valid (it is the
    CP training path) and the exchanged K/V stay connected to the graph."""
    captured_attention_tensors = []
    _install_lightweight_attention(
        monkeypatch, captured_attention_tensors=captured_attention_tensors, all_to_all_cp_size=2
    )
    processor = ArtifixerSelfAttnProcessor()
    processor._cp_mesh = _GroupedFakeCpMesh(size=2, rank=3)
    attn = _FakeWanAttention(8, 2)

    processor(attn, torch.randn(1, 5, 8), None, None, None, None)

    exchanged_key = captured_attention_tensors[-1]
    assert exchanged_key.requires_grad
    exchanged_key.sum().backward()
    assert attn.to_k.weight.grad is not None


def test_cp_self_attention_mask_free_gather_runs_under_no_grad(monkeypatch):
    """Inference-mode CP forwards attend over the K/V of every rank. The
    Ulysses exchange shards heads across ranks, so heads must divide by
    the CP size."""
    captured_attention_lengths = []
    cp_size = 3
    _install_lightweight_attention(
        monkeypatch, captured_attention_lengths=captured_attention_lengths, all_to_all_cp_size=cp_size
    )
    processor = ArtifixerSelfAttnProcessor()
    processor._cp_mesh = _GroupedFakeCpMesh(size=cp_size, rank=3)
    attn = _FakeWanAttention(6, 3)

    with torch.no_grad():
        processor(attn, torch.randn(1, 5, 6), None, None, None, None)

    assert captured_attention_lengths[-1] == cp_size * 5


def test_cp_self_attention_allows_grad_enabled_forward_when_kv_carry_no_grad(monkeypatch):
    """all_gather_cat's guard keys on the gathered tensor requiring grad, not
    on grad mode alone: a fully frozen grad-enabled forward stays valid."""
    _install_lightweight_attention(monkeypatch)
    processor = ArtifixerSelfAttnProcessor()
    processor._cp_mesh = _FakeCpMesh(size=2, rank=3)
    attn = _FakeWanAttention(8, 2)
    attn.requires_grad_(False)

    processor(attn, torch.randn(1, 5, 8), None, None, None, None)
