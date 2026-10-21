# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 Self-Forcing AR rollout tests (pipeline plumbing).

The rollout is exercised end-to-end through ArtifixerCosmos3KvCachePipeline with
the canonical tiny Cosmos3 doubles. Every rollout drives the KV cache
through the real ArtifixerCosmos3AttnProcessor, so the doubles carry it
(_TinyProcessorCosmosBaseTransformer): cache passes read and write per-layer
K/V inside attention, and a processor-less fake would starve the cache and
trip the coverage contract on the second chunk. Value-level checks live in
tests/test_cosmos3_kv_cache_rollout.py (pinned against stored expected
outputs on the real reduced class); this file pins the pipeline plumbing —
chunk partitioning, per-pass conditioning slices, truncation, RNG draws,
and the sliding window.
"""

from types import SimpleNamespace

import pytest
import torch

import model_training.pipeline.cosmos3_kv_cache_pipeline as cosmos3_kv_module
import model_training.pipeline.pipeline_base as pipeline_common_module
from model_training.net.cosmos3_transformer import ArtifixerCosmos3Transformer
from model_training.net.rollout_kv_cache import RolloutKvCache
from model_training.pipeline.cosmos3_kv_cache_pipeline import ArtifixerCosmos3KvCachePipeline
from tests._helpers import (
    LATENT_CHANNELS,
    _FakeCpMesh,
    _TinyCosmosVae,
    _TinyProcessorCosmosBaseTransformer,
    patch_kv_cache_pipeline_factories,
)

# _TinyCosmosVae carries the production VAE factors (16 spatial / 4 temporal),
# so pixel-space conditioning is latent * 16 spatially and 1 + 4N temporally.
VAE_SPATIAL = 16
VAE_TEMPORAL = 4
LATENT_HW = 4
PIXEL_HW = LATENT_HW * VAE_SPATIAL


def _kv_cache_pipe(
    ar_frames_per_chunk: int | None,
    base_transformer=None,
    *,
    local_attn_size: int = 21,
    sink_size: int = 7,
    truncate_rollout_gradients: bool = False,
):
    return ArtifixerCosmos3KvCachePipeline(
        vae=_TinyCosmosVae(),
        transformer=_TinyProcessorCosmosBaseTransformer() if base_transformer is None else base_transformer,
        gradient_checkpointing=False,
        ar_frames_per_chunk=ar_frames_per_chunk,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        truncate_rollout_gradients=truncate_rollout_gradients,
    )


def _rollout_inputs(latent_frames: int, *, opacity_value: float | None = 1.0) -> dict:
    pixel_frames = (latent_frames - 1) * VAE_TEMPORAL + 1
    opacity = (
        torch.rand(1, pixel_frames, PIXEL_HW, PIXEL_HW)
        if opacity_value is None
        else torch.full((1, pixel_frames, PIXEL_HW, PIXEL_HW), opacity_value)
    )
    inputs = dict(
        condition=torch.randn(1, LATENT_CHANNELS, latent_frames, LATENT_HW, LATENT_HW),
        rendered_opacity=opacity,
        neighbors_condition=None,
        camera_rays=torch.randn(1, latent_frames, PIXEL_HW, PIXEL_HW, 6),
        w2cs=torch.eye(4).view(1, 1, 4, 4).expand(1, latent_frames, -1, -1).clone(),
        neighbor_w2cs=None,
        Ks=torch.eye(3).view(1, 1, 3, 3).expand(1, latent_frames, -1, -1).clone(),
        neighbor_Ks=None,
        encoded_prompt=torch.tensor([[11, 12, 7, 8]]),
        prompt_attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    return inputs


def _run_rollout(
    pipe,
    inputs: dict,
    num_inference_steps: int,
    use_exit_flag: bool = False,
    *,
    condition: torch.Tensor | None = None,
    with_attention_mask: bool = True,
):
    """Single home for the 11-positional generate_samples_from_batch plumbing
    (NO seeding happens here —
    several tests below pin fresh-draw randomness). ``condition`` overrides
    inputs["condition"] (grad-tracking tests); error-path tests pass
    with_attention_mask=False to keep their exact call shape."""
    kwargs = {}
    if with_attention_mask:
        kwargs["prompt_attention_mask"] = inputs["prompt_attention_mask"]
    return pipe.generate_samples_from_batch(
        inputs["condition"] if condition is None else condition,
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
        **kwargs,
    )


def _spy_transformer_forwards(monkeypatch, captured: list):
    original_forward = ArtifixerCosmos3Transformer.forward

    def spy_forward(self, **kwargs):
        captured.append({**kwargs, "grad_enabled": torch.is_grad_enabled()})
        return original_forward(self, **kwargs)

    monkeypatch.setattr(ArtifixerCosmos3Transformer, "forward", spy_forward)


def test_ar_rollout_partitions_uniform_chunks_and_slices_conditioning_chunk_only(monkeypatch):
    """Uniform chunks in temporal order against the KV cache; per pass
    hidden_states carries ONLY the chunk's frames, kv_cache_start_frame is
    the chunk's absolute start, and every conditioning tensor is the chunk's
    absolute slice (opacity = the chunk's real pixel windows, rays/w2cs/Ks =
    the chunk's frames) — future-frame conditioning never enters a pass, and
    neither does anything before the chunk (history attends via cache
    content). Each finished chunk except the last is followed by exactly one
    seed pass (timestep None). Neighbors/text pass through unchanged on
    every call."""
    captured = []
    _spy_transformer_forwards(monkeypatch, captured)
    pipe = _kv_cache_pipe(ar_frames_per_chunk=2)
    inputs = _rollout_inputs(latent_frames=6)

    prediction = _run_rollout(pipe, inputs, num_inference_steps=2)

    assert prediction.shape == inputs["condition"].shape
    assert bool(torch.isfinite(prediction).all())
    # 3 chunks x (2 denoising steps + 1 seed pass except after the last).
    assert len(captured) == 3 * 2 + 2
    chunk_of_call = [0, 0, 0, 1, 1, 1, 2, 2]
    seed_flags = [call.get("kv_cache_seed", False) for call in captured]
    assert seed_flags == [False, False, True, False, False, True, False, False]
    for call_index, call in enumerate(captured):
        chunk_index = chunk_of_call[call_index]
        chunk_start, chunk_end = 2 * chunk_index, 2 * (chunk_index + 1)
        # Chunk-only pass at the chunk's ABSOLUTE start.
        assert call["hidden_states"].shape == (1, LATENT_CHANNELS, 2, LATENT_HW, LATENT_HW)
        assert call["kv_cache_start_frame"] == chunk_start
        assert isinstance(call["kv_cache"], RolloutKvCache)
        pixel_start = 0 if chunk_start == 0 else (chunk_start - 1) * VAE_TEMPORAL + 1
        assert torch.equal(
            call["opacity"], inputs["rendered_opacity"][:, pixel_start : (chunk_end - 1) * VAE_TEMPORAL + 1]
        )
        assert torch.equal(call["camera_rays"], inputs["camera_rays"][:, chunk_start:chunk_end])
        assert torch.equal(call["w2cs"], inputs["w2cs"][:, chunk_start:chunk_end])
        assert torch.equal(call["Ks"], inputs["Ks"][:, chunk_start:chunk_end])
        if call.get("kv_cache_seed", False):
            # Seed passes are conditioning-mode: no timestep.
            assert call["timestep"] is None
        else:
            # ONE scalar fp32 sigma per denoise pass (the fp32 timestep
            # policy via prepare_transformer_timestep).
            assert call["timestep"].shape == (1,)
            assert call["timestep"].dtype == torch.float32
        # Pass-through conditioning is never sliced.
        assert torch.equal(call["encoder_hidden_states"], inputs["encoded_prompt"])
        assert torch.equal(call["encoder_attention_mask"], inputs["prompt_attention_mask"])
        assert call["neighbor_hidden_states"] is None
    # ONE cache for the whole rollout.
    assert len({id(call["kv_cache"]) for call in captured}) == 1


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_ar_rollout_context_gradients_follow_the_truncation_flag(truncate_rollout_gradients):
    """The cross-chunk gradient path through the cache is the truncation
    flag's contract: with truncation a later chunk's loss sends ZERO gradient
    to earlier chunks' inputs (seeded K/V are stored detached); by default
    the path EXISTS — the same loss reaches chunk 0's inputs through the
    cached K/V and the seed forward. With opacity == 1, z_mix == condition
    exactly, so chunk 1's output gradient w.r.t. chunk 0's condition frames
    is the probe either way. The VALUE dependence on the context (through
    cached joint attention) is mode-independent."""
    torch.manual_seed(0)
    pipe = _kv_cache_pipe(ar_frames_per_chunk=2, truncate_rollout_gradients=truncate_rollout_gradients)
    inputs = _rollout_inputs(latent_frames=4, opacity_value=1.0)
    condition = inputs["condition"].requires_grad_(True)

    def rollout(condition_tensor: torch.Tensor) -> torch.Tensor:
        # single step => the only step per chunk runs in the ambient grad context
        return _run_rollout(pipe, inputs, num_inference_steps=1, condition=condition_tensor)

    with torch.enable_grad():
        prediction = rollout(condition)
        (condition_grad,) = torch.autograd.grad(prediction[:, :, 2:].sum(), condition)

    assert torch.count_nonzero(condition_grad[:, :, 2:]) > 0
    if truncate_rollout_gradients:
        assert (
            torch.count_nonzero(condition_grad[:, :, :2]) == 0
        ), "gradients from chunk 1 leaked into chunk 0's graph — truncation must store cached K/V detached"
    else:
        assert (
            torch.count_nonzero(condition_grad[:, :, :2]) > 0
        ), "chunk 1's loss must reach chunk 0 through the cached K/V in the default mode"

    # Non-vacuity control: chunk 1's VALUES do depend on chunk 0 (its rows
    # attend chunk 0's cached keys through the real processor), so the
    # truncation-mode zero gradient can only come from the detached store,
    # not from a missing data path.
    with torch.no_grad():
        perturbed_condition = condition.detach().clone()
        perturbed_condition[:, :, :2] += 1.0
        baseline = rollout(condition.detach())
        perturbed = rollout(perturbed_condition)
    assert not torch.equal(baseline[:, :, 2:], perturbed[:, :, 2:])


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_ar_rollout_seed_pass_x0_grad_state_follows_the_truncation_flag(monkeypatch, truncate_rollout_gradients):
    """The pipeline seam of the truncation flag: with truncation the
    hidden_states entering every seed pass have requires_grad False and no
    grad_fn (cache content can never smuggle autograd history across
    chunks); by default the seed pass receives the chunk's graph-carrying
    x0, the entry point of the cross-chunk gradient path. The rollout output
    is graph-connected either way (exit steps run under grad)."""
    captured = []
    _spy_transformer_forwards(monkeypatch, captured)
    pipe = _kv_cache_pipe(ar_frames_per_chunk=2, truncate_rollout_gradients=truncate_rollout_gradients)
    inputs = _rollout_inputs(latent_frames=4)

    with torch.enable_grad():
        # single step => every chunk's only denoise pass is its (grad-enabled) exit step
        prediction = _run_rollout(pipe, inputs, num_inference_steps=1)

    # [exit-denoise c0, seed c0, exit-denoise c1].
    assert [call.get("kv_cache_seed", False) for call in captured] == [False, True, False]
    assert prediction.requires_grad
    seed_hidden_states = captured[1]["hidden_states"]
    if truncate_rollout_gradients:
        assert seed_hidden_states.requires_grad is False
        assert seed_hidden_states.grad_fn is None
    else:
        assert seed_hidden_states.requires_grad and seed_hidden_states.grad_fn is not None


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_ar_rollout_exit_flag_truncation_is_shared_across_chunks(monkeypatch, truncate_rollout_gradients):
    """wan_kv_cache_pipeline exit-flag semantics: ONE rank-synced exit
    index for the whole rollout; per chunk, pre-exit steps run under no_grad,
    the exit step runs in the ambient grad context, then the chunk breaks
    and (except after the last chunk) the finished x0 is seeded — under
    no_grad with truncation, in the ambient grad mode by default."""
    exit_flag_calls = []

    def fake_exit_flag(num_inference_steps, *, device):
        exit_flag_calls.append(num_inference_steps)
        return 1

    monkeypatch.setattr(pipeline_common_module, "generate_and_sync_exit_flag", fake_exit_flag)
    captured = []
    _spy_transformer_forwards(monkeypatch, captured)
    pipe = _kv_cache_pipe(ar_frames_per_chunk=2, truncate_rollout_gradients=truncate_rollout_gradients)
    inputs = _rollout_inputs(latent_frames=4)

    with torch.enable_grad():
        _run_rollout(pipe, inputs, num_inference_steps=4, use_exit_flag=True)

    assert exit_flag_calls == [4], "the exit index must be drawn once and shared across chunks"
    # exit index 1 => per chunk: steps 0 and 1 then break; chunk 0 also seeds.
    assert [call.get("kv_cache_seed", False) for call in captured] == [False, False, True, False, False]
    seed_grad_enabled = not truncate_rollout_gradients
    assert [call["grad_enabled"] for call in captured] == [False, True, seed_grad_enabled, False, True]


def test_ar_rollout_draws_fresh_z_mix_per_sde_renoise(monkeypatch):
    """Every 'noise' slot is z_mix and every SDE re-noise draws a FRESH one,
    matching the Wan rollout, whose chunk-start latents and draw_fresh_latents
    closure each take a fresh prepare_latents z_mix draw — one draw at chunk
    start plus one per pre-exit step; seed passes draw NOTHING, so the cache
    adds no RNG consumption."""
    pipe = _kv_cache_pipe(ar_frames_per_chunk=2)
    draws = []
    original_prepare_latents = pipe.prepare_latents

    def spying_prepare_latents(condition, opacity, is_first_chunk, **_kwargs):
        latents = original_prepare_latents(condition, opacity, is_first_chunk)
        draws.append(latents.detach().clone())
        return latents

    monkeypatch.setattr(pipe, "prepare_latents", spying_prepare_latents)
    # opacity == 0 makes z_mix pure fresh randn, so distinct draws differ a.s.
    inputs = _rollout_inputs(latent_frames=4, opacity_value=0.0)

    _run_rollout(pipe, inputs, num_inference_steps=3)

    # 2 chunks x (1 rollout start + 2 pre-exit re-noises); the seed pass
    # between them draws nothing.
    assert len(draws) == 6
    for first, second in zip(draws, draws[1:]):
        assert not torch.equal(first, second), "SDE re-noise must draw a fresh z_mix, not reuse a cached draw"


def test_ar_rollout_windows_long_videos_with_an_anchor_and_keeps_absolute_positions(monkeypatch):
    """Anchor-plus-rolling window at the pipeline level: with
    local_attn_size=6 and sink_size=2 at 2-frame chunks (3-chunk window,
    1-chunk anchor), a 5-chunk rollout evicts before chunks 3 and 4 — each
    eviction keeps the first cached chunk and drops the oldest chunk after
    it — so every denoise pass sees at most (window - 1) chunks of cached
    history, always including the anchor, while kv_cache_start_frame stays
    ABSOLUTE (eviction moves the window, never the coordinate system). A
    training-length rollout is asserted eviction-free by
    test_every_rollout_gets_one_fresh_cache_and_training_length_never_evicts."""
    window_chunks, sink_chunks = 3, 1
    evictions = []
    original_evict = RolloutKvCache.evict_oldest_gen_chunk

    def counting_evict(self, num_anchor_chunks=0):
        evictions.append((num_anchor_chunks, self._gen_chunk_frames[0][num_anchor_chunks]))
        return original_evict(self, num_anchor_chunks)

    monkeypatch.setattr(RolloutKvCache, "evict_oldest_gen_chunk", counting_evict)
    captured = []
    _spy_transformer_forwards(monkeypatch, captured)
    pipe = _kv_cache_pipe(ar_frames_per_chunk=2, local_attn_size=6, sink_size=2)
    assert pipe.ar_window_chunks == window_chunks and pipe.ar_sink_chunks == sink_chunks
    inputs = _rollout_inputs(latent_frames=10)  # 5 chunks > the 3-chunk window

    # Rollouts longer than the window are inference-only (eval mode, no
    # grad): training rollouts must fit the window and assert otherwise.
    pipe.transformer.eval()
    with torch.no_grad():
        prediction = _run_rollout(pipe, inputs, num_inference_steps=1)

    assert prediction.shape == inputs["condition"].shape
    # Chunks 0-2 fill the window; before chunks 3 and 4 the oldest chunk
    # AFTER the 1-chunk anchor is evicted — the spy records
    # (num_anchor_chunks, evicted (start_frame, num_frames)): chunk 1 =
    # (2, 2), then chunk 2 = (4, 2). Chunk 0 is the anchor and never leaves.
    assert evictions == [(sink_chunks, (2, 2)), (sink_chunks, (4, 2))]
    denoise_calls = [call for call in captured if not call.get("kv_cache_seed", False)]
    assert [call["kv_cache_start_frame"] for call in denoise_calls] == [0, 2, 4, 6, 8]
    for call in denoise_calls:
        assert (
            call["kv_cache"].num_cached_chunks <= window_chunks - 1
        ), "a denoise pass saw more cached history than the attention window allows"
        if call["kv_cache"].num_cached_chunks:
            assert all(
                layer_records[0] == (0, 2) for layer_records in call["kv_cache"]._gen_chunk_frames
            ), "the anchor chunk must stay at the front of the retained history"
    # The final retained history is [anchor, most recent chunk]; absolute
    # coverage keeps counting evicted frames.
    assert all(layer_records == [(0, 2), (6, 2)] for layer_records in captured[-1]["kv_cache"]._gen_chunk_frames)
    assert captured[-1]["kv_cache"].total_stored_frames == 8


def test_ar_rollout_with_unlimited_local_attn_size_never_evicts(monkeypatch):
    """local_attn_size == -1 disables eviction: a 5-chunk rollout keeps every
    seeded chunk in the cache."""
    evictions = []
    original_evict = RolloutKvCache.evict_oldest_gen_chunk

    def counting_evict(self, num_anchor_chunks=0):
        evictions.append(num_anchor_chunks)
        return original_evict(self, num_anchor_chunks)

    monkeypatch.setattr(RolloutKvCache, "evict_oldest_gen_chunk", counting_evict)
    captured = []
    _spy_transformer_forwards(monkeypatch, captured)
    pipe = _kv_cache_pipe(ar_frames_per_chunk=2, local_attn_size=-1, sink_size=0)
    assert pipe.ar_window_chunks is None
    inputs = _rollout_inputs(latent_frames=10)

    with torch.enable_grad():
        prediction = _run_rollout(pipe, inputs, num_inference_steps=1)

    assert prediction.shape == inputs["condition"].shape
    assert not evictions
    # Every chunk except the global last was seeded and retained.
    assert all(
        layer_records == [(0, 2), (2, 2), (4, 2), (6, 2)]
        for layer_records in captured[-1]["kv_cache"]._gen_chunk_frames
    )


def test_generate_samples_always_routes_through_the_ar_rollout(monkeypatch):
    """The KV-cache rollout pipeline has exactly one rollout — the chunked AR
    rollout — and the chunk size is a required constructor argument, so
    forward_inference/denoise_to_latents (DMD validation) inherit the same
    routing unconditionally."""
    captured = []
    _spy_transformer_forwards(monkeypatch, captured)
    inputs = _rollout_inputs(latent_frames=1)

    result = _run_rollout(_kv_cache_pipe(ar_frames_per_chunk=1), inputs, num_inference_steps=1)
    assert result.shape == inputs["condition"].shape
    assert captured, "the rollout must reach the transformer"
    assert all(isinstance(call["kv_cache"], RolloutKvCache) for call in captured)

    # The chunk size cannot be omitted or disabled.
    with pytest.raises(TypeError):
        _kv_cache_pipe()
    with pytest.raises(ValueError, match="must be a positive latent-frame count"):
        _kv_cache_pipe(ar_frames_per_chunk=0)


def test_ar_rollout_rejects_indivisible_frames_and_odd_cp_sizes():
    pipe = _kv_cache_pipe(ar_frames_per_chunk=2)
    inputs = _rollout_inputs(latent_frames=3)

    with pytest.raises(AssertionError, match="must be divisible by ar_frames_per_chunk"):
        _run_rollout(pipe, inputs, num_inference_steps=1, with_attention_mask=False)

    # CP sizes must divide the chunk size so every chunk pass shards
    # evenly; a size-3 mesh cannot shard 2-frame chunks.
    cp_pipe = _kv_cache_pipe(ar_frames_per_chunk=2)
    cp_pipe.transformer.enable_context_parallel(_FakeCpMesh(rank=0, size=3))
    cp_inputs = _rollout_inputs(latent_frames=4)
    with pytest.raises(AssertionError, match="must divide ar_frames_per_chunk"):
        _run_rollout(cp_pipe, cp_inputs, num_inference_steps=1, with_attention_mask=False)


def test_ar_chunking_reports_the_chunk_size_as_the_cp_divisor():
    """Hybrid task+CP validation reads CP sizing from these pipeline
    methods. Every AR cache pass carries exactly ar_frames_per_chunk
    frames, so CP sizes that divide the chunk size divide every pass;
    head-gcd divisors (the tiny attention is 4/4 heads) additionally run
    padded chunks with the head-sharded cache, and the chooser weighs
    candidates by per-chunk frames."""
    pipe = _kv_cache_pipe(ar_frames_per_chunk=7)
    assert pipe._cp_frame_divisor(21) == 7
    assert pipe.cp_size_candidates(21, 8) == [1, 2, 4, 7]
    assert pipe.cp_pass_latent_frames(21) == 7
    # Blockwise cache state cannot be served by generic pre-sliced inference.
    assert not pipe.supports_local_cp_inference()


def test_get_kv_cache_pipe_forwards_the_same_chunk_and_window_options_to_both_families(monkeypatch):
    """--frames_per_block (reused, no new flag) drives ar_frames_per_chunk
    for both families' KV-cache pipes, and --local_attn_size/--sink_size and
    --truncate_rollout_gradients reach both pipes as the same options, read
    straight off the args namespace (eval namespaces set False
    explicitly)."""
    from types import SimpleNamespace

    from model_training.model_profiles import COSMOS3_NANO_MODEL_ID, WAN22_TI2V_5B_MODEL_ID
    from model_training.utils import train_utils

    constructed = []
    # Same list for both families: this test asserts the interleaved call order.
    patch_kv_cache_pipeline_factories(monkeypatch, constructed, constructed)

    cosmos3_args = SimpleNamespace(
        model_id=COSMOS3_NANO_MODEL_ID,
        gradient_checkpointing=False,
        attention_backend=None,
        frames_per_block=7,
        local_attn_size=21,
        sink_size=7,
    )
    wan_args = SimpleNamespace(
        model_id=WAN22_TI2V_5B_MODEL_ID,
        gradient_checkpointing=False,
        attention_backend=None,
        frames_per_block=7,
        local_attn_size=21,
        sink_size=7,
        checkpoint_every_n_blocks=1,
    )

    train_utils.get_kv_cache_pipe(cosmos3_args, "cpu", truncate_rollout_gradients=True)
    train_utils.get_kv_cache_pipe(wan_args, "cpu")

    cosmos3_kwargs, wan_kwargs = constructed
    assert "frames_per_block" not in cosmos3_kwargs and "frames_per_block" not in wan_kwargs
    for kwargs in (cosmos3_kwargs, wan_kwargs):
        assert kwargs["ar_frames_per_chunk"] == 7
        assert kwargs["local_attn_size"] == 21
        assert kwargs["sink_size"] == 7
    assert cosmos3_kwargs["truncate_rollout_gradients"] is True
    assert wan_kwargs["truncate_rollout_gradients"] is False


def test_ar_chunking_constructor_validation_and_s9_guard_are_independent():
    """ar_frames_per_chunk is a rollout contract of the KV-cache rollout
    subclass; the parent's diffusion-forcing validation (frames_per_block) is
    untouched and the pipeline still owns causal_block_frames=None."""
    with pytest.raises(ValueError, match="ar_frames_per_chunk"):
        _kv_cache_pipe(ar_frames_per_chunk=0)

    pipe = _kv_cache_pipe(ar_frames_per_chunk=7)
    assert pipe.ar_frames_per_chunk == 7
    assert pipe.causal_block_frames is None


# ---------------------------------------------------------------------------
# CPU-staged conditioning. run_inference keeps the full-clip opacity and
# camera tensors on the host for KV-cache eval (stages_conditioning_per_chunk);
# the rollout must hand the transformer per-chunk slices on the pipe's device.
# A VAE double on the meta device makes the two sides distinguishable on a
# CPU-only box; meta cannot run the scheduler, so the denoise loop is stubbed
# to one transformer call per chunk.
# ---------------------------------------------------------------------------


class _MetaDeviceVae(_TinyCosmosVae):
    device = torch.device("meta")


class _RecordingTransformer(torch.nn.Module):
    cp_mesh = None

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def new_rollout_kv_cache(self, *, truncate_rollout_gradients: bool):
        return SimpleNamespace(evict_to_window=lambda window_chunks, sink_chunks: None)

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return (kwargs["hidden_states"],)


def _meta_device_pipe() -> ArtifixerCosmos3KvCachePipeline:
    pipe = ArtifixerCosmos3KvCachePipeline(
        vae=_MetaDeviceVae(),
        transformer=_TinyProcessorCosmosBaseTransformer(),
        gradient_checkpointing=False,
        ar_frames_per_chunk=2,
        local_attn_size=21,
        sink_size=7,
    )
    pipe.eval()
    return pipe


def _single_call_denoise_loop(
    scheduler, timesteps, exit_flag, latents, *, call_transformer, draw_fresh_latents, device
):
    return call_transformer(latents, timesteps[exit_flag], True)


def test_rollout_moves_cpu_staged_conditioning_to_the_pipe_device(monkeypatch):
    """With opacity / camera_rays / w2cs / Ks resident on the CPU, every
    transformer call (denoise passes and cache seeds alike) receives the
    per-chunk slices on the pipe's device, and the rollout output lives
    there too."""
    pipe = _meta_device_pipe()
    pipe.transformer = _RecordingTransformer()
    device = pipe.vae.device
    monkeypatch.setattr(cosmos3_kv_module, "run_chunk_denoise_loop", _single_call_denoise_loop)
    inputs = _rollout_inputs(latent_frames=4)
    assert all(inputs[name].device.type == "cpu" for name in ("rendered_opacity", "camera_rays", "w2cs", "Ks"))

    out = pipe.generate_samples_from_batch(
        inputs["condition"].to(device),
        inputs["rendered_opacity"],
        None,
        inputs["camera_rays"],
        inputs["w2cs"],
        None,
        inputs["Ks"],
        None,
        inputs["encoded_prompt"].to(device),
        4,
        False,
        prompt_attention_mask=inputs["prompt_attention_mask"].to(device),
    )

    calls = pipe.transformer.calls
    # chunk 0 denoise, chunk 0 cache seed, chunk 1 denoise (the last chunk is never seeded)
    assert [call.get("kv_cache_seed", False) for call in calls] == [False, True, False]
    for call in calls:
        for key in ("hidden_states", "opacity", "camera_rays", "w2cs", "Ks"):
            assert call[key].device == device, (key, call[key].device)
    assert out.device == device and out.shape == inputs["condition"].shape


def test_encode_video_frames_accepts_cpu_staged_frames(monkeypatch):
    """rgb_to_latents places the frames on the VAE's device, so eval may keep
    the rendered clip on the host: the VAE sees device-resident frames and
    the latents come back on the device."""
    pipe = _meta_device_pipe()
    device = pipe.vae.device
    seen: list[torch.device] = []

    def fake_encode(frames):
        seen.append(frames.device)
        return SimpleNamespace(latents=torch.zeros(1, LATENT_CHANNELS, 1, LATENT_HW, LATENT_HW, device=device))

    monkeypatch.setattr(pipe.vae, "encode", fake_encode, raising=False)

    latents = pipe.encode_video_frames(torch.rand(1, 1, 3, PIXEL_HW, PIXEL_HW))

    assert seen == [device]
    assert latents.device == device
