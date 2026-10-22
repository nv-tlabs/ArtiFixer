# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan rollout tests on the shared RolloutKvCache, on a real reduced-config
WanTransformer3DModel: the cross-chunk gradient path (mode-parameterized),
gradient-checkpointing recompute equality, per-chunk final-call storage,
windowed eviction, and the write-once text/neighbor context slots."""

import types

import pytest
import torch
from diffusers import WanTransformer3DModel
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._pytree import tree_flatten, tree_unflatten

import model_training.pipeline.wan_kv_cache_pipeline as wan_kv_module
from model_training.net.rollout_kv_cache import RolloutKvCache
from model_training.net.wan_transformer import ArtifixerTransformer
from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline

pytestmark = pytest.mark.slow

_VAE_SPATIAL = 8
_VAE_TEMPORAL = 4
_LATENT_HW = 4
_PIXEL_HW = _LATENT_HW * _VAE_SPATIAL

# Reduced Wan config: production latent contract (16 channels, patch (1, 2,
# 2)), everything else shrunk.
_REDUCED_WAN_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=16,
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=32,
    ffn_dim=48,
    num_layers=2,
    rope_max_seq_len=256,
)


class _TinyWanVae(torch.nn.Module):
    device = torch.device("cpu")
    dtype = torch.float32
    config = types.SimpleNamespace(
        scale_factor_temporal=_VAE_TEMPORAL,
        scale_factor_spatial=_VAE_SPATIAL,
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
        z_dim=16,
    )


def _kv_cache_pipe(
    *,
    local_attn_size: int = -1,
    sink_size: int = 0,
    gradient_checkpointing: bool = False,
    truncate_rollout_gradients: bool = False,
    vae: torch.nn.Module | None = None,
) -> ArtifixerKvCachePipeline:
    torch.manual_seed(0)
    return ArtifixerKvCachePipeline(
        vae=_TinyWanVae() if vae is None else vae,
        transformer=WanTransformer3DModel(**_REDUCED_WAN_CONFIG),
        tokenizer=None,
        text_encoder=None,
        ar_frames_per_chunk=2,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        gradient_checkpointing=gradient_checkpointing,
        checkpoint_every_n_blocks=1,
        expand_timesteps=False,
        attention_backend="native",
        truncate_rollout_gradients=truncate_rollout_gradients,
    )


def _rollout_inputs(latent_frames: int, *, opacity_value: float | None = 1.0, with_neighbors: bool = False) -> dict:
    torch.manual_seed(7)
    pixel_frames = (latent_frames - 1) * _VAE_TEMPORAL + 1
    opacity = (
        torch.rand(1, pixel_frames, _PIXEL_HW, _PIXEL_HW)
        if opacity_value is None
        else torch.full((1, pixel_frames, _PIXEL_HW, _PIXEL_HW), opacity_value)
    )
    return dict(
        condition=torch.randn(1, 16, latent_frames, _LATENT_HW, _LATENT_HW),
        rendered_opacity=opacity,
        neighbors_condition=torch.randn(1, 16, 1, _LATENT_HW, _LATENT_HW) if with_neighbors else None,
        camera_rays=torch.randn(1, latent_frames, _PIXEL_HW, _PIXEL_HW, 6),
        w2cs=torch.eye(4).expand(1, latent_frames, 4, 4).clone(),
        neighbor_w2cs=torch.eye(4).expand(1, 1, 4, 4).clone() if with_neighbors else None,
        Ks=torch.eye(3).expand(1, latent_frames, 3, 3).clone(),
        neighbor_Ks=torch.eye(3).expand(1, 1, 3, 3).clone() if with_neighbors else None,
        encoded_prompt=torch.randn(1, 5, 32),
    )


def _run_rollout(pipe, inputs: dict, num_inference_steps: int, *, condition: torch.Tensor | None = None):
    torch.manual_seed(1234)
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
        False,
    )


def _spy_transformer_forwards(monkeypatch, captured: list):
    original_forward = ArtifixerTransformer.forward

    def spy_forward(self, **kwargs):
        captured.append({**kwargs, "grad_enabled": torch.is_grad_enabled()})
        return original_forward(self, **kwargs)

    monkeypatch.setattr(ArtifixerTransformer, "forward", spy_forward)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_cache_stage_survives_rebuilding_block_arguments(monkeypatch, gradient_checkpointing):
    pipe = _kv_cache_pipe(gradient_checkpointing=gradient_checkpointing)
    inputs = _rollout_inputs(latent_frames=4, opacity_value=1.0, with_neighbors=True)
    condition = inputs["condition"].requires_grad_(True)
    for block in pipe.transformer.blocks:
        original_forward = block.forward

        def rebuilt_forward(*args, _forward=original_forward, **kwargs):
            leaves, structure = tree_flatten((args, kwargs))
            rebuilt_args, rebuilt_kwargs = tree_unflatten(leaves, structure)
            return _forward(*rebuilt_args, **rebuilt_kwargs)

        monkeypatch.setattr(block, "forward", rebuilt_forward)
    with sdpa_kernel([SDPBackend.MATH]):
        output = _run_rollout(pipe, inputs, num_inference_steps=1, condition=condition)
        output[:, :, 2:].square().mean().backward()
    assert condition.grad is not None and torch.isfinite(condition.grad).all()
    assert torch.count_nonzero(condition.grad[:, :, :2]) > 0


@pytest.mark.parametrize("truncate_rollout_gradients", [False, True])
def test_rollout_context_gradients_follow_the_truncation_flag(truncate_rollout_gradients):
    """The truncation flag's contract on the Wan rollout: by default a later
    chunk's loss reaches earlier chunks' inputs through the cached K/V (the
    chunk's final-call store keeps the graph); with truncation the same
    gradient is exactly zero. With opacity == 1, z_mix == condition, so chunk
    1's output gradient w.r.t. chunk 0's condition frames is the probe."""
    pipe = _kv_cache_pipe(truncate_rollout_gradients=truncate_rollout_gradients)
    inputs = _rollout_inputs(latent_frames=4, opacity_value=1.0)
    condition = inputs["condition"].requires_grad_(True)

    with torch.enable_grad(), sdpa_kernel([SDPBackend.MATH]):
        prediction = _run_rollout(pipe, inputs, num_inference_steps=1)
        (condition_grad,) = torch.autograd.grad(prediction[:, :, 2:].sum(), condition)

    assert torch.count_nonzero(condition_grad[:, :, 2:]) > 0
    if truncate_rollout_gradients:
        assert (
            torch.count_nonzero(condition_grad[:, :, :2]) == 0
        ), "truncation must cut the gradient path through cached K/V at the chunk boundary"
    else:
        assert (
            torch.count_nonzero(condition_grad[:, :, :2]) > 0
        ), "chunk 1's loss must reach chunk 0 through the cached K/V in the default mode"

    # Non-vacuity control: chunk 1's VALUES depend on chunk 0 through the
    # cache either way, so the truncation-mode zero gradient can only come
    # from the detached store.
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        perturbed_condition = condition.detach().clone()
        perturbed_condition[:, :, :2] += 1.0
        baseline = _run_rollout(pipe, inputs, num_inference_steps=1, condition=condition.detach())
        perturbed = _run_rollout(pipe, inputs, num_inference_steps=1, condition=perturbed_condition)
    assert not torch.equal(baseline[:, :, 2:], perturbed[:, :, 2:])


def test_rollout_grads_match_with_and_without_gradient_checkpointing():
    """A checkpointed block's backward-time recompute reads the cache after
    later chunks were appended; the snapshotted history count must make it
    read exactly the slice its forward read. Pin: parameter gradients of a
    2-chunk default-mode rollout are bitwise equal with and without
    checkpointing."""
    inputs = _rollout_inputs(latent_frames=4, opacity_value=None)

    def run(gradient_checkpointing: bool) -> dict[str, torch.Tensor | None]:
        pipe = _kv_cache_pipe(gradient_checkpointing=gradient_checkpointing)
        with torch.enable_grad(), sdpa_kernel([SDPBackend.MATH]):
            prediction = _run_rollout(pipe, inputs, num_inference_steps=2)
            prediction.square().sum().backward()
        return {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in pipe.transformer.named_parameters()
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


def test_rollout_stores_each_chunks_final_call_once_except_the_last(monkeypatch):
    """Pipeline plumbing: every denoise call carries the chunk at its
    absolute frame offset against ONE cache; only each chunk's final (exit)
    call stores its K/V, and the global last chunk is never stored (no later
    reader). After the rollout the cache holds one record per stored chunk at
    the chunk's global frames."""
    captured = []
    _spy_transformer_forwards(monkeypatch, captured)
    pipe = _kv_cache_pipe()
    pipe.transformer.eval()
    inputs = _rollout_inputs(latent_frames=6, opacity_value=None, with_neighbors=True)

    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        prediction = _run_rollout(pipe, inputs, num_inference_steps=2)

    assert prediction.shape == inputs["condition"].shape
    assert bool(torch.isfinite(prediction).all())
    # 3 chunks x 2 denoise calls; the store flag marks the exit call of every
    # chunk except the last.
    assert [call["kv_cache_store_chunk"] for call in captured] == [False, True, False, True, False, False]
    assert [call["frame_offset"] for call in captured] == [0, 0, 2, 2, 4, 4]
    caches = {id(call["kv_cache"]) for call in captured}
    assert len(caches) == 1, "one cache per rollout"
    cache = captured[-1]["kv_cache"]
    assert cache._gen_chunk_frames == [[(0, 2), (2, 2)]] * len(pipe.transformer.blocks)
    # The no-grad passes filled the write-once context slots for every layer.
    for layer_index in range(len(pipe.transformer.blocks)):
        assert cache.has_text_kv(layer_index)
        assert cache.has_neighbor_kv(layer_index)


def test_windowed_rollout_evicts_after_anchor_and_training_must_fit_the_window(monkeypatch):
    """Anchor-plus-rolling window at the pipeline level: with
    local_attn_size=6 and sink_size=2 at 2-frame chunks (3-chunk window,
    1-chunk anchor), a 5-chunk eval rollout evicts before chunks 3 and 4 —
    dropping the oldest chunk after the anchor — so every denoise pass sees
    at most (window - 1) cached chunks, and absolute coverage keeps counting
    evicted frames. Under grad or train mode the same rollout is rejected:
    eviction during training would leave gradient-checkpointing recomputes
    reading the wrong history."""
    evictions = []
    original_evict = RolloutKvCache.evict_oldest_gen_chunk

    def counting_evict(self, num_anchor_chunks=0):
        evictions.append((num_anchor_chunks, self._gen_chunk_frames[0][num_anchor_chunks]))
        return original_evict(self, num_anchor_chunks)

    monkeypatch.setattr(RolloutKvCache, "evict_oldest_gen_chunk", counting_evict)
    captured = []
    _spy_transformer_forwards(monkeypatch, captured)
    pipe = _kv_cache_pipe(local_attn_size=6, sink_size=2)
    assert pipe.ar_window_chunks == 3 and pipe.ar_sink_chunks == 1
    inputs = _rollout_inputs(latent_frames=10, opacity_value=None)

    pipe.transformer.eval()
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        prediction = _run_rollout(pipe, inputs, num_inference_steps=1)

    assert prediction.shape == inputs["condition"].shape
    assert evictions == [(1, (2, 2)), (1, (4, 2))]
    cache = captured[-1]["kv_cache"]
    assert all(layer_records == [(0, 2), (6, 2)] for layer_records in cache._gen_chunk_frames)
    assert cache.total_stored_frames == 8

    pipe.transformer.train()
    with torch.enable_grad(), pytest.raises(AssertionError, match="must not evict cached history"):
        _run_rollout(pipe, inputs, num_inference_steps=1)


# ---------------------------------------------------------------------------
# CPU-staged conditioning. run_inference keeps the full-clip opacity and
# camera tensors on the host for KV-cache eval (stages_conditioning_per_chunk);
# the rollout must hand the transformer per-chunk slices on the pipe's device.
# A VAE double on the meta device makes the two sides distinguishable on a
# CPU-only box; meta cannot run the scheduler, so the denoise loop is stubbed
# to one transformer call per chunk.
# ---------------------------------------------------------------------------


class _MetaDeviceWanVae(_TinyWanVae):
    device = torch.device("meta")


class _RecordingWanTransformer(torch.nn.Module):
    cp_mesh = None
    cp_rank = 0

    def __init__(self, num_blocks: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(num_blocks))
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return (kwargs["hidden_states"],)


def test_rollout_moves_cpu_staged_conditioning_to_the_pipe_device(monkeypatch):
    """With the condition (the Wan denoise keeps it on the host) and the
    opacity / camera_rays / w2cs / Ks resident on the CPU, every transformer
    call receives the per-chunk slices on the pipe's device, and the rollout
    output lives there too."""
    pipe = _kv_cache_pipe(vae=_MetaDeviceWanVae())
    pipe.transformer = _RecordingWanTransformer(num_blocks=_REDUCED_WAN_CONFIG["num_layers"])
    pipe.eval()
    device = pipe.vae.device
    monkeypatch.setattr(
        wan_kv_module,
        "run_chunk_denoise_loop",
        lambda scheduler, timesteps, exit_flag, latents, *, call_transformer, draw_fresh_latents, device: (
            call_transformer(latents, timesteps[exit_flag], True)
        ),
    )
    inputs = _rollout_inputs(latent_frames=4)
    assert all(
        inputs[name].device.type == "cpu" for name in ("condition", "rendered_opacity", "camera_rays", "w2cs", "Ks")
    )
    inputs["encoded_prompt"] = inputs["encoded_prompt"].to(device)

    out = _run_rollout(pipe, inputs, num_inference_steps=4)

    calls = pipe.transformer.calls
    assert len(calls) == 2  # one stubbed denoise call per 2-frame chunk
    for call in calls:
        for key in ("hidden_states", "opacity", "camera_rays", "w2cs", "Ks"):
            assert call[key].device == device, (key, call[key].device)
    assert out.device == device and out.shape == inputs["condition"].shape
