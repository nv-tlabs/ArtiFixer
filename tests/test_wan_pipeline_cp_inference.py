# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-input CP equivalence for the standard Wan pipeline.

``ArtifixerPipeline.forward_inference`` must accept a complete clip while a CP
mesh is active (run_inference and trainer validation call it exactly like
that): each rank slices its temporal shard, denoises it through the real
``forward_inference_local_cp``, then all-gathers the latent shards and decodes,
so every rank returns the CP=1 video.

The test drives the real pipeline methods and a real reduced-config
WanTransformer3DModel on CPU with a fake 2-rank mesh and monkeypatched
collectives (the same doubles as tests/test_cp_attention.py and
tests/test_kv_cache_pipeline.py).  Only collectives and the attention backend
are mocked:

* ``dispatch_attention_fn`` returns the (RoPE'd) query — a token-local
  attention backend, so every rank's trajectory must reproduce the CP=1 values
  on that rank's tokens exactly, including each rank's RoPE frame offsets and
  opacity/camera/noise shard alignment.  The real cross-rank Ulysses attention
  numerics are pinned separately by tests/test_cp_attention.py.
* ``all_to_all_with_grad`` chunks over the scatter dim and concatenates over
  the gather dim, so the head-scatter/seq-gather round trip restores its input
  bitwise — exact under a token-local attention backend.
* The noise scatter (isend/irecv) runs through a mailbox, and the final latent
  all-gather runs record-then-replay: a first pass per rank captures each
  rank's latent shard (rank runs are sequential, so a rank's gather cannot see
  a peer computed later), then a second deterministic pass serves the peer
  shards through the gather and returns full videos.

No GPU required -- run with:
    pytest tests/test_wan_pipeline_cp_inference.py -v
"""

import types

import pytest
import torch
from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

import model_training.net.context_parallel as net_context_parallel_mod
import model_training.net.wan_transformer as transformer_mod
import model_training.pipeline.pipeline_base as pipeline_base_mod
from model_training.pipeline.wan_pipeline import ArtifixerPipeline
from tests._helpers import REPO_ROOT, _ChunkFaithfulWanVae, _GroupedFakeCpMesh

pytestmark = pytest.mark.slow

_WORLD_SIZE = 2
_VAE_SPATIAL = 8
_VAE_TEMPORAL = 4
_LATENT_HW = 4
_PIXEL_HW = _LATENT_HW * _VAE_SPATIAL
_TOTAL_LATENT_FRAMES = 4
_RAW_FRAMES = 1 + _VAE_TEMPORAL * (_TOTAL_LATENT_FRAMES - 1)
_NOISE_SEED = 1234

# Reduced Wan config (tests/test_wan_rollout_cache.py): production latent
# contract (16 channels, patch (1, 2, 2)), everything else shrunk.
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


class _ShardRecorded(Exception):
    """Raised by the recording all-gather fake once the shard is captured."""


def _pipe() -> ArtifixerPipeline:
    torch.manual_seed(0)
    return ArtifixerPipeline(
        vae=_ChunkFaithfulWanVae(z_dim=16, scale_factor_spatial=_VAE_SPATIAL, scale_factor_temporal=_VAE_TEMPORAL),
        scheduler=UniPCMultistepScheduler(),
        transformer=WanTransformer3DModel(**_REDUCED_WAN_CONFIG),
        tokenizer=None,
        text_encoder=None,
        default_negative_prompt_path=REPO_ROOT / "default_negative_prompt.pt",
        frames_per_block=None,
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
        expand_timesteps=False,
        attention_backend="native",
    )


def _full_clip_inputs() -> dict:
    torch.manual_seed(7)
    return dict(
        rendered_rgb=torch.rand(1, _RAW_FRAMES, 3, _PIXEL_HW, _PIXEL_HW),
        rendered_opacity=torch.rand(1, _RAW_FRAMES, _PIXEL_HW, _PIXEL_HW),
        neighbors=None,
        # Camera tensors at latent temporal resolution, as both CP consumers
        # (run_inference, trainer validation) pass them.
        camera_rays=torch.randn(1, _TOTAL_LATENT_FRAMES, _PIXEL_HW, _PIXEL_HW, 6),
        w2cs=torch.eye(4).expand(1, _TOTAL_LATENT_FRAMES, 4, 4).clone(),
        neighbor_w2cs=None,
        Ks=torch.eye(3).expand(1, _TOTAL_LATENT_FRAMES, 3, 3).clone(),
        neighbor_Ks=None,
        prompt=torch.randn(1, 5, 32),
        negative_prompt=torch.randn(1, 5, 32),
        num_inference_steps=2,
        text_guidance_scale=2.0,
    )


def _token_local_dispatch(query, key, value, **kwargs):
    """Token-local attention backend: the output is the RoPE'd query, so rank
    shards must reproduce the CP=1 per-token values exactly."""
    return query


def _chunk_preserving_all_to_all(tensor, scatter_dim, gather_dim, group=None):
    """Single-process Ulysses stand-in whose head-scatter/seq-gather round
    trip restores the input bitwise (chunks keep their order, unlike the
    repeat-chunk-0 double in tests/test_kv_cache_pipeline.py)."""
    assert group is not None, "the production CP path always exchanges through the mesh group"
    return torch.cat(tensor.chunk(_WORLD_SIZE, dim=scatter_dim), dim=gather_dim)


def test_forward_inference_under_a_fake_cp_mesh_matches_cp_disabled(monkeypatch):
    """Full-input forward_inference with an active 2-rank CP mesh must return,
    on every rank, the same video the same pipe produces with CP disabled."""
    monkeypatch.setattr(transformer_mod, "dispatch_attention_fn", _token_local_dispatch)
    monkeypatch.setattr(transformer_mod, "all_to_all_with_grad", _chunk_preserving_all_to_all)

    current_rank = {"value": 0}
    noise_mailbox = {}
    recorded_shards = {}

    def fake_isend(tensor, group=None, group_dst=None):
        noise_mailbox[group_dst] = tensor.detach().clone()
        return types.SimpleNamespace(wait=lambda: None)

    def fake_irecv(tensor, group=None, group_src=None):
        tensor.copy_(noise_mailbox.pop(current_rank["value"]))
        return types.SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(pipeline_base_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pipeline_base_mod.dist, "get_rank", lambda group=None: current_rank["value"])
    monkeypatch.setattr(pipeline_base_mod.dist, "isend", fake_isend)
    monkeypatch.setattr(pipeline_base_mod.dist, "irecv", fake_irecv)

    def record_all_gather(outputs, tensor, group=None):
        recorded_shards[current_rank["value"]] = tensor.detach().clone()
        raise _ShardRecorded

    def replay_all_gather(outputs, tensor, group=None):
        for rank, output in enumerate(outputs):
            output.copy_(tensor if rank == current_rank["value"] else recorded_shards[rank])

    inputs = _full_clip_inputs()

    reference_pipe = _pipe()
    assert reference_pipe.supports_local_cp_inference()
    torch.manual_seed(_NOISE_SEED)
    reference = reference_pipe.forward_inference(**inputs)
    assert reference.shape == (1, _RAW_FRAMES, 3, _PIXEL_HW, _PIXEL_HW)

    def run_rank(rank: int) -> torch.Tensor:
        pipe = _pipe()  # seeded factory: identical weights on every rank
        pipe.transformer.enable_context_parallel(_GroupedFakeCpMesh(size=_WORLD_SIZE, rank=rank))
        current_rank["value"] = rank
        torch.manual_seed(_NOISE_SEED)
        return pipe.forward_inference(**inputs)

    monkeypatch.setattr(net_context_parallel_mod.dist, "all_gather", record_all_gather)
    for rank in range(_WORLD_SIZE):
        with pytest.raises(_ShardRecorded):
            run_rank(rank)
    assert sorted(recorded_shards) == list(range(_WORLD_SIZE))
    frames_per_rank = _TOTAL_LATENT_FRAMES // _WORLD_SIZE
    for shard in recorded_shards.values():
        assert shard.shape == (1, 16, frames_per_rank, _LATENT_HW, _LATENT_HW)

    monkeypatch.setattr(net_context_parallel_mod.dist, "all_gather", replay_all_gather)
    outputs = [run_rank(rank) for rank in range(_WORLD_SIZE)]

    torch.testing.assert_close(outputs[0], reference)
    torch.testing.assert_close(outputs[1], reference)
    # Every rank returns the SAME full video (the deterministic replay makes
    # this exact, not merely close).
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
