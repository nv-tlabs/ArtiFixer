# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-shard CP equivalence for the cosmos3 pipeline.

``forward_inference_local_cp`` — the shared base implementation the trainer's
CP validation calls — must reproduce the CP=1 ``forward_inference`` video from
pre-sliced per-rank inputs: rank 0 scatters shards of the CP=1 noise draw, each
rank denoises only its latent shard (``local_cp_input=True`` with its absolute
``frame_offset``), and the concatenated shards decode to the reference video.

The test drives the real pipeline methods and the real reduced-config cosmos3
transformer on CPU with a fake 2-rank mesh, mirroring
tests/test_wan_pipeline_cp_inference.py: a token-local attention backend makes
per-token trajectories exactly comparable across sharding, the Ulysses
all-to-all is a chunk-preserving single-process stand-in, and the noise
scatter runs through an isend/irecv mailbox.  The prompt and negative prompt
are PADDED token ids with real attention masks at guidance 2.0, so dropping a
mask anywhere on the path shifts ``und_len`` and the mRoPE vision offsets and
fails the equality.  Any collective outside those fakes raises: the local
contract returns rank-local latents with no gather.

No GPU required -- run with:
    pytest tests/test_cosmos3_pipeline_cp_inference.py -v
"""

import types

import pytest
import torch
from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer, UniPCMultistepScheduler

import model_training.net.context_parallel as net_context_parallel_mod
import model_training.net.cosmos3_transformer as transformer_mod
import model_training.pipeline.pipeline_base as pipeline_base_mod
from model_training.common.context_parallel import cp_rank_temporal_ranges, latent_frame_count
from model_training.net.conditioning import slice_camera_tensor_frames
from model_training.pipeline.cosmos3_pipeline import ArtifixerCosmos3Pipeline
from tests._helpers import REDUCED_CONFIG, _ChunkFaithfulWanVae, _GroupedFakeCpMesh

pytestmark = pytest.mark.slow

_WORLD_SIZE = 2
_VAE_SPATIAL = 16
_VAE_TEMPORAL = 4
_LATENT_HW = 4
_PIXEL_HW = _LATENT_HW * _VAE_SPATIAL
_TOTAL_LATENT_FRAMES = 4
_RAW_FRAMES = 1 + _VAE_TEMPORAL * (_TOTAL_LATENT_FRAMES - 1)
_NOISE_SEED = 1234


def _pipe(frames_per_block: int | None = None) -> ArtifixerCosmos3Pipeline:
    torch.manual_seed(0)
    return ArtifixerCosmos3Pipeline(
        vae=_ChunkFaithfulWanVae(z_dim=48, scale_factor_spatial=_VAE_SPATIAL, scale_factor_temporal=_VAE_TEMPORAL),
        scheduler=UniPCMultistepScheduler(),
        transformer=Cosmos3OmniTransformer(**REDUCED_CONFIG),
        gradient_checkpointing=False,
        frames_per_block=frames_per_block,
    )


def _full_clip_inputs() -> dict:
    torch.manual_seed(7)
    return dict(
        rendered_rgb=torch.rand(1, _RAW_FRAMES, 3, _PIXEL_HW, _PIXEL_HW),
        rendered_opacity=torch.rand(1, _RAW_FRAMES, _PIXEL_HW, _PIXEL_HW),
        neighbors=None,
        # Camera rays at pixel spatial / latent temporal resolution, w2cs/Ks
        # at latent temporal resolution — the wrapper's contract.
        camera_rays=torch.randn(1, _TOTAL_LATENT_FRAMES, _PIXEL_HW, _PIXEL_HW, 6),
        w2cs=torch.eye(4).expand(1, _TOTAL_LATENT_FRAMES, 4, 4).clone(),
        neighbor_w2cs=None,
        Ks=torch.eye(3).expand(1, _TOTAL_LATENT_FRAMES, 3, 3).clone(),
        neighbor_Ks=None,
        # Padded ids with real masks: a dropped mask inflates und_len and
        # shifts the mRoPE vision offsets, breaking the CP=1 equality.
        prompt=torch.tensor([[11, 12, 7, 8, 0, 0]]),
        prompt_attention_mask=torch.tensor([[True, True, True, True, False, False]]),
        negative_prompt=torch.tensor([[9, 10, 0, 0]]),
        negative_prompt_attention_mask=torch.tensor([[True, True, False, False]]),
        num_inference_steps=2,
        text_guidance_scale=2.0,
    )


def _token_local_dispatch(query, *args, **kwargs):
    """Token-local attention backend: the output is the RoPE'd query, so rank
    shards must reproduce the CP=1 per-token values exactly."""
    return query


def _chunk_preserving_all_to_all(tensor, scatter_dim, gather_dim, group=None):
    """Single-process Ulysses stand-in whose head-scatter/seq-gather round
    trip restores the input bitwise."""
    assert group is not None, "the production CP path always exchanges through the mesh group"
    return torch.cat(tensor.chunk(_WORLD_SIZE, dim=scatter_dim), dim=gather_dim)


@pytest.mark.parametrize(
    ("prompt_lengths", "negative_prompt_lengths"),
    [
        pytest.param((4,), (2,), id="shorter-negative-prompt"),
        pytest.param((2,), (4,), id="longer-negative-prompt"),
        pytest.param((4, 2), (2, 3), id="mixed-length-batch"),
    ],
)
def test_cfg_video_positions_match_upstream_for_each_prompt(monkeypatch, prompt_lengths, negative_prompt_lengths):
    pipe = _pipe()
    inputs = _full_clip_inputs()
    batch_size = len(prompt_lengths)
    inputs = {
        key: value.repeat(batch_size, *([1] * (value.ndim - 1))) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    reference = types.SimpleNamespace(transformer=pipe.transformer.base_transformer)
    expected_per_step = []
    for key, mask_key, lengths in (
        ("prompt", "prompt_attention_mask", prompt_lengths),
        ("negative_prompt", "negative_prompt_attention_mask", negative_prompt_lengths),
    ):
        # Leave padding in every row so positions must use the actual text
        # length, including for the longest sample in each branch.
        width = max(lengths) + 2
        inputs[mask_key] = torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
        inputs[key] = torch.arange(1, width + 1).repeat(batch_size, 1).masked_fill(~inputs[mask_key], 0)
        for sample_index in range(batch_size):
            ids = inputs[key][sample_index][inputs[mask_key][sample_index]]
            segment = Cosmos3OmniPipeline._prepare_text_segment(reference, ids, device="cpu")
            expected_per_step.append((tuple(ids.tolist()), segment["vision_start_temporal_offset"]))

    observed = []
    prepare_text = pipe.transformer._prepare_text_segment

    def capture_segment(input_ids, *args, **kwargs):
        segment = prepare_text(input_ids, *args, **kwargs)
        observed.append((tuple(input_ids.tolist()), segment["vision_start_temporal_offset"]))
        return segment

    monkeypatch.setattr(pipe.transformer, "_prepare_text_segment", capture_segment)
    monkeypatch.setattr(transformer_mod, "dispatch_attention_fn", _token_local_dispatch)
    pipe.forward_inference(**inputs)

    assert observed == expected_per_step * inputs["num_inference_steps"]


def test_local_cp_inference_matches_cp_disabled(monkeypatch):
    """Per-rank forward_inference_local_cp on trainer-sliced inputs must
    reassemble to the video the same pipe produces with CP disabled."""
    monkeypatch.setattr(transformer_mod, "dispatch_attention_fn", _token_local_dispatch)
    monkeypatch.setattr(transformer_mod, "all_to_all_with_grad", _chunk_preserving_all_to_all)

    current_rank = {"value": 0}
    noise_mailbox = {}

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

    def _no_collectives(*args, **kwargs):
        raise AssertionError("the local-CP inference path must not run collectives outside the Ulysses exchange")

    # The local contract returns rank-local latents: any gather is a bug.
    monkeypatch.setattr(net_context_parallel_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(net_context_parallel_mod.dist, "all_gather", _no_collectives)

    inputs = _full_clip_inputs()

    reference_pipe = _pipe()
    assert reference_pipe.supports_local_cp_inference()
    torch.manual_seed(_NOISE_SEED)
    reference = reference_pipe.forward_inference(**inputs)
    assert reference.shape == (1, _RAW_FRAMES, 3, _PIXEL_HW, _PIXEL_HW)

    local_latents = {}
    for rank in range(_WORLD_SIZE):
        ranges = cp_rank_temporal_ranges(
            rank,
            _WORLD_SIZE,
            total_raw_frames=_RAW_FRAMES,
            temporal_scale=_VAE_TEMPORAL,
            # Full left halo, like Wan's full-clip CP path: the chunk-faithful
            # VAE double makes crop-of-window equal crop-of-full bitwise.
            halo_latent_frames=latent_frame_count(_RAW_FRAMES, _VAE_TEMPORAL),
        )
        latent_start, latent_end = ranges["latent"]
        conditioning_raw_start, conditioning_raw_end = ranges["conditioning_raw"]
        vae_raw_start, vae_raw_end, vae_crop_start, vae_crop_end = ranges["vae_raw"]

        pipe = _pipe()  # seeded factory: identical weights on every rank
        pipe.transformer.enable_context_parallel(_GroupedFakeCpMesh(size=_WORLD_SIZE, rank=rank))
        current_rank["value"] = rank
        torch.manual_seed(_NOISE_SEED)
        local = pipe.forward_inference_local_cp(
            rendered_rgb=inputs["rendered_rgb"][:, vae_raw_start:vae_raw_end],
            rendered_opacity=inputs["rendered_opacity"][:, conditioning_raw_start:conditioning_raw_end],
            neighbors=None,
            camera_rays=slice_camera_tensor_frames(inputs["camera_rays"], latent_start, latent_end),
            w2cs=slice_camera_tensor_frames(inputs["w2cs"], latent_start, latent_end),
            neighbor_w2cs=None,
            Ks=slice_camera_tensor_frames(inputs["Ks"], latent_start, latent_end),
            neighbor_Ks=None,
            prompt=inputs["prompt"],
            local_frame_offset=latent_start,
            local_vae_crop=(vae_crop_start, vae_crop_end),
            prompt_attention_mask=inputs["prompt_attention_mask"],
            negative_prompt=inputs["negative_prompt"],
            negative_prompt_attention_mask=inputs["negative_prompt_attention_mask"],
            num_inference_steps=inputs["num_inference_steps"],
            text_guidance_scale=inputs["text_guidance_scale"],
        )
        # The local contract returns only this rank's frames — no gather.
        assert local.shape[2] == _TOTAL_LATENT_FRAMES // _WORLD_SIZE
        local_latents[rank] = local

    video = reference_pipe.decode_latents_to_video(torch.cat([local_latents[0], local_latents[1]], dim=2))
    torch.testing.assert_close(video, reference)


def test_local_cp_inference_refuses_block_causal_configs():
    """The wrapper's block-causal mask indexes the full clip, so DF pipes must
    fail at the pipeline boundary instead of deep inside the wrapper."""
    pipe = _pipe(frames_per_block=2)
    assert not pipe.supports_local_cp_inference()
    with pytest.raises(AssertionError, match="cannot serve pre-sliced CP shards"):
        pipe.forward_inference_local_cp(
            rendered_rgb=torch.rand(1, _RAW_FRAMES, 3, _PIXEL_HW, _PIXEL_HW),
            rendered_opacity=torch.rand(1, _RAW_FRAMES, _PIXEL_HW, _PIXEL_HW),
            neighbors=None,
            camera_rays=torch.randn(1, _TOTAL_LATENT_FRAMES, _PIXEL_HW, _PIXEL_HW, 6),
            w2cs=torch.eye(4).expand(1, _TOTAL_LATENT_FRAMES, 4, 4).clone(),
            neighbor_w2cs=None,
            Ks=torch.eye(3).expand(1, _TOTAL_LATENT_FRAMES, 3, 3).clone(),
            neighbor_Ks=None,
            prompt=torch.tensor([[11, 12]]),
            local_frame_offset=0,
            local_vae_crop=(0, 2),
        )
