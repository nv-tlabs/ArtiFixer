# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor preparation and sampling shared by live validation and offline inference.

Model preparation, rank ownership, temporal padding/sharding and output writing
remain with their callers. In particular, replicated inference never performs
training's model-only FSDP2 preparation through this module.
"""

from dataclasses import dataclass

import torch

from model_training.prompt_formats import prompt_payload_for_text_dropout


@dataclass(frozen=True)
class SamplingConfig:
    num_inference_steps: int = 50
    text_guidance_scale: float | None = None
    max_neighbors_per_encode: int | None = 1
    progress_bar_leave: bool | None = False
    vae_chunk_halo_latent_frames: int | None = None


def prepare_generation_inputs(
    data: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batched: bool,
    stage_conditioning_on_cpu: bool = False,
    stage_neighbors_on_cpu: bool = False,
    use_neighbors: bool = True,
    zero_prompt: bool = False,
) -> dict:
    conditioning_device = torch.device("cpu") if stage_conditioning_on_cpu else device
    neighbor_device = torch.device("cpu") if stage_neighbors_on_cpu else device

    def move(value, target=device, dtype=None):
        if not isinstance(value, torch.Tensor):
            return value
        if not batched:
            value = value.unsqueeze(0)
        return value.to(device=target, dtype=dtype, non_blocking=True)

    prompt = prompt_payload_for_text_dropout(data, drop_text=zero_prompt)
    inputs = {
        "rendered_rgb": move(data["rgb_rendered"], conditioning_device),
        "rendered_opacity": move(data.get("opacity"), conditioning_device, dtype),
        "neighbors": move(data.get("rgb_neighbors"), neighbor_device) if use_neighbors else None,
        "camera_rays": move(data["camera_rays"], conditioning_device, dtype),
        "w2cs": move(data["w2cs"], conditioning_device),
        "Ks": move(data["Ks"], conditioning_device),
        "neighbor_w2cs": move(data.get("neighbor_w2cs")) if use_neighbors else None,
        "neighbor_Ks": move(data.get("neighbor_Ks")) if use_neighbors else None,
        "prompt": move(prompt["encoded_prompt"]),
        "prompt_attention_mask": move(prompt.get("prompt_attention_mask")),
        "negative_prompt": move(prompt.get("encoded_negative_prompt")),
        "negative_prompt_attention_mask": move(prompt.get("negative_prompt_attention_mask")),
    }
    rays = data.get("neighbor_camera_rays") if use_neighbors else None
    if rays is not None:
        inputs["neighbor_camera_rays"] = move(rays, device, dtype)
    return inputs


def generate(
    pipe,
    inputs: dict,
    sampling: SamplingConfig,
    *,
    output_latents: bool = False,
    local_cp: dict | None = None,
    noise_seed: int | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    kwargs = dict(
        inputs,
        num_inference_steps=sampling.num_inference_steps,
        max_neighbors_per_encode=sampling.max_neighbors_per_encode,
        show_progress=show_progress,
    )
    if sampling.progress_bar_leave is not None:
        kwargs["progress_bar_leave"] = sampling.progress_bar_leave
    if sampling.text_guidance_scale is not None:
        kwargs["text_guidance_scale"] = sampling.text_guidance_scale
    if noise_seed is not None:
        kwargs["noise_seed"] = noise_seed
    if local_cp is not None:
        return pipe.forward_inference_local_cp(**kwargs, **local_cp)
    if output_latents:
        if sampling.vae_chunk_halo_latent_frames is not None:
            kwargs["vae_chunk_halo_latent_frames"] = sampling.vae_chunk_halo_latent_frames
        return pipe.denoise_to_latents(**kwargs)
    return pipe.forward_inference(**kwargs)
