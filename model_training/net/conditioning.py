# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-neutral opacity/camera conditioning helpers shared by the Wan and
Cosmos3 transformer wrappers, the rollout pipelines, and the eval entry
point."""

from __future__ import annotations

import torch
from einops import rearrange


def opacity_embedding_input_dim(
    *,
    vae_scale_factor_temporal: int,
    vae_scale_factor_spatial: int,
    patch_height: int,
    patch_width: int,
) -> int:
    """Per-token opacity channel count: one generation token covers a
    ``(sf_s * patch_h) x (sf_s * patch_w)`` pixel window spatially AND
    ``vae_scale_factor_temporal`` pixel frames temporally. The temporal
    factor is easy to miss — spatial-only would be wrong."""
    return (
        vae_scale_factor_temporal * (vae_scale_factor_spatial * patch_height) * (vae_scale_factor_spatial * patch_width)
    )


def camera_embedding_input_dim(
    *,
    vae_scale_factor_spatial: int,
    patch_height: int,
    patch_width: int,
) -> int:
    """Per-token raymap channel count: 6 Plücker channels per pixel of the
    token's ``(sf_s * patch_h) x (sf_s * patch_w)`` spatial window (raymaps
    are at latent temporal resolution, so no temporal factor)."""
    return 6 * (vae_scale_factor_spatial * patch_height) * (vae_scale_factor_spatial * patch_width)


def pad_opacity_first_frame(opacity: torch.Tensor, vae_scale_factor_temporal: int) -> torch.Tensor:
    """Replicate pixel frame 0 to fill the causal VAE's 1+4N first temporal
    window (``[B, T, ...]``, frames on dim 1)."""
    return torch.cat(
        [opacity[:, :1].repeat_interleave(vae_scale_factor_temporal - 1, dim=1), opacity],
        dim=1,
    )


def pack_opacity_tokens(
    opacity: torch.Tensor,
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    vae_scale_factor_temporal: int,
    vae_scale_factor_spatial: int,
    patch_height: int,
    patch_width: int,
    pad_first_frame: bool,
) -> torch.Tensor:
    """Validate and pack a full-pixel opacity map into per-token channels,
    f_o(PixelUnshuffle(O)), shared by both families' wrappers.

    The ``[B, T_pixel, H_pixel, W_pixel]`` opacity is losslessly rearranged so
    that each generation token sees the complete opacity footprint of the
    pixels it decodes — a ``(sf_s * patch_h) x (sf_s * patch_w)`` spatial
    window times ``vae_scale_factor_temporal`` pixel frames. Returns the grid
    layout ``[B, opacity_embedding_input_dim, latent_t, grid_h, grid_w]`` (CP
    slicing on dim 2 stays valid); callers that need token-major layout
    ``flatten(2).transpose(1, 2)``.

    ``pad_first_frame`` marks a pass whose first latent frame is absolute
    frame 0: the causal VAE's 1+4N contract gives frame 0 a single real pixel
    frame, replicated to fill its window. Chunk-only passes starting mid-video
    carry complete windows (``vae_scale_factor_temporal`` pixel frames per
    latent frame) and need no pad.
    """
    assert opacity.ndim == 4, f"`opacity` must be [B, T, H, W], got shape {tuple(opacity.shape)}"
    expected_pixel_frames = (
        (num_latent_frames - 1) * vae_scale_factor_temporal + 1
        if pad_first_frame
        else num_latent_frames * vae_scale_factor_temporal
    )
    if opacity.shape[1] != expected_pixel_frames:
        # Fail loudly: a silent resize here would hide upstream frame-count
        # bugs instead of surfacing them.
        raise ValueError(
            "`opacity` must be at pixel temporal resolution "
            + (
                "(1 + vae_scale_factor_temporal * (latent_frames - 1) frames); "
                if pad_first_frame
                else "(vae_scale_factor_temporal * latent_frames frames for a chunk-only pass); "
            )
            + f"got {opacity.shape[1]} opacity frames for {num_latent_frames} latent frames "
            f"(expected {expected_pixel_frames})."
        )
    expected_height = latent_height * vae_scale_factor_spatial
    expected_width = latent_width * vae_scale_factor_spatial
    if opacity.shape[-2:] != (expected_height, expected_width):
        raise ValueError(
            "`opacity` must be at full pixel resolution for PixelUnshuffle conditioning; "
            f"got {tuple(opacity.shape[-2:])} for latent grid {(latent_height, latent_width)} "
            f"(expected {(expected_height, expected_width)})."
        )
    if pad_first_frame:
        opacity = pad_opacity_first_frame(opacity, vae_scale_factor_temporal)
    return rearrange(
        opacity,
        "b (t t4) (h h8) (w w8) -> b (h8 w8 t4) t h w",
        t4=vae_scale_factor_temporal,
        h8=vae_scale_factor_spatial * patch_height,
        w8=vae_scale_factor_spatial * patch_width,
    )


def slice_camera_tensor_frames(tensor: torch.Tensor, start_frame: int, end_frame: int) -> torch.Tensor:
    """Slice a batched camera tensor (``[B, T, ...]``) to frames
    ``[start_frame, end_frame)``.

    The single home of the camera-tensor frame-dim convention (frames on
    dim 1), shared by both families' wrapper CP shard slicing and AR rollout
    chunk slicing so the rule cannot drift. Camera tensors are batched
    everywhere in production; unbatched inputs fail loudly instead of being
    reinterpreted.
    """
    assert tensor.ndim >= 4, f"camera tensors are batched [B, T, ...]; got ndim {tensor.ndim}"
    return tensor[:, start_frame:end_frame]


def validate_source_camera_conditioning_lengths(
    *,
    camera_rays: torch.Tensor,
    w2cs: torch.Tensor,
    Ks: torch.Tensor,
    latent_num_frames: int,
) -> None:
    """Reject truncated source controls before intentional inference padding."""
    for name, tensor in (("w2cs", w2cs), ("Ks", Ks), ("camera_rays", camera_rays)):
        actual = tensor.shape[0]
        if actual != latent_num_frames:
            raise ValueError(f"{name} has {actual} source frames, expected {latent_num_frames}")
