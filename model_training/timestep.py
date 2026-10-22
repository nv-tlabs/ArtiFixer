# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def spatial_alignment(vae_scale_factor_spatial: int, patch_size: tuple[int, int, int]) -> tuple[int, int]:
    _, patch_h, patch_w = patch_size
    return vae_scale_factor_spatial * patch_h, vae_scale_factor_spatial * patch_w


def expand_transformer_timestep(
    timestep: torch.Tensor,
    hidden_states: torch.Tensor,
    patch_size: tuple[int, int, int],
    frames_per_block: int | None = None,
) -> torch.Tensor:
    """Expand scalar/block timesteps to per-token post-patch timesteps."""
    batch_size, _, latent_frames, latent_height, latent_width = hidden_states.shape
    patch_t, patch_h, patch_w = patch_size

    if latent_frames % patch_t != 0 or latent_height % patch_h != 0 or latent_width % patch_w != 0:
        raise ValueError(
            "Latent shape must be divisible by transformer patch size "
            f"{patch_size}, got {(latent_frames, latent_height, latent_width)}"
        )

    post_patch_frames = latent_frames // patch_t
    frame_seq_len = (latent_height // patch_h) * (latent_width // patch_w)
    seq_len = post_patch_frames * frame_seq_len

    if timestep.ndim == 0:
        return timestep.expand(batch_size, seq_len)

    if timestep.ndim == 2:
        if timestep.shape != (batch_size, seq_len):
            raise ValueError(f"Expanded timestep shape must be {(batch_size, seq_len)}, got {tuple(timestep.shape)}")
        return timestep

    if timestep.ndim != 1:
        raise ValueError(f"Expected scalar, 1D, or expanded 2D timestep, got shape {tuple(timestep.shape)}")

    if timestep.numel() == batch_size:
        return timestep.unsqueeze(1).expand(-1, seq_len)

    if frames_per_block is not None:
        if frames_per_block % patch_t != 0:
            raise ValueError(f"frames_per_block={frames_per_block} must be divisible by temporal patch size {patch_t}")
        if latent_frames % frames_per_block != 0:
            raise ValueError(f"latent_frames={latent_frames} must be divisible by frames_per_block={frames_per_block}")
        num_blocks = latent_frames // frames_per_block
        if timestep.numel() == batch_size * num_blocks:
            tokens_per_block = (frames_per_block // patch_t) * frame_seq_len
            return timestep.view(batch_size, num_blocks).repeat_interleave(tokens_per_block, dim=1)

    raise ValueError(
        "Cannot expand timestep with shape "
        f"{tuple(timestep.shape)} for batch={batch_size}, seq_len={seq_len}, frames_per_block={frames_per_block}"
    )
