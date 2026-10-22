# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the family-neutral conditioning and context-parallel
helpers shared by the Wan and Cosmos3 wrapper stacks."""

import pytest
import torch
from einops import rearrange

from model_training.net.conditioning import (
    camera_embedding_input_dim,
    opacity_embedding_input_dim,
    pack_opacity_tokens,
    pad_opacity_first_frame,
    slice_camera_tensor_frames,
)
from model_training.net.context_parallel import cp_frame_slice


def _pack(opacity, *, latent_frames, sf_t=4, sf_s=8, patch=2, pad_first_frame=True, latent_hw=(4, 4)):
    return pack_opacity_tokens(
        opacity,
        num_latent_frames=latent_frames,
        latent_height=latent_hw[0],
        latent_width=latent_hw[1],
        vae_scale_factor_temporal=sf_t,
        vae_scale_factor_spatial=sf_s,
        patch_height=patch,
        patch_width=patch,
        pad_first_frame=pad_first_frame,
    )


def test_pack_opacity_tokens_matches_the_previous_wan_packing():
    """At the production temporal factor 4, the shared helper is bit-identical
    to the Wan wrapper's former inline pad-3 + rearrange."""
    opacity = torch.rand(1, 13, 32, 32)  # 4 latent frames, 4x4 latent grid, sf_s=8

    grid = _pack(opacity, latent_frames=4)

    padded = torch.cat([opacity[:, :1].repeat_interleave(3, dim=1), opacity], dim=1)
    legacy = rearrange(padded, "b (t t4) (h h8) (w w8) -> b (h8 w8 t4) t h w", t4=4, h8=16, w8=16)
    torch.testing.assert_close(grid, legacy)
    assert grid.shape == (
        1,
        opacity_embedding_input_dim(
            vae_scale_factor_temporal=4, vae_scale_factor_spatial=8, patch_height=2, patch_width=2
        ),
        4,
        2,
        2,
    )


def test_pack_opacity_tokens_parametrizes_the_temporal_pad():
    """The first-frame pad follows vae_scale_factor_temporal - 1 (the Wan copy
    used to hardcode 3); a factor-2 VAE packs without error and each token
    carries 2 pixel frames of opacity."""
    opacity = torch.rand(1, 7, 32, 32)  # (4 - 1) * 2 + 1 pixel frames

    grid = _pack(opacity, latent_frames=4, sf_t=2)

    assert grid.shape[1] == 16 * 16 * 2
    # Frame 0's window is its single real pixel frame, replicated.
    torch.testing.assert_close(
        pad_opacity_first_frame(opacity, 2)[:, 0],
        opacity[:, 0],
    )


def test_pack_opacity_tokens_chunk_only_pass_needs_full_windows_and_no_pad():
    opacity = torch.rand(1, 8, 32, 32)  # 2 latent frames x 4 pixel frames

    grid = _pack(opacity, latent_frames=2, pad_first_frame=False)

    assert grid.shape[2] == 2


@pytest.mark.parametrize("pad_first_frame,frames", [(True, 8), (True, 2), (False, 9), (False, 2)])
def test_pack_opacity_tokens_rejects_unexpected_temporal_length(pad_first_frame, frames):
    opacity = torch.ones(1, frames, 32, 32)

    with pytest.raises(ValueError, match="pixel temporal resolution"):
        _pack(opacity, latent_frames=2, pad_first_frame=pad_first_frame)


def test_pack_opacity_tokens_rejects_wrong_spatial_resolution():
    opacity = torch.ones(1, 5, 4, 4)  # latent-resolution spatial dims

    with pytest.raises(ValueError, match="full pixel resolution"):
        _pack(opacity, latent_frames=2)


def test_embedding_input_dims_match_the_production_wrappers():
    # Wan reads these factors from the loaded VAE and transformer patch size;
    # the production Wan checkpoint config used in this test gives sf_t=4,
    # sf_s=8, patch (2, 2) -> 16x16 pixel window per token.
    assert (
        opacity_embedding_input_dim(
            vae_scale_factor_temporal=4, vae_scale_factor_spatial=8, patch_height=2, patch_width=2
        )
        == 4 * 16 * 16
    )
    assert camera_embedding_input_dim(vae_scale_factor_spatial=8, patch_height=2, patch_width=2) == 6 * 16 * 16
    # Cosmos3: sf_s * latent_patch_size = 32 -> 1 channel x 32^2 x 4 = 4096.
    assert (
        opacity_embedding_input_dim(
            vae_scale_factor_temporal=4, vae_scale_factor_spatial=16, patch_height=2, patch_width=2
        )
        == 4096
    )


def test_slice_camera_tensor_frames_slices_dim_1_and_rejects_unbatched():
    w2cs = torch.arange(2 * 5 * 4 * 4, dtype=torch.float32).reshape(2, 5, 4, 4)

    torch.testing.assert_close(slice_camera_tensor_frames(w2cs, 1, 3), w2cs[:, 1:3])

    with pytest.raises(AssertionError, match="batched"):
        slice_camera_tensor_frames(w2cs[0], 1, 3)


def test_cp_frame_slice_partitions_frames_contiguously():
    assert cp_frame_slice(8, 0, 2) == (0, 4)
    assert cp_frame_slice(8, 1, 2) == (4, 8)
    assert cp_frame_slice(6, 2, 3) == (4, 6)


def test_cp_frame_slice_rejects_indivisible_frame_counts():
    with pytest.raises(ValueError, match="Cannot evenly split"):
        cp_frame_slice(7, 0, 2)
