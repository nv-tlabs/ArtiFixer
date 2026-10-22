# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity pins for the forked vae-token mRoPE id builder.

model_training/net/cosmos3_transformer.py forks ``get_3d_mrope_ids_vae_tokens``
from diffusers pipeline_cosmos3_omni.py @ v0.39.0 with exactly one semantic
delta: fps modulation applies to every segment shape, grid_t == 1 included
(upstream skips it for singleton segments — a shape its action path can also
produce, which this repo only reaches when a context-parallel rank holds a
single latent frame that must keep its fps-scaled global temporal position).
These tests import the
upstream builder and assert byte-equality everywhere the semantics are shared
(every layout with grid_t > 1; grid_t == 1 without fps), and pin the
singleton-fps delta explicitly so an unnoticed upstream change or an ArtiFixer
regression fails the suite instead of shifting rope positions silently.
Signature mapping: ArtiFixer's keyword-only
``fps_reference_temporal_compression_factor`` is upstream's optional
``base_temporal_compression_factor``. (The text-token builder is imported
from diffusers directly — no fork, nothing to pin.)
"""

import pytest
import torch
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import get_3d_mrope_ids_vae_tokens as upstream_vae_mrope_ids

from model_training.net.cosmos3_transformer import get_3d_mrope_ids_vae_tokens

_VAE_PARITY_CASES = {
    # The production vision layout: 21 latent frames past the text+margin
    # offset, fps-modulated (the geometry that exposed the rope collapse).
    "vision_fps_modulated": dict(
        grid_t=21,
        grid_h=4,
        grid_w=5,
        temporal_offset=15013.0,
        reset_spatial_indices=True,
        fps=15.0,
        base_fps=24.0,
        temporal_compression_factor=4,
        start_frame_offset=0,
    ),
    # fps=None integer path (enable_fps_modulation=False configs).
    "vision_integer_positions": dict(
        grid_t=3,
        grid_h=2,
        grid_w=2,
        temporal_offset=100,
        reset_spatial_indices=True,
        fps=None,
        base_fps=24.0,
        temporal_compression_factor=4,
        start_frame_offset=0,
    ),
    # Spatial ids offset by the temporal offset when not reset.
    "vision_unreset_spatial_ids": dict(
        grid_t=2,
        grid_h=3,
        grid_w=2,
        temporal_offset=40,
        reset_spatial_indices=False,
        fps=None,
        base_fps=24.0,
        temporal_compression_factor=4,
        start_frame_offset=0,
    ),
    # The action-suffix layout: per-frame tokens (compression factor 1) scaled
    # against the vision-latent reference rate, starting at source frame 1.
    "action_suffix": dict(
        grid_t=8,
        grid_h=1,
        grid_w=1,
        temporal_offset=15013.0,
        reset_spatial_indices=True,
        fps=60.0,
        base_fps=24.0,
        temporal_compression_factor=1,
        fps_reference_temporal_compression_factor=4,
        start_frame_offset=1,
    ),
    # A context-parallel shard with more than one frame: fps-scaled
    # start_frame_offset agrees with upstream whenever grid_t > 1.
    "cp_shard_with_start_frame_offset": dict(
        grid_t=2,
        grid_h=2,
        grid_w=2,
        temporal_offset=15013.0,
        reset_spatial_indices=True,
        fps=30.0,
        base_fps=24.0,
        temporal_compression_factor=4,
        start_frame_offset=3,
    ),
}


def _upstream_kwargs(kwargs: dict) -> dict:
    """Map ArtiFixer's keyword-only signature onto upstream's parameter names."""
    mapped = dict(kwargs)
    if "fps_reference_temporal_compression_factor" in mapped:
        mapped["base_temporal_compression_factor"] = mapped.pop("fps_reference_temporal_compression_factor")
    return mapped


@pytest.mark.parametrize("case", _VAE_PARITY_CASES.values(), ids=_VAE_PARITY_CASES.keys())
def test_vae_mrope_ids_match_upstream_for_multi_frame_grids(case):
    ids, next_offset = get_3d_mrope_ids_vae_tokens(**case)
    expected_ids, expected_next_offset = upstream_vae_mrope_ids(**_upstream_kwargs(case))

    assert ids.dtype == expected_ids.dtype
    assert torch.equal(ids, expected_ids)
    assert next_offset == expected_next_offset


def test_singleton_grid_without_fps_matches_upstream():
    """grid_t == 1 on the integer path is shared semantics: parity holds."""
    case = dict(
        grid_t=1,
        grid_h=2,
        grid_w=3,
        temporal_offset=15013,
        reset_spatial_indices=True,
        fps=None,
        base_fps=24.0,
        temporal_compression_factor=4,
        start_frame_offset=5,
    )

    ids, next_offset = get_3d_mrope_ids_vae_tokens(**case)
    expected_ids, expected_next_offset = upstream_vae_mrope_ids(**_upstream_kwargs(case))

    assert ids.dtype == expected_ids.dtype
    assert torch.equal(ids, expected_ids)
    assert next_offset == expected_next_offset


def test_singleton_grid_with_fps_pins_the_deliberate_artifixer_delta():
    """The ONE audited semantic delta vs upstream: ArtiFixer keeps fps modulation
    for grid_t == 1, so a single-latent-frame CP shard at start_frame_offset
    lands on its fps-scaled global position ((0+5)/(60/4)*(24/4) + 7 = 9.0);
    upstream disables modulation for singleton grids and would place it at the
    unscaled integer position 7 + 5 = 12, breaking cross-rank temporal
    alignment. If upstream ever changes this gate, this test flags the fork
    comment + delta for re-evaluation."""
    case = dict(
        grid_t=1,
        grid_h=1,
        grid_w=1,
        temporal_offset=7.0,
        reset_spatial_indices=True,
        fps=60.0,
        base_fps=24.0,
        temporal_compression_factor=4,
        start_frame_offset=5,
    )

    ids, _ = get_3d_mrope_ids_vae_tokens(**case)
    upstream_ids, _ = upstream_vae_mrope_ids(**_upstream_kwargs(case))

    assert ids.dtype == torch.float32
    torch.testing.assert_close(ids[0, 0], torch.tensor(9.0))
    assert upstream_ids[0, 0].item() == 12
    assert not torch.equal(ids.to(torch.float64), upstream_ids.to(torch.float64))


def test_cp_shard_mrope_positions_equal_the_full_clip_slice():
    """CP slice-equality pin: a context-parallel shard reuses the
    ``start_frame_offset`` seam, so its vision mRoPE ids must equal the
    full-clip builder's rows for the same absolute positions, bitwise (fps
    modulation active). The upstream-parity cases above pin signature-mapped
    parity; this is the only direct slice-equality pin of the property the
    CP sharding relies on for cross-rank temporal alignment."""
    start, end = 4, 6
    common = dict(
        grid_h=2,
        grid_w=2,
        temporal_offset=15004.0,
        reset_spatial_indices=True,
        fps=60.0,
        base_fps=24.0,
        temporal_compression_factor=4,
    )
    full_ids, _ = get_3d_mrope_ids_vae_tokens(grid_t=end, start_frame_offset=0, **common)
    shard_ids, _ = get_3d_mrope_ids_vae_tokens(grid_t=end - start, start_frame_offset=start, **common)
    spatial = common["grid_h"] * common["grid_w"]
    assert torch.equal(shard_ids, full_ids[:, start * spatial : end * spatial])
