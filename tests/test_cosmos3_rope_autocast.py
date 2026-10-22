# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression pin for the diffusers v0.39.0 fp32 rope autocast guard.

The pre-v0.39.0 fork snapshot (784fa626) ran the Cosmos3 rope position-id matmul
under the ambient autocast. accelerate wraps the prepared model forward in bf16
autocast, and bf16 cannot represent consecutive positions past 256; the vision
temporal positions sit at offset ~15000 (text length + the 15000 modality
margin) with spacing 1.6 (base_fps 24 / conditioning fps 15 at temporal
compression 4), so all 21 latent frames collapsed onto 2 distinct rope rows in
production. diffusers v0.39.0 (a3608b512ed7248499a44c61d954965ed9bdae4d) pins
that matmul to fp32 inside Cosmos3VLTextRotaryEmbedding. These tests pin the
upstream fix against the real module with the real Cosmos3-Nano rope geometry
(head_dim 128, rope_theta 5e6, rope_axes_dim [24, 20, 20]).
"""

import torch
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

from model_training.net.cosmos3_transformer import get_3d_mrope_ids_vae_tokens

LATENT_FRAMES = 21  # 81 pixel frames under 4x temporal compression
TEMPORAL_OFFSET = 15000
TEMPORAL_SPACING = 1.6


def _real_rope() -> Cosmos3VLTextRotaryEmbedding:
    # Cosmos3-Nano rope geometry: transformer_cosmos3.py defaults / the published config.
    return Cosmos3VLTextRotaryEmbedding(head_dim=128, rope_theta=5000000.0, rope_axes_dim=(24, 20, 20))


def _collapse_prone_position_ids() -> torch.Tensor:
    """[3, 1, 21] mRoPE ids built by the real ArtiFixer vision builder: offset 15000, spacing 1.6."""
    mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=LATENT_FRAMES,
        grid_h=1,
        grid_w=1,
        temporal_offset=TEMPORAL_OFFSET,
        reset_spatial_indices=True,
        fps=15.0,
        base_fps=24.0,
        temporal_compression_factor=4,
        start_frame_offset=0,
    )
    expected = TEMPORAL_OFFSET + TEMPORAL_SPACING * torch.arange(LATENT_FRAMES, dtype=torch.float32)
    assert torch.equal(mrope_ids[0], expected), "test premise: temporal positions are 15000 + 1.6*k"
    return mrope_ids.unsqueeze(1)  # [3, N] -> [3, 1, N], as the base transformer passes them


def test_rope_temporal_rows_all_distinct_under_bf16_autocast():
    """v0.39.0 rope keeps every temporal position distinct under ambient bf16 autocast."""
    rope = _real_rope()
    position_ids = _collapse_prone_position_ids()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cos, sin = rope(position_ids=position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)

    for name, emb in (("cos", cos), ("sin", sin)):
        rows = emb[0]  # [N, head_dim]
        distinct = int(torch.unique(rows, dim=0).shape[0])
        assert distinct == LATENT_FRAMES, (
            f"temporal rope collapsed under bf16 autocast: {distinct} distinct {name} rows for "
            f"{LATENT_FRAMES} temporal positions — the fp32 rope guard regressed"
        )


def test_rope_output_unaffected_by_ambient_autocast():
    """The v0.39.0 guard makes ambient autocast inert: bitwise-equal output with and without it."""
    rope = _real_rope()
    position_ids = _collapse_prone_position_ids()

    cos_plain, sin_plain = rope(position_ids=position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cos_ac, sin_ac = rope(position_ids=position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)

    assert torch.equal(cos_ac, cos_plain)
    assert torch.equal(sin_ac, sin_plain)


def test_unguarded_bf16_position_matmul_would_collapse():
    """Premise check: without the guard these exact positions DO collapse (2 bf16 values span 21 frames).

    This emulates the pre-v0.39.0 fork path (autocast casting both matmul operands to bf16) and keeps
    the tests above honest: if the offset/spacing constants ever drift to a range bf16 can represent,
    this test fails, flagging that the regression pin no longer exercises the failure mode.
    """
    rope = _real_rope()
    position_ids = _collapse_prone_position_ids()

    inv_freq_expanded = rope.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded.to(torch.bfloat16) @ position_ids_expanded.to(torch.bfloat16)).transpose(2, 3)
    freqs = rope.apply_interleaved_mrope(freqs.float(), rope.rope_axes_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_rows = emb.cos().to(torch.bfloat16)[0]

    distinct = int(torch.unique(cos_rows, dim=0).shape[0])
    assert distinct == 2, f"expected the production collapse (21 frames -> 2 rope rows), got {distinct}"
    assert int(torch.unique(position_ids[0, 0].to(torch.bfloat16)).numel()) == 2
