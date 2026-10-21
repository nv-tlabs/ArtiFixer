# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint clean/noisy reference calculation for cached-rollout tests only.

The production transformer has no teacher-forcing API. This oracle packs a
full sequence and invokes its base model with the original clean/noisy mask;
it does not implement the KV-cache path it is used to check.
"""

import torch
from torch.nn.attention.flex_attention import BlockMask

from model_training.common.context_parallel import (
    _block_padded_length,
    _flex_block_sizes,
    _sparse_block_mask_from_intervals,
)
from model_training.net import attention_config


def teacher_forcing_block_mask(
    *,
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    frames_per_block: int,
    prefix_len: int = 0,
    block_size: int | tuple[int, int],
    attention_heads: int | None = None,
) -> BlockMask:
    """Build the joint clean/noisy reference mask without context parallelism.

    Only the test-side full-sequence oracle uses this layout for cached-rollout
    parity checks; production models do not accept clean/noisy pairs.
    The sequence is clean frames followed by noisy frames. Clean queries can
    attend through their current clean block; noisy queries attend to previous
    clean blocks plus their own noisy block. The first ``prefix_len`` KV
    columns are text keys that every generation query may attend.
    """
    assert num_frames > 0, f"num_frames must be positive, got {num_frames}"
    assert frame_seqlen > 0, f"frame_seqlen must be positive, got {frame_seqlen}"
    assert frames_per_block > 0, f"frames_per_block must be positive, got {frames_per_block}"
    assert prefix_len >= 0, f"prefix_len must be non-negative, got {prefix_len}"
    assert (
        num_frames % frames_per_block == 0
    ), f"num_frames {num_frames} must be divisible by frames_per_block {frames_per_block}"
    sequence_length = num_frames * frame_seqlen
    total_length = 2 * sequence_length
    kv_length = prefix_len + total_length
    block_token_count = frame_seqlen * frames_per_block
    # No Python scalars in attention_mask (see frame_block_causal_mask);
    # every value is a q_idx-indexed tensor sized to the padded query
    # length, matching the ends[q_idx] capture in block_mask.py.
    q_block_size, _ = _flex_block_sizes(block_size)
    padded_q = _block_padded_length(total_length, q_block_size)
    q = torch.arange(padded_q, dtype=torch.long, device=device)
    prefix_vec = torch.full((padded_q,), prefix_len, dtype=torch.long, device=device)
    clean_q_vec = q < sequence_length
    clean_limit_vec = (q // block_token_count + 1) * block_token_count
    noisy_rel = q - sequence_length
    noisy_block = noisy_rel // block_token_count
    noisy_start_vec = sequence_length + noisy_block * block_token_count
    noisy_end_vec = noisy_start_vec + block_token_count
    noisy_prev_limit_vec = noisy_block * block_token_count

    def attention_mask(b, h, q_idx, kv_idx):
        prefix = prefix_vec[q_idx]
        gen_kv_idx = kv_idx - prefix
        clean_mask = clean_q_vec[q_idx] & (gen_kv_idx < clean_limit_vec[q_idx])
        noisy_same_block = (gen_kv_idx >= noisy_start_vec[q_idx]) & (gen_kv_idx < noisy_end_vec[q_idx])
        previous_clean_blocks = gen_kv_idx < noisy_prev_limit_vec[q_idx]
        noisy_mask = (~clean_q_vec[q_idx]) & (noisy_same_block | previous_clean_blocks)
        return (kv_idx < prefix) | (q_idx == gen_kv_idx) | (clean_mask | noisy_mask)

    def query_segments(q_start: int, q_end: int) -> list[tuple[bool, int]]:
        segments: list[tuple[bool, int]] = []
        current = q_start
        while current < q_end:
            is_clean = current < sequence_length
            half_base = 0 if is_clean else sequence_length
            half_limit = sequence_length if is_clean else total_length
            block = (current - half_base) // block_token_count
            next_boundary = min(q_end, half_base + (block + 1) * block_token_count, half_limit)
            segments.append((is_clean, block))
            current = next_boundary
        return segments

    def allowed_intervals(segment: tuple[bool, int]) -> list[tuple[int, int]]:
        is_clean, block = segment
        if is_clean:
            gen_intervals = [(0, (block + 1) * block_token_count)]
        else:
            gen_intervals = [
                (0, block * block_token_count),
                (sequence_length + block * block_token_count, sequence_length + (block + 1) * block_token_count),
            ]
        return [(0, prefix_len)] + [(prefix_len + start, prefix_len + end) for start, end in gen_intervals]

    return _sparse_block_mask_from_intervals(
        device=device,
        total_length=total_length,
        kv_length=kv_length,
        block_size=block_size,
        query_segments=query_segments,
        allowed_intervals=allowed_intervals,
        mask_mod=attention_mask,
        attention_heads=attention_heads,
    )


def cosmos3_teacher_forcing_reference(
    transformer,
    *,
    hidden_states,
    clean_x,
    timestep,
    encoder_hidden_states,
    neighbor_hidden_states,
    opacity,
    camera_rays,
    encoder_attention_mask=None,
    neighbor_camera_rays=None,
    ignore_neighbors=False,
    w2cs=None,
    Ks=None,
    neighbor_w2cs=None,
    neighbor_Ks=None,
):
    """Return noisy-half predictions from an independent full-sequence weave.

    Reuse the trained modules and their conditioning primitives, but construct
    the clean/noisy positions, mask and loss rows here. No cache is read or
    written. Gradients remain live for output/parameter-gradient comparisons.
    """
    assert transformer._cp_mesh is None, "the reference calculation is single-rank"
    assert transformer.causal_block_frames is not None
    assert clean_x.shape == hidden_states.shape
    batch_size, _, num_frames, _, _ = hidden_states.shape
    device = hidden_states.device
    use_references = (
        neighbor_hidden_states is not None
        and neighbor_hidden_states.shape[2] > 0
        and not transformer._as_bool(ignore_neighbors)
    )
    if use_references:
        neighbor_hidden_states = neighbor_hidden_states.to(hidden_states)
        assert neighbor_camera_rays is not None
    opacity_tokens, camera_tokens = transformer._target_condition_tokens(
        opacity, camera_rays, hidden_states, 0, num_frames, start_frame=0
    )
    opacity_tokens = torch.cat([opacity_tokens, opacity_tokens], dim=1)
    camera_tokens = torch.cat([camera_tokens, camera_tokens], dim=1)
    outputs = []
    for sample_index in range(batch_size):
        input_ids = transformer._trim_input_ids(encoder_hidden_states, encoder_attention_mask, sample_index)
        text = transformer._prepare_text_segment(input_ids, device=device)
        noisy = hidden_states[sample_index : sample_index + 1]
        vision = transformer._prepare_vision_segment(
            input_vision_tokens=noisy,
            mrope_offset=text["vision_start_temporal_offset"],
            curr=text["und_len"],
            device=device,
        )
        tokens_per_half = vision["num_vision_tokens"]
        _, patch_h, patch_w = vision["vision_token_shapes"][0]
        vision.update(
            vision_token_shapes=[(2 * num_frames, patch_h, patch_w)],
            vision_sequence_indexes=torch.arange(
                text["und_len"], text["und_len"] + 2 * tokens_per_half, dtype=torch.long, device=device
            ),
            vision_mse_loss_indexes=vision["vision_mse_loss_indexes"] + tokens_per_half,
            vision_noisy_frame_indexes=[torch.arange(num_frames, 2 * num_frames, device=device, dtype=torch.long)],
            vision_mrope_ids=torch.cat([vision["vision_mrope_ids"], vision["vision_mrope_ids"]], dim=1),
            num_vision_tokens=2 * tokens_per_half,
        )
        sample_latents = torch.cat([clean_x[sample_index : sample_index + 1], noisy], dim=2)
        if use_references:
            sample_latents, vision = transformer._prepend_references(
                sample_latents, neighbor_hidden_states[sample_index : sample_index + 1], text, vision
            )
        block_mask = teacher_forcing_block_mask(
            device=device,
            num_frames=num_frames,
            frame_seqlen=patch_h * patch_w,
            frames_per_block=transformer.causal_block_frames,
            prefix_len=text["und_len"] + vision.get("num_reference_tokens", 0),
            block_size=attention_config.flex_block_size(),
        )
        transformer._conditioning_context = transformer._layer_conditioning_context(
            sample_index,
            opacity_tokens,
            camera_tokens,
            neighbor_hidden_states if use_references else None,
            vision,
            gen_block_mask=block_mask,
            neighbor_camera_rays=neighbor_camera_rays,
        )
        try:
            predictions, _, _ = transformer.base_transformer(
                input_ids=text["input_ids"],
                text_indexes=text["text_indexes"],
                position_ids=torch.cat([text["text_mrope_ids"], vision["vision_mrope_ids"]], dim=1),
                und_len=text["und_len"],
                sequence_length=text["und_len"] + vision["num_vision_tokens"],
                vision_tokens=[sample_latents],
                vision_token_shapes=vision["vision_token_shapes"],
                vision_sequence_indexes=vision["vision_sequence_indexes"],
                vision_mse_loss_indexes=vision["vision_mse_loss_indexes"],
                vision_timesteps=transformer._vision_timesteps(
                    timestep, sample_index, batch_size, tokens_per_half, device
                ),
                vision_noisy_frame_indexes=vision["vision_noisy_frame_indexes"],
            )
        finally:
            transformer._conditioning_context = None
            for layer in transformer.base_transformer.layers:
                transformer._processor_disarm_hook(layer.self_attn, (), None)
        outputs.append(predictions[0][:, :, vision.get("num_reference_frames", 0) + num_frames :])
    return (torch.cat(outputs, dim=0),)
