# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-neutral prompt-payload policy tests.

The Cosmos3 stored-unconditional branch is pinned by the cosmos3 suites
(test_cosmos3_integration.py, test_cosmos3_prompt_flat.py); these tests pin
the Wan branch — T5-embedding payloads with no stored unconditional drop to
zeros — the shared substitution helper both branches route through, and the
caption filename rule every eval dataset composes prompt paths with.
"""

import pytest
import torch

from model_training.prompt_formats import (
    COSMOS3_PROMPT_FORMAT,
    WAN_T5_PROMPT_FORMAT,
    caption_filename_for_format,
    prompt_payload_for_text_dropout,
    substitute_unconditional_prompt,
)


def test_wan_caption_filename_is_byte_equal_to_the_historical_bare_name():
    # On-disk contract: the Wan caption trees on the clusters were written
    # under exactly these names, so the Wan branch must never rewrite them.
    for bare_name in ("caption.hdf5", "caption_3.hdf5", "garden.h5", "frames_all_stride_1.h5"):
        assert caption_filename_for_format(bare_name, WAN_T5_PROMPT_FORMAT) == bare_name


def test_cosmos3_caption_filename_inserts_the_format_segment_before_the_extension():
    assert caption_filename_for_format("caption.hdf5", COSMOS3_PROMPT_FORMAT) == "caption.cosmos3.hdf5"
    assert caption_filename_for_format("caption_3.hdf5", COSMOS3_PROMPT_FORMAT) == "caption_3.cosmos3.hdf5"
    assert caption_filename_for_format("garden.h5", COSMOS3_PROMPT_FORMAT) == "garden.cosmos3.h5"
    assert (
        caption_filename_for_format("frames_all_stride_1.h5", COSMOS3_PROMPT_FORMAT) == "frames_all_stride_1.cosmos3.h5"
    )


def test_caption_filename_rejects_unknown_prompt_formats():
    with pytest.raises(ValueError, match="Unknown prompt format"):
        caption_filename_for_format("caption.hdf5", "not_a_format")


def test_wan_text_dropout_drops_to_zeros_without_stored_unconditional():
    t5_embedding = torch.randn(1, 3, 8, dtype=torch.float32)
    data = {"encoded_prompt": t5_embedding}

    dropped = prompt_payload_for_text_dropout(data, drop_text=True)

    # 10% of Wan training steps train the unconditional: the payload must
    # come back all-zeros (the Wan no-text convention), same shape/dtype,
    # with no mask or negative keys invented along the way.
    assert set(dropped) == {"encoded_prompt"}
    assert dropped["encoded_prompt"].shape == t5_embedding.shape
    assert dropped["encoded_prompt"].dtype == t5_embedding.dtype
    assert torch.equal(dropped["encoded_prompt"], torch.zeros_like(t5_embedding))

    kept = prompt_payload_for_text_dropout(data, drop_text=False)
    assert set(kept) == {"encoded_prompt"}
    assert torch.equal(kept["encoded_prompt"], t5_embedding)


def test_substitute_unconditional_prompt_prefers_the_stored_unconditional():
    prompt = torch.tensor([[11, 12, 7, 8, 0]])
    prompt_mask = torch.tensor([[True, True, True, True, False]])
    stored = torch.tensor([[21, 7, 8, 0, 0]])
    stored_mask = torch.tensor([[True, True, True, False, False]])

    out_prompt, out_mask = substitute_unconditional_prompt(prompt, prompt_mask, stored, stored_mask)

    assert out_prompt is stored
    assert out_mask is stored_mask


def test_substitute_unconditional_prompt_zeroes_when_no_stored_unconditional():
    prompt = torch.randn(1, 3, 8)

    out_prompt, out_mask = substitute_unconditional_prompt(prompt, None, None, None)

    assert torch.equal(out_prompt, torch.zeros_like(prompt))
    assert out_mask is None
