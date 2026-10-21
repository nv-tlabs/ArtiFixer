# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real tiny dense/MoE checkpoints; check exact parameter loading."""

import pytest
import torch
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
)

from data_processing.captioning import generate_captions


@pytest.mark.parametrize("moe", [False, True])
def test_captioner_loads_the_checkpoint_architecture(tmp_path, monkeypatch, moe):
    text = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    vision = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=1,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    cls = Qwen3VLMoeForConditionalGeneration if moe else Qwen3VLForConditionalGeneration
    config_cls = Qwen3VLMoeConfig if moe else Qwen3VLConfig
    if moe:
        text.update(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=32)
    config = config_cls(
        text_config=text,
        vision_config=vision,
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=30,
        vision_end_token_id=31,
    )
    torch.manual_seed(18)
    expected = cls(config)
    expected.save_pretrained(tmp_path)
    monkeypatch.setattr(generate_captions.Qwen3VLProcessor, "from_pretrained", lambda _: object())
    actual, _ = generate_captions.get_caption_model_and_processor(str(tmp_path))
    assert type(actual) is cls
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=0, atol=0)
