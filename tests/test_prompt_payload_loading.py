# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-payload contract has a single home.

The dataset->trainer prompt schema is load-bearing: the train path
(dl3dv_base.load_prompt_payload) and the eval path
(utils.load_prompt_payload_from_path) must decode identically, so both
delegate to data/utils.py load_prompt_payload. These tests pin the delegation
and the train/eval parity on real HDF5 files.
"""

import h5py
import numpy as np
import pytest
import torch

from model_training.constants import MAX_SEQUENCE_LENGTH
from model_training.data.dl3dv_base import load_prompt_payload as train_load_prompt_payload
from model_training.data.utils import decode_and_pad_prompt, load_prompt_payload_from_path
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT

COSMOS3_PAYLOAD_KEYS = {
    "encoded_prompt",
    "prompt_attention_mask",
    "encoded_negative_prompt",
    "negative_prompt_attention_mask",
}


def _write_cosmos3_prompt_file(path) -> None:
    with h5py.File(path, "w") as f:
        dataset = f.create_dataset("frames_81_start_0", data=np.asarray([11, 12, 7, 8], dtype=np.int32))
        dataset.attrs["caption"] = "a parity caption"
        dataset.attrs["image_indices"] = np.arange(4, dtype=np.int64)
        dataset.attrs["fps"] = 24.0
        f.attrs["negative_input_ids"] = np.asarray([21, 7, 8], dtype=np.int32)
        f.attrs["pad_token_id"] = 0


def _write_wan_prompt_file(path, *, tokens: int = 3, dim: int = 8) -> None:
    with h5py.File(path, "w") as f:
        encoded = torch.randn(tokens, dim, dtype=torch.bfloat16).view(torch.uint16).numpy()
        dataset = f.create_dataset("frames_81_start_0", data=encoded)
        dataset.attrs["caption"] = "a wan caption"


def test_cosmos3_train_and_eval_paths_decode_identical_payloads(tmp_path):
    prompt_path = tmp_path / "prompt.h5"
    _write_cosmos3_prompt_file(prompt_path)

    eval_payload, caption = load_prompt_payload_from_path(prompt_path, prompt_format=COSMOS3_PROMPT_FORMAT)
    with h5py.File(prompt_path, "r") as f:
        train_payload = train_load_prompt_payload(f, f["frames_81_start_0"], prompt_format=COSMOS3_PROMPT_FORMAT)

    assert caption == "a parity caption"
    assert set(eval_payload) == set(train_payload) == COSMOS3_PAYLOAD_KEYS
    for key in COSMOS3_PAYLOAD_KEYS:
        assert torch.equal(eval_payload[key], train_payload[key]), key


def test_wan_train_and_eval_paths_decode_identical_payloads(tmp_path):
    prompt_path = tmp_path / "prompt.h5"
    _write_wan_prompt_file(prompt_path)

    eval_payload, caption = load_prompt_payload_from_path(prompt_path, prompt_format=WAN_T5_PROMPT_FORMAT)
    with h5py.File(prompt_path, "r") as f:
        train_payload = train_load_prompt_payload(f, f["frames_81_start_0"], prompt_format=WAN_T5_PROMPT_FORMAT)

    assert caption == "a wan caption"
    assert set(eval_payload) == set(train_payload) == {"encoded_prompt"}
    assert eval_payload["encoded_prompt"].shape == (MAX_SEQUENCE_LENGTH, 8)
    assert eval_payload["encoded_prompt"].dtype == torch.bfloat16
    assert torch.equal(eval_payload["encoded_prompt"], train_payload["encoded_prompt"])


def test_cosmos3_loader_fails_on_missing_negative_prompt(tmp_path):
    prompt_path = tmp_path / "prompt.h5"
    _write_cosmos3_prompt_file(prompt_path)
    with h5py.File(prompt_path, "a") as f:
        del f.attrs["negative_input_ids"]

    with pytest.raises(KeyError):
        load_prompt_payload_from_path(prompt_path, prompt_format=COSMOS3_PROMPT_FORMAT)


def test_decode_and_pad_prompt_rejects_over_length_prompts(tmp_path):
    """The Wan length assert lives in decode_and_pad_prompt itself,
    guarding the train and eval paths alike."""
    prompt_path = tmp_path / "prompt.h5"
    _write_wan_prompt_file(prompt_path, tokens=MAX_SEQUENCE_LENGTH + 1)

    with h5py.File(prompt_path, "r") as f:
        with pytest.raises(AssertionError, match="exceeds MAX_SEQUENCE_LENGTH"):
            decode_and_pad_prompt(f["frames_81_start_0"])
