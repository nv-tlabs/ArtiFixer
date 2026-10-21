# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached prompt representations must satisfy the downstream loader contract."""

import h5py
import numpy as np
import pytest

from model_training.cosmos3_text import COSMOS3_DEFAULT_MAX_TEXT_TOKENS, pad_cosmos3_input_ids
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, validate_prompt_file_format


def _write(path, conditional, negative, pad=0):
    with h5py.File(path, "w") as hf:
        hf.attrs["negative_input_ids"] = negative
        hf.attrs["pad_token_id"] = pad
        hf.create_dataset("frame", data=conditional).attrs["fps"] = 12.5


@pytest.mark.parametrize("field", ["conditional", "negative"])
@pytest.mark.parametrize(
    "bad",
    [
        np.array([[1, 2]]),
        np.array([1.0, 2.0]),
        np.array([], dtype=np.int32),
        np.zeros(COSMOS3_DEFAULT_MAX_TEXT_TOKENS + 1, dtype=np.int32),
    ],
)
def test_rejects_malformed_token_streams(tmp_path, field, bad):
    path = tmp_path / "caption.h5"
    good = np.array([1, 2], dtype=np.int32)
    _write(path, bad if field == "conditional" else good, bad if field == "negative" else good)
    with pytest.raises(ValueError):
        validate_prompt_file_format(path, COSMOS3_PROMPT_FORMAT, expected_fps=12.5)


@pytest.mark.parametrize("bad", [np.array([0]), np.array([[0]]), 0.5, True])
def test_rejects_malformed_padding_id(tmp_path, bad):
    path = tmp_path / "caption.h5"
    _write(path, [1, 2], [3, 4], bad)
    with pytest.raises(ValueError, match="scalar integer"):
        validate_prompt_file_format(path, COSMOS3_PROMPT_FORMAT)


def test_valid_legacy_file_at_token_limit_is_loadable(tmp_path):
    path = tmp_path / "caption.h5"
    ids = np.arange(COSMOS3_DEFAULT_MAX_TEXT_TOKENS, dtype=np.int32)
    _write(path, ids, ids)
    validate_prompt_file_format(path, COSMOS3_PROMPT_FORMAT, expected_fps=12.5)
    with h5py.File(path, "r") as hf:
        for stream in [hf["frame"][:], hf.attrs["negative_input_ids"]]:
            loaded, mask = pad_cosmos3_input_ids(stream, pad_token_id=int(hf.attrs["pad_token_id"]))
            np.testing.assert_array_equal(loaded.numpy(), ids)
            assert mask.all()
