# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""load_prompt_payload_from_path reads exactly one caption file; there is no selection."""

import h5py
import numpy as np
import pytest

from model_training.constants import MAX_SEQUENCE_LENGTH
from model_training.data.utils import load_prompt_payload_from_path


def test_reads_the_given_file(tmp_path):
    prompt_path = tmp_path / "caption.hdf5"
    with h5py.File(prompt_path, "w") as f:
        dataset = f.create_dataset("frame_00001.png", data=np.zeros((4, 8), dtype=np.uint16))
        dataset.attrs["caption"] = "a forward caption"

    payload, caption = load_prompt_payload_from_path(prompt_path)

    assert caption == "a forward caption"
    assert payload["encoded_prompt"].shape[0] == MAX_SEQUENCE_LENGTH


def test_rejects_multi_record_files(tmp_path):
    # A multi-record file would silently condition on the alphabetically-first
    # record; the loader must fail loudly, naming the file and record count.
    prompt_path = tmp_path / "caption.hdf5"
    with h5py.File(prompt_path, "w") as f:
        for name in ("frame_00001.png", "frame_00002.png"):
            dataset = f.create_dataset(name, data=np.zeros((4, 8), dtype=np.uint16))
            dataset.attrs["caption"] = f"caption for {name}"

    with pytest.raises(ValueError, match=r"caption\.hdf5 holds 2 caption records.*exactly one"):
        load_prompt_payload_from_path(prompt_path)
