# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Nerfbusters train/test split convention, shared by the eval dataset and
the metrics and reconstruction scripts.

Keep imports to the standard library so all of them can use it.
"""

import os
import re


def nerfbusters_train_test_split(transforms: dict) -> tuple[set, set]:
    """
    Get train/test split based on the Nerfbusters filename convention.

    Nerfbusters naming:
    - frame_XXXXX.png = TRAIN (no _1_ in middle)
    - frame_1_XXXXX.png = TEST (has _1_ in middle)

    Returns:
        train_ids: Set of transforms indices for train frames
        test_ids: Set of transforms indices for test frames
    """
    train_ids = set()
    test_ids = set()

    train_pattern = re.compile(r"^frame_\d+\.png$")

    for idx, frame in enumerate(transforms["frames"]):
        basename = os.path.basename(frame["file_path"])
        if train_pattern.match(basename):
            train_ids.add(idx)
        else:
            test_ids.add(idx)

    return train_ids, test_ids
