# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests._helpers import argparse_default


@pytest.mark.parametrize(
    ("script_path", "expected_default"),
    [
        ("model_training/train.py", 25000),
        ("model_training/distillation.py", 1000),
    ],
)
def test_release_training_iteration_defaults_are_on_main(script_path: str, expected_default: int) -> None:
    assert argparse_default(script_path, "--max_iterations") == expected_default
