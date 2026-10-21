# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The repo stays formatted: black and isort (config in pyproject.toml) must be no-ops."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _check(module: str, *flags: str) -> None:
    pytest.importorskip(module)
    result = subprocess.run(
        [sys.executable, "-m", module, *flags, str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{module} exited {result.returncode}:\n{result.stdout}{result.stderr}"


def test_black_is_a_noop():
    _check("black", "--check")


def test_isort_is_a_noop():
    _check("isort", "--check-only", "--quiet")
