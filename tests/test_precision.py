# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""disable_tf32_matmul turns TF32 off for fp32 matmul whatever the container's
startup default (the NGC images ship TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1)."""

import pytest
import torch

from model_training.utils.precision import disable_tf32_matmul


@pytest.fixture
def restore_matmul_precision():
    previous = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(previous)


def test_the_runtime_setter_wins_over_a_tf32_default(restore_matmul_precision):
    torch.backends.cuda.matmul.allow_tf32 = True

    disable_tf32_matmul()

    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 rounding happens only in a CUDA matmul")
def test_small_batched_fp32_matmul_is_exact_after_the_call(restore_matmul_precision):
    """1 + 2**-12 has no TF32 representation (10 mantissa bits): a TF32 kernel rounds it to
    1.0 and the product is off by 2.44e-4 at the batched 4x4 camera-pose shape; an fp32
    kernel returns it exactly."""
    value = 1 + 2**-12
    batch = torch.full((8, 4, 4), value, device="cuda")
    eye = torch.eye(4, device="cuda").expand(8, 4, 4)

    torch.backends.cuda.matmul.allow_tf32 = True
    disable_tf32_matmul()

    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.equal(batch @ eye, batch)
