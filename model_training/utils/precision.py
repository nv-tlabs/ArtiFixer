# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical-precision settings shared by training and standalone eval: if only one
side sets them, panels and metrics from different clusters are not comparable."""

import torch


def disable_tf32_matmul(verbose: bool = False) -> None:
    """Disable TF32 for fp32 matmul so the camera geometry is architecture-portable.

    cuBLAS picks architecture-dependent kernels for small batched fp32 matmuls under
    TF32, so the batched 4x4 camera-pose products come out bitwise different across
    GPU generations; with TF32 off they agree. This model keeps its geometry in fp32
    (PRoPE, the Plucker decomposition) while the rest runs bf16, and those matmuls
    feed all of Q/K, so the difference reaches every attention score.

    cuDNN's TF32 is left alone. The NGC containers turn TF32 on through
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which only sets the initial precision, so
    this runtime setter takes precedence; ``verbose`` logs the effective precision
    so a silent re-enable on a container bump is visible.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    if verbose:
        print(
            f"fp32 matmul precision: {torch.get_float32_matmul_precision()} "
            f"(matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32})",
            flush=True,
        )
