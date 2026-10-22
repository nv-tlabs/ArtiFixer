# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import subprocess
import sys

import pytest
import torch

from model_eval.genfusion_lossutil import ssim as offline_ssim
from model_eval.image_metrics import psnr, ssim
from model_eval.metrics_utils import psnr_genfusion
from model_training.utils.validation_metrics import METRIC_PROTOCOL, fidelity_metrics
from model_training.utils.validation_metrics import psnr as validation_psnr
from model_training.utils.validation_metrics import ssim as validation_ssim


@pytest.mark.parametrize("delta", [0.0, 1e-7, 0.01, 0.2])
def test_shared_psnr_preserves_benchmark_and_explicit_finite_validation_policy(delta):
    a = torch.zeros(3, 3, 13, 17, dtype=torch.float64)
    b = torch.full_like(a, delta)
    benchmark = psnr_genfusion(a, b)
    assert benchmark.shape == (3, 1)
    torch.testing.assert_close(benchmark[:, 0], psnr(a, b), rtol=0, atol=0)
    expected = math.inf if delta == 0 else -20 * math.log10(delta)
    assert benchmark[:, 0].tolist() == pytest.approx([expected] * 3)
    finite = validation_psnr(a, b)
    assert finite.tolist() == pytest.approx([min(expected, 120.0)] * 3)
    torch.testing.assert_close(finite, psnr(a, b, mse_floor=METRIC_PROTOCOL["psnr"]["mse_floor"]), rtol=0, atol=0)


def test_shared_ssim_retains_established_benchmark_values_and_per_image_reduction():
    a = torch.arange(3 * 3 * 13 * 17, dtype=torch.float32).reshape(3, 3, 13, 17).remainder(37) / 36
    b = a.roll(2, dims=-1) * 0.9 + 0.03
    # Frozen from genfusion_lossutil.py at pre-refactor 5e38da66.
    expected = torch.tensor([0.6954503059387207, 0.6998953223228455, 0.7076416611671448])
    values = ssim(a, b, size_average=False)
    torch.testing.assert_close(values, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(offline_ssim(a, b, size_average=False), expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(validation_ssim(a, b), expected, rtol=0, atol=1e-7)
    assert offline_ssim(a, b).item() == pytest.approx(values.mean().item(), abs=1e-7)
    torch.testing.assert_close(ssim(a, a, size_average=False), torch.ones(3), rtol=0, atol=0)


@pytest.mark.parametrize("frames,chunk", [(1, 8), (11, 3), (19, 8)])
def test_validation_chunks_aggregate_per_image_metrics_without_changing_protocol(frames, chunk):
    a = torch.zeros(frames, 3, 13, 17)
    differences = torch.linspace(0.01, 0.2, frames)
    b = differences[:, None, None, None].expand_as(a).clone()
    expected_psnr = -20 * differences.double().log10()
    torch.testing.assert_close(psnr_genfusion(a, b)[:, 0].double(), expected_psnr, atol=1e-5, rtol=0)
    result = fidelity_metrics(a, b, device="cpu", chunk=chunk)
    assert result["psnr"] == pytest.approx(expected_psnr.mean().item(), abs=1e-5)
    assert result["ssim"] == pytest.approx(offline_ssim(a, b, size_average=False).mean().item(), abs=1e-7)


def test_tensor_metrics_import_without_perceptual_metric_dependencies():
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    assert not name.startswith(('torchmetrics', 'model_eval.lpipsPytorch', 'lpips')), name
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from model_eval.image_metrics import psnr, ssim
from model_eval.metrics_utils import psnr_genfusion
from model_training.utils.validation_metrics import fidelity_metrics
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
