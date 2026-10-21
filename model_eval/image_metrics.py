# SPDX-FileCopyrightText: Copyright (c) 2025 Sibo Wu
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Canonical per-image PSNR/SSIM tensor kernels, without perceptual-metric dependencies.

SSIM retains the GenFusion window and reduction convention. PSNR has no MSE
floor by default, including infinity for an exact match. Callers that require
finite reports must explicitly select and record their positive MSE floor.
Input normalization, device batching, and aggregation belong to callers.
"""

from math import exp, isfinite

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def psnr(img1: torch.Tensor, img2: torch.Tensor, *, mse_floor: float | None = None) -> torch.Tensor:
    """Return one PSNR per image in a [N,C,H,W] batch with data range 1."""
    if img1.shape != img2.shape or img1.ndim != 4:
        raise ValueError("PSNR expects matching [N,C,H,W] tensors")
    mse = ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean(1)
    if mse_floor is not None:
        if not isfinite(mse_floor) or mse_floor <= 0:
            raise ValueError("mse_floor must be positive and finite, or None for benchmark PSNR")
        mse = mse.clamp_min(mse_floor)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    if img1.shape != img2.shape or img1.ndim != 4:
        raise ValueError("SSIM expects matching [N,C,H,W] tensors")
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
