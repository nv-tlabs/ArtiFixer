# SPDX-FileCopyrightText: Copyright (c) 2025 Sibo Wu
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""
Taken from genfusion https://github.com/Inception3D/GenFusion/blob/main/Reconstruction/utils/loss_utils.py
"""

import torch

from model_eval.image_metrics import _ssim, create_window, gaussian, ssim


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:, 1:-1, :-2] + disp[:, 1:-1, 2:] - 2 * disp[:, 1:-1, 1:-1])
    grad_disp_y = torch.abs(disp[:, :-2, 1:-1] + disp[:, 2:, 1:-1] - 2 * disp[:, 1:-1, 1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()
