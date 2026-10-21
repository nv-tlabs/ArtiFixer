# SPDX-FileCopyrightText: Copyright (c) 2025 Sibo Wu
# SPDX-FileCopyrightText: Copyright (c) 2018 Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND BSD-2-Clause AND Apache-2.0

from collections import OrderedDict

import torch

# adapted from genfusion codebase: https://github.com/Inception3D/GenFusion/tree/main/Reconstruction/lpipsPyTorch


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = "alex", version: str = "0.1"):
    # build url
    url = (
        "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
        + f"master/lpips/weights/v{version}/{net_type}.pth"
    )

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url,
        progress=True,
        map_location="cpu",
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace("lin", "")
        new_key = new_key.replace("model.", "")
        new_state_dict[new_key] = val

    return new_state_dict
