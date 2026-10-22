# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the first-target camera origin and matching inputs in training and evaluation."""

import pytest
import torch

from model_training.data.utils import compute_camera_conditioning, compute_camera_rays
from model_training.model_profiles import requires_reference_camera_rays
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT

INTRINSICS = {"w": 4, "h": 4, "fl_x": 4.0, "fl_y": 4.0, "cx": 2.0, "cy": 2.0}


@pytest.mark.parametrize("prompt_format,required", [(WAN_T5_PROMPT_FORMAT, False), (COSMOS3_PROMPT_FORMAT, True)])
def test_reference_camera_rays_follow_the_shared_model_policy(prompt_format, required):
    assert requires_reference_camera_rays(prompt_format) is required


def _poses(count: int, offset: float) -> torch.Tensor:
    """Use different camera positions so selecting the wrong one is detectable."""
    poses = torch.eye(4).expand(count, -1, -1).clone()
    poses[:, :3, 3] = torch.arange(count, dtype=torch.float32).unsqueeze(-1) + offset
    return poses


def _transforms(c2ws: torch.Tensor, neighbor_c2ws: torch.Tensor) -> dict:
    """Build a scene record with target cameras first and reference cameras afterwards."""
    return {
        **INTRINSICS,
        "frames": [{"transform_matrix": pose.tolist()} for pose in torch.cat([c2ws, neighbor_c2ws])],
    }


@pytest.mark.parametrize("num_neighbors", [0, 3])
@pytest.mark.parametrize("pose_alignment", ["mean", "last"])
@pytest.mark.parametrize("include_neighbor_rays", [False, True])
def test_first_target_is_the_origin_with_or_without_reference_views(
    num_neighbors, pose_alignment, include_neighbor_rays
):
    c2ws, neighbors = _poses(5, offset=10.0), _poses(num_neighbors, offset=100.0)
    item = compute_camera_conditioning(
        INTRINSICS,
        c2ws,
        neighbors,
        scale=0.5,
        image_height=4,
        image_width=4,
        pose_alignment=pose_alignment,
        include_neighbor_rays=include_neighbor_rays,
    )
    torch.testing.assert_close(item["w2cs"][0], torch.eye(4))
    torch.testing.assert_close(item["camera_rays"][0, ..., :3], torch.zeros(4, 4, 3))
    torch.testing.assert_close(item["neighbor_w2cs"][:, :3, 3], -(neighbors[:, :3, 3] - c2ws[0, :3, 3]) * 0.5)
    assert ("neighbor_camera_rays" in item) == include_neighbor_rays
    if include_neighbor_rays:
        assert item["neighbor_camera_rays"].shape == (num_neighbors, 4, 4, 6)


@pytest.mark.parametrize("pose_alignment", ["mean", "last"])
@pytest.mark.parametrize("include_neighbor_rays", [False, True])
def test_train_and_eval_builders_agree_on_the_origin(pose_alignment, include_neighbor_rays):
    """The same cameras and setting must produce matching training and evaluation inputs.

    The evaluation helper requires reference views. The separate zero-reference
    test covers samples without them.
    """
    num_neighbors = 3
    c2ws, neighbors = _poses(5, offset=10.0), _poses(num_neighbors, offset=100.0)
    target_indices = list(range(5))
    neighbor_indices = list(range(5, 5 + num_neighbors))

    trained = compute_camera_conditioning(
        INTRINSICS,
        c2ws.clone(),
        neighbors.clone(),
        scale=0.5,
        image_height=4,
        image_width=4,
        pose_alignment=pose_alignment,
        include_neighbor_rays=include_neighbor_rays,
    )
    evaluated = compute_camera_rays(
        _transforms(c2ws, neighbors),
        target_indices,
        neighbor_indices,
        scale=0.5,
        image_shape=(4, 4),
        pose_alignment=pose_alignment,
        include_neighbor_rays=include_neighbor_rays,
    )

    assert trained.keys() == evaluated.keys()
    for key in trained:
        torch.testing.assert_close(trained[key], evaluated[key], msg=f"{key} disagrees")
