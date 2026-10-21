# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera pose alignment: which pose each latent frame's raymap/PRoPE geometry
uses from its raw-frame group (see CAMERA_POSE_ALIGNMENTS in data/utils.py).
"""

import pytest
import torch

from model_training.arguments import get_common_opts
from model_training.data.utils import compute_camera_conditioning
from model_training.model_profiles import default_camera_pose_alignment

NUM_RAW_FRAMES = 13  # 1 + 4 * 3 -> 4 latent frames
TEMPORAL_SCALE = 4


def _uniform_translation_c2ws(num_frames: int) -> torch.Tensor:
    """Identity rotations, camera moving one unit along x per raw frame."""
    c2ws = torch.eye(4).expand(num_frames, -1, -1).clone()
    c2ws[:, 0, 3] = torch.arange(num_frames, dtype=torch.float32)
    return c2ws


def _conditioning(pose_alignment: str, c2ws: torch.Tensor) -> dict:
    return compute_camera_conditioning(
        intrinsics={"w": 32, "h": 32, "fl_x": 32.0, "fl_y": 32.0, "cx": 16.0, "cy": 16.0},
        c2ws_world=c2ws,
        neighbor_c2ws_world=c2ws[:1],
        scale=1.0,
        image_height=32,
        image_width=32,
        pose_alignment=pose_alignment,
    )


def _camera_positions(item: dict) -> torch.Tensor:
    """Per-latent camera position along x, recovered from the returned w2cs."""
    w2cs = item["w2cs"]
    return -torch.einsum("nij,nj->ni", w2cs[:, :3, :3].transpose(1, 2), w2cs[:, :3, 3])[:, 0]


def test_last_alignment_samples_the_true_trajectory():
    """ "last" is the real camera path at raw frames 0, 4, 8, ...: uniform
    motion yields uniform per-latent steps."""
    item = _conditioning("last", _uniform_translation_c2ws(NUM_RAW_FRAMES))
    positions = _camera_positions(item)
    torch.testing.assert_close(positions, torch.tensor([0.0, 4.0, 8.0, 12.0]))


def test_mean_alignment_understates_the_first_step():
    """The documented "mean" falsehood: group means sit at times 0, 2.5,
    6.5, ..., so even perfectly uniform motion starts with a ~60% first step.
    Pinned because released checkpoints were trained against exactly this."""
    item = _conditioning("mean", _uniform_translation_c2ws(NUM_RAW_FRAMES))
    positions = _camera_positions(item)
    torch.testing.assert_close(positions, torch.tensor([0.0, 2.5, 6.5, 10.5]))


def test_last_alignment_reports_a_direction_reversal_the_mean_erases():
    """A pingpong bounce at the clip start (the greenhouse trajectory shape,
    e.g. frames 265, 264, 265, 266, ...): the mean collapses the first group
    onto the first latent's pose ("camera frozen"), while "last" keeps the
    motion visible."""
    positions = torch.tensor([1.0, 0.0, 1.0, 2.0, 3.0] + list(range(4, 4 + NUM_RAW_FRAMES - 5)), dtype=torch.float32)
    c2ws = torch.eye(4).expand(NUM_RAW_FRAMES, -1, -1).clone()
    c2ws[:, 0, 3] = positions

    mean_step = torch.diff(_camera_positions(_conditioning("mean", c2ws)))[0]
    last_step = torch.diff(_camera_positions(_conditioning("last", c2ws)))[0]

    torch.testing.assert_close(mean_step, torch.tensor(0.5))
    torch.testing.assert_close(last_step, torch.tensor(2.0))


def test_last_alignment_latent_slice_matches_the_full_clip():
    """CP training slices per-rank latent ranges; the sliced geometry must be
    the corresponding rows of the full-clip geometry."""
    c2ws = _uniform_translation_c2ws(NUM_RAW_FRAMES)
    full = _conditioning("last", c2ws)
    sliced = compute_camera_conditioning(
        intrinsics={"w": 32, "h": 32, "fl_x": 32.0, "fl_y": 32.0, "cx": 16.0, "cy": 16.0},
        c2ws_world=c2ws,
        neighbor_c2ws_world=c2ws[:1],
        scale=1.0,
        image_height=32,
        image_width=32,
        pose_alignment="last",
        latent_slice=(1, 3),
    )
    torch.testing.assert_close(sliced["w2cs"], full["w2cs"][1:3])
    torch.testing.assert_close(sliced["camera_rays"], full["camera_rays"][1:3])


def test_unknown_alignment_is_rejected():
    with pytest.raises(AssertionError, match="pose_alignment"):
        _conditioning("endpoint", _uniform_translation_c2ws(NUM_RAW_FRAMES))


def test_family_defaults():
    """Wan keeps "mean" so released checkpoints keep working; cosmos3 has
    nothing released to protect and trains on the real trajectory."""
    assert default_camera_pose_alignment("Wan-AI/Wan2.1-T2V-14B-Diffusers") == "mean"
    assert default_camera_pose_alignment("nvidia/Cosmos3-Nano") == "last"


def test_training_parser_fills_the_family_default():
    """Every training entry parses through get_common_opts, so this covers
    stage 1, diffusion forcing, and distillation launches."""
    argv = ["--split_path", "/s", "--dl3dv_dir", "/d", "--prompt_dir", "/p"]
    cosmos3 = argv + ["--model_id", "nvidia/Cosmos3-Nano"]
    assert get_common_opts().parse_args(argv).camera_pose_alignment == "mean"
    assert get_common_opts().parse_args(cosmos3).camera_pose_alignment == "last"
    forced = cosmos3 + ["--camera_pose_alignment", "mean"]
    assert get_common_opts().parse_args(forced).camera_pose_alignment == "mean"
