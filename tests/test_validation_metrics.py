# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from model_eval.genfusion_lossutil import ssim as genfusion_ssim
from model_training.data.utils import _average_temporal_c2ws_ref, invert_SE3, latent_index_representative_raw_frame
from model_training.utils.validation_metrics import (
    camera_repeat_pairs,
    fidelity_metrics,
    gain_metrics,
    mean_over_metrics,
    psnr,
    ssim,
    temporal_metrics,
    view_consistency_metrics,
)


def _w2cs(centers: torch.Tensor, rotations: torch.Tensor | None = None) -> torch.Tensor:
    num_frames = centers.shape[0]
    if rotations is None:
        rotations = torch.eye(3).expand(num_frames, -1, -1)
    w2cs = torch.eye(4).repeat(num_frames, 1, 1)
    w2cs[:, :3, :3] = rotations
    w2cs[:, :3, 3] = -torch.einsum("tij,tj->ti", rotations, centers)
    return w2cs


def _palindrome_source_indices(num_frames: int = 81) -> list[int]:
    """Frame indices as data_processing builds them for a ``*_revisit`` clip."""
    forward = list(range(num_frames))
    pivot = math.ceil(len(forward) / 2)
    return forward[:pivot] + forward[: pivot - (1 if len(forward) % 2 else 0)][::-1]


def _pingpong_source_indices(start: int, direction: int, num_frames: int, num_source: int) -> list[int]:
    """The dataset's pingpong walk (dl3dv_base._pingpong_frame_indices) for a given start/direction."""
    period = 2 * num_source - 2
    steps = start + direction * torch.arange(num_frames)
    positions = torch.minimum(steps % period, period - steps % period)
    return positions.tolist()


def _c2ws_along_a_walk(source: list[int]) -> torch.Tensor:
    """Camera-to-world poses for a walk that advances 0.5 units and yaws 0.25 degrees per source frame."""
    c2ws = torch.eye(4).repeat(len(source), 1, 1)
    for frame, index in enumerate(source):
        yaw = math.radians(0.25 * index)
        c2ws[frame, :3, :3] = torch.tensor(
            [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]]
        )
        c2ws[frame, 0, 3] = 0.5 * index
    return c2ws


def test_psnr_and_ssim_are_perfect_on_identical_input():
    video = torch.rand(4, 3, 32, 48)
    assert psnr(video, video).mean() > 100
    assert ssim(video, video).mean() == pytest.approx(1.0, abs=1e-5)


def test_ssim_matches_offline_eval_implementation():
    a, b = torch.rand(3, 3, 64, 64), torch.rand(3, 3, 64, 64)
    per_frame = ssim(a, b)
    for index in range(a.shape[0]):
        expected = genfusion_ssim(a[index : index + 1], b[index : index + 1])
        assert per_frame[index] == pytest.approx(expected.item(), abs=1e-5)


def test_psnr_matches_closed_form():
    a = torch.zeros(1, 3, 8, 8)
    b = torch.full((1, 3, 8, 8), 0.5)
    assert psnr(a, b).item() == pytest.approx(20 * math.log10(1 / 0.5), abs=1e-4)


def test_camera_repeat_pairs_recovers_palindrome_and_pairs_matching_poses():
    source = _palindrome_source_indices()
    centers = torch.tensor([[0.5 * index, 0.0, 0.0] for index in source])

    pairs = camera_repeat_pairs(_w2cs(centers))

    assert pairs
    assert all(source[i] == source[j] for i, j in pairs)
    assert all(i < j for i, j in pairs)


@pytest.mark.parametrize("alignment", ["mean", "last"])
@pytest.mark.parametrize("start,direction", [(40, 1), (57, 1), (23, -1), (30, -1)])
def test_camera_repeat_pairs_recovers_pingpong_revisits_from_per_latent_poses(alignment, start, direction):
    """The trainer only has per-LATENT poses: under ``mean`` alignment a latent pose is the average of
    its raw group, and the averages on the way out and on the way back never coincide exactly, so a
    tolerance at the raw-frame level found no pairs on most clips."""
    num_raw = 81
    source = _pingpong_source_indices(start, direction, num_raw, num_source=81)
    latent_c2ws = _average_temporal_c2ws_ref(_c2ws_along_a_walk(source), skip_vae_check=False, pose_alignment=alignment)
    latent_to_frame = [
        latent_index_representative_raw_frame(index, num_raw, pose_alignment=alignment)
        for index in range(latent_c2ws.shape[0])
    ]
    w2cs = invert_SE3(latent_c2ws)

    pairs = camera_repeat_pairs(w2cs)

    assert len(pairs) >= 3
    if alignment == "mean":
        # The group averages on the two legs never coincide, so the raw-level tolerance alone finds at
        # most one exact coincidence (latent 0 is a single frame, not an average).
        assert len(camera_repeat_pairs(w2cs, step_tolerance_fraction=0.0)) <= 1
    # Matched frames were commanded to within two raw frames (half a latent step) of each other.
    assert all(abs(source[latent_to_frame[i]] - source[latent_to_frame[j]]) <= 2 for i, j in pairs)


def test_camera_repeat_pairs_is_empty_for_monotonic_trajectory():
    centers = torch.tensor([[0.5 * index, 0.0, 0.0] for index in range(81)])
    assert camera_repeat_pairs(_w2cs(centers)) == []


def test_camera_repeat_pairs_matches_each_frame_at_most_once():
    centers = torch.zeros(40, 3)
    pairs = camera_repeat_pairs(_w2cs(centers))
    used = [index for pair in pairs for index in pair]
    assert len(used) == len(set(used))


def test_camera_repeat_pairs_rejects_shared_center_with_different_orientation():
    angle = math.radians(30)
    rotated = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    centers = torch.cat([torch.zeros(1, 3), torch.tensor([[9.0, 0.0, 0.0]]).expand(8, -1), torch.zeros(1, 3)])
    rotations = torch.cat([torch.eye(3).expand(9, -1, -1), rotated.unsqueeze(0)])

    assert (0, 9) not in camera_repeat_pairs(_w2cs(centers, rotations))


def test_camera_repeat_pairs_ignores_neighboring_frames():
    centers = torch.zeros(10, 3)
    assert all(j - i >= 4 for i, j in camera_repeat_pairs(_w2cs(centers), min_frame_gap=4))


def test_view_consistency_rewards_a_video_that_repeats_content_at_repeated_poses():
    source = _palindrome_source_indices(20)
    centers = torch.tensor([[0.5 * index, 0.0, 0.0] for index in source])
    # A video that obeys the camera shows the same content whenever the pose repeats.
    consistent = torch.rand(20, 3, 16, 16)[source]
    drifting = torch.rand(20, 3, 16, 16)

    consistent_metrics = view_consistency_metrics(
        {"generated": consistent, "rendered": consistent, "gt": consistent}, _w2cs(centers)
    )
    drifting_metrics = view_consistency_metrics({"generated": drifting, "rendered": consistent}, _w2cs(centers))

    assert consistent_metrics["num_repeat_pairs"] > 0
    assert consistent_metrics["revisit_psnr_generated"] > 100
    assert consistent_metrics["revisit_psnr_vs_rendered"] == pytest.approx(0.0, abs=1e-6)
    assert drifting_metrics["revisit_psnr_generated"] < consistent_metrics["revisit_psnr_generated"]
    assert drifting_metrics["revisit_psnr_vs_rendered"] < 0


def test_view_consistency_is_empty_without_repeated_views():
    centers = torch.tensor([[0.5 * index, 0.0, 0.0] for index in range(20)])
    assert view_consistency_metrics({"generated": torch.rand(20, 3, 8, 8)}, _w2cs(centers)) == {"num_repeat_pairs": 0.0}


def test_view_consistency_maps_latent_poses_onto_raw_frames():
    source = _palindrome_source_indices(21)
    centers = torch.tensor([[0.5 * index, 0.0, 0.0] for index in source])
    latent_to_frame = [
        latent_index_representative_raw_frame(index, 81, pose_alignment="last") for index in range(len(source))
    ]

    video = torch.rand(81, 3, 8, 8)
    metrics = view_consistency_metrics({"generated": video}, _w2cs(centers), latent_to_frame=latent_to_frame)

    assert metrics["num_repeat_pairs"] > 0
    assert max(latent_to_frame) < video.shape[0]


def test_fidelity_metrics_clamps_out_of_range_input():
    target = torch.zeros(1, 3, 8, 8)
    assert fidelity_metrics(torch.full((1, 3, 8, 8), 4.0), target)["psnr"] == pytest.approx(
        fidelity_metrics(torch.ones(1, 3, 8, 8), target)["psnr"]
    )


def test_mean_over_metrics_averages_shared_keys_and_skips_bookkeeping():
    averaged = mean_over_metrics([{"psnr": 10.0, "ssim": 0.5, "num_repeat_pairs": 3.0}, {"psnr": 20.0}])
    assert averaged == {"psnr": 15.0, "ssim": 0.5}


@pytest.mark.parametrize(
    ("latent_index", "alignment", "expected"),
    [(0, "last", 0), (1, "last", 4), (20, "last", 80), (0, "mean", 0), (1, "mean", 2)],
)
def test_latent_index_representative_raw_frame(latent_index, alignment, expected):
    assert latent_index_representative_raw_frame(latent_index, 81, pose_alignment=alignment) == expected


def test_gain_is_positive_when_the_generated_video_beats_the_render():
    gains = gain_metrics({"psnr": 24.0, "ssim": 0.80}, {"psnr": 20.0, "ssim": 0.70})

    assert gains["psnr_gain"] == pytest.approx(4.0)
    assert gains["ssim_gain"] == pytest.approx(0.10)


def test_gain_omits_metrics_missing_from_either_side():
    assert gain_metrics({"psnr": 24.0}, {"psnr": 20.0, "ssim": 0.7}) == {"psnr_gain": pytest.approx(4.0)}
    assert gain_metrics({}, {}) == {}


def test_ssim_is_unchanged_by_the_frame_chunking():
    torch.manual_seed(0)
    a, b = torch.rand(19, 3, 24, 24), torch.rand(19, 3, 24, 24)
    chunked = ssim(a, b)
    per_frame = torch.cat([ssim(a[i : i + 1], b[i : i + 1]) for i in range(a.shape[0])])
    assert torch.allclose(chunked, per_frame, atol=1e-6)


@pytest.mark.parametrize("short_frames", [1, 7, 8, 15, 16])
@pytest.mark.parametrize("reverse", [False, True])
def test_psnr_and_ssim_refuse_mismatched_shapes(short_frames, reverse):
    """Reject mismatches in either order, including surplus frames beyond a full chunk."""
    a = torch.rand(short_frames, 3, 13, 17)
    b = torch.rand(short_frames + 1, 3, 13, 17)
    if reverse:
        a, b = b, a
    with pytest.raises(ValueError, match="matching"):
        psnr(a, b)
    with pytest.raises(ValueError, match="matching"):
        ssim(a, b)


def test_rotation_only_revisits_are_scored():
    angles = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2, 0.0])
    rotations = torch.eye(3).repeat(len(angles), 1, 1)
    rotations[:, 0, 0] = rotations[:, 2, 2] = angles.cos()
    rotations[:, 0, 2] = angles.sin()
    rotations[:, 2, 0] = -angles.sin()
    pairs = camera_repeat_pairs(_w2cs(torch.zeros(len(angles), 3), rotations))
    assert (0, 8) in pairs
    assert (1, 7) in pairs
    assert camera_repeat_pairs(_w2cs(torch.zeros(len(angles), 3))) == []


def test_frozen_video_is_perfectly_repeatable_but_fails_motion_diagnostic():
    offsets = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    gt = (offsets / 4)[:, None, None, None].expand(-1, 3, 8, 8)
    frozen = gt[:1].expand_as(gt)
    poses = _w2cs(torch.stack((offsets, offsets * 0, offsets * 0), dim=1))
    assert view_consistency_metrics({"generated": frozen}, poses)["revisit_psnr_generated"] == 120.0
    metrics = temporal_metrics(frozen, gt, chunk_frames=4)
    assert metrics["motion_ratio"] == 0
    assert metrics["temporal_delta_l1"] > 0
    assert temporal_metrics(gt, gt)["temporal_delta_l1"] == 0


def test_chunk_degradation_and_boundary_error():
    gt = torch.zeros(16, 3, 8, 8)
    generated = gt.clone()
    generated[8:] = 1
    metrics = temporal_metrics(generated, gt, chunk_frames=8)
    assert metrics["chunk000_psnr"] == 120
    assert metrics["chunk001_psnr"] == 0
    assert metrics["last_minus_first_psnr"] == -120
    assert metrics["boundary_delta_l1"] == 1
    with pytest.raises(ValueError, match="shapes differ"):
        fidelity_metrics(generated[:8], gt)


def test_sampler_boundary_accounts_for_first_latents_single_raw_frame():
    gt = torch.zeros(33, 3, 8, 8)
    generated = gt.clone()
    generated[29:] = 1
    metrics = temporal_metrics(generated, gt, chunk_frames=32, sampler_chunk_latents=8, temporal_scale=4)
    assert metrics["boundary_delta_l1"] == 0
    assert metrics["sampler_boundary_delta_l1"] == 1
