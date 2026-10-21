# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared camera trajectory JSON helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_INTRINSIC_KEYS = ("w", "h", "fl_x", "fl_y", "cx", "cy")
DISTORTION_KEYS = ("k1", "k2", "p1", "p2")
CAMERA_CONVENTION_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def normalize_pose_matrix(pose: object, label: str) -> list[list[float]]:
    matrix = np.asarray(pose, dtype=np.float64)
    assert matrix.shape == (4, 4), f"{label} must be a 4x4 matrix, got {matrix.shape}"
    return matrix.tolist()


def applied_transform_matrix(transforms: Mapping[str, Any]) -> np.ndarray:
    if "applied_transform" not in transforms:
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(transforms["applied_transform"], dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
    assert matrix.shape == (4, 4), f"applied_transform must be 3x4 or 4x4, got {matrix.shape}"
    return matrix


def opencv_w2c_to_opengl_c2w(world_to_camera: object) -> np.ndarray:
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    assert world_to_camera.shape == (4, 4), f"OpenCV W2C pose must be 4x4, got {world_to_camera.shape}"
    return np.linalg.inv(world_to_camera) @ CAMERA_CONVENTION_FLIP


def opengl_c2w_to_opencv_w2c(camera_to_world: object, applied_transform: np.ndarray | None = None) -> np.ndarray:
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    assert camera_to_world.shape == (4, 4), f"OpenGL C2W pose must be 4x4, got {camera_to_world.shape}"
    if applied_transform is None:
        applied_transform = np.eye(4, dtype=np.float64)
    return CAMERA_CONVENTION_FLIP @ np.linalg.inv(camera_to_world) @ applied_transform


def camera_intrinsics_from_mapping(data: Mapping[str, Any], label: str) -> dict[str, float | int | str]:
    assert isinstance(data, Mapping), f"{label} must be a mapping with camera intrinsics"

    assert "camera_model" in data, f"{label} is missing required camera_model"
    camera_model = str(data["camera_model"])
    assert camera_model == "OPENCV", f"{label} camera_model must be OPENCV, got {camera_model!r}"
    intrinsics: dict[str, float | int | str] = {"camera_model": camera_model}
    for key in REQUIRED_INTRINSIC_KEYS:
        assert key in data, f"{label} is missing required camera intrinsic {key!r}"
        value = data[key]
        assert isinstance(value, int | float) and not isinstance(
            value, bool
        ), f"{label} camera intrinsic {key!r} must be numeric, got {value!r}"
        intrinsics[key] = int(value) if key in ("w", "h") else float(value)

    assert intrinsics["w"] > 0 and intrinsics["h"] > 0, f"{label} image size must be positive"
    assert intrinsics["fl_x"] > 0 and intrinsics["fl_y"] > 0, f"{label} focal length must be positive"

    for key in DISTORTION_KEYS:
        value = data.get(key, 0.0)
        assert isinstance(value, int | float) and not isinstance(
            value, bool
        ), f"{label} camera intrinsic {key!r} must be numeric, got {value!r}"
        intrinsics[key] = float(value)
    return intrinsics


def camera_intrinsics_for_frame(
    transforms: Mapping[str, Any], frame: Mapping[str, Any]
) -> dict[str, float | int | str]:
    frame_intrinsics = dict(transforms)
    for key in ("camera_model", *REQUIRED_INTRINSIC_KEYS, *DISTORTION_KEYS):
        if key in frame:
            frame_intrinsics[key] = frame[key]
    return camera_intrinsics_from_mapping(frame_intrinsics, "frame camera intrinsics")


def frame_with_camera_intrinsics(frame: Mapping[str, Any], intrinsics: Mapping[str, Any]) -> dict[str, object]:
    return {**frame, **camera_intrinsics_from_mapping(intrinsics, "frame camera intrinsics")}


def transforms_json(intrinsics: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    normalized_frames = []
    for index, frame in enumerate(frames):
        assert isinstance(frame, Mapping), f"Frame {index} must be a mapping"
        assert "transform_matrix" in frame, f"Frame {index} is missing transform_matrix"
        normalized = dict(frame)
        normalized["transform_matrix"] = normalize_pose_matrix(frame["transform_matrix"], f"Frame {index}")
        if "file_path" in normalized:
            normalized["file_path"] = str(normalized["file_path"])
        normalized_frames.append(normalized)
    metadata = dict(intrinsics)
    metadata.pop("frames", None)
    metadata.update(camera_intrinsics_from_mapping(intrinsics, "camera intrinsics"))
    metadata["frames"] = normalized_frames
    return metadata


def remove_applied_transform(transforms: Mapping[str, Any]) -> dict[str, object]:
    applied_transform = applied_transform_matrix(transforms)
    inverse_applied_transform = np.linalg.inv(applied_transform)
    intrinsics = dict(transforms)
    intrinsics.pop("applied_transform", None)
    frames = []
    for frame, pose in zip(transforms["frames"], trajectory_poses(transforms)):
        normalized_frame = dict(frame)
        normalized_frame["transform_matrix"] = inverse_applied_transform @ np.asarray(pose, dtype=np.float64)
        frames.append(normalized_frame)
    return transforms_json(intrinsics, frames)


def read_camera_trajectory(path: Path) -> dict[str, object]:
    data = json.loads(path.expanduser().read_text())
    assert isinstance(
        data, Mapping
    ), f"{path} must be a transforms-style JSON object with camera intrinsics and a frames list"
    assert "frames" in data, f"{path} is missing required frames"
    frames = data["frames"]
    assert isinstance(frames, list), f"{path} must contain a frames list"
    assert frames, f"{path} must contain at least one trajectory frame"
    return remove_applied_transform(transforms_json(data, frames))


def assert_target_only_trajectory(trajectory: Mapping[str, Any], label: str) -> None:
    """Reject combined target/context transforms at the user trajectory boundary."""
    frames = trajectory["frames"]
    assert isinstance(frames, list), f"{label} frames must be a list"
    source_frames = [index for index, frame in enumerate(frames) if isinstance(frame, Mapping) and "file_path" in frame]
    assert not source_frames, (
        f"{label} must contain target cameras only; frames with file_path look like source/context images: "
        f"{source_frames[:10]}"
    )


def trajectory_poses(trajectory: Mapping[str, Any]) -> list[list[list[float]]]:
    frames = trajectory["frames"]
    assert isinstance(frames, list), "trajectory frames must be a list"
    poses = []
    for index, frame in enumerate(frames):
        assert isinstance(frame, Mapping), f"Trajectory frame {index} must be a mapping"
        assert "transform_matrix" in frame, f"Trajectory frame {index} is missing transform_matrix"
        poses.append(normalize_pose_matrix(frame["transform_matrix"], f"Trajectory frame {index}"))
    return poses


def trajectory_frame_count(path: Path) -> int:
    return len(read_camera_trajectory(path)["frames"])
