# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the scene-prep CLIs and eval datasets.

Keep imports here limited to the stdlib and threedgrut dataset types:
this module is imported by eval datasets and per-phase CLIs that should
not have to load the captioning, metric-alignment, or 3DGRUT training
dependencies.
"""

from __future__ import annotations

import shutil
import struct
from collections.abc import Sequence
from pathlib import Path

from threedgrut.datasets.utils import CAMERA_MODEL_NAMES, Camera, Image


def parse_phases(value: str, valid: set[str]) -> set[str]:
    """Parse a comma-separated list of phase names, rejecting unknown or empty ones."""
    phases = {phase.strip() for phase in value.split(",") if phase.strip()}
    unknown = phases - valid
    assert not unknown, f"Unknown phases: {sorted(unknown)}"
    assert phases, f"--phases must include at least one of {sorted(valid)}"
    return phases


def reset_directory(path: Path) -> None:
    """Delete the directory if it exists and recreate it empty.

    Refuses symlinks so rmtree cannot delete a directory outside the output tree.
    """
    if path.exists():
        assert path.is_dir() and not path.is_symlink(), f"Expected output directory, got {path}"
        shutil.rmtree(path)
    path.mkdir(parents=True)


def colmap_params_to_opencv(model: str, params: Sequence[float]) -> tuple[float, float, float, float, tuple]:
    """Map a supported COLMAP camera model onto OPENCV fx, fy, cx, cy, (k1, k2, p1, p2)."""
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return f, f, cx, cy, (0.0, 0.0, 0.0, 0.0)
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        return fx, fy, cx, cy, (0.0, 0.0, 0.0, 0.0)
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        return f, f, cx, cy, (k1, 0.0, 0.0, 0.0)
    if model == "RADIAL":
        f, cx, cy, k1, k2 = params
        return f, f, cx, cy, (k1, k2, 0.0, 0.0)
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        return fx, fy, cx, cy, (k1, k2, p1, p2)
    assert False, f"Unsupported COLMAP camera model: {model}"


def image_basename(image: Image) -> str:
    return Path(image.name).name


def write_colmap_cameras(path: Path, cameras: Sequence[Camera]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            model = CAMERA_MODEL_NAMES[camera.model]
            fid.write(struct.pack("<iiQQ", camera.id, model.model_id, int(camera.width), int(camera.height)))
            fid.write(struct.pack("<" + "d" * len(camera.params), *camera.params))


def write_colmap_images(path: Path, images: Sequence[Image]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images:
            fid.write(struct.pack("<idddddddi", image.id, *image.qvec, *image.tvec, image.camera_id))
            fid.write(image_basename(image).encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image.xys)))
            for xy, point3d_id in zip(image.xys, image.point3D_ids):
                fid.write(struct.pack("<ddq", float(xy[0]), float(xy[1]), int(point3d_id)))
