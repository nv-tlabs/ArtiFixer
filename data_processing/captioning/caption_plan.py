# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caption output naming and variant helpers."""

from __future__ import annotations

import re
from pathlib import Path

from data_processing.scene_utils import load_scene_transforms_from_dir, scene_frame_count

_STRIDE_PATTERN = re.compile(r"(?:^|_)stride_(\d+)(?:_|$)")


def caption_output_path(
    scene_output_dir: Path,
    num_frames: int | None,
    frame_stride: int,
    *,
    revisit: bool = False,
    reverse: bool = False,
) -> Path:
    name_parts = [
        f"frames_{num_frames if num_frames is not None and num_frames > 0 else 'all'}",
        f"stride_{frame_stride}",
    ]
    if revisit:
        name_parts.append("revisit")
    if reverse:
        name_parts.append("reverse")
    return scene_output_dir / ("_".join(name_parts) + ".h5")


def caption_variants(
    max_frame_stride: int, *, include_reverse: bool, include_revisit: bool
) -> list[tuple[int, bool, bool]]:
    variants: list[tuple[int, bool, bool]] = []
    reverse_values = [False, True] if include_reverse else [False]
    revisit_values = [False, True] if include_revisit else [False]
    for frame_stride in range(1, max_frame_stride + 1):
        for revisit in revisit_values:
            for reverse in reverse_values:
                variants.append((frame_stride, revisit, reverse))
    return variants


def scene_limited_num_frames(num_frames: int | None, scene_path: Path) -> int | None:
    """Resolve the effective prompt window length for one scene.

    ``None``/``<= 0`` requests and scenes shorter than the window collapse to
    a single full-clip ``frames_all`` caption. The scene is only read for
    positive window lengths.
    """
    if num_frames is None or num_frames <= 0:
        return None
    if scene_path.suffix == ".zip":
        scene_num_frames = scene_frame_count(scene_path)
    else:
        scene_num_frames = len(load_scene_transforms_from_dir(scene_path)["frames"])
    if scene_num_frames < num_frames:
        return None
    return num_frames


def caption_frame_stride(path: Path) -> int:
    match = _STRIDE_PATTERN.search(path.stem)
    assert match is not None, f"Could not infer frame stride from prompt filename: {path.name}"
    frame_stride = int(match.group(1))
    assert frame_stride >= 1, f"Frame stride must be >= 1, got {frame_stride}"
    return frame_stride
