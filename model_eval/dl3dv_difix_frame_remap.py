# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiFix frame-space remapping for DL3DV eval.

DiFix train ids and visibility masks were generated against a compact DiFix
frame list. Release eval reads DL3DV-ALL-960P, whose zips still contain a few
frames that the DiFix frame space excludes. Dropping those frames before
indexing keeps render ids, train ids, masks, and GT images in the same compact
coordinate system.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_DIFIX_FRAME_REMAP_PATH = (
    Path(__file__).resolve().parents[1] / "data_processing" / "dl3dv_difix_frame_remap.json"
)


@lru_cache(maxsize=None)
def load_difix_frame_remap(remap_path: Path = DEFAULT_DIFIX_FRAME_REMAP_PATH) -> dict[str, set[str]]:
    with Path(remap_path).open() as f:
        data = json.load(f)

    scenes = data.get("scenes", {})
    assert isinstance(scenes, dict), f"{remap_path} must contain a 'scenes' mapping"
    remap: dict[str, set[str]] = {}
    for scene_id, metadata in scenes.items():
        dropped = metadata.get("dropped_file_basenames", [])
        assert isinstance(dropped, list), f"{scene_id} dropped_file_basenames must be a list"
        assert all(isinstance(name, str) for name in dropped), f"{scene_id} dropped frame names must be strings"
        assert len(set(dropped)) == len(dropped), f"{scene_id} dropped frame names must be unique"
        remap[scene_id] = set(dropped)
    return remap


def apply_difix_frame_remap(transforms: dict[str, Any], scene_id: str) -> dict[str, Any]:
    """Return transforms in DiFix's compact frame space.

    The operation is idempotent: compact DiFix transforms have already dropped
    these frames, while DL3DV-ALL-960P zip transforms have not.
    """

    dropped_basenames = load_difix_frame_remap().get(scene_id)
    if not dropped_basenames:
        return transforms

    frames = transforms.get("frames")
    assert isinstance(frames, list), f"Scene {scene_id} transforms must contain a frames list"

    frame_basenames = [os.path.basename(frame["file_path"]) for frame in frames]
    present = dropped_basenames & set(frame_basenames)
    if not present:
        return transforms
    if present != dropped_basenames:
        missing = sorted(dropped_basenames - present)
        raise ValueError(f"Scene {scene_id} has a partial DiFix frame-remap match; missing {missing}")

    filtered_frames = [
        frame for frame, basename in zip(frames, frame_basenames, strict=True) if basename not in dropped_basenames
    ]
    return {**transforms, "frames": filtered_frames}
