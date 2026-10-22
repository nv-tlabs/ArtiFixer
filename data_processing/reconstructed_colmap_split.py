# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared schema for prepared reconstructed_colmap split files.

A split file maps {split: {scene_id: entry}} where each entry stores paths
relative to the split file plus camera_scale/has_gt metadata. Written by
prepare_colmap_artifixer_inputs.py and artifixer3d.py; read by artifixer3d.py
and model_eval.datasets.reconstructed_colmap_eval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from data_processing.camera_trajectories import write_json


def read_split_scenes(split_path: Path, split: str = "test") -> dict[str, dict[str, object]]:
    """Read a split file and validate the scene-id -> metadata mapping shape."""
    split_data = json.loads(split_path.read_text())[split]
    assert isinstance(split_data, dict), f"{split_path} {split} split must be a mapping from scene id to metadata"
    for scene_id, metadata in split_data.items():
        assert isinstance(metadata, dict), f"Scene {scene_id!r} metadata must be a mapping"
    return split_data


def resolve_split_path(split_root: Path, value: object, field: str) -> Path:
    """Resolve a metadata path against the split file's directory; absolute paths pass through."""
    assert isinstance(value, str), f"Split field {field!r} must be a string path, got {value!r}"
    path = Path(value)
    return path if path.is_absolute() else split_root / path


def resolve_required_path(split_root: Path, metadata: dict[str, object], field: str) -> Path:
    """Read a required path field from split metadata; a missing field is an error."""
    assert field in metadata, f"Prepared scene is missing required field {field!r}"
    return resolve_split_path(split_root, metadata[field], field)


def resolve_optional_path(split_root: Path, metadata: dict[str, object], field: str) -> Path | None:
    return resolve_split_path(split_root, metadata[field], field) if field in metadata else None


def resolve_required_number(metadata: dict[str, object], field: str) -> float:
    """Read a required numeric field from split metadata; a missing or non-numeric field is an error."""
    assert field in metadata, f"Prepared scene is missing required field {field!r}"
    value = metadata[field]
    assert isinstance(value, int | float) and not isinstance(
        value, bool
    ), f"Prepared scene field {field!r} must be a number, got {value!r}"
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Split field {field!r} must be finite and positive, got {value!r}")
    return value


def resolve_has_gt(scene_id: str, metadata: dict[str, object]) -> bool:
    has_gt = metadata.get("has_gt", True)
    assert isinstance(has_gt, bool), f"Scene {scene_id!r} has non-boolean has_gt metadata: {has_gt!r}"
    return has_gt


def path_for_split(split_path: Path, path: Path) -> str:
    """Return the path relative to the split file when possible, else absolute.

    Relative paths keep the split usable after the scene directory is moved.
    """
    split_root = split_path.parent.resolve()
    path = path.resolve()
    try:
        return path.relative_to(split_root).as_posix()
    except ValueError:
        return str(path)


def split_entry(
    split_path: Path,
    scene_id: str,
    *,
    transforms_path: Path,
    image_root: Path,
    render_dir: Path,
    opacity_dir: Path,
    selected_indices_path: Path,
    prompt_path: Path,
    reconstruction_checkpoint: Path,
    camera_scale: float,
    target_indices_path: Path | None = None,
    has_gt: bool = True,
) -> dict[str, object]:
    """Build one scene entry with all paths stored relative to the split file."""
    entry: dict[str, object] = {
        "scene_id": scene_id,
        "transforms_path": path_for_split(split_path, transforms_path),
        "image_root": path_for_split(split_path, image_root),
        "render_dir": path_for_split(split_path, render_dir),
        "opacity_dir": path_for_split(split_path, opacity_dir),
        "selected_indices_path": path_for_split(split_path, selected_indices_path),
        "prompt_path": path_for_split(split_path, prompt_path),
        "reconstruction_checkpoint": path_for_split(split_path, reconstruction_checkpoint),
        "camera_scale": camera_scale,
    }
    if target_indices_path is not None:
        entry["target_indices_path"] = path_for_split(split_path, target_indices_path)
    if not has_gt:
        entry["has_gt"] = False
    return entry


def write_split(split_path: Path, scene_id: str, entry: dict[str, object]) -> None:
    write_json(split_path, {"test": {scene_id: entry}})
