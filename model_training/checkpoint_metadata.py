# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from model_training.model_profiles import CAMERA_POSE_ALIGNMENTS, is_wan21_t2v


def checkpoint_metadata_path(checkpoint: Path | str) -> Path:
    """Checkpoint metadata lives beside .pt files or inside DCP directories."""
    checkpoint = Path(checkpoint)
    return (
        checkpoint.with_suffix(".metadata.json")
        if checkpoint.suffix.lower() == ".pt"
        else checkpoint / "artifixer.metadata.json"
    )


def require_finished_export(checkpoint: Path | str) -> None:
    checkpoint = Path(checkpoint)
    marker = checkpoint.with_name(checkpoint.name + ".incomplete")
    if checkpoint.suffix.lower() == ".pt" and marker.exists():
        raise ValueError(f"Incomplete checkpoint export at {checkpoint}; rerun the export with --overwrite")


def write_checkpoint_metadata(checkpoint: Path | str, metadata: dict) -> None:
    """Keep checkpoint metadata with the independently loadable weights."""
    path = checkpoint_metadata_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def read_checkpoint_metadata(checkpoint: Path | str) -> dict | None:
    require_finished_export(checkpoint)
    metadata_path = checkpoint_metadata_path(checkpoint)
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    return metadata


def resolve_camera_pose_alignment(
    checkpoint: Path | str,
    model_id: str,
    *,
    camera_pose_alignment: str | None = None,
) -> str:
    """Resolve saved pose alignment and reject conflicting overrides.

    Wan2.1 weights without a record use "mean"; other models require a record.
    """
    metadata = read_checkpoint_metadata(checkpoint) or {}
    if "camera_pose_alignment" not in metadata:
        if not is_wan21_t2v(model_id):
            raise ValueError(f"{checkpoint} has no camera_pose_alignment record; required for {model_id!r}.")
        if camera_pose_alignment not in (None, "mean"):
            raise ValueError(
                f"{checkpoint} has no camera_pose_alignment record; unrecorded Wan checkpoints use 'mean', "
                f"but {camera_pose_alignment!r} was requested."
            )
        return "mean"
    trained = metadata["camera_pose_alignment"]
    if trained not in CAMERA_POSE_ALIGNMENTS:
        raise ValueError(f"{checkpoint} records camera_pose_alignment={trained!r}; expected {CAMERA_POSE_ALIGNMENTS}")
    if camera_pose_alignment is not None and camera_pose_alignment != trained:
        raise ValueError(
            f"{checkpoint} uses camera_pose_alignment={trained!r}, but {camera_pose_alignment!r} was requested."
        )
    return trained
