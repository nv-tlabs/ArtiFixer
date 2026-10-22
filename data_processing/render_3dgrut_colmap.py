#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a 3DGUT COLMAP checkpoint into the ArtiFixer reconstruction layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from threedgrut.render import Renderer

from data_processing.camera_trajectories import write_json
from model_training.utils.video_io import atomic_output_path

RENDER_FRAME_DIRS = ("renders", "opacity")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--selected_indices", type=Path, required=True)
    parser.add_argument("--num_selected_indices", type=int, default=None)
    parser.add_argument("--downsample_factor", type=float, default=None)
    parser.add_argument("--render_dataset_dir", type=Path, default=None)
    parser.add_argument(
        "--trajectory_output_subdir",
        default=None,
        help="Output subdirectory for trajectory frames; defaults to the trajectory filename stem. Use '' for the checkpoint root.",
    )
    parser.add_argument(
        "--trajectory_path",
        type=Path,
        default=None,
        help="Optional transforms-style JSON camera trajectory to render instead of the dataset test loader.",
    )
    return parser


def render_output_dir(output_root: Path, experiment_name: str, colmap_dir: Path, global_step: int, subdir: str) -> Path:
    output_dir = output_root / experiment_name / colmap_dir.name / f"ours_{global_step}"
    return output_dir / subdir if subdir else output_dir


def selected_indices_for_render(selected_indices: Path, num_selected_indices: int | None) -> list[int]:
    indices = json.loads(selected_indices.read_text())
    assert isinstance(indices, list), f"{selected_indices} must contain a JSON list"
    if num_selected_indices is not None:
        assert len(indices) >= num_selected_indices, (
            f"{selected_indices} has {len(indices)} entries, "
            f"fewer than requested num_selected_indices={num_selected_indices}"
        )
        indices = indices[:num_selected_indices]
    assert all(
        isinstance(index, int) and not isinstance(index, bool) for index in indices
    ), f"{selected_indices} must contain integer frame indices"
    return indices


def write_selected_indices(output_dir: Path, selected_indices: Path, num_selected_indices: int | None) -> None:
    indices = selected_indices_for_render(selected_indices, num_selected_indices)
    with atomic_output_path(output_dir / "selected_indices.json") as temporary:
        write_json(temporary, indices)


def render_frames_complete(output_dir: Path, frame_count: int) -> bool:
    try:
        for name in RENDER_FRAME_DIRS:
            for index in range(frame_count):
                with Image.open(output_dir / name / f"{index:05d}.png") as image:
                    if image.format != "PNG":
                        return False
                    image.verify()
    except (OSError, SyntaxError, ValueError):
        return False
    return frame_count > 0


def render_outputs_complete(
    output_dir: Path,
    frame_count: int,
    *,
    expected_selected_indices: list[int] | None = None,
) -> bool:
    selected_indices_path = output_dir / "selected_indices.json"
    if not selected_indices_path.is_file():
        return False
    try:
        existing_indices = json.loads(selected_indices_path.read_text())
    except (OSError, ValueError):
        return False
    if expected_selected_indices is not None and existing_indices != expected_selected_indices:
        return False
    return render_frames_complete(output_dir, frame_count)


def reset_render_frame_dirs(output_dir: Path) -> None:
    (output_dir / "selected_indices.json").unlink(missing_ok=True)
    for name in ("renders", "opacity", "depth"):
        path = output_dir / name
        if path.exists():
            assert path.is_dir() and not path.is_symlink(), f"Expected render output directory, got {path}"
            shutil.rmtree(path)


def render_3dgrut_colmap(
    *,
    checkpoint: Path,
    colmap_dir: Path,
    output_root: Path,
    experiment_name: str,
    selected_indices: Path,
    num_selected_indices: int | None = None,
    downsample_factor: float | None = None,
    render_dataset_dir: Path | None = None,
    trajectory_path: Path | None = None,
    trajectory_output_subdir: str | None = None,
) -> Path:
    dataset_dir = render_dataset_dir or colmap_dir
    config_overrides = {
        "path": str(colmap_dir),
        "experiment_name": experiment_name,
    }
    if trajectory_path is None:
        config_overrides["selected_indices_file"] = str(selected_indices)
    else:
        config_overrides.update(
            {
                "selected_indices_file": None,
                "train_test_split_file": None,
                "image_path_override": None,
            }
        )
    if downsample_factor is not None:
        config_overrides["dataset.downsample_factor"] = downsample_factor
    if num_selected_indices is not None:
        config_overrides["num_selected_indices"] = num_selected_indices

    renderer = Renderer.from_checkpoint(
        checkpoint_path=checkpoint,
        path=str(dataset_dir),
        out_dir=str(output_root),
        save_gt=False,
        computes_extra_metrics=True,
        config_overrides=config_overrides,
    )
    renderer.writer = None

    if trajectory_path is None:
        trajectory_subdir = ""
    else:
        trajectory_subdir = trajectory_path.stem if trajectory_output_subdir is None else trajectory_output_subdir
    output_checkpoint_dir = render_output_dir(
        output_root,
        experiment_name,
        colmap_dir,
        int(renderer.global_step),
        trajectory_subdir,
    )
    reset_render_frame_dirs(output_checkpoint_dir)

    if trajectory_path is None:
        renderer.render_all()
    else:
        renderer.render_from_file(trajectory_path, output_subdir=trajectory_subdir)

    write_selected_indices(output_checkpoint_dir, selected_indices, num_selected_indices)
    return output_checkpoint_dir


def main() -> None:
    args = build_parser().parse_args()
    render_3dgrut_colmap(
        checkpoint=args.checkpoint,
        colmap_dir=args.colmap_dir,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        selected_indices=args.selected_indices,
        num_selected_indices=args.num_selected_indices,
        downsample_factor=args.downsample_factor,
        render_dataset_dir=args.render_dataset_dir,
        trajectory_path=args.trajectory_path,
        trajectory_output_subdir=args.trajectory_output_subdir,
    )


if __name__ == "__main__":
    main()
