# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from data_processing.scene_utils import safe_extractall

RECONSTRUCTION_SUBDIR_PREFIX = "dl3dv_"


@dataclass(frozen=True)
class ReconstructionOutputs:
    data_h5: Path
    parsed_yaml: Path
    checkpoint: Path

    def complete(self) -> bool:
        return not self.incomplete_marker.exists() and all(
            path.is_file() and path.stat().st_size > 0 for path in self.paths()
        )

    @property
    def incomplete_marker(self) -> Path:
        return self.data_h5.with_name(self.data_h5.name + ".incomplete")

    def paths(self) -> tuple[Path, Path, Path]:
        return self.data_h5, self.parsed_yaml, self.checkpoint


def reconstruction_subdir_for_dl3dv_subdir(dl3dv_subdir: str) -> str:
    return f"{RECONSTRUCTION_SUBDIR_PREFIX}{dl3dv_subdir}"


def reconstruction_outputs(
    output_root: Path, reconstruction_subdir: str, scene_name: str, half: str, views: int | str
) -> ReconstructionOutputs:
    return reconstruction_outputs_for_scene_dir(
        output_root / reconstruction_subdir / scene_name, scene_name, half, views
    )


def reconstruction_outputs_for_scene_dir(
    final_dir: Path, scene_name: str, half: str, views: int | str
) -> ReconstructionOutputs:
    suffix = f"{scene_name}_{half}_{views}"
    return ReconstructionOutputs(
        data_h5=final_dir / f"data_{suffix}.h5",
        parsed_yaml=final_dir / f"parsed_{suffix}.yaml",
        checkpoint=final_dir / f"ckpt_last_{suffix}.pt",
    )


def image_dir_for_scene(scene_dir: Path, downsample_factor: float) -> Path:
    if downsample_factor == 1.0:
        return scene_dir / "images"
    factor = int(downsample_factor)
    assert float(factor) == float(
        downsample_factor
    ), f"downsample_factor must be an integer or 1.0, got {downsample_factor}"
    return scene_dir / f"images_{factor}"


def copy_reconstruction_outputs(src_dir: Path, outputs: ReconstructionOutputs) -> None:
    _transfer_reconstruction_outputs(src_dir, outputs, move=False)


def move_reconstruction_outputs(src_dir: Path, outputs: ReconstructionOutputs) -> None:
    _transfer_reconstruction_outputs(src_dir, outputs, move=True)


def _transfer_reconstruction_outputs(src_dir: Path, outputs: ReconstructionOutputs, *, move: bool) -> None:
    mapping = (
        (src_dir / "data.h5", outputs.data_h5),
        (src_dir / "parsed.yaml", outputs.parsed_yaml),
        (src_dir / "ckpt_last.pt", outputs.checkpoint),
    )
    for src, _ in mapping:
        if not src.exists():
            raise FileNotFoundError(src)

    outputs.incomplete_marker.parent.mkdir(parents=True, exist_ok=True)
    outputs.incomplete_marker.touch()
    for src, dst in mapping:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
        try:
            if move:
                shutil.move(str(src), str(tmp))
            else:
                shutil.copy2(src, tmp)
            tmp.replace(dst)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        print(f"Wrote {dst}")
    outputs.incomplete_marker.unlink()


def task_work_dir(scene_work_dir: Path, half: str, views: int | str) -> Path:
    return scene_work_dir / f"{half}@{views}"


def _zip_has_scene_root(zip_file: ZipFile, scene_name: str) -> bool:
    names = [name for name in zip_file.namelist() if name and not name.endswith("/")]
    return any(name == scene_name or name.startswith(f"{scene_name}/") for name in names)


def extract_scene_zip(scene_zip: Path, task_dir: Path, scene_name: str) -> Path:
    with ZipFile(scene_zip, "r") as zf:
        if _zip_has_scene_root(zf, scene_name):
            safe_extractall(zf, task_dir)
            return task_dir / scene_name

        scene_dir = task_dir / scene_name
        safe_extractall(zf, scene_dir)
        return scene_dir


def extract_colmap_zip(colmap_zip: Path, task_dir: Path, scene_dir: Path, scene_name: str) -> None:
    with ZipFile(colmap_zip, "r") as zf:
        safe_extractall(zf, task_dir if _zip_has_scene_root(zf, scene_name) else scene_dir)


def find_experiment_scene_dir(
    task_dir: Path,
    scene_name: str,
    *,
    expected_experiment_name: str | None = None,
    existing_dirs: set[Path] | None = None,
) -> Path:
    candidates: list[Path] = []
    if expected_experiment_name:
        expected_root = task_dir / expected_experiment_name
        expected = expected_root / scene_name
        if expected.is_dir():
            return expected
        if expected_root.is_dir():
            candidates.extend(path for path in expected_root.iterdir() if path.is_dir())

    existing_dirs = existing_dirs or set()
    for path in task_dir.iterdir():
        if not path.is_dir() or path.resolve() in existing_dirs:
            continue
        scene_dir = path / scene_name
        if scene_dir.is_dir():
            candidates.append(scene_dir)

    if not candidates:
        for path in task_dir.iterdir():
            if not path.is_dir() or path.name == scene_name:
                continue
            scene_dir = path / scene_name
            if scene_dir.is_dir():
                candidates.append(scene_dir)

    if not candidates:
        raise FileNotFoundError(f"No 3DGRUT experiment directory found under {task_dir}")

    unique_candidates = {candidate.resolve(): candidate for candidate in candidates}
    return max(unique_candidates.values(), key=lambda path: (path.parent.stat().st_mtime, path.parent.name))
