# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage DL3DV scene zips for 3DGRUT distillation jobs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from data_processing.scene_utils import read_scene_list, resolve_scene_path
from data_processing.sparse_recon.workflow import extract_colmap_zip, extract_scene_zip


def scene_items_from_split(
    split_data: dict,
    *,
    include_metric_scale: bool = False,
) -> list[tuple]:
    """Return ordered, deduplicated (scene_id, dl3dv_subdir) pairs from a DL3DV split.

    With ``include_metric_scale``, returns (scene_id, dl3dv_subdir,
    metric_scale) triples instead. A scene's entries (per-half
    reconstructions) share one subdir and agree on scale to reconstruction
    noise (measured max spread 9e-4 across the production split); the first
    entry is used. Genuinely disagreeing entries mean a corrupt split and
    raise.
    """
    assert isinstance(split_data, dict), "DL3DV split must map scene_id -> reconstruction entries"

    scene_items: list[tuple] = []
    for scene_id, reconstructions in split_data.items():
        if not reconstructions:
            continue
        subdirs = {reconstruction["subdir"] for reconstruction in reconstructions}
        if include_metric_scale and any("scale" not in r for r in reconstructions):
            # Defaulting a missing scale would silently change what the eval
            # conditions on (camera_scale = scale x dataset scaling factor).
            raise ValueError(
                f"Scene {scene_id!r} split entry has no 'scale' (the reconstruction's metric "
                "scale), which this eval conditions on. Regenerate the split with the prepare "
                "tooling, or add the scale to the entry."
            )
        scales = (
            [float(reconstruction["scale"]) for reconstruction in reconstructions] if include_metric_scale else None
        )
        # 1% = 10x the worst scale spread observed across all 10,421 scenes
        # of the production split; anything larger is corruption, not noise.
        scales_consistent = scales is None or all(
            math.isclose(scale, scales[0], rel_tol=1e-2, abs_tol=1e-8) for scale in scales[1:]
        )
        if len(subdirs) != 1 or not scales_consistent:
            raise ValueError(
                f"Scene {scene_id!r} has inconsistent reconstruction entries "
                f"(subdirs={sorted(subdirs)}, scales={scales}); the split is corrupt — "
                "regenerate it with the prepare tooling"
            )

        subdir = reconstructions[0]["subdir"]
        if not include_metric_scale:
            scene_items.append((scene_id, subdir))
            continue
        scale = scales[0]
        # The winning entry's scale feeds camera_scale regardless of mode.
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Scene {scene_id!r} has an invalid metric scale: {scale}")
        scene_items.append((scene_id, subdir, scale))
    return scene_items


def has_transforms(scene_dir: Path) -> bool:
    return (scene_dir / "transforms.json").is_file()


def has_normalizable_transforms(scene_dir: Path) -> bool:
    return has_transforms(scene_dir) or (scene_dir / "nerfstudio" / "transforms.json").is_file()


def normalize_transforms(scene_dir: Path) -> None:
    root_transforms = scene_dir / "transforms.json"
    if root_transforms.is_file():
        return
    nerfstudio_transforms = scene_dir / "nerfstudio" / "transforms.json"
    if not nerfstudio_transforms.is_file():
        raise FileNotFoundError(
            f"Missing transforms.json for staged DL3DV scene. Expected {root_transforms} "
            f"or {nerfstudio_transforms} before normalization."
        )
    shutil.copy2(nerfstudio_transforms, root_transforms)


def has_colmap(scene_dir: Path) -> bool:
    for sparse_dir in (scene_dir / "colmap" / "sparse" / "0", scene_dir / "sparse" / "0"):
        if all((sparse_dir / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
            return True
    return False


def stage_colmap_source(colmap_source: Path, output_root: Path, scene_dir: Path, scene_id: str) -> None:
    if colmap_source.is_file():
        extract_colmap_zip(colmap_source, output_root, scene_dir, scene_id)
        return

    if not colmap_source.is_dir():
        raise FileNotFoundError(f"COLMAP source does not exist: {colmap_source}")

    copied = False
    for subdir_name in ("colmap", "sparse"):
        source_subdir = colmap_source / subdir_name
        if not source_subdir.exists():
            continue
        shutil.copytree(source_subdir, scene_dir / subdir_name, dirs_exist_ok=True)
        copied = True

    if not copied:
        raise FileNotFoundError(f"COLMAP source has no colmap/ or sparse/ directory: {colmap_source}")


def stage_dl3dv_distill_dataset(
    dl3dv_dir: Path,
    split_path: Path,
    scene_list_file: Path,
    output_root: Path,
    *,
    colmap_zip_root: Path | None = None,
    split: str = "test",
    overwrite: bool = False,
) -> list[Path]:
    split_json = json.loads(split_path.read_text())
    if split not in split_json:
        raise ValueError(f"{split_path} does not contain split {split!r}")

    scene_ids = read_scene_list(scene_list_file)
    scene_subdirs = dict(scene_items_from_split(split_json[split]))
    missing_scenes = [scene_id for scene_id in scene_ids if scene_id not in scene_subdirs]
    if missing_scenes:
        raise ValueError(f"Scene list contains scenes missing from {split_path}: {', '.join(missing_scenes)}")

    scene_zips = {scene_id: dl3dv_dir / scene_subdirs[scene_id] / f"{scene_id}.zip" for scene_id in scene_ids}
    missing_zips = [path for path in scene_zips.values() if not path.is_file()]
    if missing_zips:
        raise FileNotFoundError(f"Missing DL3DV scene zip: {missing_zips[0]}")
    colmap_zips: dict[str, Path] = {}
    if colmap_zip_root is not None:
        colmap_zips = {scene_id: resolve_scene_path(colmap_zip_root, scene_id) for scene_id in scene_ids}

    output_root.mkdir(parents=True, exist_ok=True)
    staged_paths = []
    for scene_id in scene_ids:
        scene_dir = output_root / scene_id
        if scene_dir.exists():
            if overwrite:
                shutil.rmtree(scene_dir)
            elif has_normalizable_transforms(scene_dir) and has_colmap(scene_dir):
                normalize_transforms(scene_dir)
                print(f"Skipping staged scene {scene_id}: {scene_dir}")
                staged_paths.append(scene_dir)
                continue
            else:
                raise FileExistsError(
                    f"{scene_dir} exists but does not contain transforms.json plus COLMAP sparse files"
                )

        extracted_scene_dir = extract_scene_zip(scene_zips[scene_id], output_root, scene_id)
        if colmap_zip_root is not None:
            stage_colmap_source(colmap_zips[scene_id], output_root, extracted_scene_dir, scene_id)
        normalize_transforms(extracted_scene_dir)
        if not has_transforms(extracted_scene_dir):
            raise FileNotFoundError(
                f"Extracted {scene_zips[scene_id]} to {extracted_scene_dir}, but no transforms.json was found"
            )
        if not has_colmap(extracted_scene_dir):
            raise FileNotFoundError(
                f"Extracted {scene_zips[scene_id]} to {extracted_scene_dir}, but no COLMAP sparse files were found"
            )
        print(f"Staged scene {scene_id}: {extracted_scene_dir}")
        staged_paths.append(extracted_scene_dir)
    return staged_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dl3dv_dir", type=Path, required=True)
    parser.add_argument("--split_path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scene_list_file", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--colmap_zip_root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    staged_paths = stage_dl3dv_distill_dataset(
        dl3dv_dir=args.dl3dv_dir,
        split_path=args.split_path,
        split=args.split,
        scene_list_file=args.scene_list_file,
        output_root=args.output_root,
        colmap_zip_root=args.colmap_zip_root,
        overwrite=args.overwrite,
    )
    print(f"Staged {len(staged_paths)} DL3DV scenes under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
