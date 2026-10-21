# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from model_eval.dl3dv_difix_frame_remap import apply_difix_frame_remap

DIFIX_TRAIN_IDS_FILENAME = "difix_train_ids.json"

CAMERA_KEYS = (
    "camera_angle_x",
    "camera_angle_y",
    "fl_x",
    "fl_y",
    "cx",
    "cy",
    "w",
    "h",
    "k1",
    "k2",
    "k3",
    "k4",
    "p1",
    "p2",
)


@dataclass(frozen=True)
class PreparedSceneDataset:
    scene_dataset_path: Path
    selected_indices_file: Path
    num_selected_indices: int


def _uses_difix_frame_space(selected_indices_file: str | Path | None) -> bool:
    """DiFix train ids index the compact DiFix frame space, not DL3DV-ALL-960P."""
    return selected_indices_file is not None and Path(selected_indices_file).name == DIFIX_TRAIN_IDS_FILENAME


def _load_transforms(scene_dataset_path: Path, *, use_difix_frame_remap: bool = False) -> dict:
    transforms_path = scene_dataset_path / "transforms.json"
    with transforms_path.open() as f:
        transforms = json.load(f)

    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {transforms_path}")
    if use_difix_frame_remap:
        # The canonical prepared layout is <root>/SCENE_ID/transforms.json, so the
        # directory name keys the DiFix frame remap (a no-op for unknown scenes and
        # for transforms that are already in the compact DiFix frame space).
        transforms = apply_difix_frame_remap(transforms, scene_dataset_path.name)
    return transforms


def _load_frame_count(scene_dataset_path: Path, *, use_difix_frame_remap: bool = False) -> int:
    transforms = _load_transforms(scene_dataset_path, use_difix_frame_remap=use_difix_frame_remap)
    frames = transforms.get("frames", [])
    return len(frames)


def _scene_dataset_path(dataset_root: Path, scene_id: str) -> Path:
    scene_root = dataset_root / scene_id
    transforms_path = scene_root / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(
            "Missing prepared DL3DV scene dataset. Expected "
            f"{transforms_path}. Stage DL3DV distillation inputs into the canonical "
            "SCENE_ID/transforms.json layout before launching 3DGRUT."
        )
    return scene_root


def _frame_signature(transforms: dict, frame: dict) -> dict:
    signature = {"file_path": frame.get("file_path"), "transform_matrix": frame.get("transform_matrix")}
    for key in CAMERA_KEYS:
        if key in frame:
            signature[key] = frame[key]
        elif key in transforms:
            signature[key] = transforms[key]
    return signature


def _override_frame_indices(scene_image_override: Path) -> list[int]:
    indices = []
    for path in scene_image_override.glob("*.png"):
        if path.stem.endswith("_mask") or not path.stem.isdigit():
            continue
        indices.append(int(path.stem))

    if not indices:
        raise ValueError(f"No numeric PNG override frames found in {scene_image_override}")
    return sorted(indices)


def _load_selected_source_indices(selected_indices_file: Path, num_selected_indices: int) -> set[int]:
    return set(_load_selected_source_index_list(selected_indices_file, num_selected_indices))


def _load_selected_source_index_list(selected_indices_file: Path, num_selected_indices: int) -> list[int]:
    with selected_indices_file.open() as f:
        selected_indices = json.load(f)
    if num_selected_indices < 0:
        raise ValueError(f"num_selected_indices must be non-negative, got {num_selected_indices}")
    loaded = [int(idx) for idx in selected_indices[:num_selected_indices]]
    negative_indices = [idx for idx in loaded if idx < 0]
    if negative_indices:
        preview = ", ".join(str(idx) for idx in negative_indices[:10])
        raise ValueError(f"Selected source indices must be non-negative, got {preview}")
    return loaded


def _colmap_sparse_dirs(scene_dataset_path: Path) -> list[Path]:
    return [
        scene_dataset_path / "sparse" / "0",
        scene_dataset_path / "colmap" / "sparse" / "0",
    ]


def _find_colmap_file(scene_dataset_path: Path, *names: str) -> Path | None:
    for sparse_dir in _colmap_sparse_dirs(scene_dataset_path):
        for name in names:
            path = sparse_dir / name
            if path.is_file():
                return path
    return None


def _validate_3dgrut_assets(scene_dataset_path: Path) -> None:
    if _find_colmap_file(scene_dataset_path, "points3D.bin", "points3D.txt") is None:
        raise ValueError(
            f"Missing COLMAP points for 3DGRUT initialization under {scene_dataset_path}/sparse/0 "
            f"or {scene_dataset_path}/colmap/sparse/0"
        )


def _source_image_path(scene_dataset_path: Path, frame: dict, downsample_factor: int) -> Path | None:
    frame_path = frame.get("file_path")
    if frame_path is None:
        return None
    image_name = Path(frame_path).name
    images_folder = "images" if downsample_factor == 1 else f"images_{downsample_factor}"
    return scene_dataset_path / images_folder / image_name


def _symlink_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Make src absolute: a relative link target would be resolved against the
    # staging dir the link lives in, not the caller's working dir, and dangle.
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def prepare_3dgrut_scene_dataset(
    scene_dataset_path: str | Path,
    runtime_scene_dataset_path: str | Path,
    selected_indices_file: str | Path,
    num_selected_indices: int,
    *,
    downsample_factor: int = 4,
    allow_missing_source_images: bool = False,
) -> PreparedSceneDataset:
    scene_dataset_path = Path(scene_dataset_path).resolve()
    runtime_scene_dataset_path = Path(runtime_scene_dataset_path).resolve()
    if scene_dataset_path.is_relative_to(runtime_scene_dataset_path):
        raise ValueError("Runtime scene dataset must not be the source directory or an ancestor of it")
    selected_indices_file = Path(selected_indices_file)
    use_difix_frame_remap = _uses_difix_frame_space(selected_indices_file)
    transforms = _load_transforms(scene_dataset_path, use_difix_frame_remap=use_difix_frame_remap)
    frames = transforms.get("frames", [])
    selected_indices = _load_selected_source_index_list(selected_indices_file, num_selected_indices)
    _validate_3dgrut_assets(scene_dataset_path)

    filtered_indices = []
    dropped_indices = []
    for idx in selected_indices:
        if idx >= len(frames):
            raise ValueError(f"Selected source index {idx} is outside the scene frame table")
        source_image_path = _source_image_path(scene_dataset_path, frames[idx], downsample_factor)
        if source_image_path is not None and source_image_path.is_file():
            filtered_indices.append(idx)
        else:
            dropped_indices.append(idx)

    if not filtered_indices:
        raise ValueError(
            "No selected source frames have available source images for 3DGRUT training. "
            f"Dropped indices: {dropped_indices[:10]}"
        )
    if dropped_indices:
        preview = ", ".join(str(idx) for idx in dropped_indices[:10])
        message = (
            "Selected source frames are missing source images for 3DGRUT training: "
            f"{preview}. This would change the requested context protocol."
        )
        if not allow_missing_source_images:
            raise ValueError(message)
        print(f"warning: {message}", file=sys.stderr)

    if runtime_scene_dataset_path.exists():
        shutil.rmtree(runtime_scene_dataset_path)
    runtime_scene_dataset_path.mkdir(parents=True)
    if use_difix_frame_remap:
        # DiFix train ids and overrides index the compact DiFix frame space, so the
        # runtime dataset must train against the remapped frame table rather than
        # the staged DL3DV-ALL-960P transforms.json.
        (runtime_scene_dataset_path / "transforms.json").write_text(json.dumps(transforms))
    else:
        _symlink_path(scene_dataset_path / "transforms.json", runtime_scene_dataset_path / "transforms.json")

    images_folder = "images" if downsample_factor == 1 else f"images_{downsample_factor}"
    source_images_dir = scene_dataset_path / images_folder
    if source_images_dir.is_dir():
        _symlink_path(source_images_dir, runtime_scene_dataset_path / images_folder)

    runtime_sparse_dir = runtime_scene_dataset_path / "colmap" / "sparse" / "0"
    for name in ("cameras.bin", "cameras.txt", "points3D.bin", "points3D.txt"):
        src = _find_colmap_file(scene_dataset_path, name)
        if src is not None:
            _symlink_path(src, runtime_sparse_dir / name)

    runtime_selected_indices_file = runtime_scene_dataset_path / "selected_indices.json"
    runtime_selected_indices_file.write_text(json.dumps(filtered_indices))
    return PreparedSceneDataset(
        scene_dataset_path=runtime_scene_dataset_path,
        selected_indices_file=runtime_selected_indices_file,
        num_selected_indices=len(filtered_indices),
    )


def validate_override_frame_indices(
    scene_dataset_path: str | Path,
    scene_image_override: str | Path,
    reference_scene_dataset_path: str | Path | None = None,
    selected_indices_file: str | Path | None = None,
    num_selected_indices: int | None = None,
) -> None:
    scene_dataset_path = Path(scene_dataset_path)
    scene_image_override = Path(scene_image_override)
    use_difix_frame_remap = _uses_difix_frame_space(selected_indices_file)
    frame_count = _load_frame_count(scene_dataset_path, use_difix_frame_remap=use_difix_frame_remap)
    indices = _override_frame_indices(scene_image_override)
    index_set = set(indices)
    invalid_indices = [idx for idx in indices if idx < 0 or idx >= frame_count]
    if invalid_indices:
        max_valid = frame_count - 1
        preview = ", ".join(str(idx) for idx in invalid_indices[:10])
        raise ValueError(
            "Image override directory contains frame indices outside the scene frame table: "
            f"{preview}. Scene has {frame_count} frames, so the max valid index is {max_valid}. "
            "This usually means DATASET_PATH does not match the ArtiFixer render frame table."
        )

    if (selected_indices_file is None) != (num_selected_indices is None):
        raise ValueError("selected_indices_file and num_selected_indices must be provided together")
    if selected_indices_file is not None and num_selected_indices is not None:
        source_indices = _load_selected_source_indices(Path(selected_indices_file), num_selected_indices)
        invalid_source_indices = [idx for idx in sorted(source_indices) if idx < 0 or idx >= frame_count]
        if invalid_source_indices:
            preview = ", ".join(str(idx) for idx in invalid_source_indices[:10])
            raise ValueError(
                "Selected source indices are outside the scene frame table: "
                f"{preview}. Scene has {frame_count} frames."
            )
        required_override_indices = set(range(frame_count)) - source_indices
        missing_override_indices = sorted(required_override_indices - index_set)
        if missing_override_indices:
            preview = ", ".join(str(idx) for idx in missing_override_indices[:10])
            raise ValueError(
                "Image override directory is missing predictions for non-source training frames: "
                f"{preview}. For all-frame DL3DV ArtiFixer outputs, use video_frames/batch_0000/pred "
                "instead of frames/batch_0000/pred."
            )

    if reference_scene_dataset_path is None:
        return

    transforms = _load_transforms(scene_dataset_path, use_difix_frame_remap=use_difix_frame_remap)
    frames = transforms.get("frames", [])
    frame_paths = [frame["file_path"] for frame in frames]
    reference_scene_dataset_path = Path(reference_scene_dataset_path)
    reference_transforms = _load_transforms(reference_scene_dataset_path)
    reference_frames = reference_transforms.get("frames", [])
    reference_frame_paths = [frame["file_path"] for frame in reference_frames]
    invalid_reference_indices = [idx for idx in indices if idx >= len(reference_frame_paths)]
    if invalid_reference_indices:
        preview = ", ".join(str(idx) for idx in invalid_reference_indices[:10])
        raise ValueError(
            "Image override directory contains frame indices outside the reference frame table: "
            f"{preview}. Reference has {len(reference_frame_paths)} frames."
        )

    mismatched_indices = [
        idx
        for idx in indices
        if _frame_signature(transforms, frames[idx]) != _frame_signature(reference_transforms, reference_frames[idx])
    ]
    if mismatched_indices:
        idx = mismatched_indices[0]
        detail = (
            f"{frame_paths[idx]} != {reference_frame_paths[idx]}"
            if frame_paths[idx] != reference_frame_paths[idx]
            else "camera metadata differs"
        )
        raise ValueError(
            "Scene frame table has a different frame at override index "
            f"{idx}: {detail}. "
            "Use a DATASET_PATH with the same transforms.json ordering as the ArtiFixer renders."
        )


def select_matching_scene_dataset_path(
    dataset_root: str | Path,
    scene_id: str,
    scene_image_override: str | Path,
    *,
    selected_indices_file: str | Path | None = None,
    num_selected_indices: int | None = None,
    require_3dgrut_assets: bool = False,
) -> Path:
    """Validate and return the canonical prepared DL3DV scene root."""

    dataset_root = Path(dataset_root)
    scene_image_override = Path(scene_image_override)
    candidate = _scene_dataset_path(dataset_root, scene_id)
    validate_override_frame_indices(
        candidate,
        scene_image_override,
        selected_indices_file=selected_indices_file,
        num_selected_indices=num_selected_indices,
    )
    if require_3dgrut_assets:
        _validate_3dgrut_assets(candidate)
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DL3DV distillation override frame indices.")
    parser.add_argument("scene_dataset_path", type=Path)
    parser.add_argument("scene_image_override", type=Path, nargs="?")
    parser.add_argument(
        "--prepare_only", action="store_true", help="Prepare a reconstruction dataset without predicted frames."
    )
    parser.add_argument("--reference_scene_dataset_path", type=Path, default=None)
    parser.add_argument("--scene_id", default=None)
    parser.add_argument("--print_scene_dataset_path", action="store_true")
    parser.add_argument("--require_3dgrut_assets", action="store_true")
    parser.add_argument("--runtime_scene_dataset_path", type=Path, default=None)
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--allow_missing_source_images", action="store_true")
    parser.add_argument("--print_runtime_scene_info", action="store_true")
    parser.add_argument("--selected_indices_file", type=Path, default=None)
    parser.add_argument("--num_selected_indices", type=int, default=None)
    args = parser.parse_args(argv)
    if args.prepare_only:
        if args.runtime_scene_dataset_path is None or args.selected_indices_file is None:
            parser.error("--prepare_only requires --runtime_scene_dataset_path and --selected_indices_file")
        count = args.num_selected_indices
        if count is None:
            count = len(json.loads(args.selected_indices_file.read_text()))
        prepared = prepare_3dgrut_scene_dataset(
            args.scene_dataset_path,
            args.runtime_scene_dataset_path,
            args.selected_indices_file,
            count,
            downsample_factor=args.downsample_factor,
            allow_missing_source_images=args.allow_missing_source_images,
        )
        print(prepared.scene_dataset_path)
        return 0
    if args.scene_image_override is None:
        parser.error("scene_image_override is required unless --prepare_only is set")

    try:
        if args.scene_id is not None:
            selected_scene_dataset_path = select_matching_scene_dataset_path(
                args.scene_dataset_path,
                args.scene_id,
                args.scene_image_override,
                selected_indices_file=args.selected_indices_file,
                num_selected_indices=args.num_selected_indices,
                require_3dgrut_assets=args.require_3dgrut_assets,
            )
            if args.runtime_scene_dataset_path is not None:
                if args.selected_indices_file is None or args.num_selected_indices is None:
                    raise ValueError(
                        "runtime_scene_dataset_path requires selected_indices_file and num_selected_indices"
                    )
                prepared = prepare_3dgrut_scene_dataset(
                    selected_scene_dataset_path,
                    args.runtime_scene_dataset_path,
                    args.selected_indices_file,
                    args.num_selected_indices,
                    downsample_factor=args.downsample_factor,
                    allow_missing_source_images=args.allow_missing_source_images,
                )
                validate_override_frame_indices(
                    prepared.scene_dataset_path,
                    args.scene_image_override,
                    selected_indices_file=prepared.selected_indices_file,
                    num_selected_indices=prepared.num_selected_indices,
                )
                selected_scene_dataset_path = prepared.scene_dataset_path
                if args.print_runtime_scene_info:
                    print(prepared.scene_dataset_path)
                    print(prepared.selected_indices_file)
                    print(prepared.num_selected_indices)
            if args.print_scene_dataset_path:
                print(selected_scene_dataset_path)
        else:
            validate_override_frame_indices(
                args.scene_dataset_path,
                args.scene_image_override,
                reference_scene_dataset_path=args.reference_scene_dataset_path,
                selected_indices_file=args.selected_indices_file,
                num_selected_indices=args.num_selected_indices,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
