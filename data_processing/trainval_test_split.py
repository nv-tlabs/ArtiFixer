# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import h5py
import numpy as np
from tqdm import tqdm

from data_processing.scene_utils import (
    downsampled_image_path,
    load_source_scene_transforms,
    read_scene_list,
    scene_zip_member,
)
from data_processing.sparse_recon.workflow import RECONSTRUCTION_SUBDIR_PREFIX
from model_training.data.reconstruction_path import parse_reconstruction_hdf5_path
from model_training.utils.video_io import decode_video_frame_count

BAD_SCENE_IDS: frozenset[str] = frozenset(Path(__file__).with_name("bad_scene_ids.txt").read_text().split())
DATASET_SCALING_FACTOR = 0.01
MAX_CAMERA_TRANSFORM_ABS = 10.0


def notify(message: str) -> None:
    print(message, flush=True)


def dl3dv_subdir_from_reconstruction_subdir(subdir: str, prefix: str) -> str:
    if not is_reconstruction_subdir(subdir, prefix):
        raise ValueError(f"Unexpected reconstruction subdir {subdir!r}; expected prefix {prefix!r}")
    return subdir.removeprefix(prefix)


def dl3dv_zip_path(dl3dv_dir: Path, scene_id: str, subdir: str, prefix: str) -> Path:
    return dl3dv_dir / dl3dv_subdir_from_reconstruction_subdir(subdir, prefix) / f"{scene_id}.zip"


def is_reconstruction_subdir(subdir: str, prefix: str) -> bool:
    return subdir.startswith(prefix)


def invert_se3(transforms: np.ndarray) -> np.ndarray:
    assert transforms.shape[-2:] == (4, 4)
    rinv = np.swapaxes(transforms[..., :3, :3], -1, -2)
    out = np.zeros_like(transforms)
    out[..., :3, :3] = rinv
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def load_source_transforms(dl3dv_dir: Path, scene_id: str, subdir: str, prefix: str) -> tuple[dict, str]:
    return load_source_scene_transforms(dl3dv_dir, dl3dv_subdir_from_reconstruction_subdir(subdir, prefix), scene_id)


def validate_scene_geometry(scene_id: str, transforms: dict, zip_path: str, recons: list[dict]) -> dict | None:
    scales = [float(recon["scale"]) * DATASET_SCALING_FACTOR for recon in recons]
    if not scales:
        return {"scene_id": scene_id, "reason": "no_scale"}
    invalid_scales = [scale for scale in scales if not (math.isfinite(scale) and scale > 0)]
    if invalid_scales:
        return {"scene_id": scene_id, "reason": "invalid_scale", "scales": scales}

    c2ws_world = np.asarray([frame["transform_matrix"] for frame in transforms["frames"]], dtype=np.float64)
    if c2ws_world.ndim != 3 or c2ws_world.shape[1:] != (4, 4):
        return {
            "scene_id": scene_id,
            "reason": "invalid_transform_shape",
            "shape": list(c2ws_world.shape),
            "zip_path": zip_path,
        }
    if not np.isfinite(c2ws_world).all():
        return {
            "scene_id": scene_id,
            "reason": "non_finite_transform",
            "zip_path": zip_path,
        }
    scale = max(scales)
    c2ws_world[:, :3, 3] *= scale
    w2cs_world = invert_se3(c2ws_world)
    max_abs = float(np.max(np.abs(np.concatenate([w2cs_world, c2ws_world], axis=0))))
    if not math.isfinite(max_abs):
        return {"scene_id": scene_id, "reason": "non_finite_camera_transform", "zip_path": zip_path}
    if max_abs > MAX_CAMERA_TRANSFORM_ABS:
        return {
            "scene_id": scene_id,
            "reason": "camera_transform_too_large",
            "max_abs": max_abs,
            "scale": scale,
            "zip_path": zip_path,
        }
    return None


def raise_for_geometry_bad_entries(geometry_bad_entries: list[dict]) -> None:
    if not geometry_bad_entries:
        return
    lines = [
        f"Found {len(geometry_bad_entries)} scene(s) with invalid scaled camera geometry. The split JSON is "
        f"only valid if max(abs(c2w/w2c)) <= {MAX_CAMERA_TRANSFORM_ABS} after applying "
        f"scale * {DATASET_SCALING_FACTOR}. Fix scale/source poses or add the scene id to "
        "data_processing/bad_scene_ids.txt before regenerating."
    ]
    for entry in sorted(geometry_bad_entries, key=lambda e: e["scene_id"]):
        detail = " ".join(f"{k}={v}" for k, v in entry.items() if k != "scene_id")
        lines.append(f"    scene_id={entry['scene_id']} {detail}".rstrip())
    raise ValueError("\n".join(lines))


def validate_reconstruction_payload(
    f: h5py.File,
    expected_selected_indices: int,
    decode_reconstruction_videos: bool,
) -> tuple[dict | None, int | None, np.ndarray | None]:
    """Return the first payload issue that would make the training loader fail or silently corrupt data."""
    if "selected_indices" not in f:
        return {"reason": "selected_indices_missing"}, None, None

    selected_indices = f["selected_indices"][:]
    if selected_indices.ndim != 1:
        return {"reason": "selected_indices_invalid_shape", "shape": list(selected_indices.shape)}, None, None
    if len(selected_indices) != expected_selected_indices:
        return (
            {
                "reason": "selected_indices_count_mismatch",
                "expected": expected_selected_indices,
                "actual": len(selected_indices),
            },
            None,
            None,
        )
    if not np.issubdtype(selected_indices.dtype, np.integer):
        return {"reason": "selected_indices_not_integer", "dtype": str(selected_indices.dtype)}, None, None
    if len(np.unique(selected_indices)) != len(selected_indices):
        return {"reason": "selected_indices_not_unique"}, None, None

    frame_count = None
    for dataset_name in ("render_video", "opacity_video"):
        if dataset_name not in f:
            return {"reason": f"{dataset_name}_missing"}, None, None
        if len(f[dataset_name]) == 0:
            return {"reason": f"{dataset_name}_empty"}, None, None
        if not decode_reconstruction_videos:
            continue
        try:
            dataset_frame_count = decode_video_frame_count(f[dataset_name][:].tobytes())
        except Exception:
            return {"reason": f"{dataset_name}_decode_error"}, None, None
        if dataset_frame_count == 0:
            return {"reason": f"{dataset_name}_empty"}, None, None
        if frame_count is None:
            frame_count = dataset_frame_count
        elif dataset_frame_count != frame_count:
            return (
                {
                    "reason": "video_frame_count_mismatch",
                    "render_or_opacity_frame_count": dataset_frame_count,
                    "first_video_frame_count": frame_count,
                },
                None,
                None,
            )

    return None, frame_count, selected_indices


def validate_source_scene(
    dl3dv_dir: Path, scene_id: str, subdir: str, reconstruction_subdir_prefix: str
) -> tuple[dict | None, dict | None]:
    """Return source-data issue and loaded transforms if the DL3DV zip is loader-compatible."""
    source_subdir = dl3dv_subdir_from_reconstruction_subdir(subdir, reconstruction_subdir_prefix)
    zip_path = dl3dv_zip_path(dl3dv_dir, scene_id, subdir, reconstruction_subdir_prefix)
    context = {"scene_id": scene_id, "subdir": source_subdir, "zip_path": str(zip_path)}

    try:
        transforms, prefix = load_source_transforms(dl3dv_dir, scene_id, subdir, reconstruction_subdir_prefix)
        for key in ("cx", "cy", "fl_x", "fl_y", "w", "h"):
            if key not in transforms:
                return {**context, "reason": "intrinsics_missing", "member": key}, None

        try:
            frames = transforms["frames"]
        except KeyError:
            return {**context, "reason": "frames_missing"}, None
        if not isinstance(frames, list) or not frames:
            return {**context, "reason": "frames_missing_or_empty"}, None

        with ZipFile(zip_path, "r") as zf:
            for frame in frames:
                if not isinstance(frame, dict):
                    return {**context, "reason": "frame_invalid"}, None
                try:
                    file_path = frame["file_path"]
                except KeyError:
                    return {**context, "reason": "frame_file_path_missing"}, None
                if not isinstance(file_path, str):
                    return {**context, "reason": "frame_file_path_invalid"}, None
                try:
                    matrix = frame["transform_matrix"]
                except KeyError:
                    return {**context, "reason": "frame_transform_missing", "member": file_path}, None
                if (
                    not isinstance(matrix, list)
                    or len(matrix) != 4
                    or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
                ):
                    return {**context, "reason": "frame_transform_invalid", "member": file_path}, None
                image_member = scene_zip_member(prefix.rstrip("/"), downsampled_image_path(file_path, 4))
                try:
                    zf.getinfo(image_member)
                except KeyError:
                    return {**context, "reason": "frame_image_missing", "member": image_member}, None
    except FileNotFoundError as e:
        reason = "zip_missing" if not zip_path.is_file() else "transforms_missing"
        return {**context, "reason": reason, "error": str(e)}, None
    except BadZipFile as e:
        return {**context, "reason": "bad_zip", "error": repr(e)}, None
    except Exception as e:
        return {**context, "reason": "zip_read_error", "error": repr(e)}, None

    return None, transforms


def process_scene(
    data_path: str,
    scene_id: str,
    subdir: str,
    dl3dv_dir: Path,
    required_splits: set[int],
    decode_reconstruction_videos: bool,
    reconstruction_subdir_prefix: str,
) -> tuple[str, list[dict], list[dict], dict | None, dict | None, tuple[int, int] | None]:
    """Scan one scene's h5 reconstructions. Missing-scale, zero-scale,
    unreasonably-large-scale, and unreadable files are NOT silently dropped —
    BAD_SCENE_IDS was filtered out before this point, so any file with those
    issues is a not-yet-quarantined scene and needs triage. We collect offending
    files as ``bad_entries`` and the caller raises with the full list after
    scanning every scene, so a single regen surfaces every problem at once.

    Returns ``(scene_id, per_scene_data, bad_entries, source_issue, geometry_issue,
    source_resolution)`` where ``source_resolution`` is the scene's transforms
    ``(w, h)`` when the source zip was validated, else None."""
    source_issue = None
    geometry_issue = None
    source_resolution = None

    scene_path = os.path.join(data_path, subdir, scene_id)
    by_key: dict[tuple[str, int], dict] = {}
    frame_counts: dict[str, int] = {}
    selected_indices_by_path: dict[str, np.ndarray] = {}
    bad_entries: list[dict] = []
    for hdf5_path in sorted(glob.glob(os.path.join(scene_path, "*.h5"))):
        try:
            parsed_scene_id, scene_half, num_selected_indices = parse_reconstruction_hdf5_path(hdf5_path)
        except ValueError as e:
            bad_entries.append(
                {
                    "hdf5_path": hdf5_path,
                    "scene_id": scene_id,
                    "scene_half": None,
                    "num_selected_indices": None,
                    "reason": "filename_parse_error",
                    "error": str(e),
                    "blocks_split": True,
                }
            )
            continue
        if parsed_scene_id != scene_id:
            bad_entries.append(
                {
                    "hdf5_path": hdf5_path,
                    "scene_id": scene_id,
                    "scene_half": scene_half,
                    "num_selected_indices": num_selected_indices,
                    "reason": "filename_scene_id_mismatch",
                    "parsed_scene_id": parsed_scene_id,
                    "blocks_split": True,
                }
            )
            continue

        context = {
            "hdf5_path": hdf5_path,
            "scene_id": scene_id,
            "scene_half": scene_half,
            "num_selected_indices": num_selected_indices,
        }
        try:
            with h5py.File(hdf5_path, "r") as f:
                if "scale" not in f:
                    bad_entries.append({**context, "reason": "scale_missing"})
                    continue
                scale = float(f["scale"][()])
                if not math.isfinite(scale):
                    bad_entries.append({**context, "reason": "scale_non_finite", "scale": scale})
                    continue
                if scale == 0.0:
                    bad_entries.append({**context, "reason": "scale_zero"})
                    continue
                if scale < 0:
                    bad_entries.append({**context, "reason": "scale_negative", "scale": scale})
                    continue
                if scale > 100:
                    bad_entries.append({**context, "reason": "scale_too_large", "scale": scale})
                    continue
                frame_count = None
                if num_selected_indices in required_splits:
                    payload_issue, frame_count, selected_indices = validate_reconstruction_payload(
                        f,
                        num_selected_indices,
                        decode_reconstruction_videos,
                    )
                    if payload_issue is not None:
                        bad_entries.append({**context, **payload_issue})
                        continue
                    selected_indices_by_path[hdf5_path] = selected_indices
        except Exception as e:
            bad_entries.append({**context, "reason": "read_error", "error": repr(e)})
            continue

        candidate = {
            "scene_half": scene_half,
            "num_selected_indices": num_selected_indices,
            "subdir": dl3dv_subdir_from_reconstruction_subdir(subdir, reconstruction_subdir_prefix),
            "scale": scale,
            "hdf5_path": hdf5_path,
        }
        if frame_count is not None:
            frame_counts[hdf5_path] = frame_count
        key = (scene_half, num_selected_indices)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = candidate
        else:
            bad_entries.append(
                {
                    **context,
                    "reason": "duplicate_reconstruction",
                    "existing_hdf5_path": existing["hdf5_path"],
                    "duplicate_hdf5_path": candidate["hdf5_path"],
                }
            )

    if any(recon["num_selected_indices"] in required_splits for recon in by_key.values()):
        source_issue, transforms = validate_source_scene(dl3dv_dir, scene_id, subdir, reconstruction_subdir_prefix)
        source_zip_path = str(dl3dv_zip_path(dl3dv_dir, scene_id, subdir, reconstruction_subdir_prefix))
        required_recons = [recon for recon in by_key.values() if recon["num_selected_indices"] in required_splits]
        if source_issue is None:
            source_resolution = (int(transforms["w"]), int(transforms["h"]))
            source_frame_count = len(transforms["frames"])
            for recon in required_recons:
                hdf5_path = recon["hdf5_path"]
                if decode_reconstruction_videos:
                    recon_frame_count = frame_counts[hdf5_path]
                    if recon_frame_count != source_frame_count:
                        bad_entries.append(
                            {
                                "hdf5_path": hdf5_path,
                                "scene_id": scene_id,
                                "scene_half": recon["scene_half"],
                                "num_selected_indices": recon["num_selected_indices"],
                                "reason": "reconstruction_frame_count_mismatch",
                                "reconstruction_frame_count": recon_frame_count,
                                "source_frame_count": source_frame_count,
                            }
                        )
                selected_indices = selected_indices_by_path[hdf5_path]
                if selected_indices.min() < 0 or selected_indices.max() >= source_frame_count:
                    bad_entries.append(
                        {
                            "hdf5_path": hdf5_path,
                            "scene_id": scene_id,
                            "scene_half": recon["scene_half"],
                            "num_selected_indices": recon["num_selected_indices"],
                            "reason": "selected_indices_out_of_range",
                            "min_index": int(selected_indices.min()),
                            "max_index": int(selected_indices.max()),
                            "source_frame_count": source_frame_count,
                        }
                    )
            try:
                geometry_issue = validate_scene_geometry(scene_id, transforms, source_zip_path, required_recons)
            except Exception as e:
                geometry_issue = {"scene_id": scene_id, "reason": "geometry_read_error", "error": repr(e)}

    return scene_id, list(by_key.values()), bad_entries, source_issue, geometry_issue, source_resolution


def raise_for_source_bad_entries(source_bad_entries: list[dict]) -> None:
    if not source_bad_entries:
        return

    by_reason: dict[str, list[dict]] = defaultdict(list)
    for entry in source_bad_entries:
        by_reason[entry["reason"]].append(entry)
    for entries in by_reason.values():
        entries.sort(key=lambda e: (e["scene_id"], e["subdir"]))

    lines = [
        f"Found {len(source_bad_entries)} scene(s) with DL3DV source-data issues. The split JSON is "
        "only valid if every included scene has a loadable source zip with transforms.json. Fix the "
        "source data or add the scene id to data_processing/bad_scene_ids.txt before regenerating."
    ]
    for reason in sorted(by_reason):
        entries = by_reason[reason]
        lines.append(f"\n  {reason} ({len(entries)} scene(s)):")
        for e in entries:
            detail = e.get("error") or e.get("member") or ""
            lines.append(f"    scene_id={e['scene_id']} subdir={e['subdir']} zip={e['zip_path']} {detail}".rstrip())
    raise ValueError("\n".join(lines))


def entry_blocks_split(entry: dict, required_splits: set[int]) -> bool:
    return entry.get("blocks_split", False) or entry.get("num_selected_indices") in required_splits


def format_entry_details(entry: dict) -> str:
    detail_keys = (
        "scale",
        "member",
        "expected",
        "actual",
        "dtype",
        "shape",
        "reconstruction_frame_count",
        "source_frame_count",
        "min_index",
        "max_index",
        "parsed_scene_id",
        "existing_hdf5_path",
        "duplicate_hdf5_path",
        "error",
    )
    return " ".join(f"{key}={entry[key]}" for key in detail_keys if key in entry)


def load_test_scenes() -> set[str]:
    repo_root = Path(__file__).resolve().parents[1]
    test_scenes: set[str] = set()
    for name in ("dl3dv_benchmark_ours_test_scenes.txt", "dl3dv_benchmark_difix_test_scenes.txt"):
        test_scenes.update(read_scene_list(repo_root / "data_processing" / name))
    return test_scenes


def validate_split_membership(split_data: dict, test_scenes: set[str]) -> None:
    actual_test_scenes = set(split_data["test"])
    actual_trainval_scenes = set(split_data["trainval"])

    overlap = sorted(actual_test_scenes & actual_trainval_scenes)
    if overlap:
        raise ValueError(f"Scenes appeared in both test and trainval: {overlap}")

    missing_test = sorted(test_scenes - actual_test_scenes)
    unexpected_test = sorted(actual_test_scenes - test_scenes)
    if missing_test or unexpected_test:
        lines = [
            "Generated test split does not exactly match the benchmark scene lists.",
        ]
        if missing_test:
            lines.append(f"Missing benchmark test scenes: {missing_test}")
        if unexpected_test:
            lines.append(f"Unexpected test scenes: {unexpected_test}")
        raise ValueError("\n".join(lines))


def _resolve_max_workers(requested: int | None, num_scenes: int) -> int:
    if requested is not None:
        return max(1, min(requested, num_scenes))

    if "SLURM_CPUS_PER_TASK" in os.environ:
        slurm_cpus = os.environ["SLURM_CPUS_PER_TASK"]
        try:
            return max(1, min(int(slurm_cpus), num_scenes))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, num_scenes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Root directory of artifixer reconstructions")
    parser.add_argument("--output_root", type=str, required=True, help="Directory to write trainval_test_split.json")
    parser.add_argument("--dl3dv_dir", type=Path, required=True, help="Root directory of DL3DV source scene zips")
    parser.add_argument(
        "--required_splits",
        type=int,
        nargs="+",
        default=[2, 3, 6, 12],
        help=(
            "Every (scene_id, scene_half) must contribute exactly one reconstruction for each of these "
            "num_selected_indices values. Raises if any pair is missing a required value; extras are pruned."
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes used while scanning scenes. Defaults to "
            "SLURM_CPUS_PER_TASK when set, otherwise os.cpu_count(), capped by scene count."
        ),
    )
    parser.add_argument(
        "--decode_reconstruction_videos",
        action="store_true",
        help=(
            "Decode render_video and opacity_video payloads for required reconstructions and compare their frame "
            "counts against the DL3DV source scene. This is comprehensive but much slower than schema-only checks."
        ),
    )
    parser.add_argument(
        "--reconstruction_subdir_prefix",
        type=str,
        default=RECONSTRUCTION_SUBDIR_PREFIX,
        help=(
            "Directory-name prefix of the reconstruction subdirs under --data_path; the DL3DV source subdir "
            "is the remainder after the prefix."
        ),
    )
    parser.add_argument(
        "--trainval_image_resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help=(
            "Keep only trainval scenes whose training images are exactly this size (the renders use the 4x "
            "downsampled DL3DV images, so this matches scenes with transforms w/h of 4*WIDTH x 4*HEIGHT). "
            "Scenes at other resolutions are dropped from trainval with a printed summary; the test split is "
            "never filtered. Training gathers residual tensors across ranks, which needs one uniform resolution."
        ),
    )
    args = parser.parse_args()

    test_scenes = load_test_scenes()
    split_data: dict = {"test": defaultdict(list), "trainval": defaultdict(list)}

    scene_subdirs: dict[str, str] = {}
    skipped_bad_scenes = 0
    skipped_non_reconstruction_subdirs = 0
    for subdir in sorted(os.listdir(args.data_path)):
        if not is_reconstruction_subdir(subdir, args.reconstruction_subdir_prefix):
            skipped_non_reconstruction_subdirs += 1
            continue
        subdir_path = os.path.join(args.data_path, subdir)
        if os.path.isdir(subdir_path):
            for scene_id in sorted(os.listdir(subdir_path)):
                if scene_id in BAD_SCENE_IDS:
                    skipped_bad_scenes += 1
                    continue
                previous_subdir = scene_subdirs.setdefault(scene_id, subdir)
                if previous_subdir != subdir:
                    raise ValueError(
                        f"Scene {scene_id} appears in multiple reconstruction subdirs: "
                        f"{previous_subdir} and {subdir}"
                    )
    if skipped_non_reconstruction_subdirs:
        print(f"Skipped {skipped_non_reconstruction_subdirs} non-reconstruction top-level directories")
    if skipped_bad_scenes:
        print(f"Skipped {skipped_bad_scenes} scene directories listed in BAD_SCENE_IDS")
    scene_ids = sorted(scene_subdirs.items())

    required = sorted(set(args.required_splits))
    required_set = set(required)
    target_source_resolution = None
    if args.trainval_image_resolution is not None:
        target_source_resolution = (4 * args.trainval_image_resolution[0], 4 * args.trainval_image_resolution[1])
    source_resolution_counts: dict[tuple[int, int], int] = defaultdict(int)
    excluded_trainval_resolution_counts: dict[tuple[int, int], int] = defaultdict(int)
    all_bad_entries: list[dict] = []
    source_bad_entries_by_scene_id: dict[str, dict] = {}
    geometry_bad_entries_by_scene_id: dict[str, dict] = {}
    max_workers = _resolve_max_workers(args.max_workers, len(scene_ids))
    print(
        f"Scanning {len(scene_ids)} scenes with max_workers={max_workers} "
        f"decode_reconstruction_videos={args.decode_reconstruction_videos}"
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_scene,
                args.data_path,
                sid,
                sub,
                args.dl3dv_dir,
                required_set,
                args.decode_reconstruction_videos,
                args.reconstruction_subdir_prefix,
            )
            for sid, sub in scene_ids
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning scenes"):
            scene_id, per_scene_data, bad_entries, source_issue, geometry_issue, source_resolution = future.result()
            all_bad_entries.extend(bad_entries)
            if source_issue is not None:
                source_bad_entries_by_scene_id[scene_id] = source_issue
            if geometry_issue is not None:
                geometry_bad_entries_by_scene_id[scene_id] = geometry_issue
            if source_resolution is not None:
                source_resolution_counts[source_resolution] += 1
            if (
                target_source_resolution is not None
                and source_resolution is not None
                and source_resolution != target_source_resolution
                and scene_id not in test_scenes
            ):
                excluded_trainval_resolution_counts[source_resolution] += 1
                continue
            for recon in per_scene_data:
                bucket = "test" if scene_id in test_scenes else "trainval"
                split_data[bucket][scene_id].append(recon)

    if source_resolution_counts:
        formatted = ", ".join(
            f"{w}x{h}: {count}" for (w, h), count in sorted(source_resolution_counts.items(), key=lambda kv: -kv[1])
        )
        print(f"Source scene resolutions: {formatted}")
    if excluded_trainval_resolution_counts:
        formatted = ", ".join(
            f"{w}x{h}: {count}"
            for (w, h), count in sorted(excluded_trainval_resolution_counts.items(), key=lambda kv: -kv[1])
        )
        total_excluded = sum(excluded_trainval_resolution_counts.values())
        print(
            f"Excluded {total_excluded} trainval scene(s) whose source resolution is not "
            f"{target_source_resolution[0]}x{target_source_resolution[1]}: {formatted}"
        )

    # Only raise on bad entries whose num_selected_indices is in required_splits — those are the
    # files that the training/eval pipeline actually needs. Out-of-required bad entries are logged
    # so users know about them but don't block regen.
    blocking = [e for e in all_bad_entries if entry_blocks_split(e, required_set)]
    non_blocking = [e for e in all_bad_entries if not entry_blocks_split(e, required_set)]
    if non_blocking:
        print(
            f"Note: {len(non_blocking)} reconstruction file(s) with data issues in non-required "
            f"splits (num_selected_indices not in {required}); ignoring."
        )
    if blocking:
        by_reason: dict[str, list[dict]] = defaultdict(list)
        for entry in blocking:
            by_reason[entry["reason"]].append(entry)
        # Stable per-reason ordering so users can diff across regen runs.
        for reason, entries in by_reason.items():
            entries.sort(key=lambda e: (e["scene_id"], e["scene_half"], e["num_selected_indices"]))
        lines = [
            f"Found {len(blocking)} reconstruction file(s) in required_splits with unresolved data "
            "issues. These scenes were NOT in BAD_SCENE_IDS, which means they need triage — either "
            "the affected h5 needs to be re-run, or (if every required split of the scene is "
            "affected) the scene id should be added to data_processing/bad_scene_ids.txt."
        ]
        for reason in sorted(by_reason):
            entries = by_reason[reason]
            lines.append(f"\n  {reason} ({len(entries)} file(s)):")
            for e in entries:
                detail = format_entry_details(e)
                lines.append(
                    f"    scene_id={e['scene_id']} half={e['scene_half']} "
                    f"n={e['num_selected_indices']} path={e['hdf5_path']} {detail}".rstrip()
                )
        raise ValueError("\n".join(lines))

    if required:
        for bucket in ("test", "trainval"):
            kept: dict = defaultdict(list)
            for scene_id, recons in split_data[bucket].items():
                by_half: dict[str, dict[int, dict]] = defaultdict(dict)
                for r in recons:
                    half = r["scene_half"]
                    n = int(r["num_selected_indices"])
                    # process_scene already rejected duplicate (half, n) files and invalid scales.
                    if n not in required_set:
                        continue
                    by_half[half][n] = r
                for half, by_n in by_half.items():
                    missing = required_set - by_n.keys()
                    if missing:
                        raise ValueError(
                            f"({scene_id}, half={half}) is missing required splits {sorted(missing)}; "
                            f"has {sorted(by_n)}"
                        )
                    scales_required = {n: by_n[n]["scale"] for n in required}
                    canonical = next(iter(scales_required.values()))
                    # rel_tol=1e-3 (0.1%) tolerates small numerical noise from independent
                    # reconstruction runs (observed drift is ~5e-5) while still catching
                    # corruption-level disagreements (2x scale errors and larger).
                    for s in list(scales_required.values())[1:]:
                        if not math.isclose(s, canonical, rel_tol=1e-3, abs_tol=1e-8):
                            raise ValueError(
                                f"({scene_id}, half={half}) has disagreeing scales across required "
                                f"splits: {scales_required}. Expected values to match within rel_tol=1e-3."
                            )
                    kept[scene_id].extend(by_n[n] for n in required)
            split_data[bucket] = kept

    validate_split_membership(split_data, test_scenes)
    included_scene_ids = set(split_data["test"]) | set(split_data["trainval"])
    raise_for_source_bad_entries(
        [
            source_bad_entries_by_scene_id[scene_id]
            for scene_id in sorted(included_scene_ids & source_bad_entries_by_scene_id.keys())
        ]
    )
    raise_for_geometry_bad_entries(
        [
            geometry_bad_entries_by_scene_id[scene_id]
            for scene_id in sorted(included_scene_ids & geometry_bad_entries_by_scene_id.keys())
        ]
    )

    test_recon_count = sum(len(v) for v in split_data["test"].values())
    trainval_recon_count = sum(len(v) for v in split_data["trainval"].values())
    print(f"Found {test_recon_count} reconstructions for {len(split_data['test'])} test scenes")
    print(f"Found {trainval_recon_count} reconstructions for {len(split_data['trainval'])} trainval scenes")

    os.makedirs(args.output_root, exist_ok=True)
    output_path = os.path.join(args.output_root, "trainval_test_split.json")
    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=4)

    notify(
        f"Split complete. Test: {test_recon_count} reconstructions ({len(split_data['test'])} scenes), "
        f"Trainval: {trainval_recon_count} reconstructions ({len(split_data['trainval'])} scenes). "
        f"Saved to {output_path}"
    )


if __name__ == "__main__":
    main()
