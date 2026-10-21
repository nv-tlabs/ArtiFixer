#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ArtiFixer3D utilities for distilling corrected views back into 3DGRUT."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation
from threedgrut.datasets.utils import Camera
from threedgrut.datasets.utils import Image as ColmapImage
from threedgrut.datasets.utils import read_colmap_extrinsics_binary, read_colmap_intrinsics_binary

import data_processing.threedgrut_training as threedgrut_training
from data_processing.camera_trajectories import (
    applied_transform_matrix,
    camera_intrinsics_for_frame,
    opengl_c2w_to_opencv_w2c,
    trajectory_frame_count,
    write_json,
)
from data_processing.prep_utils import (
    colmap_params_to_opencv,
    parse_phases,
    reset_directory,
    write_colmap_cameras,
    write_colmap_images,
)
from data_processing.reconstructed_colmap_split import (
    read_split_scenes,
    resolve_has_gt,
    resolve_optional_path,
    resolve_required_number,
    resolve_required_path,
    split_entry,
    write_split,
)
from data_processing.render_3dgrut_colmap import (
    render_3dgrut_colmap,
    render_outputs_complete,
    selected_indices_for_render,
)
from model_training.utils.video_io import atomic_output_path

ARTIFIXER3D_EXPERIMENT = "artifixer3d"

# Canonical AF3D split pose contract:
#
# Every prepared reconstructed_colmap split stores transforms.json
# `transform_matrix` values as NeRFStudio/OpenGL camera-to-world poses. That is
# the convention produced by prepare_colmap_artifixer_inputs.py through
# opencv_w2c_to_opengl_c2w(). AF3D always materializes generated frames back to
# COLMAP/3DGRUT OpenCV world-to-camera by applying the OpenGL->OpenCV camera
# flip in opengl_c2w_to_opencv_w2c().
POSE_VALIDATION_ROTATION_TOLERANCE_DEGREES = 1e-3
POSE_VALIDATION_TRANSLATION_TOLERANCE = 1e-4


@dataclass(frozen=True)
class PreparedScene:
    # Key from split.json; reused unchanged in ArtiFixer3D+ inference metadata.
    scene_id: str
    # Prepared scene directory; used for default ArtiFixer3D output paths.
    scene_root: Path
    # Source transforms.json defining frame order, intrinsics, and generated-frame poses.
    transforms_path: Path
    # Prepared COLMAP root containing real anchor images and sparse/0.
    colmap_dir: Path
    # Caption embedding file carried into ArtiFixer3D+ inference metadata.
    prompt_path: Path
    # Geometry scale expected by ArtiFixer inference; copied from the prepared
    # split, where it was computed as (reconstruction-units-to-meters factor)
    # x the fixed training conditioning constant.
    camera_scale: float
    # Preserve generated-only metadata as has_gt=False for ArtiFixer3D+ inference.
    has_gt: bool
    # Frame indices that keep real COLMAP images; the complement uses ArtiFixer PNGs.
    selected_indices: list[int]
    # Optional target filter passed through so ArtiFixer3D+ uses the same target frames.
    target_indices_path: Path | None
    # Prepared 3DGRUT checkpoint used as the default ArtiFixer3D resume point.
    reconstruction_checkpoint: Path | None
    # Total transforms frames; used to materialize every COLMAP entry and validate render completeness.
    frame_count: int


@dataclass(frozen=True)
class Artifixer3DPaths:
    """Generated paths for this ArtiFixer3D run; PreparedScene stores existing inputs."""

    # 3DGRUT training output root created by this run and passed as out_dir.
    run_root: Path
    # New COLMAP scene built from real anchors and ArtiFixer predictions, distinct from scene.colmap_dir.
    distillation_input_dir: Path
    # File where scene.selected_indices is written for 3DGRUT training and render validation.
    distillation_selected_indices_path: Path
    # Prediction RGB subdirectory inside distillation_input_dir referenced by image_path_override.
    override_image_dir: Path
    # Render output root created by this run and passed to render_3dgrut_colmap.
    render_output_root: Path
    # Concrete ours_STEP render directory used for reuse checks and the ArtiFixer3D+ split.
    render_checkpoint_dir: Path
    # reconstructed_colmap split consumed by the ArtiFixer3D+ inference pass.
    artifixer3d_plus_inference_split_path: Path


def load_json_list(path: Path) -> list[int]:
    """Read selected-index JSON files and assert they contain integer frame ids."""
    values = json.loads(path.read_text())
    assert isinstance(values, list), f"{path} must contain a JSON list"
    assert all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ), f"{path} must contain integer frame indices"
    return values


def read_frame_count(transforms_path: Path) -> int:
    """Derive frame count from transforms.json, the source of frame order for distillation."""
    transforms = json.loads(transforms_path.read_text())
    frames = transforms["frames"]
    assert isinstance(frames, list), f"{transforms_path} must contain a frames list"
    return len(frames)


def load_prepared_scene(scene_root: Path, split_path: Path | None, scene_id: str | None) -> PreparedScene:
    """Resolve and validate the prepared split into the contract ArtiFixer3D consumes."""
    scene_root = scene_root.resolve()
    split_path = (split_path or scene_root / "split.json").resolve()
    split_data = read_split_scenes(split_path)

    if scene_id is None:
        assert len(split_data) == 1, f"--scene_id is required when {split_path} contains {len(split_data)} scenes"
        scene_id = next(iter(split_data))

    metadata = split_data[scene_id]
    split_root = split_path.parent

    transforms_path = resolve_required_path(split_root, metadata, "transforms_path")
    colmap_dir = resolve_required_path(split_root, metadata, "image_root")
    selected_indices_path = resolve_required_path(split_root, metadata, "selected_indices_path")
    target_indices_path = resolve_optional_path(split_root, metadata, "target_indices_path")
    prompt_path = resolve_required_path(split_root, metadata, "prompt_path")
    reconstruction_checkpoint = resolve_optional_path(split_root, metadata, "reconstruction_checkpoint")
    has_gt = resolve_has_gt(scene_id, metadata)

    selected_indices = load_json_list(selected_indices_path)
    frame_count = read_frame_count(transforms_path)
    camera_scale = resolve_required_number(metadata, "camera_scale")

    required_paths = (transforms_path, colmap_dir, selected_indices_path, prompt_path)
    if target_indices_path is not None:
        required_paths = (*required_paths, target_indices_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Prepared scene {scene_id!r} references missing path: {path}")

    return PreparedScene(
        scene_id=scene_id,
        scene_root=scene_root,
        transforms_path=transforms_path,
        colmap_dir=colmap_dir,
        prompt_path=prompt_path,
        camera_scale=camera_scale,
        has_gt=has_gt,
        selected_indices=selected_indices,
        target_indices_path=target_indices_path,
        reconstruction_checkpoint=reconstruction_checkpoint,
        frame_count=frame_count,
    )


def artifixer3d_paths(
    scene: PreparedScene, output_root: Path | None, artifixer3d_plus_inference_split_path: Path | None, steps: int
) -> Artifixer3DPaths:
    """Centralize output layout so all ArtiFixer3D phases agree on paths."""
    output_root = (output_root or scene.scene_root / "artifixer3d").resolve()
    output_name = scene.scene_id
    artifixer3d_plus_inference_split_path = (
        artifixer3d_plus_inference_split_path or scene.scene_root / "split_artifixer3d_plus.json"
    ).resolve()
    distillation_input_dir = output_root / "distillation_input" / output_name
    render_output_root = output_root / "recon_results" / output_name
    render_checkpoint_dir = render_output_root / ARTIFIXER3D_EXPERIMENT / distillation_input_dir.name / f"ours_{steps}"
    return Artifixer3DPaths(
        run_root=output_root / "runs",
        distillation_input_dir=distillation_input_dir,
        distillation_selected_indices_path=output_root / "distillation_input" / f"{output_name}_selected_indices.json",
        override_image_dir=distillation_input_dir / "artifixer_predictions",
        render_output_root=render_output_root,
        render_checkpoint_dir=render_checkpoint_dir,
        artifixer3d_plus_inference_split_path=artifixer3d_plus_inference_split_path,
    )


def artifixer3d_checkpoint(scene: PreparedScene, paths: Artifixer3DPaths, steps: int) -> Path:
    """Mirror 3DGRUT checkpoint naming so reuse checks target the expected file."""
    return paths.run_root / scene.scene_id / paths.distillation_input_dir.name / f"ours_{steps}" / f"ckpt_{steps}.pt"


def generated_frame_indices(scene: PreparedScene) -> list[int]:
    """Compute pseudo-view indices as every non-anchor frame in the prepared split."""
    selected = set(scene.selected_indices)
    assert len(selected) == len(
        scene.selected_indices
    ), f"Selected indices contain duplicates: {scene.selected_indices}"
    if selected:
        assert (
            min(selected) >= 0 and max(selected) < scene.frame_count
        ), f"Selected indices must be in [0, {scene.frame_count - 1}], got {scene.selected_indices}"
    return [index for index in range(scene.frame_count) if index not in selected]


def require_frames(directory: Path, indices: list[int], description: str) -> None:
    """Raise FileNotFoundError naming up to 10 of the missing frame PNGs."""
    missing = [index for index in indices if not (directory / f"{index:05d}.png").is_file()]
    if missing:
        preview = ", ".join(f"{index:05d}.png" for index in missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"Missing {description} in {directory}: {preview}{suffix}")


def validate_artifixer_frames(scene: PreparedScene, artifixer_frames_dir: Path) -> None:
    """Fail before training if any required ArtiFixer prediction PNG is missing."""
    artifixer_frames_dir = artifixer_frames_dir.resolve()
    if not artifixer_frames_dir.is_dir():
        raise FileNotFoundError(f"Missing ArtiFixer prediction frame directory: {artifixer_frames_dir}")

    generated_indices = generated_frame_indices(scene)
    assert generated_indices, (
        "ArtiFixer3D needs at least one generated/predicted frame. "
        "The prepared selected_indices cover every frame, so there is nothing to distill."
    )
    require_frames(artifixer_frames_dir, generated_indices, "ArtiFixer prediction frames")


def source_frame_path(image_root: Path, file_path: str) -> Path:
    """Resolve source frame file_path entries against the prepared COLMAP root."""
    path = Path(file_path)
    return path if path.is_absolute() else image_root / path


def prediction_image_name(index: int) -> str:
    """Name generated frames by index so they cannot collide with source image basenames."""
    return f"frame_{index:05d}.png"


def selected_image_name(index: int, source: Path) -> str:
    """Name real anchors by frame index while preserving the original image extension."""
    return f"frame_{index:05d}{source.suffix.lower()}"


def symlink_frame(source: Path, target: Path) -> None:
    """Share image bytes without copying and fail if the expected source is absent."""
    if not source.is_file():
        raise FileNotFoundError(f"Missing frame source: {source}")
    target.symlink_to(source.resolve())


def opencv_camera_from_mapping(camera_id: int, intrinsics: dict[str, object]) -> Camera:
    """Build the trajectory OPENCV camera from transforms.json intrinsics."""
    width = int(intrinsics["w"])
    height = int(intrinsics["h"])
    params = np.array(
        [
            float(intrinsics["fl_x"]),
            float(intrinsics["fl_y"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
            float(intrinsics.get("k1") or 0.0),
            float(intrinsics.get("k2") or 0.0),
            float(intrinsics.get("p1") or 0.0),
            float(intrinsics.get("p2") or 0.0),
        ],
        dtype=np.float64,
    )
    return Camera(id=camera_id, model="OPENCV", width=width, height=height, params=params)


def opencv_camera_from_colmap(camera_id: int, camera: Camera, image_size: tuple[int, int]) -> Camera:
    """Convert source COLMAP cameras to the OPENCV Camera object written to cameras.bin."""
    fx, fy, cx, cy, distortion = colmap_params_to_opencv(camera.model, camera.params)
    scale_x = image_size[0] / camera.width
    scale_y = image_size[1] / camera.height
    params = np.array(
        [fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y, *distortion],
        dtype=np.float64,
    )
    return Camera(id=camera_id, model="OPENCV", width=image_size[0], height=image_size[1], params=params)


def opencv_world_to_camera_from_transforms_frame(frame: dict[str, object], applied_transform: np.ndarray) -> np.ndarray:
    """Convert canonical OpenGL C2W split poses into COLMAP/3DGRUT OpenCV W2C."""
    assert "transform_matrix" in frame, "Distillation frame is missing transform_matrix"
    return opengl_c2w_to_opencv_w2c(frame["transform_matrix"], applied_transform)


def colmap_qvec_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized COLMAP qvec in wxyz order."""
    xyzw = Rotation.from_matrix(rotation).as_quat()
    qvec = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    if qvec[0] < 0:
        qvec *= -1
    return qvec / np.linalg.norm(qvec)


def colmap_pose_from_transforms_frame(
    frame: dict[str, object], applied_transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a split transform_matrix to COLMAP OpenCV W2C qvec/tvec."""
    world_to_camera = opencv_world_to_camera_from_transforms_frame(frame, applied_transform)
    return colmap_qvec_from_rotation(world_to_camera[:3, :3]), world_to_camera[:3, 3]


def colmap_rotation_from_qvec(qvec: np.ndarray) -> Rotation:
    """Read a COLMAP qvec in wxyz order as a scipy rotation."""
    qvec = np.asarray(qvec, dtype=np.float64)
    assert qvec.shape == (4,), f"COLMAP qvec must have shape (4,), got {qvec.shape}"
    return Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])


def validate_selected_anchor_poses(
    scene: PreparedScene,
    frames: list[object],
    applied_transform: np.ndarray,
    source_images: dict[str, ColmapImage],
) -> None:
    """Verify selected split frames round-trip to their exact source COLMAP poses."""
    for index in scene.selected_indices:
        assert 0 <= index < len(frames), f"Selected frame {index} is outside transforms frame range"
        frame = frames[index]
        assert isinstance(frame, dict), f"Selected frame {index} metadata must be a mapping"
        assert "file_path" in frame, f"Selected frame {index} is missing file_path"
        source_basename = Path(str(frame["file_path"])).name
        assert source_basename in source_images, f"Selected frame {source_basename!r} is not in source COLMAP"

        candidate_qvec, candidate_tvec = colmap_pose_from_transforms_frame(frame, applied_transform)
        source_image = source_images[source_basename]
        rotation_error_degrees = (
            (colmap_rotation_from_qvec(candidate_qvec) * colmap_rotation_from_qvec(source_image.qvec).inv()).magnitude()
            * 180.0
            / np.pi
        )
        translation_error = float(np.linalg.norm(np.asarray(candidate_tvec) - np.asarray(source_image.tvec)))
        if (
            rotation_error_degrees > POSE_VALIDATION_ROTATION_TOLERANCE_DEGREES
            or translation_error > POSE_VALIDATION_TRANSLATION_TOLERANCE
        ):
            raise AssertionError(
                f"Selected frame {index} ({source_basename}) does not match source COLMAP under "
                "the canonical OpenGL C2W camera pose contract: "
                f"rotation_error_degrees={rotation_error_degrees:.6g}, "
                f"translation_error={translation_error:.6g}. "
                "Regenerate the prepared split instead of reusing bad ArtiFixer3D outputs."
            )


def copy_source_points3d(source_colmap_dir: Path, path: Path) -> None:
    """Seed scratch ArtiFixer3D initialization from the source COLMAP points."""
    source = source_colmap_dir / "sparse" / "0" / "points3D.bin"
    if not source.is_file():
        raise FileNotFoundError(f"Missing source COLMAP points for ArtiFixer3D initialization: {source}")
    data = source.read_bytes()
    if len(data) < 8:
        raise ValueError(f"Source COLMAP points file is too small: {source}")
    point_count = struct.unpack("<Q", data[:8])[0]
    if point_count == 0:
        raise ValueError(f"Source COLMAP points file is empty: {source}")
    path.write_bytes(data)


def source_colmap_lookup(colmap_dir: Path) -> tuple[dict[str, ColmapImage], dict[int, Camera]]:
    """Load source COLMAP metadata to preserve exact real-anchor poses and cameras."""
    sparse_dir = colmap_dir / "sparse" / "0"
    images = read_colmap_extrinsics_binary(sparse_dir / "images.bin")
    cameras = read_colmap_intrinsics_binary(sparse_dir / "cameras.bin")
    images_by_basename = {Path(image.name).name: image for image in images}
    return images_by_basename, cameras


def materialize_distillation_input(
    scene: PreparedScene,
    paths: Artifixer3DPaths,
    artifixer_frames_dir: Path,
) -> None:
    """Build a COLMAP scene combining real anchors with ArtiFixer prediction views."""
    validate_artifixer_frames(scene, artifixer_frames_dir)
    transforms = json.loads(scene.transforms_path.read_text())
    frames = transforms["frames"]
    assert isinstance(frames, list), f"{scene.transforms_path} must contain a frames list"
    applied_transform = applied_transform_matrix(transforms)

    image_dir = paths.distillation_input_dir / "images"
    sparse_dir = paths.distillation_input_dir / "sparse" / "0"
    reset_directory(paths.distillation_input_dir)
    image_dir.mkdir()
    sparse_dir.mkdir(parents=True)
    paths.override_image_dir.mkdir()

    source_images, source_cameras = source_colmap_lookup(scene.colmap_dir)
    validate_selected_anchor_poses(scene, frames, applied_transform, source_images)
    selected = set(scene.selected_indices)
    colmap_cameras = []
    colmap_images = []
    for index, frame in enumerate(frames):
        assert isinstance(frame, dict), f"Frame {index} metadata must be a mapping"
        if index in selected:
            assert "file_path" in frame, f"Selected frame {index} is missing file_path"
            source = source_frame_path(scene.colmap_dir, str(frame["file_path"]))
            image_name = selected_image_name(index, source)
            source_basename = Path(str(frame["file_path"])).name
            assert source_basename in source_images, f"Selected frame {source_basename!r} is not in source COLMAP"
            source_image = source_images[source_basename]
            with PILImage.open(source) as image:
                image_size = image.size
            camera_id = len(colmap_cameras) + 1
            colmap_cameras.append(
                opencv_camera_from_colmap(camera_id, source_cameras[source_image.camera_id], image_size)
            )
            qvec, tvec = source_image.qvec, source_image.tvec
        else:
            source = artifixer_frames_dir / f"{index:05d}.png"
            image_name = prediction_image_name(index)
            camera_id = len(colmap_cameras) + 1
            colmap_cameras.append(opencv_camera_from_mapping(camera_id, camera_intrinsics_for_frame(transforms, frame)))
            symlink_frame(source, paths.override_image_dir / f"{index:05d}.png")
            qvec, tvec = colmap_pose_from_transforms_frame(frame, applied_transform)
        symlink_frame(source, image_dir / image_name)

        colmap_images.append(
            ColmapImage(
                id=index + 1,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty(0, dtype=np.int64),
            )
        )

    write_colmap_cameras(sparse_dir / "cameras.bin", colmap_cameras)
    write_colmap_images(sparse_dir / "images.bin", colmap_images)
    copy_source_points3d(scene.colmap_dir, sparse_dir / "points3D.bin")
    write_json(paths.distillation_selected_indices_path, scene.selected_indices)


def train_artifixer3d(
    scene: PreparedScene,
    paths: Artifixer3DPaths,
    *,
    artifixer_frames_dir: Path,
    base_checkpoint: Path | None,
    config_name: str,
    steps: int,
    use_wandb: bool,
    replace: bool,
) -> tuple[Path, bool]:
    """Run or reuse the 3DGRUT distillation checkpoint for the prepared scene."""
    checkpoint = artifixer3d_checkpoint(scene, paths, steps)
    artifixer_frames_dir = artifixer_frames_dir.resolve()
    if checkpoint.is_file() and not replace:
        print(f"Skipping ArtiFixer3D distillation; found {checkpoint}", flush=True)
        return checkpoint, True

    validate_artifixer_frames(scene, artifixer_frames_dir)
    if base_checkpoint is not None:
        base_checkpoint = base_checkpoint.resolve()
        if not base_checkpoint.is_file():
            raise FileNotFoundError(f"Missing initial 3DGRUT checkpoint: {base_checkpoint}")
        if base_checkpoint.is_relative_to(checkpoint.parent.resolve()):
            raise ValueError("Use a base checkpoint outside the ArtiFixer3D output checkpoint directory")
    paths.artifixer3d_plus_inference_split_path.unlink(missing_ok=True)
    reset_directory(paths.render_output_root)
    reset_directory(checkpoint.parent)
    materialize_distillation_input(scene, paths, artifixer_frames_dir)

    overrides = [
        f"path={paths.distillation_input_dir}",
        f"out_dir={paths.run_root}",
        f"selected_indices_file={paths.distillation_selected_indices_path}",
        f"image_path_override={paths.override_image_dir.name}",
        "test_last=False",
        "export_ingp.enabled=False",
        f"experiment_name={scene.scene_id}",
        f"n_iterations={steps}",
        f"use_wandb={'True' if use_wandb else 'False'}",
        f"checkpoint.iterations=[{steps}]",
    ]
    if base_checkpoint is not None:
        overrides.append(f"resume={base_checkpoint}")

    threedgrut_training.train_3dgrut(config_name, overrides, threedgrut_training.DEFAULT_THREEDGRUT_CONFIG_DIR)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"3DGRUT did not write expected ArtiFixer3D checkpoint: {checkpoint}")
    return checkpoint, False


def render_artifixer3d(
    scene: PreparedScene,
    paths: Artifixer3DPaths,
    *,
    checkpoint: Path,
    checkpoint_reused: bool,
    replace: bool,
    render_trajectory_path: Path,
) -> Path:
    """Render the distilled checkpoint on the requested camera trajectory."""
    render_trajectory_path = render_trajectory_path.resolve()
    if not render_trajectory_path.is_file():
        raise FileNotFoundError(f"Missing ArtiFixer3D render trajectory: {render_trajectory_path}")
    render_frame_count = trajectory_frame_count(render_trajectory_path)
    use_distillation_trajectory = render_trajectory_path == scene.transforms_path.resolve()
    trajectory_output_subdir = "" if use_distillation_trajectory else None
    expected_render_dir = (
        paths.render_checkpoint_dir
        if trajectory_output_subdir == ""
        else paths.render_checkpoint_dir / render_trajectory_path.stem
    )
    if (
        not replace
        and checkpoint_reused
        and render_outputs_complete(
            expected_render_dir,
            render_frame_count,
            expected_selected_indices=selected_indices_for_render(paths.distillation_selected_indices_path, None),
        )
    ):
        print(f"Skipping ArtiFixer3D render; found complete outputs in {expected_render_dir}", flush=True)
        return expected_render_dir

    output_dir = render_3dgrut_colmap(
        checkpoint=checkpoint,
        colmap_dir=paths.distillation_input_dir,
        output_root=paths.render_output_root,
        experiment_name=ARTIFIXER3D_EXPERIMENT,
        selected_indices=paths.distillation_selected_indices_path,
        trajectory_path=render_trajectory_path,
        trajectory_output_subdir=trajectory_output_subdir,
    )
    return output_dir


def require_render_outputs(
    render_dir: Path,
    frame_count: int,
    expected_selected_indices_path: Path | None = None,
) -> None:
    """Check ArtiFixer3D renders before writing the next inference split."""
    if not render_dir.is_dir():
        raise FileNotFoundError(f"Missing ArtiFixer3D render checkpoint directory: {render_dir}")

    selected_indices_path = render_dir / "selected_indices.json"
    if not selected_indices_path.is_file():
        raise FileNotFoundError(f"Missing ArtiFixer3D selected-indices file: {selected_indices_path}")
    if expected_selected_indices_path is not None:
        assert load_json_list(selected_indices_path) == load_json_list(
            expected_selected_indices_path
        ), f"ArtiFixer3D selected indices at {selected_indices_path} do not match {expected_selected_indices_path}"

    for name in ("renders", "opacity"):
        directory = render_dir / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing ArtiFixer3D render output directory: {directory}")
        require_frames(directory, list(range(frame_count)), f"ArtiFixer3D {name} frames")


def write_artifixer3d_plus_inference_split(
    scene: PreparedScene,
    paths: Artifixer3DPaths,
    render_dir: Path,
    reconstruction_checkpoint: Path,
) -> Path:
    """Write the reconstructed_colmap split consumed by ArtiFixer3D+ inference."""
    output_split = paths.artifixer3d_plus_inference_split_path
    require_render_outputs(
        render_dir,
        scene.frame_count,
        expected_selected_indices_path=paths.distillation_selected_indices_path,
    )
    if not reconstruction_checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing ArtiFixer3D checkpoint for ArtiFixer3D+ inference: {reconstruction_checkpoint}"
        )

    entry = split_entry(
        output_split,
        scene.scene_id,
        transforms_path=scene.transforms_path,
        image_root=scene.colmap_dir,
        render_dir=render_dir / "renders",
        opacity_dir=render_dir / "opacity",
        selected_indices_path=render_dir / "selected_indices.json",
        prompt_path=scene.prompt_path,
        reconstruction_checkpoint=reconstruction_checkpoint,
        camera_scale=scene.camera_scale,
        target_indices_path=scene.target_indices_path,
        has_gt=scene.has_gt,
    )
    with atomic_output_path(output_split) as temporary:
        write_split(temporary, scene.scene_id, entry)
    print(f"Wrote ArtiFixer3D+ inference split: {output_split}", flush=True)
    return output_split


def run_artifixer3d(args: argparse.Namespace) -> None:
    """Orchestrate requested ArtiFixer3D phases with reuse and replace semantics."""
    phases = parse_phases(args.phases, {"distill", "render", "prepare_artifixer3d_plus"})
    scene = load_prepared_scene(args.scene_root, args.split_path, args.scene_id)
    paths = artifixer3d_paths(
        scene, args.output_root, args.artifixer3d_plus_inference_split_path, args.artifixer3d_steps
    )

    checkpoint = artifixer3d_checkpoint(scene, paths, args.artifixer3d_steps)
    render_trajectory_path = (args.render_trajectory_path or scene.transforms_path).resolve()
    if "prepare_artifixer3d_plus" in phases and render_trajectory_path != scene.transforms_path.resolve():
        raise ValueError(
            "--phases prepare_artifixer3d_plus requires rendering the distillation trajectory. "
            "Use --phases render with --render_trajectory_path for arbitrary post-training renders."
        )

    checkpoint_reused = True
    if "distill" in phases:
        assert (
            args.artifixer_frames_dir is not None
        ), "--artifixer_frames_dir is required when --phases includes distill"
        checkpoint, checkpoint_reused = train_artifixer3d(
            scene,
            paths,
            artifixer_frames_dir=args.artifixer_frames_dir,
            base_checkpoint=args.base_checkpoint,
            config_name=args.config_name,
            steps=args.artifixer3d_steps,
            use_wandb=args.use_wandb,
            replace=args.replace,
        )
    elif "render" in phases and not checkpoint.is_file():
        raise FileNotFoundError(f"Missing ArtiFixer3D checkpoint for render phase: {checkpoint}")

    render_dir = paths.render_checkpoint_dir
    if "render" in phases:
        paths.artifixer3d_plus_inference_split_path.unlink(missing_ok=True)
        render_dir = render_artifixer3d(
            scene,
            paths,
            checkpoint=checkpoint,
            checkpoint_reused=checkpoint_reused,
            replace=args.replace,
            render_trajectory_path=render_trajectory_path,
        )
    if "prepare_artifixer3d_plus" in phases:
        write_artifixer3d_plus_inference_split(scene, paths, render_dir, checkpoint)

    if "distill" in phases or "render" in phases:
        print(f"artifixer3d_checkpoint={checkpoint}", flush=True)
    if "render" in phases or "prepare_artifixer3d_plus" in phases:
        print(f"artifixer3d_render_dir={render_dir}", flush=True)
    if "prepare_artifixer3d_plus" in phases:
        print(f"artifixer3d_plus_inference_split={paths.artifixer3d_plus_inference_split_path}", flush=True)
