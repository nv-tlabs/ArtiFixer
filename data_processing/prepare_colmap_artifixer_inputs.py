#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a COLMAP scene for ArtiFixer inference.

This command turns a COLMAP scene into the dataset layout consumed by
`model_eval.run_inference`. It prepares the input files, trains and renders a
3DGUT reconstruction, estimates metric scale, and generates captions. It does
not run ArtiFixer inference.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from threedgrut.datasets.utils import (
    Camera,
    Image,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_intrinsics_binary,
)

from data_processing.camera_trajectories import (
    assert_target_only_trajectory,
    camera_intrinsics_for_frame,
    frame_with_camera_intrinsics,
    opencv_w2c_to_opengl_c2w,
    read_camera_trajectory,
    trajectory_frame_count,
    transforms_json,
    write_json,
)
from data_processing.captioning.generate_captions import generate_caption_hdf5
from data_processing.captioning.prompt_hdf5 import default_prompt_model_id
from data_processing.prep_utils import (
    colmap_params_to_opencv,
    image_basename,
    parse_phases,
    reset_directory,
    write_colmap_cameras,
    write_colmap_images,
)
from data_processing.reconstructed_colmap_split import path_for_split, resolve_required_path, split_entry, write_split
from data_processing.render_3dgrut_colmap import (
    render_3dgrut_colmap,
    render_outputs_complete,
    selected_indices_for_render,
)
from data_processing.scene_utils import COLMAP_SPARSE_FILES, COLMAP_SPARSE_SUBDIRS, find_colmap_sparse_dir
from data_processing.sparse_recon.metric_alignment import align_colmap_to_metric_scale
from data_processing.threedgrut_training import DEFAULT_THREEDGRUT_CONFIG_DIR, train_3dgrut
from model_training.constants import DL3DV_DATASET_FPS
from model_training.prompt_formats import (
    PROMPT_FORMAT_CHOICES,
    WAN_T5_PROMPT_FORMAT,
    caption_filename_for_format,
    validate_prompt_file_format,
)
from model_training.utils.video_io import atomic_output_path

DEFAULT_THREEDGRUT_CONFIG = "apps/colmap_3dgut_sparse_mcmc"
DEFAULT_CAPTIONING_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_TEXT_ENCODER_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DEFAULT_CAPTION_FILENAME = "caption.h5"
DEFAULT_RECON_SUBDIR = "reconstruction"
DEFAULT_CAMERA_CONDITIONING_SCALE = 0.01
SUPPORTED_CAMERA_MODELS = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"}

# Canonical pose contract for prepared ArtiFixer/AF3D reconstructed_colmap splits.
# The writer converts source COLMAP OpenCV W2C poses into NeRFStudio/OpenGL C2W
# transform_matrix values; AF3D converts them back through the inverse helper and
# validates selected anchors against the original COLMAP images.bin.


@dataclass(frozen=True)
class ColmapScene:
    cameras: list[Camera]
    images: list[Image]


@dataclass(frozen=True)
class PreparedPaths:
    scene_id: str
    scene_root: Path
    threedgrut_input_dir: Path
    reconstruction_run_dir: Path
    render_output_root: Path
    render_checkpoint_dir: Path
    eval_dataset_root: Path
    eval_scene_dir: Path
    recon_results_dir: Path
    prompt_dir: Path
    split_path: Path
    render_result_path: Path
    selected_indices_path: Path
    scale_dir: Path
    prompt_filename: str = DEFAULT_CAPTION_FILENAME


@dataclass(frozen=True)
class PreparedTrajectory:
    root: Path
    render_dataset_dir: Path
    transforms_path: Path
    render_trajectory_path: Path
    selected_indices_path: Path
    target_indices_path: Path
    render_checkpoint_dir: Path


def prepared_paths(
    scene_root: Path,
    scene_id: str,
    reconstruction_steps: int,
    prompt_format: str = WAN_T5_PROMPT_FORMAT,
) -> PreparedPaths:
    threedgrut_input_dir = scene_root / "3dgrut_input" / scene_id
    recon_results_dir = scene_root / "recon_results"
    render_output_root = recon_results_dir / scene_id
    render_checkpoint_dir = render_output_root / DEFAULT_RECON_SUBDIR / scene_id / f"ours_{reconstruction_steps}"
    return PreparedPaths(
        scene_id=scene_id,
        scene_root=scene_root,
        threedgrut_input_dir=threedgrut_input_dir,
        reconstruction_run_dir=scene_root / "3dgrut_runs",
        render_output_root=render_output_root,
        render_checkpoint_dir=render_checkpoint_dir,
        eval_dataset_root=threedgrut_input_dir.parent,
        eval_scene_dir=threedgrut_input_dir,
        recon_results_dir=recon_results_dir,
        prompt_dir=scene_root / "captions",
        split_path=scene_root / "split.json",
        render_result_path=scene_root / "render.json",
        selected_indices_path=scene_root / "selected_indices.json",
        scale_dir=scene_root / "metric_alignment",
        prompt_filename=caption_filename_for_format(DEFAULT_CAPTION_FILENAME, prompt_format),
    )


def resolve_image_dir(scene_dir: Path) -> Path:
    """Return `images/`, or the single downsampled `images_<factor>/` directory when `images/` is absent."""
    image_dir = scene_dir / "images"
    if image_dir.is_dir():
        return image_dir
    downsampled = sorted(
        path for path in scene_dir.glob("images_*") if path.is_dir() and path.name.removeprefix("images_").isdigit()
    )
    if len(downsampled) == 1:
        return downsampled[0]
    if downsampled:
        names = ", ".join(path.name for path in downsampled)
        raise ValueError(f"Missing {image_dir}; found several downsampled image directories ({names})")
    raise FileNotFoundError(f"Missing COLMAP images directory: {image_dir}")


def resolve_colmap_paths(colmap_dir: Path) -> tuple[Path, Path]:
    colmap_dir = colmap_dir.resolve()
    image_dir = resolve_image_dir(colmap_dir)
    sparse_dir = find_colmap_sparse_dir(colmap_dir)
    if sparse_dir is None:
        expected = " or ".join(str(colmap_dir / rel) for rel in COLMAP_SPARSE_SUBDIRS)
        raise FileNotFoundError(f"Missing COLMAP sparse model: expected {expected}")
    for name in COLMAP_SPARSE_FILES:
        if not (sparse_dir / name).is_file():
            raise FileNotFoundError(f"Missing COLMAP sparse file: {sparse_dir / name}")
    return image_dir, sparse_dir


def resolve_pyramid_factor(image_dir: Path, pyramid_factor: int | None) -> int | None:
    """Use an explicit factor or the selected ``images_<factor>`` directory.

    Raster dimensions alone do not identify a pyramid: a small crop can have
    the same dimensions as a rounded reduction.
    """
    suffix = image_dir.name.removeprefix("images_")
    directory_factor = int(suffix) if image_dir.name.startswith("images_") and suffix.isdigit() else None
    if directory_factor is not None and directory_factor < 1:
        raise ValueError(f"Image pyramid factor must be a positive integer: {image_dir}")
    if pyramid_factor is not None:
        if isinstance(pyramid_factor, bool) or not isinstance(pyramid_factor, int) or pyramid_factor < 1:
            raise ValueError("Image pyramid factor must be a positive integer")
        if directory_factor is not None and pyramid_factor != directory_factor:
            raise ValueError(f"Pyramid factor {pyramid_factor} conflicts with image directory {image_dir}")
        return pyramid_factor
    return directory_factor


def read_colmap_scene(sparse_dir: Path) -> ColmapScene:
    cameras = read_colmap_intrinsics_binary(sparse_dir / "cameras.bin")
    return ColmapScene(
        cameras=[cameras[camera_id] for camera_id in sorted(cameras)],
        images=read_colmap_extrinsics_binary(sparse_dir / "images.bin"),
    )


def read_selected_image_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_selected_indices(args: argparse.Namespace, images: Sequence[Image]) -> list[int]:
    if args.selected_image_names_file is not None:
        name_to_index = {Path(image.name).name: index for index, image in enumerate(images)}
        names = read_selected_image_names(args.selected_image_names_file)
        basenames = [Path(name).name for name in names]
        missing = [name for name in basenames if name not in name_to_index]
        assert not missing, f"Selected image names are not in the COLMAP model: {missing}"
        selected = [name_to_index[name] for name in basenames]
    else:
        selected = list(range(len(images)))

    assert selected, "At least one training view is required"
    assert len(set(selected)) == len(selected), f"Selected indices contain duplicates: {selected}"
    assert min(selected) >= 0 and max(selected) < len(
        images
    ), f"Selected indices must be in [0, {len(images) - 1}], got {selected}"
    return selected


def require_unique_basenames(images: Sequence[Image]) -> None:
    basenames = [Path(image.name).name for image in images]
    assert len(set(basenames)) == len(basenames), "COLMAP image basenames must be unique"


def source_image_path(image_dir: Path, image: Image) -> Path:
    for candidate in (image_dir / image.name, image_dir / Path(image.name).name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find image for COLMAP entry {image.name!r} under {image_dir}")


def scaled_camera_params(camera: Camera, sx: float, sy: float) -> np.ndarray:
    params = camera.params.astype(np.float64).copy()
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        assert np.isclose(sx, sy), f"Camera {camera.id} requires uniform scaling for {camera.model}"
        params[0] *= sx
        params[1] *= sx
        params[2] *= sy
    elif camera.model in {"PINHOLE", "OPENCV"}:
        params[0] *= sx
        params[1] *= sy
        params[2] *= sx
        params[3] *= sy
    else:
        assert False, f"Unsupported camera model for image-resolution scaling: {camera.model}"
    return params


def scale_image_observations(image: Image, sx: float, sy: float) -> Image:
    xys = image.xys.astype(np.float64).copy()
    if len(xys):
        xys[:, 0] *= sx
        xys[:, 1] *= sy
    return image._replace(xys=xys)


def scale_colmap_scene_to_images(
    image_dir: Path, scene: ColmapScene, *, pyramid_factor: int | None = None
) -> ColmapScene:
    """Match calibration to image sizes, with rounding only for a known pyramid.

    An explicit integer factor permits at most one output pixel of rounding
    per axis and preserves the common calibration scale. Without a factor,
    images must have the same horizontal and vertical scale.
    """
    if pyramid_factor is not None and (
        isinstance(pyramid_factor, bool) or not isinstance(pyramid_factor, int) or pyramid_factor < 1
    ):
        raise ValueError("Image pyramid factor must be a positive integer")
    camera_by_id = {camera.id: camera for camera in scene.cameras}
    image_info_by_camera: dict[int, tuple[int, int, str, Path]] = {}
    for image in scene.images:
        assert (
            image.camera_id in camera_by_id
        ), f"COLMAP image {image.name!r} references missing camera {image.camera_id}"
        image_path = source_image_path(image_dir, image)
        with PILImage.open(image_path) as pil_image:
            width, height = pil_image.size
        previous = image_info_by_camera.setdefault(image.camera_id, (width, height, image.name, image_path))
        assert previous[:2] == (width, height), (
            f"COLMAP camera {image.camera_id} is shared by images with different sizes: "
            f"{previous[0]}x{previous[1]} and {width}x{height}"
        )

    scaled_cameras = []
    scale_by_camera: dict[int, tuple[float, float]] = {}
    for camera in scene.cameras:
        image_info = image_info_by_camera.get(camera.id)
        if image_info is None:
            scaled_cameras.append(camera)
            scale_by_camera[camera.id] = (1.0, 1.0)
            continue
        width, height, image_name, image_path = image_info
        if (width, height) == (int(camera.width), int(camera.height)):
            scaled_cameras.append(camera)
            scale_by_camera[camera.id] = (1.0, 1.0)
            continue
        sx = width / float(camera.width)
        sy = height / float(camera.height)
        if pyramid_factor is not None:
            tolerance = 1 if pyramid_factor > 1 else 0
            assert (
                abs(float(camera.width) / pyramid_factor - width) <= tolerance
                and abs(float(camera.height) / pyramid_factor - height) <= tolerance
            ), (
                f"COLMAP camera/image size mismatch for {image_name!r}: "
                f"camera {camera.id} is {int(camera.width)}x{int(camera.height)} at pyramid factor "
                f"{pyramid_factor}, but {image_path} is {width}x{height}"
            )
            sx = sy = 1.0 / pyramid_factor
        assert np.isclose(sx, sy), (
            f"COLMAP camera/image size mismatch for {image_name!r}: "
            f"camera {camera.id} is {int(camera.width)}x{int(camera.height)}, "
            f"but {image_path} is {width}x{height}"
        )
        params = scaled_camera_params(camera, sx, sy)
        print(
            f"Scaling COLMAP camera {camera.id} from {int(camera.width)}x{int(camera.height)} "
            f"to {width}x{height} to match input images",
            flush=True,
        )
        scaled_cameras.append(camera._replace(width=width, height=height, params=params))
        scale_by_camera[camera.id] = (sx, sy)

    scaled_images = [scale_image_observations(image, *scale_by_camera[image.camera_id]) for image in scene.images]
    return ColmapScene(cameras=scaled_cameras, images=scaled_images)


def reset_prepared_scene(path: Path) -> None:
    reset_directory(path)
    (path / "images").mkdir()
    (path / "sparse/0").mkdir(parents=True)


def symlink_images(source_image_dir: Path, target_image_dir: Path, images: Sequence[Image]) -> None:
    for image in images:
        (target_image_dir / image_basename(image)).symlink_to(source_image_path(source_image_dir, image).resolve())


def write_sparse_model(source_sparse_dir: Path, target_sparse_dir: Path, scene: ColmapScene) -> None:
    write_colmap_cameras(target_sparse_dir / "cameras.bin", scene.cameras)
    write_colmap_images(target_sparse_dir / "images.bin", scene.images)
    (target_sparse_dir / "points3D.bin").symlink_to((source_sparse_dir / "points3D.bin").resolve())


def image_to_nerfstudio_transform(image: Image) -> list[list[float]]:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = qvec_to_so3(image.qvec)
    world_to_camera[:3, 3] = image.tvec
    return opencv_w2c_to_opengl_c2w(world_to_camera).tolist()


def camera_intrinsics(camera: Camera) -> dict[str, float | int]:
    fx, fy, cx, cy, (k1, k2, p1, p2) = colmap_params_to_opencv(camera.model, camera.params)
    values = {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "p1": p1, "p2": p2}
    return {
        "camera_model": "OPENCV",
        "w": int(camera.width),
        "h": int(camera.height),
        **{key: float(value) for key, value in values.items()},
    }


def shared_camera_intrinsics(scene: ColmapScene) -> dict[str, float | int]:
    camera_by_id = {camera.id: camera for camera in scene.cameras}
    used_cameras = [camera_by_id[camera_id] for camera_id in sorted({image.camera_id for image in scene.images})]
    unsupported = sorted({camera.model for camera in used_cameras} - SUPPORTED_CAMERA_MODELS)
    assert not unsupported, f"Unsupported camera models for ArtiFixer transforms: {unsupported}"

    intrinsics = camera_intrinsics(used_cameras[0])
    for camera in used_cameras[1:]:
        assert camera_intrinsics(camera) == intrinsics, "ArtiFixer COLMAP prep expects one shared intrinsic calibration"
    return intrinsics


def source_frame(image: Image) -> dict[str, object]:
    return {
        "file_path": f"images/{image_basename(image)}",
        "transform_matrix": image_to_nerfstudio_transform(image),
    }


def write_transforms_json(path: Path, scene: ColmapScene) -> None:
    write_json(
        path,
        transforms_json(
            shared_camera_intrinsics(scene),
            [source_frame(image) for image in scene.images],
        ),
    )


def reuse_prepared_scene(
    args: argparse.Namespace, paths: PreparedPaths, scene: ColmapScene
) -> tuple[ColmapScene, list[int]]:
    """Keep saved frame order and anchors when a later phase reuses prepared inputs."""
    prepared = read_colmap_scene(paths.threedgrut_input_dir / "sparse/0")
    source_by_name = {image_basename(image): image for image in scene.images}
    saved_names = [image_basename(image) for image in prepared.images]
    if set(saved_names) != set(source_by_name):
        raise ValueError("COLMAP images differ from the prepared scene; use a new --output_root")
    ordered = ColmapScene(scene.cameras, [source_by_name[name] for name in saved_names])
    saved_geometry = transforms_json(
        shared_camera_intrinsics(prepared), [source_frame(image) for image in prepared.images]
    )
    current_geometry = transforms_json(
        shared_camera_intrinsics(ordered), [source_frame(image) for image in ordered.images]
    )
    if current_geometry != saved_geometry:
        raise ValueError("COLMAP cameras differ from the prepared scene; use a new --output_root")
    selected = selected_indices_for_render(paths.selected_indices_path, None)
    if not selected or len(selected) != len(set(selected)) or min(selected) < 0 or max(selected) >= len(saved_names):
        raise ValueError(f"Invalid saved training-view selection: {paths.selected_indices_path}")
    if args.selected_image_names_file is not None:
        requested = resolve_selected_indices(args, prepared.images)
        if requested != selected:
            raise ValueError(
                "Training-view selection differs from the prepared scene; use a new --output_root "
                "or rerun all phases with --replace"
            )
    return prepared, selected


def eval_split_entry(
    paths: PreparedPaths,
    *,
    transforms_path: Path,
    image_root: Path,
    render_dir: Path,
    opacity_dir: Path,
    selected_indices_path: Path,
    prompt_path: Path,
    metric_scale: float,
    reconstruction_checkpoint_path: Path,
    target_indices_path: Path | None = None,
    has_gt: bool = True,
) -> dict[str, object]:
    return split_entry(
        paths.split_path,
        paths.scene_id,
        transforms_path=transforms_path,
        image_root=image_root,
        render_dir=render_dir,
        opacity_dir=opacity_dir,
        selected_indices_path=selected_indices_path,
        prompt_path=prompt_path,
        reconstruction_checkpoint=reconstruction_checkpoint_path,
        camera_scale=metric_scale * DEFAULT_CAMERA_CONDITIONING_SCALE,
        target_indices_path=target_indices_path,
        has_gt=has_gt,
    )


def trajectory_split_entry(
    paths: PreparedPaths,
    trajectory: PreparedTrajectory,
    metric_scale: float,
    reconstruction_checkpoint_path: Path,
) -> dict[str, object]:
    prompt_path = paths.prompt_dir / paths.scene_id / paths.prompt_filename
    return eval_split_entry(
        paths,
        transforms_path=trajectory.transforms_path,
        image_root=paths.threedgrut_input_dir,
        render_dir=trajectory.render_checkpoint_dir / "renders",
        opacity_dir=trajectory.render_checkpoint_dir / "opacity",
        selected_indices_path=trajectory.render_checkpoint_dir / "selected_indices.json",
        prompt_path=prompt_path,
        metric_scale=metric_scale,
        reconstruction_checkpoint_path=reconstruction_checkpoint_path,
        target_indices_path=trajectory.target_indices_path,
        has_gt=False,
    )


def render_checkpoint_dir(paths: PreparedPaths, trajectory: PreparedTrajectory | None) -> Path:
    return paths.render_checkpoint_dir if trajectory is None else trajectory.render_checkpoint_dir


def render_frame_count(paths: PreparedPaths, trajectory: PreparedTrajectory | None) -> int:
    transforms_path = (
        paths.threedgrut_input_dir / "nerfstudio/transforms.json"
        if trajectory is None
        else trajectory.render_trajectory_path
    )
    return trajectory_frame_count(transforms_path)


def save_render_result(paths: PreparedPaths, trajectory: PreparedTrajectory | None, checkpoint: Path) -> None:
    result = {
        "render_dir": path_for_split(paths.render_result_path, render_checkpoint_dir(paths, trajectory)),
        "reconstruction_checkpoint": path_for_split(paths.render_result_path, checkpoint),
        "trajectory": trajectory is not None,
    }
    with atomic_output_path(paths.render_result_path) as temporary:
        write_json(temporary, result)


def reuse_render_result(paths: PreparedPaths) -> tuple[PreparedPaths, PreparedTrajectory | None, Path]:
    """Recover completed rendering when a later phase has not yet written the split."""
    result = json.loads(paths.render_result_path.read_text())
    root = paths.render_result_path.parent
    render_dir = resolve_required_path(root, result, "render_dir")
    checkpoint = resolve_required_path(root, result, "reconstruction_checkpoint")
    is_trajectory = result["trajectory"]
    assert isinstance(is_trajectory, bool), f"Invalid trajectory flag in {paths.render_result_path}"
    if is_trajectory:
        trajectory = replace(trajectory_paths(paths), render_checkpoint_dir=render_dir)
    else:
        paths = replace(paths, render_checkpoint_dir=render_dir)
        trajectory = None
    selection_path = paths.selected_indices_path if trajectory is None else trajectory.selected_indices_path
    if not checkpoint.is_file() or not render_outputs_complete(
        render_dir,
        render_frame_count(paths, trajectory),
        expected_selected_indices=selected_indices_for_render(selection_path, None),
    ):
        raise ValueError("Saved rendering is incomplete; rerun the render phase")
    return paths, trajectory, checkpoint


def write_eval_split(
    paths: PreparedPaths,
    metric_scale: float,
    trajectory: PreparedTrajectory | None,
    reconstruction_checkpoint_path: Path,
) -> None:
    prompt_path = paths.prompt_dir / paths.scene_id / paths.prompt_filename
    if trajectory is None:
        entry = eval_split_entry(
            paths,
            transforms_path=paths.threedgrut_input_dir / "nerfstudio/transforms.json",
            image_root=paths.threedgrut_input_dir,
            render_dir=paths.render_checkpoint_dir / "renders",
            opacity_dir=paths.render_checkpoint_dir / "opacity",
            selected_indices_path=paths.render_checkpoint_dir / "selected_indices.json",
            prompt_path=prompt_path,
            metric_scale=metric_scale,
            reconstruction_checkpoint_path=reconstruction_checkpoint_path,
        )
    else:
        entry = trajectory_split_entry(paths, trajectory, metric_scale, reconstruction_checkpoint_path)
    with atomic_output_path(paths.split_path) as temporary:
        write_split(temporary, paths.scene_id, entry)


def prepare_files(
    args: argparse.Namespace,
    paths: PreparedPaths,
    image_dir: Path,
    sparse_dir: Path,
    scene: ColmapScene,
    selected_indices: list[int],
) -> None:
    if paths.threedgrut_input_dir.exists() and not args.replace:
        print(f"Skipping prepare; found {paths.threedgrut_input_dir}", flush=True)
        return

    reset_prepared_scene(paths.threedgrut_input_dir)
    symlink_images(image_dir, paths.threedgrut_input_dir / "images", scene.images)
    write_sparse_model(sparse_dir, paths.threedgrut_input_dir / "sparse/0", scene)
    write_transforms_json(paths.threedgrut_input_dir / "nerfstudio/transforms.json", scene)
    write_selected_indices(paths, selected_indices, scene.images)


def write_selected_indices(paths: PreparedPaths, selected_indices: list[int], images: Sequence[Image]) -> None:
    selected_images = [image_basename(images[index]) for index in selected_indices]
    write_json(paths.selected_indices_path, selected_indices)
    (paths.scene_root / "selected_images.txt").write_text("\n".join(selected_images) + "\n")


def reconstruction_output_dir(paths: PreparedPaths) -> Path:
    return paths.reconstruction_run_dir / paths.scene_id / paths.scene_id


def reconstruction_checkpoint(paths: PreparedPaths, args: argparse.Namespace) -> Path:
    if args.reconstruction_checkpoint is not None:
        return args.reconstruction_checkpoint
    return (
        reconstruction_output_dir(paths) / f"ours_{args.reconstruction_steps}" / f"ckpt_{args.reconstruction_steps}.pt"
    )


def require_reconstruction_checkpoint(paths: PreparedPaths, args: argparse.Namespace) -> Path:
    checkpoint = reconstruction_checkpoint(paths, args)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing 3DGUT checkpoint: {checkpoint}")
    return checkpoint


def threedgrut_train_overrides(
    paths: PreparedPaths, args: argparse.Namespace, num_selected_views: int, num_images: int
) -> list[str]:
    if num_selected_views >= num_images:
        # selected_indices_file would leave 3DGRUT an empty validation split, which crashes
        # dataset construction; interval 0 uses every image for both splits (validation never runs).
        split_overrides = ["dataset.test_split_interval=0"]
    else:
        split_overrides = [f"selected_indices_file={paths.selected_indices_path}"]
    return [
        f"path={paths.threedgrut_input_dir}",
        f"out_dir={paths.reconstruction_run_dir}",
        *split_overrides,
        "test_last=False",
        "export_ingp.enabled=False",
        f"experiment_name={paths.scene_id}",
        f"n_iterations={args.reconstruction_steps}",
        f"checkpoint.iterations=[{args.reconstruction_steps}]",
    ]


def run_reconstruction(
    args: argparse.Namespace, paths: PreparedPaths, num_selected_views: int, num_images: int
) -> bool:
    checkpoint = reconstruction_checkpoint(paths, args)
    if args.reconstruction_checkpoint is not None:
        require_reconstruction_checkpoint(paths, args)
        print(f"Skipping reconstruction; using {checkpoint}", flush=True)
        return True
    if checkpoint.is_file() and not args.replace:
        print(f"Skipping reconstruction; found {checkpoint}", flush=True)
        return True
    paths.split_path.unlink(missing_ok=True)
    paths.render_result_path.unlink(missing_ok=True)
    reset_directory(paths.render_output_root)
    reset_directory(checkpoint.parent)
    train_3dgrut(
        DEFAULT_THREEDGRUT_CONFIG,
        threedgrut_train_overrides(paths, args, num_selected_views, num_images),
        DEFAULT_THREEDGRUT_CONFIG_DIR,
    )
    require_reconstruction_checkpoint(paths, args)
    return False


def render_reconstruction(args: argparse.Namespace, paths: PreparedPaths, checkpoint_reused: bool) -> Path:
    source_trajectory_path = paths.threedgrut_input_dir / "nerfstudio/transforms.json"
    checkpoint = require_reconstruction_checkpoint(paths, args)
    frame_count = render_frame_count(paths, None)
    if (
        not args.replace
        and checkpoint_reused
        and args.reconstruction_checkpoint is None
        and render_outputs_complete(
            paths.render_checkpoint_dir,
            frame_count,
            expected_selected_indices=selected_indices_for_render(paths.selected_indices_path, None),
        )
    ):
        print(f"Skipping render; found complete outputs in {paths.render_checkpoint_dir}", flush=True)
        return paths.render_checkpoint_dir

    paths.render_checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    return render_3dgrut_colmap(
        checkpoint=checkpoint,
        colmap_dir=paths.threedgrut_input_dir,
        output_root=paths.render_output_root,
        experiment_name=DEFAULT_RECON_SUBDIR,
        selected_indices=paths.selected_indices_path,
        trajectory_path=source_trajectory_path,
        trajectory_output_subdir="",
    )


def trajectory_paths(paths: PreparedPaths) -> PreparedTrajectory:
    root = paths.scene_root / "trajectory"
    return PreparedTrajectory(
        root=root,
        render_dataset_dir=root / "render_dataset",
        transforms_path=root / "transforms.json",
        render_trajectory_path=root / "trajectory.json",
        selected_indices_path=root / "selected_indices.json",
        target_indices_path=root / "target_indices.json",
        render_checkpoint_dir=paths.render_checkpoint_dir / "trajectory",
    )


def write_trajectory_render_dataset(trajectory: PreparedTrajectory, trajectory_data: dict[str, object]) -> None:
    first_frame = trajectory_data["frames"][0]
    write_json(
        trajectory.render_dataset_dir / "transforms.json",
        transforms_json(
            trajectory_data,
            [frame_with_camera_intrinsics(first_frame, camera_intrinsics_for_frame(trajectory_data, first_frame))],
        ),
    )


def write_trajectory_inputs(
    args: argparse.Namespace,
    paths: PreparedPaths,
    scene: ColmapScene,
    selected_indices: list[int],
) -> PreparedTrajectory:
    trajectory = trajectory_paths(paths)
    trajectory_data = read_camera_trajectory(args.trajectory_path)
    assert_target_only_trajectory(trajectory_data, str(args.trajectory_path))
    trajectory_frames = trajectory_data["frames"]

    target_frames = [
        frame_with_camera_intrinsics(frame, camera_intrinsics_for_frame(trajectory_data, frame))
        for frame in trajectory_frames
    ]
    selected_start = len(target_frames)
    source_intrinsics = shared_camera_intrinsics(scene)
    selected_frames = [
        frame_with_camera_intrinsics(source_frame(scene.images[index]), source_intrinsics) for index in selected_indices
    ]

    reset_directory(trajectory.root)
    write_json(
        trajectory.transforms_path,
        transforms_json(trajectory_data, target_frames + selected_frames),
    )
    write_json(
        trajectory.render_trajectory_path,
        transforms_json(trajectory_data, target_frames),
    )
    write_trajectory_render_dataset(trajectory, trajectory_data)
    write_json(trajectory.selected_indices_path, list(range(selected_start, selected_start + len(selected_indices))))
    write_json(trajectory.target_indices_path, list(range(len(target_frames))))
    return trajectory


def render_trajectory(
    args: argparse.Namespace,
    paths: PreparedPaths,
    scene: ColmapScene,
    selected_indices: list[int],
) -> PreparedTrajectory:
    checkpoint = require_reconstruction_checkpoint(paths, args)
    trajectory = write_trajectory_inputs(args, paths, scene, selected_indices)
    output_dir = render_3dgrut_colmap(
        checkpoint=checkpoint,
        colmap_dir=paths.threedgrut_input_dir,
        output_root=paths.render_output_root,
        experiment_name=DEFAULT_RECON_SUBDIR,
        selected_indices=trajectory.selected_indices_path,
        render_dataset_dir=trajectory.render_dataset_dir,
        trajectory_path=trajectory.render_trajectory_path,
    )
    return replace(trajectory, render_checkpoint_dir=output_dir)


def run_metric_alignment(args: argparse.Namespace, paths: PreparedPaths) -> None:
    scale_info = paths.scale_dir / "scale_info.txt"
    if args.metric_scale is not None:
        if scale_info.is_file() and not args.replace:
            existing_scale = read_metric_scale(scale_info)
            assert np.isclose(existing_scale, args.metric_scale), (
                f"Existing metric scale at {scale_info} is {existing_scale}, "
                f"but --metric_scale={args.metric_scale}. Re-run with --replace."
            )
            print(f"Skipping metric alignment; found {scale_info}", flush=True)
            return
        paths.scale_dir.mkdir(parents=True, exist_ok=True)
        scale_info.write_text(f"Scale factor: {args.metric_scale}\n")
        return

    if scale_info.is_file() and not args.replace:
        print(f"Skipping metric alignment; found {scale_info}", flush=True)
        return

    # Align against the prepared model: its intrinsics/keypoints are rescaled to the
    # actual image resolution and its image names match the flat prepared image dir,
    # unlike the original COLMAP model in args.colmap_dir.
    colmap_dir = paths.threedgrut_input_dir / "sparse/0"
    image_dir = paths.threedgrut_input_dir / "images"
    paths.scale_dir.mkdir(parents=True, exist_ok=True)
    scale, _, _ = align_colmap_to_metric_scale(
        colmap_dir=colmap_dir,
        image_dir=image_dir,
        output_dir=paths.scale_dir,
        debug=False,
        downsample_factor=1,
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Metric alignment returned a scale that is not finite and positive")
    scale_info.write_text(f"Scale factor: {scale}\n")


def generate_caption(args: argparse.Namespace, paths: PreparedPaths) -> None:
    caption_path = paths.prompt_dir / paths.scene_id / paths.prompt_filename
    prompt_format = args.prompt_format
    if caption_path.is_file() and not args.replace:
        validate_prompt_file_format(caption_path, prompt_format, expected_fps=args.dataset_fps)
        print(f"Skipping captioning; found {caption_path}", flush=True)
        return

    generate_caption_hdf5(
        input_path=paths.eval_scene_dir,
        output_path=caption_path,
        dataset_downsample_factor=1,
        dataset_fps=args.dataset_fps,
        prompt_format=prompt_format,
        captioning_model_id=args.captioning_model_id,
        prompt_model_id=(
            args.text_encoder_model_id
            if prompt_format == WAN_T5_PROMPT_FORMAT
            else default_prompt_model_id(prompt_format)
        ),
    )


def read_metric_scale(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("Scale factor:"):
            scale = float(line.split(":", 1)[1].split()[0])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Metric scale in {path} must be finite and positive")
            return scale
    assert False, f"Could not find scale factor in {path}"


def metric_scale(args: argparse.Namespace, paths: PreparedPaths) -> float | None:
    if args.metric_scale is not None:
        return args.metric_scale
    scale_info = paths.scale_dir / "scale_info.txt"
    return read_metric_scale(scale_info) if scale_info.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--colmap_dir",
        type=Path,
        required=True,
        help="Input COLMAP scene directory containing images/ (or a single downsampled images_<factor>/) and "
        "sparse/0/ or colmap/sparse/0/ with cameras.bin, images.bin, and points3D.bin. An extracted DL3DV scene "
        "with its DL3DV-ALL-ColmapCache archive extracted alongside matches this layout.",
    )
    parser.add_argument(
        "--pyramid_factor",
        type=int,
        default=None,
        help="Known integer image-pyramid reduction factor. Defaults to the selected images_<factor>/ suffix. "
        "Set this for downsampled images stored in images/; without a factor, only uniform resizing is accepted.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Scene output directory. The directory name is used as the scene id.",
    )
    parser.add_argument(
        "--selected_image_names_file",
        type=Path,
        default=None,
        help="Optional newline-delimited file of image filenames to use as 3DGRUT training views. "
        "Defaults to all COLMAP images.",
    )
    parser.add_argument(
        "--trajectory_path",
        type=Path,
        default=None,
        help="Optional transforms-style JSON camera trajectory to render and register for ArtiFixer inference. "
        "The JSON must include camera intrinsics and frames with 4x4 camera-to-world matrices.",
    )
    parser.add_argument(
        "--phases",
        default="prepare,reconstruct,render,scale,caption",
        help="Comma-separated phases to run. Valid phases: prepare,reconstruct,render,scale,caption.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        default=False,
        help="Overwrite existing prepared files, reconstruction renders, metric scale, and caption outputs.",
    )
    parser.add_argument(
        "--text_encoder_model_id",
        default=DEFAULT_TEXT_ENCODER_MODEL_ID,
        help="Wan text encoder model id used to store caption embeddings for ArtiFixer "
        "inference; accepts a local snapshot path for offline nodes.",
    )
    parser.add_argument(
        "--captioning_model_id",
        default=DEFAULT_CAPTIONING_MODEL_ID,
        help="Vision-language model that writes the scene caption. The default needs ~60 GB of "
        "GPU memory in bf16; Qwen/Qwen3-VL-4B-Instruct fits a 24 GB GPU.",
    )
    parser.add_argument("--prompt_format", choices=PROMPT_FORMAT_CHOICES, default=WAN_T5_PROMPT_FORMAT)
    parser.add_argument(
        "--dataset_fps", type=float, default=DL3DV_DATASET_FPS, help="Source capture FPS used in caption metadata."
    )
    parser.add_argument(
        "--reconstruction_steps",
        type=int,
        default=10000,
        help="3DGRUT MCMC training iterations and checkpoint step.",
    )
    parser.add_argument(
        "--reconstruction_checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional existing 3DGRUT checkpoint. When provided, reconstruction is skipped "
            "and this checkpoint is rendered."
        ),
    )
    parser.add_argument(
        "--metric_scale",
        type=float,
        default=None,
        help="Optional precomputed metric scale. When omitted, the scale phase runs MoGe alignment.",
    )

    return parser


def prepare_colmap_scene(args: argparse.Namespace) -> None:
    phases = parse_phases(args.phases, {"prepare", "reconstruct", "render", "scale", "caption"})
    if args.metric_scale is not None and (not np.isfinite(args.metric_scale) or args.metric_scale <= 0):
        raise ValueError("--metric_scale must be finite and positive")

    args.colmap_dir = args.colmap_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.selected_image_names_file is not None:
        args.selected_image_names_file = args.selected_image_names_file.expanduser().resolve()
    if args.reconstruction_checkpoint is not None:
        args.reconstruction_checkpoint = args.reconstruction_checkpoint.expanduser().resolve()
    if args.trajectory_path is not None:
        args.trajectory_path = args.trajectory_path.expanduser().resolve()
        assert "render" in phases, "--trajectory_path requires the render phase"
    image_dir, sparse_dir = resolve_colmap_paths(args.colmap_dir)
    pyramid_factor = resolve_pyramid_factor(image_dir, args.pyramid_factor)
    scene = read_colmap_scene(sparse_dir)
    require_unique_basenames(scene.images)
    scene = scale_colmap_scene_to_images(image_dir, scene, pyramid_factor=pyramid_factor)

    if not np.isfinite(args.dataset_fps) or args.dataset_fps <= 0:
        raise ValueError("--dataset_fps must be finite and positive")
    paths = prepared_paths(args.output_root, args.output_root.name, args.reconstruction_steps, args.prompt_format)
    scale_info = paths.scale_dir / "scale_info.txt"
    if (
        scale_info.is_file()
        and not ("scale" in phases and args.replace)
        and (args.metric_scale is None or "scale" in phases)
    ):
        read_metric_scale(scale_info)
    paths.scene_root.mkdir(parents=True, exist_ok=True)
    replacing_prepared = "prepare" in phases and args.replace
    if paths.threedgrut_input_dir.exists() and not replacing_prepared:
        scene, selected_indices = reuse_prepared_scene(args, paths, scene)
    else:
        selected_indices = resolve_selected_indices(args, scene.images)
    if replacing_prepared and paths.threedgrut_input_dir.exists():
        if (
            phases != {"prepare", "reconstruct", "render", "scale", "caption"}
            or args.reconstruction_checkpoint is not None
        ):
            raise ValueError(
                "Replacing prepared inputs requires all phases together with a new reconstruction; "
                "use a new --output_root to reuse an external checkpoint"
            )
        paths.split_path.unlink(missing_ok=True)
        paths.render_result_path.unlink(missing_ok=True)
        # Discard results before changing their inputs. If a later phase is
        # interrupted, a normal rerun cannot reuse the previous reconstruction.
        for directory in (
            paths.reconstruction_run_dir,
            paths.recon_results_dir,
            paths.scale_dir,
            paths.prompt_dir,
            trajectory_paths(paths).root,
        ):
            reset_directory(directory)

    if "prepare" in phases:
        prepare_files(args, paths, image_dir, sparse_dir, scene, selected_indices)
    if {"reconstruct", "render"} & phases:
        assert (
            paths.selected_indices_path.is_file()
        ), f"Missing prepared selected indices: {paths.selected_indices_path}"

    checkpoint_reused = True
    if "reconstruct" in phases:
        checkpoint_reused = run_reconstruction(args, paths, len(selected_indices), len(scene.images))
    checkpoint = reconstruction_checkpoint(paths, args)
    if "render" in phases:
        paths.split_path.unlink(missing_ok=True)
        paths.render_result_path.unlink(missing_ok=True)
        if args.trajectory_path is None:
            paths = replace(paths, render_checkpoint_dir=render_reconstruction(args, paths, checkpoint_reused))
            trajectory = None
        else:
            trajectory = render_trajectory(args, paths, scene, selected_indices)
        save_render_result(paths, trajectory, checkpoint)
    elif not paths.split_path.is_file() and paths.render_result_path.is_file():
        paths, trajectory, checkpoint = reuse_render_result(paths)
        if args.reconstruction_checkpoint is not None and args.reconstruction_checkpoint != checkpoint:
            raise ValueError("The checkpoint differs from the saved render; rerun the render phase")
    else:
        trajectory = None
    if "scale" in phases:
        run_metric_alignment(args, paths)
    scale = metric_scale(args, paths)
    if "caption" in phases:
        generate_caption(args, paths)
    render_dir = render_checkpoint_dir(paths, trajectory)
    preserve_split = paths.split_path.is_file() and "render" not in phases
    if preserve_split or (
        paths.render_result_path.is_file()
        and scale is not None
        and render_outputs_complete(render_dir, render_frame_count(paths, trajectory))
    ):
        prompt_path = paths.prompt_dir / paths.scene_id / paths.prompt_filename
        validate_prompt_file_format(prompt_path, args.prompt_format, expected_fps=args.dataset_fps)
        if preserve_split:
            # Phase-only reruns retain the rendered trajectory, selection and
            # reconstruction identity; only prompt selection and scale change.
            split_data = json.loads(paths.split_path.read_text())
            entry = split_data["test"][paths.scene_id]
            checkpoint = resolve_required_path(paths.split_path.parent, entry, "reconstruction_checkpoint")
            entry["prompt_path"] = path_for_split(paths.split_path, prompt_path)
            if scale is not None:
                entry["camera_scale"] = scale * DEFAULT_CAMERA_CONDITIONING_SCALE
            with atomic_output_path(paths.split_path) as temporary:
                write_json(temporary, split_data)
        else:
            write_eval_split(paths, scale, trajectory, checkpoint)

    print(f"prepared_scene={paths.scene_id}", flush=True)
    print(f"scene_root={paths.scene_root}", flush=True)
    print(f"selected_views={len(selected_indices)}", flush=True)
    if paths.split_path.is_file():
        print(f"split_path={paths.split_path}", flush=True)
    else:
        print("split_path=<not written; complete render, scale, and caption phases>", flush=True)
    print(f"eval_dataset_root={paths.eval_dataset_root}", flush=True)
    print(f"prompt_dir={paths.prompt_dir}", flush=True)
    print(f"recon_results_dir={paths.recon_results_dir}", flush=True)
    print(f"reconstruction_checkpoint={checkpoint}", flush=True)
    if scale is not None:
        print(f"metric_scale={scale}", flush=True)
        print(f"camera_scale={scale * DEFAULT_CAMERA_CONDITIONING_SCALE}", flush=True)
    else:
        print("metric_scale=<missing>", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    prepare_colmap_scene(args)


if __name__ == "__main__":
    main()
