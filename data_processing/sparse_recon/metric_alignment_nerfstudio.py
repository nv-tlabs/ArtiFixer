# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metric alignment for nerfstudio datasets without images.bin.

Uses transforms.json for poses and projects all 3D points to images
instead of relying on 2D-3D correspondences from images.bin.

Usage:
    python \
        metric_alignment_nerfstudio.py \
        --scene_dir /path/to/scene/nerfstudio \
        --output_dir ./output
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from data_processing.sparse_recon.metric_alignment import (
    MoGeModelWrapper,
    debug_report,
    extract_monodepth_depths,
    read_colmap_points3D_binary,
    solve_scale_factor,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def read_nerfstudio_transforms(transforms_path):
    """Read camera intrinsics and poses from nerfstudio transforms.json."""
    with open(transforms_path) as f:
        data = json.load(f)

    # Extract camera intrinsics
    intrinsics = {
        "w": data["w"],
        "h": data["h"],
        "fl_x": data["fl_x"],
        "fl_y": data["fl_y"],
        "cx": data["cx"],
        "cy": data["cy"],
    }

    # Extract applied_transform if present (used to transform COLMAP points to match poses)
    applied_transform = None
    if "applied_transform" in data:
        applied_transform = np.array(data["applied_transform"])
        # Ensure it's 4x4
        if applied_transform.shape == (3, 4):
            full_transform = np.eye(4)
            full_transform[:3, :] = applied_transform
            applied_transform = full_transform

    # Extract frames with poses
    frames = []
    for frame in data["frames"]:
        c2w = np.array(frame["transform_matrix"])

        # Compute W2C and apply transforms (following 3dgrut convention)
        w2c = np.zeros((4, 4), dtype=np.float32)
        w2c[:3, :3] = c2w[:3, :3].T
        w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]
        w2c[3, 3] = 1.0

        # Apply the inverse applied_transform to "undo" nerfstudio's transform
        if applied_transform is not None:
            w2c = w2c @ applied_transform

        # Convert from OpenGL (Y-up, -Z forward) to OpenCV (Y-down, +Z forward)
        w2c[1:3, :] *= -1

        frames.append(
            {
                "file_path": frame["file_path"],
                "c2w": np.linalg.inv(w2c),
                "w2c": w2c,
                "R": w2c[:3, :3],
                "t": w2c[:3, 3],
            }
        )

    return intrinsics, frames


def project_points_to_frame(points3D, frame, intrinsics):
    """Project 3D points to image using frame pose and intrinsics."""
    R = frame["R"]
    t = frame["t"].reshape(3, 1)

    # Transform to camera coordinates
    points_cam = R @ points3D.T + t  # 3 x N

    # Filter points behind camera
    valid_depth = points_cam[2] > 0.1  # minimum depth threshold

    if valid_depth.sum() == 0:
        return np.array([]), np.array([]), np.array([])

    points_cam_valid = points_cam[:, valid_depth]
    depths = points_cam_valid[2]

    # Project to image
    fx, fy = intrinsics["fl_x"], intrinsics["fl_y"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    x = points_cam_valid[0] / points_cam_valid[2] * fx + cx
    y = points_cam_valid[1] / points_cam_valid[2] * fy + cy

    # Filter points outside image bounds
    w, h = intrinsics["w"], intrinsics["h"]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    return x[valid], y[valid], depths[valid]


def filter_occluded_points(x, y, depths, intrinsics, cell_size=8, depth_tolerance=1.1):
    """Approximate z-buffer visibility test for projected points.

    Keeps only points whose depth is close to the minimum projected depth within
    their local pixel cell; points well behind that minimum are likely occluded
    and must not be paired with monocular depth of the occluder.
    """
    grid_w = (int(intrinsics["w"]) + cell_size - 1) // cell_size
    grid_h = (int(intrinsics["h"]) + cell_size - 1) // cell_size
    cells = (y.astype(int) // cell_size) * grid_w + (x.astype(int) // cell_size)
    min_depths = np.full(grid_w * grid_h, np.inf)
    np.minimum.at(min_depths, cells, depths)
    return depths <= min_depths[cells] * depth_tolerance


def extract_projected_depths(frames, intrinsics, points3D, image_dir):
    """Extract COLMAP depths by projecting all 3D points to each frame."""
    correspondences = []

    for frame in frames:
        # Get image path
        file_path = frame["file_path"]
        if file_path.startswith("images/"):
            image_name = file_path.split("/")[-1]
        else:
            image_name = file_path

        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        # Project all 3D points to this frame and drop occluded ones
        x, y, depths_colmap = project_points_to_frame(points3D, frame, intrinsics)

        if len(depths_colmap) == 0:
            continue

        visible = filter_occluded_points(x, y, depths_colmap, intrinsics)
        x, y, depths_colmap = x[visible], y[visible], depths_colmap[visible]

        if len(depths_colmap) == 0:
            continue

        xys = np.stack([x, y], axis=1)

        correspondences.append(
            {
                "image_name": image_name,
                "xys": xys,
                "depths_colmap": depths_colmap,
                "image_path": image_path,
                "camera": {
                    "width": intrinsics["w"],
                    "height": intrinsics["h"],
                    "params": np.array([intrinsics["fl_x"], intrinsics["fl_y"], intrinsics["cx"], intrinsics["cy"]]),
                },
            }
        )

    return correspondences


def align_nerfstudio_to_metric_scale(
    scene_dir,
    output_dir=None,
    num_images=None,
    debug=True,
    downsample_factor=1.0,
):
    """
    Align nerfstudio scene to metric scale.

    Args:
        scene_dir: Path to nerfstudio scene directory (contains transforms.json)
        output_dir: Output directory for results
        num_images: Limit number of images to process
        debug: Enable debug visualizations
        downsample_factor: Image downsample factor
    """
    scene_dir = Path(scene_dir)

    # Find transforms.json
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found in {scene_dir}")

    # Find colmap directory for points3D.bin
    colmap_dir = scene_dir / "colmap" / "sparse" / "0"
    if not colmap_dir.exists():
        colmap_dir = scene_dir / "colmap"

    points3D_path = colmap_dir / "points3D.bin"
    if not points3D_path.exists():
        raise FileNotFoundError(f"points3D.bin not found in {colmap_dir}")

    # Determine image directory
    if downsample_factor == 1.0:
        image_dir = scene_dir / "images"
    else:
        image_dir = scene_dir / f"images_{int(downsample_factor)}"

    if not image_dir.exists():
        # Try parent directory
        image_dir = (
            scene_dir.parent / "images"
            if downsample_factor == 1.0
            else scene_dir.parent / f"images_{int(downsample_factor)}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Load data
    logger.info("Loading nerfstudio transforms...")
    intrinsics, frames = read_nerfstudio_transforms(transforms_path)
    logger.info(f"Loaded {len(frames)} frames")
    logger.info(f"Original intrinsics: {intrinsics['w']}x{intrinsics['h']}, fx={intrinsics['fl_x']:.1f}")

    # Scale intrinsics for downsampled images
    if downsample_factor != 1.0:
        intrinsics["w"] = int(intrinsics["w"] / downsample_factor)
        intrinsics["h"] = int(intrinsics["h"] / downsample_factor)
        intrinsics["fl_x"] = intrinsics["fl_x"] / downsample_factor
        intrinsics["fl_y"] = intrinsics["fl_y"] / downsample_factor
        intrinsics["cx"] = intrinsics["cx"] / downsample_factor
        intrinsics["cy"] = intrinsics["cy"] / downsample_factor
        logger.info(f"Scaled intrinsics: {intrinsics['w']}x{intrinsics['h']}, fx={intrinsics['fl_x']:.1f}")

    logger.info("Loading COLMAP points3D...")
    point_ids, points_xyz, points_rgb, points_error = read_colmap_points3D_binary(str(points3D_path))
    logger.info(f"Loaded {len(points_xyz)} 3D points")

    if num_images is not None:
        frames = frames[:num_images]

    # Extract projected depths
    logger.info(f"Projecting 3D points to {len(frames)} frames...")
    correspondences = extract_projected_depths(frames, intrinsics, points_xyz, str(image_dir))
    logger.info(f"Found {len(correspondences)} frames with valid projections")

    # Run monodepth. xys are already in downsampled image coordinates (the
    # intrinsics were scaled above), so no further coordinate scaling is needed.
    logger.info("\nRunning monodepth...")
    monodepth_model = MoGeModelWrapper()
    correspondences = extract_monodepth_depths(correspondences, monodepth_model, downsample_factor=1.0)

    # Filter out inf/nan values from metric depths
    for corr in correspondences:
        if corr.get("depths_metric") is not None:
            valid = np.isfinite(corr["depths_metric"])
            corr["depths_metric"] = corr["depths_metric"][valid]
            corr["depths_colmap"] = corr["depths_colmap"][valid]
            corr["xys"] = corr["xys"][valid]
            if corr.get("mask") is not None:
                corr["mask"] = corr["mask"][valid]

    # Solve for scale
    logger.info("\nSolving for scale factor...")
    scale, stats = solve_scale_factor(correspondences)

    logger.info("\nResults:")
    logger.info(f"  Scale factor: {scale:.6f}")
    logger.info(f"  RMSE: {stats['rmse']:.4f} (weighted: {stats['weighted_rmse']:.4f})")
    logger.info(f"  Mean error: {stats['mean_error']:.4f} (weighted: {stats['weighted_mean_error']:.4f})")
    logger.info(f"  Median error: {stats['median_error']:.4f}")
    logger.info(f"  Correspondences: {stats['num_correspondences']}")

    if debug and output_dir:
        debug_dir = Path(output_dir) / "debug_visualizations"
        logger.info(f"\nWriting debug report to {debug_dir}")
        # xys are already in downsampled coordinates (intrinsics were scaled above), so downsample_factor=1.0.
        debug_report(correspondences, scale, stats, debug_dir, num_samples=10)

    # Save scale info
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "scale_info.txt", "w") as f:
            f.write(f"Scale factor: {scale}\n")
            f.write(f"RMSE: {stats['rmse']:.6f}\n")
            f.write(f"Median error: {stats['median_error']:.6f}\n")
            f.write(f"Num correspondences: {stats['num_correspondences']}\n")

    return scale, stats, correspondences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric alignment for nerfstudio datasets (without images.bin)")
    parser.add_argument(
        "--scene_dir", required=True, help="Path to nerfstudio scene directory (contains transforms.json)"
    )
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--num_images", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug visualizations")
    parser.add_argument("--downsample_factor", type=float, default=1.0, help="Image downsample factor")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join("./debug_nerfstudio", Path(args.scene_dir).name)

    scale, stats, correspondences = align_nerfstudio_to_metric_scale(
        args.scene_dir,
        args.output_dir,
        args.num_images,
        debug=not args.no_debug,
        downsample_factor=args.downsample_factor,
    )
