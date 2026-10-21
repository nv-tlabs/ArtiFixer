# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import struct
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from threedgrut.datasets.utils import qvec_to_so3, read_colmap_extrinsics_binary, read_colmap_intrinsics_binary

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class MoGeModelWrapper:
    def __init__(self, model_variant="Ruicheng/moge-2-vitl-normal", device="cuda"):
        if "MOGE_MODEL_PATH" in os.environ and Path(os.environ["MOGE_MODEL_PATH"]).exists():
            local_model_path = os.environ["MOGE_MODEL_PATH"]
            logger.info(f"Loading MoGe from local path: {local_model_path}")
            model = MoGeModel.from_pretrained(local_model_path).to(device)
        else:
            logger.info(f"Loading MoGe from HuggingFace: {model_variant}")
            model = MoGeModel.from_pretrained(model_variant).to(device)
        self.model = model

    @torch.no_grad()
    def __call__(self, rgb_origin):
        output = self.model.infer(rgb_origin)
        sky_mask = output["depth"] == torch.inf
        return output["depth"], output["mask"] & ~sky_mask, output["intrinsics"]


def read_colmap_points3D_binary(path_to_model_file):
    # Parse the whole file from memory; per-point file reads are far too slow for
    # the 10^5-10^6 point models this runs on.
    with open(path_to_model_file, "rb") as fid:
        data = memoryview(fid.read())

    num_points = struct.unpack_from("<Q", data, 0)[0]
    offset = 8
    point_ids = np.empty(num_points, dtype=np.int64)
    xyzs = np.empty((num_points, 3), dtype=np.float64)
    rgbs = np.empty((num_points, 3), dtype=np.int32)
    errors = np.empty(num_points, dtype=np.float64)
    for i in range(num_points):
        point_properties = struct.unpack_from("<QdddBBBd", data, offset)
        offset += 43
        point_ids[i] = point_properties[0]
        xyzs[i] = point_properties[1:4]
        rgbs[i] = point_properties[4:7]
        errors[i] = point_properties[7]
        track_length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8 + 8 * track_length

    return point_ids, xyzs, rgbs, errors.reshape(-1, 1)


def extract_colmap_depths(images, cameras, point3D_ids, points3D_xyz, image_dir):
    """Extract COLMAP depths for visible 3D points in each image."""
    # Sort the point ids once so each image's observations can be matched with searchsorted.
    sort_order = np.argsort(point3D_ids)
    sorted_point_ids = point3D_ids[sort_order]

    correspondences = []
    if len(sorted_point_ids) == 0:
        return correspondences

    for img in images:
        # Get valid 3D point observations
        valid_mask = img.point3D_ids != -1
        if valid_mask.sum() == 0:
            continue

        img_point3D_ids = img.point3D_ids[valid_mask]
        xys = img.xys[valid_mask]

        # Match observations against the model's point ids
        positions = np.searchsorted(sorted_point_ids, img_point3D_ids)
        positions = np.clip(positions, 0, len(sorted_point_ids) - 1)
        found = sorted_point_ids[positions] == img_point3D_ids

        if not found.any():
            continue

        points = points3D_xyz[sort_order[positions[found]]]
        xys = xys[found]

        # Compute depths
        R = qvec_to_so3(img.qvec)
        t = img.tvec.reshape(3, 1)
        points_cam = R @ points.T + t
        depths = points_cam[2]

        # Store correspondences
        camera = cameras[img.camera_id]
        correspondences.append(
            {
                "image_name": img.name,
                "xys": xys,
                "depths_colmap": depths,
                "camera": camera,
                "image_path": os.path.join(image_dir, img.name),
            }
        )

    return correspondences


def extract_monodepth_depths(correspondences, monodepth_model, downsample_factor=1.0):
    """Extract Monodepth depth predictions at COLMAP point locations."""
    for corr in correspondences:
        image_path = corr["image_path"]
        if not os.path.exists(image_path):
            corr["depths_metric"] = None
            corr["mask"] = None
            continue

        input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)

        pred_depth, mask, _ = monodepth_model(input_image)
        pred_depth = pred_depth.cpu().numpy()
        mask = mask.cpu().numpy()

        # Squeeze mask if it has extra dimensions
        if mask.ndim > 2:
            mask = mask.squeeze()

        # Sample depths and mask at point locations (scale coordinates to match downsampled image)
        xys = corr["xys"] / downsample_factor
        depth_h, depth_w = pred_depth.shape
        x_idx = np.clip(xys[:, 0].astype(int), 0, depth_w - 1)
        y_idx = np.clip(xys[:, 1].astype(int), 0, depth_h - 1)

        depths_metric = pred_depth[y_idx, x_idx]
        mask_sampled = mask[y_idx, x_idx]

        corr["depths_metric"] = depths_metric
        corr["mask"] = mask_sampled

    return correspondences


def _concat_correspondence_depths(correspondences):
    """Concatenate COLMAP depths, metric depths, and masks from every correspondence that
    has metric depths. A correspondence without a mask contributes an all-ones mask."""
    valid = [corr for corr in correspondences if corr["depths_metric"] is not None]
    colmap_depths = np.concatenate([corr["depths_colmap"] for corr in valid])
    metric_depths = np.concatenate([corr["depths_metric"] for corr in valid])
    masks = np.concatenate(
        [corr["mask"] if corr.get("mask") is not None else np.ones_like(corr["depths_colmap"]) for corr in valid]
    )
    return colmap_depths, metric_depths, masks


def solve_scale_factor(correspondences):
    """Solve for scale factor: metric_depth = scale * colmap_depth using mask-weighted L1 loss.

    Args:
        correspondences: List of correspondence dictionaries
    """
    colmap_depths, metric_depths, masks = _concat_correspondence_depths(correspondences)

    # Filter outliers
    valid = (metric_depths > 0) & (colmap_depths > 0)
    colmap_depths = colmap_depths[valid]
    metric_depths = metric_depths[valid]
    masks = masks[valid]

    # Normalize masks to have mean 1 (so they act as weights)
    if masks.sum() > 0:
        masks = masks / masks.mean()
    else:
        masks = np.ones_like(colmap_depths)

    # Mask-weighted L1 loss optimization
    # For L1 loss, the optimal scale is the weighted median of depth ratios
    ratios = metric_depths / colmap_depths

    # Compute weighted median
    sorted_idx = np.argsort(ratios)
    sorted_ratios = ratios[sorted_idx]
    sorted_weights = masks[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2.0)
    scale = sorted_ratios[median_idx]

    # Compute statistics
    scaled_colmap = scale * colmap_depths
    errors = np.abs(scaled_colmap - metric_depths)

    # Compute both weighted and unweighted statistics
    weighted_mean_error = (masks * errors).sum() / masks.sum()
    weighted_rmse = np.sqrt((masks * errors**2).sum() / masks.sum())

    stats = {
        "scale": scale,
        "mean_error": errors.mean(),
        "weighted_mean_error": weighted_mean_error,
        "median_error": np.median(errors),
        "rmse": np.sqrt((errors**2).mean()),
        "weighted_rmse": weighted_rmse,
        "num_correspondences": len(errors),
        "mean_mask": masks.mean(),
        "median_mask": np.median(masks),
    }

    return scale, stats


def debug_report(correspondences, scale, stats, output_dir, num_samples=5, downsample_factor=1.0):
    """Save per-image depth comparisons, a correlation/residual plot, and sanity-check logs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = [corr for corr in correspondences if corr["depths_metric"] is not None]

    for idx, corr in enumerate(valid[:num_samples]):
        rgb = cv2.imread(corr["image_path"])[:, :, ::-1]
        xys = corr["xys"] / downsample_factor
        panels = (
            ("COLMAP depth", corr["depths_colmap"], {"cmap": "plasma"}),
            ("Metric depth", corr["depths_metric"], {"cmap": "plasma"}),
            (
                "Metric - scaled COLMAP",
                corr["depths_metric"] - scale * corr["depths_colmap"],
                {"cmap": "RdBu", "vmin": -5, "vmax": 5},
            ),
        )
        fig, axes = plt.subplots(1, len(panels) + 1, figsize=(6 * (len(panels) + 1), 6))
        axes[0].imshow(rgb)
        axes[0].scatter(xys[:, 0], xys[:, 1], c="red", s=1, alpha=0.5)
        axes[0].set_title(corr["image_name"])
        axes[0].axis("off")
        for ax, (title, values, kwargs) in zip(axes[1:], panels):
            points = ax.scatter(xys[:, 0], xys[:, 1], c=values, s=10, **kwargs)
            ax.set(title=title, xlim=(0, rgb.shape[1]), ylim=(rgb.shape[0], 0))
            fig.colorbar(points, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"depth_comparison_{idx:03d}.png", dpi=150)
        plt.close(fig)

    colmap_depths, metric_depths, masks = _concat_correspondence_depths(correspondences)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    points = axes[0].scatter(colmap_depths, metric_depths, c=masks, cmap="viridis", alpha=0.3, s=1, vmin=0, vmax=1)
    axes[0].axline((0, 0), slope=scale, color="r", label=f"Fitted scale={scale:.3f}")
    axes[0].set(xlabel="COLMAP depth", ylabel="Metric depth", title="Depth correlation (colored by mask)")
    axes[0].legend()
    fig.colorbar(points, ax=axes[0], label="Mask")
    residuals = metric_depths - scale * colmap_depths
    points = axes[1].scatter(colmap_depths, residuals, c=masks, cmap="viridis", alpha=0.3, s=1, vmin=0, vmax=1)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set(xlabel="COLMAP depth", ylabel="Residual (metric - scaled COLMAP)", title="Residuals vs depth")
    fig.colorbar(points, ax=axes[1], label="Mask")
    fig.tight_layout()
    fig.savefig(output_dir / "depth_correlation.png", dpi=150)
    plt.close(fig)

    coverage = len(valid) / len(correspondences) * 100 if correspondences else 0.0
    checks = (
        (0.1 <= scale <= 100, f"scale factor: {scale:.4f}"),
        (len(colmap_depths) >= 1000, f"correspondence count: {len(colmap_depths)}"),
        (stats["median_error"] <= 1.0, f"median error: {stats['median_error']:.3f}m"),
        (coverage >= 50, f"image coverage: {len(valid)}/{len(correspondences)} ({coverage:.1f}%)"),
    )
    for ok, message in checks:
        logger.info(f"{'OK' if ok else 'WARNING'}: {message}")
    logger.info(
        f"Depth ranges: COLMAP [{colmap_depths.min():.2f}, {colmap_depths.max():.2f}], "
        f"metric [{metric_depths.min():.2f}, {metric_depths.max():.2f}], "
        f"scaled COLMAP [{(colmap_depths * scale).min():.2f}, {(colmap_depths * scale).max():.2f}]"
    )


def align_colmap_to_metric_scale(
    colmap_dir,
    image_dir,
    output_dir=None,
    num_images=None,
    debug=True,
    downsample_factor=1.0,
):
    """Main function to align COLMAP reconstruction to metric scale.

    Args:
        colmap_dir: Path to COLMAP sparse reconstruction directory
        image_dir: Path to images directory
        output_dir: Output directory for scaled reconstruction
        num_images: Limit number of images to process (for testing)
        debug: Enable debug visualizations
        downsample_factor: Factor by which images are downsampled (e.g., 4.0 for images_4)
    """

    # Load COLMAP data
    logger.info("Loading COLMAP data...")

    model_dir = Path(colmap_dir)
    required_files = ("points3D.bin", "cameras.bin", "images.bin")
    if not all((model_dir / name).exists() for name in required_files) and model_dir.parts[-3:] == (
        "colmap",
        "sparse",
        "0",
    ):
        # Some scenes ship the model under <scene>/nerfstudio/colmap/sparse/0 instead of <scene>/colmap/sparse/0.
        fallback_dir = model_dir.parents[2] / "nerfstudio" / "colmap" / "sparse" / "0"
        if all((fallback_dir / name).exists() for name in required_files):
            model_dir = fallback_dir

    point_ids, points_xyz, points_rgb, points_error = read_colmap_points3D_binary(str(model_dir / "points3D.bin"))
    cameras = read_colmap_intrinsics_binary(str(model_dir / "cameras.bin"))
    images = read_colmap_extrinsics_binary(str(model_dir / "images.bin"))

    if num_images is not None:
        images = images[:num_images]

    # Extract COLMAP depths
    logger.info(f"Extracting COLMAP depths from {len(images)} images...")
    correspondences = extract_colmap_depths(images, cameras, point_ids, points_xyz, image_dir)
    logger.info(f"Found {len(correspondences)} images with valid points")

    # Run Monodepth
    logger.info("\nRunning Monodepth...")
    monodepth_model = MoGeModelWrapper()
    correspondences = extract_monodepth_depths(correspondences, monodepth_model, downsample_factor=downsample_factor)

    # Solve for scale
    logger.info("\nSolving for scale factor (mask-weighted L1)...")
    scale, stats = solve_scale_factor(correspondences)

    logger.info("\nResults:")
    logger.info(f"  Scale factor: {scale:.6f}")
    logger.info(f"  RMSE: {stats['rmse']:.4f} (weighted: {stats['weighted_rmse']:.4f})")
    logger.info(f"  Mean error: {stats['mean_error']:.4f} (weighted: {stats['weighted_mean_error']:.4f})")
    logger.info(f"  Median error: {stats['median_error']:.4f}")
    logger.info(f"  Correspondences: {stats['num_correspondences']}")
    logger.info(f"  Mean mask: {stats['mean_mask']:.4f}")
    logger.info(f"  Median mask: {stats['median_mask']:.4f}")

    if debug and output_dir:
        debug_dir = Path(output_dir) / "debug_visualizations"
        logger.info(f"\nWriting debug report to {debug_dir}")
        debug_report(correspondences, scale, stats, debug_dir, num_samples=10, downsample_factor=downsample_factor)

    return scale, stats, correspondences


if __name__ == "__main__":
    """
    python metric_alignment.py \
        --dataset_dir /path/to/DL3DV-ALL-960P/1K \
        --scene_id <scene_id>
    """
    parser = argparse.ArgumentParser(description="Align COLMAP reconstruction to metric scale")

    # Option 1: Use dataset_dir + scene_id
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Path to dataset root directory (e.g., DL3DV-ALL-960P)",
    )
    parser.add_argument("--scene_id", default=None, help="Scene ID/name within dataset")

    # Option 2: Use explicit paths (overrides dataset_dir/scene_id if provided)
    parser.add_argument(
        "--colmap_dir",
        default=None,
        help="Path to COLMAP sparse/0 directory (overrides dataset_dir/scene_id)",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Path to images directory (overrides dataset_dir/scene_id)",
    )

    # Common arguments
    parser.add_argument("--output_dir", default=None, help="Output directory for scaled reconstruction")
    parser.add_argument("--num_images", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug visualizations")
    parser.add_argument(
        "--downsample_factor",
        type=float,
        default=4.0,
        help="Factor by which images are downsampled (e.g., 4.0 for images_4 folder)",
    )

    args = parser.parse_args()

    # Construct paths from dataset_dir and scene_id if not explicitly provided
    if args.colmap_dir is None or args.image_dir is None:
        if args.dataset_dir is None or args.scene_id is None:
            parser.error("Either provide --dataset_dir and --scene_id, or provide both --colmap_dir and --image_dir")

        scene_path = os.path.join(args.dataset_dir, args.scene_id)

        if args.colmap_dir is None:
            args.colmap_dir = os.path.join(scene_path, "colmap", "sparse", "0")

        if args.image_dir is None:
            # Determine image directory based on downsample factor
            if args.downsample_factor == 1.0:
                image_folder = "images"
            else:
                image_folder = f"images_{int(args.downsample_factor)}"
            args.image_dir = os.path.join(scene_path, image_folder)

        # Set default output_dir if not provided
        if args.output_dir is None:
            args.output_dir = os.path.join("./debug", args.scene_id, "scaled")

    # Validate paths exist
    if not os.path.exists(args.colmap_dir):
        parser.error(f"COLMAP directory does not exist: {args.colmap_dir}")
    if not os.path.exists(args.image_dir):
        parser.error(f"Image directory does not exist: {args.image_dir}")

    scale, stats, correspondences = align_colmap_to_metric_scale(
        args.colmap_dir,
        args.image_dir,
        args.output_dir,
        args.num_images,
        debug=not args.no_debug,
        downsample_factor=args.downsample_factor,
    )

    with open(os.path.join(args.output_dir, "scale_info.txt"), "w") as f:
        f.write(f"Scale factor: {scale}\n")
