# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compute metrics for Nerfbusters evaluation outputs.

Computes PSNR, SSIM, LPIPS, and FID for predicted vs ground truth images.
Only evaluates on TEST frames (frame_1_XXXXX.png naming convention).
"""

import argparse
import glob
import json
import os
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from data_processing.nerfbusters import NERFBUSTERS_SCENES, nerfbusters_gt_relpath
from model_eval.datasets.nerfbusters_split import nerfbusters_train_test_split
from model_eval.metrics_utils import (
    MetricSuite,
    build_sorted_test_mask_index,
    load_image_as_tensor,
    load_visibility_mask,
    validate_frame_coverage,
)


def resize_to_gt_if_needed(image: torch.Tensor, gt: torch.Tensor, *, label: str, frame_path: Path) -> torch.Tensor:
    if image.shape[-2:] == gt.shape[-2:]:
        return image
    warnings.warn(
        f"{label} frame {frame_path} has shape {tuple(image.shape[-2:])}, "
        f"expected GT shape {tuple(gt.shape[-2:])}; resizing for metrics.",
        RuntimeWarning,
        stacklevel=2,
    )
    return F.interpolate(image, size=gt.shape[-2:], mode="bicubic", align_corners=False)


def main(
    render_name: str,
    num_views: int,
    *,
    eval_base_path: Path,
    nerfbusters_dir: Path,
    visibility_masks_dir: Path | None = None,
    use_masks: bool = True,
    output_suffix: str = "",
    neighbor_selection_mode: str = "evenly_spaced",
    masked_lpips_backend: str = "torchmetrics",
    lpips_protocol: str = "standard",
):
    if use_masks and visibility_masks_dir is None:
        raise ValueError("--visibility_masks_dir is required unless --no_masks is set")
    # Mirrors the per-evalset subtree run_inference writes under the render name.
    output_suffix_part = f"_{output_suffix}" if output_suffix else ""
    eval_mode = f"nerfbusters_{num_views}_{neighbor_selection_mode}{output_suffix_part}"
    path = eval_base_path / render_name / eval_mode

    if not path.exists():
        raise FileNotFoundError(f"Eval path not found: {path}")

    device = torch.device("cuda")
    suite = MetricSuite(
        device, use_masks=use_masks, masked_lpips_backend=masked_lpips_backend, lpips_protocol=lpips_protocol
    )

    with torch.inference_mode():
        for scene_id in NERFBUSTERS_SCENES:
            scene_path = path / scene_id
            if not scene_path.is_dir():
                raise FileNotFoundError(f"Scene {scene_id} not found at {scene_path}")

            print(f"Processing scene {scene_id}...")

            # Load transforms to get train/test split and frame paths
            transforms_path = nerfbusters_dir / scene_id / "transforms.json"
            with open(transforms_path, "r") as f:
                transforms = json.load(f)

            train_ids, test_ids = nerfbusters_train_test_split(transforms)
            transforms_idx_to_mask_idx = build_sorted_test_mask_index(transforms, test_ids)

            # Build frame index to GT path mapping
            frame_idx_to_path = {}
            for i, frame in enumerate(transforms["frames"]):
                frame_idx_to_path[i] = nerfbusters_gt_relpath(scene_id, frame["file_path"])

            pred_dir = scene_path / "frames" / "batch_0000" / "pred"
            rendered_dir = scene_path / "frames" / "batch_0000" / "rendered"
            pred_files = sorted(
                f
                for f in glob.glob(os.path.join(pred_dir, "*.png"))
                if not os.path.splitext(os.path.basename(f))[0].endswith("_mask")
            )
            rendered_files = sorted(
                f
                for f in glob.glob(os.path.join(rendered_dir, "*.png"))
                if not os.path.splitext(os.path.basename(f))[0].endswith("_mask")
            )
            pred_names = [os.path.basename(path) for path in pred_files]
            rendered_names = [os.path.basename(path) for path in rendered_files]
            if pred_names != rendered_names:
                raise ValueError(f"Mismatched frame names between pred and rendered outputs in {scene_path}")
            frame_indices = [int(Path(name).stem) for name in pred_names]
            validate_frame_coverage(
                [index for index in frame_indices if index not in train_ids], test_ids, label=str(scene_path)
            )

            for pred_file, rendered_file, frame_idx in zip(pred_files, rendered_files, frame_indices):
                # Predictions are named by transforms.json index
                # Only evaluate on TEST frames
                if frame_idx not in test_ids:
                    continue

                gt_file_path = nerfbusters_dir / scene_id / frame_idx_to_path[frame_idx]
                if not gt_file_path.is_file():
                    raise FileNotFoundError(f"GT not found: {gt_file_path}")

                gt = load_image_as_tensor(gt_file_path, device)
                pred = load_image_as_tensor(pred_file, device)
                rendered = load_image_as_tensor(rendered_file, device)
                pred = resize_to_gt_if_needed(pred, gt, label="prediction", frame_path=Path(pred_file))
                rendered = resize_to_gt_if_needed(rendered, gt, label="3D-GUT", frame_path=Path(rendered_file))

                # Non-masked metrics using the shared RGB convention.
                suite.update(scene_id, pred, rendered, gt)

                # Masked metrics using torchmetrics
                # Masks indexed by sorted test filename order, predictions by transforms.json index
                if use_masks:
                    mask_idx = transforms_idx_to_mask_idx[frame_idx]
                    mask_path = visibility_masks_dir / scene_id / f"{mask_idx:05d}.png"
                    if not mask_path.is_file():
                        raise FileNotFoundError(f"Missing visibility mask: {mask_path}")
                    target_size = (gt.shape[3], gt.shape[2])  # (W, H)
                    mask = load_visibility_mask(mask_path, target_size, device)
                    suite.update_masked(scene_id, pred, rendered, gt, mask)

            suite.gut.print_scene_summary(scene_id, prefix="3D-GUT ")
            suite.ours.print_scene_summary(scene_id)

        suite.report(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render_name",
        type=str,
        required=True,
        help="The render run's --render_name; renders are read from EVAL_BASE_PATH/<render_name>.",
    )
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--no_masks", action="store_true")
    parser.add_argument("--eval_base_path", type=Path, required=True)
    parser.add_argument("--nerfbusters_dir", type=Path, required=True)
    parser.add_argument("--visibility_masks_dir", type=Path)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument(
        "--neighbor_selection_mode",
        choices=["consecutive", "evenly_spaced", "covisibility"],
        default="evenly_spaced",
        help="Neighbor selection mode used by the render run (part of its output directory name).",
    )
    parser.add_argument(
        "--masked_lpips_backend",
        choices=("torchmetrics", "alex", "squeeze", "vgg"),
        default="torchmetrics",
    )
    parser.add_argument(
        "--lpips_protocol",
        choices=("standard", "legacy"),
        default="standard",
        help="Use standard LPIPS normalization; legacy only reproduces historical scores.",
    )
    args = parser.parse_args()
    main(
        args.render_name,
        args.num_views,
        eval_base_path=args.eval_base_path,
        nerfbusters_dir=args.nerfbusters_dir,
        visibility_masks_dir=args.visibility_masks_dir,
        use_masks=not args.no_masks,
        output_suffix=args.output_suffix,
        neighbor_selection_mode=args.neighbor_selection_mode,
        masked_lpips_backend=args.masked_lpips_backend,
        lpips_protocol=args.lpips_protocol,
    )
