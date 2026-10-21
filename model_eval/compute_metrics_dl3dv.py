# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metrics computation for DL3DV test-scene eval, including ArtiFixer and DiFix variants."""

import argparse
import glob
import json
import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import h5py
import torch

from data_processing.scene_utils import read_scene_list
from model_eval.dl3dv_difix_frame_remap import apply_difix_frame_remap
from model_eval.dl3dv_reconstruction_evalsets import RECON_SUBDIR_BY_EVALSET, default_dl3dv_num_views
from model_eval.metrics_utils import (
    MetricSuite,
    build_sorted_test_mask_index,
    downsampled_image_path,
    load_dl3dv_scene_transforms,
    load_dl3dv_scene_zip_bytes,
    load_image_as_tensor,
    load_subdir_by_scene_id,
    load_visibility_mask,
    validate_frame_coverage,
)
from model_eval.reconstructed_colmap_evalsets import num_views_output_label
from model_training.utils.video_io import probe_video_frame_count

DL3DV_EVALSETS = ("3dgrut_dl3dv_ours", "3dgrut_dl3dv_difix", "artifixer3d_dl3dv_ours", "artifixer3d_dl3dv_difix")
DIFIX_EVALSETS = {"3dgrut_dl3dv_difix", "artifixer3d_dl3dv_difix"}
OURS_EVALSETS = {"3dgrut_dl3dv_ours", "artifixer3d_dl3dv_ours"}

DATA_PROCESSING_DIR = Path(__file__).resolve().parents[1] / "data_processing"


@lru_cache(maxsize=None)
def _load_split(split_path: Path) -> dict:
    with split_path.open() as f:
        return json.load(f)


def _hdf5_reconstruction_path(scene_id: str, num_views: int, split_path: Path) -> Path:
    split = _load_split(split_path)
    reconstructions = []
    for bucket in ("test", "trainval"):
        reconstructions.extend(split[bucket].get(scene_id, []))

    matches = [
        reconstruction
        for reconstruction in reconstructions
        if reconstruction["scene_half"] == "0" and reconstruction["num_selected_indices"] == num_views
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one DL3DV HDF5 path for scene_id={scene_id} num_views={num_views}, got {len(matches)}"
        )

    return Path(matches[0]["hdf5_path"])


def load_metric_scene_ids(split_path: Path) -> list[str]:
    split = _load_split(split_path)
    return sorted(split["test"])


def load_hdf5_training_indices(scene_id: str, num_views: int, split_path: Path) -> set[int]:
    hdf5_file = _hdf5_reconstruction_path(scene_id, num_views, split_path)
    with h5py.File(hdf5_file, "r") as f:
        return set(f["selected_indices"][:].tolist())


def load_hdf5_frame_count(scene_id: str, num_views: int, split_path: Path) -> int:
    hdf5_file = _hdf5_reconstruction_path(scene_id, num_views, split_path)
    with h5py.File(hdf5_file, "r") as handle:
        count = handle.attrs.get("n_frames")
        if count is None and "depth_np_array" in handle:
            count = handle["depth_np_array"].shape[0]
        if count is None and "render_video" in handle:
            count = probe_video_frame_count(handle["render_video"][:].tobytes())
        if count is None:
            raise ValueError(
                f"Cannot determine source frame count from {hdf5_file}: missing n_frames, depth data, and render_video"
            )
    frame_count = int(count)
    if frame_count <= 0 or frame_count != count:
        raise ValueError(f"Invalid source frame count in {hdf5_file}: {count!r}")
    return frame_count


def load_difix_training_indices(train_ids_dir: Path, scene_id: str) -> set[int]:
    """Load held-in DiFix training frame ids so metrics skip source-view frames."""
    with (train_ids_dir / scene_id / "difix_train_ids.json").open() as f:
        return set(json.load(f))


def parse_num_views(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("--num_views must be positive")
        return value
    if value == "auto":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--num_views must be positive")
    return parsed


def _reconstruction_checkpoint_dir(
    recon_results_dir: Path, scene_id: str, checkpoint: str, recon_subdir: str | None
) -> Path:
    if recon_subdir is None:
        return (
            recon_results_dir
            / "difix_split_reconstruction_eval"
            / scene_id
            / "difix_split_reconstruction"
            / scene_id
            / f"ours_{checkpoint}"
        )
    return recon_results_dir / scene_id / recon_subdir / scene_id / f"ours_{checkpoint}"


def load_reconstruction_training_indices(
    recon_results_dir: Path,
    scene_id: str,
    *,
    checkpoint: str,
    recon_subdir: str | None,
) -> set[int]:
    checkpoint_dir = _reconstruction_checkpoint_dir(recon_results_dir, scene_id, checkpoint, recon_subdir)
    with (checkpoint_dir / "selected_indices.json").open() as f:
        return set(json.load(f))


def load_reconstruction_frame_count(
    recon_results_dir: Path, scene_id: str, *, checkpoint: str, recon_subdir: str | None
) -> int:
    checkpoint_dir = _reconstruction_checkpoint_dir(recon_results_dir, scene_id, checkpoint, recon_subdir)
    render_indices = sorted(int(path.stem) for path in (checkpoint_dir / "renders").glob("*.png"))
    opacity_indices = sorted(int(path.stem) for path in (checkpoint_dir / "opacity").glob("*.png"))
    if render_indices != opacity_indices:
        raise ValueError(f"Mismatched reconstruction render/opacity frame IDs in {checkpoint_dir}")
    validate_frame_coverage(render_indices, set(range(len(render_indices))), label=str(checkpoint_dir))
    return len(render_indices)


def build_difix_mask_index(transforms: dict, training_indices: set[int]) -> dict[int, int]:
    """Map transform frame ids to compact DiFix visibility-mask ids."""
    test_indices = set(range(len(transforms["frames"]))) - training_indices
    return build_sorted_test_mask_index(transforms, test_indices)


def resolve_difix_mask_path(
    visibility_masks_dir: Path, scene_id: str, frame_idx: int, mask_idx: int | None
) -> Path | None:
    mask_index = frame_idx if mask_idx is None else mask_idx
    path = visibility_masks_dir / scene_id / "visibility_median_count" / f"{mask_index:05d}.png"
    return path if path.is_file() else None


def compute_metrics(
    render_name: str,
    num_views: int | None,
    evalset: str,
    *,
    split_path: Path,
    dl3dv_dir: Path,
    eval_base_path: Path,
    use_masks: bool = True,
    render_trajectory: str = "val_frames",
    visibility_masks_dir: Path | None = None,
    test_scene_ids: Sequence[str] | None = None,
    training_indices_for_scene: Callable[[str], set[int]] | None = None,
    frame_count_for_scene: Callable[[str], int] | None = None,
    metrics_suffix: str = "",
    masked_lpips_backend: str = "torchmetrics",
    lpips_protocol: str = "standard",
    neighbor_selection_mode: str = "evenly_spaced",
    output_suffix: str = "",
) -> None:
    """Compute per-scene and aggregated metrics for rendered eval outputs.

    Args:
        render_name: the render run's directory name under eval_base_path.
        num_views: number of training views used for evaluation.
        evalset: label used to construct the render output directory (e.g. "3dgrut_dl3dv_ours").
        use_masks: also compute visibility-masked variants of each metric.
        split_path: split JSON used by eval and by ours HDF5 selected_indices lookup.
        dl3dv_dir: DL3DV zip root.
        eval_base_path: root containing rendered eval outputs.
        render_trajectory: trajectory suffix used by the renderer output directory.
        visibility_masks_dir: root containing optional visibility masks.
        test_scene_ids: explicit scene ids to evaluate for custom comparison paths.
        metrics_suffix: optional suffix for sidecar metric YAMLs.
        masked_lpips_backend: LPIPS implementation for masked metrics.
        lpips_protocol: standard normalization, or legacy to reproduce historical scores.
        neighbor_selection_mode: neighbor selection mode used by the render run.
        output_suffix: output-directory suffix used by the render run.
        training_indices_for_scene: optional callback for custom comparison eval paths.
            Defaults to reading selected_indices from the ArtiFixer HDF5 referenced by split_path.
        frame_count_for_scene: available reconstruction frames. Defaults to HDF5 metadata when
            it supplies training indices, otherwise the source transforms' frame count.
    """
    if use_masks and visibility_masks_dir is None:
        raise ValueError("--visibility_masks_dir is required unless --no_masks is set")
    if test_scene_ids is None:
        test_scene_ids = load_metric_scene_ids(split_path)
    test_scene_ids = tuple(test_scene_ids)
    if training_indices_for_scene is None:
        training_indices_for_scene = lambda scene_id: load_hdf5_training_indices(scene_id, num_views, split_path)
        if frame_count_for_scene is None:
            frame_count_for_scene = lambda scene_id: load_hdf5_frame_count(scene_id, num_views, split_path)

    # Mirrors the per-evalset subtree run_inference writes under the render name.
    trajectory_part = "" if render_trajectory == "val_frames" else f"_{render_trajectory}"
    output_suffix_part = f"_{output_suffix}" if output_suffix else ""
    eval_mode = (
        f"distilled_views_{evalset}_{num_views_output_label(num_views)}_{neighbor_selection_mode}"
        f"{trajectory_part}{output_suffix_part}"
    )
    path = eval_base_path / render_name / eval_mode

    if not path.exists():
        raise FileNotFoundError(f"Eval path not found: {path}")

    missing_scene_ids = [scene_id for scene_id in test_scene_ids if not (path / scene_id).is_dir()]
    if missing_scene_ids:
        formatted_scene_ids = ", ".join(missing_scene_ids)
        raise FileNotFoundError(
            f"Missing {len(missing_scene_ids)} expected DL3DV scene output directories under {path}: "
            f"{formatted_scene_ids}"
        )

    scene_id_to_subdir = load_subdir_by_scene_id(split_path)

    device = torch.device("cuda")
    suite = MetricSuite(
        device, use_masks=use_masks, masked_lpips_backend=masked_lpips_backend, lpips_protocol=lpips_protocol
    )

    with torch.inference_mode():
        for scene_id in test_scene_ids:
            print(f"Processing scene {scene_id}...")

            subdir = scene_id_to_subdir[scene_id]
            training_indices = training_indices_for_scene(scene_id)

            transforms = load_dl3dv_scene_transforms(dl3dv_dir, subdir, scene_id)
            if evalset in DIFIX_EVALSETS:
                transforms = apply_difix_frame_remap(transforms, scene_id)
            total_frames = len(transforms["frames"])
            if frame_count_for_scene is not None:
                render_frames = frame_count_for_scene(scene_id)
                if evalset in DIFIX_EVALSETS and render_frames != total_frames:
                    raise ValueError(f"Scene {scene_id} has {render_frames} renders but {total_frames} remapped frames")
                total_frames = min(total_frames, render_frames)
            training_indices = {index for index in training_indices if 0 <= index < total_frames}
            expected_indices = set(range(total_frames)) - training_indices
            frame_idx_to_path = {
                i: downsampled_image_path(frame["file_path"], 4)
                for i, frame in enumerate(transforms["frames"][:total_frames])
            }
            difix_mask_index = build_difix_mask_index(transforms, training_indices) if evalset in DIFIX_EVALSETS else {}

            pred_dir = path / scene_id / "frames" / "batch_0000" / "pred"
            rendered_dir = path / scene_id / "frames" / "batch_0000" / "rendered"
            pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
            rendered_files = sorted(glob.glob(os.path.join(rendered_dir, "*.png")))
            pred_names = [os.path.basename(path) for path in pred_files]
            rendered_names = [os.path.basename(path) for path in rendered_files]
            if pred_names != rendered_names:
                raise ValueError(f"Mismatched frame names between pred and rendered outputs for {scene_id}")
            frame_indices = [int(os.path.splitext(name)[0]) for name in pred_names]
            validate_frame_coverage(
                [index for index in frame_indices if index not in training_indices],
                expected_indices,
                label=str(path / scene_id),
            )
            needed_relpaths = [frame_idx_to_path[index] for index in frame_indices if index in expected_indices]
            gt_bytes_by_path = load_dl3dv_scene_zip_bytes(dl3dv_dir, subdir, scene_id, needed_relpaths)

            for pred_file, rendered_file, frame_idx in zip(pred_files, rendered_files, frame_indices):
                if frame_idx in training_indices:
                    continue

                gt = load_image_as_tensor(gt_bytes_by_path[frame_idx_to_path[frame_idx]], device)
                pred = load_image_as_tensor(pred_file, device)
                rendered = load_image_as_tensor(rendered_file, device)

                suite.update(scene_id, pred, rendered, gt)

                if use_masks:
                    mask_path = resolve_difix_mask_path(
                        visibility_masks_dir,
                        scene_id,
                        frame_idx,
                        difix_mask_index.get(frame_idx),
                    )
                    if mask_path is None:
                        raise FileNotFoundError(
                            f"Missing visibility mask for scene {scene_id}, frame {frame_idx} in {visibility_masks_dir}"
                        )
                    mask = load_visibility_mask(mask_path, (gt.shape[3], gt.shape[2]), device, threshold=1)
                    suite.update_masked(scene_id, pred, rendered, gt, mask)

            suite.gut.print_scene_summary(scene_id, prefix="3D-GUT ")
            suite.ours.print_scene_summary(scene_id)
            if use_masks:
                suite.gut_masked.print_scene_summary(scene_id, prefix="3D-GUT (masked) ")
                suite.ours_masked.print_scene_summary(scene_id, prefix="(masked) ")

        suite.ours.require_values("psnr", path, evalset=evalset)
        suite.gut.require_values("psnr", path, evalset=evalset)

        suite.report(path, metrics_suffix=metrics_suffix)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render_name",
        type=str,
        required=True,
        help="The render run's --render_name; renders are read from EVAL_BASE_PATH/<render_name>.",
    )
    parser.add_argument(
        "--num_views",
        default=None,
        help="Reference-view count used by inference, or auto for per-scene selection. "
        "Defaults to 6 for 3dgrut_dl3dv_ours and auto for reconstruction-directory evalsets.",
    )
    parser.add_argument("--evalset", choices=DL3DV_EVALSETS, required=True)
    parser.add_argument("--render_trajectory", choices=["val_frames", "all_frames"], default="val_frames")
    parser.add_argument(
        "--neighbor_selection_mode",
        choices=["consecutive", "evenly_spaced", "covisibility"],
        default="evenly_spaced",
        help="Neighbor selection mode used by the render run (part of its output directory name).",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Output-directory suffix used by the render run.",
    )
    parser.add_argument("--no_masks", action="store_true")
    parser.add_argument("--split_path", type=Path, required=True)
    parser.add_argument("--dl3dv_dir", type=Path, required=True)
    parser.add_argument("--eval_base_path", type=Path, required=True)
    parser.add_argument("--visibility_masks_dir", type=Path)
    parser.add_argument("--metrics_suffix", type=str, default="")
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
    parser.add_argument(
        "--difix_train_ids_dir",
        type=Path,
        help="Directory containing per-scene difix_train_ids.json; required for DiFix evalsets.",
    )
    parser.add_argument(
        "--recon_results_dir",
        type=Path,
        help="Reconstruction output root; required for directory-based reconstruction evalsets.",
    )
    parser.add_argument(
        "--dl3dv_recon_subdir",
        type=str,
        default=None,
        help="Reconstruction experiment subdir under each DL3DV scene.",
    )
    parser.add_argument(
        "--dl3dv_reconstruction_checkpoint",
        type=str,
        default="30000",
        help="Reconstruction checkpoint iteration used for selected_indices.json lookup.",
    )
    args = parser.parse_args(argv)
    try:
        args.num_views = (
            default_dl3dv_num_views(args.evalset) if args.num_views is None else parse_num_views(args.num_views)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.num_views is None and args.evalset == "3dgrut_dl3dv_ours":
        parser.error("--num_views auto is not supported for 3dgrut_dl3dv_ours; pass a numeric view count")
    if args.evalset in RECON_SUBDIR_BY_EVALSET and args.recon_results_dir is None:
        parser.error("--recon_results_dir is required for directory-based reconstruction evalsets")
    return args


def main() -> None:
    args = parse_args()

    num_views = args.num_views
    test_scene_ids = None
    training_indices_for_scene = None
    frame_count_for_scene = None
    recon_subdir = args.dl3dv_recon_subdir or RECON_SUBDIR_BY_EVALSET.get(args.evalset)
    if args.recon_results_dir is not None and args.evalset in RECON_SUBDIR_BY_EVALSET:
        frame_count_for_scene = lambda scene_id: load_reconstruction_frame_count(
            args.recon_results_dir,
            scene_id,
            checkpoint=args.dl3dv_reconstruction_checkpoint,
            recon_subdir=recon_subdir,
        )
    if args.evalset in DIFIX_EVALSETS:
        if args.difix_train_ids_dir is None:
            raise ValueError("--difix_train_ids_dir is required for DiFix DL3DV evalsets")
        test_scene_ids = read_scene_list(DATA_PROCESSING_DIR / "dl3dv_benchmark_difix_test_scenes.txt")
        training_indices_for_scene = lambda scene_id: load_difix_training_indices(args.difix_train_ids_dir, scene_id)
    elif args.evalset in OURS_EVALSETS:
        test_scene_ids = read_scene_list(DATA_PROCESSING_DIR / "dl3dv_benchmark_ours_test_scenes.txt")
        if args.evalset in RECON_SUBDIR_BY_EVALSET:
            training_indices_for_scene = lambda scene_id: load_reconstruction_training_indices(
                args.recon_results_dir,
                scene_id,
                checkpoint=args.dl3dv_reconstruction_checkpoint,
                recon_subdir=recon_subdir,
            )

    compute_metrics(
        render_name=args.render_name,
        num_views=num_views,
        evalset=args.evalset,
        use_masks=not args.no_masks,
        render_trajectory=args.render_trajectory,
        split_path=args.split_path,
        dl3dv_dir=args.dl3dv_dir,
        eval_base_path=args.eval_base_path,
        visibility_masks_dir=args.visibility_masks_dir,
        test_scene_ids=test_scene_ids,
        training_indices_for_scene=training_indices_for_scene,
        frame_count_for_scene=frame_count_for_scene,
        metrics_suffix=args.metrics_suffix,
        masked_lpips_backend=args.masked_lpips_backend,
        lpips_protocol=args.lpips_protocol,
        neighbor_selection_mode=args.neighbor_selection_mode,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
