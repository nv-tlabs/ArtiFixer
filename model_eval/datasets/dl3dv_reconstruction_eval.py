# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DL3DV reconstruction-render eval paths."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from zipfile import ZipFile

from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset, ReconstructedColmapScene
from model_eval.dl3dv_difix_frame_remap import apply_difix_frame_remap
from model_eval.dl3dv_reconstruction_evalsets import RECON_SUBDIR_BY_EVALSET
from model_eval.prepare_dl3dv_distill_dataset import scene_items_from_split
from model_training.data.scene_utils import downsampled_image_path, load_scene_transforms_from_zip
from model_training.data.utils import NeighborSelectionMode
from model_training.model_profiles import prompt_format_for_model, spatial_resize_multiple
from model_training.prompt_formats import caption_filename_for_format


def create_dl3dv_reconstruction_dataset(args, selection_mode, max_test_frames, include_all_frames, rank: int):
    recon_subdir = args.dl3dv_recon_subdir or RECON_SUBDIR_BY_EVALSET[args.evalset]
    filter_scene_ids = args.scene_ids
    if filter_scene_ids is None and args.scene_id is not None:
        filter_scene_ids = [args.scene_id]
    return DL3DVReconstructionEvalDataset(
        split="test",
        split_path=args.split_path,
        dl3dv_dir=args.dl3dv_dir,
        recon_results_dir=args.recon_results_dir,
        prompt_dir=args.prompt_dir,
        num_views=args.num_views,
        dataset_scaling_factor=args.dataset_scaling_factor,
        checkpoint=args.dl3dv_reconstruction_checkpoint,
        neighbor_selection_mode=selection_mode,
        max_test_frames=max_test_frames,
        include_all_frames=include_all_frames,
        filter_scene_ids=filter_scene_ids,
        resize_multiple=spatial_resize_multiple(args.model_id),
        prompt_format=prompt_format_for_model(args.model_id),
        camera_pose_alignment=args.camera_pose_alignment,
        verbose=(rank == 0),
        recon_subdir=recon_subdir,
        use_difix_frame_remap=args.evalset.endswith("_difix"),
    )


class DL3DVReconstructionEvalDataset(ReconstructedColmapEvalDataset):
    """Adapt existing DL3DV reconstruction-render layouts to the reconstructed-COLMAP eval loader."""

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        recon_results_dir: Path,
        prompt_dir: Path,
        num_views: int | None,
        dataset_scaling_factor: float,
        image_downsample_factor: int = 4,
        checkpoint: str = "30000",
        neighbor_selection_mode: NeighborSelectionMode = NeighborSelectionMode.COVISIBILITY,
        max_test_frames: int | None = None,
        include_all_frames: bool = False,
        filter_scene_ids: Collection[str] | None = None,
        resize_multiple: int = 16,
        *,
        prompt_format: str,
        camera_pose_alignment: str = "mean",
        verbose: bool = False,
        recon_subdir: str | None = None,
        use_difix_frame_remap: bool = False,
    ):
        self.dl3dv_dir = Path(dl3dv_dir)
        self.recon_results_dir = Path(recon_results_dir)
        self.prompt_dir = Path(prompt_dir)
        self.dataset_scaling_factor = dataset_scaling_factor
        self.image_downsample_factor = image_downsample_factor
        self.checkpoint = checkpoint
        self.recon_subdir = recon_subdir
        self.use_difix_frame_remap = use_difix_frame_remap
        self.filter_scene_ids = set(filter_scene_ids) if filter_scene_ids is not None else None
        super().__init__(
            split=split,
            split_path=split_path,
            num_views=num_views,
            neighbor_selection_mode=neighbor_selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            resize_multiple=resize_multiple,
            prompt_format=prompt_format,
            camera_pose_alignment=camera_pose_alignment,
            verbose=verbose,
        )

    def _get_scene_dir(self, scene_id: str) -> Path:
        if self.recon_subdir is not None:
            return self.recon_results_dir / scene_id / self.recon_subdir / scene_id
        return (
            self.recon_results_dir
            / "difix_split_reconstruction_eval"
            / scene_id
            / "difix_split_reconstruction"
            / scene_id
        )

    def _get_checkpoint_dir(self, scene_id: str) -> Path:
        return self._get_scene_dir(scene_id) / f"ours_{self.checkpoint}"

    @staticmethod
    def _rendered_frame_count(render_dir: Path, opacity_dir: Path) -> int:
        render_indices = sorted(int(path.stem) for path in render_dir.glob("*.png"))
        opacity_indices = sorted(int(path.stem) for path in opacity_dir.glob("*.png"))
        assert render_indices, f"{render_dir} has no rendered frames"
        assert render_indices == opacity_indices, f"{render_dir} and {opacity_dir} frame sets differ"
        assert render_indices == list(range(len(render_indices))), f"{render_dir} frame indices must be contiguous"
        return len(render_indices)

    def _load_scene_data(
        self,
        split_path: Path,
        split: str,
        filter_scene_id: str | None,  # always None here; this dataset filters via filter_scene_ids
        verbose: bool,
    ) -> None:
        with open(split_path) as f:
            split_data = json.load(f)[split]

        self._reset_scene_data()
        scene_items = tuple(
            scene_items_from_split(
                split_data,
                include_metric_scale=True,
            )
        )
        if self.filter_scene_ids is not None:
            split_scene_ids = {scene_id for scene_id, _, _ in scene_items}
            unknown_scene_ids = sorted(self.filter_scene_ids - split_scene_ids)
            if unknown_scene_ids:
                raise ValueError(
                    f"Requested DL3DV scene IDs are absent from the {split!r} split: " f"{', '.join(unknown_scene_ids)}"
                )

        for scene_id, source_subdir, metric_scale in scene_items:
            if self.filter_scene_ids is not None and scene_id not in self.filter_scene_ids:
                continue

            checkpoint_dir = self._get_checkpoint_dir(scene_id)
            render_dir = checkpoint_dir / "renders"
            opacity_dir = checkpoint_dir / "opacity"
            selected_indices_path = checkpoint_dir / "selected_indices.json"
            transforms_path = self.dl3dv_dir / source_subdir / f"{scene_id}.zip"

            required_paths = (render_dir, opacity_dir, selected_indices_path, transforms_path)
            missing = [path for path in required_paths if not path.exists()]
            if missing:
                if self.filter_scene_ids is not None:
                    raise FileNotFoundError(
                        f"Requested DL3DV scene {scene_id!r} cannot be evaluated: missing {missing[0]}"
                    )
                if verbose:
                    print(f"Skipping {scene_id}: missing {missing[0]}")
                continue

            # Evaluation always conditions on the scene's canonical
            # forward-order caption, never on the reverse/revisit training
            # variants stored next to it. A missing canonical prompt is a data
            # error, not a scene-not-reconstructed-here condition, so it
            # raises even when scenes without renders are skipped above.
            # Unlike the HDF5 eval, only scenes this run evaluates are
            # checked: parallel evaluation uses --scene_ids, and one shard
            # must not fail on another shard's scene.
            canonical_prompt_path = (
                self.prompt_dir
                / source_subdir
                / scene_id
                / caption_filename_for_format("frames_all_stride_1.h5", self.prompt_format)
            )
            if not canonical_prompt_path.exists():
                raise FileNotFoundError(
                    f"Scene {scene_id} has no canonical evaluation prompt {canonical_prompt_path} "
                    "(regenerate the scene's prompt files)"
                )

            with ZipFile(transforms_path, "r") as zip_file:
                transforms, _ = load_scene_transforms_from_zip(zip_file, scene_id)
            assert isinstance(transforms["frames"], list), f"{transforms_path} must contain a frames list"
            if self.use_difix_frame_remap:
                transforms = apply_difix_frame_remap(transforms, scene_id)
            rendered_frame_count = self._rendered_frame_count(render_dir, opacity_dir)
            if self.use_difix_frame_remap and rendered_frame_count != len(transforms["frames"]):
                raise ValueError(
                    f"Scene {scene_id} has {rendered_frame_count} renders but {len(transforms['frames'])} remapped frames"
                )
            transforms["frames"] = [
                {**frame, "file_path": downsampled_image_path(frame["file_path"], self.image_downsample_factor)}
                for frame in transforms["frames"][:rendered_frame_count]
            ]
            scene = ReconstructedColmapScene(
                scene_id=scene_id,
                transforms_path=transforms_path,
                image_root=transforms_path,
                render_dir=render_dir,
                opacity_dir=opacity_dir,
                selected_indices_path=selected_indices_path,
                target_indices_path=None,
                prompt_path=canonical_prompt_path,
                camera_scale=metric_scale * self.dataset_scaling_factor,
                has_gt=True,
            )
            self._register_scene(scene, transforms, verbose)

        if verbose:
            print(f"Found {len(self.scene_ids)} DL3DV scenes with reconstruction renders")
