# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DL3DV HDF5 Evaluation Dataset

Dataset for evaluating on DL3DV test scenes using pre-computed HDF5 files.
Train/test indices are read from the HDF5 file's selected_indices field.
Rendered images are also read from the HDF5 file.
"""

from pathlib import Path
from typing import Any, List, Tuple

import h5py
import numpy as np
import torch

from model_training.data.dl3dv_test import DL3DVPairedDatasetTest
from model_training.data.reconstruction_path import parse_reconstruction_hdf5_path
from model_training.data.utils import (
    InferencePair,
    NeighborSelectionMode,
    generate_inference_pairs,
    visualize_inference_pairs,
)
from model_training.prompt_formats import caption_filename_for_format
from model_training.utils.video_io import probe_video_frame_count


class DL3DVHDF5EvalDataset(DL3DVPairedDatasetTest):
    """
    Dataset for evaluating on DL3DV using pre-computed HDF5 files.

    Train IDs come from HDF5's selected_indices field.
    Rendered images are read from the HDF5 file.
    Test frames are the complement of train indices.
    """

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        prompt_dir: Path,
        num_views: int,
        dataset_scaling_factor: float,
        frames_per_block: int = 7,
        neighbor_selection_mode: NeighborSelectionMode = NeighborSelectionMode.COVISIBILITY,
        max_test_frames: int | None = None,
        include_all_frames: bool = False,
        filter_scene_id: str | None = None,
        resize_multiple: int = 16,
        *,
        prompt_format: str,
        camera_pose_alignment: str = "mean",
        verbose: bool = False,
    ):
        super().__init__(
            split=split,
            split_path=split_path,
            dl3dv_dir=dl3dv_dir,
            prompt_dir=prompt_dir,
            num_frames=None,
            frames_per_block=frames_per_block,
            num_views=num_views,
            dataset_scaling_factor=dataset_scaling_factor,
            max_neighbors=num_views,
            verbose=verbose,
            resize_multiple=resize_multiple,
            prompt_format=prompt_format,
            skip_vae_check=True,
            camera_pose_alignment=camera_pose_alignment,
        )

        self.num_views = num_views
        self.neighbor_selection_mode = neighbor_selection_mode
        self.inference_items: List[Tuple[int, InferencePair]] = []
        filtered_scene_ids = set()

        for data_idx, data_dict in enumerate(self.data):
            if num_views not in data_dict:
                raise ValueError(f"num_views {num_views} not found in data_dict (available: {list(data_dict.keys())})")
            hdf5_path = data_dict[num_views][0]
            scene_id, half, n_views = parse_reconstruction_hdf5_path(hdf5_path)

            if n_views != num_views:
                continue

            if half != "0":
                continue

            if filter_scene_id is not None and scene_id != filter_scene_id:
                continue

            filtered_scene_ids.add(scene_id)

            with h5py.File(hdf5_path, "r") as f:
                source_train_ids = set(f["selected_indices"][:].tolist())
                render_total_frames = probe_video_frame_count(f["render_video"][:].tobytes())

            transforms = self.transforms_by_scene_id[scene_id]
            total_frames = min(len(transforms["frames"]), render_total_frames)
            train_ids = {frame_idx for frame_idx in source_train_ids if frame_idx < total_frames}
            if verbose and total_frames != len(transforms["frames"]):
                print(
                    f"Scene {scene_id}: limiting eval frames to {total_frames} HDF5 renders "
                    f"(transforms has {len(transforms['frames'])} frames)"
                )

            extrinsics_c2w = None
            if neighbor_selection_mode == NeighborSelectionMode.COVISIBILITY:
                extrinsics_c2w = np.array([transforms["frames"][i]["transform_matrix"] for i in range(total_frames)])

            pairs = generate_inference_pairs(
                train_ids=train_ids,
                total_frames=total_frames,
                num_train_context=num_views,
                max_test_frames=max_test_frames,
                selection_mode=neighbor_selection_mode,
                extrinsics_c2w=extrinsics_c2w,
                include_all_frames=include_all_frames,
            )

            for pair in pairs:
                pair.scene_id = scene_id
                self.inference_items.append((data_idx, pair))

        if verbose:
            print(
                f"Generated {len(self.inference_items)} inference items "
                f"across {len(filtered_scene_ids)} matched scene(s) "
                f"(base split contains {len(self.data)} scene entries)"
            )

    def _prompt_paths_for_scene(
        self, scene_prompt_dir: Path, num_frames: int | None, scene_num_frames: int
    ) -> list[Path]:
        """Exactly one caption per scene: the canonical forward-order file, one record.

        The reverse/revisit files next to it are training augmentation; any
        draw among files or records would tie the caption to the item's
        position in the job, which changes with sharding. Missing or
        ambiguous captions fail the run rather than shrink the benchmark.
        """
        canonical_prompt_path = scene_prompt_dir / caption_filename_for_format(
            "frames_all_stride_1.h5", self.prompt_format
        )
        if not canonical_prompt_path.exists():
            raise FileNotFoundError(
                f"Scene {scene_prompt_dir.name} has no canonical evaluation prompt {canonical_prompt_path} "
                "(regenerate the scene's prompt files)"
            )
        with h5py.File(canonical_prompt_path, "r") as f:
            num_records = len(f)
        if num_records != 1:
            raise ValueError(
                f"Canonical evaluation prompt {canonical_prompt_path} holds {num_records} caption records; "
                "evaluation conditions on exactly one"
            )
        return [canonical_prompt_path]

    def __len__(self) -> int:
        return len(self.inference_items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data_idx, pair = self.inference_items[idx]
        # Evaluation samples nothing: construction validated one caption file
        # with one record per scene, and frame/neighbor indices come from the
        # pair. The local generator keeps get_item_inner's now-inert draws
        # from consuming global RNG state.
        generator = torch.Generator().manual_seed(self.validation_seed + idx)
        item = self.get_item_inner(
            data_idx,
            self.num_views,
            len(pair.neighbor_indices),
            override_frame_indices=pair.test_indices,
            override_neighbor_indices=pair.neighbor_indices,
            generator=generator,
        )
        item["frame_indices"] = torch.tensor(pair.test_indices, dtype=torch.long)
        item["neighbor_indices"] = torch.tensor(pair.neighbor_indices, dtype=torch.long)
        item["scene_id"] = pair.scene_id
        item["chunk_idx"] = pair.chunk_idx
        # valid_frames_mask: True for actual frames, False for VAE padding (added later by eval script)
        num_inference_frames = len(pair.test_indices)
        item["valid_frames_mask"] = torch.ones(num_inference_frames, dtype=torch.bool)
        item["is_test_frame"] = (
            torch.tensor(pair.is_test_frame, dtype=torch.bool)
            if pair.is_test_frame is not None
            else torch.ones(num_inference_frames, dtype=torch.bool)
        )
        item["gt_index"] = torch.where(item["is_test_frame"], item["frame_indices"], -1)
        return item

    def visualize_splits(self, output_path: str | Path | None = None, title: str | None = None) -> None:
        """Visualize train/test splits for this dataset."""

        def get_total_frames(data_idx: int) -> int:
            hdf5_path = next(iter(self.data[data_idx].values()))[0]
            scene_id, _, _ = parse_reconstruction_hdf5_path(hdf5_path)
            return len(self.transforms_by_scene_id[scene_id]["frames"])

        visualize_inference_pairs(
            inference_items=self.inference_items,
            get_total_frames=get_total_frames,
            output_path=output_path,
            title=title or f"Ours - {self.neighbor_selection_mode.value.upper()}: Train (green) / Test (blue)",
            selection_mode=self.neighbor_selection_mode,
        )
