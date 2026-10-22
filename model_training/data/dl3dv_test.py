# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import torch

from model_training.data.dl3dv_base import DL3DVPairedDatasetBase
from model_training.data.reconstruction_path import parse_reconstruction_hdf5_path
from model_training.prompt_formats import WAN_T5_PROMPT_FORMAT
from model_training.utils.validation_artifacts import validate_manifest


class DL3DVPairedDatasetTest(DL3DVPairedDatasetBase):

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        prompt_dir: Path,
        num_frames: int | None,
        frames_per_block: int | None,
        num_views: int,
        dataset_scaling_factor: float,
        validation_seed: int = 42,
        max_neighbors: int | None = None,
        verbose: bool = False,
        skip_vae_check: bool = False,
        resize_multiple: int = 16,
        prompt_format: str = WAN_T5_PROMPT_FORMAT,
        camera_pose_alignment: str = "mean",
        validation_manifest: dict | None = None,
        capture_validation_metadata: bool = False,
    ):
        self.validation_manifest = validation_manifest
        if validation_manifest is not None:
            validate_manifest(validation_manifest)
        super().__init__(
            split=split,
            split_path=split_path,
            dl3dv_dir=dl3dv_dir,
            prompt_dir=prompt_dir,
            num_frames=num_frames,
            frames_per_block=frames_per_block,
            return_unencoded_prompt=True,
            dataset_scaling_factor=dataset_scaling_factor,
            max_neighbors=max_neighbors,
            verbose=verbose,
            skip_vae_check=skip_vae_check,
            resize_multiple=resize_multiple,
            prompt_format=prompt_format,
            camera_pose_alignment=camera_pose_alignment,
            capture_validation_metadata=capture_validation_metadata,
        )

        self.num_views = num_views
        self.validation_seed = validation_seed
        self.manifest_indices = None
        if validation_manifest is not None:
            lookup = {parse_reconstruction_hdf5_path(recons[3][0])[:2]: index for index, recons in enumerate(self.data)}
            self.manifest_indices = [
                lookup[(sample["scene_id"], sample["half"])] for sample in validation_manifest["samples"]
            ]

    def _prompt_paths_for_scene(self, scene_prompt_dir, num_frames, scene_num_frames):
        if self.validation_manifest is None:
            return super()._prompt_paths_for_scene(scene_prompt_dir, num_frames, scene_num_frames)
        names = {
            sample["prompt_file"]
            for sample in self.validation_manifest["samples"]
            if sample["scene_id"] == scene_prompt_dir.name
        }
        return [scene_prompt_dir / name for name in sorted(names)]

    def __len__(self):
        if self.manifest_indices is not None:
            return len(self.manifest_indices)
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.data:
            raise ValueError(
                "empty validation dataset: no DL3DV items were discovered. "
                "Check num_frames, prompt_dir, and split_path."
            )
        data_idx = idx % len(self.data)
        generator = torch.Generator().manual_seed(self.validation_seed + idx)
        if self.manifest_indices is not None:
            index = idx % len(self.manifest_indices)
            selection = self.validation_manifest["samples"][index]
            return self.get_item_inner(
                self.manifest_indices[index],
                self.num_views,
                self.num_views,
                override_frame_indices=selection["frame_indices"],
                override_neighbor_indices=selection["neighbor_indices"] if self.num_views == 3 else None,
                generator=generator,
                validation_selection=selection,
            )
        return self.get_item_inner(data_idx, self.num_views, self.num_views, generator=generator)
