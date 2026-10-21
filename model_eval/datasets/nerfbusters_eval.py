# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nerfbusters Evaluation Dataset

Dataset for evaluating on Nerfbusters benchmark scenes.
Each scene has:
- transforms.json: Camera parameters and frame paths
- images/: Full resolution images
- images_2/, images_4/, images_8/: Downsampled images

Image folder is selected per-scene to achieve height=960:
- 1080x1920 scenes (aloe, car, garbage, table): use images_2 -> 540x960
- 540x960 scenes (art, century, flowers, picnic, pipe): use images -> 540x960
- 720x960 scenes (pikachu, plant, roses): use images -> 720x960

Nerfbusters naming convention for train/test split:
- frame_XXXXX.png = TRAIN (no _1_ in middle)
- frame_1_XXXXX.png = TEST (has _1_ in middle)

3DGRUT (after commit 3b99e43) renders ALL frames in sequential order when split="test"
So render_idx == frame_idx (1:1 mapping).
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from PIL import Image

from data_processing.nerfbusters import nerfbusters_gt_relpath, nerfbusters_image_folder
from model_eval.datasets.nerfbusters_split import nerfbusters_train_test_split
from model_training.data.utils import (
    InferencePair,
    NeighborSelectionMode,
    compute_camera_rays,
    generate_inference_pairs,
    load_indexed_frames,
    load_prompt_payload_from_path,
    resize_to_multiple,
)
from model_training.model_profiles import requires_reference_camera_rays
from model_training.prompt_formats import caption_filename_for_format


class NerfbustersEvalDataset(torch.utils.data.Dataset):
    """
    Dataset for evaluating on Nerfbusters benchmark scenes.

    Expects 3DGRUT reconstruction results in a directory structure:
        recon_results_dir/
            scene_name/
                {recon_experiment_name}/
                    scene_name/
                        ours_{step}/
                            renders/
                                00000.png, 00001.png, ...
                            opacity/
                                00000.png, 00001.png, ...
    """

    def __init__(
        self,
        nerfbusters_dir: Path,
        recon_results_dir: Path,
        captions_dir: Path,
        num_views: int,
        recon_experiment_name: str,
        dataset_scaling_factor: int = 1,
        neighbor_selection_mode: NeighborSelectionMode = NeighborSelectionMode.EVENLY_SPACED,
        max_test_frames: int | None = None,
        include_all_frames: bool = False,
        filter_scene_id: str | None = None,
        checkpoint: str = "30000",
        image_folder: str | None = None,  # If None, uses per-scene mapping for height=960
        resize_multiple: int = 16,
        *,
        prompt_format: str,
        camera_pose_alignment: str = "mean",
        verbose: bool = False,
    ):
        assert recon_experiment_name, "NerfBusters eval requires the 3DGRUT reconstruction experiment name"
        self.nerfbusters_dir = Path(nerfbusters_dir)
        self.recon_results_dir = Path(recon_results_dir)
        self.recon_experiment_name = recon_experiment_name
        self.captions_dir = Path(captions_dir)
        self.num_views = num_views
        self.dataset_scaling_factor = dataset_scaling_factor
        self.neighbor_selection_mode = neighbor_selection_mode
        self.checkpoint = checkpoint
        self.image_folder = image_folder  # None means use per-scene mapping
        self.resize_multiple = resize_multiple
        self.prompt_format = prompt_format
        self.camera_pose_alignment = camera_pose_alignment

        self.transforms_by_scene_id = {}
        self.video_frame_indices_by_scene_id = {}
        self.train_ids_by_scene_id = {}
        self.test_ids_by_scene_id = {}
        self.captions_by_scene_id: dict[str, Path] = {}
        self.inference_items: List[Tuple[str, InferencePair]] = []

        # Discover scenes
        scene_dirs = sorted(
            [d for d in self.nerfbusters_dir.iterdir() if d.is_dir() and (d / "transforms.json").exists()]
        )

        for scene_dir in scene_dirs:
            scene_id = scene_dir.name

            if filter_scene_id is not None and scene_id != filter_scene_id:
                continue

            render_dir = self._get_render_dir(scene_id)
            if not render_dir.exists():
                raise FileNotFoundError(f"No reconstruction results for {scene_id} at {render_dir}")

            # Load transforms
            with open(scene_dir / "transforms.json") as f:
                transforms = json.load(f)

            total_frames = len(transforms["frames"])

            # Get train/test split based on filename convention
            # frame_XXXXX.png = TRAIN, frame_1_XXXXX.png = TEST
            train_ids, test_ids = self._get_train_test_split(transforms)

            self.transforms_by_scene_id[scene_id] = transforms
            ordered_indices = self._sort_indices_by_frame_number(transforms, list(range(total_frames)))
            self.video_frame_indices_by_scene_id[scene_id] = {
                source_index: video_index for video_index, source_index in enumerate(ordered_indices)
            }
            self.train_ids_by_scene_id[scene_id] = train_ids
            self.test_ids_by_scene_id[scene_id] = test_ids

            caption_path = self.captions_dir / caption_filename_for_format(f"{scene_id}.h5", self.prompt_format)
            if not caption_path.exists():
                raise FileNotFoundError(f"Scene {scene_id} has no evaluation caption {caption_path}")
            self.captions_by_scene_id[scene_id] = caption_path

            # Get extrinsics for covisibility-based selection
            extrinsics_c2w = None
            if neighbor_selection_mode == NeighborSelectionMode.COVISIBILITY:
                extrinsics_c2w = np.array([transforms["frames"][i]["transform_matrix"] for i in range(total_frames)])

            # Generate inference pairs
            pairs = generate_inference_pairs(
                train_ids=train_ids,
                total_frames=total_frames,
                num_train_context=num_views,
                max_test_frames=max_test_frames,
                selection_mode=neighbor_selection_mode,
                extrinsics_c2w=extrinsics_c2w,
                test_ids=test_ids,
                include_all_frames=include_all_frames,
            )

            for pair in pairs:
                pair.scene_id = scene_id
                for split_pair in self._split_pair_by_frame_number_gaps(transforms, pair):
                    self.inference_items.append((scene_id, split_pair))

        if verbose:
            print(f"Loaded {len(self.transforms_by_scene_id)} scenes, {len(self.inference_items)} inference items")

    def _get_train_test_split(self, transforms: dict) -> tuple[set, set]:
        return nerfbusters_train_test_split(transforms)

    def _extract_frame_number(self, file_path: str) -> int:
        """Extract numeric frame number from filename.

        Examples:
            frame_00001.png -> 1
            frame_1_00186.png -> 186
        """
        basename = os.path.basename(file_path)
        match = re.search(r"frame_(?:1_)?(\d+)\.png$", basename)
        return int(match.group(1)) if match else 0

    def _sort_indices_by_frame_number(self, transforms: dict, indices: List[int]) -> List[int]:
        """Sort frame indices by the frame number in the filename.

        This ensures temporal ordering since transforms.json is NOT in order.
        """
        indexed = [(idx, self._extract_frame_number(transforms["frames"][idx]["file_path"])) for idx in indices]
        indexed.sort(key=lambda x: (x[1], x[0]))
        return [idx for idx, _ in indexed]

    def _split_pair_by_frame_number_gaps(self, transforms: dict, pair: InferencePair) -> List[InferencePair]:
        """Split an inference pair whenever Nerfbusters frame numbers are not consecutive."""
        indexed = sorted(
            enumerate(pair.test_indices),
            key=lambda item: self._extract_frame_number(transforms["frames"][item[1]]["file_path"]),
        )
        if not indexed:
            return []

        segments = []
        current_segment = [indexed[0]]
        for item in indexed[1:]:
            prev_idx = current_segment[-1][1]
            curr_idx = item[1]
            prev_num = self._extract_frame_number(transforms["frames"][prev_idx]["file_path"])
            curr_num = self._extract_frame_number(transforms["frames"][curr_idx]["file_path"])
            if curr_num - prev_num > 1:
                segments.append(current_segment)
                current_segment = []
            current_segment.append(item)
        segments.append(current_segment)

        split_pairs = []
        for segment_idx, segment in enumerate(segments):
            segment_indices = [idx for _, idx in segment]
            if pair.is_test_frame is not None and len(pair.is_test_frame) == len(pair.test_indices):
                is_test_frame = [bool(pair.is_test_frame[orig_pos]) for orig_pos, _ in segment]
            else:
                is_test_frame = [True] * len(segment_indices)
            split_pairs.append(
                InferencePair(
                    neighbor_indices=list(pair.neighbor_indices),
                    test_indices=segment_indices,
                    reversed=pair.reversed,
                    chunk_idx=pair.chunk_idx * 1000 + segment_idx,
                    scene_id=pair.scene_id,
                    is_test_frame=is_test_frame,
                )
            )
        return split_pairs

    def _get_checkpoint_dir(self, scene_id: str) -> Path:
        """3DGRUT outputs to {recon_results_dir}/{scene_id}/{experiment_name}/{scene_id}/ours_{step}/."""
        return self.recon_results_dir / scene_id / self.recon_experiment_name / scene_id / f"ours_{self.checkpoint}"

    def _get_render_dir(self, scene_id: str) -> Path:
        return self._get_checkpoint_dir(scene_id) / "renders"

    def _get_image_folder_for_scene(self, scene_id: str) -> str:
        """Get the appropriate image folder for a scene to get closest to target resolution."""
        if self.image_folder is not None:
            return self.image_folder
        return nerfbusters_image_folder(scene_id)

    def _load_frames_from_scene(
        self,
        scene_id: str,
        frame_indices: List[int],
    ) -> torch.Tensor:
        """Load ground truth frames from nerfbusters dataset."""
        scene_dir = self.nerfbusters_dir / scene_id
        transforms = self.transforms_by_scene_id[scene_id]
        image_folder = self._get_image_folder_for_scene(scene_id)

        frames = []
        for idx in frame_indices:
            file_path = transforms["frames"][idx]["file_path"]
            if self.image_folder is None:
                file_path = nerfbusters_gt_relpath(scene_id, file_path)
            elif file_path.startswith("images/"):
                file_path = file_path.replace("images/", f"{image_folder}/", 1)

            img_path = scene_dir / file_path
            if not img_path.exists():
                raise FileNotFoundError(f"Missing frame: {img_path}")

            image = Image.open(img_path)
            frames.append(np.asarray(image))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tensor = torch.FloatTensor(np.stack(frames)).permute(0, 3, 1, 2) / 255.0

        return tensor

    def __len__(self) -> int:
        return len(self.inference_items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scene_id, pair = self.inference_items[idx]

        transforms = self.transforms_by_scene_id[scene_id]
        # Sort frame indices by frame number for temporal ordering
        # (transforms.json is NOT in temporal order)
        frame_indices = self._sort_indices_by_frame_number(transforms, list(pair.test_indices))
        # Also sort neighbor indices by frame number for consistency
        neighbor_indices = self._sort_indices_by_frame_number(transforms, list(pair.neighbor_indices))

        item = {}

        prompt_payload, caption = load_prompt_payload_from_path(
            self.captions_by_scene_id[scene_id],
            prompt_format=self.prompt_format,
        )
        item.update(prompt_payload)
        item["prompt"] = caption

        # Load rendered images from reconstruction results
        render_dir = self._get_render_dir(scene_id)
        rgb_rendered = load_indexed_frames(render_dir, frame_indices, filename_format="{:05d}.png")
        item["rgb_rendered"] = resize_to_multiple(rgb_rendered, self.resize_multiple)

        # Load ground truth
        rgb_gt = self._load_frames_from_scene(scene_id, frame_indices)
        item["target_h"] = rgb_gt.shape[-2]
        item["target_w"] = rgb_gt.shape[-1]
        item["rgb_gt"] = resize_to_multiple(rgb_gt, self.resize_multiple)

        # Load neighbor frames
        rgb_neighbors = self._load_frames_from_scene(scene_id, neighbor_indices)
        item["rgb_neighbors"] = resize_to_multiple(rgb_neighbors, self.resize_multiple)

        opacity_dir = self._get_checkpoint_dir(scene_id) / "opacity"
        opacity = load_indexed_frames(opacity_dir, frame_indices, filename_format="{:05d}.png", grayscale=True)
        item["opacity"] = resize_to_multiple(opacity, self.resize_multiple).squeeze(1)

        # Compute camera rays
        camera_items = compute_camera_rays(
            transforms=transforms,
            frame_indices=list(frame_indices),
            neighbor_indices=list(neighbor_indices),
            scale=self.dataset_scaling_factor,
            image_shape=item["rgb_rendered"].shape[-2:],
            skip_vae_check=True,  # Eval script handles padding
            pose_alignment=self.camera_pose_alignment,
            include_neighbor_rays=requires_reference_camera_rays(self.prompt_format),
        )
        item.update(camera_items)

        # Metadata
        item["scene_id"] = scene_id
        item["chunk_idx"] = pair.chunk_idx
        item["frame_indices"] = torch.tensor(frame_indices, dtype=torch.long)
        # Train and test filenames can share a frame number. Use their unique
        # position in the full scene's temporal order, including across chunks.
        item["video_frame_indices"] = torch.tensor(
            [self.video_frame_indices_by_scene_id[scene_id][index] for index in frame_indices],
            dtype=torch.long,
        )
        item["neighbor_indices"] = torch.tensor(neighbor_indices, dtype=torch.long)

        # Determine which frames are test frames
        test_ids = self.test_ids_by_scene_id[scene_id]
        is_test = [idx in test_ids for idx in frame_indices]
        item["is_test_frame"] = torch.tensor(is_test, dtype=torch.bool)

        # gt_index contains transforms.json indices for output naming
        # This allows metrics script to correctly look up GT paths via frame_idx_to_path[gt_index]
        # Set to -1 for train frames so they're skipped in individual frame output
        # (they're still included in video output for temporal continuity)
        gt_index = [idx if is_test[i] else -1 for i, idx in enumerate(frame_indices)]
        item["gt_index"] = torch.tensor(gt_index, dtype=torch.long)

        # Valid frames mask
        num_total_frames = item["rgb_gt"].shape[0]
        valid_mask = torch.ones(num_total_frames, dtype=torch.bool)
        item["valid_frames_mask"] = valid_mask

        return item
