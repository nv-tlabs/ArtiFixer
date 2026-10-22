# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from model_training.common.context_parallel import cp_rank_temporal_ranges
from model_training.data.dl3dv_base import DL3DVPairedDatasetBase
from model_training.prompt_formats import WAN_T5_PROMPT_FORMAT


class DL3DVPairedDatasetTrain(DL3DVPairedDatasetBase):

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        prompt_dir: Path,
        num_frames: int,
        frames_per_block: int,
        num_fetches: int,
        dataset_scaling_factor: float,
        max_neighbors: int = 12,
        verbose: bool = False,
        resize_multiple: int = 16,
        prompt_format: str = WAN_T5_PROMPT_FORMAT,
        context_parallel_size: int = 1,
        vae_chunk_halo_latent_frames: int = 0,
        camera_pose_alignment: str = "mean",
    ):
        """
        num_fetches is the TOTAL number of DataLoader fetches the training run will perform
        across every rank (num_fetches = max_iterations × gradient_accumulation_steps ×
        num_processes). CP ranks in the same group consume one shared source sample, so the
        random fetch plan is sized by logical CP-group fetches while __len__ remains sized
        for Accelerate's per-rank sharding.
        """
        assert context_parallel_size >= 1, "context_parallel_size must be positive"
        assert vae_chunk_halo_latent_frames >= 0, "vae_chunk_halo_latent_frames must be non-negative"
        assert (
            num_fetches % context_parallel_size == 0
        ), f"num_fetches {num_fetches} must divide by context_parallel_size {context_parallel_size}"
        super().__init__(
            split=split,
            split_path=split_path,
            dl3dv_dir=dl3dv_dir,
            prompt_dir=prompt_dir,
            num_frames=num_frames,
            frames_per_block=frames_per_block,
            return_unencoded_prompt=False,
            dataset_scaling_factor=dataset_scaling_factor,
            max_neighbors=max_neighbors,
            verbose=verbose,
            filter_pose_jump_windows=True,
            resize_multiple=resize_multiple,
            prompt_format=prompt_format,
            camera_pose_alignment=camera_pose_alignment,
        )

        assert self.available_splits, "available_splits is empty"
        assert all(s > 0 for s in self.available_splits), f"available_splits must be positive: {self.available_splits}"
        assert all(
            a < b for a, b in zip(self.available_splits, self.available_splits[1:])
        ), f"available_splits must be strictly increasing: {self.available_splits}"

        # intervals[i] = (lo, hi) inclusive range of num_neighbors for split available_splits[i].
        # Each split "owns" neighbor counts between its predecessor (exclusive) and itself (inclusive),
        # with the first split starting at 1. For (2, 3, 6, 12) this gives (1,2), (3,3), (4,6), (7,12).
        prev = [0, *self.available_splits[:-1]]
        self._intervals = torch.tensor([(lo + 1, hi) for lo, hi in zip(prev, self.available_splits)], dtype=torch.int64)

        self.num_fetches = num_fetches
        self.context_parallel_size = context_parallel_size
        self.vae_chunk_halo_latent_frames = vae_chunk_halo_latent_frames
        self.num_group_fetches = num_fetches // context_parallel_size
        # Populated by set_random_splits; shape (num_group_fetches, 2), cols = (split, num_neighbors).
        self.iter_plan: torch.Tensor | None = None
        # Populated by set_random_splits; shape (len(self.data),) permutation of scene indices.
        self.data_perm: torch.Tensor | None = None
        self.random_seed: int | None = None

    def set_random_splits(self, device: torch.device | str, seed: int) -> None:
        """Pre-sample one (split, num_neighbors) pair per logical CP-group fetch and a one-time scene
        permutation, then broadcast rank-0's values to all ranks so every rank agrees on the plan."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        iter_plan = torch.empty((self.num_group_fetches, 2), dtype=torch.int64, device=device)
        data_perm = torch.empty((len(self.data),), dtype=torch.int64, device=device)

        if rank == 0:
            generator = torch.Generator().manual_seed(seed)
            split_idx = torch.randint(
                0, len(self.available_splits), (self.num_group_fetches,), generator=generator, dtype=torch.int64
            )
            lo = self._intervals[split_idx, 0]
            hi = self._intervals[split_idx, 1]
            # Uniform integer sampling from [lo, hi] inclusive: continuous uniform [0, 1) mapped to
            # integer offsets in [0, hi - lo] via floor, then shifted by lo.
            num_neighbors = lo + (torch.rand(self.num_group_fetches, generator=generator) * (hi - lo + 1)).long()
            iter_plan.copy_(torch.stack([hi, num_neighbors], dim=1))
            # Scene permutation decouples the slight over-representation of scenes in
            # [0, num_group_fetches mod len(self.data)) from self.data's default sort order so the
            # over-weighted subset rotates with the seed. Same generator → deterministic across
            # ranks' broadcast views.
            data_perm.copy_(torch.randperm(len(self.data), generator=generator))

        if dist.is_initialized():
            dist.broadcast(iter_plan, src=0)
            dist.broadcast(data_perm, src=0)

        self.iter_plan = iter_plan.cpu()
        self.data_perm = data_perm.cpu()
        self.random_seed = int(seed)

    def __len__(self) -> int:
        return self.num_fetches

    def _group_fetch_index(self, idx: int) -> int:
        return idx // self.context_parallel_size

    def _cp_rank_for_index(self, idx: int) -> int:
        return idx % self.context_parallel_size

    def _fetch_generator(self, group_idx: int) -> torch.Generator:
        if self.random_seed is None:
            raise RuntimeError("set_random_splits() must be called before iterating the dataset")
        generator = torch.Generator()
        generator.manual_seed(self.random_seed + group_idx)
        return generator

    def _cp_temporal_slice(self, cp_rank: int) -> dict[str, tuple[int, int] | tuple[int, int, int, int]] | None:
        if self.context_parallel_size <= 1:
            return None
        return cp_rank_temporal_ranges(
            cp_rank,
            self.context_parallel_size,
            total_raw_frames=self.num_frames,
            temporal_scale=4,
            halo_latent_frames=self.vae_chunk_halo_latent_frames,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.iter_plan is None or self.data_perm is None:
            raise RuntimeError("set_random_splits() must be called before iterating the dataset")

        group_idx = self._group_fetch_index(idx)
        chosen_split, num_neighbors = self.iter_plan[group_idx].tolist()
        temporal_slice = self._cp_temporal_slice(self._cp_rank_for_index(idx))
        # Sampling is a function of the logical fetch, including CP=1. Worker
        # RNG state is not restored when Accelerate skips batches on resume.
        generator = self._fetch_generator(group_idx)
        # A scene with pose jumps sits out draws it cannot serve from one
        # jump-free segment; the next viable scene in the shuffled order
        # serves the fetch instead, so __len__ stays fixed for accelerate's
        # plan-slot bookkeeping (the replacement scene is served twice that
        # epoch). The scan depends only on group-level state, so every CP
        # rank in a group lands on the same scene.
        for attempt in range(len(self.data)):
            data_idx = self.data_perm[(group_idx + attempt) % len(self.data)].item()
            if not self.can_supply_same_segment_neighbors(data_idx, chosen_split, num_neighbors):
                continue
            return self.get_item_inner(
                data_idx,
                chosen_split,
                num_neighbors,
                generator=generator,
                temporal_slice=temporal_slice,
            )
        raise RuntimeError(
            f"No train scene could serve split {chosen_split}: every scene lacks a jump-free "
            f"segment with {num_neighbors} training views."
        )
