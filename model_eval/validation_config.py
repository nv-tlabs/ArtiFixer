# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.distributed as dist

from model_eval.generation import SamplingConfig
from model_training.model_profiles import is_cosmos3_model


@dataclass
class EvaluationConfig:
    model_id: str
    camera_pose_alignment: str = "mean"
    dataset_scaling_factor: float = 0.01
    validation_metrics: bool = True
    validation_strict: bool = False
    validation_chunk_frames: int = 32
    test_panel_scenes: int = 0
    max_validation_tasks: int | None = None
    frames_per_block: int | None = None
    context_parallel_size: int = 1
    local_attn_size: int | None = None
    sink_size: int | None = None
    vae_chunk_halo_latent_frames: int = 0
    attention_backend: str | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    @classmethod
    def from_args(cls, args):
        # Only evaluation inputs cross this boundary; no optimizers, loaders,
        # checkpoint scheduling or training-algorithm state is retained.
        fields = {name: getattr(args, name) for name in cls.__dataclass_fields__ if hasattr(args, name)}
        fields["validation_strict"] = getattr(args, "validation_strict", False) or getattr(args, "validate_only", False)
        fields["sampling"] = SamplingConfig(
            num_inference_steps=getattr(args, "num_inference_steps", 50),
            vae_chunk_halo_latent_frames=(
                None if is_cosmos3_model(args.model_id) else getattr(args, "vae_chunk_halo_latent_frames", 0)
            ),
        )
        return cls(**fields)


@dataclass
class EvaluationContext:
    device: torch.device
    print: Callable = print
    cp_meshes: dict = field(default_factory=dict)

    @property
    def rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1


@dataclass(frozen=True)
class EvaluationIdentity:
    project_dir: Path
    step: int = 0
    checkpoint: Path | None = None
    initial_checkpoint: Path | None = None
    weights: Literal["raw", "ema", "loaded_checkpoint"] = "raw"
    training_algorithm: str | None = None
