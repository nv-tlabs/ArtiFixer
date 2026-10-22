# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""get_train_dataloader with the REAL DL3DVPairedDatasetTrain.

Every other test of this entry point fakes the dataset class, so a kwarg
added at the call site but missing from DL3DVPairedDatasetTrain.__init__
passes the whole suite and kills every rank of a real launch before step 1.
This test builds the real dataset from a tiny synthetic scene and pulls one
batch through the DataLoader."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from model_training.utils.train_utils import get_train_dataloader
from tests.test_pose_jump_window_filtering import NUM_FRAMES, _entry, _write_scene, _write_split


@pytest.mark.requires_host_ffmpeg
def test_get_train_dataloader_builds_and_serves_the_real_train_dataset(tmp_path):
    _write_scene(
        tmp_path,
        "scene",
        x_positions=[0.05 * i for i in range(NUM_FRAMES)],
        prompt_windows={"w0": [0, 1, 2, 3, 4]},
    )
    split_path = _write_split(tmp_path, {"scene": [_entry(tmp_path, "scene", "first", 1.0)]})
    args = argparse.Namespace(
        max_iterations=1,
        gradient_accumulation_steps=1,
        split_path=split_path,
        dl3dv_dir=tmp_path / "dl3dv",
        prompt_dir=tmp_path / "prompts",
        num_frames=NUM_FRAMES,
        dataset_scaling_factor=0.01,
        # A Wan2.1 id keeps resize_multiple at 16, matching the synthetic
        # scene's 16x16 frames.
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        context_parallel_size=1,
        vae_chunk_halo_latent_frames=0,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
        seed=123,
        camera_pose_alignment="mean",
    )
    accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"), is_main_process=False)

    dataloader = get_train_dataloader(args, accelerator, frames_per_block=None)
    batch = next(iter(dataloader))

    assert batch["rgb_gt"].shape == (1, NUM_FRAMES, 3, 16, 16)


@pytest.mark.requires_host_ffmpeg
@pytest.mark.parametrize("workers", [0, 2])
def test_native_dataset_resume_replays_frames_prompts_and_neighbors(tmp_path, isolated_accelerator_state, workers):
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    from model_training.utils.train_utils import resume_training_from_checkpoint

    _write_scene(
        tmp_path,
        "scene",
        x_positions=[0.05 * i for i in range(13)],
        prompt_windows={"first": [0, 1, 2, 3, 4], "last": [8, 9, 10, 11, 12]},
        selected_indices=[1, 10],
    )
    split_path = _write_split(tmp_path, {"scene": [_entry(tmp_path, "scene", "first", 1.0)]})
    args = argparse.Namespace(
        max_iterations=8,
        gradient_accumulation_steps=1,
        split_path=split_path,
        dl3dv_dir=tmp_path / "dl3dv",
        prompt_dir=tmp_path / "prompts",
        num_frames=NUM_FRAMES,
        dataset_scaling_factor=0.01,
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        context_parallel_size=1,
        vae_chunk_halo_latent_frames=0,
        dataloader_num_workers=workers,
        dataloader_prefetch_factor=2,
        seed=123,
        camera_pose_alignment="mean",
    )
    project = tmp_path / "training"
    accelerator = Accelerator(
        cpu=True,
        project_config=ProjectConfiguration(
            project_dir=project,
            automatic_checkpoint_naming=True,
        ),
    )
    loader = accelerator.prepare_data_loader(get_train_dataloader(args, accelerator, None))
    iterator = iter(loader)
    for _ in range(3):
        with accelerator.accumulate():
            next(iterator)
    accelerator.project_configuration.iteration = 3
    checkpoint = accelerator.save_state()
    expected = list(iterator)
    assert len(expected) == 5

    resumed_accelerator = Accelerator(cpu=True, project_config=ProjectConfiguration(project_dir=project))
    resumed_loader = resumed_accelerator.prepare_data_loader(get_train_dataloader(args, resumed_accelerator, None))
    args.resume_from_checkpoint = checkpoint
    resumed_loader, state = resume_training_from_checkpoint(args, resumed_accelerator, resumed_loader)
    actual = list(resumed_loader)
    assert state.step_offset == 0
    assert len(actual) == len(expected)
    for restored, continuous in zip(actual, expected, strict=True):
        torch.testing.assert_close(restored, continuous, rtol=0, atol=0)
    # The fixture must not reduce replay to a constant frame window.
    assert any(not torch.equal(expected[0]["rgb_gt"], batch["rgb_gt"]) for batch in expected[1:])
