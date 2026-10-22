# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The factory glue between run_inference arguments and the DL3DV dataset.

The dataset tests cover filter_scene_ids. This checks the factory handoff:
if it stops forwarding --scene_ids, each evaluation shard silently evaluates
every scene.
The dataset takes one filter parameter, so the factory also maps the CLI's
single --scene_id into it.
"""

from argparse import Namespace
from unittest import mock

from model_eval.datasets import dl3dv_reconstruction_eval


def _factory_args(**overrides):
    args = Namespace(
        evalset="artifixer3d_dl3dv_ours",
        split_path="/data/split.json",
        dl3dv_dir="/data/dl3dv",
        recon_results_dir="/runs/recon",
        prompt_dir="/data/prompts",
        num_views=6,
        dataset_scaling_factor=1.0,
        dl3dv_reconstruction_checkpoint="30000",
        scene_id=None,
        scene_ids=["scene-a", "scene-b"],
        dl3dv_recon_subdir=None,
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        camera_pose_alignment="mean",
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _factory_call_kwargs(args):
    with mock.patch.object(dl3dv_reconstruction_eval, "DL3DVReconstructionEvalDataset") as dataset_cls:
        dl3dv_reconstruction_eval.create_dl3dv_reconstruction_dataset(
            args, selection_mode="evenly_spaced", max_test_frames=None, include_all_frames=True, rank=0
        )
    return dataset_cls.call_args.kwargs


def test_factory_forwards_scene_ids_to_the_dataset():
    kwargs = _factory_call_kwargs(_factory_args())
    assert kwargs["filter_scene_ids"] == ["scene-a", "scene-b"]
    assert "filter_scene_id" not in kwargs
    assert kwargs["recon_subdir"] == "distill_ours_eval"


def test_factory_maps_single_scene_id_into_the_filter():
    kwargs = _factory_call_kwargs(_factory_args(scene_id="scene-only", scene_ids=None))
    assert kwargs["filter_scene_ids"] == ["scene-only"]
