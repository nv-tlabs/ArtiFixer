# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-GPU validation-only integration using reduced real models and synthetic disk assets.

Only model sizes, VAE computation and scheduler construction use offline fixtures.
Pipeline construction, FSDP2 preparation, DCP loading, CP inference, dataset loading,
metric scoring, manifest capture and replay use the production implementation.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from diffusers import UniPCMultistepScheduler
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import FSDPModule

import model_training.distillation as distillation
import model_training.train as train
from model_training.checkpoint_metadata import write_checkpoint_metadata
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID, flow_shift_for_model, is_cosmos3_model
from model_training.utils import train_utils
from model_training.utils.validation_artifacts import validate_manifest
from tests._helpers import _activate_conditioning_adapters
from tests.stage3_smoke import FixtureVae, fixture_transformer
from tests.test_pose_jump_window_filtering import _entry, _write_scene


class ValidationVae(FixtureVae):
    def decode(self, latents, return_dict=False):
        size = self.config.scale_factor_spatial
        video = (
            F.interpolate(
                latents[:, :3].float(),
                size=((latents.shape[2] - 1) * 4 + 1, latents.shape[3] * size, latents.shape[4] * size),
                mode="trilinear",
                align_corners=False,
            )
            .tanh()
            .to(self.dtype)
        )
        return (video,)


def fixture_vae(model_id, device, model_revision=None):
    return ValidationVae(is_cosmos3_model(model_id)).to(device=device, dtype=torch.bfloat16)


def fixture_scheduler(model_id, revision=None):
    return UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=flow_shift_for_model(model_id)
    )


def arguments(root, family, path, replay):
    entry = distillation if path == "kv" else train
    command = [
        "--validate_only",
        "--model_id",
        COSMOS3_NANO_MODEL_ID if family == "cosmos3" else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--base_checkpoint_dir",
        str(root / path / "weights"),
        "--project_dir",
        str(root / path / ("replay" if replay else "capture")),
        "--split_path",
        str(root / "split.json"),
        "--dl3dv_dir",
        str(root / "dl3dv"),
        "--prompt_dir",
        str(root / "prompts"),
        "--default_negative_prompt_path",
        str(root / "negative_prompt.pt"),
        "--num_frames",
        "21",
        "--context_parallel_size",
        "2",
        "--shard_size",
        "2",
        "--test_panel_scenes",
        "1",
        "--max_validation_tasks",
        "3",
        "--log_with",
        "none",
        "--camera_pose_alignment",
        "last" if family == "cosmos3" else "mean",
        "--process_group_timeout_minutes",
        "5",
    ]
    if path == "kv":
        command += ["--frames_per_block", "2", "--local_attn_size", "6", "--sink_size", "2"]
    if replay:
        command += ["--validation_manifest", str(root / path / "capture/val/step_000000/manifest.json")]
    args = entry.build_parser().parse_args(command)
    args.num_inference_steps = 2
    return entry, args


def prepare(root, family):
    root.mkdir(parents=True, exist_ok=False)
    prompt_file = "frames_21_stride_1.h5"
    _write_scene(
        root,
        "scene",
        x_positions=[min(i, 20 - i) * 0.1 for i in range(21)],
        prompt_windows={"w0": list(range(21))},
        splits=(3, 6, 12),
        prompt_filename=prompt_file,
        image_size=64,
    )
    with h5py.File(root / "prompts/960P/scene" / prompt_file, "a") as prompt:
        del prompt["w0"]
        if family == "cosmos3":
            dataset = prompt.create_dataset("w0", data=np.array([3, 4, 5, 6], dtype=np.int32))
            prompt.attrs["negative_input_ids"] = np.array([7, 8, 9, 10], dtype=np.int32)
            prompt.attrs["pad_token_id"] = 0
        else:
            dataset = prompt.create_dataset(
                "w0", data=torch.zeros(5, 32, dtype=torch.bfloat16).view(torch.uint16).numpy()
            )
        dataset.attrs["image_indices"] = np.arange(21)
        dataset.attrs["caption"] = "synthetic validation scene"
    (root / "split.json").write_text(
        json.dumps({"trainval": {}, "test": {"scene": [_entry(root, "scene", "first", 1.0, n) for n in (3, 6, 12)]}})
    )
    torch.save(torch.zeros(1, 5, 32, dtype=torch.bfloat16), root / "negative_prompt.pt")
    for path in ("standard", "kv"):
        _, args = arguments(root, family, path, False)
        pipe = (
            train_utils.get_kv_cache_pipe(args, "cpu")
            if path == "kv"
            else train_utils.get_pipe(args, False, None, "cpu")
        )
        if family == "cosmos3":
            _activate_conditioning_adapters(pipe.transformer)
        with torch.no_grad():
            next(pipe.transformer.parameters()).add_(0.01)
        weights = pipe.transformer.state_dict()
        dcp.save({"model": weights}, checkpoint_id=args.base_checkpoint_dir)
        write_checkpoint_metadata(
            args.base_checkpoint_dir,
            {"camera_pose_alignment": args.camera_pose_alignment},
        )
        torch.save(weights, root / path / "expected.pt")
        datasets = train_utils.get_val_datasets(
            args, SimpleNamespace(is_main_process=False), args.num_frames, 2 if path == "kv" else None
        )
        for dataset in datasets.values():
            assert len(dataset) == 1
            sample = dataset[0]
            assert sample["rgb_gt"].shape == (21, 3, 64, 64)
            assert sample["encoded_prompt"].ndim == (1 if family == "cosmos3" else 2)


def run(root, family, path, replay):
    entry, args = arguments(root, family, path, replay)
    loaded = []
    loader = train_utils.load_model_weights_from_dcp

    def load(model, checkpoint):
        assert isinstance(model, FSDPModule), "validation-only must prepare the real FSDP2 model"
        loader(model, checkpoint)
        full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if dist.get_rank() == 0:
            expected = torch.load(root / path / "expected.pt", weights_only=True)
            assert full.keys() == expected.keys()
            for key in expected:
                torch.testing.assert_close(full[key], expected[key], rtol=0, atol=0)
        loaded.append(model)

    with patch.object(train_utils, "load_model_weights_from_dcp", load):
        entry.main(args)
    assert len(loaded) == 1
    assert loaded[0].cp_mesh.size() == 2
    rank = dist.get_rank()
    phase = "replay" if replay else "capture"
    proof = {
        "family": family,
        "path": path,
        "phase": phase,
        "rank": rank,
        "fsdp2": True,
        "cp_size": 2,
        "exact_dcp_weights": True,
    }
    if rank == 0:
        directory = args.project_dir / "val/step_000000"
        manifest = json.loads((directory / "manifest.json").read_text())
        report = json.loads((directory / "results.json").read_text())
        validate_manifest(manifest)
        assert report["status"] == report["validation"]["status"] == report["panel"]["status"] == "complete"
        assert report["validation"]["complete_tasks"] == 3
        assert report["panel"]["complete_tasks"] == 2
        assert {task["cp_size"] for task in report["validation"]["tasks"].values()} == {2}
        if replay:
            original = json.loads((root / path / "capture/val/step_000000/results.json").read_text())
            assert original["manifest_id"] == report["manifest_id"]
            for section in ("validation", "panel"):
                for key, result in report[section]["tasks"].items():
                    # The manifest pins three-reference selections. The 6/12-view
                    # clips must be complete, but their reference order is not replayed.
                    if section == "validation" and key != "default_3":
                        continue
                    before = original[section]["tasks"][key]
                    assert result["sample"] == before["sample"]
                    assert result["noise_seed"] == before["noise_seed"]
                    for metric in ("psnr", "ssim", "temporal_delta_l1"):
                        assert abs(result["metrics"][metric] - before["metrics"][metric]) < 1e-5, (section, key, metric)
        proof["manifest_id"] = report["manifest_id"]
    dist.barrier()
    (root / path / f"{phase}-rank{rank}.json").write_text(json.dumps(proof, indent=2) + "\n")
    dist.barrier()
    print("VALIDATION_SMOKE_PASS " + json.dumps(proof), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--family", choices=("wan", "cosmos3"), required=True)
    parser.add_argument("--path", choices=("standard", "kv"), default="standard")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        stack.enter_context(patch.object(train_utils, "_transformer_from_config", fixture_transformer))
        stack.enter_context(patch.object(train_utils, "_vae_for_model", fixture_vae))
        stack.enter_context(patch.object(train_utils, "_scheduler_for_model", fixture_scheduler))
        if args.prepare:
            prepare(args.root, args.family)
        else:
            run(args.root, args.family, args.path, args.replay)


if __name__ == "__main__":
    main()
