# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, Dataset

import model_training.distillation as entry
import model_training.utils.train_utils as train_utils
from model_training import stages
from model_training.checkpoint_metadata import write_checkpoint_metadata
from model_training.common.context_parallel import cp_rank_temporal_ranges
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID, is_cosmos3_model
from model_training.trainers.dmd import DMD
from tests._helpers import REDUCED_CONFIG, _activate_conditioning_adapters, _joint_reference_kwargs
from tests.test_wan_rollout_cache import _REDUCED_WAN_CONFIG, _rollout_inputs


class FixtureVae(torch.nn.Module):
    def __init__(self, cosmos):
        super().__init__()
        channels, spatial = (48, 16) if cosmos else (16, 8)
        self.register_buffer("anchor", torch.empty(0))
        self.config = SimpleNamespace(
            scale_factor_temporal=4,
            scale_factor_spatial=spatial,
            latents_mean=[0.0] * channels,
            latents_std=[1.0] * channels,
            z_dim=channels,
        )

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def encode(self, rgb):
        spatial = self.config.scale_factor_spatial
        latent = F.avg_pool3d(rgb.float(), (1, spatial, spatial))[:, :, ::4].mean(dim=1, keepdim=True)
        latent = latent.expand(-1, self.config.z_dim, -1, -1, -1).contiguous().to(self.dtype)
        return SimpleNamespace(latents=latent)


class FixtureDataset(Dataset):
    def __init__(self, cosmos, context_parallel_size, vae_chunk_halo_latent_frames):
        self.context_parallel_size = context_parallel_size
        self.vae_chunk_halo_latent_frames = vae_chunk_halo_latent_frames
        if cosmos:
            inputs = _joint_reference_kwargs(total_frames=6)
            self.data = {
                key: inputs[key][0]
                for key in (
                    "opacity",
                    "camera_rays",
                    "w2cs",
                    "Ks",
                    "neighbor_w2cs",
                    "neighbor_Ks",
                    "neighbor_camera_rays",
                )
            }
            self.data.update(
                encoded_prompt=inputs["encoder_hidden_states"][0],
                prompt_attention_mask=inputs["encoder_attention_mask"][0],
                encoded_negative_prompt=torch.tensor([9, 10, 11, 12]),
                negative_prompt_attention_mask=torch.ones(4, dtype=torch.bool),
            )
            size, neighbors = 64, 3
        else:
            inputs = _rollout_inputs(6, opacity_value=None, with_neighbors=True)
            self.data = {key: inputs[key][0] for key in ("camera_rays", "w2cs", "Ks", "neighbor_w2cs", "neighbor_Ks")}
            self.data.update(
                opacity=inputs["rendered_opacity"][0],
                encoded_prompt=inputs["encoded_prompt"][0].to(torch.bfloat16),
            )
            size, neighbors = 32, 1
        generator = torch.Generator().manual_seed(42)
        rgb = torch.rand(21, 3, size, size, generator=generator)
        self.data.update(
            rgb_rendered=rgb,
            rgb_gt=rgb.clone(),
            rgb_neighbors=torch.rand(neighbors, 3, size, size, generator=generator),
        )

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if self.context_parallel_size == 1:
            return self.data
        raw_frames = self.data["rgb_rendered"].shape[0]
        ranges = cp_rank_temporal_ranges(
            index % self.context_parallel_size,
            self.context_parallel_size,
            total_raw_frames=raw_frames,
            temporal_scale=4,
            halo_latent_frames=self.vae_chunk_halo_latent_frames,
        )
        raw_start, raw_end, crop_start, crop_end = ranges["vae_raw"]
        opacity_start, opacity_end = ranges["conditioning_raw"]
        start, end = ranges["latent"]
        local = dict(self.data)
        for key in ("rgb_rendered", "rgb_gt"):
            local[key] = self.data[key][raw_start:raw_end]
        local["opacity"] = self.data["opacity"][opacity_start:opacity_end]
        for key in ("camera_rays", "w2cs", "Ks"):
            local[key] = self.data[key][start:end]
        local.update(
            _cp_data_sharded=torch.tensor(True),
            _cp_vae_crop=torch.tensor([crop_start, crop_end]),
            _cp_vae_raw_start=torch.tensor(raw_start),
            _cp_conditioning_raw_start=torch.tensor(opacity_start),
            _cp_global_raw_frames=torch.tensor(raw_frames),
        )
        return local


def fixture_transformer(cls, model_id, model_revision=None):
    torch.manual_seed(0)
    return cls(**(REDUCED_CONFIG if is_cosmos3_model(model_id) else _REDUCED_WAN_CONFIG))


def fixture_vae(model_id, device, model_revision=None):
    return FixtureVae(is_cosmos3_model(model_id)).to(device=device, dtype=torch.bfloat16)


class ReportingDMD(DMD):
    instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ReportingDMD.instance = self
        self.smoke_updates = []
        assert isinstance(self.pipe.transformer, FSDPModule)
        assert isinstance(self.fake_score, FSDPModule)
        assert isinstance(self.real_score, FSDPModule)
        print(
            f"NORMAL_STAGE3_READY rank={self.accelerator.process_index} distributed={self.accelerator.distributed_type} cp={self.args.context_parallel_size}",
            flush=True,
        )

    def training_batch(self, data):
        loss, metrics = super().training_batch(data)
        self.smoke_metrics = {key: float(value) for key, value in metrics.items()}
        return loss, metrics

    def on_before_optimizer_step(self):
        super().on_before_optimizer_step()
        record = {"step": self.get_step(), "student_phase": self.is_student_phase(), **self.smoke_metrics}
        for role, model in (("student", self.pipe.transformer), ("critic", self.fake_score)):
            squared_norm = torch.zeros((), device=self.accelerator.device, dtype=torch.float64)
            present = 0
            for parameter in model.parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
                assert torch.isfinite(gradient).all(), role
                squared_norm += gradient.detach().double().square().sum()
                present += 1
            dist.all_reduce(squared_norm)
            record[f"{role}_grad_norm"] = float(squared_norm.sqrt())
            record[f"{role}_gradient_tensors"] = present
        assert record["critic_grad_norm"] > 0
        if record["student_phase"]:
            assert record["student_gradient_tensors"] > 0
        self.smoke_updates.append(record)
        print(json.dumps({"rank": self.accelerator.process_index, **record}), flush=True)


def arguments(root, family, cp_size):
    return entry.build_parser().parse_args(
        [
            "--model_id",
            COSMOS3_NANO_MODEL_ID if family == "cosmos3" else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "--project_dir",
            str(root / f"run-cp{cp_size}"),
            "--base_checkpoint_dir",
            str(root / "student"),
            "--base_checkpoint_dir_critic",
            str(root / "critic"),
            "--split_path",
            str(root / "split.json"),
            "--dl3dv_dir",
            str(root),
            "--prompt_dir",
            str(root),
            "--default_negative_prompt_path",
            str(root / "negative_prompt.pt"),
            "--num_frames",
            "21",
            "--frames_per_block",
            "2",
            "--local_attn_size",
            "6",
            "--sink_size",
            "2",
            "--context_parallel_size",
            str(cp_size),
            "--shard_size",
            "2",
            "--max_iterations",
            "6",
            "--save_steps",
            "1000",
            "--validation_steps",
            "1000",
            "--dropout_rate",
            "0",
            "--dataloader_num_workers",
            "0",
            "--log_with",
            "none",
            "--camera_pose_alignment",
            "last" if family == "cosmos3" else "mean",
            "--process_group_timeout_minutes",
            "5",
            "--check_state_finite",
        ]
    )


def main(root, family, prepare, cp_size):
    args = arguments(root, family, cp_size)
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(patch.object(train_utils, "_transformer_from_config", fixture_transformer))
        stack.enter_context(patch.object(stages, "_transformer_from_config", fixture_transformer))
        stack.enter_context(patch.object(train_utils, "_vae_for_model", fixture_vae))
        if prepare:
            pipe = train_utils.get_kv_cache_pipe(args, "cpu")
            if family == "cosmos3":
                _activate_conditioning_adapters(pipe.transformer)
            dcp.save({"model": pipe.transformer.state_dict()}, checkpoint_id=root / "student")
            critic = stages.build_score_transformer(args, args.model_id, pipe, gradient_checkpointing=True)
            dcp.save({"model": critic.state_dict()}, checkpoint_id=root / "critic")
            for slot in ("student", "critic"):
                write_checkpoint_metadata(root / slot, {"camera_pose_alignment": args.camera_pose_alignment})
            torch.save(torch.zeros(1, 5, 32, dtype=torch.bfloat16), root / "negative_prompt.pt")
            print(f"PREPARED {family} {root}", flush=True)
            return
        stack.enter_context(
            patch.object(
                stages,
                "get_train_dataloader",
                lambda *unused: DataLoader(
                    FixtureDataset(family == "cosmos3", args.context_parallel_size, args.vae_chunk_halo_latent_frames),
                    batch_size=1,
                    num_workers=0,
                ),
            )
        )
        stack.enter_context(patch.object(stages, "get_val_datasets", lambda *unused: {}))
        stack.enter_context(patch.object(stages, "DMD", ReportingDMD))
        entry.main(args)
        trainer = ReportingDMD.instance
        assert trainer.get_step() == 6
        assert len(trainer.smoke_updates) == 6
        assert [update["step"] for update in trainer.smoke_updates if update["student_phase"]] == [1, 6]
        assert trainer.smoke_updates[-1]["student_grad_norm"] > 0
        report = {
            "family": family,
            "cp_size": cp_size,
            "rank": trainer.accelerator.process_index,
            "passed": True,
            "updates": trainer.smoke_updates,
        }
        (root / f"smoke-cp{cp_size}-rank{trainer.accelerator.process_index}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(f"STAGE3_SMOKE_PASS {family} cp={cp_size} rank={trainer.accelerator.process_index}", flush=True)
        trainer.accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--family", choices=("wan", "cosmos3"), required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cp-size", type=int, default=2)
    options = parser.parse_args()
    main(options.root, options.family, options.prepare, options.cp_size)
