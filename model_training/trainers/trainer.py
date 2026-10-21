# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
from accelerate import Accelerator
from torch import nn

from model_training.model_profiles import flow_shift_for_model
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.schedulers.flow_match import FlowMatchScheduler
from model_training.trainers.trainer_base import TrainerBase


class Trainer(TrainerBase):

    pipe: ArtifixerPipelineBase

    def __init__(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataloader: torch.utils.data.DataLoader,
        val_datasets: dict[int, torch.utils.data.Dataset],
        pipe: ArtifixerPipelineBase,
        frames_per_block: int | None,
        step_offset: int = 0,
    ):
        super().__init__(args, accelerator, pipe, [optimizer], [scheduler], train_dataloader, val_datasets, step_offset)
        self.frames_per_block = frames_per_block

        self.train_scheduler = FlowMatchScheduler(
            shift=flow_shift_for_model(args.model_id),
            sigma_min=0.0,
            extra_one_step=True,
            device=self.pipe.vae.device,
        )

    def training_batch(self, data: dict) -> tuple[torch.Tensor, dict]:
        # Per-block supervised objective (diffusion forcing); the bidirectional
        # stage overrides this in ContextParallelTrainer.
        assert self.frames_per_block is not None
        data = self.process_inputs(data)
        batch_size = data["rgb_gt"].shape[0]
        # Flow-matching source z_mix (paper eq. 3): the degraded latent where
        # opacity is high, Gaussian noise elsewhere — not pure noise.
        source_latent = self.pipe.prepare_latents(data["condition"], data["opacity"], True)
        latent_gt = self.pipe.encode_video_frames(data["rgb_gt"])

        timestep_id = torch.randint(
            0,
            self.train_scheduler.num_train_timesteps,
            (batch_size * latent_gt.shape[2] // self.frames_per_block,),
        )
        timestep = self.train_scheduler.timesteps[timestep_id]
        model_input = self.train_scheduler.add_noise_multi_timestep(latent_gt, source_latent, timestep).to(
            latent_gt.dtype
        )

        with torch.no_grad():
            training_target = self.train_scheduler.training_target(latent_gt, source_latent)

        transformer_timestep = self.pipe.prepare_transformer_timestep(
            timestep,
            model_input,
            frames_per_block=self.frames_per_block,
        )

        prompt_kwargs = self._prompt_kwargs(data)
        noise_pred = self.pipe.transformer(
            hidden_states=model_input,
            timestep=transformer_timestep,
            neighbor_hidden_states=data["neighbors_condition"],
            ignore_neighbors=data["ignore_neighbors"],
            opacity=data["opacity"],
            camera_rays=data["camera_rays"],
            w2cs=data["w2cs"],
            neighbor_w2cs=data["neighbor_w2cs"],
            Ks=data["Ks"],
            neighbor_Ks=data["neighbor_Ks"],
            **self._neighbor_camera_kwargs(data.get("neighbor_camera_rays")),
            return_dict=False,
            **prompt_kwargs,
        )[0]

        mse_loss = torch.nn.functional.mse_loss(
            noise_pred.float(),
            training_target.float(),
            reduction="none",
        )
        # Report a detached tensor, not .item(): the train loop converts it to
        # a float after backward is queued, so the host does not block on the
        # forward pass.
        metrics = {
            "mse_loss": mse_loss.detach().mean(),
        }
        weights = self.train_scheduler.training_weight_multi_timestep(timestep)
        weights = weights.repeat_interleave(mse_loss.shape[0] * mse_loss.shape[2] // weights.shape[0]).view(
            mse_loss.shape[0], 1, mse_loss.shape[2], 1, 1
        )
        loss = (mse_loss * weights).mean()

        return loss, metrics

    def to_accumulate(self) -> list[nn.Module]:
        return [self.pipe.transformer]

    def on_before_optimizer_step(self) -> None:
        # https://github.com/huggingface/accelerate/issues/3789
        self.accelerator.unscale_gradients()
        torch.nn.utils.clip_grad_norm_(self.pipe.transformer.parameters(), self.args.max_grad_norm)

    def on_before_zero_grad(self) -> None:
        pass
