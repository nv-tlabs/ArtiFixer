# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn

from model_training.common.context_parallel import conditioning_raw_frame_bounds
from model_training.model_profiles import is_cosmos3_model
from model_training.net.conditioning import pad_opacity_first_frame
from model_training.net.context_parallel import all_gather_cat
from model_training.net.wan_transformer import ArtifixerTransformer
from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline
from model_training.trainers.context_parallel import ContextParallelVideoMixin
from model_training.trainers.trainer_base import TrainerBase
from model_training.utils.ema import ModelEMA


class DMD(ContextParallelVideoMixin, TrainerBase):

    pipe: ArtifixerKvCachePipeline

    def __init__(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        optimizers: list[torch.optim.Optimizer],
        schedulers: list[torch.optim.lr_scheduler.LRScheduler],
        train_dataloader: torch.utils.data.DataLoader,
        val_datasets: dict[int, torch.utils.data.Dataset],
        pipe: ArtifixerKvCachePipeline,
        real_score: ArtifixerTransformer,
        fake_score: ArtifixerTransformer,
        step_offset: int = 0,
    ):
        super().__init__(args, accelerator, pipe, optimizers, schedulers, train_dataloader, val_datasets, step_offset)
        self.real_score = real_score
        self.fake_score = fake_score
        self.ema = ModelEMA(
            pipe.transformer,
            accelerator,
            self.get_step,
            ema_weight=args.ema_weight,
            ema_start_step=args.ema_start_step,
            initialize_after_updates=0,
            training_stage="dmd",
            camera_pose_alignment=args.camera_pose_alignment,
        )
        self.negative_prompt = (
            None
            if is_cosmos3_model(args.model_id)
            else torch.load(args.default_negative_prompt_path, weights_only=True).to(pipe.vae.device)
        )

        self.min_step = int(0.02 * self.pipe.scheduler.num_train_timesteps)
        self.max_step = int(0.98 * self.pipe.scheduler.num_train_timesteps)

    def _to_device(self, value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is None:
            return value.to(device=self.accelerator.device, non_blocking=True)
        return value.to(device=self.accelerator.device, dtype=dtype, non_blocking=True)

    def _gather_cp_conditioning(self, tensor: torch.Tensor, frames_per_chunk: int) -> torch.Tensor:
        cp_mesh = self._cp_mesh()
        assert cp_mesh is not None
        cp_size = cp_mesh.size()
        local_frames = tensor.shape[1]
        output = tensor.new_empty((tensor.shape[0], local_frames * cp_size, *tensor.shape[2:]), device="cpu")
        for start in range(0, local_frames, frames_per_chunk):
            chunk = tensor[:, start : start + frames_per_chunk]
            gathered = all_gather_cat(self._to_device(chunk), cp_mesh, dim=1).cpu()
            for rank, shard in enumerate(gathered.chunk(cp_size, dim=1)):
                output_start = rank * local_frames + start
                output[:, output_start : output_start + chunk.shape[1]].copy_(shard)
        return output

    def process_inputs(self, data: dict) -> dict[str, torch.Tensor | bool]:
        # DMD keeps three Wan transformers resident. Leave the raw video batch on
        # CPU and move only encoded/chunked tensors to GPU; otherwise Accelerate's
        # default full-batch device copy OOMs before CP can reduce temporal memory.
        # One dropout decision per CP group.  The generator sample, real critic,
        # and fake critic all consume the same conditioning, so adjacent temporal
        # shards must not independently drop prompts/rendered suffixes.
        processed_data = super().process_inputs(data)
        processed_data.pop("rgb_gt")
        processed_data.pop("_cp_vae_crop", None)
        cp_mesh = self._cp_mesh()
        if cp_mesh is not None:
            assert (
                data["rgb_gt"].shape[0] == processed_data["condition"].shape[0] == 1
            ), "DMD CP expects a per-rank batch size of 1"
            assert (
                self._cp_batch_scalar(data["_cp_global_raw_frames"]) == self.args.num_frames
            ), "DMD CP global raw frames must match num_frames"
            local_frames = processed_data["condition"].shape[2]
            latent_start = self._local_frame_start(local_frames)
            temporal_scale = self.pipe.vae.config.scale_factor_temporal
            expected_raw_start, raw_end = conditioning_raw_frame_bounds(
                latent_start, latent_start + local_frames, temporal_scale
            )
            raw_start = self._cp_batch_scalar(data["_cp_conditioning_raw_start"])
            assert (
                raw_start == expected_raw_start
            ), f"DMD CP conditioning starts at raw frame {raw_start}, expected {expected_raw_start}"
            for key in ("opacity", "camera_rays", "w2cs", "Ks"):
                expected_frames = raw_end - raw_start if key == "opacity" else local_frames
                assert processed_data[key].shape[:2] == (1, expected_frames), (
                    f"DMD CP {key} must have batch size 1 and {expected_frames} frames, "
                    f"got shape {tuple(processed_data[key].shape)}"
                )

            processed_data["condition"] = all_gather_cat(processed_data["condition"], cp_mesh, dim=2)
            opacity = processed_data["opacity"]
            if raw_start == 0:
                opacity = pad_opacity_first_frame(opacity, temporal_scale)
            processed_data["opacity"] = self._gather_cp_conditioning(
                opacity, self.pipe.ar_frames_per_chunk * temporal_scale
            )[:, temporal_scale - 1 :]
            processed_data["camera_rays"] = self._gather_cp_conditioning(
                processed_data["camera_rays"], self.pipe.ar_frames_per_chunk
            )
            for key in ("w2cs", "Ks"):
                processed_data[key] = all_gather_cat(self._to_device(processed_data[key]), cp_mesh, dim=1).cpu()

        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor) and key not in ("opacity", "camera_rays", "w2cs", "Ks"):
                processed_data[key] = self._to_device(value)
        return processed_data

    def training_batch(self, data: dict) -> tuple[torch.Tensor, dict]:
        # Process inputs once so critic and generator see the same dropout masks
        # and conditioning — matching the official DMD2 implementation where
        # guidance_data_dict is shared between turns.
        data = self.process_inputs(data)

        student_phase = self.is_student_phase()

        # Generate x_fake once; shared between critic (detached) and generator.
        # Grad flows only on student phases.
        prompt_generation_kwargs = {}
        if "prompt_attention_mask" in data:
            prompt_generation_kwargs["prompt_attention_mask"] = data["prompt_attention_mask"]
        generate_context = torch.enable_grad() if student_phase else torch.no_grad()
        with generate_context:
            prediction = self.pipe.generate_samples_from_batch(
                data["condition"],
                data["opacity"],
                data["neighbors_condition"],
                data["camera_rays"],
                data["w2cs"],
                data["neighbor_w2cs"],
                data["Ks"],
                data["neighbor_Ks"],
                data["encoded_prompt"],
                self.args.num_inference_steps,
                True,
                ignore_neighbors=data["ignore_neighbors"],
                **self._neighbor_camera_kwargs(data.get("neighbor_camera_rays")),
                **prompt_generation_kwargs,
            )

        critic_loss = self.critic_loss(data, prediction.detach())
        loss = critic_loss
        # Detached tensors, not .item(): the train loop converts to floats after
        # backward is enqueued so the host does not sync on the forward pass.
        metrics = {"critic_loss": critic_loss.detach()}
        if student_phase:
            generator_loss = self.generator_loss(data, prediction)
            loss = loss + generator_loss
            metrics["generator_loss"] = generator_loss.detach()

        return loss, metrics

    def _score_batch(self, data: dict, prediction: torch.Tensor) -> tuple[dict, torch.Tensor, int, bool]:
        cp_rank, cp_size = self._cp_rank_size()
        local_data = dict(data)
        if data.get("neighbor_camera_rays") is not None:
            local_data["neighbor_camera_rays"] = self._to_device(data["neighbor_camera_rays"], self.pipe.vae.dtype)
        if cp_size == 1:
            local_data["condition"] = self._to_device(local_data["condition"])
            local_data["opacity"] = self._to_device(local_data["opacity"], self.pipe.vae.dtype)
            local_data["camera_rays"] = self._to_device(local_data["camera_rays"], self.pipe.vae.dtype)
            local_data["w2cs"] = self._to_device(local_data["w2cs"])
            local_data["Ks"] = self._to_device(local_data["Ks"])
            local_data["neighbor_w2cs"] = self._to_device(local_data["neighbor_w2cs"])
            local_data["neighbor_Ks"] = self._to_device(local_data["neighbor_Ks"])
            local_data["encoded_prompt"] = self._to_device(local_data["encoded_prompt"])
            local_data["neighbors_condition"] = self._to_device(local_data["neighbors_condition"])
            return local_data, prediction, 0, False

        latent_frames = prediction.shape[2]
        assert (
            latent_frames % cp_size == 0
        ), f"DMD score latent frames ({latent_frames}) must divide by CP size ({cp_size})"
        local_frames = latent_frames // cp_size
        latent_start = cp_rank * local_frames
        latent_end = latent_start + local_frames
        raw_start, raw_end = conditioning_raw_frame_bounds(
            latent_start,
            latent_end,
            self.pipe.vae.config.scale_factor_temporal,
        )
        camera_rays = data["camera_rays"][:, latent_start:latent_end]

        local_data["condition"] = data["condition"][:, :, latent_start:latent_end]
        local_data["opacity"] = self._to_device(data["opacity"][:, raw_start:raw_end], self.pipe.vae.dtype)
        local_data["camera_rays"] = self._to_device(camera_rays, self.pipe.vae.dtype)
        local_data["w2cs"] = self._to_device(data["w2cs"][:, latent_start:latent_end])
        local_data["Ks"] = self._to_device(data["Ks"][:, latent_start:latent_end])
        return local_data, prediction[:, :, latent_start:latent_end], latent_start, True

    def _score_kwargs(
        self, data: dict, noisy_latent: torch.Tensor, timestep: torch.Tensor, frame_offset: int, cp: bool
    ):
        """Shared transformer inputs for the real/fake score evaluations of one noised sample."""
        transformer_timestep = self.pipe.prepare_transformer_timestep(timestep, noisy_latent)
        return dict(
            hidden_states=noisy_latent,
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
            local_cp_input=cp,
            frame_offset=frame_offset,
        )

    def _negative_prompt_kwargs(self, data: dict, batch_size: int) -> dict:
        """Unconditional-prompt kwargs for the CFG branch: the batch's stored
        unconditional when present, else the loaded default negative embedding."""
        if "encoded_negative_prompt" in data:
            neg_kwargs = {"encoder_hidden_states": data["encoded_negative_prompt"]}
            if "negative_prompt_attention_mask" in data:
                neg_kwargs["encoder_attention_mask"] = data["negative_prompt_attention_mask"]
            return neg_kwargs
        assert self.negative_prompt is not None, "DMD text guidance needs a negative prompt for token-prompt models."
        return {"encoder_hidden_states": self.negative_prompt.expand(batch_size, -1, -1)}

    def generator_loss(self, data: dict, prediction: torch.Tensor):
        """Compute the DMD distribution matching loss (DMD2 paper eq. 7-8)."""
        data, prediction, frame_offset, cp = self._score_batch(data, prediction)
        with torch.no_grad():
            timestep = self.get_timestep(prediction.shape[0])
            # Opacity-mixed flow-matching source z_mix (paper eq. 3), not pure noise.
            source_latent = self.pipe.prepare_latents(
                data["condition"],
                data["opacity"],
                frame_offset == 0,
                broadcast_cp_noise=not cp,
            )
            noisy_latent = self.pipe.scheduler.add_noise(prediction, source_latent, timestep).detach()

            score_kwargs = self._score_kwargs(data, noisy_latent, timestep, frame_offset, cp)
            prompt_kwargs = self._prompt_kwargs(data)

            fake_noise_pred = self.fake_score(**prompt_kwargs, **score_kwargs)[0]
            fake_predicted_x0 = self.pipe.scheduler.step(fake_noise_pred, timestep, noisy_latent, to_final=True)

            real_noise_pred = self.real_score(**prompt_kwargs, **score_kwargs)[0]
            real_predicted_x0 = self.pipe.scheduler.step(real_noise_pred, timestep, noisy_latent, to_final=True)

            if self.args.text_guidance_scale > 1:
                neg_kwargs = self._negative_prompt_kwargs(data, prediction.shape[0])
                fake_noise_pred_uncond_text = self.fake_score(**neg_kwargs, **score_kwargs)[0]
                fake_predicted_x0_uncond_text = self.pipe.scheduler.step(
                    fake_noise_pred_uncond_text, timestep, noisy_latent, to_final=True
                )
                fake_predicted_x0 = (
                    fake_predicted_x0_uncond_text
                    + (fake_predicted_x0 - fake_predicted_x0_uncond_text) * self.args.text_guidance_scale
                )

                real_noise_pred_uncond_text = self.real_score(**neg_kwargs, **score_kwargs)[0]
                real_predicted_x0_uncond_text = self.pipe.scheduler.step(
                    real_noise_pred_uncond_text, timestep, noisy_latent, to_final=True
                )
                real_predicted_x0 = (
                    real_predicted_x0_uncond_text
                    + (real_predicted_x0 - real_predicted_x0_uncond_text) * self.args.text_guidance_scale
                )

            # DMD gradient (eq. 7) with normalization (eq. 8).
            grad = fake_predicted_x0 - real_predicted_x0
            p_real = prediction - real_predicted_x0
            # FP32 or wider avoids half-precision rounding and overflow when combining shards.
            # Equal-sized shards make the average of their means the full-video mean.
            normalizer_dtype = torch.promote_types(p_real.dtype, torch.float32)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True, dtype=normalizer_dtype)
            if cp:
                cp_mesh = self._cp_mesh()
                dist.all_reduce(normalizer, group=cp_mesh.get_group())
                normalizer /= cp_mesh.size()
            grad = torch.nan_to_num(grad / normalizer.to(p_real.dtype))

        return 0.5 * F.mse_loss(prediction.double(), (prediction.double() - grad.double()).detach(), reduction="mean")

    def critic_loss(self, data: dict, prediction: torch.Tensor):
        """Train the fake score network on the generator's detached output (DMD2 Sec 4.5)."""
        data, prediction, frame_offset, cp = self._score_batch(data, prediction)
        timestep = self.get_timestep(prediction.shape[0])
        # Opacity-mixed flow-matching source z_mix (paper eq. 3), not pure noise.
        source_latent = self.pipe.prepare_latents(
            data["condition"],
            data["opacity"],
            frame_offset == 0,
            broadcast_cp_noise=not cp,
        )
        noisy_latent = self.pipe.scheduler.add_noise(prediction, source_latent, timestep)
        score_kwargs = self._score_kwargs(data, noisy_latent, timestep, frame_offset, cp)
        prompt_kwargs = self._prompt_kwargs(data)
        fake_noise_pred = self.fake_score(**prompt_kwargs, **score_kwargs)[0]

        with torch.no_grad():
            training_target = self.pipe.scheduler.training_target(prediction, source_latent)
        return F.mse_loss(fake_noise_pred.float(), training_target.float())

    def to_accumulate(self) -> list[nn.Module]:
        return [self.pipe.transformer, self.fake_score]

    def on_before_optimizer_step(self) -> None:
        # https://github.com/huggingface/accelerate/issues/3789
        self.accelerator.unscale_gradients()
        torch.nn.utils.clip_grad_norm_(self.pipe.transformer.parameters(), self.args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.fake_score.parameters(), self.args.max_grad_norm)

    def on_before_zero_grad(self) -> None:
        if not self.is_student_phase() or self.accelerator.optimizer_step_was_skipped:
            return

        if self.accelerator.sync_gradients:
            self.ema.update(average=self.get_step() >= self.args.ema_start_step)

    def is_student_phase(self) -> bool:
        # The phase must be constant across every micro-batch of one gradient-
        # accumulation window. accelerator.step advances once per micro-batch,
        # so get_step()'s floor division only flips on the window's LAST
        # micro-batch; derive the in-progress (1-indexed) optimizer step with a
        # ceiling division instead. Identical to get_step() when
        # gradient_accumulation_steps == 1 and at sync boundaries.
        current_step = self.step_offset + (self.accelerator.step - 1) // self.args.gradient_accumulation_steps + 1
        return (current_step - 1) % self.args.dfake_gen_update_ratio == 0

    def get_timestep(self, batch_size: int) -> torch.Tensor:
        # A CP mesh only exists in distributed runs; without an initialized
        # process group get_timestep needs nothing beyond scheduler + vae.
        cp_mesh = self._cp_mesh() if dist.is_initialized() else None
        if cp_mesh is None or self.pipe.transformer.cp_rank == 0:
            timestep = torch.randint(
                0,
                self.pipe.scheduler.num_train_timesteps,
                (batch_size,),
                device=self.pipe.vae.device,
                dtype=torch.long,
            )
        else:
            timestep = torch.empty(batch_size, device=self.pipe.vae.device, dtype=torch.long)

        if cp_mesh is not None:
            dist.broadcast(timestep, src=cp_mesh.mesh.view(-1)[0].item(), group=cp_mesh.get_group())

        # The noising warp and the rollout/step schedule are one shift BY
        # CONSTRUCTION: both read the pipeline scheduler's profile shift
        # (there is no per-run shift knob; a shift change edits the profile).
        shift = self.pipe.scheduler.shift
        if shift > 1:
            timestep = (
                shift
                * (timestep / self.pipe.scheduler.num_train_timesteps)
                / (1 + (shift - 1) * (timestep / self.pipe.scheduler.num_train_timesteps))
                * self.pipe.scheduler.num_train_timesteps
            )

        return timestep.clamp(self.min_step, self.max_step)
