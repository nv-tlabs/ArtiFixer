# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import math

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from model_training.net.transformer import ArtifixerTransformer
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.schedulers.flow_match import FlowMatchScheduler

logger = logging.getLogger(__name__)


class ArtifixerKvCachePipeline(ArtifixerPipelineBase):

    def __init__(
        self,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        tokenizer: AutoTokenizer | None,
        text_encoder: UMT5EncoderModel | None,
        frames_per_block: int,
        local_attn_size: int,
        sink_size: int,
        gradient_checkpointing: bool,
        checkpoint_every_n_blocks: int,
        attention_backend: str | None = None,
    ):
        super().__init__(vae, tokenizer, text_encoder)

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True, device=self.vae.device)

        self.frames_per_block = frames_per_block
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        transformer.train()
        transformer.requires_grad_(True)
        if attention_backend is not None:
            transformer.set_attention_backend(attention_backend)

        self.transformer = ArtifixerTransformer(
            transformer,
            frames_per_block=None,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            vae_scale_factor_spatial=self.vae.config.scale_factor_spatial,
            vae_scale_factor_temporal=self.vae.config.scale_factor_temporal,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_every_n_blocks=checkpoint_every_n_blocks,
        )

        self.kv_cache1 = None
        self.crossattn_cache = None
        self.neighbor_crossattn_cache = None

    def cp_frame_divisor(self, total_latent_frames: int) -> int:
        """CP group size must divide frames_per_block, sink_size, and local_attn_size.

        Returns ``gcd(frames_per_block, sink_size, local_attn_size)`` over
        whichever values are active (positive).  The CP group size must divide
        all chunk-level parameters so that each rank processes whole chunks.

        Examples::

            # frames_per_block=7, sink_size=0, local_attn_size=0
            # gcd(7) = 7 -> valid CP sizes: {1, 7}

            # frames_per_block=7, sink_size=3, local_attn_size=0
            # gcd(7, 3) = 1 -> only CP=1 works (7 and 3 are coprime)

            # frames_per_block=6, sink_size=3, local_attn_size=0
            # gcd(6, 3) = 3 -> valid CP sizes: {1, 3}
        """
        values = [self.frames_per_block]
        if self.sink_size > 0:
            values.append(self.sink_size)
        if self.local_attn_size > 0 and self.local_attn_size != -1:
            values.append(self.local_attn_size)
        return math.gcd(*values)

    @torch.no_grad()
    def forward_inference(
        self,
        rendered_rgb: torch.Tensor,
        rendered_opacity: torch.Tensor,
        neighbors: torch.Tensor | None,
        camera_rays: torch.Tensor,
        w2cs: torch.Tensor,
        neighbor_w2cs: torch.Tensor | None,
        Ks: torch.Tensor,
        neighbor_Ks: torch.Tensor | None,
        prompt: str | torch.Tensor,
        negative_prompt: str | torch.Tensor | None = None,  # unused — KV-cache pipeline does not do CFG
        num_inference_steps: int = 4,
        text_guidance_scale: float = 5.0,  # unused — KV-cache pipeline does not do CFG
        *,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
    ) -> torch.Tensor:
        if negative_prompt is not None:
            logger.warning("KV-cache pipeline does not support classifier-free guidance; negative_prompt is ignored.")
        if text_guidance_scale != 5.0:
            logger.warning(
                "KV-cache pipeline does not support classifier-free guidance; text_guidance_scale is ignored."
            )

        latents = self.denoise_to_latents(
            rendered_rgb,
            rendered_opacity,
            neighbors,
            camera_rays,
            w2cs,
            neighbor_w2cs,
            Ks,
            neighbor_Ks,
            prompt,
            num_inference_steps,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
            max_neighbors_per_encode=max_neighbors_per_encode,
        )
        return self.decode_latents_to_video(latents)

    @torch.no_grad()
    def denoise_to_latents(
        self,
        rendered_rgb: torch.Tensor,
        rendered_opacity: torch.Tensor,
        neighbors: torch.Tensor | None,
        camera_rays: torch.Tensor,
        w2cs: torch.Tensor,
        neighbor_w2cs: torch.Tensor | None,
        Ks: torch.Tensor,
        neighbor_Ks: torch.Tensor | None,
        prompt: str | torch.Tensor,
        num_inference_steps: int = 4,
        *,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
    ) -> torch.Tensor:
        condition = self.encode_video_frames(rendered_rgb.to(self.vae.device)).cpu()

        batch_size = rendered_rgb.shape[0]
        if neighbors is not None:
            assert neighbors.shape[0] == batch_size, "`neighbors` batch size has to match `rendered_rgb`"
            neighbors_condition = self.encode_neighbors(neighbors, max_neighbors_per_encode)
        else:
            neighbors_condition = None

        if isinstance(prompt, torch.Tensor):
            assert prompt.ndim == 3, "`prompt` has to be a 3D tensor"
            assert prompt.shape[0] == batch_size, "`prompt` batch size has to match `rendered_rgb`"
            prompt_embeds = prompt.to(self.vae.device)
        else:
            prompt_embeds = self.get_t5_prompt_embeds(prompt).expand(batch_size, -1, -1)

        # Keyword args from use_exit_flag onward: passing show_progress /
        # progress_bar_leave positionally used to land show_progress on
        # ignore_neighbors, silently dropping the reference views on any
        # process with show_progress=True (i.e. rank 0 -- every single-GPU
        # run). See generate_samples_from_batch's signature.
        return self.generate_samples_from_batch(
            condition,
            rendered_opacity,
            neighbors_condition,
            camera_rays,
            w2cs,
            neighbor_w2cs,
            Ks,
            neighbor_Ks,
            prompt_embeds,
            num_inference_steps,
            use_exit_flag=False,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
        )

    @torch.no_grad()
    def decode_latents_to_video(self, latents: torch.Tensor) -> torch.Tensor:
        self.clear_inference_caches()
        video = self.latents_to_rgb(latents)
        return self.video_processor.postprocess_video(video, output_type="pt")

    def clear_inference_caches(self) -> None:
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.neighbor_crossattn_cache = None
        torch.cuda.empty_cache()
        gc.collect()

    def generate_samples_from_batch(
        self,
        condition: torch.Tensor,
        rendered_opacity: torch.Tensor,
        neighbors_condition: torch.Tensor | None,
        camera_rays: torch.Tensor,
        w2cs: torch.Tensor,
        neighbor_w2cs: torch.Tensor | None,
        Ks: torch.Tensor,
        neighbor_Ks: torch.Tensor | None,
        encoded_prompt: torch.Tensor,
        num_inference_steps: int,
        use_exit_flag: bool,
        # Keyword-only: trailing boolean flags must not bind positionally (a
        # positional call once silently swapped show_progress into
        # ignore_neighbors, dropping the reference views).
        *,
        ignore_neighbors: bool = False,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
    ) -> torch.Tensor:
        batch_size, _, latent_num_frames, latent_height, latent_width = condition.shape
        p_t, p_h, p_w = self.transformer.patch_size
        post_patch_num_frames = latent_num_frames // p_t
        post_patch_height = latent_height // p_h
        post_patch_width = latent_width // p_w

        # If training is enabled, we need the kv cache to span all frames
        # for gradient checkpointing to work correctly.
        if torch.is_grad_enabled():
            num_cache_frames = post_patch_num_frames
        else:
            num_cache_frames = self.local_attn_size if self.local_attn_size != -1 else post_patch_num_frames

        frame_seq_length = post_patch_height * post_patch_width
        self._initialize_kv_cache(batch_size, frame_seq_length, num_cache_frames)
        self._initialize_crossattn_cache("crossattn_cache")
        self._initialize_crossattn_cache("neighbor_crossattn_cache")

        current_start_frame = 0
        current_uncompressed_start_frame = 0
        assert (
            latent_num_frames % self.frames_per_block == 0
        ), "Latent number of frames must be divisible by the number of frames per block"
        all_num_frames = [self.frames_per_block] * (latent_num_frames // self.frames_per_block)

        exit_flag = self._generate_and_sync_exit_flag(num_inference_steps) if use_exit_flag else num_inference_steps - 1

        if neighbor_w2cs is not None:
            neighbor_w2cs = neighbor_w2cs.to(self.vae.device)
        if neighbor_Ks is not None:
            neighbor_Ks = neighbor_Ks.to(self.vae.device)

        output = torch.zeros_like(condition, device=self.vae.device)
        for current_num_frames in tqdm(all_num_frames, disable=not show_progress, leave=progress_bar_leave):
            current_end_frame = current_start_frame + current_num_frames
            current_uncompressed_num_frames = (current_num_frames - 1) * self.vae.config.scale_factor_temporal + (
                1 if current_start_frame == 0 else self.vae.config.scale_factor_temporal
            )
            current_uncompressed_end_frame = current_uncompressed_start_frame + current_uncompressed_num_frames

            chunk_condition = condition[:, :, current_start_frame:current_end_frame].to(self.vae.device)
            chunk_opacity = rendered_opacity[:, current_uncompressed_start_frame:current_uncompressed_end_frame].to(
                self.vae.device
            )

            latents = self.prepare_latents(chunk_condition, chunk_opacity, current_start_frame == 0)

            chunk_camera_rays = camera_rays[:, current_start_frame:current_end_frame].to(self.vae.device)
            chunk_w2cs = w2cs[:, current_start_frame:current_end_frame].to(self.vae.device)
            chunk_Ks = Ks[:, current_start_frame:current_end_frame].to(self.vae.device)

            transformer_kwargs = dict(
                hidden_states=latents,
                encoder_hidden_states=encoded_prompt,
                neighbor_hidden_states=neighbors_condition,
                ignore_neighbors=ignore_neighbors,
                opacity=chunk_opacity,
                camera_rays=chunk_camera_rays,
                w2cs=chunk_w2cs,
                neighbor_w2cs=neighbor_w2cs,
                Ks=chunk_Ks,
                neighbor_Ks=neighbor_Ks,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                neighbor_crossattn_cache=self.neighbor_crossattn_cache,
                current_start=current_start_frame * frame_seq_length,
                frame_offset=current_start_frame,
                return_dict=False,
            )

            timesteps = self.create_denoising_step_list(num_inference_steps)

            # Step 3.1: Spatial denoising loop
            for index, t in enumerate(timesteps):
                should_exit = index == exit_flag

                timestep = t.expand(batch_size)
                transformer_kwargs["hidden_states"] = latents

                if not should_exit:
                    with torch.no_grad():
                        noise_pred = self.transformer(**transformer_kwargs, timestep=timestep.to(latents.dtype))[0]
                        latents = self.scheduler.step(noise_pred, timestep, latents, to_final=True)
                        next_timestep = timesteps[index + 1]
                        latents = self.scheduler.add_noise(
                            latents,
                            self.prepare_latents(chunk_condition, chunk_opacity, current_start_frame == 0),
                            next_timestep
                            * torch.ones(
                                (batch_size,),
                                device=self.vae.device,
                                dtype=torch.long,
                            ),
                        )
                else:
                    # for getting real output
                    noise_pred = self.transformer(**transformer_kwargs, timestep=timestep.to(latents.dtype))[0]
                    latents = self.scheduler.step(noise_pred, timestep, latents, to_final=True)
                    break

            # Step 3.2: record the model's output
            output[:, :, current_start_frame:current_end_frame] = latents

            # Step 3.4: update the start and end frame indices
            current_start_frame = current_end_frame
            current_uncompressed_start_frame = current_uncompressed_end_frame

        return output

    def _generate_and_sync_exit_flag(self, num_inference_steps: int) -> int:
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            exit_flag = torch.randint(low=0, high=num_inference_steps, size=(1,), device=self.vae.device)
        else:
            exit_flag = torch.empty(1, dtype=torch.long, device=self.vae.device)

        if dist.is_initialized():
            dist.broadcast(exit_flag, src=0)  # Broadcast the random indices to all ranks
        return exit_flag.item()

    def _initialize_kv_cache(self, batch_size, frame_seq_length: int, num_frames: int) -> None:
        # With CP each rank stores 1/cp_size of the total cache frames.
        cp_world_size = self.transformer._cp_world_size
        local_num_frames = num_frames // cp_world_size
        total_tokens = local_num_frames * frame_seq_length

        # Reuse existing cache if it already has the right shape — just zero it.
        can_reuse = (
            self.kv_cache1 is not None
            and self.kv_cache1[0]["k"].shape[0] == batch_size
            and self.kv_cache1[0]["k"].shape[1] == total_tokens
        )
        if can_reuse:
            for kv_cache in self.kv_cache1:
                for value in kv_cache.values():
                    value.zero_()
        else:
            # Free the old cache (if any) before allocating a new one.
            del self.kv_cache1
            torch.cuda.empty_cache()
            gc.collect()

            self.kv_cache1 = [
                {
                    "k": torch.zeros(
                        [batch_size, total_tokens, block.attn1.heads, block.attn1.inner_dim // block.attn1.heads],
                        dtype=self.vae.dtype,
                        device=self.vae.device,
                    ),
                    "v": torch.zeros(
                        [batch_size, total_tokens, block.attn1.heads, block.attn1.inner_dim // block.attn1.heads],
                        dtype=self.vae.dtype,
                        device=self.vae.device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=self.vae.device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=self.vae.device),
                }
                for block in self.transformer.blocks
            ]

    def _initialize_crossattn_cache(self, attr_name: str) -> None:
        cache = [{"is_init": False} for _ in self.transformer.blocks]
        setattr(self, attr_name, cache)

    def create_denoising_step_list(self, num_inference_steps: int) -> torch.Tensor:
        denoising_step_list = torch.linspace(
            1000, 0, num_inference_steps + 1, dtype=torch.long, device=self.vae.device
        )[:-1]

        # This will change the denoising step list from long type to float type (as in the official self-forcing implementation)
        timesteps = torch.cat(
            (
                self.scheduler.timesteps,
                torch.tensor([0], dtype=torch.float32, device=self.vae.device),
            )
        )
        denoising_step_list = timesteps[self.scheduler.num_train_timesteps - denoising_step_list]
        return denoising_step_list
