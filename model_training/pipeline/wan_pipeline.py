# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel

from model_training.common.context_parallel import cp_rank_temporal_ranges, latent_frame_count
from model_training.net.conditioning import slice_camera_tensor_frames
from model_training.net.context_parallel import all_gather_cat
from model_training.net.wan_transformer import ArtifixerTransformer
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase, prepare_transformer_for_training


class ArtifixerPipeline(ArtifixerPipelineBase):

    def __init__(
        self,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        transformer: WanTransformer3DModel,
        tokenizer: AutoTokenizer | None,
        text_encoder: UMT5EncoderModel | None,
        default_negative_prompt_path: Path,
        frames_per_block: int | None,
        gradient_checkpointing: bool,
        checkpoint_every_n_blocks: int,
        expand_timesteps: bool,
        attention_backend: str | None = None,
    ):
        super().__init__(
            vae,
            tokenizer,
            text_encoder,
            expand_timesteps=expand_timesteps,
            frames_per_block=frames_per_block,
            attention_backend=attention_backend,
        )

        self.default_negative_prompt = nn.Buffer(
            torch.load(default_negative_prompt_path, weights_only=True), persistent=False
        )
        self.scheduler = scheduler

        transformer.requires_grad_(True)
        prepare_transformer_for_training(transformer, attention_backend)

        self.transformer = ArtifixerTransformer(
            transformer,
            frames_per_block=frames_per_block,
            use_kv_cache_attention=False,
            vae_scale_factor_spatial=self.vae.config.scale_factor_spatial,
            vae_scale_factor_temporal=self.vae.config.scale_factor_temporal,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_every_n_blocks=checkpoint_every_n_blocks,
        )

    def _cp_frame_divisor(self, total_latent_frames: int) -> int:
        """The standard pipeline accepts a full-clip ``forward_inference`` call
        at any latent frame boundary while CP is active: each rank shards the
        clip internally and gathers latents before decoding (see
        ``_forward_inference_full_clip_cp``)."""
        return total_latent_frames

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
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt: str | torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 5.0,
        *,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        assert (
            prompt_attention_mask is None and negative_prompt_attention_mask is None
        ), "Wan prompts are pre-encoded T5 embeddings; prompt attention masks are not supported"
        cp_mesh = self.transformer.cp_mesh
        if cp_mesh is not None and cp_mesh.size() > 1:
            return self._forward_inference_full_clip_cp(
                rendered_rgb=rendered_rgb,
                rendered_opacity=rendered_opacity,
                neighbors=neighbors,
                camera_rays=camera_rays,
                w2cs=w2cs,
                neighbor_w2cs=neighbor_w2cs,
                Ks=Ks,
                neighbor_Ks=neighbor_Ks,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                text_guidance_scale=text_guidance_scale,
                show_progress=show_progress,
                progress_bar_leave=progress_bar_leave,
                noise_seed=noise_seed,
                max_neighbors_per_encode=max_neighbors_per_encode,
            )
        latents = self.denoise_to_latents(
            rendered_rgb=rendered_rgb,
            rendered_opacity=rendered_opacity,
            neighbors=neighbors,
            camera_rays=camera_rays,
            w2cs=w2cs,
            neighbor_w2cs=neighbor_w2cs,
            Ks=Ks,
            neighbor_Ks=neighbor_Ks,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
            max_neighbors_per_encode=max_neighbors_per_encode,
            noise_seed=noise_seed,
        )
        return self.decode_latents_to_video(latents)

    @torch.no_grad()
    def _forward_inference_full_clip_cp(
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
        negative_prompt: str | torch.Tensor | None,
        num_inference_steps: int,
        text_guidance_scale: float,
        show_progress: bool,
        progress_bar_leave: bool,
        max_neighbors_per_encode: int | None,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        """Serve a full-clip inference request while a CP mesh is active.

        run_inference and trainer validation enable CP on the transformer and
        still pass complete clips.  Each rank slices its own temporal shard
        here, denoises it through ``forward_inference_local_cp``, then
        all-gathers the latent shards and decodes once, so every rank returns
        the same full video a CP=1 run produces.  Camera tensors are at latent
        temporal resolution (the transformer requires that of ``camera_rays``,
        and both CP consumers pad ``w2cs``/``Ks`` to the same layout), so all
        three are sliced by the latent range.
        """
        cp_mesh = self.transformer.cp_mesh
        temporal_scale = self.vae.config.scale_factor_temporal
        num_raw_frames = rendered_rgb.shape[1]
        ranges = cp_rank_temporal_ranges(
            self.transformer.cp_rank,
            cp_mesh.size(),
            total_raw_frames=num_raw_frames,
            temporal_scale=temporal_scale,
            # Full left halo: the Wan VAE is temporally causal, so encoding
            # each rank's window with its complete left context (right side
            # truncated) reproduces the CP=1 full-clip encode exactly.
            halo_latent_frames=latent_frame_count(num_raw_frames, temporal_scale),
        )
        latent_start, latent_end = ranges["latent"]
        conditioning_raw_start, conditioning_raw_end = ranges["conditioning_raw"]
        vae_raw_start, vae_raw_end, vae_crop_start, vae_crop_end = ranges["vae_raw"]

        local_latents = self.forward_inference_local_cp(
            rendered_rgb=rendered_rgb[:, vae_raw_start:vae_raw_end],
            rendered_opacity=rendered_opacity[:, conditioning_raw_start:conditioning_raw_end],
            neighbors=neighbors,
            camera_rays=slice_camera_tensor_frames(camera_rays, latent_start, latent_end),
            w2cs=slice_camera_tensor_frames(w2cs, latent_start, latent_end),
            neighbor_w2cs=neighbor_w2cs,
            Ks=slice_camera_tensor_frames(Ks, latent_start, latent_end),
            neighbor_Ks=neighbor_Ks,
            prompt=prompt,
            local_frame_offset=latent_start,
            local_vae_crop=(vae_crop_start, vae_crop_end),
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
            max_neighbors_per_encode=max_neighbors_per_encode,
            noise_seed=noise_seed,
        )
        latents = all_gather_cat(local_latents, cp_mesh, dim=2)
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
        negative_prompt: str | torch.Tensor | None = None,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 5.0,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        return self._denoise_to_latents(
            rendered_rgb=rendered_rgb,
            rendered_opacity=rendered_opacity,
            neighbors=neighbors,
            camera_rays=camera_rays,
            w2cs=w2cs,
            neighbor_w2cs=neighbor_w2cs,
            Ks=Ks,
            neighbor_Ks=neighbor_Ks,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
            max_neighbors_per_encode=max_neighbors_per_encode,
            local_cp_input=False,
            local_frame_offset=0,
            local_vae_crop=None,
            noise_seed=noise_seed,
        )

    def supports_local_cp_inference(self) -> bool:
        # Every Wan config serves pre-sliced shards: the wrapper builds
        # block-causal masks at global geometry under local inputs.
        return True

    def _resolve_text_conditioning(
        self,
        prompt,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt,
        negative_prompt_attention_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[dict, dict | None]:
        assert (
            prompt_attention_mask is None and negative_prompt_attention_mask is None
        ), "Wan prompts are pre-encoded T5 embeddings; prompt attention masks are not supported"
        prompt_embeds = self._resolve_prompt_embeds(prompt, batch_size)

        if negative_prompt is None:
            negative_prompt_embeds = self.default_negative_prompt.to(device).expand(batch_size, -1, -1)
        else:
            negative_prompt_embeds = self._resolve_prompt_embeds(negative_prompt, batch_size)
        return (
            {"encoder_hidden_states": prompt_embeds},
            {"encoder_hidden_states": negative_prompt_embeds},
        )
