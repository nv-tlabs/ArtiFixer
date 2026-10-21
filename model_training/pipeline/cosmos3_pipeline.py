# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler

from model_training.net.cosmos3_transformer import ArtifixerCosmos3Transformer
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase, prepare_transformer_for_training
from model_training.schedulers.flow_match import FlowMatchScheduler
from model_training.timestep import expand_transformer_timestep


class ArtifixerCosmos3Pipeline(ArtifixerPipelineBase):
    def __init__(
        self,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler | FlowMatchScheduler,
        transformer,
        gradient_checkpointing: bool,
        attention_backend: str | None = None,
        frames_per_block: int | None = None,
    ):
        super().__init__(
            vae,
            tokenizer=None,
            text_encoder=None,
            expand_timesteps=False,
            frames_per_block=frames_per_block,
            attention_backend=attention_backend,
        )
        self.scheduler = scheduler
        prepare_transformer_for_training(transformer, attention_backend)
        self.transformer = ArtifixerCosmos3Transformer(
            transformer,
            vae_scale_factor_spatial=self.vae.config.scale_factor_spatial,
            vae_scale_factor_temporal=self.vae.config.scale_factor_temporal,
            gradient_checkpointing=gradient_checkpointing,
            causal_block_frames=frames_per_block,
        )

    def _cp_frame_divisor(self, total_latent_frames: int) -> int:
        # The whole clip is one transformer call, so validation CP can split
        # at any latent frame boundary — even block-causal: the mask is built
        # at global geometry and row-sliced per CP shard (cp_size_candidates
        # additionally gcds with the block count for DF pipes).
        return total_latent_frames

    def cp_size_candidates(self, num_latent_frames: int, world_size: int) -> list[int]:
        # Any size dividing the attention-head gcd works on any clip length
        # (21-frame production clips included): the wrapper pads the
        # latent-frame axis to the CP multiple, drops pad keys after the
        # all-to-all exchange, and cuts pad frames after the output gather —
        # so no frame rule applies. Falls back to the base frame rule when
        # the transformer exposes no head counts (reduced test stubs).
        head_divisor = self.cp_head_divisor()
        if head_divisor is None:
            return super().cp_size_candidates(num_latent_frames, world_size)
        return [size for size in range(1, world_size + 1) if head_divisor % size == 0]

    def cp_raw_frame_padding(self, num_frames: int, cp_size: int, vae_temporal_scale: int) -> int:
        # The transformer pads its own frame axis (see cp_size_candidates),
        # so eval must NOT pre-pad the raw clip: the model sees the same
        # frames at every CP size.
        return 0

    def validate_cp_size(self, cp_size: int, num_latent_frames: int) -> None:
        head_divisor = self.cp_head_divisor(required=True)
        assert head_divisor % cp_size == 0, (
            f"CP size {cp_size} must divide the transformer's attention head divisor {head_divisor}: CP splits "
            "the attention heads across ranks, so every rank needs a whole number of query and KV heads. The "
            "transformer pads its own frame axis, so no frame rule applies."
        )

    def supports_local_cp_inference(self) -> bool:
        # The wrapper's block-causal mask indexes the full clip, so only
        # bidirectional configs serve pre-sliced shards; DF configs stay on
        # full-input internal sharding.
        return self.causal_block_frames is None

    def dummy_validation_prompt(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Cosmos3 prompts are token ids, not embeddings: one pad-free token
        # per branch.
        prompt = torch.ones(1, 1, device=device, dtype=torch.long)
        prompt_attention_mask = torch.ones(1, 1, device=device, dtype=torch.bool)
        return prompt, prompt_attention_mask, prompt.clone(), prompt_attention_mask.clone()

    def prepare_transformer_timestep(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        frames_per_block: int | None = None,
    ) -> torch.Tensor:
        """Expand per-block diffusion-forcing timesteps to the wrapper's
        per-token fp32 contract.

        Per-block tensors expand sample-major through the shared
        expand_transformer_timestep, matching the flow-match scheduler's
        ordering. Half-precision per-block input is rejected rather than
        silently cast to fp32. Scalar, per-sample, and any other 1-D shape
        pass through untouched so the wrapper's strict shape rejection
        still fires.
        """
        fpb = self.causal_block_frames
        if frames_per_block is not None:
            assert fpb is not None and frames_per_block == fpb, (
                f"frames_per_block={frames_per_block} was passed to prepare_transformer_timestep but the "
                f"pipeline was constructed with causal_block_frames={fpb}; the pipeline is the "
                "single owner of the diffusion-forcing contract."
            )
        if (
            fpb is not None
            and isinstance(timestep, torch.Tensor)
            and timestep.ndim == 1
            and hidden_states.shape[2] % fpb == 0
            and timestep.numel() == hidden_states.shape[0] * (hidden_states.shape[2] // fpb)
            and timestep.numel() != hidden_states.shape[0]
        ):
            assert timestep.dtype in (torch.int32, torch.int64, torch.float32, torch.float64), (
                f"Cosmos3 per-block timesteps must be integer or full-precision float, got {timestep.dtype}; "
                "half-precision timesteps quantize t before embedding."
            )
            return expand_transformer_timestep(timestep.to(torch.float32), hidden_states, self.patch_size(), fpb)
        return super().prepare_transformer_timestep(timestep, hidden_states, frames_per_block)

    def _normalize_prompt_inputs(
        self,
        prompt: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        assert isinstance(prompt, torch.Tensor), "Cosmos3 prompts must be pre-tokenized tensors"
        prompt = prompt.to(device=device, dtype=torch.long)
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(device=device, dtype=torch.bool)
        if negative_prompt is not None:
            assert isinstance(negative_prompt, torch.Tensor), "Cosmos3 negative prompts must be pre-tokenized tensors"
            negative_prompt = negative_prompt.to(device=device, dtype=torch.long)
        if negative_prompt_attention_mask is not None:
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device=device, dtype=torch.bool)
        return prompt, prompt_attention_mask, negative_prompt, negative_prompt_attention_mask

    def _resolve_text_conditioning(
        self,
        prompt,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt,
        negative_prompt_attention_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[dict, dict | None]:
        # There is no default negative-prompt buffer, so CFG additionally
        # needs negative_prompt (the no-text render diagnostic passes None
        # for a single unguided pass).
        prompt, prompt_attention_mask, negative_prompt, negative_prompt_attention_mask = self._normalize_prompt_inputs(
            prompt,
            prompt_attention_mask,
            negative_prompt,
            negative_prompt_attention_mask,
            device=device,
        )
        cond_kwargs = {"encoder_hidden_states": prompt, "encoder_attention_mask": prompt_attention_mask}
        if negative_prompt is None:
            return cond_kwargs, None
        return cond_kwargs, {
            "encoder_hidden_states": negative_prompt,
            "encoder_attention_mask": negative_prompt_attention_mask,
        }

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
        prompt: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 5.0,
        *,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
        noise_seed: int | None = None,
        neighbor_camera_rays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rendered_opacity = rendered_opacity.to(device=self.vae.device, dtype=self.vae.dtype)
        latents = self._denoise_to_latents(
            rendered_rgb=rendered_rgb,
            rendered_opacity=rendered_opacity,
            neighbors=neighbors,
            camera_rays=camera_rays,
            w2cs=w2cs,
            neighbor_w2cs=neighbor_w2cs,
            Ks=Ks,
            neighbor_Ks=neighbor_Ks,
            prompt=prompt,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt=negative_prompt,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
            max_neighbors_per_encode=max_neighbors_per_encode,
            noise_seed=noise_seed,
            local_cp_input=False,
            local_frame_offset=0,
            local_vae_crop=None,
            neighbor_camera_rays=neighbor_camera_rays,
        )
        return self.decode_latents_to_video(latents)
