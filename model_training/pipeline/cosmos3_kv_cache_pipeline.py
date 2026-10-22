# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import math

import torch
from diffusers import AutoencoderKLWan
from tqdm import tqdm

from model_training.model_profiles import COSMOS3_FLOW_SHIFT
from model_training.net.conditioning import slice_camera_tensor_frames
from model_training.net.context_parallel import transformer_cp_head_counts
from model_training.pipeline.cosmos3_pipeline import ArtifixerCosmos3Pipeline
from model_training.pipeline.pipeline_base import (
    assert_training_rollout_fits_window,
    create_denoising_step_list,
    resolve_exit_step,
    run_chunk_denoise_loop,
    uncompressed_chunk_frames,
    validate_latent_frames_divisible,
    validate_positive_frame_count,
    validate_rollout_window_options,
)
from model_training.schedulers.flow_match import FlowMatchScheduler

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _warn_once(message: str) -> None:
    """Emit a warning once per process instead of once per call.

    Cosmos3 validation batches always carry an encoded negative prompt (the
    shared trainer validation path passes it unconditionally), so without
    deduplication the CFG-ignored notices below would repeat on every
    forward_inference call on every rank each validation round.
    """
    logger.warning(message)


class ArtifixerCosmos3KvCachePipeline(ArtifixerCosmos3Pipeline):
    uses_classifier_free_guidance = False

    stages_conditioning_per_chunk = True

    def __init__(
        self,
        vae: AutoencoderKLWan,
        transformer,
        gradient_checkpointing: bool,
        attention_backend: str | None = None,
        flow_shift: float = COSMOS3_FLOW_SHIFT,
        *,
        ar_frames_per_chunk: int,
        local_attn_size: int,
        sink_size: int,
        truncate_rollout_gradients: bool = False,
    ):
        validate_positive_frame_count("ar_frames_per_chunk", ar_frames_per_chunk)
        validate_rollout_window_options(local_attn_size, sink_size)
        # The rollout evicts at whole-chunk granularity, so the latent-frame
        # options convert to chunk counts: local_attn_size == -1 means never
        # evict, and any positive sink_size keeps at least one anchor chunk
        # once eviction starts.
        ar_window_chunks = None if local_attn_size == -1 else max(1, local_attn_size // ar_frames_per_chunk)
        ar_sink_chunks = max(1, sink_size // ar_frames_per_chunk) if sink_size > 0 else 0
        if ar_window_chunks is not None and ar_sink_chunks >= ar_window_chunks:
            raise ValueError(
                f"`sink_size` ({sink_size}) and `local_attn_size` ({local_attn_size}) must leave at least one "
                f"non-anchor chunk in the attention window at ar_frames_per_chunk={ar_frames_per_chunk}"
            )
        # KV-cache rollouts and validation both run on the FlowMatch schedule
        # (wan_kv_cache_pipeline.py does the same for Wan); it is the pipeline's
        # one real scheduler, passed through the scheduler parameter instead
        # of building a throwaway UniPCMultistepScheduler and overwriting it.
        super().__init__(
            vae=vae,
            scheduler=FlowMatchScheduler(
                shift=flow_shift,
                sigma_min=0.0,
                extra_one_step=True,
                device=vae.device,
            ),
            transformer=transformer,
            gradient_checkpointing=gradient_checkpointing,
            attention_backend=attention_backend,
        )
        # Stage-2 Self-Forcing AR rollout chunking: uniform chunks of
        # frames_per_block latent frames, matching the Wan rollout (a
        # smaller first chunk would give the first frames a different
        # denoising trajectory than every later chunk).
        # Distinct from the parent's frames_per_block diffusion-forcing
        # guard, which stays untouched: AR chunking is a rollout contract of
        # this KV-cache rollout subclass, not a full-sequence block-causal
        # timestep contract.
        self.ar_frames_per_chunk = ar_frames_per_chunk
        # Attention window of the cached AR rollout, in whole chunks (same
        # latent-frame options as the Wan pipeline). Each denoise pass attends
        # [text + up to (window - 1) chunks of cached K/V + its own chunk];
        # when the window is full, the oldest cached chunk after the
        # ar_sink_chunks anchor is evicted, so long videos run at constant
        # per-chunk cost. ar_window_chunks is None when local_attn_size is -1:
        # nothing is ever evicted. Training rollouts must fit the window (see
        # assert_training_rollout_fits_window).
        self.ar_window_chunks = ar_window_chunks
        self.ar_sink_chunks = ar_sink_chunks
        # Threaded into every rollout's RolloutKvCache (the flag is defined
        # there); also selects the seed pass's grad mode, the one place the
        # pipeline itself branches on it (generate_samples_from_batch).
        self.truncate_rollout_gradients = truncate_rollout_gradients

    def _cp_frame_divisor(self, total_latent_frames: int) -> int:
        # Every AR cache pass carries exactly ar_frames_per_chunk frames, and
        # the wrapper shards each pass equally across the CP mesh — so CP
        # sizes that divide the chunk size divide every pass with the
        # frame-sharded cache layout (cp_size_candidates adds the sizes that
        # run the padded head-sharded cache instead).
        return self.ar_frames_per_chunk

    def cp_size_candidates(self, num_latent_frames: int, world_size: int) -> list[int]:
        # Chunk-divisor sizes run the frame-sharded cache with no head rule
        # (e.g. CP=7 with 7-frame chunks); head-gcd divisors additionally
        # run padded chunks with the head-sharded cache.
        chunk_sizes = {size for size in range(1, world_size + 1) if self.ar_frames_per_chunk % size == 0}
        return sorted(chunk_sizes.union(super().cp_size_candidates(num_latent_frames, world_size)))

    def cp_pass_latent_frames(self, num_latent_frames: int) -> int:
        # Every rollout pass carries one chunk.
        return self.ar_frames_per_chunk

    def latent_frame_multiple(self) -> int:
        # A rollout consumes whole ar_frames_per_chunk chunks.
        return self.ar_frames_per_chunk

    def validate_cp_size(self, cp_size: int, num_latent_frames: int) -> None:
        frames_per_chunk = self.ar_frames_per_chunk
        if frames_per_chunk % cp_size == 0:
            # Frame-sharded cache: queries keep all heads, no head rule.
            return
        # A padded chunk splits heads, so the transformer must expose its
        # head counts (required=True raises for foreign backbones).
        num_query_heads, num_kv_heads = transformer_cp_head_counts(self.transformer, required=True)
        assert num_query_heads % cp_size == 0 and num_kv_heads % cp_size == 0, (
            f"CP size {cp_size} must divide ar_frames_per_chunk ({frames_per_chunk}) for the "
            f"frame-sharded cache, or the transformer's attention head divisor "
            f"{math.gcd(num_query_heads, num_kv_heads)} ({num_query_heads} query / {num_kv_heads} KV heads) "
            "for the padded head-sharded cache"
        )

    def supports_local_cp_inference(self) -> bool:
        # Blockwise cache state cannot be represented by generic pre-sliced
        # inference; validation CP runs full-input per chunk.
        return False

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
        num_inference_steps: int = 4,
        text_guidance_scale: float = 5.0,  # unused — the distilled few-step path does not do CFG
        *,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
        noise_seed: int | None = None,
        neighbor_camera_rays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_guidance_scale != 5.0:
            _warn_once(
                "The Cosmos3 KV-cache rollout does not use classifier-free guidance; text_guidance_scale is ignored."
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
            prompt_attention_mask,
            negative_prompt,
            negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
            max_neighbors_per_encode=max_neighbors_per_encode,
            noise_seed=noise_seed,
            neighbor_camera_rays=neighbor_camera_rays,
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
        prompt: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        num_inference_steps: int = 4,
        *,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
        noise_seed: int | None = None,
        neighbor_camera_rays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if negative_prompt is not None:
            _warn_once(
                "The Cosmos3 KV-cache rollout does not use classifier-free guidance; negative_prompt is ignored."
            )
        condition = self.encode_video_frames(rendered_rgb)
        neighbors_condition = self._encode_neighbors_checked(
            neighbors,
            condition.shape[0],
            max_neighbors_per_encode,
        )
        # Same pre-tokenized-prompt hardening as the parent's forward_inference
        # (device move + long token ids + bool mask); idempotent for inputs
        # that already satisfy the contract. The distilled few-step rollout
        # ignores the negative prompt (no CFG), so those slots pass None.
        prompt, prompt_attention_mask, _, _ = self._normalize_prompt_inputs(
            prompt,
            prompt_attention_mask,
            None,
            None,
            device=condition.device,
        )
        return self.generate_samples_from_batch(
            condition,
            rendered_opacity,
            neighbors_condition,
            camera_rays,
            w2cs,
            neighbor_w2cs,
            Ks,
            neighbor_Ks,
            prompt,
            num_inference_steps,
            False,
            prompt_attention_mask=prompt_attention_mask,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
            noise_seed=noise_seed,
            neighbor_camera_rays=neighbor_camera_rays,
        )

    @torch.no_grad()
    def decode_latents_to_video(self, latents: torch.Tensor, temporal_chunk_size: int | None = None) -> torch.Tensor:
        return super().decode_latents_to_video(latents, temporal_chunk_size=temporal_chunk_size)

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
        # Keyword-only; same guard as
        # wan_kv_cache_pipeline.generate_samples_from_batch.
        *,
        ignore_neighbors: bool = False,
        prompt_attention_mask: torch.Tensor | None = None,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        noise_seed: int | None = None,
        neighbor_camera_rays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Self-Forcing AR chunked rollout (stage-2 SF-DMD), KV-cached and
        windowed.

        Chunks of ``ar_frames_per_chunk`` latent frames are denoised in
        temporal order: each denoise pass carries ONLY the current chunk
        (conditioning sliced chunk-only at absolute positions; history
        attends via cache content, no mask), and each finished chunk except
        the global last is seeded into the cache by one conditioning-mode
        forward over its x0 (``timestep=None``, no RNG draw; grad mode per
        ``RolloutKvCache.truncate_rollout_gradients``). Before each chunk,
        cached history beyond
        (``ar_window_chunks`` - 1) chunks is evicted, keeping the first
        ``ar_sink_chunks`` as a fixed anchor, so long videos denoise at
        constant per-chunk cost; ``local_attn_size == -1`` and
        training-length rollouts never evict. The cache lives for this one
        call; see ``RolloutKvCache`` for the cache contracts (write-once
        seeding, the gradient storage policy, coverage bookkeeping).
        """
        # One generator per rollout, on the device the chunks are drawn on (the condition may be
        # CPU-staged). None for the DMD training rollout, whose noise stays on the global stream.
        noise_generator = self._noise_generator(noise_seed, self.vae.device)
        rendered_opacity = rendered_opacity.to(dtype=condition.dtype)
        frames_per_chunk = self.ar_frames_per_chunk
        latent_frames = condition.shape[2]
        validate_latent_frames_divisible(latent_frames, frames_per_chunk, "ar_frames_per_chunk")
        assert_training_rollout_fits_window(
            latent_frames, self.ar_window_chunks, frames_per_chunk, training=self.transformer.training
        )
        if self.transformer.cp_mesh is not None:
            self.validate_cp_size(self.transformer.cp_world_size, latent_frames)
        # All chunks must use slices of one full-clip camera-ray array.
        # Cached and new frames must use the same reference camera.
        assert camera_rays.shape[1] == latent_frames, (
            f"the rollout needs one clip-global raymap covering all {latent_frames} latent "
            f"frames, got {camera_rays.shape[1]}"
        )
        scale_factor_temporal = self.vae.config.scale_factor_temporal
        timesteps = create_denoising_step_list(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=self.vae.device,
        )
        exit_flag = resolve_exit_step(num_inference_steps, use_exit_flag, device=self.vae.device)

        reference_camera_kwargs = {}
        if neighbor_camera_rays is not None:
            reference_camera_kwargs["neighbor_camera_rays"] = neighbor_camera_rays.to(condition.device)

        kv_cache = self.transformer.new_rollout_kv_cache(truncate_rollout_gradients=self.truncate_rollout_gradients)
        clean_chunks: list[torch.Tensor] = []
        uncompressed_start_frame = 0
        chunk_start_frames = range(0, latent_frames, frames_per_chunk)
        for chunk_start in tqdm(chunk_start_frames, disable=not show_progress, leave=progress_bar_leave):
            chunk_end = chunk_start + frames_per_chunk
            kv_cache.evict_to_window(self.ar_window_chunks, self.ar_sink_chunks)
            uncompressed_num_frames = uncompressed_chunk_frames(chunk_start, frames_per_chunk, scale_factor_temporal)
            uncompressed_end_frame = uncompressed_start_frame + uncompressed_num_frames
            chunk_condition = condition[:, :, chunk_start:chunk_end]
            chunk_opacity = rendered_opacity[:, uncompressed_start_frame:uncompressed_end_frame].to(condition.device)

            latents = self.prepare_latents(chunk_condition, chunk_opacity, chunk_start == 0, generator=noise_generator)

            visible_camera_rays = slice_camera_tensor_frames(camera_rays, chunk_start, chunk_end).to(condition.device)
            visible_w2cs = slice_camera_tensor_frames(w2cs, chunk_start, chunk_end).to(condition.device)
            visible_Ks = slice_camera_tensor_frames(Ks, chunk_start, chunk_end).to(condition.device)

            # ONE conditioning set per chunk, shared verbatim by every
            # denoise pass of the chunk and by the cache-seed forward below —
            # the seeded K/V must be computed under exactly the conditioning
            # the denoise passes saw, so the keys are deliberately not
            # hand-copied per call site. Per-pass deltas (hidden_states /
            # timestep / kv-cache flags) stay explicit at each transformer
            # call.
            chunk_cond_kwargs = dict(
                encoder_hidden_states=encoded_prompt,
                encoder_attention_mask=prompt_attention_mask,
                neighbor_hidden_states=neighbors_condition,
                ignore_neighbors=ignore_neighbors,
                opacity=chunk_opacity,
                camera_rays=visible_camera_rays,
                w2cs=visible_w2cs,
                neighbor_w2cs=neighbor_w2cs,
                Ks=visible_Ks,
                neighbor_Ks=neighbor_Ks,
                return_dict=False,
                **reference_camera_kwargs,
            )

            def call_transformer(latents, t, is_exit_step):
                # is_exit_step is unused: cosmos3 caches K/V in a separate
                # conditioning-mode seed pass after the loop, not on the
                # exit denoise.
                # ONE scalar fp32 sigma per pass (pipeline dtype policy).
                transformer_timestep = self.prepare_transformer_timestep(t, latents)
                return self.transformer(
                    hidden_states=latents,
                    timestep=transformer_timestep,
                    kv_cache=kv_cache,
                    kv_cache_start_frame=chunk_start,
                    **chunk_cond_kwargs,
                )[0]

            def draw_fresh_latents():
                # Every "noise" slot is a FRESH opacity-mixed z_mix draw.
                return self.prepare_latents(chunk_condition, chunk_opacity, chunk_start == 0, generator=noise_generator)

            latents = run_chunk_denoise_loop(
                self.scheduler,
                timesteps,
                exit_flag,
                latents,
                call_transformer=call_transformer,
                draw_fresh_latents=draw_fresh_latents,
                device=self.vae.device,
            )
            clean_chunks.append(latents)
            if chunk_end < latent_frames:
                # Seeding exists so later chunks can read the history; the
                # final chunk has no later reader, so it is not seeded.
                # Truncation seeds under no_grad on the detached x0 (the
                # store would detach anyway); see
                # RolloutKvCache.truncate_rollout_gradients.
                with torch.set_grad_enabled(torch.is_grad_enabled() and not self.truncate_rollout_gradients):
                    self.transformer(
                        hidden_states=latents.detach() if self.truncate_rollout_gradients else latents,
                        timestep=None,
                        kv_cache=kv_cache,
                        kv_cache_seed=True,
                        kv_cache_start_frame=chunk_start,
                        **chunk_cond_kwargs,
                    )
            uncompressed_start_frame = uncompressed_end_frame
        return torch.cat(clean_chunks, dim=2)
