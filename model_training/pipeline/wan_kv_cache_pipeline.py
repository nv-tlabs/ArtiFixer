# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from model_training.common.context_parallel import all_gather_cat_with_grad, conditioning_raw_frame_bounds
from model_training.model_profiles import WAN_FLOW_SHIFT
from model_training.net.conditioning import slice_camera_tensor_frames
from model_training.net.rollout_kv_cache import RolloutKvCache
from model_training.net.wan_transformer import ArtifixerTransformer
from model_training.pipeline.pipeline_base import (
    ArtifixerPipelineBase,
    assert_training_rollout_fits_window,
    create_denoising_step_list,
    prepare_transformer_for_training,
    resolve_exit_step,
    run_chunk_denoise_loop,
    validate_latent_frames_divisible,
    validate_positive_frame_count,
    validate_rollout_window_options,
)
from model_training.schedulers.flow_match import FlowMatchScheduler

logger = logging.getLogger(__name__)


class ArtifixerKvCachePipeline(ArtifixerPipelineBase):
    uses_classifier_free_guidance = False

    stages_conditioning_per_chunk = True

    def __init__(
        self,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        tokenizer: AutoTokenizer | None,
        text_encoder: UMT5EncoderModel | None,
        ar_frames_per_chunk: int,
        local_attn_size: int,
        sink_size: int,
        gradient_checkpointing: bool,
        checkpoint_every_n_blocks: int,
        expand_timesteps: bool,
        attention_backend: str | None = None,
        flow_shift: float = WAN_FLOW_SHIFT,
        truncate_rollout_gradients: bool = False,
    ):
        validate_positive_frame_count("ar_frames_per_chunk", ar_frames_per_chunk)
        validate_rollout_window_options(local_attn_size, sink_size)
        if local_attn_size != -1:
            window_size = local_attn_size - sink_size
            if sink_size % ar_frames_per_chunk != 0 or window_size % ar_frames_per_chunk != 0:
                raise ValueError("`sink_size` and `local_attn_size - sink_size` must align to `ar_frames_per_chunk`")

        super().__init__(
            vae, tokenizer, text_encoder, expand_timesteps=expand_timesteps, attention_backend=attention_backend
        )

        # Defaults to the profile shift; the DMD noising warp
        # (trainers/dmd.py get_timestep) reads this scheduler's shift, so the
        # warp and the rollout/step schedule cannot desynchronize.
        self.scheduler = FlowMatchScheduler(
            shift=flow_shift, sigma_min=0.0, extra_one_step=True, device=self.vae.device
        )

        self.ar_frames_per_chunk = ar_frames_per_chunk
        # Attention window of the cached rollout in whole chunks (the
        # alignment checks above make the conversion exact): each denoise
        # pass attends [up to (window - 1) chunks of cached K/V + its own
        # chunk], the oldest non-anchor chunk is evicted once the window is
        # full, and local_attn_size == -1 (ar_window_chunks None) never
        # evicts. Training rollouts must fit the window (see
        # assert_training_rollout_fits_window).
        self.ar_window_chunks = None if local_attn_size == -1 else local_attn_size // ar_frames_per_chunk
        self.ar_sink_chunks = sink_size // ar_frames_per_chunk
        # Threaded into every rollout's RolloutKvCache; the flag is defined
        # there.
        self.truncate_rollout_gradients = truncate_rollout_gradients

        transformer.requires_grad_(True)
        prepare_transformer_for_training(transformer, attention_backend)

        self.transformer = ArtifixerTransformer(
            transformer,
            frames_per_block=None,
            use_kv_cache_attention=True,
            vae_scale_factor_spatial=self.vae.config.scale_factor_spatial,
            vae_scale_factor_temporal=self.vae.config.scale_factor_temporal,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_every_n_blocks=checkpoint_every_n_blocks,
        )

    def _cp_frame_divisor(self, total_latent_frames: int) -> int:
        # Every rollout pass carries exactly ar_frames_per_chunk frames, and
        # the rollout shards each pass equally across the CP mesh — so CP
        # sizes that divide the chunk size divide every pass.
        return self.ar_frames_per_chunk

    def cp_head_divisor(self, required: bool = False) -> int | None:
        # The rollout cache path never splits heads: queries stay rank-local
        # with all heads, so only the frames-per-chunk rule constrains CP
        # sizes (e.g. production CP=7 rollouts stay legal at any head count).
        return None

    def cp_pass_latent_frames(self, num_latent_frames: int) -> int:
        # Every rollout pass carries one chunk.
        return self.ar_frames_per_chunk

    def latent_frame_multiple(self) -> int:
        # A rollout consumes whole ar_frames_per_chunk chunks.
        return self.ar_frames_per_chunk

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
        negative_prompt: str | torch.Tensor | None = None,  # unused — KV-cache pipeline does not do CFG
        negative_prompt_attention_mask: torch.Tensor | None = None,
        num_inference_steps: int = 4,
        text_guidance_scale: float = 5.0,  # unused — KV-cache pipeline does not do CFG
        *,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
        noise_seed: int | None = None,
        vae_chunk_halo_latent_frames: int = 0,
    ) -> torch.Tensor:
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
            prompt_attention_mask,
            negative_prompt,
            negative_prompt_attention_mask,
            num_inference_steps,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
            max_neighbors_per_encode=max_neighbors_per_encode,
            noise_seed=noise_seed,
            vae_chunk_halo_latent_frames=vae_chunk_halo_latent_frames,
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
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt: str | torch.Tensor | None = None,  # unused — KV-cache pipeline does not do CFG
        negative_prompt_attention_mask: torch.Tensor | None = None,
        num_inference_steps: int = 4,
        *,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
        noise_seed: int | None = None,
        vae_chunk_halo_latent_frames: int = 0,
    ) -> torch.Tensor:
        assert (
            prompt_attention_mask is None and negative_prompt_attention_mask is None
        ), "Wan prompts are pre-encoded T5 embeddings; prompt attention masks are not supported"
        if negative_prompt is not None:
            logger.warning("KV-cache pipeline does not support classifier-free guidance; negative_prompt is ignored.")
        assert not self.transformer.training, (
            "KV-cache inference requires the transformer in eval mode (call `pipe.transformer.eval()`): "
            "train mode disables the rollout's windowed eviction, so long videos would run full attention "
            "instead of the deployed windowed/sink attention"
        )
        # With CP active, each rank VAE-encodes only its temporal shard (plus
        # a halo for the VAE's receptive field) and the shards are gathered;
        # without CP this is a plain full encode.
        condition = self.encode_video_frames_chunked_for_cp(rendered_rgb, vae_chunk_halo_latent_frames).cpu()

        batch_size = rendered_rgb.shape[0]
        neighbors_condition = self._encode_neighbors_checked(neighbors, batch_size, max_neighbors_per_encode)
        prompt_embeds = self._resolve_prompt_embeds(prompt, batch_size)

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
            noise_seed=noise_seed,
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
        # Keyword-only: trailing boolean flags must not bind positionally (a
        # positional call once silently swapped show_progress into
        # ignore_neighbors, dropping the reference views).
        *,
        ignore_neighbors: bool = False,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        # One generator per rollout, on the device the chunks are drawn on (the condition may be
        # CPU-staged). None for the DMD training rollout, whose noise stays on the global stream.
        noise_generator = self._noise_generator(noise_seed, self.vae.device)
        latent_num_frames = condition.shape[2]
        validate_latent_frames_divisible(latent_num_frames, self.ar_frames_per_chunk, "ar_frames_per_chunk")
        assert_training_rollout_fits_window(
            latent_num_frames, self.ar_window_chunks, self.ar_frames_per_chunk, training=self.transformer.training
        )

        cp_mesh = self.transformer.cp_mesh
        cp_active = cp_mesh is not None and cp_mesh.size() > 1
        cp_size = cp_mesh.size() if cp_active else 1
        cp_rank = self.transformer.cp_rank if cp_active else 0
        if cp_active:
            # Each rank denoises 1/cp_size of every rollout chunk, so the
            # chunk must split evenly across the CP mesh. Window and sink
            # spans are whole chunks by constructor validation, so per-chunk
            # sharding keeps every rank's cache eviction in lockstep.
            assert (
                self.ar_frames_per_chunk % cp_size == 0
            ), f"ar_frames_per_chunk ({self.ar_frames_per_chunk}) must be divisible by CP size ({cp_size})"

        kv_cache = RolloutKvCache(
            len(self.transformer.blocks),
            truncate_rollout_gradients=self.truncate_rollout_gradients,
        )
        current_start_frame = 0
        all_num_frames = [self.ar_frames_per_chunk] * (latent_num_frames // self.ar_frames_per_chunk)

        exit_flag = resolve_exit_step(num_inference_steps, use_exit_flag, device=self.vae.device)

        if neighbor_w2cs is not None:
            neighbor_w2cs = neighbor_w2cs.to(self.vae.device)
        if neighbor_Ks is not None:
            neighbor_Ks = neighbor_Ks.to(self.vae.device)

        output = None if cp_active else torch.zeros_like(condition, device=self.vae.device)
        output_blocks = [] if cp_active else None
        temporal_scale = self.vae.config.scale_factor_temporal
        cp_group = cp_mesh.get_group() if cp_active else None

        for current_num_frames in tqdm(all_num_frames, disable=not show_progress, leave=progress_bar_leave):
            current_end_frame = current_start_frame + current_num_frames
            kv_cache.evict_to_window(self.ar_window_chunks, self.ar_sink_chunks)
            if cp_active:
                local_num_frames = current_num_frames // cp_size
                local_start_frame = current_start_frame + cp_rank * local_num_frames
                local_end_frame = local_start_frame + local_num_frames
            else:
                local_start_frame = current_start_frame
                local_end_frame = current_end_frame

            raw_start, raw_end = conditioning_raw_frame_bounds(local_start_frame, local_end_frame, temporal_scale)
            chunk_condition = condition[:, :, local_start_frame:local_end_frame].to(self.vae.device)
            chunk_opacity = rendered_opacity[:, raw_start:raw_end].to(self.vae.device)

            latents = self.prepare_latents(
                chunk_condition,
                chunk_opacity,
                local_start_frame == 0,
                broadcast_cp_noise=not cp_active,
                generator=noise_generator,
            )

            chunk_camera_rays = slice_camera_tensor_frames(camera_rays, local_start_frame, local_end_frame).to(
                self.vae.device
            )
            chunk_w2cs = slice_camera_tensor_frames(w2cs, local_start_frame, local_end_frame).to(self.vae.device)
            chunk_Ks = slice_camera_tensor_frames(Ks, local_start_frame, local_end_frame).to(self.vae.device)

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
                kv_cache=kv_cache,
                frame_offset=local_start_frame,
                return_dict=False,
                local_cp_input=cp_active,
            )

            timesteps = create_denoising_step_list(
                self.scheduler,
                num_inference_steps=num_inference_steps,
                device=self.vae.device,
            )

            # The chunk's exit-step call feeds the cache: only its K/V are
            # ever read by later chunks (intermediate-step K/V never are),
            # and the global last chunk has no later reader.
            store_chunk_on_exit = current_end_frame < latent_num_frames

            def call_transformer(latents, t, is_exit_step):
                transformer_kwargs["hidden_states"] = latents
                transformer_timestep = self.prepare_transformer_timestep(t, latents)
                return self.transformer(
                    **transformer_kwargs,
                    timestep=transformer_timestep,
                    kv_cache_store_chunk=store_chunk_on_exit and is_exit_step,
                )[0]

            def draw_fresh_latents():
                return self.prepare_latents(
                    chunk_condition,
                    chunk_opacity,
                    local_start_frame == 0,
                    broadcast_cp_noise=not cp_active,
                    generator=noise_generator,
                )

            latents = run_chunk_denoise_loop(
                self.scheduler,
                timesteps,
                exit_flag,
                latents,
                call_transformer=call_transformer,
                draw_fresh_latents=draw_fresh_latents,
                device=self.vae.device,
            )

            if cp_active:
                assert output_blocks is not None and cp_group is not None
                # DMD scoring consumes global generated latents, while each CP
                # rank denoises only its local block for the current KV-cache
                # step. Gather after every block so the returned tensor matches
                # the non-CP global-latent contract.
                output_blocks.append(all_gather_cat_with_grad(latents, dim=2, group=cp_group))
            else:
                assert output is not None
                output[:, :, current_start_frame:current_end_frame] = latents

            current_start_frame = current_end_frame

        if cp_active:
            assert output_blocks is not None
            return torch.cat(output_blocks, dim=2)
        assert output is not None
        return output
