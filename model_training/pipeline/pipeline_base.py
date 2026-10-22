# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
from diffusers.video_processor import VideoProcessor
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from model_training.common.context_parallel import latent_range_to_raw_frame_slice
from model_training.constants import MAX_SEQUENCE_LENGTH
from model_training.net.conditioning import pad_opacity_first_frame
from model_training.net.context_parallel import all_gather_cat, transformer_cp_head_counts
from model_training.timestep import expand_transformer_timestep
from model_training.timestep import spatial_alignment as _spatial_alignment
from model_training.utils.validation_artifacts import tensor_identity


def validate_positive_frame_count(name: str, value: int | None) -> None:
    """Reject non-positive latent-frame-count knobs; None (knob unset) passes."""
    if value is not None and value < 1:
        raise ValueError(f"`{name}` must be a positive latent-frame count, got {value}")


def validate_rollout_window_options(local_attn_size: int, sink_size: int) -> None:
    """Range checks for the rollout-window options both families share:
    ``local_attn_size`` is -1 (unlimited window) or a positive latent-frame
    count, and ``sink_size`` is a non-negative latent-frame anchor that must
    fit inside a bounded window."""
    if local_attn_size != -1 and local_attn_size <= 0:
        raise ValueError("`local_attn_size` must be -1 (unlimited attention window) or a positive frame count")
    if sink_size < 0:
        raise ValueError("`sink_size` must be non-negative")
    if local_attn_size != -1 and sink_size >= local_attn_size:
        raise ValueError("`sink_size` must be smaller than `local_attn_size`")


def assert_training_rollout_fits_window(
    latent_frames: int, ar_window_chunks: int | None, ar_frames_per_chunk: int, *, training: bool
) -> None:
    """Training rollouts must fit the fixed attention window so they never
    evict cached history: a gradient-checkpointing recompute during backward
    re-reads the cache, and an evicted chunk would leave it reading the wrong
    history. Critic-only DMD phases run without gradients but must see the
    same attention span as student phases — hence the check on the
    transformer's train mode."""
    if not (torch.is_grad_enabled() or training):
        return
    assert ar_window_chunks is None or latent_frames <= ar_window_chunks * ar_frames_per_chunk, (
        f"Training rollouts must not evict cached history: {latent_frames} latent frames exceed the "
        f"{ar_window_chunks}-chunk attention window (ar_frames_per_chunk={ar_frames_per_chunk})"
    )


def validate_latent_frames_divisible(latent_frames: int, chunk_frames: int, knob_name: str) -> None:
    assert (
        latent_frames % chunk_frames == 0
    ), f"Latent frame count ({latent_frames}) must be divisible by {knob_name} ({chunk_frames})"


def prepare_transformer_for_training(transformer: nn.Module, attention_backend: str | None) -> None:
    """Set train mode and apply the requested attention backend: a requested
    backend is always applied, and a transformer without
    ``set_attention_backend`` raises instead of silently training on the
    default backend.
    """
    transformer.train()
    if attention_backend is None:
        return
    transformer.set_attention_backend(attention_backend)


def create_denoising_step_list(scheduler, *, num_inference_steps: int, device: torch.device | str) -> torch.Tensor:
    denoising_step_list = torch.linspace(
        scheduler.num_train_timesteps,
        0,
        num_inference_steps + 1,
        dtype=torch.long,
        device=device,
    )[:-1]
    timesteps = torch.cat(
        (
            scheduler.timesteps,
            torch.tensor([0], dtype=torch.float32, device=device),
        )
    )
    return timesteps[scheduler.num_train_timesteps - denoising_step_list]


def generate_and_sync_exit_flag(num_inference_steps: int, *, device: torch.device | str) -> int:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        exit_flag = torch.randint(low=0, high=num_inference_steps, size=(1,), device=device)
    else:
        exit_flag = torch.empty(1, dtype=torch.long, device=device)

    if dist.is_initialized():
        dist.broadcast(exit_flag, src=0)
    return int(exit_flag.item())


def resolve_exit_step(num_inference_steps: int, use_exit_flag: bool, *, device: torch.device | str) -> int:
    """Index of the denoising step whose output the rollout keeps.

    With the exit flag, one rank-synced random index is shared across every
    chunk of the rollout (truncated Self-Forcing training); without it, the
    rollout runs all steps and exits on the last one.
    """
    if use_exit_flag:
        return generate_and_sync_exit_flag(num_inference_steps, device=device)
    return num_inference_steps - 1


def uncompressed_chunk_frames(
    chunk_start_latent_frame: int, chunk_latent_frames: int, scale_factor_temporal: int
) -> int:
    """Pixel-frame count of a rollout chunk under the VAE's 1+4N temporal
    contract: the first chunk owns pixel frame 0."""
    return (chunk_latent_frames - 1) * scale_factor_temporal + (
        1 if chunk_start_latent_frame == 0 else scale_factor_temporal
    )


def run_chunk_denoise_loop(
    scheduler,
    timesteps: torch.Tensor,
    exit_flag: int,
    latents: torch.Tensor,
    *,
    call_transformer: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
    draw_fresh_latents: Callable[[], torch.Tensor],
    device: torch.device | str,
) -> torch.Tensor:
    """Few-step denoise loop for one AR rollout chunk, shared by both
    families' rollout pipelines.

    Pre-exit steps run under no_grad, step to the final estimate, and SDE
    re-noise toward a fresh ``draw_fresh_latents()`` draw at the next
    timestep; the exit step runs in the ambient grad context and breaks.
    ``call_transformer(latents, t, is_exit_step)`` closes over each family's
    conditioning kwargs (and timestep preparation) for one transformer
    forward; ``is_exit_step`` marks the exit call so callers key exit-only
    work (the Wan chunk store) off the loop's own state.
    """
    for index, t in enumerate(timesteps):
        should_exit = index == exit_flag

        if not should_exit:
            with torch.no_grad():
                noise_pred = call_transformer(latents, t, should_exit)
                latents = scheduler.step(noise_pred, t, latents, to_final=True)
                next_timestep = timesteps[index + 1]
                latents = scheduler.add_noise(
                    latents,
                    draw_fresh_latents(),
                    next_timestep
                    * torch.ones(
                        (latents.shape[0],),
                        device=device,
                        dtype=torch.long,
                    ),
                )
        else:
            noise_pred = call_transformer(latents, t, should_exit)
            latents = scheduler.step(noise_pred, t, latents, to_final=True)
            break
    return latents


class ArtifixerPipelineBase(nn.Module, ABC):
    uses_classifier_free_guidance = True

    def validation_negative_conditioning(
        self, prompt, prompt_attention_mask, negative_prompt, negative_prompt_attention_mask
    ) -> dict | None:
        """Fingerprint the resolved unconditional tensors consumed by validation CFG."""
        if not self.uses_classifier_free_guidance:
            return None
        _, unconditional = self._resolve_text_conditioning(
            prompt, prompt_attention_mask, negative_prompt, negative_prompt_attention_mask, 1, self.vae.device
        )
        return {name: tensor_identity(value) for name, value in unconditional.items()} if unconditional else None

    # Eval reads this to decide where the full-clip conditioning (rendered
    # frames, opacity, camera tensors) lives: rollout pipelines consume it in
    # per-chunk slices and move each slice to the device themselves, so the
    # caller may stage the whole clip on the CPU. Bidirectional pipelines
    # take the whole clip in one pass and expect it on the device.
    stages_conditioning_per_chunk = False

    def __init__(
        self,
        vae: AutoencoderKLWan,
        tokenizer: AutoTokenizer | None,
        text_encoder: UMT5EncoderModel | None,
        expand_timesteps: bool,
        frames_per_block: int | None = None,
        attention_backend: str | None = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.attention_backend = attention_backend

        if text_encoder is not None:
            assert tokenizer is not None, "`tokenizer` is required when `text_encoder` is provided"
            self.text_encoder = text_encoder
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
        else:
            self.text_encoder = None

        self.vae = vae
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.scale_factor_spatial)
        self.expand_timesteps = expand_timesteps
        # Latent frames per block of the block-causal attention mask; None =
        # bidirectional. Trainers read it here, not from transformer
        # internals. Validation CP group sizes must divide the block count.
        # Validates the knob passed to this constructor; pipelines with their
        # own frame-count knobs (e.g. the rollout chunk size) validate those
        # through the same shared helper.
        validate_positive_frame_count("frames_per_block", frames_per_block)
        self.causal_block_frames: int | None = frames_per_block

        self.latents_mean = torch.nn.Buffer(
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype),
            persistent=False,
        )
        self.latents_inv_std = torch.nn.Buffer(
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype),
            persistent=False,
        )

    def _cp_frame_divisor(self, total_latent_frames: int) -> int:
        """Return the largest latent-frame count whose divisors are valid CP
        group sizes for this pipeline's validation path, or 1 to opt out.

        CP consumers (trainer validation and run_inference) enable CP on the
        transformer and still call ``forward_inference`` with complete clips,
        so a pipeline may only advertise a divisor above 1 if that full-input
        call works under an active CP mesh — by sharding per rank inside the
        pipeline (the Wan pipeline slices its shard, denoises it through
        ``forward_inference_local_cp``, and gathers the latent shards before
        decoding) or inside the transformer (Cosmos3).  Neither mechanism is
        generic, so the base class opts out.
        """
        return 1

    def cp_head_divisor(self, required: bool = False) -> int | None:
        """The gcd of the transformer's query and KV attention head counts —
        the value a CP size must divide wherever this pipeline splits heads
        across the CP group. The base rule: full-input CP inference always
        splits heads, so the gcd always applies; the Wan rollout pipeline
        overrides this to None because its cache path keeps all heads on
        every rank. ``required=True`` raises instead of returning None when
        the transformer exposes no CP-capable attention (see
        ``transformer_cp_head_counts``)."""
        head_counts = transformer_cp_head_counts(self.transformer, required=required)
        if head_counts is None:
            return None
        return math.gcd(*head_counts)

    def cp_size_candidates(self, num_latent_frames: int, world_size: int) -> list[int]:
        """The CP group sizes (ascending, 1 always included) this pipeline
        can run full-input validation inference at. Base rule: every rank
        needs whole latent frames — the size must divide the frame divisor,
        and for block-causal configs also the block count so each rank holds
        whole temporal blocks — and the attention head gcd
        (``cp_head_divisor``; None means no head rule)."""
        frame_divisor = self._cp_frame_divisor(num_latent_frames)
        if self.causal_block_frames is not None:
            frame_divisor = math.gcd(frame_divisor, num_latent_frames // self.causal_block_frames)
        head_divisor = self.cp_head_divisor()
        return [
            size
            for size in range(1, world_size + 1)
            if frame_divisor % size == 0 and (head_divisor is None or head_divisor % size == 0)
        ]

    def cp_pass_latent_frames(self, num_latent_frames: int) -> int:
        """Latent frames one transformer pass shards across the CP group —
        the whole clip here; rollout pipelines override with their chunk
        size. Validation weighs CP-size candidates by this per-pass cost."""
        return num_latent_frames

    def cp_raw_frame_padding(self, num_frames: int, cp_size: int, vae_temporal_scale: int) -> int:
        """Raw frames eval must append so this pipeline can run at
        ``cp_size``: the transformer slices whole frame shards per rank, so
        the latent count must divide the CP size. Cosmos3 overrides this to
        0 — its transformer pads its own frame axis and cuts the pads from
        the output, so eval feeds the SAME raw frames at every CP size (the
        one padding scheme validation uses, keeping eval and validation
        numbers comparable for the same clip)."""
        if cp_size <= 1:
            return 0
        padded_num_frames = num_frames
        while (((padded_num_frames - 1) // vae_temporal_scale) + 1) % cp_size != 0:
            padded_num_frames += vae_temporal_scale
        return padded_num_frames - num_frames

    def validate_cp_size(self, cp_size: int, num_latent_frames: int) -> None:
        """Assert ``cp_size`` is runnable before any collective starts: the
        same mistake caught later surfaces as shape errors or a peer hang
        inside the first forward's all-to-alls, not a named rule. Admits
        exactly the sizes ``cp_size_candidates`` lists (keep the two in
        step), split into per-rule asserts so the error names the rule that
        refused the size."""
        frame_divisor = self._cp_frame_divisor(num_latent_frames)
        assert frame_divisor % cp_size == 0, (
            f"CP size {cp_size} must divide the valid CP frame divisor {frame_divisor} "
            f"({num_latent_frames} latent frames): this pipeline's transformer cannot pad its own frame axis."
        )
        head_divisor = self.cp_head_divisor(required=True)
        assert head_divisor is None or head_divisor % cp_size == 0, (
            f"CP size {cp_size} must divide the transformer's attention head divisor {head_divisor}: CP splits "
            "the attention heads across ranks, so every rank needs a whole number of query and KV heads."
        )

    def latent_frame_multiple(self) -> int:
        """Inference clips must hold a whole multiple of this many latent
        frames; eval pads raw frames up to it. Rollout pipelines override
        with their chunk size — a rollout consumes whole chunks. 1: any
        latent count runs."""
        return 1

    def patch_size(self) -> tuple[int, int, int]:
        return tuple(self.transformer.patch_size)

    def spatial_alignment(self) -> tuple[int, int]:
        return _spatial_alignment(self.vae.config.scale_factor_spatial, self.patch_size())

    def dummy_validation_prompt(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Batch-of-one (prompt, prompt_attention_mask, negative_prompt,
        negative_prompt_attention_mask) for validation forwards whose output
        is discarded (spectator ranks and idle groups). This family feeds
        prompt embeddings: a zero embedding at the transformer's text width,
        in bfloat16 like dataset-encoded prompts; no masks, and no negative
        prompt (None disables CFG)."""
        prompt_dim = int(self.transformer.condition_embedder.text_embedder.linear_1.in_features)
        prompt = torch.zeros(1, MAX_SEQUENCE_LENGTH, prompt_dim, device=device, dtype=torch.bfloat16)
        return prompt, None, None, None

    def prepare_transformer_timestep(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        frames_per_block: int | None = None,
    ) -> torch.Tensor:
        """Shape and dtype timesteps for the active backbone.

        Wan2.1 accepts one timestep per sample, while Wan2.2 TI2V-5B's
        Diffusers config sets ``expand_timesteps=true`` and the reference
        WanPipeline sends one timestep per post-patch latent token.

        Per-block training (diffusion forcing) may already
        sample one timestep per latent frame block; in that case expand each
        block's timestep across the tokens belonging to that block.

        Note on cost: for every timestep pattern this repo currently produces
        (uniform, per-sample, per-block), the expanded path is numerically
        identical to feeding the transformer the un-expanded tensor — the
        expansion exists to match the reference WanPipeline exactly and to
        allow future per-token conditioning (e.g. the TI2V image mask, which
        sets t=0 on first-frame tokens). It materializes per-token embeddings
        (roughly seq_len x 6 x dim per sample), which is significant at high
        resolutions.

        The returned tensor is always float32, for every family: timesteps
        stay float32 into the embedder because half precision quantizes the
        [0, 1000] timestep grid before the sinusoid projection (bf16 rounds
        999 to 1000 and shifts the highest-frequency embedding band). Both
        families' time embedders accept fp32 and cast only after the
        sinusoid, so callers must not re-cast the result to the latents'
        dtype.
        """
        if not self.expand_timesteps:
            shaped = timestep.expand(hidden_states.shape[0]) if timestep.ndim == 0 else timestep
        else:
            shaped = expand_transformer_timestep(timestep, hidden_states, self.patch_size(), frames_per_block)
        return shaped.to(torch.float32)

    def clear_inference_caches(self) -> None:
        """Drop any inference caches that outlive a forward_inference call.

        The default is a no-op: pipelines whose caches are local to one call
        have nothing to drop. The Wan distilled pipeline overrides this to
        release its persistent attention caches between the denoise and
        decode phases of evaluation.
        """

    def _validate_spatial_alignment(self, height: int, width: int, *, label: str) -> None:
        height_multiple, width_multiple = self.spatial_alignment()
        if height % height_multiple != 0 or width % width_multiple != 0:
            raise ValueError(
                f"`{label}` height/width must be divisible by ({height_multiple}, {width_multiple}) "
                f"for VAE scale and transformer patchification, got ({height}, {width})"
            )

    @abstractmethod
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
        neighbor_camera_rays: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def rgb_to_latents(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.to(device=self.vae.device, dtype=self.vae.dtype)
        latents = retrieve_latents(self.vae.encode(rgb), sample_mode="argmax")
        return (latents - self.latents_mean) * self.latents_inv_std

    def latents_to_rgb(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.vae.dtype)
        latents = latents / self.latents_inv_std + self.latents_mean
        return self.vae.decode(latents, return_dict=False)[0]

    def decode_latents_to_video(self, latents: torch.Tensor, temporal_chunk_size: int | None = None) -> torch.Tensor:
        if temporal_chunk_size is not None and latents.shape[2] > temporal_chunk_size:
            chunks = []
            for latent_start in range(0, latents.shape[2], temporal_chunk_size):
                latent_end = min(latents.shape[2], latent_start + temporal_chunk_size)
                decode_start = max(0, latent_start - 1)
                video = self.latents_to_rgb(latents[:, :, decode_start:latent_end])
                video = self.video_processor.postprocess_video(video, output_type="pt")
                if latent_start > 0:
                    video = video[:, 1:]
                chunks.append(video.cpu())
            return torch.cat(chunks, dim=1)

        video = self.latents_to_rgb(latents)
        return self.video_processor.postprocess_video(video, output_type="pt")

    def supports_local_cp_inference(self) -> bool:
        # True only when the wrapped transformer accepts local_cp_input frame
        # shards for this pipeline's active config; the trainer routes CP
        # validation through forward_inference_local_cp when True.
        return False

    def use_local_cp_inference(self, cp_size: int, num_latent_frames: int) -> bool:
        """How a CP validation task reaches this pipeline: True to slice the
        inputs per rank and call ``forward_inference_local_cp``, False to
        give every rank the full inputs for ``forward_inference``. The local
        path serves whole latent frames per rank, so a clip length the CP
        size does not divide runs full-input even when the active config
        supports slicing."""
        return self.supports_local_cp_inference() and num_latent_frames % cp_size == 0

    def _resolve_text_conditioning(
        self,
        prompt,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt,
        negative_prompt_attention_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[dict, dict | None]:
        """Transformer text kwargs for the conditional pass and the
        unconditional CFG pass; a None unconditional disables CFG."""
        raise NotImplementedError

    @torch.no_grad()
    def forward_inference_local_cp(
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
        *,
        local_frame_offset: int,
        local_vae_crop: tuple[int, int],
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt: str | torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 5.0,
        show_progress: bool = False,
        progress_bar_leave: bool = True,
        max_neighbors_per_encode: int | None = None,
        noise_seed: int | None = None,
        neighbor_camera_rays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CP-validation path for pipelines whose transformer serves
        pre-sliced frame shards.

        The caller has already sliced temporal inputs for this CP rank:
        ``rendered_rgb`` is the raw VAE window and may include left halo,
        ``local_vae_crop`` trims that encode back to this rank's latent shard,
        ``rendered_opacity`` is the raw conditioning window for that shard, and
        camera tensors may be raw-temporal or latent-temporal depending on their
        source layout.  Return local latents instead of decoded video; the
        caller gathers latent shards and decodes the full clip once.
        """
        assert self.supports_local_cp_inference(), (
            f"{type(self).__name__} cannot serve pre-sliced CP shards with its active config "
            "(see supports_local_cp_inference)"
        )
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
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt=negative_prompt,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            show_progress=show_progress,
            progress_bar_leave=progress_bar_leave,
            max_neighbors_per_encode=max_neighbors_per_encode,
            noise_seed=noise_seed,
            local_cp_input=True,
            local_frame_offset=local_frame_offset,
            local_vae_crop=local_vae_crop,
            neighbor_camera_rays=neighbor_camera_rays,
        )

    @staticmethod
    def _noise_generator(noise_seed: int | None, device: torch.device) -> torch.Generator | None:
        """A generator on ``device`` seeded with ``noise_seed``, or None for the global stream.
        Every rank seeds identically, which keeps the CP paths' shared noise shared."""
        if noise_seed is None:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(int(noise_seed) % (2**63 - 1))
        return generator

    def _denoise_to_latents(
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
        local_cp_input: bool,
        local_frame_offset: int,
        local_vae_crop: tuple[int, int] | None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        noise_seed: int | None = None,
        *,
        neighbor_camera_rays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self.encode_video_frames(rendered_rgb)
        if local_vae_crop is not None:
            assert (
                0 <= local_vae_crop[0] < local_vae_crop[1] <= condition.shape[2]
            ), f"local_vae_crop {local_vae_crop} is outside encoded latent length {condition.shape[2]}"
            condition = condition[:, :, local_vae_crop[0] : local_vae_crop[1]].contiguous()
        if local_cp_input:
            temporal_scale = self.vae.config.scale_factor_temporal
            expected_opacity_frames = (
                1 + temporal_scale * (condition.shape[2] - 1)
                if local_frame_offset == 0
                else temporal_scale * condition.shape[2]
            )
            assert rendered_opacity.shape[1] == expected_opacity_frames, (
                f"rendered_opacity has {rendered_opacity.shape[1]} raw frames, expected "
                f"{expected_opacity_frames} for {condition.shape[2]} local latent frames"
            )

        batch_size = rendered_rgb.shape[0]
        neighbors_condition = self._encode_neighbors_checked(neighbors, batch_size, max_neighbors_per_encode)

        latents = self.prepare_latents(
            condition,
            rendered_opacity,
            local_frame_offset == 0,
            broadcast_cp_noise=not local_cp_input,
            generator=self._noise_generator(noise_seed, condition.device),
        )

        cond_kwargs, uncond_kwargs = self._resolve_text_conditioning(
            prompt,
            prompt_attention_mask,
            negative_prompt,
            negative_prompt_attention_mask,
            batch_size,
            latents.device,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=self.vae.device)
        timesteps = self.scheduler.timesteps

        for t in tqdm(timesteps, disable=not show_progress, leave=progress_bar_leave):
            timestep = self.prepare_transformer_timestep(t, latents)

            common_kwargs = dict(
                hidden_states=latents,
                timestep=timestep,
                neighbor_hidden_states=neighbors_condition,
                opacity=rendered_opacity,
                camera_rays=camera_rays,
                w2cs=w2cs,
                neighbor_w2cs=neighbor_w2cs,
                Ks=Ks,
                neighbor_Ks=neighbor_Ks,
                return_dict=False,
                local_cp_input=local_cp_input,
                frame_offset=local_frame_offset,
            )
            if neighbor_camera_rays is not None:
                common_kwargs["neighbor_camera_rays"] = neighbor_camera_rays

            noise_pred = self.transformer(**cond_kwargs, **common_kwargs)[0]

            if uncond_kwargs is not None and text_guidance_scale > 1:
                noise_uncond = self.transformer(**uncond_kwargs, **common_kwargs)[0]
                noise_pred = noise_uncond + text_guidance_scale * (noise_pred - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents

    def _sample_cp_local_noise_like(
        self, reference: torch.Tensor, cp_mesh, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        assert reference.ndim == 5, f"latent noise reference must be B,C,T,H,W, got {tuple(reference.shape)}"
        assert dist.is_initialized(), "local CP latent noise synchronization requires distributed training"

        cp_group = cp_mesh.get_group()
        cp_rank = dist.get_rank(group=cp_group)
        cp_size = cp_mesh.size()
        batch_size, channels, local_frames, height, width = reference.shape

        if cp_rank == 0:
            global_noise = torch.randn(
                (batch_size, channels, local_frames * cp_size, height, width),
                generator=generator,
                device=reference.device,
                dtype=reference.dtype,
            )
            local_noise = None
            for rank_index in range(cp_size):
                chunk = global_noise[:, :, rank_index * local_frames : (rank_index + 1) * local_frames].contiguous()
                if rank_index == 0:
                    local_noise = chunk
                else:
                    request = dist.isend(chunk, group=cp_group, group_dst=rank_index)
                    request.wait()
                    del chunk
            assert local_noise is not None, "CP source rank did not keep its local latent-noise slice"
            return local_noise

        local_noise = torch.empty_like(reference)
        request = dist.irecv(local_noise, group=cp_group, group_src=0)
        request.wait()
        return local_noise

    def prepare_latents(
        self,
        condition: torch.Tensor,
        opacity: torch.Tensor,
        is_first_chunk: bool,
        *,
        broadcast_cp_noise: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Mix the conditioning latents with noise. ``generator`` is None in training (the global
        stream) and set by validation, whose draws must not depend on training's."""
        # Add a channel dim in both branches: max_pool3d treats a 4D
        # [B, T, H, W] input as unbatched, and the pooled mask's batch dim
        # would then broadcast onto the latents' channel dim.
        opacity_for_mixing = (
            pad_opacity_first_frame(opacity, self.vae.config.scale_factor_temporal) if is_first_chunk else opacity
        ).unsqueeze(1)
        opacity_for_mixing = torch.nn.functional.max_pool3d(
            opacity_for_mixing,
            (
                self.vae.config.scale_factor_temporal,
                self.vae.config.scale_factor_spatial,
                self.vae.config.scale_factor_spatial,
            ),
        )
        cp_mesh = self.transformer.cp_mesh
        if cp_mesh is not None and cp_mesh.size() > 1 and not broadcast_cp_noise:
            # Local-CP callers pass one temporal shard per rank.  Rank 0 samples
            # the same global noise tensor a CP=1 run would consume, then scatters
            # contiguous temporal chunks so later RNG draws remain CP=1-aligned.
            noise = self._sample_cp_local_noise_like(condition, cp_mesh, generator=generator)
        elif generator is None:
            # Training's original call, kept verbatim so training's noise is unchanged.
            noise = torch.randn_like(condition)
        else:
            noise = torch.randn(condition.shape, generator=generator, device=condition.device, dtype=condition.dtype)

        latents = condition * opacity_for_mixing + noise * (1 - opacity_for_mixing)

        # Full-input CP paths still need identical complete latents on every CP
        # rank before the transformer shards internally.
        if broadcast_cp_noise and cp_mesh is not None:
            src_rank = cp_mesh.mesh.view(-1)[0].item()
            dist.broadcast(latents, src=src_rank, group=cp_mesh.get_group())

        return latents

    def encode_video_frames(self, video_frames: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _, height, width = video_frames.shape
        self._validate_spatial_alignment(height, width, label="video_frames")
        assert (
            num_frames - 1
        ) % self.vae.config.scale_factor_temporal == 0, (
            f"`num_frames - 1` has to be divisible by {self.vae.config.scale_factor_temporal}"
        )

        video_frames = (
            self.video_processor.preprocess(video_frames.flatten(0, 1))
            .view(batch_size, num_frames, 3, height, width)
            .permute(0, 2, 1, 3, 4)
        )

        return self.rgb_to_latents(video_frames)

    @torch.no_grad()
    def encode_video_frames_chunked_for_cp(
        self,
        video_frames: torch.Tensor,
        halo_latent_frames: int,
        *,
        gather: bool = True,
    ) -> torch.Tensor:
        cp_mesh = self.transformer.cp_mesh
        if cp_mesh is None or cp_mesh.size() == 1:
            return self.encode_video_frames(video_frames)
        assert halo_latent_frames >= 0, "halo_latent_frames must be non-negative"

        batch_size, num_frames, _, height, width = video_frames.shape
        self._validate_spatial_alignment(height, width, label="video_frames")
        temporal_scale = self.vae.config.scale_factor_temporal
        assert (num_frames - 1) % temporal_scale == 0, f"`num_frames - 1` has to be divisible by {temporal_scale}"

        total_latent_frames = (num_frames - 1) // temporal_scale + 1
        cp_size = cp_mesh.size()
        assert (
            total_latent_frames % cp_size == 0
        ), f"Latent frame count {total_latent_frames} must be divisible by CP size {cp_size}"

        cp_group = cp_mesh.get_group()
        cp_rank = dist.get_rank(group=cp_group)
        frames_per_rank = total_latent_frames // cp_size
        latent_start = cp_rank * frames_per_rank
        latent_end = latent_start + frames_per_rank
        raw_start, raw_end, crop_start, crop_end = latent_range_to_raw_frame_slice(
            latent_start,
            latent_end,
            total_raw_frames=num_frames,
            temporal_scale=temporal_scale,
            halo_latent_frames=halo_latent_frames,
        )

        local_latents = self.encode_video_frames(video_frames[:, raw_start:raw_end])[:, :, crop_start:crop_end]
        if not gather:
            return local_latents.contiguous()
        return all_gather_cat(local_latents, cp_mesh, dim=2)

    def _encode_neighbors_batch(self, neighbor_frames: torch.Tensor) -> torch.Tensor:
        # Callers may stage neighbors on CPU (e.g. DMD validation batches).
        neighbor_frames = neighbor_frames.to(self.vae.device)
        batch_size, num_neighbors, _, neighbors_height, neighbors_width = neighbor_frames.shape
        self._validate_spatial_alignment(neighbors_height, neighbors_width, label="neighbor_frames")

        # Not using temporal compression atm
        neighbors_condition = (
            self.video_processor.preprocess(neighbor_frames.flatten(0, 1))
            .view(-1, 1, 3, neighbors_height, neighbors_width)
            .permute(0, 2, 1, 3, 4)
        )
        neighbors_condition = self.rgb_to_latents(neighbors_condition)
        neighbors_condition = neighbors_condition.squeeze(2)
        neighbors_condition = neighbors_condition.view(batch_size, num_neighbors, *neighbors_condition.shape[1:])
        return neighbors_condition.permute(0, 2, 1, 3, 4)

    def encode_neighbors(
        self, neighbor_frames: torch.Tensor, max_neighbors_per_encode: int | None = None
    ) -> torch.Tensor:
        if max_neighbors_per_encode is None:
            return self._encode_neighbors_batch(neighbor_frames.to(self.vae.device))
        if max_neighbors_per_encode <= 0:
            raise ValueError("`max_neighbors_per_encode` must be positive when set")

        chunks = [
            self._encode_neighbors_batch(neighbor)
            for neighbor in neighbor_frames.split(max_neighbors_per_encode, dim=1)
        ]
        return torch.cat(chunks, dim=2)

    def _encode_neighbors_checked(
        self, neighbors: torch.Tensor | None, batch_size: int, max_neighbors_per_encode: int | None
    ) -> torch.Tensor | None:
        if neighbors is None:
            return None
        if neighbors.shape[0] != batch_size:
            raise ValueError("`neighbors` batch size has to match the rendering batch size")
        return self.encode_neighbors(neighbors, max_neighbors_per_encode)

    def _resolve_prompt_embeds(self, prompt: str | torch.Tensor, batch_size: int) -> torch.Tensor:
        if isinstance(prompt, torch.Tensor):
            if prompt.ndim != 3:
                raise ValueError("An encoded prompt has to be a 3D tensor")
            if prompt.shape[0] != batch_size:
                raise ValueError("Encoded prompt batch size has to match the rendering batch size")
            return prompt.to(self.vae.device)
        return self.get_t5_prompt_embeds(prompt).expand(batch_size, -1, -1)

    def get_t5_prompt_embeds(self, prompt: str) -> torch.Tensor:
        assert self.tokenizer is not None, "`tokenizer` is required when getting T5 prompt embeddings"
        assert self.text_encoder is not None, "`text_encoder` is required when getting T5 prompt embeddings"

        text_inputs = self.tokenizer(
            prompt_clean(prompt),
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            mask.to(self.text_encoder.device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.vae.dtype, device=self.vae.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(MAX_SEQUENCE_LENGTH - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        return prompt_embeds
