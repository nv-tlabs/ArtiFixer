# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from diffusers import WanTransformer3DModel
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanRotaryPosEmbed,
    WanTransformerBlock,
    _get_added_kv_projections,
    _get_qkv_projections,
)
from einops import rearrange
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from model_training.common.context_parallel import all_to_all_with_grad, frame_block_causal_mask, gather_kv_cat
from model_training.net import attention_config
from model_training.net.conditioning import camera_embedding_input_dim, opacity_embedding_input_dim, pack_opacity_tokens
from model_training.net.context_parallel import clear_context_parallel_state, set_context_parallel_state
from model_training.net.prope import PropeApplyFns, PropeDotProductAttention
from model_training.net.rng import rank_invariant_init
from model_training.net.rollout_kv_cache import RolloutKvCache

flex_attention = torch.compile(attention_config.flex_attention_with_options, dynamic=False)


# Context-parallel contract:
# CP callers pass the local latent-frame shard plus its global frame_offset.
# The transformer never owns the full CP sequence under CP; it keeps MLP,
# modulation, projection, and output tensors local, while self-attention uses
# Ulysses all-to-all to exchange sequence/head shards. Validation gathers local
# latent outputs outside the transformer when it needs to save a global video.


# ---------------------------------------------------------------------------
# Frame-shaped Wan modulation helpers
# ---------------------------------------------------------------------------


def _wan_block_frame_modulation_value(
    scale_shift_table: torch.Tensor,
    temb: torch.Tensor,
    index: int,
    *,
    batch_size: int,
    hidden_seq_len: int,
    frame_seqlen: int,
) -> torch.Tensor:
    """Return block AdaLN values as [B, frames, 1, C] without token expansion."""
    assert (
        hidden_seq_len % frame_seqlen == 0
    ), f"hidden_seq_len={hidden_seq_len} must be divisible by frame_seqlen={frame_seqlen}"
    frames = hidden_seq_len // frame_seqlen
    if temb.ndim == 4:
        assert temb.shape[1] in (
            frames,
            hidden_seq_len,
        ), f"Expanded timestep length must be {frames} frames or {hidden_seq_len} tokens, got {temb.shape[1]}"
        if temb.shape[1] == frames:
            frame_temb = temb[:, :, index]
        else:
            frame_temb = temb[:, ::frame_seqlen, index]
        return scale_shift_table[:, index].view(1, 1, 1, -1) + frame_temb.float().unsqueeze(2)

    value = scale_shift_table[:, index] + temb[:, index].float()
    if value.shape[0] == batch_size:
        return value.unsqueeze(1).unsqueeze(2).expand(-1, frames, -1, -1)
    value = value.unflatten(0, (batch_size, value.shape[0] // batch_size))
    assert frames % value.shape[1] == 0, f"Cannot broadcast {value.shape[1]} timestep segments over {frames} frames"
    return value.repeat_interleave(frames // value.shape[1], dim=1).unsqueeze(2)


def _wan_output_frame_modulation_value(
    scale_shift_table: torch.Tensor,
    temb: torch.Tensor,
    index: int,
    *,
    batch_size: int,
    hidden_seq_len: int,
    frame_seqlen: int,
) -> torch.Tensor:
    """Return output AdaLN values as [B, frames, 1, C] without token expansion."""
    assert (
        hidden_seq_len % frame_seqlen == 0
    ), f"hidden_seq_len={hidden_seq_len} must be divisible by frame_seqlen={frame_seqlen}"
    frames = hidden_seq_len // frame_seqlen
    if temb.ndim == 3:
        assert temb.shape[1] in (
            frames,
            hidden_seq_len,
        ), f"Output timestep length must be {frames} frames or {hidden_seq_len} tokens, got {temb.shape[1]}"
        if temb.shape[1] == frames:
            frame_temb = temb
        else:
            frame_temb = temb[:, ::frame_seqlen]
        return scale_shift_table[:, index].view(1, 1, 1, -1) + frame_temb.float().unsqueeze(2)

    value = scale_shift_table[:, index] + temb.float()
    if value.shape[0] == batch_size:
        return value.unsqueeze(1).unsqueeze(2).expand(-1, frames, -1, -1)
    value = value.unflatten(0, (batch_size, value.shape[0] // batch_size))
    assert frames % value.shape[1] == 0, f"Cannot broadcast {value.shape[1]} timestep segments over {frames} frames"
    return value.repeat_interleave(frames // value.shape[1], dim=1).unsqueeze(2)


def _modulated_layer_norm(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    frame_seqlen: int,
) -> torch.Tensor:
    frames = hidden_states.shape[1] // frame_seqlen
    value = norm(hidden_states.float()).unflatten(1, (frames, frame_seqlen))
    return (value * (1 + scale) + shift).flatten(1, 2).type_as(hidden_states)


def _apply_frame_gate(
    hidden_states: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    *,
    frame_seqlen: int,
) -> torch.Tensor:
    frames = hidden_states.shape[1] // frame_seqlen
    # The AdaLN gate arrives in fp32; the gate product and the residual add
    # must also run in fp32, rounding back to the model dtype only once.
    gated = update.float().unflatten(1, (frames, frame_seqlen)) * gate
    residual = hidden_states.float().unflatten(1, (frames, frame_seqlen)) + gated
    return residual.flatten(1, 2).type_as(hidden_states)


@dataclass
class _KvCacheStage:
    """Keep cache writes shared when FSDP rebuilds argument containers."""

    layers: dict[int, tuple[torch.Tensor, torch.Tensor]]


class ArtifixerTransformer(nn.Module):
    # Context-parallelism state — set via enable/disable_context_parallel.
    _cp_mesh: torch.distributed.DeviceMesh | None = None
    _cp_rank: int = 0
    _cp_world_size: int = 1

    @property
    def cp_mesh(self) -> torch.distributed.DeviceMesh | None:
        return self._cp_mesh

    @property
    def cp_rank(self) -> int:
        return self._cp_rank

    @property
    def cp_world_size(self) -> int:
        return self._cp_world_size

    @rank_invariant_init
    def __init__(
        self,
        base_transformer: WanTransformer3DModel,
        frames_per_block: int | None,
        use_kv_cache_attention: bool,
        vae_scale_factor_spatial: int,
        vae_scale_factor_temporal: int,
        gradient_checkpointing: bool,
        checkpoint_every_n_blocks: int,
    ):
        super().__init__()

        self.rope = OffsetableWanRotaryPosEmbed(
            attention_head_dim=base_transformer.rope.attention_head_dim,
            patch_size=base_transformer.rope.patch_size,
            max_seq_len=base_transformer.rope.max_seq_len,
        )

        # TODO: Add clip image embeddings?
        self.patch_embedding = base_transformer.patch_embedding

        self.patch_size = base_transformer.config.patch_size
        self.condition_embedder = base_transformer.condition_embedder

        opacity_embedding_dim = opacity_embedding_input_dim(
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_height=self.patch_size[1],
            patch_width=self.patch_size[2],
        )
        camera_embedding_dim = camera_embedding_input_dim(
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_height=self.patch_size[1],
            patch_width=self.patch_size[2],
        )

        self.frames_per_block = frames_per_block
        self.block_masks = {}
        self.blocks = nn.ModuleList(
            [
                ArtifixerTransformerBlock(
                    block,
                    opacity_embedding_dim,
                    camera_embedding_dim,
                    use_kv_cache_attention,
                )
                for block in base_transformer.blocks
            ]
        )

        self.norm_out = base_transformer.norm_out
        self.proj_out = base_transformer.proj_out
        self.scale_shift_table = base_transformer.scale_shift_table
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_every_n_blocks = checkpoint_every_n_blocks

        # PRoPE for neighbor cross-attention. Names follow the PRoPE paper:
        # `target` holds the target-frame cameras (transforms applied to Q and
        # O), `reference` the reference-view cameras (applied to K/V).
        self.prope_target = PropeDotProductAttention(head_dim=self.rope.attention_head_dim)
        self.prope_reference = PropeDotProductAttention(head_dim=self.rope.attention_head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        neighbor_hidden_states: torch.Tensor | None,
        opacity: torch.Tensor,
        camera_rays: torch.Tensor,
        w2cs: torch.Tensor,
        neighbor_w2cs: torch.Tensor | None,
        Ks: torch.Tensor,
        neighbor_Ks: torch.Tensor | None,
        ignore_neighbors: bool = False,
        kv_cache: RolloutKvCache | None = None,
        kv_cache_store_chunk: bool = False,
        frame_offset: int = 0,
        return_dict: bool = False,
        local_cp_input: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if kv_cache is not None:
            # Under CP the cache stores this rank's frame shard, so it counts
            # frames in LOCAL coordinates while frame_offset is the shard's
            # GLOBAL start. Subtracting the rank offset reconstructs the global
            # chunk start (shard start = chunk_start + rank * frames_per_rank,
            # see cp_frame_slice) and dividing by the CP world size maps it to
            # this rank's cumulative stored-frame count. RoPE and first-frame
            # padding below keep the global frame_offset.
            if self._cp_mesh is None:
                cache_offset = frame_offset
            else:
                cache_offset = (frame_offset - self._cp_rank * post_patch_num_frames) // self._cp_world_size
            # Snapshot once per pass: checkpointed recomputes must read the
            # slice this forward read (see RolloutKvCache).
            kv_history_chunks = kv_cache.num_cached_chunks
            kv_cache.assert_pass_starts_at(cache_offset)
            # The chunk's final denoise call stages its self-attention K/V
            # here; committed after the block loop (see
            # RolloutKvCache.store_gen_chunk).
            kv_cache_stage = _KvCacheStage({}) if kv_cache_store_chunk else None
        else:
            assert not kv_cache_store_chunk, "kv_cache_store_chunk requires kv_cache"
            kv_history_chunks = 0
            kv_cache_stage = None
        # Text/neighbor cross-attention slots are consulted only when
        # gradients are off: their projections train, so grad-enabled passes
        # must recompute them fresh.
        context_kv_cache = kv_cache if not torch.is_grad_enabled() else None

        rotary_emb = self.rope(hidden_states, frame_offset)

        hidden_states = self.patch_embedding(hidden_states)

        opacity_extra_patches = pack_opacity_tokens(
            opacity,
            num_latent_frames=post_patch_num_frames,
            latent_height=post_patch_height * p_h,
            latent_width=post_patch_width * p_w,
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            patch_height=p_h,
            patch_width=p_w,
            pad_first_frame=frame_offset == 0,
        )
        opacity_extra_patches = opacity_extra_patches.flatten(2).transpose(1, 2)

        if camera_rays.shape[1] != hidden_states.shape[2]:
            # The per-block camera_embedding Linear is sized for raymaps at
            # latent temporal resolution (channel dim c * h8 * w8).
            raise ValueError(
                "`camera_rays` must be at latent temporal resolution (one raymap per latent frame); "
                f"got {camera_rays.shape[1]} raymaps for {hidden_states.shape[2]} latent frames."
            )
        camera_extra_patches = rearrange(
            camera_rays,
            "b t (h h8) (w w8) c -> b (c h8 w8) t h w",
            h8=self.vae_scale_factor_spatial * self.patch_size[1],
            w8=self.vae_scale_factor_spatial * self.patch_size[2],
        )
        camera_extra_patches = camera_extra_patches.flatten(2).transpose(1, 2)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # PRoPE transforms are rebuilt on every forward and passed to the
        # blocks as args, never stored on the shared modules: gradient
        # checkpointing re-runs blocks at backward time, and by then another
        # rollout chunk may have run with different cameras.
        prope_target_fns: PropeApplyFns | None = None
        prope_reference_fns: PropeApplyFns | None = None

        # Once every block's neighbor K/V slot is filled, the cached
        # post-PRoPE K/V are used directly, so the reference views do not
        # need re-embedding and their PRoPE transforms are not rebuilt.
        # neighbor_hidden_states then only signals that the neighbor
        # cross-attention branch is active.
        neighbor_caches_warm = context_kv_cache is not None and all(
            context_kv_cache.has_neighbor_kv(layer_index) for layer_index in range(len(self.blocks))
        )

        if neighbor_hidden_states is not None:
            if self.prope_target.patches_x != post_patch_width or self.prope_target.patches_y != post_patch_height:
                self.prope_target.update_coeffs(post_patch_width, post_patch_height, hidden_states.device)

            if not neighbor_caches_warm:
                if (
                    self.prope_reference.patches_x != post_patch_width
                    or self.prope_reference.patches_y != post_patch_height
                ):
                    self.prope_reference.update_coeffs(
                        post_patch_width, post_patch_height, neighbor_hidden_states.device
                    )
                prope_reference_fns = self.prope_reference.precompute_apply_fns(neighbor_w2cs, neighbor_Ks)

                neighbor_hidden_states = self.patch_embedding(neighbor_hidden_states)
                neighbor_hidden_states = neighbor_hidden_states.flatten(2).transpose(1, 2)

        frame_seqlen = post_patch_height * post_patch_width
        if timestep.ndim == 2:
            assert (
                timestep.shape[1] % frame_seqlen == 0
            ), f"Expanded timestep length {timestep.shape[1]} must be divisible by frame_seqlen={frame_seqlen}"
            timestep = timestep[:, ::frame_seqlen].contiguous()

        # timestep shape: (batch_size,), flat (batch_size * num_blocks,) for
        # diffusion forcing, or (batch_size, frame_count) for Wan2.2 TI2V.
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * frame_count
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep, encoder_hidden_states, None, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        block_size = attention_config.flex_block_size()
        cp_active = self._cp_mesh is not None
        if cp_active:
            assert (
                local_cp_input
            ), "context-parallel transformer inputs must be local frame shards (outputs also stay local)"
        if self.frames_per_block is None:
            block_mask = None
        elif cp_active:
            # Build the global block-causal mask below, after validating
            # the local frame shard and Ulysses attention-head layout.
            block_mask = None
        else:
            mask_key = (
                "block_causal",
                post_patch_num_frames,
                frame_seqlen,
                self.frames_per_block,
                block_size,
            )
            if mask_key not in self.block_masks:
                self.block_masks[mask_key] = frame_block_causal_mask(
                    device=hidden_states.device,
                    num_frames=post_patch_num_frames,
                    frame_seqlen=frame_seqlen,
                    frames_per_block=self.frames_per_block,
                    block_size=block_size,
                )
            block_mask = self.block_masks[mask_key]

        # Context parallelism: callers already sliced the sequence at frame boundaries.
        # Self-attention uses Ulysses-style all-to-all over heads/sequence.
        if cp_active:
            frames_per_rank = post_patch_num_frames
            total_frames = frames_per_rank * self._cp_world_size

            assert self.frames_per_block is None or frames_per_rank % self.frames_per_block == 0, (
                f"CP shard ({frames_per_rank} frames/rank) must be divisible by "
                f"frames_per_block ({self.frames_per_block}) for block-causal CP."
            )

            hidden_states = hidden_states.contiguous()

            if self.frames_per_block is not None:
                local_attention_heads = self.blocks[0].attn1.heads // self._cp_world_size
                cp_cache_key = (
                    "block_causal_cp_ulysses",
                    total_frames,
                    frame_seqlen,
                    self.frames_per_block,
                    local_attention_heads,
                    block_size,
                )
                if cp_cache_key not in self.block_masks:
                    self.block_masks[cp_cache_key] = frame_block_causal_mask(
                        device=hidden_states.device,
                        num_frames=total_frames,
                        frame_seqlen=frame_seqlen,
                        frames_per_block=self.frames_per_block,
                        block_size=block_size,
                        attention_heads=local_attention_heads,
                    )
                block_mask = self.block_masks[cp_cache_key]

            opacity_extra_patches = opacity_extra_patches.contiguous()
            camera_extra_patches = camera_extra_patches.contiguous()

            # PRoPE target: build transforms for this rank's camera/frame range
            # only (the reference side was built above with the full neighbor set).
            if neighbor_hidden_states is not None:
                prope_target_fns = self.prope_target.precompute_apply_fns(w2cs, Ks)
        else:
            # Without CP, build PRoPE target transforms for all cameras.
            if neighbor_hidden_states is not None:
                prope_target_fns = self.prope_target.precompute_apply_fns(w2cs, Ks)

        # 4. Transformer blocks
        use_checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing

        for i, block in enumerate(self.blocks):
            args = (
                hidden_states,
                encoder_hidden_states,
                neighbor_hidden_states,
                ignore_neighbors,
                timestep_proj,
                rotary_emb,
                opacity_extra_patches,
                camera_extra_patches,
                kv_cache,
                i,
                kv_history_chunks,
                kv_cache_stage,
                context_kv_cache,
                frame_seqlen,
                prope_target_fns,
                prope_reference_fns,
                block_mask,
            )

            if use_checkpointing and i % self.checkpoint_every_n_blocks == 0:
                hidden_states = torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)
            else:
                hidden_states = block(*args)

        if kv_cache_stage is not None:
            kv_cache.commit_staged_chunk(kv_cache_stage.layers, cache_offset, post_patch_num_frames)

        output_post_patch_num_frames = post_patch_num_frames
        if cp_active:
            output_post_patch_num_frames = frames_per_rank

        # 5. Output norm, projection & unpatchify
        output_temb = temb.to(hidden_states.device)
        output_table = self.scale_shift_table.to(hidden_states.device)
        shift = _wan_output_frame_modulation_value(
            output_table,
            output_temb,
            0,
            batch_size=batch_size,
            hidden_seq_len=hidden_states.shape[1],
            frame_seqlen=frame_seqlen,
        )
        scale = _wan_output_frame_modulation_value(
            output_table,
            output_temb,
            1,
            batch_size=batch_size,
            hidden_seq_len=hidden_states.shape[1],
            frame_seqlen=frame_seqlen,
        )
        output_frames = hidden_states.shape[1] // frame_seqlen
        hidden_dtype = hidden_states.dtype
        hidden_states = self.norm_out(hidden_states.float()).unflatten(1, (output_frames, frame_seqlen))
        hidden_states = (hidden_states * (1 + scale) + shift).to(dtype=hidden_dtype)
        hidden_states = self.proj_out(hidden_states).flatten(1, 2)

        hidden_states = hidden_states.reshape(
            batch_size,
            output_post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def enable_context_parallel(self, mesh: torch.distributed.DeviceMesh) -> None:
        """Enable frame-sharded context parallelism.

        The caller supplies a local frame shard and ``frame_offset``. Each rank
        processes that local shard. Self-attention uses Ulysses-style
        all-to-all to gather sequence tokens while sharding attention heads.

        Compatible with block-causal autoregressive attention when
        frames_per_rank is divisible by frames_per_block.  The block mask is
        built once for the gathered global sequence and shared by all ranks.
        """
        set_context_parallel_state(self, mesh)
        for block in self.blocks:
            block.attn1.processor.set_context_parallel_mesh(mesh)

    def disable_context_parallel(self) -> None:
        """Disable context parallelism."""
        clear_context_parallel_state(self)
        for block in self.blocks:
            block.attn1.processor.set_context_parallel_mesh(None)


class ArtifixerTransformerBlock(nn.Module):

    def __init__(
        self,
        base_block: WanTransformerBlock,
        opacity_embedding_dim: int,
        camera_embedding_dim: int,
        use_kv_cache_attention: bool,
    ):
        super().__init__()

        if opacity_embedding_dim > 0:
            self.opacity_embedding = nn.Linear(
                opacity_embedding_dim,
                base_block.attn1.inner_dim,
                dtype=base_block.attn1.to_q.weight.dtype,
                device=base_block.attn1.to_q.weight.device,
                bias=True,
            )
            self.opacity_embedding.weight.data.zero_()
            self.opacity_embedding.bias.data.zero_()
        else:
            self.opacity_embedding = None

        if camera_embedding_dim > 0:
            self.camera_embedding = nn.Linear(
                camera_embedding_dim,
                base_block.attn1.inner_dim,
                dtype=base_block.attn1.to_q.weight.dtype,
                device=base_block.attn1.to_q.weight.device,
                bias=True,
            )
            self.camera_embedding.weight.data.zero_()
            self.camera_embedding.bias.data.zero_()
        else:
            self.camera_embedding = None

        # 1. Self-attention
        self.norm1 = base_block.norm1
        self.attn1 = base_block.attn1

        attn1_backend = base_block.attn1.processor._attention_backend
        if use_kv_cache_attention:
            self.attn1.processor = KvCacheWanSelfAttnProcessor()
        else:
            self.attn1.processor = ArtifixerSelfAttnProcessor()
        self.attn1.processor._attention_backend = attn1_backend

        # 2. Cross-attention
        self.attn2 = base_block.attn2
        self.norm2 = base_block.norm2

        # 3. Add projections for neighbor cross-attention
        self.attn2.add_k_proj = nn.Linear(
            self.attn2.inner_dim,
            self.attn2.inner_dim,
            bias=True,
            dtype=self.attn2.to_k.weight.dtype,
            device=self.attn2.to_k.weight.device,
        )

        # Zero-initialize to not affect the pretrained model initialization
        self.attn2.add_v_proj = nn.Linear(
            self.attn2.inner_dim,
            self.attn2.inner_dim,
            bias=True,
            dtype=self.attn2.to_v.weight.dtype,
            device=self.attn2.to_v.weight.device,
        )
        self.attn2.add_v_proj.weight.data.zero_()
        self.attn2.add_v_proj.bias.data.zero_()
        self.attn2.norm_added_k = nn.RMSNorm(
            self.attn2.norm_k.normalized_shape,
            eps=self.attn2.norm_k.eps,
            elementwise_affine=True,
            dtype=self.attn2.norm_k.weight.dtype,
            device=self.attn2.norm_k.weight.device,
        )

        attn2_backend = base_block.attn2.processor._attention_backend
        self.attn2.processor = ArtifixerCrossAttnProcessor()
        self.attn2.processor._attention_backend = attn2_backend

        # 4. Feed-forward
        self.ffn = base_block.ffn
        self.norm3 = base_block.norm3
        self.scale_shift_table = base_block.scale_shift_table

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        neighbor_hidden_states: torch.Tensor | None,
        ignore_neighbors: bool,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        opacity_extra_patches: torch.Tensor,
        camera_extra_patches: torch.Tensor,
        kv_cache: RolloutKvCache | None,
        layer_index: int,
        kv_cache_num_history_chunks: int,
        kv_cache_stage: _KvCacheStage | None,
        context_kv_cache: RolloutKvCache | None,
        frame_seqlen: int,
        prope_target: PropeApplyFns | None,
        prope_reference: PropeApplyFns | None,
        block_mask: BlockMask | None,
    ) -> torch.Tensor:
        def frame_modulation(index: int) -> torch.Tensor:
            return _wan_block_frame_modulation_value(
                self.scale_shift_table,
                temb,
                index,
                batch_size=hidden_states.shape[0],
                hidden_seq_len=hidden_states.shape[1],
                frame_seqlen=frame_seqlen,
            )

        # 1. Self-attention
        shift_msa = frame_modulation(0)
        scale_msa = frame_modulation(1)
        norm_hidden_states = _modulated_layer_norm(
            self.norm1, hidden_states, scale_msa, shift_msa, frame_seqlen=frame_seqlen
        )
        if self.opacity_embedding is not None:
            norm_hidden_states = norm_hidden_states + self.opacity_embedding(opacity_extra_patches)
        if self.camera_embedding is not None:
            norm_hidden_states = norm_hidden_states + self.camera_embedding(camera_extra_patches)

        if kv_cache is not None:
            attn_output = self.attn1(
                norm_hidden_states,
                None,
                None,
                rotary_emb,
                kv_cache=kv_cache,
                layer_index=layer_index,
                num_history_chunks=kv_cache_num_history_chunks,
                kv_cache_stage=kv_cache_stage.layers if kv_cache_stage is not None else None,
            )
        else:
            attn_output = self.attn1(
                norm_hidden_states,
                None,
                None,
                rotary_emb,
                block_mask=block_mask,
            )
        gate_msa = frame_modulation(2)
        hidden_states = _apply_frame_gate(hidden_states, attn_output, gate_msa, frame_seqlen=frame_seqlen)
        del attn_output, gate_msa

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)

        if neighbor_hidden_states is not None:
            neighbor_hidden_states = neighbor_hidden_states.type_as(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            None,
            None,
            neighbor_hidden_states=neighbor_hidden_states,
            ignore_neighbors=ignore_neighbors,
            context_kv_cache=context_kv_cache,
            layer_index=layer_index,
            prope_target=prope_target,
            prope_reference=prope_reference,
        )
        hidden_states = hidden_states + attn_output

        # 4. Feed-forward
        c_shift_msa = frame_modulation(3)
        c_scale_msa = frame_modulation(4)
        norm_hidden_states = _modulated_layer_norm(
            self.norm3, hidden_states, c_scale_msa, c_shift_msa, frame_seqlen=frame_seqlen
        )
        del c_shift_msa, c_scale_msa
        ff_output = self.ffn(norm_hidden_states)
        c_gate_msa = frame_modulation(5)
        hidden_states = _apply_frame_gate(hidden_states, ff_output, c_gate_msa, frame_seqlen=frame_seqlen)
        del ff_output, c_gate_msa

        return hidden_states


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _qkv_with_rotary(
    attn: WanAttention,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV projections, Q/K norms, head split, and RoPE, shared by the self-attention processors."""
    query, key, value = _get_qkv_projections(attn, hidden_states, None)

    query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
    key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    if rotary_emb is not None:
        query = _apply_rotary_emb(query, *rotary_emb)
        key = _apply_rotary_emb(key, *rotary_emb)

    return query, key, value


def _project_attn_out(attn: WanAttention, hidden_states: torch.Tensor) -> torch.Tensor:
    """Merge heads and apply the output projection."""
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = attn.to_out[0](hidden_states)
    return attn.to_out[1](hidden_states)


class ArtifixerCrossAttnProcessor:
    _attention_backend = None

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        neighbor_hidden_states: torch.Tensor | None = None,
        ignore_neighbors: bool = False,
        context_kv_cache: RolloutKvCache | None = None,
        layer_index: int = 0,
        prope_target: PropeApplyFns | None = None,
        prope_reference: PropeApplyFns | None = None,
    ) -> torch.Tensor:
        if context_kv_cache is not None and context_kv_cache.has_text_kv(layer_index):
            query = attn.norm_q(attn.to_q(hidden_states))
            key, value = context_kv_cache.read_text_kv(layer_index)
        else:
            query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if context_kv_cache is not None:
                context_kv_cache.store_text_kv(layer_index, key, value)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Add neighbor cross-attention
        hidden_states_neighbor = None
        if neighbor_hidden_states is not None:
            query_dtype = query.dtype
            if context_kv_cache is not None and context_kv_cache.has_neighbor_kv(layer_index):
                # Cached K/V already carry the reference-camera PRoPE transform,
                # which depends only on the (fixed) reference views.
                key_neighbor, value_neighbor = context_kv_cache.read_neighbor_kv(layer_index)
            else:
                key_neighbor, value_neighbor = _get_added_kv_projections(attn, neighbor_hidden_states)
                key_neighbor = attn.norm_added_k(key_neighbor)

                key_neighbor = key_neighbor.unflatten(2, (attn.heads, -1))
                value_neighbor = value_neighbor.unflatten(2, (attn.heads, -1))

                key_neighbor = (
                    prope_reference.apply_to_kv(key_neighbor.transpose(2, 1).float()).transpose(2, 1).to(query_dtype)
                )
                value_neighbor = (
                    prope_reference.apply_to_kv(value_neighbor.transpose(2, 1).float()).transpose(2, 1).to(query_dtype)
                )

                if context_kv_cache is not None:
                    context_kv_cache.store_neighbor_kv(layer_index, key_neighbor, value_neighbor)

            query_neighbor = prope_target.apply_to_q(query.transpose(2, 1).float()).transpose(2, 1).to(query_dtype)

            hidden_states_neighbor = dispatch_attention_fn(
                query_neighbor,
                key_neighbor,
                value_neighbor,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )

            hidden_states_neighbor = (
                prope_target.apply_to_o(hidden_states_neighbor.transpose(2, 1).float()).transpose(2, 1).to(query_dtype)
            )

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        if hidden_states_neighbor is not None:
            hidden_states = hidden_states + hidden_states_neighbor * (0 if ignore_neighbors else 1)

        return _project_attn_out(attn, hidden_states)


class ArtifixerSelfAttnProcessor:
    _attention_backend = None
    _cp_mesh: torch.distributed.DeviceMesh | None = None

    def set_context_parallel_mesh(self, mesh: torch.distributed.DeviceMesh | None) -> None:
        self._cp_mesh = mesh

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        block_mask: BlockMask | None,
    ) -> torch.Tensor:
        query, key, value = _qkv_with_rotary(attn, hidden_states, rotary_emb)

        cp_group = None
        if self._cp_mesh is not None:
            cp_group = self._cp_mesh.get_group()
            assert (
                attn.heads % self._cp_mesh.size() == 0
            ), f"self-attention heads ({attn.heads}) must be divisible by CP size ({self._cp_mesh.size()})"
            query = all_to_all_with_grad(query, scatter_dim=2, gather_dim=1, group=cp_group)
            key = all_to_all_with_grad(key, scatter_dim=2, gather_dim=1, group=cp_group)
            value = all_to_all_with_grad(value, scatter_dim=2, gather_dim=1, group=cp_group)

        if block_mask is not None:
            # Block-causal attention; under CP the mask covers the gathered
            # global sequence and is used as-is on every rank.
            hidden_states = flex_attention(
                query=query.transpose(2, 1),
                key=key.transpose(2, 1),
                value=value.transpose(2, 1),
                block_mask=block_mask,
            ).transpose(2, 1)
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask if self._cp_mesh is None else None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )

        if cp_group is not None:
            hidden_states = all_to_all_with_grad(hidden_states, scatter_dim=1, gather_dim=2, group=cp_group)

        return _project_attn_out(attn, hidden_states)


class KvCacheWanSelfAttnProcessor:
    _attention_backend = None
    _cp_mesh: torch.distributed.DeviceMesh | None = None

    def set_context_parallel_mesh(self, mesh: torch.distributed.DeviceMesh | None) -> None:
        self._cp_mesh = mesh

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        kv_cache: RolloutKvCache,
        layer_index: int,
        num_history_chunks: int,
        kv_cache_stage: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> torch.Tensor:
        query, key, value = _qkv_with_rotary(attn, hidden_states, rotary_emb)

        if kv_cache_stage is not None:
            kv_cache_stage[layer_index] = (key, value)

        # Key order is load-bearing: cached chunks in temporal order, then
        # the own-pass rows — the attention reduction order over keys is
        # fixed by that layout. Under CP each rank stores and reads its own
        # frame shard, and the gather below concatenates the per-rank
        # [history + own] sequences in rank order.
        k_history, v_history = kv_cache.read_gen_history(layer_index, num_history_chunks)
        cached_k = torch.cat([*k_history, key], dim=1)
        cached_v = torch.cat([*v_history, value], dim=1)
        if self._cp_mesh is not None:
            # gather_kv_cat owns the grad-mode dispatch between the two
            # collectives (DMD rollouts backprop through these gathered K/V).
            cached_k = gather_kv_cat(cached_k, self._cp_mesh, dim=1)
            cached_v = gather_kv_cat(cached_v, self._cp_mesh, dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            cached_k,
            cached_v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        return _project_attn_out(attn, hidden_states)


class OffsetableWanRotaryPosEmbed(WanRotaryPosEmbed):

    def forward(self, hidden_states: torch.Tensor, frame_offset: int) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][frame_offset : frame_offset + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][frame_offset : frame_offset + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin
