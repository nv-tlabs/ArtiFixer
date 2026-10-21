# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import get_3d_mrope_ids_text_tokens
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_mask
from torch.utils.checkpoint import checkpoint

from model_training.common.context_parallel import all_gather_cat_with_grad, all_to_all_with_grad, gather_kv_cat
from model_training.model_profiles import COSMOS3_ROPE_FPS
from model_training.net import attention_config
from model_training.net.block_mask import create_prefix_block_causal_block_mask
from model_training.net.conditioning import opacity_embedding_input_dim, pack_opacity_tokens
from model_training.net.context_parallel import (
    all_gather_cat,
    clear_context_parallel_state,
    cp_frame_slice,
    cp_padded_frame_count,
    set_context_parallel_state,
)
from model_training.net.rng import rank_invariant_init
from model_training.net.rollout_kv_cache import RolloutKvCache
from model_training.net.scope import ScopeQKInjection, full_tensor, token_plucker_rays
from model_training.pipeline.pipeline_base import validate_positive_frame_count
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT

# Same GPU-arch kernel options as the Wan family's flex_attention, but
# compiled with automatic dynamic shapes: the packed [prompt ⊕ vision] length
# varies with every trimmed prompt, and a fixed-shape compile (Wan's
# dynamic=False) would recompile per prompt length until dynamo's recompile
# limit forced eager execution.
flex_attention = torch.compile(attention_config.flex_attention_with_options)


def conditioning_adapter_weight_norms(transformer: nn.Module) -> dict[str, float]:
    """Report how large each layer's opacity weights are, or {} if there are none.

    All workers must call this together when weights are split between them.
    """
    adapters = getattr(transformer, "opacity_embeddings", None)
    if adapters is None:
        return {}
    logs: dict[str, float] = {}
    for index, adapter in enumerate(adapters):
        data = adapter.weight.data if hasattr(adapter, "weight") else adapter.data
        logs[f"opacity_embeddings/weight_norm_{index:02d}"] = full_tensor(data).detach().float().norm().item()
    return logs


def scope_to_query_ratios(samples: list[tuple[float, float]]) -> dict[str, float]:
    """Compare the camera's change to each query with the original query size.

    Weight sizes alone do not show the effect after normalization. Return {} if
    no usable measurements were collected.
    """
    if not samples:
        return {}
    ratios = [delta / query for delta, query in samples if query > 0]
    if not ratios:
        return {}
    return {
        "scope_to_query/mean": sum(ratios) / len(ratios),
        "scope_to_query/max": max(ratios),
    }


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    *,
    reset_spatial_indices: bool,
    fps: float | None,
    base_fps: float,
    temporal_compression_factor: int,
    fps_reference_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    """Cosmos3 RoPE positions, based on diffusers 0.39.0.

    Unlike upstream, FPS scaling also applies to single-frame shards so their
    absolute positions match the unsharded clip.
    """
    fps_modulation_enabled = fps is not None
    # Converts this token stream's FPS to tokens/sec (vision uses the VAE
    # temporal compression factor).
    tokens_per_second = fps / temporal_compression_factor if fps_modulation_enabled else None
    # Converts the base FPS to the reference tokens/sec for mRoPE scaling.
    # ``fps_reference_temporal_compression_factor`` maps to upstream's
    # ``base_temporal_compression_factor`` (see the docstring); byte-parity
    # with the pinned fork requires keeping it.
    if fps_reference_temporal_compression_factor is None:
        fps_reference_temporal_compression_factor = temporal_compression_factor
    reference_tokens_per_second = (
        base_fps / fps_reference_temporal_compression_factor if fps_modulation_enabled else None
    )
    if fps_modulation_enabled:
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (
            frame_indices + start_frame_offset
        ) / tokens_per_second * reference_tokens_per_second + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    if not reset_spatial_indices:
        h_index = h_index + int(temporal_offset)
        w_index = w_index + int(temporal_offset)

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)
    return mrope_ids, math.ceil(mrope_ids.max().item()) + 1


def freeze_cosmos3_reasoner_parameters(transformer: nn.Module) -> None:
    transformer.requires_grad_(False)
    transformer.proj_in.requires_grad_(True)
    transformer.proj_out.requires_grad_(True)
    transformer.time_embedder.requires_grad_(True)
    transformer.norm_moe_gen.requires_grad_(True)

    for layer in transformer.layers:
        layer.input_layernorm_moe_gen.requires_grad_(True)
        layer.post_attention_layernorm_moe_gen.requires_grad_(True)
        layer.mlp_moe_gen.requires_grad_(True)

        attn = layer.self_attn
        attn.add_q_proj.requires_grad_(True)
        attn.add_k_proj.requires_grad_(True)
        attn.add_v_proj.requires_grad_(True)
        attn.to_add_out.requires_grad_(True)
        attn.norm_added_q.requires_grad_(True)
        attn.norm_added_k.requires_grad_(True)


class _Fp32TimestepEmbedding(nn.Module):
    """Keep timestep calculations in fp32, including with bf16 weights or autocast.

    Reuse the original child modules so pretrained state-dict keys still match.
    """

    def __init__(self, time_embedder: nn.Module):
        super().__init__()
        assert isinstance(getattr(time_embedder, "linear_1", None), nn.Linear) and isinstance(
            getattr(time_embedder, "linear_2", None), nn.Linear
        ), f"{type(time_embedder).__name__} must expose TimestepEmbedding's linear_1/linear_2"
        assert getattr(time_embedder, "cond_proj", None) is None, "conditioned TimestepEmbedding is unsupported"
        assert getattr(time_embedder, "post_act", None) is None, "post-activation TimestepEmbedding is unsupported"
        self.linear_1 = time_embedder.linear_1
        self.act = time_embedder.act
        self.linear_2 = time_embedder.linear_2

    @staticmethod
    def _fp32_linear(sample: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        bias = linear.bias.float() if linear.bias is not None else None
        return F.linear(sample, linear.weight.float(), bias)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=sample.device.type, enabled=False):
            sample = self._fp32_linear(sample.float(), self.linear_1)
            sample = self.act(sample)
            return self._fp32_linear(sample, self.linear_2)


@dataclass
class _LayerConditioningContext:
    """Per-call inputs restored during gradient-checkpoint recomputation."""

    opacity_tokens: torch.Tensor
    gen_block_mask: BlockMask | None = None
    kv_cache: RolloutKvCache | None = None
    kv_cache_stage: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    kv_cache_num_history_chunks: int = 0
    kv_cache_num_vision_tokens: int = 0
    scope_rays: torch.Tensor | None = None
    cp_pad_tokens: int = 0
    num_reference_tokens: int = 0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class ArtifixerCosmos3AttnProcessor:
    _attention_backend = None
    _parallel_config = None
    _scope = None
    _scope_rays = None
    _scope_accumulator = None
    _cp_mesh: torch.distributed.DeviceMesh | None = None
    _gen_block_mask: BlockMask | None = None
    _kv_cache_context: tuple[RolloutKvCache, int, int, dict | None, int] | None = None
    _cp_pad_tokens: int = 0
    _num_reference_tokens: int = 0

    def _masked_gen_attention(
        self,
        q_gen: torch.Tensor,
        all_k: torch.Tensor,
        all_v: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        assert self._parallel_config is None, (
            "masked Cosmos3 attention runs flex_attention outside dispatch_attention_fn and cannot "
            "honor a diffusers parallel_config"
        )
        query = q_gen.unsqueeze(0).transpose(1, 2)
        key = all_k.unsqueeze(0).transpose(1, 2)
        value = all_v.unsqueeze(0).transpose(1, 2)
        if query.device.type == "cpu":
            # torch ships no CPU flex_attention kernel (no torch.compile
            # lowering, no backward), so CPU (unit-test) forwards materialize
            # the BlockMask's own mask_mod into a boolean mask and run torch
            # SDPA — one mask definition either way.
            attn_mask = create_mask(block_mask.mask_mod, None, None, query.shape[2], key.shape[2], query.device)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, enable_gqa=True)
        else:
            if torch.is_autocast_enabled("cuda"):
                # RoPE/SCoPE keep Q/K in FP32 while V uses the projected dtype.
                # Materialize homogeneous inputs before torch.compile: the
                # no-grad autocast lowering of mixed Q/K/V can disagree with
                # both dense attention and the gradient-enabled flex path.
                query = query.to(value.dtype)
                key = key.to(value.dtype)
            out = flex_attention(query, key, value, block_mask=block_mask, enable_gqa=True)
        return out.transpose(1, 2)

    def __call__(
        self,
        attn: nn.Module,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)

        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        q_und = q_und * cos_und.unsqueeze(1) + _rotate_half(q_und) * sin_und.unsqueeze(1)
        k_und = k_und * cos_und.unsqueeze(1) + _rotate_half(k_und) * sin_und.unsqueeze(1)
        q_gen = q_gen * cos_gen.unsqueeze(1) + _rotate_half(q_gen) * sin_gen.unsqueeze(1)
        k_gen = k_gen * cos_gen.unsqueeze(1) + _rotate_half(k_gen) * sin_gen.unsqueeze(1)

        # Add camera information while each worker still has all heads for its tokens.
        # Save the updated keys so later chunks use the same camera reference.
        if self._scope is not None:
            scoped_q, k_gen = self._scope(q_gen, k_gen, self._scope_rays)
            if self._scope_accumulator is not None:
                self._scope_accumulator.append(
                    ((scoped_q - q_gen).detach().float().norm().item(), q_gen.detach().float().norm().item())
                )
            q_gen = scoped_q

        num_references = self._num_reference_tokens
        if num_references:
            q_ref, q_gen = q_gen[:num_references], q_gen[num_references:]
            k_ref, k_gen = k_gen[:num_references], k_gen[num_references:]
            v_ref, v_gen = v_gen[:num_references], v_gen[num_references:]
            reference_out = dispatch_attention_fn(
                q_ref.unsqueeze(0),
                torch.cat([k_und, k_ref], dim=0).unsqueeze(0),
                torch.cat([v_und, v_ref], dim=0).unsqueeze(0),
                is_causal=False,
                enable_gqa=True,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            ).squeeze(0)

        causal_out = dispatch_attention_fn(
            q_und.unsqueeze(0),
            k_und.unsqueeze(0),
            v_und.unsqueeze(0),
            is_causal=True,
            enable_gqa=True,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        causal_out = causal_out.squeeze(0).flatten(-2, -1)

        kv_cache_context = self._kv_cache_context
        # Trailing pad rows on the GLOBAL gen token axis: the wrapper pads
        # the latent-frame axis so any head-compatible CP size shards evenly.
        # Pad keys are dropped right after the exchange below, so no real
        # query ever attends to one; pad QUERY rows stay in (their outputs
        # are cut by the wrapper after the gather).
        cp_pad_tokens = self._cp_pad_tokens
        if kv_cache_context is not None:
            cache, layer_index, num_history_chunks, kv_cache_stage, num_vision_rows = kv_cache_context
            # Cache passes carry only the current chunk: no earlier frames
            # in-pass, so no mask — causality is enforced by cache content,
            # and the mask-free dispatch below stays flash-eligible.
            assert (
                self._gen_block_mask is None
            ), "KV-cache passes take no attention mask — causality is enforced by cache content."
            assert k_gen.shape[0] == num_vision_rows, (
                f"KV-cache pass gen row count {k_gen.shape[0]} does not match this rank's "
                f"vision-token count {num_vision_rows}"
            )
        # Padded chunks need head-sharded caches: removing pads from frame shards
        # would leave unequal row counts for the history gather. Even chunks keep frame shards.
        exchange_heads = self._cp_mesh is not None and (kv_cache_context is None or cp_pad_tokens > 0)
        cp_gen_group = None
        if exchange_heads:
            # Ulysses exchange: each rank trades its frame shard (all heads)
            # for the FULL gen sequence — the clip on non-cache forwards,
            # the chunk on cache passes — with a contiguous 1/CP chunk of
            # the heads. RoPE already ran above with global mRoPE positions,
            # so exchanged rows carry their absolute positions. Cosmos3 has
            # real GQA, so BOTH head counts must split evenly across the CP
            # group; the query/KV ratio is unchanged by the split, so
            # enable_gqa below applies as-is.
            cp_size = self._cp_mesh.size()
            cp_rank = self._cp_mesh.get_local_rank()
            cp_gen_group = self._cp_mesh.get_group()
            assert attn.num_attention_heads % cp_size == 0, (
                f"Cosmos3 Ulysses CP shards attention heads: num_attention_heads "
                f"({attn.num_attention_heads}) must be divisible by the CP size ({cp_size})"
            )
            assert attn.num_key_value_heads % cp_size == 0, (
                f"Cosmos3 Ulysses CP shards key/value heads too (GQA): num_key_value_heads "
                f"({attn.num_key_value_heads}) must be divisible by the CP size ({cp_size})"
            )
            q_gen = all_to_all_with_grad(q_gen, scatter_dim=1, gather_dim=0, group=cp_gen_group)
            k_gen = all_to_all_with_grad(k_gen, scatter_dim=1, gather_dim=0, group=cp_gen_group)
            v_gen = all_to_all_with_grad(v_gen, scatter_dim=1, gather_dim=0, group=cp_gen_group)
            if cp_pad_tokens:
                # Pad rows sit at the global tail (the wrapper pads the last
                # frames); dropping them here hands the attention below
                # exactly the CP=1 key set and keeps them out of the KV
                # cache. Pad QUERY rows stay in q_gen so the reverse
                # exchange stays even-sized — their outputs are stripped
                # after the wrapper's gather.
                k_gen = k_gen[:-cp_pad_tokens]
                v_gen = v_gen[:-cp_pad_tokens]
            # The replicated text K/V join the gen keys with the SAME
            # contiguous rank-order head chunking the all_to_all uses
            # (tensor.chunk on the head dim); any other slicing would pair
            # text heads with the wrong gen heads. The und causal attention
            # above still runs with full heads.
            k_und_kv = k_und.chunk(cp_size, dim=1)[cp_rank]
            v_und_kv = v_und.chunk(cp_size, dim=1)[cp_rank]
        else:
            assert cp_pad_tokens == 0, (
                "CP pad rows exist only on head-exchange paths: the wrapper pads frames only when it "
                "shards them across an active CP mesh"
            )
            k_und_kv = k_und
            v_und_kv = v_und
        reference_keys, reference_values = (), ()
        if num_references:
            if exchange_heads:
                k_ref = k_ref.chunk(cp_size, dim=1)[cp_rank]
                v_ref = v_ref.chunk(cp_size, dim=1)[cp_rank]
            reference_keys, reference_values = (k_ref,), (v_ref,)
        if kv_cache_context is not None:
            # Frozen text weights and a fixed prompt make detached text K/V reusable.
            if not cache.has_text_kv(layer_index):
                cache.store_text_kv(layer_index, k_und_kv, v_und_kv)
            k_und_cached, v_und_cached = cache.read_text_kv(layer_index)
            assert k_und_cached.shape == k_und_kv.shape, (
                f"prompt changed mid-rollout: cached text K/V cover {k_und_cached.shape[0]} tokens, "
                f"this pass carries {k_und_kv.shape[0]}"
            )
            if kv_cache_stage is not None:
                # Store only this rank's clean target K/V, excluding references and padding.
                kv_cache_stage[layer_index] = (k_gen, v_gen)
            k_vision_history, v_vision_history = cache.read_gen_history(layer_index, num_history_chunks)
            gen_k = torch.cat([*k_vision_history, k_gen], dim=0)
            gen_v = torch.cat([*v_vision_history, v_gen], dim=0)
            if self._cp_mesh is not None and not exchange_heads:
                # Gather both tensors in one collective. Key order does not affect attention.
                gathered_kv = gather_kv_cat(torch.stack([gen_k, gen_v]), self._cp_mesh, dim=1)
                gen_k, gen_v = gathered_kv.unbind(0)
            # Replicated text and references are added once, outside the target gather.
            all_k = torch.cat([k_und_cached, *reference_keys, gen_k], dim=0)
            all_v = torch.cat([v_und_cached, *reference_values, gen_v], dim=0)
        else:
            all_k = torch.cat([k_und_kv, *reference_keys, k_gen], dim=0)
            all_v = torch.cat([v_und_kv, *reference_values, v_gen], dim=0)
        block_mask = self._gen_block_mask
        if block_mask is None:
            full_out = dispatch_attention_fn(
                q_gen.unsqueeze(0),
                all_k.unsqueeze(0),
                all_v.unsqueeze(0),
                is_causal=False,
                enable_gqa=True,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            # Every head shard uses the same global target-query/key mask.
            assert block_mask.shape[-2:] == (q_gen.shape[0], all_k.shape[0]), (
                f"block-causal BlockMask geometry {tuple(block_mask.shape[-2:])} does not match "
                f"[S_gen, S_und + S_gen] = {(q_gen.shape[0], all_k.shape[0])}"
            )
            full_out = self._masked_gen_attention(q_gen, all_k, all_v, block_mask)
        full_out = full_out.squeeze(0)
        if cp_gen_group is not None:
            # Reverse Ulysses exchange: trade the global rows / head shard
            # back for this rank's frame rows with all heads, so to_add_out
            # and the residual stream stay rank-local. The row scatter is
            # always even: cp_frame_slice enforces equal frame shards.
            full_out = all_to_all_with_grad(full_out, scatter_dim=0, gather_dim=1, group=cp_gen_group)
        if num_references:
            full_out = torch.cat([reference_out, full_out], dim=0)
        full_out = full_out.flatten(-2, -1)

        und_out = attn.to_out(causal_out)
        gen_out = attn.to_add_out(full_out)
        return und_out, gen_out


class ArtifixerCosmos3Transformer(nn.Module):
    """Cosmos3 with shared reference/target self-attention, camera rays, and opacity.

    Text stays frozen. References attend to text and other references, never
    target frames. Target attention follows the training or rollout mask.
    """

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
        base_transformer: nn.Module,
        *,
        vae_scale_factor_spatial: int,
        vae_scale_factor_temporal: int,
        gradient_checkpointing: bool,
        causal_block_frames: int | None = None,
    ):
        super().__init__()
        validate_positive_frame_count("causal_block_frames", causal_block_frames)
        self.causal_block_frames = causal_block_frames
        self.base_transformer = base_transformer
        self.base_transformer.time_embedder = _Fp32TimestepEmbedding(self.base_transformer.time_embedder)
        self.config = base_transformer.config
        self.latent_patch_size = int(self.config.latent_patch_size)
        self.patch_size = (1, self.latent_patch_size, self.latent_patch_size)
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.gradient_checkpointing = gradient_checkpointing
        self._install_attention_processors()

        opacity_embedding_dim = opacity_embedding_input_dim(
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_height=self.latent_patch_size,
            patch_width=self.latent_patch_size,
        )
        self.opacity_embeddings = nn.ModuleList()
        self.scope = nn.ModuleList()
        self._scope_accumulator: list[tuple[float, float]] | None = None
        self.block_masks: dict[tuple, BlockMask] = {}
        self._conditioning_context: _LayerConditioningContext | None = None
        for layer_index, layer in enumerate(self.base_transformer.layers):
            attn = layer.self_attn
            q_proj = attn.add_q_proj
            opacity_embedding = nn.Linear(
                opacity_embedding_dim, q_proj.in_features, dtype=q_proj.weight.dtype, device=q_proj.weight.device
            )
            nn.init.zeros_(opacity_embedding.weight)
            nn.init.zeros_(opacity_embedding.bias)
            self.opacity_embeddings.append(opacity_embedding)
            self.scope.append(
                ScopeQKInjection(
                    num_attention_heads=int(attn.num_attention_heads),
                    num_key_value_heads=int(attn.num_key_value_heads),
                    head_dim=int(attn.head_dim),
                ).to(device=q_proj.weight.device)
            )
            attn.register_forward_pre_hook(self._make_conditioning_pre_hook(layer_index), with_kwargs=True)
            attn.register_forward_hook(self._processor_disarm_hook)

        freeze_cosmos3_reasoner_parameters(self.base_transformer)
        if gradient_checkpointing:
            self.base_transformer.enable_gradient_checkpointing(self._gradient_checkpointing_func)
        else:
            self.base_transformer.disable_gradient_checkpointing()

    def _install_attention_processors(self) -> None:
        for layer in self.base_transformer.layers:
            attn = layer.self_attn
            set_processor = attn.set_processor
            previous_processor = attn.processor
            processor = ArtifixerCosmos3AttnProcessor()
            processor._attention_backend = previous_processor._attention_backend
            processor._parallel_config = previous_processor._parallel_config
            set_processor(processor)

    def _gradient_checkpointing_func(self, module, *args):
        # Save these inputs so rerunning a layer during training uses the same
        # camera, opacity, and reference data.
        conditioning_context = self._conditioning_context

        def run_with_conditioning_context(*inner_args):
            previous_context = self._conditioning_context
            self._conditioning_context = conditioning_context
            try:
                return module(*inner_args)
            finally:
                self._conditioning_context = previous_context

        return checkpoint(run_with_conditioning_context, *args, use_reentrant=False)

    def _make_conditioning_pre_hook(self, layer_index: int):
        # The base model scales video features before self_attn. Add opacity at that point.
        def hook(_module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]):
            processor = _module.processor
            assert isinstance(
                processor, ArtifixerCosmos3AttnProcessor
            ), "Cosmos3 target conditioning requires ArtifixerCosmos3AttnProcessor."
            context = self._conditioning_context
            if context is None:
                processor._gen_block_mask = None
                processor._kv_cache_context = None
                processor._cp_pad_tokens = 0
                processor._num_reference_tokens = 0
                processor._scope = None
                processor._scope_rays = None
                processor._scope_accumulator = None
                return None
            processor._gen_block_mask = context.gen_block_mask
            processor._cp_pad_tokens = context.cp_pad_tokens
            processor._num_reference_tokens = context.num_reference_tokens
            processor._scope = self.scope[layer_index]
            processor._scope_rays = context.scope_rays
            processor._scope_accumulator = self._scope_accumulator
            if context.kv_cache is None:
                processor._kv_cache_context = None
            else:
                processor._kv_cache_context = (
                    context.kv_cache,
                    layer_index,
                    context.kv_cache_num_history_chunks,
                    context.kv_cache_stage,
                    context.kv_cache_num_vision_tokens,
                )
            if len(inputs) >= 2:
                gen_seq = inputs[1]
                from_args = True
            else:
                gen_seq = kwargs.get("gen_seq")
                from_args = False
            assert (
                isinstance(gen_seq, torch.Tensor) and gen_seq.ndim == 2
            ), "Cosmos3 self_attn gen input must be [S,D] for target conditioning"
            opacity_tokens = context.opacity_tokens.to(device=gen_seq.device, dtype=gen_seq.dtype)
            # Add opacity to the video features here; camera rays are added inside attention.
            delta = self.opacity_embeddings[layer_index](opacity_tokens)
            if context.num_reference_tokens:
                delta = F.pad(delta, (0, 0, context.num_reference_tokens, 0))
            # Add opacity to target rows only; clean reference rows stay unchanged.
            assert delta.shape == gen_seq.shape, (
                f"target conditioning delta shape {tuple(delta.shape)} does not match "
                f"gen tokens {tuple(gen_seq.shape)}"
            )
            gen_seq = gen_seq + delta
            if from_args:
                inputs = list(inputs)
                inputs[1] = gen_seq
                return tuple(inputs), kwargs
            return inputs, {**kwargs, "gen_seq": gen_seq}

        return hook

    @staticmethod
    def _processor_disarm_hook(module: nn.Module, _inputs: tuple[Any, ...], output):
        """Release per-call attention state; checkpoint recomputation restores it."""
        processor = module.processor
        if isinstance(processor, ArtifixerCosmos3AttnProcessor):
            processor._gen_block_mask = None
            processor._kv_cache_context = None
            processor._cp_pad_tokens = 0
            processor._num_reference_tokens = 0
        processor._scope = None
        processor._scope_rays = None
        processor._scope_accumulator = None
        return output

    def set_attention_backend(self, attention_backend: str | None) -> None:
        if attention_backend is not None:
            self.base_transformer.set_attention_backend(attention_backend)

    def new_rollout_kv_cache(self, *, truncate_rollout_gradients: bool) -> RolloutKvCache:
        """One AR rollout's write-once K/V cache: one entry set per
        decoder layer; lifetime = one ``generate_samples_from_batch`` call."""
        return RolloutKvCache(
            len(self.base_transformer.layers),
            truncate_rollout_gradients=truncate_rollout_gradients,
        )

    def enable_context_parallel(self, mesh: torch.distributed.DeviceMesh) -> None:
        set_context_parallel_state(self, mesh)
        self._set_attention_processor_cp_mesh(mesh)

    def disable_context_parallel(self) -> None:
        self._set_attention_processor_cp_mesh(None)
        clear_context_parallel_state(self)

    def _set_attention_processor_cp_mesh(self, mesh: torch.distributed.DeviceMesh | None) -> None:
        for layer in self.base_transformer.layers:
            processor = layer.self_attn.processor
            assert isinstance(
                processor, ArtifixerCosmos3AttnProcessor
            ), "Cosmos3 context parallelism requires ArtifixerCosmos3AttnProcessor."
            processor._cp_mesh = mesh

    def _cp_frame_slice(self, total_latent_frames: int) -> tuple[int, int]:
        if self._cp_mesh is None:
            return 0, total_latent_frames
        return cp_frame_slice(total_latent_frames, self._cp_rank, self._cp_world_size)

    def _gather_cp_output(self, output: torch.Tensor) -> torch.Tensor:
        if self._cp_mesh is None:
            return output
        if torch.is_grad_enabled():
            # Grad-enabled CP forwards backprop through the gather: each
            # rank's slice gradient routes back to the rank that produced it.
            return all_gather_cat_with_grad(output, dim=2, group=self._cp_mesh.get_group())
        return all_gather_cat(output, self._cp_mesh, dim=2)

    def _opacity_tokens(self, opacity: torch.Tensor, target: torch.Tensor, start_frame_offset: int = 0) -> torch.Tensor:
        """Pack the full-pixel opacity map into per-token channels, f_o(PixelUnshuffle(O)).

        Validation, pad, and rearrange live in the shared
        ``net/conditioning.py::pack_opacity_tokens`` (also used by the Wan
        wrapper): the [B, T_pixel, H_pixel, W_pixel] opacity is losslessly
        rearranged so that each generation token sees the complete opacity
        footprint of the pixels it decodes instead of a single maxpooled
        latent-resolution value per latent cell. Returns
        [B, opacity_embedding_dim, latent_t, patch_h, patch_w], the same
        grid layout as ``_camera_grid`` so CP slicing on dim 2 stays valid.

        ``start_frame_offset`` (KV-cache rollout chunks, CP shard inputs) marks a
        pass whose first latent frame is absolute frame ``start_frame_offset > 0``: the causal
        VAE's 1+4N contract folds pixel frames ``[4k-3, 4k]`` into latent frame
        ``k >= 1``, so a mid-video chunk of ``T`` latent frames covers exactly
        ``T * vae_scale_factor_temporal`` REAL pixel frames — no first-frame
        replication pad (that pad exists only to fill frame 0's window). The
        resulting per-token channels are bit-identical to the corresponding
        frames of the full-clip packing (pinned by test).
        """
        _, _, latent_t, latent_h, latent_w = target.shape
        assert opacity.ndim == 4, f"`opacity` must be [B, T, H, W], got shape {tuple(opacity.shape)}"
        assert (
            opacity.shape[0] == target.shape[0]
        ), f"`opacity` batch size {opacity.shape[0]} does not match latent batch size {target.shape[0]}"
        opacity = opacity.to(device=target.device, dtype=target.dtype)
        return pack_opacity_tokens(
            opacity,
            num_latent_frames=latent_t,
            latent_height=latent_h,
            latent_width=latent_w,
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            patch_height=self.latent_patch_size,
            patch_width=self.latent_patch_size,
            pad_first_frame=start_frame_offset == 0,
        )

    def _camera_grid(self, camera_rays: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, _, latent_t, latent_h, latent_w = target.shape
        camera = camera_rays.to(device=target.device, dtype=target.dtype)
        expected_shape = (
            batch_size,
            latent_t,
            latent_h * self.vae_scale_factor_spatial,
            latent_w * self.vae_scale_factor_spatial,
            6,
        )
        assert (
            camera.shape == expected_shape
        ), f"camera_rays must have shape {expected_shape}, got {tuple(camera.shape)}"
        patch_size = self.latent_patch_size
        assert (
            latent_h % patch_size == 0 and latent_w % patch_size == 0
        ), f"latent camera grid {(latent_h, latent_w)} must be divisible by patch_size={patch_size}"
        camera_patch_size = self.vae_scale_factor_spatial * patch_size
        patch_h = latent_h // patch_size
        patch_w = latent_w // patch_size
        camera = camera.reshape(
            batch_size,
            latent_t,
            patch_h,
            camera_patch_size,
            patch_w,
            camera_patch_size,
            6,
        )
        return camera.permute(0, 6, 3, 5, 1, 2, 4).reshape(
            batch_size,
            6 * camera_patch_size * camera_patch_size,
            latent_t,
            patch_h,
            patch_w,
        )

    def _target_condition_tokens(
        self,
        opacity: torch.Tensor,
        camera_rays: torch.Tensor,
        target: torch.Tensor,
        cp_start_frame: int,
        cp_end_frame: int,
        start_frame: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        opacity_grid = self._opacity_tokens(opacity, target, start_frame_offset=start_frame)
        assert camera_rays is not None, "camera_rays are required for Cosmos3 target conditioning"
        camera_target = target
        if self._cp_mesh is not None:
            camera_rays = camera_rays[:, cp_start_frame:cp_end_frame].contiguous()
            camera_target = target[:, :, cp_start_frame:cp_end_frame]
        camera_grid = self._camera_grid(camera_rays, camera_target)
        if self._cp_mesh is not None:
            opacity_grid = opacity_grid[:, :, cp_start_frame:cp_end_frame].contiguous()
        return opacity_grid.flatten(2).transpose(1, 2), camera_grid.flatten(2).transpose(1, 2)

    def _patchify_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        bsz, channels, frames, height, width = latents.shape
        latent_channels = int(self.config.latent_channel)
        patch_size = self.latent_patch_size
        assert channels == latent_channels, f"expected {latent_channels} latent channels, got {channels}"
        assert (
            height % patch_size == 0 and width % patch_size == 0
        ), f"latent spatial shape {(height, width)} must be divisible by patch_size={patch_size}"
        patch_h = height // patch_size
        patch_w = width // patch_size
        tokens = latents.reshape(bsz, channels, frames, patch_h, patch_size, patch_w, patch_size)
        # Per-token feature order [patch_y, patch_x, channel]: proj_in is shared
        # with the base transformer's own patchify (transformer_cosmos3.py
        # einsum cthpwq->thwpqc), so reference tokens must use its layout.
        tokens = tokens.permute(0, 2, 3, 5, 4, 6, 1).reshape(
            bsz,
            frames * patch_h * patch_w,
            patch_size * patch_size * channels,
        )
        return tokens, (frames, patch_h, patch_w)

    def _layer_conditioning_context(
        self,
        sample_index: int,
        opacity_tokens: torch.Tensor,
        camera_tokens: torch.Tensor,
        neighbor_hidden_states: torch.Tensor | None,
        vision_segment: dict[str, Any],
        *,
        gen_block_mask: BlockMask | None = None,
        cp_pad_tokens: int = 0,
        kv_cache: RolloutKvCache | None = None,
        kv_cache_stage: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        kv_cache_num_history_chunks: int = 0,
        kv_cache_num_vision_tokens: int = 0,
        neighbor_camera_rays: torch.Tensor | None = None,
    ) -> _LayerConditioningContext:
        scope_rays = token_plucker_rays(camera_tokens[sample_index])
        num_reference_tokens = vision_segment.get("num_reference_tokens", 0)
        if num_reference_tokens:
            reference_grid = self._camera_grid(
                neighbor_camera_rays[sample_index : sample_index + 1],
                neighbor_hidden_states[sample_index : sample_index + 1],
            )
            reference_rays = token_plucker_rays(reference_grid.flatten(2).transpose(1, 2)[0])
            assert reference_rays.shape[0] == num_reference_tokens
            scope_rays = torch.cat([reference_rays, scope_rays], dim=0)
        return _LayerConditioningContext(
            opacity_tokens=opacity_tokens[sample_index],
            scope_rays=scope_rays,
            num_reference_tokens=num_reference_tokens,
            gen_block_mask=gen_block_mask,
            cp_pad_tokens=cp_pad_tokens,
            kv_cache=kv_cache,
            kv_cache_stage=kv_cache_stage,
            kv_cache_num_history_chunks=kv_cache_num_history_chunks,
            kv_cache_num_vision_tokens=kv_cache_num_vision_tokens,
        )

    @staticmethod
    def _as_bool(value: bool | torch.Tensor) -> bool:
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1, f"expected scalar bool tensor, got shape {tuple(value.shape)}"
            return bool(value.item())
        return bool(value)

    def _trim_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        sample_index: int,
    ) -> torch.Tensor:
        sample_ids = input_ids[sample_index]
        if attention_mask is None:
            return sample_ids
        sample_mask = attention_mask[sample_index].to(dtype=torch.bool, device=sample_ids.device)
        return sample_ids[sample_mask]

    def _prepare_text_segment(
        self,
        input_ids: torch.Tensor,
        device: torch.device | str,
    ) -> dict[str, Any]:
        und_len = int(input_ids.numel())
        text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
            und_len,
            temporal_offset=0,
            use_float_positions=bool(self.config.enable_fps_modulation),
        )
        return {
            "input_ids": input_ids.to(device=device, dtype=torch.long),
            "text_indexes": torch.arange(und_len, dtype=torch.long, device=device),
            "und_len": und_len,
            "text_mrope_ids": text_mrope_ids.to(device),
            "vision_start_temporal_offset": next_mrope_offset
            + int(self.config.unified_3d_mrope_temporal_modality_margin),
        }

    def _prepare_vision_segment(
        self,
        input_vision_tokens: torch.Tensor,
        mrope_offset: int | float,
        curr: int,
        device: torch.device | str,
        start_frame_offset: int = 0,
        noisy_frame_indexes: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
        patch_h = math.ceil(latent_h / self.latent_patch_size)
        patch_w = math.ceil(latent_w / self.latent_patch_size)
        num_vision_tokens = latent_t * patch_h * patch_w
        spatial_tokens_per_frame = patch_h * patch_w
        if noisy_frame_indexes is None:
            noisy_frame_indexes = torch.arange(0, latent_t, device=device, dtype=torch.long)
            mse_loss_indexes = torch.arange(curr, curr + num_vision_tokens, dtype=torch.long, device=device)
        else:
            # Conditioning-only reference or KV-cache seed frames have an
            # empty noisy set: no frame is denoised or receives a timestep. The
            # base transformer reads predictions at vision_mse_loss_indexes
            # ordered (frame, spatial) per noisy frame (transformer_cosmos3.py
            # _unpatchify_and_unpack_latents), so the sequence positions are
            # built from the frame list, not a flat arange.
            noisy_frame_indexes = noisy_frame_indexes.to(device=device, dtype=torch.long)
            mse_loss_indexes = (
                curr
                + (noisy_frame_indexes * spatial_tokens_per_frame).unsqueeze(-1)
                + torch.arange(spatial_tokens_per_frame, dtype=torch.long, device=device)
            ).flatten()
        # The MRoPE temporal scale is pinned to the profile constant; see the
        # COSMOS3_ROPE_FPS comment in model_profiles.py. This is the single
        # consumption line; a per-call fps parameter would scatter silent
        # defaults across call sites.
        fps = COSMOS3_ROPE_FPS if bool(self.config.enable_fps_modulation) else None
        vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=mrope_offset,
            reset_spatial_indices=bool(self.config.unified_3d_mrope_reset_spatial_ids),
            fps=fps,
            base_fps=float(self.config.base_fps),
            temporal_compression_factor=self.vae_scale_factor_temporal,
            start_frame_offset=start_frame_offset,
        )
        return {
            "vision_token_shapes": [(latent_t, patch_h, patch_w)],
            "vision_sequence_indexes": torch.arange(curr, curr + num_vision_tokens, dtype=torch.long, device=device),
            "vision_mse_loss_indexes": mse_loss_indexes,
            "vision_noisy_frame_indexes": [noisy_frame_indexes],
            "vision_mrope_ids": vision_mrope_ids.to(device),
            "num_vision_tokens": num_vision_tokens,
            # Scatter-src row count for _vision_timesteps: the base transformer
            # embeds one timestep row per NOISY vision token
            # (transformer_cosmos3.py _apply_timestep_embeds_to_noisy_tokens).
            "num_noisy_vision_tokens": int(noisy_frame_indexes.numel()) * spatial_tokens_per_frame,
        }

    def _prepend_references(
        self,
        target_latents: torch.Tensor,
        reference_latents: torch.Tensor,
        text_segment: dict[str, Any],
        vision_segment: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        assert reference_latents.shape[:2] == target_latents.shape[:2]
        assert (
            reference_latents.shape[3:] == target_latents.shape[3:]
        ), "joint references and target must have matching latent patch grids"
        reference_frames = reference_latents.shape[2]
        assert reference_frames > 0
        target_frames, patch_h, patch_w = vision_segment["vision_token_shapes"][0]
        reference_tokens = reference_frames * patch_h * patch_w
        total_tokens = reference_tokens + vision_segment["num_vision_tokens"]
        reference_segment = self._prepare_vision_segment(
            input_vision_tokens=reference_latents[:, :, :1],
            mrope_offset=text_segment["vision_start_temporal_offset"],
            curr=text_segment["und_len"],
            device=target_latents.device,
            noisy_frame_indexes=torch.empty(0, device=target_latents.device, dtype=torch.long),
        )
        segment = {
            **vision_segment,
            "vision_token_shapes": [(reference_frames + target_frames, patch_h, patch_w)],
            "vision_sequence_indexes": torch.arange(
                text_segment["und_len"],
                text_segment["und_len"] + total_tokens,
                device=target_latents.device,
            ),
            "vision_mse_loss_indexes": vision_segment["vision_mse_loss_indexes"] + reference_tokens,
            "vision_noisy_frame_indexes": [
                indexes + reference_frames for indexes in vision_segment["vision_noisy_frame_indexes"]
            ],
            "vision_mrope_ids": torch.cat(
                [reference_segment["vision_mrope_ids"].repeat(1, reference_frames), vision_segment["vision_mrope_ids"]],
                dim=1,
            ),
            "num_vision_tokens": total_tokens,
            "num_reference_tokens": reference_tokens,
            "num_reference_frames": reference_frames,
        }
        return torch.cat([reference_latents, target_latents], dim=2), segment

    def _vision_timesteps(
        self,
        timestep: torch.Tensor,
        sample_index: int,
        batch_size: int,
        num_noisy_vision_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-token timesteps sized to the NOISY vision tokens only.

        The base transformer scatter-adds one timestep-embedding row per noisy
        vision token (transformer_cosmos3.py
        _apply_timestep_embeds_to_noisy_tokens: ``src`` rows must align with
        the flattened noisy indexes), so the row count is
        ``num_noisy_vision_tokens`` — equal to ``num_vision_tokens`` on every
        denoise path (KV-cache seed passes bypass this helper entirely: zero
        noisy tokens, no timestep).
        """
        # Rejected rather than cast: this guard catches callers that reach
        # the transformer without going through the pipeline. The fp32
        # rationale lives on prepare_transformer_timestep.
        if isinstance(timestep, torch.Tensor):
            assert timestep.dtype in (torch.int32, torch.int64, torch.float32, torch.float64), (
                f"Cosmos3 timesteps must be integer or full-precision float, got {timestep.dtype}; "
                "half-precision timesteps quantize t before embedding."
            )
        dtype = torch.float32
        # Strict shapes only: [B] or [B, num_noisy_vision_tokens].
        assert isinstance(timestep, torch.Tensor) and timestep.ndim >= 1, (
            f"Cosmos3 timesteps must be a tensor of ndim >= 1, got {type(timestep).__name__}"
            f"{'' if not isinstance(timestep, torch.Tensor) else f' with ndim {timestep.ndim}'}."
        )
        if timestep.ndim == 1:
            if timestep.numel() != batch_size:
                raise ValueError(
                    f"Cosmos3 1-D timesteps must be per-sample [B]={batch_size}, got {timestep.numel()} entries. "
                    "Per-block diffusion-forcing timesteps are not supported here; the pipeline expands them to "
                    "per-token [B, num_noisy_vision_tokens] before the transformer sees them."
                )
            return torch.full(
                (num_noisy_vision_tokens,), float(timestep[sample_index].item()), device=device, dtype=dtype
            )
        if timestep.ndim == 2:
            if timestep.shape != (batch_size, num_noisy_vision_tokens):
                raise ValueError(
                    f"Cosmos3 2-D timesteps must be exactly [B, num_noisy_vision_tokens]="
                    f"{(batch_size, num_noisy_vision_tokens)}, got {tuple(timestep.shape)}."
                )
            return timestep[sample_index].to(device=device, dtype=dtype)
        raise ValueError(
            f"Cosmos3 timesteps must be scalar, [B], or [B, num_noisy_vision_tokens]; got shape "
            f"{tuple(timestep.shape)}."
        )

    def _armed_block_causal_block_mask(
        self,
        *,
        und_len: int,
        global_latent_frames: int,
        block_frames: int,
        spatial_tokens_per_frame: int,
        device: torch.device | str,
        cp_pad_tokens: int = 0,
    ) -> BlockMask:
        """Targets see the text/reference prefix, earlier blocks, and their own block.

        CP exchanges preserve global token order, so every rank uses this mask.
        Padding extends queries only; padded keys have already been removed.
        """
        assert global_latent_frames % block_frames == 0, (
            f"block-granular causality needs a uniform block grid: {global_latent_frames} frames are not "
            f"divisible by block_frames={block_frames}"
        )
        block_size = attention_config.flex_block_size()
        key = (
            und_len,
            global_latent_frames,
            block_frames,
            spatial_tokens_per_frame,
            # Padded and unpadded forwards differ in query-axis geometry,
            # so they must not share a mask.
            cp_pad_tokens,
            torch.device(device),
            block_size,
        )
        cached = self.block_masks.get(key)
        if cached is not None:
            return cached
        block_mask = create_prefix_block_causal_block_mask(
            prefix_len=und_len,
            total_tokens=global_latent_frames * spatial_tokens_per_frame,
            frames_per_block=block_frames,
            frame_seqlen=spatial_tokens_per_frame,
            device=device,
            block_size=block_size,
            pad_query_tokens=cp_pad_tokens,
        )
        while len(self.block_masks) >= 2:
            # Evict oldest-inserted first (dicts preserve insertion order).
            self.block_masks.pop(next(iter(self.block_masks)))
        self.block_masks[key] = block_mask
        return block_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        neighbor_hidden_states: torch.Tensor | None,
        opacity: torch.Tensor,
        camera_rays: torch.Tensor,
        w2cs: torch.Tensor | None = None,
        neighbor_w2cs: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        neighbor_Ks: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        ignore_neighbors: bool = False,
        kv_cache: RolloutKvCache | None = None,
        kv_cache_seed: bool = False,
        kv_cache_start_frame: int = 0,
        frame_offset: int = 0,
        local_cp_input: bool = False,
        return_dict: bool = False,
        neighbor_camera_rays: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        """Use camera rays; accept but ignore w2cs/Ks and their neighbor counterparts for the shared Wan interface."""
        assert encoder_hidden_states.dtype in (
            torch.int32,
            torch.int64,
            torch.long,
        ), f"Cosmos3 prompts must be integer token ids. Retokenize prompts with prompt_format={COSMOS3_PROMPT_FORMAT}."
        assert (
            encoder_hidden_states.ndim == 2
        ), f"Cosmos3 prompt ids must be [B,L], got {tuple(encoder_hidden_states.shape)}"
        assert encoder_hidden_states.shape[0] == hidden_states.shape[0], (
            f"Cosmos3 prompt batch size {encoder_hidden_states.shape[0]} does not match latent batch size "
            f"{hidden_states.shape[0]}"
        )
        assert encoder_attention_mask is None or encoder_attention_mask.shape == encoder_hidden_states.shape, (
            f"Cosmos3 prompt attention mask shape {tuple(encoder_attention_mask.shape)} must match prompt ids shape "
            f"{tuple(encoder_hidden_states.shape)}"
        )

        use_reference_attention = (
            neighbor_hidden_states is not None
            and neighbor_hidden_states.shape[2] > 0
            and not self._as_bool(ignore_neighbors)
        )
        batch_size = hidden_states.shape[0]
        if local_cp_input:
            # The local-shard contract covers plain supervised forwards only:
            # the KV cache's row/commit math and the block-causal mask both
            # index the full clip.
            assert kv_cache is None, "local_cp_input does not support KV-cache passes"
            assert self.causal_block_frames is None, "local_cp_input does not support block-causal forwards"
        assert (
            kv_cache is None or frame_offset == 0
        ), "KV-cache passes address their chunk through kv_cache_start_frame; frame_offset must stay 0"
        if self.causal_block_frames is not None:
            # Full-sequence training masks and chunked cache passes are separate paths.
            assert kv_cache is None, (
                "Cosmos3 block-causal supervised forwards never take KV-cache passes: the cache is a "
                "rollout-only contract."
            )
        kv_history_chunks = 0
        if kv_cache is not None:
            # Inputs contain the current chunk; earlier frames come from the cache.
            assert (
                batch_size == 1
            ), f"Cosmos3 KV-cache rollouts are batch-1 (the text K/V cache holds one prompt), got batch {batch_size}"
            assert kv_cache_start_frame >= 0, f"kv_cache_start_frame must be >= 0, got {kv_cache_start_frame}"
            kv_history_chunks = kv_cache.num_cached_chunks
            kv_cache.assert_pass_starts_at(kv_cache_start_frame, "seed pass" if kv_cache_seed else "denoise pass")
            if kv_cache_seed:
                assert (
                    timestep is None
                ), "seed passes are pure conditioning mode (no time embedding): pass timestep=None"
            else:
                assert timestep is not None, "Cosmos3 KV-cache denoise passes require a timestep"
        else:
            assert (
                not kv_cache_seed and kv_cache_start_frame == 0
            ), "kv_cache_seed / kv_cache_start_frame require kv_cache"
        input_latent_frames = int(hidden_states.shape[2])
        input_start = kv_cache_start_frame + frame_offset
        real_latent_frames = input_latent_frames
        cp_pad_frames = 0
        if self._cp_mesh is not None and not local_cp_input:
            cp_pad_frames = cp_padded_frame_count(input_latent_frames, self._cp_world_size) - input_latent_frames
        if cp_pad_frames:
            # Whole-frame padding makes equal CP shards; attention excludes padded keys.
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, cp_pad_frames))
            opacity = F.pad(opacity, (0, 0, 0, 0, 0, cp_pad_frames * self.vae_scale_factor_temporal))
            camera_rays = F.pad(camera_rays, (0, 0, 0, 0, 0, 0, 0, cp_pad_frames))
            if isinstance(timestep, torch.Tensor) and timestep.ndim == 2:
                pad_spatial_tokens = math.ceil(hidden_states.shape[3] / self.latent_patch_size) * math.ceil(
                    hidden_states.shape[4] / self.latent_patch_size
                )
                timestep = F.pad(timestep, (0, cp_pad_frames * pad_spatial_tokens))
            input_latent_frames += cp_pad_frames
        if local_cp_input:
            if self._cp_mesh is not None:
                assert frame_offset == self._cp_rank * input_latent_frames, (
                    f"frame_offset {frame_offset} does not match CP rank {self._cp_rank} "
                    f"x local latent frames {input_latent_frames}"
                )
            cp_start_frame, cp_end_frame = 0, input_latent_frames
        else:
            cp_start_frame, cp_end_frame = self._cp_frame_slice(input_latent_frames)
        kv_cache_stage = {} if kv_cache is not None and kv_cache_seed else None
        noisy_frame_indexes = None
        if kv_cache is not None and kv_cache_seed:
            # Cache writes contain clean frames, with no timestep conditioning.
            noisy_frame_indexes = torch.empty(0, dtype=torch.long, device=hidden_states.device)
        opacity_tokens, camera_tokens = self._target_condition_tokens(
            opacity,
            camera_rays,
            hidden_states,
            cp_start_frame,
            cp_end_frame,
            start_frame=input_start,
        )
        if self._cp_mesh is not None:
            hidden_states = hidden_states[:, :, cp_start_frame:cp_end_frame].contiguous()
        if use_reference_attention:
            neighbor_hidden_states = neighbor_hidden_states.to(device=hidden_states.device, dtype=hidden_states.dtype)
            assert neighbor_camera_rays is not None, "Cosmos3 references require neighbor_camera_rays"
            reference_ray_shape = (
                batch_size,
                neighbor_hidden_states.shape[2],
                neighbor_hidden_states.shape[3] * self.vae_scale_factor_spatial,
                neighbor_hidden_states.shape[4] * self.vae_scale_factor_spatial,
                6,
            )
            assert (
                neighbor_camera_rays.shape == reference_ray_shape
            ), f"neighbor_camera_rays must have shape {reference_ray_shape}, got {tuple(neighbor_camera_rays.shape)}"
        device = hidden_states.device
        outputs = []
        for sample_index in range(batch_size):
            input_ids = self._trim_input_ids(encoder_hidden_states, encoder_attention_mask, sample_index)
            text_segment = self._prepare_text_segment(input_ids, device=device)
            sample_latents = hidden_states[sample_index : sample_index + 1]
            vision_segment = self._prepare_vision_segment(
                input_vision_tokens=sample_latents,
                mrope_offset=text_segment["vision_start_temporal_offset"],
                curr=text_segment["und_len"],
                device=device,
                # mRoPE positions are always full-clip absolute: where the
                # input starts plus the slice within it (cp_start_frame is 0
                # when inputs arrive pre-sliced).
                start_frame_offset=input_start + cp_start_frame,
                noisy_frame_indexes=noisy_frame_indexes,
            )
            if use_reference_attention:
                sample_latents, vision_segment = self._prepend_references(
                    sample_latents,
                    neighbor_hidden_states[sample_index : sample_index + 1],
                    text_segment,
                    vision_segment,
                )
            num_reference_tokens = vision_segment.get("num_reference_tokens", 0)
            position_ids = torch.cat([text_segment["text_mrope_ids"], vision_segment["vision_mrope_ids"]], dim=1)
            sequence_length = text_segment["und_len"] + vision_segment["num_vision_tokens"]
            # One frame's spatial token count (patch_h * patch_w), shared by
            # the timestep window, the block mask, and the KV-cache row
            # contract below.
            _, patch_h, patch_w = vision_segment["vision_token_shapes"][0]
            spatial_tokens_per_frame = patch_h * patch_w
            if kv_cache is not None and kv_cache_seed:
                vision_timesteps = torch.empty(0, dtype=torch.float32, device=device)
            else:
                sample_timestep = timestep
                if self._cp_mesh is not None and isinstance(timestep, torch.Tensor) and timestep.ndim == 2:
                    # 2-D per-token timesteps (diffusion-forcing training)
                    # arrive over the GLOBAL token axis; take this rank's
                    # frame window before _vision_timesteps' strict
                    # [B, num_noisy_vision_tokens] shape check. Scalar and
                    # 1-D timesteps need no slicing (per-sample expansion
                    # inside _vision_timesteps).
                    sample_timestep = timestep[
                        :, cp_start_frame * spatial_tokens_per_frame : cp_end_frame * spatial_tokens_per_frame
                    ]
                vision_timesteps = self._vision_timesteps(
                    sample_timestep,
                    sample_index,
                    batch_size,
                    vision_segment["num_noisy_vision_tokens"],
                    device,
                )
            gen_block_mask = None
            if self.causal_block_frames is not None:
                # Per sample: und_len varies with the trimmed prompt.
                gen_block_mask = self._armed_block_causal_block_mask(
                    und_len=text_segment["und_len"] + num_reference_tokens,
                    # Full clip by construction: local_cp_input is asserted
                    # off for block-causal forwards. REAL frames only —
                    # CP pad frames extend the query axis via
                    # cp_pad_tokens, never the block grid.
                    global_latent_frames=real_latent_frames,
                    block_frames=self.causal_block_frames,
                    spatial_tokens_per_frame=spatial_tokens_per_frame,
                    device=device,
                    cp_pad_tokens=cp_pad_frames * spatial_tokens_per_frame,
                )
            kv_cache_kwargs = {}
            if kv_cache is not None:
                kv_cache_kwargs = dict(
                    kv_cache=kv_cache,
                    kv_cache_stage=kv_cache_stage,
                    kv_cache_num_history_chunks=kv_history_chunks,
                    # Count this rank's target rows before any head exchange.
                    kv_cache_num_vision_tokens=(cp_end_frame - cp_start_frame) * spatial_tokens_per_frame,
                )
            self._conditioning_context = self._layer_conditioning_context(
                sample_index,
                opacity_tokens,
                camera_tokens,
                neighbor_hidden_states if use_reference_attention else None,
                vision_segment,
                gen_block_mask=gen_block_mask,
                cp_pad_tokens=cp_pad_frames * spatial_tokens_per_frame,
                neighbor_camera_rays=neighbor_camera_rays,
                **kv_cache_kwargs,
            )
            try:
                preds_vision, _, _ = self.base_transformer(
                    input_ids=text_segment["input_ids"],
                    text_indexes=text_segment["text_indexes"],
                    position_ids=position_ids,
                    und_len=text_segment["und_len"],
                    sequence_length=sequence_length,
                    vision_tokens=[sample_latents],
                    vision_token_shapes=vision_segment["vision_token_shapes"],
                    vision_sequence_indexes=vision_segment["vision_sequence_indexes"],
                    vision_mse_loss_indexes=vision_segment["vision_mse_loss_indexes"],
                    vision_timesteps=vision_timesteps,
                    vision_noisy_frame_indexes=vision_segment["vision_noisy_frame_indexes"],
                )
            finally:
                self._conditioning_context = None
                # Exception-path disarm (see _processor_disarm_hook).
                for layer in self.base_transformer.layers:
                    self._processor_disarm_hook(layer.self_attn, (), None)
            outputs.append(preds_vision[0][:, :, vision_segment.get("num_reference_frames", 0) :])
        if kv_cache_stage is not None:
            # Coverage bookkeeping counts REAL frames: pad rows never enter
            # the cache (the padded branch drops them right after the
            # exchange).
            kv_cache.commit_staged_chunk(kv_cache_stage, kv_cache_start_frame, real_latent_frames)
        output = torch.cat(outputs, dim=0)
        if not local_cp_input:
            output = self._gather_cp_output(output)
            if cp_pad_frames:
                # The caller must get back exactly the frame count it passed
                # in; the pad frames end here and never reach saved output.
                output = output[:, :, :real_latent_frames]
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
