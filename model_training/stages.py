# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
from accelerate.utils.fsdp_utils import fsdp2_prepare_model
from diffusers import Cosmos3OmniTransformer, WanTransformer3DModel

from model_training.checkpoint_metadata import resolve_camera_pose_alignment
from model_training.common.context_parallel import (
    configure_context_parallel_shard_size,
    create_context_parallel_mesh,
    validate_latent_frame_layout,
)
from model_training.model_profiles import is_cosmos3_model
from model_training.net.cosmos3_transformer import ArtifixerCosmos3Transformer
from model_training.net.wan_transformer import ArtifixerTransformer
from model_training.pipeline.pipeline_base import assert_training_rollout_fits_window
from model_training.trainers.context_parallel import ContextParallelTrainer
from model_training.trainers.dmd import DMD
from model_training.utils.train_utils import (
    _transformer_from_config,
    barrier_if_distributed,
    get_accelerator,
    get_kv_cache_pipe,
    get_run_id_and_should_resume,
    get_train_dataloader,
    get_val_datasets,
    load_model_weights_from_dcp,
    maybe_write_run_id,
    resume_training_from_checkpoint,
    run_supervised_training,
    run_validation_only,
    trainable_parameters,
)


def _trainer_factory(
    args, accelerator, optimizer, scheduler, train_dataloader, val_datasets, pipe, frames_per_block, step_offset
):
    # The stage is bidirectional, so ContextParallelTrainer takes no frames_per_block.
    return ContextParallelTrainer(
        args, accelerator, optimizer, scheduler, train_dataloader, val_datasets, pipe, step_offset
    )


def run_bidirectional(args: argparse.Namespace):
    run_supervised_training(
        args,
        frames_per_block=None,
        base_checkpoint_dir=args.base_checkpoint_dir,
        context_parallel=True,
        trainer_factory=_trainer_factory,
    )


def run_diffusion_forcing(args: argparse.Namespace):
    if args.context_parallel_size != 1:
        raise ValueError("Diffusion forcing requires --context_parallel_size 1; its trainer does not support CP.")
    run_supervised_training(args, frames_per_block=args.frames_per_block, base_checkpoint_dir=args.base_checkpoint_dir)


def build_score_transformer(args: argparse.Namespace, model_id: str, pipe, *, gradient_checkpointing: bool):
    # model_id may be --model_id_critic rather than args.model_id.
    # Reuse the student's resolved backend without resetting process-wide attention settings.
    attention_backend = pipe.attention_backend
    if is_cosmos3_model(model_id):
        wrapped = ArtifixerCosmos3Transformer(
            _transformer_from_config(Cosmos3OmniTransformer, model_id),
            vae_scale_factor_spatial=pipe.vae.config.scale_factor_spatial,
            vae_scale_factor_temporal=pipe.vae.config.scale_factor_temporal,
            gradient_checkpointing=gradient_checkpointing,
        )
        if attention_backend is not None:
            wrapped.set_attention_backend(attention_backend)
        return wrapped

    transformer = _transformer_from_config(WanTransformer3DModel, model_id)
    if attention_backend is not None:
        # Before wrapping: the Wan wrapper copies each block's backend from
        # the base at wrap time (same order as the student pipeline).
        transformer.set_attention_backend(attention_backend)
    return ArtifixerTransformer(
        transformer,
        frames_per_block=None,
        use_kv_cache_attention=False,
        vae_scale_factor_spatial=pipe.vae.config.scale_factor_spatial,
        vae_scale_factor_temporal=pipe.vae.config.scale_factor_temporal,
        gradient_checkpointing=gradient_checkpointing,
        checkpoint_every_n_blocks=args.checkpoint_every_n_blocks if gradient_checkpointing else 1,
    )


def run_distillation(args: argparse.Namespace):
    if args.model_id_critic is None:
        args.model_id_critic = args.model_id
    if is_cosmos3_model(args.model_id_critic) != is_cosmos3_model(args.model_id):
        raise ValueError(
            "DMD critic and student model families must match; mixed Cosmos3/Wan critics need separate "
            "prompt and conditioning plumbing."
        )
    if not getattr(args, "validate_only", False):
        resolve_camera_pose_alignment(
            args.base_checkpoint_dir_critic, args.model_id_critic, camera_pose_alignment=args.camera_pose_alignment
        )

    run_id, should_resume = get_run_id_and_should_resume(args)
    configure_context_parallel_shard_size(args)
    accelerator = get_accelerator(args, run_id)
    pipe = get_kv_cache_pipe(args, accelerator.device, truncate_rollout_gradients=args.truncate_rollout_gradients)

    latent_frames = validate_latent_frame_layout(
        num_frames=args.num_frames,
        temporal_scale=pipe.vae.config.scale_factor_temporal,
        frames_per_block=args.frames_per_block,
        context_parallel_size=args.context_parallel_size,
    )
    if not getattr(args, "validate_only", False):
        assert_training_rollout_fits_window(
            latent_frames, pipe.ar_window_chunks, pipe.ar_frames_per_chunk, training=True
        )
    assert (
        args.local_attn_size == -1 or args.local_attn_size >= args.sink_size
    ), f"local_attn_size {args.local_attn_size} must be -1 or at least sink_size {args.sink_size}"
    if args.context_parallel_size > 1:
        assert args.frames_per_block % args.context_parallel_size == 0, (
            f"frames_per_block {args.frames_per_block} must be divisible by context_parallel_size "
            f"{args.context_parallel_size}"
        )
        if args.sink_size > 0:
            assert (
                args.sink_size % args.frames_per_block == 0
            ), f"sink_size {args.sink_size} must be divisible by frames_per_block {args.frames_per_block}"
            assert (
                args.sink_size % args.context_parallel_size == 0
            ), f"sink_size {args.sink_size} must be divisible by context_parallel_size {args.context_parallel_size}"
        if args.local_attn_size > 0:
            assert (
                args.local_attn_size % args.frames_per_block == 0
            ), f"local_attn_size {args.local_attn_size} must be divisible by frames_per_block {args.frames_per_block}"
            assert args.local_attn_size % args.context_parallel_size == 0, (
                f"local_attn_size {args.local_attn_size} must be divisible by context_parallel_size "
                f"{args.context_parallel_size}"
            )

    # HOST-MEMORY INVARIANT: never hold more than ONE unsharded transformer
    # per rank. _transformer_from_config materializes full fp32 weights on
    # host RAM, so each transformer (student, critics, teacher) is built only
    # AFTER the previous one has been FSDP2-sharded, which frees the full
    # host copy. Weights load from DCP into the shards afterwards, so the
    # discarded random init is the only thing this reorders.

    if getattr(args, "validate_only", False):
        return run_validation_only(
            args, accelerator, pipe, frames_per_block=args.frames_per_block, context_parallel=True
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters(pipe.transformer), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Wait for all ranks before the first distributed prepare call; otherwise
    # slow model setup leaves faster ranks stuck inside FSDP initialization.
    barrier_if_distributed()

    transformer, optimizer, scheduler = accelerator.prepare(
        pipe.transformer,
        optimizer,
        scheduler,
    )
    pipe.transformer = transformer

    train_dataloader = get_train_dataloader(args, accelerator, args.frames_per_block)
    train_dataloader = accelerator.prepare_data_loader(train_dataloader, device_placement=False)
    # Raw video batches must stay on CPU until the DMD trainer applies CP-local
    # VAE/model slicing. Accelerate's default device placement would move the
    # full 1533-frame batch to every GPU before that slicing and can OOM.
    train_loader_device = getattr(train_dataloader, "device", None)
    if train_loader_device is not None:
        raise RuntimeError(f"Stage 3 training dataloader must stay on CPU, got device={train_loader_device}")

    # Student is sharded; only now materialize the fake-score critic.
    fake_score = build_score_transformer(
        args, args.model_id_critic, pipe, gradient_checkpointing=args.gradient_checkpointing
    )

    optimizer_fake_score = torch.optim.AdamW(
        trainable_parameters(fake_score), lr=args.learning_rate_critic, weight_decay=args.weight_decay
    )
    scheduler_fake_score = torch.optim.lr_scheduler.ConstantLR(optimizer_fake_score)

    fake_score, optimizer_fake_score, scheduler_fake_score = accelerator.prepare(
        fake_score, optimizer_fake_score, scheduler_fake_score
    )

    # Frozen teacher last, same one-unsharded-model-at-a-time invariant. It is
    # frozen (no optimizer), so it takes the plain fsdp2_prepare_model wrap
    # instead of accelerator.prepare; sharding still frees the full host copy.
    real_score = build_score_transformer(args, args.model_id_critic, pipe, gradient_checkpointing=False)
    real_score.eval()
    real_score.requires_grad_(False)
    real_score = fsdp2_prepare_model(accelerator, real_score)
    load_model_weights_from_dcp(real_score, args.base_checkpoint_dir_critic)

    cp_mesh = create_context_parallel_mesh(args.context_parallel_size, accelerator.device.type)
    if cp_mesh is not None:
        pipe.transformer.enable_context_parallel(cp_mesh)
        fake_score.enable_context_parallel(cp_mesh)
        real_score.enable_context_parallel(cp_mesh)

    trainer = DMD(
        args,
        accelerator,
        [optimizer, optimizer_fake_score],
        [scheduler, scheduler_fake_score],
        train_dataloader,
        {},
        pipe,
        real_score,
        fake_score,
    )
    # Should only load state after prepare is called
    if should_resume:
        train_dataloader, resume_state = resume_training_from_checkpoint(args, accelerator, train_dataloader)
        trainer.train_dataloader = train_dataloader
        trainer.step_offset = resume_state.step_offset
    else:
        load_model_weights_from_dcp(transformer, args.base_checkpoint_dir)
        load_model_weights_from_dcp(fake_score, args.base_checkpoint_dir_critic)
    trainer.ema.restore_after_model_load()
    maybe_write_run_id(accelerator, args.project_dir, run_id, args.log_with)

    # Stage 3 trains on the target rollout length but validates on full-clip
    # frames_all prompts to exercise the autoregressive KV-cache path.
    trainer.val_datasets = get_val_datasets(args, accelerator, None, args.frames_per_block)
    trainer.train()
