# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import gc
import math
import time
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh

from model_eval.generation import generate
from model_eval.validation_config import EvaluationConfig, EvaluationContext, EvaluationIdentity
from model_eval.validation_reporting import finish_evaluation
from model_eval.validation_scoring import EvaluationPass, ValidationRecorder
from model_eval.validation_tasks import (
    EvaluationPlan,
    ValidationTask,
    build_validation_task,
    task_generation_inputs,
    validation_noise_seed,
)
from model_training.common.context_parallel import conditioning_raw_frame_bounds, latent_range_to_raw_frame_slice
from model_training.net.context_parallel import all_gather_cat
from model_training.utils.train_utils import barrier_if_distributed

_VALIDATION_SYNC_TIMEOUT = timedelta(minutes=30)


@contextlib.contextmanager
def evaluation_mode(pipe):
    was_training = pipe.transformer.training
    pipe.transformer.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        pipe.transformer.train(was_training)


class EvaluationRunner:
    """Execute explicit task plans against an already prepared pipeline.

    Owns no training algorithm, optimizer, dataloader, critic, EMA, or scheduling.
    Distributed model preparation belongs to the caller. CP sampling meshes are
    cached in the execution context so repeated training validations reuse them.
    """

    def __init__(self, pipe, config: EvaluationConfig, context: EvaluationContext):
        self.pipe = pipe
        self.config = config
        self.context = context
        self.recorder = None

    @torch.no_grad()
    def run_pass(self, datasets, task_plan, save_dir: Path | None, *, weights="raw") -> EvaluationPass:
        started = time.perf_counter()
        self.recorder = ValidationRecorder(
            self.config, self.context, self.pipe.vae.config.scale_factor_temporal, weights=weights
        )
        self.generate_tasks(datasets, task_plan, save_dir)
        return self.recorder.finish(time.perf_counter() - started)

    def finish(self, plan, identity, regular, panel, save_dir):
        return finish_evaluation(
            self.config, self.context, plan, identity, regular, panel, save_dir, pipeline_name=type(self.pipe).__name__
        )

    def run(self, plan: EvaluationPlan, save_dir: Path, identity: EvaluationIdentity):
        with evaluation_mode(self.pipe):
            regular = self.run_pass(plan.datasets, plan.regular, save_dir, weights=identity.weights)
            panel = self.run_pass(plan.datasets, plan.panel, None, weights=identity.weights)
        return self.finish(plan, identity, regular, panel, save_dir)

    def generate_tasks(self, datasets, task_plan, save_dir):
        if not task_plan:
            return
        mesh = getattr(self.pipe.transformer, "cp_mesh", None)
        if getattr(self.pipe, "stages_conditioning_per_chunk", False) and mesh is not None and mesh.size() > 1:
            self._generate_kv_cp(datasets, task_plan, save_dir)
        else:
            self._generate_standard(datasets, task_plan, save_dir)

    def _slice_validation_task_for_cp(
        self,
        task: ValidationTask,
        *,
        cp_group_size: int,
        group_rank: int,
        global_num_frames: int,
    ) -> tuple[ValidationTask, int, tuple[int, int] | None]:
        if cp_group_size <= 1:
            return task, 0, None

        # Validation CP is explicit temporal slicing around a normal inference
        # call.  Each rank gets a raw RGB VAE window with optional halo, but all
        # transformer-space tensors are sliced to exactly the rank's latent
        # interval.  The returned crop removes the halo after VAE encoding.
        temporal_scale = self.pipe.vae.config.scale_factor_temporal
        total_latent_frames = (global_num_frames - 1) // temporal_scale + 1
        assert total_latent_frames % cp_group_size == 0, "validation latent frames must divide by CP group size"
        frames_per_rank = total_latent_frames // cp_group_size
        latent_start = group_rank * frames_per_rank
        latent_end = latent_start + frames_per_rank
        raw_start, raw_end, crop_start, crop_end = latent_range_to_raw_frame_slice(
            latent_start,
            latent_end,
            total_raw_frames=global_num_frames,
            temporal_scale=temporal_scale,
            halo_latent_frames=self.config.vae_chunk_halo_latent_frames,
        )
        # Validation inference does not need the VAE halo in transformer-space
        # inputs; only the raw RGB window passed to the VAE keeps that halo.
        cond_raw_start, cond_raw_end = conditioning_raw_frame_bounds(latent_start, latent_end, temporal_scale)
        local_raw_start = raw_start + crop_start * temporal_scale
        local_raw_end = min(global_num_frames, raw_start + (crop_end - 1) * temporal_scale + 1)

        inference_task = replace(
            task,
            rendered_rgb=task.rendered_rgb[:, raw_start:raw_end],
            rendered_opacity=task.rendered_opacity[:, cond_raw_start:cond_raw_end],
            camera_rays=task.camera_rays[:, latent_start:latent_end],
            w2cs=task.w2cs[:, latent_start:latent_end],
            Ks=task.Ks[:, latent_start:latent_end],
            rgb_gt=task.rgb_gt[:, local_raw_start:local_raw_end],
        )
        return inference_task, latent_start, (crop_start, crop_end)

    def _move_validation_task_to_device(self, task: ValidationTask) -> ValidationTask:
        return replace(
            task,
            **{
                field.name: getattr(task, field.name).to(self.context.device, non_blocking=True)
                for field in fields(task)
                if field.name != "rgb_gt" and isinstance(getattr(task, field.name), torch.Tensor)
            },
        )

    @staticmethod
    def _choose_cp_group_size(
        cp_size_candidates: list[int],
        world_size: int,
        num_tasks: int,
        pass_latent_frames: int,
    ) -> int:
        """Pick the candidate CP group size that minimizes validation
        wall-clock time.

        ``cp_size_candidates`` comes from ``pipe.cp_size_candidates`` and
        ``pass_latent_frames`` from ``pipe.cp_pass_latent_frames`` (the
        latent frames one pass shards). Each candidate costs
        ``rounds * ceil(pass_latent_frames / cp_size)`` —
        the frames one rank computes across its rounds — so CP frame padding
        counts against a candidate, and ascending iteration with a strict
        ``<`` keeps the smallest size on ties (a pad-free divisor beats a
        padded size of equal cost).

        Ranks left over when ``world_size % cp_group_size != 0`` become
        spectators: they run a minimal dummy forward pass so FSDP collectives
        stay aligned.  FSDP all-gathers only reconstruct parameters, so their
        cost does not depend on input size and the dummy forwards do not slow
        down the CP-active ranks.

        Examples::

            # Frame-divisor candidates [1, 3, 7] (21 frames), world_size=9,
            # num_tasks=12:
            #   d=1: groups=9, rounds=ceil(12/9)=2,  cost=2*21=42
            #   d=3: groups=3, rounds=ceil(12/3)=4,  cost=4*7=28  <-- best
            #   d=7: groups=1, rounds=ceil(12/1)=12, cost=12*3=36
            # Returns 3

            # Cosmos3 production geometry: head-gcd candidates [1, 2, 4, 8]
            # (21 frames x 32/8 heads — the transformer pads the frame
            # axis), world_size=8, num_tasks=12:
            #   d=1: groups=8, rounds=2,  cost=2*21=42
            #   d=2: groups=4, rounds=3,  cost=3*ceil(21/2)=33  <-- best
            #   d=4: groups=2, rounds=6,  cost=6*ceil(21/4)=36
            #   d=8: groups=1, rounds=12, cost=12*ceil(21/8)=36
            # Returns 2 (frame divisors of 21 alone would force CP=1)
        """
        assert pass_latent_frames > 0, f"pass_latent_frames must be positive, got {pass_latent_frames}"
        best_size, best_time = 1, float("inf")
        for size in sorted(set(cp_size_candidates)):
            num_groups = world_size // size
            rounds = math.ceil(num_tasks / num_groups)
            time = rounds * math.ceil(pass_latent_frames / size)
            if time < best_time:
                best_time = time
                best_size = size
        return best_size

    def _build_minimal_validation_task(
        self,
        key: str,
        num_frames: int,
        num_latent_frames: int,
        num_inference_steps: int,
        device: torch.device,
        vae_dtype: torch.dtype,
    ) -> ValidationTask:
        """Build a minimal-resolution task for ranks that only need FSDP sync.

        Inactive validation groups must still call ``forward_inference`` so that
        FSDP all-gathers fire, but their output is discarded.  Keep the **same
        temporal extent** as real tasks (so the KV-cache pipeline produces the
        same number of chunks / transformer calls — required for FSDP sync)
        but uses the smallest spatial resolution accepted by the active VAE
        and transformer patchification, making per-layer compute negligible.

        ``vae_dtype`` (typically bf16) is used for opacity and optional target
        trajectory tensors to match the dtype that ``_build_validation_task``
        produces. These tensors bypass the VAE and are mixed directly with bf16
        latents, so a float32 mismatch would upcast everything.
        """
        min_hw = max(self.pipe.spatial_alignment())
        pixel_dtype = torch.float32  # pixels go through the VAE, which casts
        B = 1

        prompt, prompt_attention_mask, negative_prompt, negative_prompt_attention_mask = (
            self.pipe.dummy_validation_prompt(device)
        )

        camera_rays = torch.zeros(B, num_latent_frames, min_hw, min_hw, 6, device=device, dtype=vae_dtype)

        return ValidationTask(
            key=key,
            rendered_rgb=torch.zeros(B, num_frames, 3, min_hw, min_hw, device=device, dtype=pixel_dtype),
            rendered_opacity=torch.zeros(B, num_frames, min_hw, min_hw, device=device, dtype=vae_dtype),
            neighbors=None,  # None → skips encode_neighbors, PRoPE, and neighbor cross-attn entirely
            camera_rays=camera_rays,
            w2cs=torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, num_latent_frames, -1, -1),
            neighbor_w2cs=None,
            Ks=torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, num_latent_frames, -1, -1),
            neighbor_Ks=None,
            prompt=prompt,
            prompt_attention_mask=prompt_attention_mask,
            encoded_negative_prompt=negative_prompt,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            rgb_gt=torch.zeros(B, num_frames, 3, min_hw, min_hw, device=device, dtype=pixel_dtype),
        )

    def _gather_cp_validation_latents(self, local_latents: torch.Tensor) -> torch.Tensor | None:
        cp_mesh = self.pipe.transformer.cp_mesh
        assert cp_mesh is not None, "CP validation latent gather requires an active CP mesh"
        if cp_mesh.size() == 1:
            return local_latents

        gathered = all_gather_cat(local_latents, cp_mesh, dim=2)
        return gathered if dist.get_rank(group=cp_mesh.get_group()) == 0 else None

    def _generate_standard(self, val_datasets, task_plan, save_dir):
        num_inference_steps = self.config.sampling.num_inference_steps
        task_specs = [spec for spec, _ in task_plan]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        validation_sync_group = (
            dist.new_group(backend="nccl", timeout=_VALIDATION_SYNC_TIMEOUT) if dist.is_initialized() else None
        )

        try:
            if rank == 0 and save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)

            barrier_if_distributed(validation_sync_group)

            # Hybrid task-parallel + context-parallel validation.
            # GPUs are split into CP groups; each group cooperates on one task via
            # all-gather K/V attention, and different groups run different tasks in
            # parallel.  For block-causal autoregressive attention, CP is still
            # possible but frames_per_rank must be divisible by frames_per_block,
            # which constrains valid CP sizes to divisors of num_blocks.
            vae_temporal = self.pipe.vae.config.scale_factor_temporal
            reference_dataset = val_datasets[task_specs[0].num_neighbors]
            num_latent_frames = (reference_dataset.num_frames - 1) // vae_temporal + 1
            cp_group_size = self._choose_cp_group_size(
                self.pipe.cp_size_candidates(num_latent_frames, world_size),
                world_size,
                len(task_specs),
                self.pipe.cp_pass_latent_frames(num_latent_frames),
            )
            num_groups = world_size // cp_group_size
            group_index = rank // cp_group_size
            group_rank = rank % cp_group_size
            # Ranks that don't fit into a complete group are spectators.
            # Spectators run a tiny dummy forward to keep FSDP all-gathers in sync
            # without bottlenecking CP-active ranks (FSDP collectives are
            # input-size-independent).
            is_spectator = group_index >= num_groups
            num_spectators = world_size - num_groups * cp_group_size
            rounds = math.ceil(len(task_specs) / num_groups)

            if rank == 0:
                print(
                    f"Validation CP: cp_size={cp_group_size}, groups={num_groups}, "
                    f"spectators={num_spectators}, tasks={len(task_specs)}, "
                    f"rounds={rounds}",
                    flush=True,
                )

            # Built lazily only on ranks/rounds without real validation work.
            # Some ranks never need it, so the cache may remain empty.
            cached_dummy_task: ValidationTask | None = None
            previous_cp_mesh = self.pipe.transformer.cp_mesh

            if cp_group_size > 1:
                if cp_group_size not in self.context.cp_meshes:
                    # All ranks must participate in DeviceMesh creation (it involves
                    # dist.new_group which is a world-collective in some PyTorch versions).
                    # Build the full list of CP group rank-lists so every rank calls
                    # new_group the same number of times. Cache the result: DeviceMesh
                    # process groups register NCCL communicators that are not freed
                    # during the run.
                    all_group_ranks = [
                        list(range(g * cp_group_size, (g + 1) * cp_group_size)) for g in range(num_groups)
                    ]
                    # Spectators form their own singleton group(s) for the collective.
                    spectator_ranks = list(range(num_groups * cp_group_size, world_size))
                    if spectator_ranks:
                        all_group_ranks.append(spectator_ranks)

                    my_mesh = None
                    for group_ranks in all_group_ranks:
                        mesh = DeviceMesh(self.context.device.type, group_ranks)
                        if rank in group_ranks:
                            my_mesh = mesh
                    self.context.cp_meshes[cp_group_size] = my_mesh

                if is_spectator:
                    self.pipe.transformer.disable_context_parallel()
                else:
                    self.pipe.transformer.enable_context_parallel(self.context.cp_meshes[cp_group_size])
            else:
                self.pipe.transformer.disable_context_parallel()

            try:
                # Round-robin tasks across active groups; inactive groups run a
                # minimal forward so FSDP collectives stay aligned without
                # increasing full-resolution validation memory fanout.
                # The sync group uses a longer timeout because only group_rank 0 saves results.
                for round_idx in range(rounds):
                    barrier_if_distributed(validation_sync_group)
                    task_index = round_idx * num_groups + (group_index if not is_spectator else 0)
                    is_real_task = not is_spectator and task_index < len(task_specs)
                    cp_validation_active = not is_spectator and cp_group_size > 1
                    use_local_cp = cp_validation_active and self.pipe.use_local_cp_inference(
                        cp_group_size, num_latent_frames
                    )

                    if is_real_task:
                        spec, sample_index = task_plan[task_index]
                        task = build_validation_task(
                            self.pipe,
                            self.config,
                            self.context,
                            val_datasets,
                            spec,
                            sample_index,
                            copy_to_device=not use_local_cp,
                        )
                    else:
                        if cached_dummy_task is None:
                            cached_dummy_task = self._build_minimal_validation_task(
                                "__dummy__",
                                reference_dataset.num_frames,
                                num_latent_frames,
                                num_inference_steps,
                                self.context.device,
                                self.pipe.vae.dtype,
                            )
                        task = cached_dummy_task

                    save_task = task
                    local_frame_offset = 0
                    local_vae_crop = None
                    if use_local_cp:
                        save_task = task if group_rank == 0 else None
                        # CP validation slices raw RGB with VAE halo, while
                        # transformer-space tensors use this rank's exact
                        # latent-frame range. The latent shard is gathered and
                        # decoded on group_rank 0 after inference.
                        task, local_frame_offset, local_vae_crop = self._slice_validation_task_for_cp(
                            task,
                            cp_group_size=cp_group_size,
                            group_rank=group_rank,
                            global_num_frames=reference_dataset.num_frames,
                        )
                        task = self._move_validation_task_to_device(task)

                    inputs = task_generation_inputs(task)
                    self.recorder.start_generation()
                    local_cp = (
                        {"local_frame_offset": local_frame_offset, "local_vae_crop": local_vae_crop}
                        if use_local_cp
                        else None
                    )
                    output = generate(
                        self.pipe,
                        inputs,
                        self.config.sampling,
                        local_cp=local_cp,
                        noise_seed=validation_noise_seed(task.key),
                        show_progress=rank == 0,
                    )
                    if is_real_task:
                        if use_local_cp:
                            global_latents = self._gather_cp_validation_latents(output)
                            if group_rank == 0:
                                assert save_task is not None
                                assert global_latents is not None
                                global_output = self.pipe.decode_latents_to_video(
                                    global_latents,
                                    temporal_chunk_size=output.shape[2],
                                )
                                self.recorder.save(global_output, save_task, save_dir, cp_size=cp_group_size)
                                del global_output
                            del global_latents
                        elif group_rank == 0:
                            self.recorder.save(output, save_task, save_dir, cp_size=cp_group_size)

                    del output
                    del task
                    del save_task
                    torch.cuda.empty_cache()
                    gc.collect()
            finally:
                if previous_cp_mesh is None:
                    self.pipe.transformer.disable_context_parallel()
                else:
                    self.pipe.transformer.enable_context_parallel(previous_cp_mesh)

            barrier_if_distributed(validation_sync_group)
        finally:
            if validation_sync_group is not None:
                dist.destroy_process_group(validation_sync_group)

    def _generate_kv_cp(self, val_datasets, task_plan, save_dir):
        if self.context.is_main_process and save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
        barrier_if_distributed()
        for spec, index in task_plan:
            task = build_validation_task(
                self.pipe, self.config, self.context, val_datasets, spec, index, copy_to_device=False
            )
            self.recorder.start_generation()
            latents = generate(
                self.pipe,
                task_generation_inputs(task),
                self.config.sampling,
                output_latents=True,
                noise_seed=validation_noise_seed(task.key),
                show_progress=self.context.rank == 0,
            )
            if self.context.is_main_process:
                cp_size = self.pipe.transformer.cp_mesh.size()
                chunk_size = max(latents.shape[2] // cp_size, 1)
                output = self.pipe.decode_latents_to_video(latents, temporal_chunk_size=chunk_size)
                self.recorder.save(output, task, save_dir, cp_size=cp_size)
                del output
            del latents, task
            torch.cuda.empty_cache()
            barrier_if_distributed()
