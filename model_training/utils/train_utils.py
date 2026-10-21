# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate import Accelerator, DataLoaderConfiguration, FullyShardedDataParallelPlugin, ParallelismConfig
from accelerate.utils import GradientAccumulationPlugin, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKLWan, Cosmos3OmniTransformer, UniPCMultistepScheduler, WanTransformer3DModel
from torch.distributed.fsdp import MixedPrecisionPolicy

from model_training.arguments import LOG_WITH_CHOICES, LOG_WITH_WANDB
from model_training.checkpoint_metadata import resolve_camera_pose_alignment, write_checkpoint_metadata
from model_training.common.context_parallel import (
    configure_context_parallel_shard_size,
    enable_training_context_parallel,
    validate_latent_frame_layout,
)
from model_training.data.dl3dv_test import DL3DVPairedDatasetTest
from model_training.data.dl3dv_train import DL3DVPairedDatasetTrain
from model_training.model_profiles import (
    flow_shift_for_model,
    is_cosmos3_model,
    prompt_format_for_model,
    spatial_resize_multiple,
    uses_expanded_timesteps,
    validate_model_profile,
)
from model_training.net.attention_config import configure_attention
from model_training.pins import dcp_cache_dir_name, hf_revision_for_model
from model_training.pipeline.cosmos3_kv_cache_pipeline import ArtifixerCosmos3KvCachePipeline
from model_training.pipeline.cosmos3_pipeline import ArtifixerCosmos3Pipeline
from model_training.pipeline.pipeline_base import validate_latent_frames_divisible
from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline
from model_training.pipeline.wan_pipeline import ArtifixerPipeline
from model_training.utils.precision import disable_tf32_matmul
from model_training.utils.validation_artifacts import VALIDATION_SEED

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"

_CHECKPOINT_COMPLETE_FILENAME = "checkpoint_complete"
# The requeue protocol's three W&B identities: fresh episode = never (do
# not attach to a stale run), authorized resume = must (reattach to the same
# run id), unset = allow (wandb default). Anything else is a typo.
_WANDB_RESUME_CHOICES = {"allow", "must", "never"}


@dataclass(frozen=True)
class ResumeState:
    step_offset: int = 0


def _validate_training_pose_alignment(args: argparse.Namespace, checkpoint: Path | str) -> None:
    args.camera_pose_alignment = resolve_camera_pose_alignment(
        checkpoint,
        args.model_id,
        camera_pose_alignment=args.camera_pose_alignment,
    )


def _validate_training_resume(args: argparse.Namespace, accelerator: Accelerator, checkpoint: Path | str) -> None:
    _validate_training_pose_alignment(args, checkpoint)
    for index, model in enumerate(accelerator._models):
        slot = Path(checkpoint) / f"pytorch_model_fsdp_{index}"
        if (slot / ".metadata").is_file():
            _validate_training_pose_alignment(args, slot)
            _validate_dcp_model_keys(model.state_dict(), dcp.FileSystemReader(str(slot)))


def _register_checkpoint_metadata_hooks(args: argparse.Namespace, accelerator: Accelerator) -> None:
    """Save pose alignment with each independently loadable model slot."""
    metadata = {"camera_pose_alignment": args.camera_pose_alignment}

    def save_metadata(models, weights, output_dir) -> None:
        if accelerator.is_main_process:
            output_dir = Path(output_dir)
            checkpoint_paths = [output_dir, *(output_dir / f"pytorch_model_fsdp_{i}" for i in range(len(models)))]
            for checkpoint in checkpoint_paths:
                write_checkpoint_metadata(checkpoint, metadata)

    accelerator.register_save_state_pre_hook(save_metadata)


def _checkpoint_complete_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / _CHECKPOINT_COMPLETE_FILENAME


def _checkpoint_step_from_path(checkpoint_dir: Path | str) -> int:
    name = Path(checkpoint_dir).name
    if not name.startswith("checkpoint_"):
        raise ValueError(f"Checkpoint directory must be named checkpoint_<step>, got {checkpoint_dir}")
    try:
        return int(name.removeprefix("checkpoint_"))
    except ValueError as e:
        raise ValueError(f"Checkpoint directory must be named checkpoint_<step>, got {checkpoint_dir}") from e


def checkpoint_is_complete(project_dir: Path, step: int, accelerator: Accelerator | None = None) -> bool:
    checkpoint_dir = project_dir / "checkpoints" / f"checkpoint_{step}"
    complete = _checkpoint_complete_path(checkpoint_dir).is_file()
    if accelerator is not None and dist.is_initialized():
        complete_tensor = torch.tensor(
            [int(complete if accelerator.is_main_process else False)],
            device=accelerator.device,
            dtype=torch.int32,
        )
        dist.broadcast(complete_tensor, src=0)
        complete = bool(complete_tensor.item())
    return complete


def mark_checkpoint_complete(project_dir: Path, step: int, accelerator: Accelerator) -> None:
    checkpoint_dir = project_dir / "checkpoints" / f"checkpoint_{step}"
    local_marker_succeeded = True
    caught_error = None

    if accelerator.is_main_process:
        try:
            _checkpoint_complete_path(checkpoint_dir).touch()
        except Exception as e:
            local_marker_succeeded = False
            caught_error = e

    if dist.is_initialized():
        marker_succeeded_tensor = torch.tensor(
            [int(local_marker_succeeded)],
            device=accelerator.device,
            dtype=torch.int32,
        )
        dist.all_reduce(marker_succeeded_tensor, op=dist.ReduceOp.MIN)
        marker_succeeded = bool(marker_succeeded_tensor.item())
    else:
        marker_succeeded = local_marker_succeeded

    if not marker_succeeded:
        if caught_error is not None:
            raise caught_error
        raise RuntimeError(f"Failed to mark checkpoint complete: {checkpoint_dir}")


def uses_wandb_logging(log_with: str) -> bool:
    assert log_with in LOG_WITH_CHOICES, f"log_with must be one of {LOG_WITH_CHOICES}, got {log_with!r}"
    return log_with == LOG_WITH_WANDB


def wandb_resume_mode() -> str:
    resume_mode = os.environ.get("WANDB_RESUME", "allow")
    if resume_mode not in _WANDB_RESUME_CHOICES:
        raise ValueError(f"WANDB_RESUME must be one of {sorted(_WANDB_RESUME_CHOICES)}, got {resume_mode!r}")
    return resume_mode


def read_wandb_run_id(project_dir: Path) -> str:
    run_id_path = project_dir / "run_id.txt"
    if not run_id_path.is_file():
        raise FileNotFoundError(f"Cannot resume W&B logging without run id file: {run_id_path}")
    run_id = run_id_path.read_text().strip()
    if not run_id:
        raise ValueError(f"W&B run id file is empty: {run_id_path}")
    return run_id


def _has_completed_checkpoint(checkpoints_dir: Path) -> bool:
    if not checkpoints_dir.is_dir():
        return False
    return any(
        folder.is_dir() and folder.name.startswith("checkpoint_") and _checkpoint_complete_path(folder).is_file()
        for folder in checkpoints_dir.iterdir()
    )


def get_run_id_and_should_resume(args: argparse.Namespace) -> tuple[str | None, bool]:
    if getattr(args, "validate_only", False):
        if args.resume_from_checkpoint is not None:
            raise ValueError("--validate_only loads weights with --base_checkpoint_dir; omit --resume_from_checkpoint")
        if getattr(args, "base_checkpoint_dir", None) is None:
            raise ValueError("--validate_only requires --base_checkpoint_dir")
        _validate_training_pose_alignment(args, args.base_checkpoint_dir)
        manifest_path = getattr(args, "validation_manifest", None)
        if manifest_path is not None:
            from model_training.utils.validation_artifacts import validate_manifest

            manifest = json.loads(manifest_path.read_text())
            validate_manifest(manifest)
            args.num_frames = manifest["num_frames"]
        elif getattr(args, "validation_num_frames", None) is not None:
            args.num_frames = args.validation_num_frames
        return None, False
    run_id = None
    should_resume = False
    if uses_wandb_logging(args.log_with) and (args.project_dir / "run_id.txt").is_file():
        run_id = read_wandb_run_id(args.project_dir)

    if args.resume_from_checkpoint is not None:
        auto_resume = args.resume_from_checkpoint == "auto"
        checkpoints_dir = args.project_dir / "checkpoints"
        if (not auto_resume) or _has_completed_checkpoint(checkpoints_dir):
            should_resume = True
            if uses_wandb_logging(args.log_with) and run_id is None:
                run_id = read_wandb_run_id(args.project_dir)
        elif checkpoints_dir.exists():
            # A hard kill during the very first save can leave only incomplete
            # checkpoint dirs. Start fresh in that case: resuming would make
            # load_checkpoint_state raise, so every requeued job would fail
            # the same way.
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"No completed checkpoints in {checkpoints_dir}; starting fresh")

    return run_id, should_resume


def load_checkpoint_state(args: argparse.Namespace, accelerator: Accelerator) -> int:
    if args.resume_from_checkpoint != "auto":
        checkpoint_step = _checkpoint_step_from_path(args.resume_from_checkpoint)
        _validate_training_resume(args, accelerator, args.resume_from_checkpoint)
        accelerator.load_state(args.resume_from_checkpoint)
        return checkpoint_step

    input_dir = None
    checkpoint_step = None
    failure_message = None
    if accelerator.is_main_process:
        checkpoints_dir = Path(accelerator.project_dir) / "checkpoints"
        completed_folders = []
        if not checkpoints_dir.exists():
            failure_message = f"No checkpoints directory found at {checkpoints_dir}"
        else:
            checkpoint_folders = []
            for folder in checkpoints_dir.iterdir():
                if not folder.is_dir() or not folder.name.startswith("checkpoint_"):
                    continue
                try:
                    step = _checkpoint_step_from_path(folder)
                except ValueError:
                    continue
                checkpoint_folders.append((step, folder))

            checkpoint_folders.sort(key=lambda pair: pair[0])
            completed_folders = [
                (step, folder) for step, folder in checkpoint_folders if _checkpoint_complete_path(folder).is_file()
            ]

        if not completed_folders and failure_message is None:
            failure_message = f"No completed checkpoints left to resume in {checkpoints_dir}"
        elif failure_message is None:
            checkpoint_step, input_dir_path = completed_folders[-1]
            input_dir = str(input_dir_path)

    values = [input_dir, checkpoint_step] if accelerator.is_main_process else [None, None]
    failure_values = [failure_message] if accelerator.is_main_process else [None]
    if dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
        dist.broadcast_object_list(failure_values, src=0)

    if failure_values[0] is not None:
        raise ValueError(failure_values[0])

    _validate_training_resume(args, accelerator, values[0])
    accelerator.load_state(values[0])
    return int(values[1])


def resume_training_from_checkpoint(
    args: argparse.Namespace,
    accelerator: Accelerator,
    train_dataloader: torch.utils.data.DataLoader,
) -> tuple[torch.utils.data.DataLoader, ResumeState]:
    checkpoint_step = load_checkpoint_state(args, accelerator)
    restored_step = accelerator.step // args.gradient_accumulation_steps
    step_offset = checkpoint_step - restored_step
    if step_offset < 0:
        raise RuntimeError(
            f"Checkpoint {checkpoint_step} is older than restored accelerator step {restored_step}; "
            "cannot compute a consistent resume offset"
        )

    batches_to_skip = checkpoint_step * args.gradient_accumulation_steps
    if batches_to_skip > 0:
        if not hasattr(accelerator, "skip_first_batches"):
            raise RuntimeError("This Accelerate version cannot skip dataloader batches on resume")
        train_dataloader = accelerator.skip_first_batches(train_dataloader, batches_to_skip)

    accelerator.print(
        f"Resumed from checkpoint_{checkpoint_step}: restored_step={restored_step}, "
        f"step_offset={step_offset}, skipped_batches_per_rank={batches_to_skip}"
    )
    return train_dataloader, ResumeState(step_offset=step_offset)


def barrier_if_distributed(group: dist.ProcessGroup | None = None) -> None:
    if dist.is_initialized():
        dist.barrier(group=group)


def _validate_dcp_model_keys(state_dict: dict, reader: dcp.FileSystemReader) -> None:
    expected = {f"model.{name}" for name in state_dict}
    stored = {name for name in reader.read_metadata().state_dict_metadata if name.startswith("model.")}
    missing, unexpected = sorted(expected - stored), sorted(stored - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"{reader.path} does not match the model state dict: "
            f"{len(missing)} missing key(s) {missing[:3]}, "
            f"{len(unexpected)} unexpected key(s) {unexpected[:3]}"
        )


def load_model_weights_from_dcp(model: torch.nn.Module, checkpoint_dir: Path | str) -> None:
    state_dict = model.state_dict()
    reader = dcp.FileSystemReader(str(checkpoint_dir))
    _validate_dcp_model_keys(state_dict, reader)
    dcp.load({"model": state_dict}, reader)
    model.load_state_dict(state_dict)


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def get_accelerator(args: argparse.Namespace, run_id: str | None) -> Accelerator:
    # No need to sync with the end of the dataloader since we don't really have a notion of epochs
    grad_acc_plugin = GradientAccumulationPlugin(sync_with_dataloader=False, num_steps=args.gradient_accumulation_steps)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=True,
        mixed_precision_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=(
            ["Cosmos3VLTextMoTDecoderLayer"] if is_cosmos3_model(args.model_id) else ["ArtifixerTransformerBlock"]
        ),
    )
    world_size = int(os.environ["ROLE_WORLD_SIZE"]) if "ROLE_WORLD_SIZE" in os.environ else 1
    if args.shard_size is None:
        local_size = int(os.environ["LOCAL_WORLD_SIZE"]) if "LOCAL_WORLD_SIZE" in os.environ else 1
    else:
        local_size = args.shard_size
    if world_size % local_size != 0:
        raise ValueError(f"Local shard size {local_size} must divide world size {world_size}")
    parallelism_config = ParallelismConfig(
        dp_replicate_size=world_size // local_size,
        dp_shard_size=local_size,
    )

    accelerator_log_with = LOG_WITH_WANDB if uses_wandb_logging(args.log_with) else None

    accelerator = Accelerator(
        project_config=ProjectConfiguration(
            project_dir=args.project_dir,
            automatic_checkpoint_naming=True,
            # Accelerate prunes before saving; TrainerBase applies retention
            # only after the replacement has a distributed completion marker.
            total_limit=None,
        ),
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
        # bf16 compute is fixed by the FSDP2 MixedPrecisionPolicy above; keep both in sync.
        mixed_precision="bf16",
        gradient_accumulation_plugin=grad_acc_plugin,
        fsdp_plugin=fsdp_plugin,
        parallelism_config=parallelism_config,
        log_with=accelerator_log_with,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(minutes=args.process_group_timeout_minutes)),
        ],
    )

    _register_checkpoint_metadata_hooks(args, accelerator)

    init_kwargs = dict()

    if uses_wandb_logging(args.log_with):
        wandb_init_kwargs = {
            "name": (args.tracker_run_name if args.tracker_run_name else args.project_dir.name),
            "dir": args.project_dir / "wandb",
        }
        if run_id is not None:
            wandb_init_kwargs["id"] = run_id
            wandb_init_kwargs["resume"] = wandb_resume_mode()

        init_kwargs["wandb"] = wandb_init_kwargs

    if accelerator_log_with is not None:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)), init_kwargs=init_kwargs)

    # Seed after Accelerator() so process_index is available. device_specific=True gives
    # each rank args.seed + process_index so dropout/noise are decorrelated across DP
    # ranks (training runs pure DP+FSDP, not CP). CP paths that require rank-identical
    # RNG resolve divergence explicitly via dist.broadcast — see e.g. pipeline_base.py's
    # latent broadcast and generate_and_sync_exit_flag.
    set_seed(args.seed, device_specific=True)

    disable_tf32_matmul(verbose=accelerator.is_main_process)

    return accelerator


def _transformer_cls_for_model(model_id: str):
    return Cosmos3OmniTransformer if is_cosmos3_model(model_id) else WanTransformer3DModel


def _transformer_from_config(transformer_cls, model_id: str, revision: str | None = None):
    """Materialize a randomly initialized transformer from the hub config at
    the pin-registry revision.

    ``from_config`` ignores ``torch_dtype`` and always builds fp32 modules —
    FSDP2 keeps the fp32 masters and computes in bf16 — so no dtype kwarg
    belongs here.
    """
    resolved_revision = revision if revision is not None else hf_revision_for_model(model_id)
    return transformer_cls.from_config(
        model_id,
        subfolder="transformer",
        revision=resolved_revision,
    )


def _vae_for_model(
    model_id: str,
    device: torch.device | str,
    revision: str | None = None,
) -> AutoencoderKLWan:
    """Load the VAE at the pin-registry revision onto ``device``."""
    resolved_revision = revision if revision is not None else hf_revision_for_model(model_id)
    return AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        revision=resolved_revision,
    ).to(device)


def _scheduler_for_model(model_id: str, revision: str | None = None) -> UniPCMultistepScheduler:
    if is_cosmos3_model(model_id):
        return UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=flow_shift_for_model(model_id),
        )

    resolved_revision = revision if revision is not None else hf_revision_for_model(model_id)
    return UniPCMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        torch_dtype=torch.bfloat16,
        revision=resolved_revision,
    )


def _save_replicated_state_dict_to_dcp(state_dict: dict[str, torch.Tensor], cache_path: Path) -> None:
    dcp.save(
        {"model": state_dict},
        storage_writer=dcp.FileSystemWriter(str(cache_path)),
        no_dist=True,
    )


def _load_replicated_state_dict_from_dcp(state_dict: dict[str, torch.Tensor], cache_path: Path) -> None:
    dcp.load(
        {"model": state_dict},
        storage_reader=dcp.FileSystemReader(str(cache_path)),
        no_dist=True,
    )


def _ensure_dcp_cache(model_id: str, dcp_cache_dir: Path) -> Path:
    """Ensure a DCP-format cache of the HF transformer weights exists, creating it if needed.

    Rank zero builds missing weights and broadcasts completion or failure to peers.
    Returns the path to the DCP cache directory.
    """
    # Keyed by the pin fingerprint so a pin bump stops resolving to the old conversion.
    cache_path = dcp_cache_dir / dcp_cache_dir_name(model_id) / "transformer"
    marker = cache_path / ".dcp_complete"

    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    error = [None]
    if rank == 0:
        try:
            if not marker.exists():
                print(f"DCP cache not found at {cache_path}, converting from HF pretrained weights...")
                transformer_cls = _transformer_cls_for_model(model_id)
                transformer, loading_info = transformer_cls.from_pretrained(
                    model_id,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                    local_files_only="HF_HUB_OFFLINE" in os.environ and os.environ["HF_HUB_OFFLINE"] == "1",
                    revision=hf_revision_for_model(model_id),
                    output_loading_info=True,
                )
                missing_keys = sorted(loading_info.get("missing_keys", ()))
                if missing_keys:
                    raise RuntimeError(
                        f"{model_id} at revision {hf_revision_for_model(model_id)} does not ship "
                        f"{len(missing_keys)} transformer key(s) its config declares; HF random-filled them on "
                        f"this rank and the DCP cache would freeze that draw for this cluster: {missing_keys[:8]}"
                    )
                cache_path.mkdir(parents=True, exist_ok=True)
                _save_replicated_state_dict_to_dcp(transformer.state_dict(), cache_path)
                marker.touch()
                del transformer
                torch.cuda.empty_cache()
                print(f"DCP cache saved to {cache_path}")
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    # Every rank participates even if it observed the completion marker at a
    # different time. Also release peers when rank zero cannot build the cache.
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"DCP cache preparation failed: {error[0]}")

    return cache_path


def get_pipe(
    args: argparse.Namespace,
    load_pretrained_transformer_weights: bool,
    frames_per_block: int | None,
    device: torch.device | str,
) -> ArtifixerPipeline | ArtifixerCosmos3Pipeline:
    attention_backend = configure_attention(device, attention_backend=args.attention_backend)
    model_revision = getattr(args, "model_revision", None)
    scheduler = _scheduler_for_model(args.model_id, model_revision)
    transformer = _transformer_from_config(
        _transformer_cls_for_model(args.model_id),
        args.model_id,
        model_revision,
    )

    if load_pretrained_transformer_weights:
        dcp_path = _ensure_dcp_cache(args.model_id, args.dcp_cache_dir)
        state_dict = transformer.state_dict()
        _load_replicated_state_dict_from_dcp(state_dict, dcp_path)
        transformer.load_state_dict(state_dict)

    vae = _vae_for_model(args.model_id, device, model_revision)

    if is_cosmos3_model(args.model_id):
        pipe = ArtifixerCosmos3Pipeline(
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
            gradient_checkpointing=args.gradient_checkpointing,
            attention_backend=attention_backend,
            # None keeps the pipeline bidirectional; set, it arms the pipeline's diffusion-forcing contract.
            frames_per_block=frames_per_block,
        )
        validate_model_profile(args.model_id, pipe)
        return pipe

    pipe = ArtifixerPipeline(
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
        tokenizer=None,
        text_encoder=None,
        default_negative_prompt_path=args.default_negative_prompt_path,
        frames_per_block=frames_per_block,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_every_n_blocks=args.checkpoint_every_n_blocks,
        expand_timesteps=uses_expanded_timesteps(args.model_id),
        attention_backend=attention_backend,
    )
    validate_model_profile(args.model_id, pipe)
    return pipe


def get_kv_cache_pipe(
    args: argparse.Namespace,
    device: torch.device | str,
    *,
    truncate_rollout_gradients: bool = False,
) -> ArtifixerKvCachePipeline | ArtifixerCosmos3KvCachePipeline:
    """Build the family's KV-cache rollout pipeline for args.model_id.

    args must carry frames_per_block, local_attn_size, and sink_size; the
    scheduler shift comes from the model profile. truncate_rollout_gradients
    is a training option (inert under no_grad), so only distillation passes it.
    """
    attention_backend = configure_attention(device, attention_backend=args.attention_backend)
    resolved_flow_shift = flow_shift_for_model(args.model_id)
    model_revision = getattr(args, "model_revision", None)
    vae = _vae_for_model(args.model_id, device, model_revision)
    if is_cosmos3_model(args.model_id):
        pipe = ArtifixerCosmos3KvCachePipeline(
            vae=vae,
            transformer=_transformer_from_config(
                Cosmos3OmniTransformer,
                args.model_id,
                model_revision,
            ),
            gradient_checkpointing=args.gradient_checkpointing,
            attention_backend=attention_backend,
            flow_shift=resolved_flow_shift,
            ar_frames_per_chunk=args.frames_per_block,
            local_attn_size=args.local_attn_size,
            sink_size=args.sink_size,
            truncate_rollout_gradients=truncate_rollout_gradients,
        )
        validate_model_profile(args.model_id, pipe)
        return pipe

    pipe = ArtifixerKvCachePipeline(
        vae=vae,
        transformer=_transformer_from_config(
            WanTransformer3DModel,
            args.model_id,
            model_revision,
        ),
        tokenizer=None,
        text_encoder=None,
        ar_frames_per_chunk=args.frames_per_block,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_every_n_blocks=args.checkpoint_every_n_blocks,
        expand_timesteps=uses_expanded_timesteps(args.model_id),
        attention_backend=attention_backend,
        flow_shift=resolved_flow_shift,
        truncate_rollout_gradients=truncate_rollout_gradients,
    )
    validate_model_profile(args.model_id, pipe)
    return pipe


def get_train_dataloader(
    args: argparse.Namespace, accelerator: Accelerator, frames_per_block: int | None
) -> torch.utils.data.DataLoader:
    # num_fetches covers every DataLoader fetch across the full training run:
    # max_iterations optimizer steps × gradient_accumulation_steps fetches/optimizer_step ×
    # num_processes ranks. accelerate.prepare() shards this into num_fetches / num_processes
    # items per rank, so each rank sees exactly one DataLoader epoch. With context
    # parallelism, ranks in a CP group must train on the same source video; the dataset maps
    # adjacent shard indices to one logical CP-group fetch without wrap-around reuse.
    assert accelerator.num_processes % args.context_parallel_size == 0, (
        f"num_processes {accelerator.num_processes} must be divisible by context_parallel_size "
        f"{args.context_parallel_size}"
    )
    num_fetches = args.max_iterations * args.gradient_accumulation_steps * accelerator.num_processes
    train_dataset = DL3DVPairedDatasetTrain(
        split="trainval",
        split_path=args.split_path,
        dl3dv_dir=args.dl3dv_dir,
        prompt_dir=args.prompt_dir,
        num_frames=args.num_frames,
        frames_per_block=frames_per_block,
        num_fetches=num_fetches,
        dataset_scaling_factor=args.dataset_scaling_factor,
        verbose=accelerator.is_main_process,
        resize_multiple=spatial_resize_multiple(args.model_id),
        prompt_format=prompt_format_for_model(args.model_id),
        context_parallel_size=args.context_parallel_size,
        vae_chunk_halo_latent_frames=args.vae_chunk_halo_latent_frames,
        camera_pose_alignment=args.camera_pose_alignment,
    )

    # Use accelerator.device for the distributed broadcast path (NCCL requires GPU tensors);
    # DL3DVPairedDatasetTrain stores CPU copies before DataLoader workers read them.
    train_dataset.set_random_splits(accelerator.device, seed=args.seed)
    # Keep DataLoader order fixed so Accelerate's per-rank shards are adjacent
    # within each CP group. Randomness comes from iter_plan and data_perm above.
    return torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        worker_init_fn=_seed_dataloader_worker,
    )


def _seed_dataloader_worker(worker_id: int) -> None:
    """Seed numpy and the python random module inside each dataloader worker.

    torch seeds its per-worker RNG automatically from the DataLoader's ``generator`` argument;
    this function extends that determinism to numpy and stdlib random, which the default
    worker_init_fn does not touch. This keeps current and future dataloader-side
    randomness deterministic.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_val_datasets(
    args: argparse.Namespace,
    accelerator: Accelerator,
    num_frames: int | None,
    frames_per_block: int | None,
) -> dict[int, DL3DVPairedDatasetTest]:
    manifest_path = getattr(args, "validation_manifest", None)
    manifest = json.loads(manifest_path.read_text()) if manifest_path is not None else None
    if manifest is not None:
        from model_training.utils.validation_artifacts import validate_manifest

        validate_manifest(manifest)
        num_frames = manifest["num_frames"]
        for key in ("dataset_scaling_factor", "camera_pose_alignment"):
            if getattr(args, key) != manifest[key]:
                raise ValueError(f"Validation manifest {key}={manifest[key]!r} differs from the run")
        requested = getattr(args, "validation_num_frames", None)
        if requested is not None and requested != num_frames:
            raise ValueError("--validation_num_frames differs from the manifest")
        if args.test_panel_scenes not in (0, len(manifest["samples"])):
            raise ValueError("--test_panel_scenes must match the manifest sample count")
        args.test_panel_scenes = len(manifest["samples"])
    elif getattr(args, "validation_num_frames", None) is not None:
        num_frames = args.validation_num_frames
    elif getattr(args, "validate_only", False):
        num_frames = args.num_frames

    # The three variants differ only in num_views, and most of the
    # construction cost is the rank-0 scene scan and broadcast, so build one
    # dataset and shallow-copy it for the other view counts.
    base = DL3DVPairedDatasetTest(
        split="test",
        split_path=args.split_path,
        dl3dv_dir=args.dl3dv_dir,
        prompt_dir=args.prompt_dir,
        num_frames=num_frames,
        frames_per_block=frames_per_block,
        num_views=3,
        dataset_scaling_factor=args.dataset_scaling_factor,
        validation_seed=VALIDATION_SEED,
        verbose=accelerator.is_main_process,
        resize_multiple=spatial_resize_multiple(args.model_id),
        prompt_format=prompt_format_for_model(args.model_id),
        camera_pose_alignment=args.camera_pose_alignment,
        validation_manifest=manifest,
        capture_validation_metadata=getattr(args, "validation_metrics", True) or manifest is not None,
    )

    val_datasets = {}
    for num_neighbors in (3, 6, 12):
        val_datasets[num_neighbors] = copy.copy(base)
        val_datasets[num_neighbors].num_views = num_neighbors

    return val_datasets


def run_validation_only(args, accelerator, pipe, *, frames_per_block, context_parallel=False):
    """Shard and load only the evaluated transformer. No training loader, optimizer or critics."""
    from model_eval.validation import EvaluationRunner
    from model_eval.validation_config import EvaluationConfig, EvaluationContext, EvaluationIdentity
    from model_eval.validation_reporting import validation_media
    from model_eval.validation_tasks import build_evaluation_plan

    pipe.transformer.requires_grad_(False)
    barrier_if_distributed()
    if accelerator.is_fsdp2:
        # The model-only FSDP2 path also configures per-block wrapping before sharding.
        (pipe.transformer,) = accelerator._prepare_fsdp2(pipe.transformer)
    else:
        pipe.transformer = accelerator.prepare_model(pipe.transformer)
    load_model_weights_from_dcp(pipe.transformer, args.base_checkpoint_dir)
    if context_parallel:
        enable_training_context_parallel(pipe, args.context_parallel_size, accelerator.device.type)
    val_datasets = get_val_datasets(args, accelerator, args.num_frames, frames_per_block)
    config = EvaluationConfig.from_args(args)
    context = EvaluationContext(accelerator.device, accelerator.print)
    runner = EvaluationRunner(pipe, config, context)
    plan = build_evaluation_plan(val_datasets, config, 0)
    identity = EvaluationIdentity(
        project_dir=args.project_dir,
        checkpoint=args.base_checkpoint_dir,
        initial_checkpoint=args.base_checkpoint_dir,
        weights="loaded_checkpoint",
    )
    save_dir = args.project_dir / "val" / "step_000000"
    result = runner.run(plan, save_dir, identity)
    if uses_wandb_logging(args.log_with):
        values = dict(result.scalars)
        if context.is_main_process:
            values.update(validation_media(plan, save_dir))
        accelerator.log(values, step=0)
    barrier_if_distributed()
    return result


def run_supervised_training(
    args: argparse.Namespace,
    frames_per_block: int | None,
    base_checkpoint_dir: Path | None,
    *,
    context_parallel: bool = False,
    trainer_factory: Callable | None = None,
) -> None:
    """Run supervised training for stage 1 (train.py)
    and stage 2 (diffusion_forcing.py).  The stages differ in the block-causal
    settings, in whether a base checkpoint is loaded, and — for the
    conditioned stage — in the context-parallel setup and trainer class."""
    run_id, should_resume = get_run_id_and_should_resume(args)
    if context_parallel:
        # CP may set args.shard_size, so this must run before get_accelerator
        # builds the FSDP config.
        configure_context_parallel_shard_size(args)
    accelerator = get_accelerator(args, run_id)
    pipe = get_pipe(args, base_checkpoint_dir is None and not should_resume, frames_per_block, accelerator.device)

    if context_parallel:
        latent_num_frames = validate_latent_frame_layout(
            num_frames=args.num_frames,
            temporal_scale=pipe.vae.config.scale_factor_temporal,
            context_parallel_size=args.context_parallel_size,
        )
        pipe.validate_cp_size(args.context_parallel_size, latent_num_frames)

    if frames_per_block is not None:
        latent_num_frames = (args.num_frames - 1) // pipe.vae.config.scale_factor_temporal + 1
        validate_latent_frames_divisible(latent_num_frames, frames_per_block, "frames_per_block")

    if getattr(args, "validate_only", False):
        return run_validation_only(
            args, accelerator, pipe, frames_per_block=frames_per_block, context_parallel=context_parallel
        )

    optimizer = torch.optim.AdamW(
        pipe.transformer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    train_dataloader = get_train_dataloader(args, accelerator, frames_per_block)

    # Keep ranks aligned after model and dataloader setup; large model loading can
    # otherwise leave faster ranks waiting inside distributed preparation.
    barrier_if_distributed()

    transformer, optimizer, train_dataloader, scheduler = accelerator.prepare(
        pipe.transformer, optimizer, train_dataloader, scheduler
    )

    # Should only load state after prepare is called
    resume_state = ResumeState()
    if should_resume:
        train_dataloader, resume_state = resume_training_from_checkpoint(args, accelerator, train_dataloader)
    elif base_checkpoint_dir is not None:
        load_model_weights_from_dcp(transformer, base_checkpoint_dir)

    pipe.transformer = transformer
    if context_parallel:
        # CP wraps the transformer, so it must follow accelerator.prepare.
        enable_training_context_parallel(pipe, args.context_parallel_size, accelerator.device.type)

    maybe_write_run_id(accelerator, args.project_dir, run_id, args.log_with)

    # Build validation datasets after loading the checkpoint so validation uses
    # the final split/model configuration.
    val_datasets = get_val_datasets(args, accelerator, args.num_frames, None)

    if trainer_factory is None:
        # Imported here to avoid a circular import (trainer_base imports this module).
        from model_training.trainers.trainer import Trainer

        trainer_factory = Trainer

    trainer = trainer_factory(
        args,
        accelerator,
        optimizer,
        scheduler,
        train_dataloader,
        val_datasets,
        pipe,
        frames_per_block,
        resume_state.step_offset,
    )
    trainer.train()


def maybe_write_run_id(accelerator: Accelerator, project_dir: Path, run_id: str | None, log_with: str) -> None:
    if uses_wandb_logging(log_with) and run_id is None and accelerator.is_main_process:
        project_dir.mkdir(parents=True, exist_ok=True)
        run_id = accelerator.get_tracker("wandb").run.id
        with open(project_dir / "run_id.txt", "w") as f:
            f.write(run_id)
