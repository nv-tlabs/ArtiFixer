# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate import Accelerator, DataLoaderConfiguration, FullyShardedDataParallelPlugin, ParallelismConfig
from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration, set_seed
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
from torch.distributed.fsdp import MixedPrecisionPolicy

from model_training.data.dl3dv_test import DL3DVPairedDatasetTest
from model_training.data.dl3dv_train import DL3DVPairedDatasetTrain
from model_training.net.transformer import _DEFAULT_ATTENTION_BACKEND
from model_training.pipeline.kv_cache_pipeline import ArtifixerKvCachePipeline
from model_training.pipeline.pipeline import ArtifixerPipeline

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"

_CHECKPOINT_COMPLETE_FILENAME = "checkpoint_complete"
_LOG_WITH_WANDB = "wandb"
_LOG_WITH_NONE = "none"
_LOG_WITH_CHOICES = (_LOG_WITH_WANDB, _LOG_WITH_NONE)


@dataclass(frozen=True)
class ResumeState:
    checkpoint_step: int = 0
    step_offset: int = 0


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


def tracker_logging_enabled(log_with: str) -> bool:
    assert log_with in _LOG_WITH_CHOICES, f"log_with must be one of {_LOG_WITH_CHOICES}, got {log_with!r}"
    return log_with == _LOG_WITH_WANDB


def uses_wandb_logging(log_with: str) -> bool:
    return tracker_logging_enabled(log_with)


def read_wandb_run_id(project_dir: Path) -> str:
    run_id_path = project_dir / "run_id.txt"
    if not run_id_path.is_file():
        raise FileNotFoundError(f"Cannot resume W&B logging without run id file: {run_id_path}")
    run_id = run_id_path.read_text().strip()
    if not run_id:
        raise ValueError(f"W&B run id file is empty: {run_id_path}")
    return run_id


def get_run_id_and_should_resume(args: argparse.Namespace) -> tuple[str | None, bool]:
    run_id = None
    should_resume = False
    if uses_wandb_logging(args.log_with) and (args.project_dir / "run_id.txt").is_file():
        run_id = read_wandb_run_id(args.project_dir)

    if args.resume_from_checkpoint is not None:
        auto_resume = args.resume_from_checkpoint == "auto"
        if (not auto_resume) or (args.project_dir / "checkpoints").exists():
            should_resume = True
            if uses_wandb_logging(args.log_with) and run_id is None:
                run_id = read_wandb_run_id(args.project_dir)

    return run_id, should_resume


def load_checkpoint_state(args: argparse.Namespace, accelerator: Accelerator) -> int:
    if args.resume_from_checkpoint != "auto":
        checkpoint_step = _checkpoint_step_from_path(args.resume_from_checkpoint)
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
    return train_dataloader, ResumeState(checkpoint_step=checkpoint_step, step_offset=step_offset)


def barrier_if_distributed(group: dist.ProcessGroup | None = None) -> None:
    if dist.is_initialized():
        dist.barrier(group=group)


def load_model_weights_from_dcp(model: torch.nn.Module, checkpoint_dir: Path | str) -> None:
    state_dict = model.state_dict()
    dcp.load({"model": state_dict}, dcp.FileSystemReader(str(checkpoint_dir)))
    model.load_state_dict(state_dict)


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
        transformer_cls_names_to_wrap=["ArtifixerTransformerBlock"],
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

    accelerator_log_with = _LOG_WITH_WANDB if uses_wandb_logging(args.log_with) else None

    accelerator = Accelerator(
        project_config=ProjectConfiguration(
            project_dir=args.project_dir,
            automatic_checkpoint_naming=True,
            total_limit=args.max_checkpoints,
        ),
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
        mixed_precision=args.mixed_precision,
        gradient_accumulation_plugin=grad_acc_plugin,
        fsdp_plugin=fsdp_plugin,
        parallelism_config=parallelism_config,
        log_with=accelerator_log_with,
    )

    init_kwargs = dict()

    if uses_wandb_logging(args.log_with):
        wandb_init_kwargs = {
            "name": (args.tracker_run_name if args.tracker_run_name else args.project_dir.name),
            "dir": args.project_dir / "wandb",
        }
        if run_id is not None:
            wandb_init_kwargs["id"] = run_id
            wandb_init_kwargs["resume"] = "allow"

        init_kwargs["wandb"] = wandb_init_kwargs

    if accelerator_log_with is not None:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)), init_kwargs=init_kwargs)

    # Seed after Accelerator() so process_index is available. device_specific=True gives
    # each rank args.seed + process_index so dropout/noise are decorrelated across DP
    # ranks (training runs pure DP+FSDP, not CP). CP paths that require rank-identical
    # RNG resolve divergence explicitly via dist.broadcast — see e.g. pipeline_base.py's
    # latent broadcast and kv_cache_pipeline._generate_and_sync_exit_flag.
    set_seed(args.seed, device_specific=True)

    return accelerator


def _ensure_dcp_cache(model_id: str, dcp_cache_dir: Path) -> Path:
    """Ensure a DCP-format cache of the HF transformer weights exists, creating it if needed.

    Only rank 0 performs the conversion; other ranks wait at a barrier.
    Returns the path to the DCP cache directory.
    """
    cache_path = dcp_cache_dir / model_id.replace("/", "--") / "transformer"
    marker = cache_path / ".dcp_complete"

    if marker.exists():
        return cache_path

    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        print(f"DCP cache not found at {cache_path}, converting from HF pretrained weights...")
        transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only="HF_HUB_OFFLINE" in os.environ and os.environ["HF_HUB_OFFLINE"] == "1",
        )
        cache_path.mkdir(parents=True, exist_ok=True)
        dcp.save({"model": transformer.state_dict()}, dcp.FileSystemWriter(str(cache_path)))
        marker.touch()
        del transformer
        torch.cuda.empty_cache()
        print(f"DCP cache saved to {cache_path}")

    barrier_if_distributed()

    return cache_path


def get_pipe(
    args: argparse.Namespace,
    load_pretrained_transformer_weights: bool,
    frames_per_block: int | None,
    device: torch.device | str,
) -> ArtifixerPipeline:
    scheduler = UniPCMultistepScheduler.from_pretrained(
        args.model_id, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    transformer = WanTransformer3DModel.from_config(args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)

    if load_pretrained_transformer_weights:
        dcp_path = _ensure_dcp_cache(args.model_id, args.dcp_cache_dir)
        state_dict = transformer.state_dict()
        dcp.load({"model": state_dict}, dcp.FileSystemReader(str(dcp_path)))
        transformer.load_state_dict(state_dict)

    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.bfloat16).to(device)

    return ArtifixerPipeline(
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
        tokenizer=None,
        text_encoder=None,
        default_negative_prompt_path=args.default_negative_prompt_path,
        frames_per_block=frames_per_block,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_every_n_blocks=args.checkpoint_every_n_blocks,
        attention_backend=args.attention_backend if args.attention_backend is not None else _DEFAULT_ATTENTION_BACKEND,
    )


def get_kv_cache_pipe(args: argparse.Namespace, device: torch.device | str) -> ArtifixerKvCachePipeline:
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.bfloat16).to(device)

    return ArtifixerKvCachePipeline(
        vae=vae,
        transformer=WanTransformer3DModel.from_config(
            args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        ).to(device),
        tokenizer=None,
        text_encoder=None,
        frames_per_block=args.frames_per_block,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_every_n_blocks=args.checkpoint_every_n_blocks,
        attention_backend=args.attention_backend if args.attention_backend is not None else _DEFAULT_ATTENTION_BACKEND,
    )


def get_train_dataloader(
    args: argparse.Namespace, accelerator: Accelerator, frames_per_block: int | None
) -> torch.utils.data.DataLoader:
    # num_fetches covers every DataLoader fetch across the full training run:
    # max_iterations optimizer steps × gradient_accumulation_steps fetches/optimizer_step ×
    # num_processes ranks. accelerate.prepare() shards this into num_fetches / num_processes
    # items per rank, so each rank sees exactly one DataLoader epoch and each iter_plan slot
    # is consumed exactly once (no wrap-around reuse of the same (split, num_neighbors,
    # scene) triple).
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
    )

    train_dataset.set_random_splits(accelerator.device, seed=args.seed)
    # Rank-identical generator by design: all ranks produce the same shuffle permutation;
    # accelerate.prepare() then shards it so each rank sees a disjoint subset. Also seeds
    # the per-worker torch RNG (each worker gets base_seed + worker_id). Model-side RNG
    # diversification across ranks is handled separately by set_seed(device_specific=True).
    return torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
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
    val_datasets = {}
    for neighbor_idx, num_neighbors in enumerate([3, 6, 12]):
        val_datasets[num_neighbors] = DL3DVPairedDatasetTest(
            split="test",
            split_path=args.split_path,
            dl3dv_dir=args.dl3dv_dir,
            prompt_dir=args.prompt_dir,
            num_frames=num_frames,
            frames_per_block=frames_per_block,
            num_views=num_neighbors,
            start_index=0,
            dataset_scaling_factor=args.dataset_scaling_factor,
            validation_seed=42,
            verbose=accelerator.is_main_process and neighbor_idx == 0,
        )

    return val_datasets


def maybe_write_run_id(accelerator: Accelerator, project_dir: Path, run_id: str | None, log_with: str) -> None:
    if uses_wandb_logging(log_with) and run_id is None and accelerator.is_main_process:
        project_dir.mkdir(parents=True, exist_ok=True)
        run_id = accelerator.get_tracker("wandb").run.id
        with open(project_dir / "run_id.txt", "w") as f:
            f.write(run_id)


def add_required_data_path_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split_path", type=Path, required=True)
    parser.add_argument("--dl3dv_dir", type=Path, required=True)
    parser.add_argument("--prompt_dir", type=Path, required=True)


def add_common_runtime_opts(parser: argparse.ArgumentParser, include_num_frames: bool = True) -> None:
    parser.add_argument("--dataset_scaling_factor", default=0.01, type=float)

    parser.add_argument("--log_with", default=_LOG_WITH_WANDB, choices=_LOG_WITH_CHOICES, type=str)
    parser.add_argument("--tracker_project_name", default="artifixer", type=str)
    parser.add_argument("--tracker_run_name", default=None, type=str)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)

    # Available models: Wan-AI/Wan2.1-T2V-1.3B-Diffusers, Wan-AI/Wan2.1-T2V-14B-Diffusers
    parser.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-14B-Diffusers", type=str)
    parser.add_argument(
        "--dcp_cache_dir",
        default=Path.home() / ".cache" / "artifixer" / "dcp_weights",
        type=Path,
        help="Directory to cache DCP-format transformer weights (converted from HF pretrained on first use)",
    )

    parser.add_argument(
        "--default_negative_prompt_path",
        default=Path(os.path.dirname(os.path.realpath(__file__))).parent.parent / "default_negative_prompt.pt",
        type=Path,
    )
    if include_num_frames:
        parser.add_argument("--num_frames", default=81, type=int)
    parser.add_argument("--dropout_rate", default=0.1, type=float)

    parser.add_argument(
        "--attention_backend",
        default=None,
        type=str,
        help="Diffusers attention backend override. When omitted, auto-selects based on "
        "GPU arch: cuDNN SDPA on A100, native flash (FA3) on H100, native flash "
        "(FA4) on GB200. Valid values include '_native_flash', 'native', 'flash', 'flex'.",
    )

    parser.add_argument("--max_checkpoints", default=5, type=int)

    parser.add_argument("--mixed_precision", default="bf16", type=str)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--shard_size", default=None, type=int)
    parser.add_argument(
        "--max_validation_task_groups",
        default=None,
        type=int,
        help=(
            "Maximum number of CP groups that run full validation tasks concurrently. "
            "Remaining groups run minimal dummy forwards so FSDP collectives stay aligned."
        ),
    )
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--checkpoint_every_n_blocks", default=1, type=int)

    parser.add_argument("--dataloader_num_workers", default=8, type=int)
    parser.add_argument("--dataloader_prefetch_factor", default=4, type=int)

    # Used to signal to the training process that it should save a checkpoint and exit with timeout signal for requeue
    parser.add_argument("--should_checkpoint_flag", default=None, type=Path)

    parser.add_argument("--check_state_finite", action="store_true")

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Base RNG seed. Applied via accelerate.utils.set_seed(device_specific=True) at startup, "
        "and threaded into the train DataLoader generator and DL3DVPairedDatasetTrain.set_random_splits. "
        "Validation dataloaders remain pinned to 42 for cross-run metric comparability.",
    )


def get_common_opts() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_required_data_path_opts(parser)
    add_common_runtime_opts(parser)
    return parser


def get_eval_common_opts() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_runtime_opts(parser, include_num_frames=False)
    return parser
