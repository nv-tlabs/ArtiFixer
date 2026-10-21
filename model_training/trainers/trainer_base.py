# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import gc
import shutil
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch import nn
from tqdm import tqdm

from model_eval.validation import EvaluationRunner
from model_eval.validation_config import EvaluationConfig, EvaluationContext, EvaluationIdentity
from model_eval.validation_reporting import validation_media
from model_eval.validation_tasks import build_evaluation_plan
from model_training.net.cosmos3_transformer import conditioning_adapter_weight_norms, scope_to_query_ratios
from model_training.net.scope import scope_statistics
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.prompt_formats import prompt_payload_for_text_dropout
from model_training.utils.train_utils import (
    barrier_if_distributed,
    checkpoint_is_complete,
    mark_checkpoint_complete,
    uses_wandb_logging,
)


class TrainerBase(ABC):

    def __init__(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        pipe: ArtifixerPipelineBase,
        optimizers: list[torch.optim.Optimizer],
        schedulers: list[torch.optim.lr_scheduler.LRScheduler],
        train_dataloader: torch.utils.data.DataLoader,
        val_datasets: dict[int, torch.utils.data.Dataset],
        step_offset: int = 0,
    ):
        self.args = args
        self.accelerator = accelerator
        self.pipe = pipe
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.train_dataloader = train_dataloader
        self.val_datasets = val_datasets
        self.step_offset = step_offset
        self._zero_encode_cache: dict[tuple, torch.Tensor] = {}
        self.evaluation = EvaluationRunner(
            pipe, EvaluationConfig.from_args(args), EvaluationContext(accelerator.device, accelerator.print)
        )

    @abstractmethod
    def training_batch(self, data: dict) -> tuple[torch.Tensor, dict]:
        pass

    @abstractmethod
    def to_accumulate(self) -> list[nn.Module]:
        pass

    @abstractmethod
    def on_before_optimizer_step(self) -> None:
        pass

    @abstractmethod
    def on_before_zero_grad(self) -> None:
        pass

    def _prompt_kwargs(self, data: dict) -> dict[str, torch.Tensor]:
        """Positive-prompt transformer kwargs; the attention mask exists only for token-id prompts."""
        prompt_kwargs = {"encoder_hidden_states": data["encoded_prompt"]}
        if "prompt_attention_mask" in data:
            prompt_kwargs["encoder_attention_mask"] = data["prompt_attention_mask"]
        return prompt_kwargs

    @staticmethod
    def _neighbor_camera_kwargs(neighbor_camera_rays: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Only joint data supplies reference rays; omit the keyword for legacy/Wan callers."""
        return {} if neighbor_camera_rays is None else {"neighbor_camera_rays": neighbor_camera_rays}

    def _should_checkpoint_for_timeout(self) -> bool:
        if self.args.should_checkpoint_flag is None:
            return False

        # Slurm pre-timeout handlers touch this flag so the job can save a
        # checkpoint before exiting with the requeue-friendly timeout code.
        should_checkpoint = self.args.should_checkpoint_flag.exists()
        if dist.is_initialized():
            # If any rank sees the flag, every rank must enter the checkpoint
            # path together; otherwise distributed collectives can diverge.
            should_checkpoint_tensor = torch.tensor(
                [int(should_checkpoint)],
                device=self.accelerator.device,
                dtype=torch.int32,
            )
            dist.all_reduce(should_checkpoint_tensor, op=dist.ReduceOp.MAX)
            should_checkpoint = bool(should_checkpoint_tensor.item())

        return should_checkpoint

    def _save_state_with_commit_marker(self, step: int) -> None:
        project_dir = Path(self.accelerator.project_dir)
        checkpoint_dir = project_dir / "checkpoints" / f"checkpoint_{step}"
        if checkpoint_is_complete(project_dir, step, self.accelerator):
            return

        if self.accelerator.is_main_process and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        barrier_if_distributed()

        self.accelerator.project_configuration.iteration = step

        local_save_succeeded = True
        caught_error = None
        try:
            self.accelerator.save_state()
        except Exception as e:
            local_save_succeeded = False
            caught_error = e
            self.accelerator.print(
                f"Checkpoint save failed on rank {self.accelerator.process_index} at iteration {step}:\n"
                f"{traceback.format_exc()}"
            )

        if dist.is_initialized():
            save_succeeded_tensor = torch.tensor(
                [int(local_save_succeeded)],
                device=self.accelerator.device,
                dtype=torch.int32,
            )
            dist.all_reduce(save_succeeded_tensor, op=dist.ReduceOp.MIN)
            save_succeeded = bool(save_succeeded_tensor.item())
        else:
            save_succeeded = local_save_succeeded

        if not save_succeeded:
            checkpoint_complete = checkpoint_is_complete(project_dir, step, self.accelerator)
            if self.accelerator.is_main_process and not checkpoint_complete:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            barrier_if_distributed()
            if caught_error is not None:
                raise caught_error
            raise RuntimeError(f"Distributed checkpoint save failed at iteration {step}")

        barrier_if_distributed()
        mark_checkpoint_complete(project_dir, step, self.accelerator)
        barrier_if_distributed()
        self._prune_completed_checkpoints(step)
        barrier_if_distributed()

    def _prune_completed_checkpoints(self, saved_step: int) -> None:
        limit = getattr(self.args, "max_checkpoints", None)
        if not self.accelerator.is_main_process or limit is None:
            return
        checkpoints_dir = Path(self.accelerator.project_dir) / "checkpoints"
        # Leave incomplete saves and unrelated entries alone. Only this
        # completed replacement authorizes removal of older complete saves.
        try:
            completed = sorted(
                (
                    (int(path.name.removeprefix("checkpoint_")), path)
                    for path in checkpoints_dir.iterdir()
                    if path.name.startswith("checkpoint_")
                    and path.name.isascii()
                    and path.name.removeprefix("checkpoint_").isdigit()
                    and not path.is_symlink()
                    and (path / "checkpoint_complete").is_file()
                ),
                key=lambda entry: entry[0],
            )
            if any(step > saved_step for step, _ in completed):
                self.accelerator.print("Skipping checkpoint retention: newer completed checkpoints already exist")
                return
            for _, path in completed[:-limit]:
                shutil.rmtree(path)
        except OSError as error:
            # The new checkpoint is already committed. A retention failure
            # must not strand peers at the following barrier or revoke it.
            self.accelerator.print(f"Checkpoint saved, but pruning older checkpoints failed: {error}")

    def _save_timeout_checkpoint_and_ack(self, step: int) -> None:
        project_dir = Path(self.accelerator.project_dir)
        if checkpoint_is_complete(project_dir, step, self.accelerator):
            raise RuntimeError(f"Timeout checkpoint_{step} is already complete; refusing to acknowledge a stale save")

        self._save_state_with_commit_marker(step)

        local_ack_succeeded = True
        caught_error = None
        if self.accelerator.is_main_process:
            try:
                request_flag = Path(self.args.should_checkpoint_flag)
                request_token = request_flag.read_text().strip().replace("\n", " ")
                if not request_token:
                    # An empty token cannot be bound to a requeue
                    # authorization, so refuse it instead of writing an ack
                    # the launcher's gate cannot validate.
                    raise RuntimeError(
                        f"Checkpoint request flag {request_flag} carries no request token; "
                        "the launcher must write a unique token into the flag (see "
                        "model_training/slurm/lib/requeue_protocol.sh)."
                    )
                ack_path = Path(f"{request_flag}.complete")
                tmp_path = ack_path.with_name(f".{ack_path.name}.tmp")
                tmp_path.write_text(f"request={request_token}\nstep={step}\n")
                tmp_path.replace(ack_path)
            except Exception as e:
                local_ack_succeeded = False
                caught_error = e

        if dist.is_initialized():
            ack_succeeded_tensor = torch.tensor(
                [int(local_ack_succeeded)],
                device=self.accelerator.device,
                dtype=torch.int32,
            )
            dist.all_reduce(ack_succeeded_tensor, op=dist.ReduceOp.MIN)
            ack_succeeded = bool(ack_succeeded_tensor.item())
        else:
            ack_succeeded = local_ack_succeeded

        if not ack_succeeded:
            if caught_error is not None:
                raise caught_error
            raise RuntimeError(f"Failed to acknowledge timeout checkpoint_{step}")

        barrier_if_distributed()

    def _encode_zeros_cached(self, reference: torch.Tensor, *, neighbors: bool) -> torch.Tensor:
        """Encode an all-zeros input shaped like ``reference`` through the frozen VAE, with caching.

        The argmax-mode encoding of a zeros input is deterministic per shape
        (and is not itself zeros), so it only needs to be computed once.
        """
        key = (neighbors, tuple(reference.shape), reference.dtype, reference.device)
        cached = self._zero_encode_cache.get(key)
        if cached is None:
            zeros = torch.zeros_like(reference)
            cached = self.pipe.encode_neighbors(zeros) if neighbors else self.pipe.encode_video_frames(zeros)
            self._zero_encode_cache[key] = cached
        return cached

    def _sample_input_dropout(
        self,
        num_frames: int,
        *,
        device: torch.device | None = None,
        is_source_rank: bool = True,
        broadcast: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[bool, int, bool, bool]:
        """Choose conditioning inputs to drop, optionally synchronizing the choices across ranks."""
        dropout_probs = torch.rand(3, device=device) if is_source_rank else torch.empty(3, device=device)
        if broadcast is not None:
            broadcast(dropout_probs)
        drop_rendered, drop_text, ignore_neighbors = (dropout_probs < self.args.dropout_rate).tolist()
        frames_to_drop = 0
        if not drop_rendered:
            frames_to_drop_tensor = (
                torch.randint(0, num_frames, (1,), device=device)
                if is_source_rank
                else torch.empty(1, device=device, dtype=torch.long)
            )
            if broadcast is not None:
                broadcast(frames_to_drop_tensor)
            frames_to_drop = int(frames_to_drop_tensor.item())
        return drop_rendered, frames_to_drop, drop_text, ignore_neighbors

    # Non-CP batch preprocessing (main's contract for Trainer.training_batch).
    # CP-aware trainers override this via ContextParallelVideoMixin, which is
    # listed before TrainerBase in their MRO.
    def process_inputs(self, data: dict) -> dict[str, torch.Tensor | bool]:
        num_frames = data["rgb_gt"].shape[1]
        processed_data = {"rgb_gt": data["rgb_gt"]}

        drop_rendered, frames_to_drop, drop_text, ignore_neighbors = self._sample_input_dropout(num_frames)

        opacity = data["opacity"].to(dtype=self.pipe.vae.dtype)
        if drop_rendered:
            condition = self._encode_zeros_cached(data["rgb_rendered"], neighbors=False)
            opacity = torch.zeros_like(opacity)
        else:
            if frames_to_drop > 0:
                condition = self.pipe.encode_video_frames(
                    torch.cat(
                        [
                            data["rgb_rendered"][:, : num_frames - frames_to_drop],
                            torch.zeros_like(data["rgb_rendered"][:, -frames_to_drop:]),
                        ],
                        dim=1,
                    )
                )
                if opacity is data["opacity"]:
                    opacity = opacity.clone()
                opacity[:, -frames_to_drop:] = 0
            else:
                condition = self.pipe.encode_video_frames(data["rgb_rendered"])

        processed_data["condition"] = condition
        processed_data["opacity"] = opacity

        processed_data.update(prompt_payload_for_text_dropout(data, drop_text=drop_text))

        # Wan zeroes the reference output; Cosmos3 omits reference tokens.
        # A cached placeholder avoids encoding images that will not affect the output.
        if ignore_neighbors:
            processed_data["neighbors_condition"] = self._encode_zeros_cached(data["rgb_neighbors"], neighbors=True)
        else:
            processed_data["neighbors_condition"] = self.pipe.encode_neighbors(data["rgb_neighbors"])
        processed_data["ignore_neighbors"] = ignore_neighbors
        processed_data["neighbor_w2cs"] = data["neighbor_w2cs"]
        processed_data["neighbor_Ks"] = data["neighbor_Ks"]
        processed_data["camera_rays"] = data["camera_rays"].to(dtype=self.pipe.vae.dtype)
        if data.get("neighbor_camera_rays") is not None:
            processed_data["neighbor_camera_rays"] = data["neighbor_camera_rays"].to(dtype=self.pipe.vae.dtype)
        processed_data["w2cs"] = data["w2cs"]
        processed_data["Ks"] = data["Ks"]

        return processed_data

    @contextlib.contextmanager
    def validation_context(self):
        transformer = self.pipe.transformer
        was_training = transformer.training
        transformer.eval()
        ema = getattr(self, "ema", None)
        ema_context = ema.apply(enabled=ema.validation_active) if ema is not None else contextlib.nullcontext()
        try:
            with torch.no_grad(), ema_context:
                yield
        finally:
            transformer.train(was_training)

    def _check_models_finite(self, kind: str, data: dict, loss: torch.Tensor) -> None:
        for i, model in enumerate(self.accelerator._models):
            for name, param in model.named_parameters():
                value = param.grad if kind == "grad" else param
                if value is None or torch.isfinite(value).all():
                    continue
                torch.save(
                    [data, loss, param],
                    self.args.project_dir
                    / f"train_error-{kind}-{i}-{self.accelerator.process_index}-{self.get_step()}.pt",
                )
                noun = "parameter gradient" if kind == "grad" else "parameter"
                raise ValueError(f"Model {i} {noun} {name} contains non-finite values")

    def _maybe_checkpoint(self) -> None:
        # Check only at synchronized step boundaries so all ranks
        # save the same logical iteration before Slurm requeues.
        if self._should_checkpoint_for_timeout():
            self.accelerator.print(
                f"Saving checkpoint at iteration {self.get_step()} and exiting with timeout signal for requeue"
            )
            self._save_timeout_checkpoint_and_ack(self.get_step())
            self.accelerator.print(f"Exiting with timeout signal for requeue at iteration {self.get_step()}")
            sys.exit(124)

        if self.get_step() % self.args.save_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()
            self._save_state_with_commit_marker(self.get_step())

    def _validation_step(self, log_dict: dict) -> None:
        """Training owns when to evaluate and which raw/EMA weights to expose."""
        step = self.get_step()
        index = max(step // self.args.validation_steps - 1, 0)
        save_dir = self.args.project_dir / "val" / f"step_{step:06d}"
        runner = self.evaluation
        plan = build_evaluation_plan(self.val_datasets, runner.config, index)
        ema = getattr(self, "ema", None)
        identity = EvaluationIdentity(
            project_dir=self.args.project_dir,
            step=step,
            initial_checkpoint=getattr(self.args, "base_checkpoint_dir", None),
            weights="ema" if ema is not None and ema.validation_active else "raw",
            training_algorithm=type(self).__name__,
        )
        if ema is not None:
            log_dict["val/ema_active"] = int(ema.validation_active)
            log_dict["val/ema_updates"] = ema.ema_updates
        log_dict.update(conditioning_adapter_weight_norms(self.pipe.transformer))
        log_dict.update(scope_statistics(self.pipe.transformer))
        scope_samples = []
        capture_scope = hasattr(self.pipe.transformer, "_scope_accumulator")
        with self.validation_context():
            if capture_scope:
                self.pipe.transformer._scope_accumulator = scope_samples
            try:
                regular = runner.run_pass(plan.datasets, plan.regular, save_dir, weights=identity.weights)
            finally:
                if capture_scope:
                    self.pipe.transformer._scope_accumulator = None
            panel = runner.run_pass(plan.datasets, plan.panel, None, weights=identity.weights)
        log_dict.update(scope_to_query_ratios(scope_samples))
        result = runner.finish(plan, identity, regular, panel, save_dir)
        log_dict.update(result.scalars)
        if self.accelerator.is_main_process and uses_wandb_logging(self.args.log_with):
            log_dict.update(validation_media(plan, save_dir))
        torch.cuda.empty_cache()
        gc.collect()

    def train(self) -> None:
        pbar = tqdm(total=self.args.max_iterations, disable=not self.accelerator.is_local_main_process)
        pbar.update(self.get_step())
        while self.get_step() < self.args.max_iterations:
            data_loading_start_time = time.time()
            for data in self.train_dataloader:
                log_dict = {"train/data_time": time.time() - data_loading_start_time}
                if self.get_step() % 20 == 0:
                    torch.cuda.empty_cache()

                with self.accelerator.accumulate(*self.to_accumulate()):
                    loss, metrics = self.training_batch(data)
                    self.accelerator.backward(loss)

                    # Synced after backward is enqueued so the host does not
                    # stall on the forward pass before launching it.
                    if not torch.isfinite(loss).all():
                        torch.save(
                            [data, loss],
                            self.args.project_dir
                            / f"train_error-{self.accelerator.process_index}-{self.get_step()}.pt",
                        )
                        raise ValueError(
                            f"{self.accelerator.process_index} {self.get_step()}: Loss is not finite: {loss}"
                        )

                    log_dict["train/loss"] = loss.item()
                    log_dict.update({f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in metrics.items()})

                    if self.accelerator.sync_gradients:
                        # Clip only on the syncing micro-batch: clipping partial
                        # accumulated gradients would distort the window.
                        self.on_before_optimizer_step()

                    if self.args.check_state_finite:
                        self._check_models_finite("grad", data, loss)

                    for optimizer in self.optimizers:
                        optimizer.step()
                    for scheduler in self.schedulers:
                        scheduler.step()

                    if self.args.check_state_finite:
                        self._check_models_finite("param", data, loss)

                    self.on_before_zero_grad()

                    for optimizer in self.optimizers:
                        optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self._maybe_checkpoint()

                    is_validation_step = self.get_step() % self.args.validation_steps == 0
                    if is_validation_step:
                        self._validation_step(log_dict)

                    if uses_wandb_logging(self.args.log_with):
                        self.accelerator.log(log_dict, step=self.get_step())
                    if is_validation_step:
                        # Barrier after logging: rank 0 may spend extra time uploading
                        # validation images/videos via wandb while other ranks only log
                        # scalars. Without this barrier the faster ranks re-enter the
                        # dataloader (triggering accelerate's RNG-sync broadcast) before
                        # rank 0 is ready, causing an NCCL timeout.
                        barrier_if_distributed()
                    pbar.update(1)
                    data_loading_start_time = time.time()

        self._save_state_with_commit_marker(self.get_step())

    def get_step(self) -> int:
        return self.step_offset + self.accelerator.step // self.args.gradient_accumulation_steps
