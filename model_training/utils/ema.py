# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from accelerate import Accelerator
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from model_training.checkpoint_metadata import write_checkpoint_metadata

EMA_CHECKPOINT_VERSION = 1
EMA_PROGRESS_FILENAME = "ema.json"
EMA_WEIGHTS_DIRECTORY = "pytorch_model_fsdp_ema"


class DTensorFastEmaModelUpdater:
    """
    Similar as FastEmaModelUpdater
    """

    def __init__(self):
        # Flag to indicate whether the cache is taken or not. Useful to avoid cache overwrite
        self.is_cached = False

    @staticmethod
    def _local_tensor(param: torch.Tensor) -> torch.Tensor:
        return param.to_local() if isinstance(param, DTensor) else param

    @torch.no_grad()
    def copy_to(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters(), strict=True):
            self._local_tensor(tgt_params).copy_(self._local_tensor(src_params))

    @torch.no_grad()
    def update_average(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module, beta: float = 0.9999) -> None:
        target_list = []
        source_list = []
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters(), strict=True):
            assert (
                tgt_params.dtype == torch.float32
            ), f"EMA model only works in FP32 dtype, got {tgt_params.dtype} instead."
            target_list.append(self._local_tensor(tgt_params))
            source_list.append(self._local_tensor(src_params))
        torch._foreach_mul_(target_list, beta)
        torch._foreach_add_(target_list, source_list, alpha=1.0 - beta)

    @torch.no_grad()
    def cache(self, parameters: Any) -> None:
        assert self.is_cached is False, "EMA cache is already taken. Did you forget to restore it?"
        self.collected_params = [self._local_tensor(param).clone() for param in parameters]
        self.is_cached = True

    @torch.no_grad()
    def restore(self, parameters: Any) -> None:
        assert self.is_cached, "EMA cache is not taken yet."
        for c_param, param in zip(self.collected_params, parameters, strict=True):
            local_param = self._local_tensor(param)
            local_param.copy_(c_param.to(dtype=local_param.dtype))
        self.collected_params = []
        # Release the cache after we call restore
        self.is_cached = False


class EMAProgressMissingError(ValueError):
    """The shared checkpoint's EMA progress file is unavailable."""


def read_ema_progress(checkpoint_dir: Path) -> dict:
    path = checkpoint_dir / EMA_PROGRESS_FILENAME
    if not path.is_file():
        retired = [name for name in ("long_ar_ema.json", "pytorch_model_fsdp_2") if (checkpoint_dir / name).exists()]
        layout = f" Retired EMA layout found: {', '.join(retired)}." if retired else ""
        raise EMAProgressMissingError(
            f"EMA checkpoint metadata is unavailable: the shared schema requires {path}.{layout} "
            "Older layouts are not migrated."
        )
    progress = json.loads(path.read_text())
    if not isinstance(progress, dict):
        raise ValueError(f"EMA progress: expected a JSON object, got {type(progress).__name__} in {path}")
    if progress.get("version") != EMA_CHECKPOINT_VERSION:
        raise ValueError(f"EMA version: expected {EMA_CHECKPOINT_VERSION}, got {progress.get('version')} in {path}")
    stage = progress.get("training_stage")
    if stage not in ("long_ar", "dmd"):
        raise ValueError(f"EMA training_stage: expected long_ar or dmd, got {stage!r} in {path}")
    for field in ("checkpoint_step", "optimizer_updates", "ema_start_step", "initialize_after_updates", "ema_updates"):
        value = progress.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"EMA {field}: expected a nonnegative integer, got {value!r} in {path}")
    weight = progress.get("ema_weight")
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight >= 1:
        raise ValueError(f"EMA ema_weight: expected a finite decay below 1, got {weight!r} in {path}")
    updates = progress["optimizer_updates"]
    if updates > progress["checkpoint_step"]:
        raise ValueError(
            f"EMA optimizer_updates: expected at most {progress['checkpoint_step']}, got {updates} in {path}"
        )
    minimum = progress["initialize_after_updates"]
    initialized = progress["ema_initialized_update"]
    if initialized is None:
        if weight > 0 and updates >= minimum:
            raise ValueError(f"EMA ema_initialized_update: expected {minimum} through {updates}, got None in {path}")
    elif type(initialized) is not int or not minimum <= initialized <= updates or weight <= 0:
        raise ValueError(
            f"EMA ema_initialized_update: expected {minimum} through {updates} with EMA enabled, got {initialized!r} in {path}"
        )
    expected_updates = 0 if initialized is None else updates - initialized
    if progress["ema_updates"] != expected_updates:
        raise ValueError(f"EMA ema_updates: expected {expected_updates}, got {progress['ema_updates']} in {path}")
    return progress


def ema_export_metadata(checkpoint_dir: Path, checkpoint_step: int, slot: str) -> dict[str, str | int | float]:
    state_path = checkpoint_dir.parent / EMA_PROGRESS_FILENAME
    is_ema = checkpoint_dir.name == EMA_WEIGHTS_DIRECTORY
    if not is_ema and (checkpoint_dir.name != "pytorch_model_fsdp_0" or not state_path.is_file()):
        return {}
    progress = read_ema_progress(checkpoint_dir.parent)
    for field, expected, actual in (
        ("checkpoint", progress["checkpoint_step"], checkpoint_step),
        ("slot", checkpoint_dir.name, slot),
    ):
        if actual != expected:
            raise ValueError(
                f"Export {field} for slot {checkpoint_dir.name}: expected {expected}, got {actual} in {state_path}"
            )
    metadata = {"weight_type": "ema" if is_ema else "raw", "training_stage": progress["training_stage"]}
    if is_ema:
        if progress["ema_initialized_update"] is None:
            raise ValueError(
                f"EMA export ema_initialized_update: expected an initialized EMA, got None in {state_path}"
            )
        metadata.update(
            {key: progress[key] for key in ("ema_weight", "ema_start_step", "ema_initialized_update", "ema_updates")}
        )
        metadata["ema_optimizer_updates"] = progress["optimizer_updates"]
    return metadata


class ModelEMA:
    """FP32 parameter-shard EMA for DMD.

    Raw model state remains in Accelerate's slot 0. ``pytorch_model_fsdp_ema``
    contains EMA parameters plus the model's current buffers when EMA is active.
    Its ``artifixer.metadata.json`` records the training pose alignment for
    independent export and evaluation.
    ``ema.json`` retains the shared checkpoint schema. The metadata reader
    also accepts historical ``training_stage="long_ar"`` artifacts for export:

    - ``version``: shared serialization format version, not an algorithm version.
    - ``training_stage``: ``dmd`` (or ``long_ar`` in historical artifacts);
      must match on resume.
    - ``checkpoint_step``: trainer step, including DMD critic-only iterations.
    - ``optimizer_updates``: updates admitted by the trainer's step guards;
      successful DMD generator updates (all model updates in historical artifacts).
    - ``ema_weight`` / ``ema_start_step``: configured decay and start threshold.
    - ``initialize_after_updates``: tracked-update count required for allocation
      (zero for DMD's post-load initialization; older artifacts may use a positive threshold).
    - ``ema_initialized_update``: tracked-update index of the last initialization
      or copy/reset, or null while uninitialized or disabled.
    - ``ema_updates``: averaging updates since that last copy/reset, equal to
      ``optimizer_updates - ema_initialized_update`` when active, otherwise zero.

    DMD initializes after loading the student and copies it on each
    generator update until the trainer step reaches 200; subsequent generator
    updates average with decay 0.99. Pre-start copies reset the averaging count.
    Resume requires this schema and is deferred until raw weights and trainer
    step/offset are restored. Configuration changes and old layouts are rejected.
    """

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        get_step: Callable[[], int],
        *,
        ema_weight: float,
        ema_start_step: int,
        initialize_after_updates: int,
        training_stage: str,
        camera_pose_alignment: str,
    ):
        self.model = model
        self.accelerator = accelerator
        self.get_step = get_step
        self.camera_pose_alignment = camera_pose_alignment
        self.config = {
            "ema_weight": ema_weight,
            "ema_start_step": ema_start_step,
            "initialize_after_updates": initialize_after_updates,
            "training_stage": training_stage,
        }
        self.weights: nn.ParameterList | None = None
        self.worker = DTensorFastEmaModelUpdater()
        self.optimizer_updates = 0
        self.initialized_update: int | None = None
        self._checkpoint_dir: Path | None = None
        accelerator.register_save_state_pre_hook(self._save_hook)
        accelerator.register_load_state_pre_hook(self._load_hook)

    @property
    def active(self) -> bool:
        return self.weights is not None

    @property
    def validation_active(self) -> bool:
        return self.active and self.get_step() >= self.config["ema_start_step"]

    @property
    def is_cached(self) -> bool:
        return self.worker.is_cached

    @property
    def ema_updates(self) -> int:
        return 0 if self.initialized_update is None else self.optimizer_updates - self.initialized_update

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.weights.parameters() if self.active else iter(())

    def _reshard(self) -> None:
        # https://github.com/pytorch/pytorch/issues/144289
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

    @torch.no_grad()
    def _initialize(self) -> None:
        # The EMA model is not optimized directly, but it still needs the same
        # FSDP2 wrapping as the trainable transformer.
        self._reshard()
        self.weights = nn.ParameterList(
            nn.Parameter(param.detach().clone().float(), requires_grad=False) for param in self.model.parameters()
        )
        self.initialized_update = self.optimizer_updates

    @torch.no_grad()
    def update(self, *, average: bool = True) -> None:
        if self.is_cached:
            raise RuntimeError("Cannot update EMA while EMA validation weights are applied")
        self.optimizer_updates += 1
        if self.config["ema_weight"] <= 0 or self.optimizer_updates < self.config["initialize_after_updates"]:
            return
        if not self.active:
            self._initialize()
            self.accelerator.print(
                f"Initialized {self.config['training_stage']} EMA from optimizer update {self.optimizer_updates} "
                f"with decay {self.config['ema_weight']}"
            )
        elif average:
            self.worker.update_average(self.model, self.weights, beta=self.config["ema_weight"])
        else:
            self.worker.copy_to(self.model, self.weights)
            self.initialized_update = self.optimizer_updates

    @contextmanager
    def apply(self, *, enabled: bool = True) -> Iterator[None]:
        if self.is_cached:
            raise RuntimeError("EMA validation weights are already applied")
        was_training = self.model.training
        try:
            if enabled and self.active:
                self._reshard()
                self.worker.cache(self.model.parameters())
                self.worker.copy_to(self.weights, self.model)
            self.model.eval()
            yield
        finally:
            try:
                if self.is_cached:
                    self._reshard()
                    self.worker.restore(self.model.parameters())
            finally:
                self.model.train(was_training)

    def _parameter_state(self) -> dict[str, torch.Tensor]:
        shadows = {
            id(param): shadow.detach() for param, shadow in zip(self.model.parameters(), self.parameters(), strict=True)
        }
        return {name: shadows[id(param)] for name, param in self.model.named_parameters(remove_duplicate=False)}

    def _save_hook(self, models, weights, output_dir: str) -> None:
        self.save_checkpoint(Path(output_dir))

    def _load_hook(self, models, input_dir: str) -> None:
        # Accelerate restores its step after this hook; step_offset is resolved
        # by the entrypoint afterwards. Defer EMA loading until both are known.
        self._checkpoint_dir = Path(input_dir)

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        if self.is_cached:
            raise RuntimeError("Cannot save a training checkpoint while EMA validation weights are applied")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.active:
            state = self.model.state_dict() | self._parameter_state()
            dcp.save({"model": state}, checkpoint_id=checkpoint_dir / EMA_WEIGHTS_DIRECTORY)
        if self.accelerator.is_main_process:
            if self.active:
                write_checkpoint_metadata(
                    checkpoint_dir / EMA_WEIGHTS_DIRECTORY, {"camera_pose_alignment": self.camera_pose_alignment}
                )
            progress = self.config | {
                "version": EMA_CHECKPOINT_VERSION,
                "checkpoint_step": self.get_step(),
                "optimizer_updates": self.optimizer_updates,
                "ema_initialized_update": self.initialized_update,
                "ema_updates": self.ema_updates,
            }
            (checkpoint_dir / EMA_PROGRESS_FILENAME).write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")

    @torch.no_grad()
    def restore_after_model_load(self) -> None:
        if self.is_cached:
            raise RuntimeError("Cannot restore a training checkpoint while EMA validation weights are applied")
        if self._checkpoint_dir is None:
            if not self.active and self.config["ema_weight"] > 0 and self.config["initialize_after_updates"] == 0:
                self._initialize()
            return
        try:
            progress = read_ema_progress(self._checkpoint_dir)
        except EMAProgressMissingError as error:
            raise ValueError(
                f"Cannot resume EMA training state: {error} "
                "Resume with the revision that wrote the checkpoint, or warm-start a new --project_dir "
                f"using --base_checkpoint_dir {self._checkpoint_dir / 'pytorch_model_fsdp_0'}. "
                "For DMD, keep --base_checkpoint_dir_critic pointed at the original stage-1 teacher checkpoint; "
                "the trained fake critic is not a replacement for that teacher."
            ) from error
        state_path = self._checkpoint_dir / EMA_PROGRESS_FILENAME
        for key, expected in (self.config | {"checkpoint_step": self.get_step()}).items():
            if progress[key] != expected:
                raise ValueError(f"EMA {key}: expected {expected}, got {progress[key]} in {state_path}")
        initialized = progress["ema_initialized_update"]
        weights_path = self._checkpoint_dir / EMA_WEIGHTS_DIRECTORY
        metadata = weights_path / ".metadata"
        if initialized is None and weights_path.exists():
            raise ValueError(f"EMA ema_initialized_update: expected no weights when uninitialized, got {weights_path}")
        if initialized is not None and not metadata.is_file():
            raise FileNotFoundError(
                f"EMA weights: expected checkpoint metadata file {metadata}, got missing or non-file"
            )
        self.optimizer_updates = progress["optimizer_updates"]
        if initialized is not None:
            self._initialize()
            dcp.load({"model": self._parameter_state()}, checkpoint_id=weights_path)
        else:
            self.weights = None
        self.initialized_update = initialized
        self.accelerator.print(
            f"Restored {self.config['training_stage']} EMA progress: {self.optimizer_updates} optimizer updates, "
            f"initialized at {initialized}, {self.ema_updates} averaging updates"
        )
        self._checkpoint_dir = None
