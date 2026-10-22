# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import torch

from model_eval.generation import prepare_generation_inputs
from model_eval.validation_config import EvaluationConfig, EvaluationContext
from model_training.utils.validation_artifacts import VALIDATION_SEED

PANEL_SPEC_KEYS = ("default", "no_rendered_rgb")


@dataclass(frozen=True)
class ValidationTaskSpec:
    key: str
    num_neighbors: int
    use_neighbors: bool = True
    zero_rendered: bool = False
    last_drop_fraction: float | None = None
    zero_prompt: bool = False

    @property
    def is_ablation(self) -> bool:
        """Diagnostic intervention, excluded from the aggregate for unmodified tasks."""
        return self.zero_rendered or self.zero_prompt or not self.use_neighbors or self.last_drop_fraction is not None

    @property
    def has_unmodified_rendered_baseline(self) -> bool:
        """Whether scoring may use the original rendered baseline, without render ablations."""
        return not self.zero_rendered and self.last_drop_fraction is None


@dataclass
class ValidationTask:
    key: str
    rendered_rgb: torch.Tensor
    rendered_opacity: torch.Tensor
    neighbors: torch.Tensor | None
    camera_rays: torch.Tensor
    w2cs: torch.Tensor
    neighbor_w2cs: torch.Tensor | None
    Ks: torch.Tensor
    neighbor_Ks: torch.Tensor | None
    prompt: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    encoded_negative_prompt: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    num_inference_steps: int
    rgb_gt: torch.Tensor
    has_unmodified_rendered_baseline: bool = True
    sample_metadata: dict | None = None
    negative_conditioning: dict | None = None
    neighbor_camera_rays: torch.Tensor | None = None


def build_validation_task_specs(val_datasets: dict[int, torch.utils.data.Dataset]) -> list[ValidationTaskSpec]:
    specs = []

    for num_neighbors in sorted(val_datasets):
        specs.append(ValidationTaskSpec(key=f"default_{num_neighbors}", num_neighbors=num_neighbors))

    fewest_neighbors = min(val_datasets.keys())

    for last_drop in [0.25, 0.5, 0.75]:
        specs.append(
            ValidationTaskSpec(
                key=f"last_drop_{last_drop}_{fewest_neighbors}",
                num_neighbors=fewest_neighbors,
                last_drop_fraction=last_drop,
            )
        )

    specs.extend(
        [
            ValidationTaskSpec(
                key=f"text_only_{fewest_neighbors}",
                num_neighbors=fewest_neighbors,
                use_neighbors=False,
                zero_rendered=True,
            ),
            ValidationTaskSpec(
                key=f"no_rendered_rgb_{fewest_neighbors}",
                num_neighbors=fewest_neighbors,
                zero_rendered=True,
            ),
            ValidationTaskSpec(
                key=f"neighbors_only_{fewest_neighbors}",
                num_neighbors=fewest_neighbors,
                zero_rendered=True,
                zero_prompt=True,
            ),
            ValidationTaskSpec(
                key=f"no_text_{fewest_neighbors}",
                num_neighbors=fewest_neighbors,
                zero_prompt=True,
            ),
            ValidationTaskSpec(
                key=f"no_neighbors_{fewest_neighbors}",
                num_neighbors=fewest_neighbors,
                use_neighbors=False,
            ),
            ValidationTaskSpec(
                key=f"no_neighbors_no_text_{fewest_neighbors}",
                num_neighbors=fewest_neighbors,
                use_neighbors=False,
                zero_prompt=True,
            ),
        ]
    )

    return specs


def build_test_panel_plan(
    val_datasets: dict[int, torch.utils.data.Dataset], num_scenes: int, max_tasks: int | None = None
) -> list[tuple[ValidationTaskSpec, int]]:
    """Pair each panel configuration with each of the first ``num_scenes`` samples."""
    by_key = {spec.key: spec for spec in build_validation_task_specs(val_datasets)}
    # The fewest-neighbour variant is the one the ablation specs are keyed on.
    fewest = min(val_datasets)
    wanted = [f"{key}_{fewest}" for key in PANEL_SPEC_KEYS]
    specs = [by_key[key] for key in wanted if key in by_key]
    assert len(specs) == len(wanted), f"panel specs {wanted} not all present among {sorted(by_key)}"

    plan = [
        (replace(spec, key=f"{spec.key}@scene{index:02d}"), index)
        for index in range(min(num_scenes, len(val_datasets[fewest])))
        for spec in specs
    ]
    # --max_validation_tasks caps the panel the way it caps the validation tasks.
    cap = max_tasks
    return plan if cap is None else plan[:cap]


def validation_noise_seed(task_key: str) -> int:
    """Stable task noise, independent of training step and run RNG.
    Panel scene suffixes distinguish the fixed samples within each configuration."""
    digest = hashlib.sha256(f"{VALIDATION_SEED}:{task_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class EvaluationPlan:
    datasets: dict[int, torch.utils.data.Dataset]
    regular: list[tuple[ValidationTaskSpec, int]]
    panel: list[tuple[ValidationTaskSpec, int]]
    expected_panel: list[str]
    validation_index: int


def build_evaluation_plan(datasets, config: EvaluationConfig, validation_index: int) -> EvaluationPlan:
    specs = build_validation_task_specs(datasets)
    cap = config.max_validation_tasks
    if cap is not None:
        if cap <= 0:
            raise ValueError("max_validation_tasks must be positive")
        specs = specs[:cap]
    regular = [(spec, validation_index) for spec in specs]
    panel = []
    expected_panel = []
    if config.validation_metrics and config.test_panel_scenes:
        panel = build_test_panel_plan(datasets, config.test_panel_scenes, cap)
        expected_panel = [
            f"{key}_{min(datasets)}@scene{index:02d}"
            for index in range(config.test_panel_scenes)
            for key in PANEL_SPEC_KEYS
        ]
    return EvaluationPlan(datasets, regular, panel, expected_panel, validation_index)


def load_validation_batch(datasets, num_neighbors, index):
    return torch.utils.data.default_collate([datasets[num_neighbors][index]])


def build_validation_task(
    pipe,
    config: EvaluationConfig,
    context: EvaluationContext,
    datasets,
    spec: ValidationTaskSpec,
    index: int,
    *,
    copy_to_device: bool = True,
):
    data = load_validation_batch(datasets, spec.num_neighbors, index)
    inputs = prepare_generation_inputs(
        data,
        device=context.device if copy_to_device else torch.device("cpu"),
        dtype=pipe.vae.dtype,
        batched=True,
        use_neighbors=spec.use_neighbors,
        zero_prompt=spec.zero_prompt,
        stage_conditioning_on_cpu=getattr(pipe, "stages_conditioning_per_chunk", False),
        stage_neighbors_on_cpu=not copy_to_device,
    )
    if spec.zero_rendered:
        inputs["rendered_rgb"] = torch.zeros_like(inputs["rendered_rgb"])
        inputs["rendered_opacity"] = torch.zeros_like(inputs["rendered_opacity"])
    elif spec.last_drop_fraction is not None:
        frames_to_drop = int(inputs["rendered_rgb"].shape[1] * spec.last_drop_fraction)
        inputs["rendered_rgb"] = inputs["rendered_rgb"].clone()
        inputs["rendered_opacity"] = inputs["rendered_opacity"].clone()
        if frames_to_drop > 0:
            inputs["rendered_rgb"][:, -frames_to_drop:] = 0
            inputs["rendered_opacity"][:, -frames_to_drop:] = 0
    # Keep the unconditional branch for zero-prompt interventions too: concurrent
    # FSDP task groups must issue the same number of transformer forwards.
    inputs["encoded_negative_prompt"] = inputs.pop("negative_prompt")
    task = ValidationTask(
        key=spec.key,
        **inputs,
        num_inference_steps=config.sampling.num_inference_steps,
        rgb_gt=data["rgb_gt"],
        has_unmodified_rendered_baseline=spec.has_unmodified_rendered_baseline,
        sample_metadata=json.loads(data["validation_sample_json"][0]) if "validation_sample_json" in data else None,
    )
    if task.sample_metadata is not None:
        task.negative_conditioning = pipe.validation_negative_conditioning(
            task.prompt, task.prompt_attention_mask, task.encoded_negative_prompt, task.negative_prompt_attention_mask
        )
        manifest = getattr(datasets[spec.num_neighbors], "validation_manifest", None)
        if manifest is not None:
            sample = manifest["samples"][index % len(manifest["samples"])]
            if task.negative_conditioning != sample["negative_conditioning"]:
                raise ValueError("Validation manifest negative_conditioning mismatch")
    return task


def task_generation_inputs(task: ValidationTask) -> dict:
    result = {
        name: getattr(task, name)
        for name in (
            "rendered_rgb",
            "rendered_opacity",
            "neighbors",
            "camera_rays",
            "w2cs",
            "neighbor_w2cs",
            "Ks",
            "neighbor_Ks",
            "prompt",
            "prompt_attention_mask",
            "negative_prompt_attention_mask",
        )
    }
    result["negative_prompt"] = task.encoded_negative_prompt
    if task.neighbor_camera_rays is not None:
        result["neighbor_camera_rays"] = task.neighbor_camera_rays
    return result
