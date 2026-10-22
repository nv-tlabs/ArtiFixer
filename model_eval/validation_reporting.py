# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from model_eval.validation_config import EvaluationConfig, EvaluationContext, EvaluationIdentity
from model_eval.validation_scoring import EvaluationPass
from model_eval.validation_tasks import VALIDATION_SEED, EvaluationPlan
from model_training.utils import validation_artifacts, validation_metrics


@dataclass(frozen=True)
class EvaluationResult:
    scalars: dict
    report: dict | None
    save_dir: Path


def finish_evaluation(
    config: EvaluationConfig,
    context: EvaluationContext,
    plan: EvaluationPlan,
    identity: EvaluationIdentity,
    regular: EvaluationPass,
    panel: EvaluationPass,
    save_dir: Path,
    *,
    pipeline_name: str,
) -> EvaluationResult:
    scalars = {"val/wall_clock_seconds": regular.seconds}
    for key, values in regular.metrics.items():
        scalars.update({f"val/{key}/{name}": value for name, value in values.items()})
    # Diagnostic interventions have different inputs; only unmodified tasks enter this mean.
    unmodified = {spec.key for spec, _ in plan.regular if not spec.is_ablation}
    means = validation_metrics.mean_over_metrics([m for key, m in regular.metrics.items() if key in unmodified])
    scalars.update({f"val/mean/{name}": value for name, value in means.items()})
    if plan.expected_panel:
        scalars["test/wall_clock_seconds"] = panel.seconds
        by_config = {}
        for key, values in panel.metrics.items():
            name, index = key.rsplit("@scene", 1)
            by_config.setdefault(name, []).append(values)
            scalars.update({f"test/{name}/scene{index}/{metric}": value for metric, value in values.items()})
        for name, values in by_config.items():
            scalars.update(
                {
                    f"test/{name}/{metric}": value
                    for metric, value in validation_metrics.mean_over_metrics(values).items()
                }
            )
    if not config.validation_metrics:
        return EvaluationResult(scalars, None, save_dir)

    regular_report = validation_artifacts.pass_report(
        [spec.key for spec, _ in plan.regular], regular.metrics, regular.records
    )
    panel_report = None
    manifest = None
    if plan.expected_panel:
        panel_report = validation_artifacts.pass_report(plan.expected_panel, panel.metrics, panel.records)
        fewest = min(plan.datasets)
        samples = [
            panel.records.get(f"default_{fewest}@scene{index:02d}", {}).get("sample")
            for index in range(config.test_panel_scenes)
        ]
        if all(sample is not None for sample in samples):
            samples = [
                {
                    **sample,
                    "negative_conditioning": panel.records[f"default_{fewest}@scene{index:02d}"][
                        "negative_conditioning"
                    ],
                }
                for index, sample in enumerate(samples)
            ]
            manifest = {
                "schema_version": validation_artifacts.VALIDATION_SCHEMA_VERSION,
                "noise_seed": VALIDATION_SEED,
                "num_neighbors": fewest,
                "num_frames": len(samples[0]["frame_indices"]),
                "dataset_scaling_factor": config.dataset_scaling_factor,
                "camera_pose_alignment": config.camera_pose_alignment,
                "samples": samples,
            }
        else:
            panel_report["status"] = "incomplete"
            panel_report["manifest_error"] = "Missing panel input metadata"
    complete = regular_report["status"] == "complete" and (panel_report is None or panel_report["status"] == "complete")
    report = {
        "schema_version": validation_artifacts.VALIDATION_SCHEMA_VERSION,
        "status": "complete" if complete else "incomplete",
        "step": identity.step,
        "checkpoint": str(identity.checkpoint) if identity.checkpoint is not None else None,
        "initial_checkpoint": str(identity.initial_checkpoint) if identity.initial_checkpoint is not None else None,
        "project_dir": str(identity.project_dir),
        "weights": identity.weights,
        "model_id": config.model_id,
        "trainer": identity.training_algorithm,
        "pipeline": pipeline_name,
        "world_size": context.world_size,
        "device": str(context.device),
        "device_name": torch.cuda.get_device_name(context.device) if context.device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "sampling": {
            name: getattr(config, name)
            for name in (
                "frames_per_block",
                "context_parallel_size",
                "local_attn_size",
                "sink_size",
                "vae_chunk_halo_latent_frames",
                "validation_chunk_frames",
                "attention_backend",
            )
        },
        "metric_protocol": copy.deepcopy(validation_metrics.METRIC_PROTOCOL),
        "manifest_id": validation_artifacts.content_id(manifest) if manifest is not None else None,
        "validation": regular_report,
        "panel": panel_report,
    }
    error = [None]
    if context.is_main_process:
        try:
            report["code_revision"] = validation_artifacts.code_revision()
            if manifest is not None:
                validation_artifacts.write_json(save_dir / "manifest.json", manifest)
            validation_artifacts.write_json(save_dir / "results.json", report)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"Could not write validation artifacts: {error[0]}")
    if config.validation_strict and not complete:
        raise RuntimeError(f"Incomplete validation; see {save_dir / 'results.json'}")
    return EvaluationResult(scalars, report, save_dir)


def validation_media(plan: EvaluationPlan, save_dir: Path) -> dict:
    """Optional tracking adapter; scoring and standalone JSON output do not require it."""
    import wandb

    first = plan.datasets[min(plan.datasets)][plan.validation_index]
    caption = f"{first['prompt_file']}: {first['prompt']}"
    media = {
        f"val/{file.stem}": wandb.Image(file, caption=caption)
        for file in save_dir.glob("*.png")
        if not file.name.startswith(".")
    }
    media.update(
        {
            f"val/{file.stem}": wandb.Video(str(file), caption=caption, fps=15, format="mp4")
            for file in save_dir.glob("*.mp4")
            if not file.name.startswith(".")
        }
    )
    return media
