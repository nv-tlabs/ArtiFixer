# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from model_eval.validation_tasks import ValidationTask, validation_noise_seed
from model_training.data.utils import latent_index_representative_raw_frame
from model_training.utils import validation_metrics
from model_training.utils.video_io import save_video


def gather_task_dict(local: dict) -> dict:
    if not dist.is_initialized():
        return local
    shards = [None] * dist.get_world_size()
    dist.all_gather_object(shards, local)
    return {key: value for shard in shards if shard for key, value in shard.items()}


@dataclass(frozen=True)
class EvaluationPass:
    metrics: dict
    records: dict
    seconds: float = 0.0


class ValidationRecorder:
    def __init__(self, config, context, temporal_scale, *, weights="raw"):
        self.config = config
        self.context = context
        self.temporal_scale = temporal_scale
        self.weights = weights
        self.metrics = {}
        self.records = {}

    def finish(self, seconds=0.0):
        return EvaluationPass(gather_task_dict(self.metrics), gather_task_dict(self.records), seconds)

    def start_generation(self) -> None:
        if self.context.device.type == "cuda":
            torch.cuda.synchronize(self.context.device)
        self._validation_generation_started = time.perf_counter()

    def save(self, output: torch.Tensor, task: ValidationTask, save_dir: Path | None, *, cp_size: int = 1) -> None:
        """Score the clip and, unless ``save_dir`` is None, write the composite."""
        generated = output.cpu().squeeze(0)
        started = getattr(self, "_validation_generation_started", None)
        generation_seconds = time.perf_counter() - started if started is not None else None
        rgb_gt = task.rgb_gt.cpu().squeeze(0)
        rendered_rgb = task.rendered_rgb.cpu().squeeze(0)
        self.record(generated, rgb_gt, rendered_rgb, task, cp_size=cp_size)
        if self.config.validation_metrics and generation_seconds is not None and task.key in self.metrics:
            self.metrics[task.key]["generation_seconds"] = generation_seconds
            self.metrics[task.key]["generation_fps"] = len(generated) / max(generation_seconds, 1e-9)
        if save_dir is None:
            return
        if self.weights == "ema":
            task = replace(task, key=f"{task.key}_ema")
        result_rows = [torch.cat([generated, rgb_gt, rendered_rgb], dim=-1)]
        if task.neighbors is not None:
            for neighbor_index in range(0, task.neighbors.shape[1], 3):
                result_rows.append(
                    torch.cat([x.cpu() for x in task.neighbors[0, neighbor_index : neighbor_index + 3]], dim=-1)
                    .unsqueeze(0)
                    .expand(result_rows[0].shape[0], -1, -1, -1)
                )
        result = torch.cat(result_rows, dim=-2)
        if result.shape[0] == 1:
            Image.fromarray((result.clamp(0, 1).squeeze(0).permute(1, 2, 0) * 255).byte().numpy()).save(
                save_dir / f"{task.key}.png"
            )
        else:
            save_video(result.clamp(0, 1), save_dir / f"{task.key}.mp4", fps=15)

    def record(
        self,
        generated: torch.Tensor,
        rgb_gt: torch.Tensor,
        rendered_rgb: torch.Tensor,
        task: ValidationTask,
        *,
        cp_size: int = 1,
    ) -> None:
        """Score one clip into ``self.metrics``. Each metric family is guarded on
        its own: they fail for unrelated reasons and one failure must not discard the rest."""
        if not self.config.validation_metrics:
            return

        device = self.context.device
        temporal_scale = self.temporal_scale
        # A zeroed render is not a baseline the model beat, so it gets no rendered-side numbers.
        scored_rendered = rendered_rgb if task.has_unmodified_rendered_baseline else None

        metrics: dict[str, float] = {}
        errors = {}
        scoring_started = time.time()

        def collect(label: str, produce):
            try:
                values = produce() or {}
                if any(not math.isfinite(value) for value in values.values()):
                    raise ValueError("Metric family returned nonfinite values")
                metrics.update(values)
            except Exception as exc:
                errors[label] = f"{type(exc).__name__}: {exc}"
                self.context.print(f"Validation metric {label!r} failed for task {task.key}:\n{traceback.format_exc()}")

        def fidelity(video, target):
            return validation_metrics.fidelity_metrics(video, target, device=device)

        def check_inputs():
            for name, video in (("generated", generated), ("gt", rgb_gt), ("rendered", rendered_rgb)):
                for start in range(0, len(video), validation_metrics.FRAME_CHUNK):
                    if not torch.isfinite(video[start : start + validation_metrics.FRAME_CHUNK]).all():
                        raise ValueError(f"{name} contains nonfinite pixels")
            return {}

        collect("inputs", check_inputs)
        collect("fidelity", lambda: fidelity(generated, rgb_gt))
        collect(
            "temporal",
            lambda: validation_metrics.temporal_metrics(
                generated,
                rgb_gt,
                chunk_frames=getattr(self.config, "validation_chunk_frames", 32),
                device=device,
                sampler_chunk_latents=getattr(self.config, "frames_per_block", None),
                temporal_scale=temporal_scale,
            ),
        )

        def rendered_and_gain():
            if scored_rendered is None:
                return {}
            baseline = fidelity(scored_rendered, rgb_gt)
            result = {f"rendered_{name}": value for name, value in baseline.items()}
            result.update(validation_metrics.gain_metrics(metrics, baseline))
            return result

        collect("rendered_and_gain", rendered_and_gain)

        collect(
            "view_consistency",
            lambda: validation_metrics.view_consistency_metrics(
                {"generated": generated, "rendered": scored_rendered, "gt": rgb_gt},
                task.w2cs.detach().cpu().squeeze(0),
                device=device,
                latent_to_frame=[
                    latent_index_representative_raw_frame(
                        latent_index,
                        generated.shape[0],
                        temporal_scale=temporal_scale,
                        pose_alignment=self.config.camera_pose_alignment,
                    )
                    for latent_index in range(task.w2cs.shape[1])
                ],
            ),
        )
        if metrics:
            metrics["scoring_seconds"] = time.time() - scoring_started
            self.metrics[task.key] = metrics
        required = {"psnr", "ssim", "num_repeat_pairs", "chunk000_psnr", "last_minus_first_psnr"}
        if len(generated) > 1:
            required.update(("temporal_delta_l1", "motion_generated", "motion_gt"))
        if scored_rendered is not None:
            required.update(("rendered_psnr", "rendered_ssim", "psnr_gain", "ssim_gain"))
        if metrics.get("num_repeat_pairs", 0) > 0:
            required.update(("revisit_psnr_generated", "revisit_ssim_generated", "revisit_psnr_gt"))
        self.records[task.key] = {
            "sample": task.sample_metadata,
            "negative_conditioning": task.negative_conditioning,
            "noise_seed": validation_noise_seed(task.key),
            "num_frames": len(generated),
            "resolution": list(generated.shape[-2:]),
            "num_inference_steps": task.num_inference_steps,
            "cp_size": cp_size,
            "errors": errors,
            "missing_metrics": sorted(required - metrics.keys()),
            "revisit_status": (
                "failed"
                if "view_consistency" in errors
                else "scored" if metrics.get("num_repeat_pairs", 0) else "not_applicable"
            ),
        }
