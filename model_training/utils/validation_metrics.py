# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numeric metrics for the validation pass, computed from tensors the save path already
holds (nothing is re-generated and nothing reaches the network).

- ``fidelity_metrics``: PSNR and SSIM against the ground truth.
- ``gain_metrics``: improvement over the rendered conditioning on the same clip.
- ``view_consistency_metrics``: a video against itself at frames commanded to the same camera
  pose. This measures repeatability, not camera accuracy: a frozen video can score perfectly. Repeated poses
  are recovered from the commanded extrinsics, not from the clip name: pingpong trajectories
  start at a random frame in a random direction, so their pivot is not the caption's.

Scoring runs on the accelerator when a device is passed; SSIM on the host takes minutes per clip.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from model_eval.image_metrics import psnr as image_psnr
from model_eval.image_metrics import ssim as image_ssim

FRAME_CHUNK = 8
VALIDATION_PSNR_MSE_FLOOR = 1e-12
METRIC_PROTOCOL = {
    "psnr": {"data_range": 1.0, "mse_floor": VALIDATION_PSNR_MSE_FLOOR, "reduction": "mean_per_frame_db"},
    "ssim": {"window_size": 11, "sigma": 1.5, "padding": "zero", "reduction": "mean_per_frame"},
}


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-frame PSNR with the explicit finite-JSON validation policy."""
    return image_psnr(a, b, mse_floor=VALIDATION_PSNR_MSE_FLOOR)


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shared per-image SSIM, with bounded validation execution batches."""
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError("SSIM expects matching [N,C,H,W] tensors")
    return torch.cat(
        [
            image_ssim(a[start : start + FRAME_CHUNK], b[start : start + FRAME_CHUNK], size_average=False)
            for start in range(0, len(a), FRAME_CHUNK)
        ]
    )


def camera_centers(w2cs: torch.Tensor) -> torch.Tensor:
    """World-space camera centres from world-to-camera matrices ``[T, 4, 4]``."""
    rotation = w2cs[:, :3, :3]
    translation = w2cs[:, :3, 3]
    return -torch.einsum("tji,tj->ti", rotation, translation)


def camera_repeat_pairs(
    w2cs: torch.Tensor,
    rotation_tolerance_deg: float = 1.0,
    translation_tolerance_fraction: float = 0.01,
    min_frame_gap: int = 4,
    step_tolerance_fraction: float = 0.5,
) -> list[tuple[int, int]]:
    """Frame pairs that share a camera pose, each frame matched at most once.

    Poses match within ``rotation_tolerance_deg`` and ``translation_tolerance_fraction`` of the
    trajectory's extent (DL3DV poses carry an arbitrary SfM scale), widened to
    ``step_tolerance_fraction`` of the median frame-to-frame motion: the poses are per latent
    frame, i.e. averages or samples of the raw poses, so two latent frames revisiting the same
    raw pose can sit up to half a latent step of motion apart.
    """
    if w2cs.ndim != 3 or w2cs.shape[0] <= min_frame_gap:
        return []

    w2cs = w2cs.detach().double()
    centers = camera_centers(w2cs)
    distance = torch.cdist(centers, centers)

    extent = distance.max()
    if not torch.isfinite(w2cs).all():
        raise ValueError("Camera poses must be finite")

    rotation = w2cs[:, :3, :3]
    relative_trace = torch.einsum("aij,bij->ab", rotation, rotation)
    angle = torch.rad2deg(torch.arccos(((relative_trace - 1) / 2).clamp(-1, 1)))

    # A rotation-only trajectory can revisit a pose without translating. Fully static
    # trajectories do not provide evidence of camera control.
    if extent <= 0 and angle.max() <= rotation_tolerance_deg:
        return []

    step_distance = distance.diagonal(offset=1).median()
    step_angle = angle.diagonal(offset=1).median()
    translation_tolerance = max(extent * translation_tolerance_fraction, step_distance * step_tolerance_fraction)
    rotation_tolerance = max(rotation_tolerance_deg, step_angle * step_tolerance_fraction)

    frames = torch.arange(w2cs.shape[0], device=w2cs.device)
    eligible = (
        (angle <= rotation_tolerance)
        & (distance <= translation_tolerance)
        & ((frames[None, :] - frames[:, None]).abs() >= min_frame_gap)
    )

    pairs: list[tuple[int, int]] = []
    matched: set[int] = set()
    for i in range(w2cs.shape[0]):
        if i in matched:
            continue
        candidates = [j for j in torch.nonzero(eligible[i]).flatten().tolist() if j > i and j not in matched]
        if not candidates:
            continue
        j = min(candidates, key=lambda j: (float(distance[i, j]), float(angle[i, j])))
        pairs.append((i, j))
        matched.update((i, j))
    return pairs


def fidelity_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    device: torch.device | str | None = None,
    chunk: int = FRAME_CHUNK,
) -> dict:
    """PSNR and SSIM between two ``[T, C, H, W]`` videos in ``[0, 1]``, scored ``chunk`` frames at
    a time on ``device`` (None: wherever the tensors already are)."""
    if generated.shape != target.shape:
        raise ValueError(f"Video shapes differ: {generated.shape} vs {target.shape}")
    generated = generated.clamp(0, 1).float()
    target = target.clamp(0, 1).float()

    psnr_values, ssim_values = [], []
    for start in range(0, generated.shape[0], chunk):
        a, b = generated[start : start + chunk], target[start : start + chunk]
        if device is not None:
            a, b = a.to(device), b.to(device)
        psnr_values.append(psnr(a, b).cpu())
        ssim_values.append(ssim(a, b).cpu())

    return {
        "psnr": torch.cat(psnr_values).mean().item(),
        "ssim": torch.cat(ssim_values).mean().item(),
    }


def view_consistency_metrics(
    videos: dict[str, torch.Tensor],
    w2cs: torch.Tensor,
    latent_to_frame: Sequence[int] | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """Agreement between frames of each video that share a commanded camera pose.

    ``w2cs`` is per latent frame and the videos are per raw frame; ``latent_to_frame`` maps
    between them (default: identity). Scoring the rendered conditioning and the ground truth
    alongside the generated video gives its number a reference on this clip. Returns ``{}``
    when the trajectory never revisits a pose.
    """
    pairs = camera_repeat_pairs(w2cs)
    if not pairs:
        return {"num_repeat_pairs": 0.0}

    if latent_to_frame is not None:
        pairs = [(latent_to_frame[i], latent_to_frame[j]) for i, j in pairs]

    left = torch.tensor([i for i, _ in pairs])
    right = torch.tensor([j for _, j in pairs])

    metrics: dict[str, float] = {"num_repeat_pairs": float(len(pairs))}
    for name, video in videos.items():
        if video is None:
            continue
        psnr_values, ssim_values = [], []
        for start in range(0, len(pairs), FRAME_CHUNK):
            first = video[left[start : start + FRAME_CHUNK]].to(device=device, dtype=torch.float32).clamp(0, 1)
            second = video[right[start : start + FRAME_CHUNK]].to(device=device, dtype=torch.float32).clamp(0, 1)
            psnr_values.append(psnr(first, second).cpu())
            ssim_values.append(ssim(first, second).cpu())
        metrics[f"revisit_psnr_{name}"] = torch.cat(psnr_values).mean().item()
        metrics[f"revisit_ssim_{name}"] = torch.cat(ssim_values).mean().item()

    if "revisit_psnr_generated" in metrics and "revisit_psnr_rendered" in metrics:
        metrics["revisit_psnr_vs_rendered"] = metrics["revisit_psnr_generated"] - metrics["revisit_psnr_rendered"]
    return metrics


def temporal_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_frames: int = 32,
    device: torch.device | str | None = None,
    sampler_chunk_latents: int | None = None,
    temporal_scale: int = 4,
) -> dict[str, float]:
    """Motion-difference error and fidelity by raw-frame chunk, all relative to GT.

    Motion is a pixel-space diagnostic, not estimated camera pose. A frozen prediction
    has zero motion amplitude but nonzero error when the target moves. Chunk boundaries
    use a fixed raw-frame grid so different training stages can share the measurement.
    """
    if generated.shape != target.shape or generated.ndim != 4 or generated.shape[0] == 0:
        raise ValueError("Expected matching nonempty [T,C,H,W] videos")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    result = {}
    per_frame_psnr = []
    for start in range(0, len(generated), FRAME_CHUNK):
        a = generated[start : start + FRAME_CHUNK].to(device=device, dtype=torch.float32).clamp(0, 1)
        b = target[start : start + FRAME_CHUNK].to(device=device, dtype=torch.float32).clamp(0, 1)
        per_frame_psnr.append(psnr(a, b).cpu())
    per_frame_psnr = torch.cat(per_frame_psnr)
    for start in range(0, len(generated), chunk_frames):
        result[f"chunk{start // chunk_frames:03d}_psnr"] = per_frame_psnr[start : start + chunk_frames].mean().item()
    last = (len(generated) - 1) // chunk_frames
    result["last_minus_first_psnr"] = result[f"chunk{last:03d}_psnr"] - result["chunk000_psnr"]
    if len(generated) < 2:
        return result

    errors, generated_motion, target_motion = [], [], []
    for start in range(1, len(generated), FRAME_CHUNK):
        end = min(start + FRAME_CHUNK, len(generated))
        a = generated[start - 1 : end].to(device=device, dtype=torch.float32).clamp(0, 1)
        b = target[start - 1 : end].to(device=device, dtype=torch.float32).clamp(0, 1)
        da, db = a[1:] - a[:-1], b[1:] - b[:-1]
        errors.append((da - db).abs().mean(dim=(1, 2, 3)).cpu())
        generated_motion.append(da.abs().mean(dim=(1, 2, 3)).cpu())
        target_motion.append(db.abs().mean(dim=(1, 2, 3)).cpu())
    error = torch.cat(errors)
    result["temporal_delta_l1"] = error.mean().item()
    result["motion_generated"] = torch.cat(generated_motion).mean().item()
    result["motion_gt"] = torch.cat(target_motion).mean().item()
    if result["motion_gt"] > 1e-8:
        result["motion_ratio"] = result["motion_generated"] / result["motion_gt"]
    if len(error) >= chunk_frames:
        boundaries = torch.arange(chunk_frames - 1, len(error), chunk_frames)
        result["boundary_delta_l1"] = error[boundaries].mean().item()
    if sampler_chunk_latents is not None:
        if sampler_chunk_latents <= 0 or temporal_scale <= 0:
            raise ValueError("Sampler chunk size and temporal scale must be positive")
        # Latent 0 owns one raw frame; subsequent latents own temporal_scale frames.
        # For 8 latents/chunk and scale 4 the first boundary is raw frame 29, not 32.
        first = (sampler_chunk_latents - 1) * temporal_scale
        if first < len(error):
            boundaries = torch.arange(first, len(error), sampler_chunk_latents * temporal_scale)
            result["sampler_boundary_delta_l1"] = error[boundaries].mean().item()
    return result


# Bookkeeping, not quality; never averaged.
NON_METRIC_KEYS = frozenset({"scoring_seconds", "num_repeat_pairs"})


def mean_over_metrics(scored: Iterable[dict[str, float]]) -> dict:
    """Average each metric across scored clips, skipping the bookkeeping keys."""
    totals: dict[str, list[float]] = {}
    for metrics in scored:
        for name, value in metrics.items():
            if name in NON_METRIC_KEYS:
                continue
            totals.setdefault(name, []).append(value)
    return {name: sum(values) / len(values) for name, values in totals.items()}


def gain_metrics(direct: dict, baseline: dict) -> dict:
    """``direct`` minus ``baseline`` for PSNR and SSIM; both are higher-is-better, so a positive
    gain means the generated video beat the render it was conditioned on."""
    return {
        f"{name}_gain": direct[name] - baseline[name]
        for name in ("psnr", "ssim")
        if name in direct and name in baseline
    }
