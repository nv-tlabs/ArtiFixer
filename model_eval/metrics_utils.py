# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from data_processing.scene_utils import (
    downsampled_image_path,
    load_scene_transforms_from_dir,
    load_scene_transforms_from_zip,
    scene_zip_member,
)
from model_eval.image_metrics import psnr


def _scene_frame_root(dl3dv_dir: Path, scene_id: str) -> Path | None:
    scene_dir = dl3dv_dir / scene_id
    transforms_path = scene_dir / "transforms.json"
    if transforms_path.is_file():
        return scene_dir
    return None


def load_image_as_tensor(source: Path | str | bytes | io.BytesIO, device: torch.device) -> torch.Tensor:
    """Load image and convert to (1, C, H, W) tensor normalized to [0, 1].

    Accepts a filesystem path or raw image bytes (for zip-sourced frames).
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as image:
        img = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0).to(device)


def _resolve_zip_root(zf: ZipFile, scene_id: str) -> str:
    """Return the scene-root member prefix used by the shared DL3DV zip loader."""
    _, scene_root = load_scene_transforms_from_zip(zf, scene_id)
    return scene_root


def load_dl3dv_scene_zip_bytes(
    dl3dv_dir: Path, subdir: str, scene_id: str, relative_paths: list[str]
) -> dict[str, bytes]:
    """Read files from a DL3DV zip or extracted scene root, keyed by relative path."""
    frame_root = _scene_frame_root(dl3dv_dir, scene_id)
    if frame_root is not None:
        return {rel: (frame_root / rel).read_bytes() for rel in relative_paths}

    zip_path = dl3dv_dir / subdir / f"{scene_id}.zip"
    with ZipFile(zip_path, "r") as zf:
        scene_root = _resolve_zip_root(zf, scene_id)
        return {rel: zf.read(scene_zip_member(scene_root, rel)) for rel in relative_paths}


def load_dl3dv_scene_transforms(dl3dv_dir: Path, subdir: str, scene_id: str) -> dict:
    """Read transforms.json from a DL3DV zip or extracted scene root."""
    frame_root = _scene_frame_root(dl3dv_dir, scene_id)
    if frame_root is not None:
        return load_scene_transforms_from_dir(frame_root)

    zip_path = dl3dv_dir / subdir / f"{scene_id}.zip"
    with ZipFile(zip_path, "r") as zf:
        transforms, _ = load_scene_transforms_from_zip(zf, scene_id)
        return transforms


def load_subdir_by_scene_id(split_path: Path) -> dict[str, str]:
    """Return a mapping of scene_id -> DL3DV subdir (e.g. "1K") derived from the split file."""
    with open(split_path) as f:
        split = json.load(f)
    mapping: dict[str, str] = {}
    for bucket in ("test", "trainval"):
        for scene_id, recons in split[bucket].items():
            if recons:
                mapping[scene_id] = recons[0]["subdir"]
    return mapping


def validate_frame_coverage(frame_indices: list[int], expected_indices: set[int], *, label: str) -> None:
    actual_indices = set(frame_indices)
    if len(actual_indices) != len(frame_indices):
        raise ValueError(f"Duplicate frame IDs in {label}")
    if not expected_indices:
        raise ValueError(f"No evaluation frames expected for {label}")
    missing = sorted(expected_indices - actual_indices)
    unexpected = sorted(actual_indices - expected_indices)
    if missing or unexpected:
        raise ValueError(
            f"Incomplete frame coverage for {label}: expected {len(expected_indices)}, found {len(actual_indices)}; "
            f"missing {missing[:5]}, unexpected {unexpected[:5]}"
        )


class MetricsAggregator:
    """Aggregates per-frame metrics across scenes."""

    def __init__(self, metric_names: list[str]):
        self.metric_names = metric_names
        self.scene_metrics: dict[str, dict[str, list[float]]] = {}

    def add(self, scene_id: str, **metrics):
        scene_metrics = self.scene_metrics.setdefault(scene_id, defaultdict(list))
        for name, value in metrics.items():
            scene_metrics[name].append(value)

    def print_scene_summary(self, scene_id: str, prefix: str = ""):
        metrics = self.scene_metrics.get(scene_id)
        if not metrics:
            return
        parts = [f"{name}: {np.mean(values):.4f}" for name in self.metric_names if (values := metrics.get(name))]
        if parts:
            print(f"  {prefix}{', '.join(parts)}")

    def print_overall_summary(self, prefix: str = ""):
        for name in self.metric_names:
            all_vals = [v for scene in self.scene_metrics.values() for v in scene.get(name, [])]
            if all_vals:
                print(f"{prefix}{name}: {np.mean(all_vals):.4f}")

    def get_all_values(self, metric_name: str) -> list[float]:
        return [v for scene in self.scene_metrics.values() for v in scene.get(metric_name, [])]

    def require_values(self, metric_name: str, path: Path, *, evalset: str) -> None:
        if self.get_all_values(metric_name):
            return
        raise ValueError(
            f"No DL3DV frames were processed for evalset {evalset!r} at {path}. "
            "Check the eval output directory, num_views label, and scene list."
        )

    def to_results_dict(self, include_per_frame: bool = False) -> dict:
        results = {
            "scenes": {},
            "summary": {},
        }
        for scene_id, metrics in self.scene_metrics.items():
            scene_data = {}
            for name in self.metric_names:
                values = metrics.get(name)
                if values:
                    scene_data[f"{name}_mean"] = float(np.mean(values))
                    if include_per_frame:
                        scene_data[f"{name}_per_frame"] = values
            results["scenes"][scene_id] = scene_data

        for name in self.metric_names:
            all_vals = self.get_all_values(name)
            if all_vals:
                results["summary"][name] = float(np.mean(all_vals))

        return results

    def save_to_yaml(
        self, path: Path, include_per_frame: bool = False, *, protocol: dict | None = None, **extra_metrics: Any
    ):
        results = self.to_results_dict(include_per_frame)
        if protocol is not None:
            results["protocol"] = protocol
        for key, value in extra_metrics.items():
            if value is not None:
                results["summary"][key] = float(value)
        with open(path, "w") as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to {path}")


# GenFusion PSNR implementation
def psnr_genfusion(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """PSNR from GenFusion: https://github.com/Inception3D/GenFusion"""
    return psnr(img1, img2).unsqueeze(1)


def lpips_protocol_metadata(lpips_protocol: str, backbone: str = "vgg") -> dict:
    """Describe the LPIPS convention saved alongside benchmark scores."""
    if lpips_protocol not in {"standard", "legacy"}:
        raise ValueError(f"Unknown LPIPS protocol: {lpips_protocol}")
    return {
        "lpips_protocol": lpips_protocol,
        "lpips_backbone": backbone,
        "lpips_input_range": [-1, 1] if lpips_protocol == "standard" else [0, 1],
    }


def compute_rgb_metrics(
    image: torch.Tensor, target: torch.Tensor, lpips_net_type: str = "vgg", *, lpips_protocol: str
) -> dict[str, float]:
    """Score [0,1] RGB; standard LPIPS converts to its required [-1,1] range.

    Every caller must select its protocol explicitly. Evaluators must record the
    selection and rescore comparison baselines before changing it.
    """
    from model_eval.genfusion_lossutil import ssim
    from model_eval.lpipsPytorch import lpips

    if lpips_protocol not in {"standard", "legacy"}:
        raise ValueError(f"Unknown LPIPS protocol: {lpips_protocol}")
    perceptual_image, perceptual_target = (
        (2 * image - 1, 2 * target - 1) if lpips_protocol == "standard" else (image, target)
    )

    return {
        "psnr": psnr_genfusion(image, target).item(),
        "ssim": ssim(image, target).item(),
        "lpips": lpips(perceptual_image, perceptual_target, net_type=lpips_net_type).item(),
    }


class FIDTracker:
    """Tracks and computes FID score across all images.

    Images are fed to the FID metric in fixed-size chunks, so host memory holds
    at most chunk_size images at a time. The metric only accumulates fixed-size
    Inception feature statistics, so the score is the same as if every image
    were added at the end.
    """

    def __init__(self, device: torch.device, resize_to: int = 512, chunk_size: int = 64):
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.device = device
        self.resize = T.Resize((resize_to, resize_to))
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True, compute_on_cpu=True).to(device)
        self.chunk_size = chunk_size
        self.num_images = 0
        self.pred_images: list[torch.Tensor] = []
        self.gt_images: list[torch.Tensor] = []

    def _flush(self):
        if not self.pred_images:
            return
        self.fid.update(torch.cat(self.pred_images).to(self.device), real=False)
        self.fid.update(torch.cat(self.gt_images).to(self.device), real=True)
        self.pred_images.clear()
        self.gt_images.clear()

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """Add a pred/gt image pair (both [1, C, H, W] tensors)."""
        self.pred_images.append(self.resize(pred.detach().cpu()))
        self.gt_images.append(self.resize(gt.detach().cpu()))
        self.num_images += 1
        if len(self.pred_images) >= self.chunk_size:
            self._flush()

    def compute(self) -> float | None:
        """Compute FID score. Returns None if no images were added."""
        if self.num_images == 0:
            return None
        self._flush()
        return self.fid.compute().item()


def load_visibility_mask(
    mask_path: Path, target_size: tuple[int, int], device: torch.device, threshold: int = 1
) -> torch.Tensor:
    """Load a visibility mask, resize to (W, H) with nearest resampling, and threshold count values."""
    resample_nearest = getattr(Image, "Resampling", Image).NEAREST
    mask_image = Image.open(mask_path).convert("L").resize(target_size, resample_nearest)
    mask = torch.from_numpy(np.array(mask_image)).to(device)
    return (mask >= threshold).float().unsqueeze(0).unsqueeze(0)


def build_sorted_test_mask_index(transforms: dict, test_indices: set) -> dict[int, int]:
    """Map transforms.json frame ids to compact visibility-mask ids.

    Visibility masks are numbered by sorted test frame name, while rendered
    predictions are named by transforms.json frame index.
    """
    test_frames = []
    for frame_idx, frame in enumerate(transforms["frames"]):
        if frame_idx not in test_indices:
            continue
        test_frames.append((frame_idx, os.path.basename(frame["file_path"])))
    test_frames.sort(key=lambda item: item[1])
    return {frame_idx: mask_idx for mask_idx, (frame_idx, _) in enumerate(test_frames)}


def _make_masked_lpips_fn(
    device: torch.device, backend: str, *, lpips_protocol: str
) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """Build the masked-LPIPS callable used by MetricSuite."""
    if backend == "torchmetrics":
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        # This backend already normalized correctly in historical evaluations.
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

        def masked_lpips(image: torch.Tensor, target: torch.Tensor) -> float:
            return metric(image.clamp(0, 1), target.clamp(0, 1)).detach().cpu().item()

    else:
        from model_eval.lpipsPytorch import lpips

        def masked_lpips(image: torch.Tensor, target: torch.Tensor) -> float:
            image, target = image.clamp(0, 1), target.clamp(0, 1)
            if lpips_protocol == "standard":
                image, target = 2 * image - 1, 2 * target - 1
            return lpips(image, target, net_type=backend).item()

    return masked_lpips


class MetricSuite:
    """Paired ours-vs-3D-GUT metric trackers shared by the benchmark metric scripts.

    Tracks unmasked and visibility-masked PSNR/SSIM/LPIPS plus FID for both the
    model output ("ours") and the 3D-GUT reconstruction baseline.
    """

    def __init__(
        self,
        device: torch.device,
        use_masks: bool,
        masked_lpips_backend: str = "torchmetrics",
        *,
        lpips_protocol: str = "standard",
    ):
        from torchmetrics import PeakSignalNoiseRatio

        self.protocol = lpips_protocol_metadata(lpips_protocol)
        self.lpips_protocol = lpips_protocol
        self.masked_lpips_backend = masked_lpips_backend
        metric_names = ["psnr", "ssim", "lpips"]
        self.use_masks = use_masks
        self.ours = MetricsAggregator(metric_names)
        self.gut = MetricsAggregator(metric_names)
        self.ours_masked = MetricsAggregator(metric_names)
        self.gut_masked = MetricsAggregator(metric_names)
        self.ours_fid = FIDTracker(device)
        self.gut_fid = FIDTracker(device)
        self.ours_fid_masked = FIDTracker(device) if use_masks else None
        self.gut_fid_masked = FIDTracker(device) if use_masks else None
        self._masked_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device) if use_masks else None
        self._masked_lpips = (
            _make_masked_lpips_fn(device, masked_lpips_backend, lpips_protocol=lpips_protocol) if use_masks else None
        )

    def update(self, scene_id: str, pred: torch.Tensor, rendered: torch.Tensor, gt: torch.Tensor) -> None:
        """Record unmasked PSNR/SSIM/LPIPS and FID for one pred/rendered/gt frame."""
        self.ours.add(scene_id, **compute_rgb_metrics(pred, gt, lpips_protocol=self.lpips_protocol))
        self.gut.add(scene_id, **compute_rgb_metrics(rendered, gt, lpips_protocol=self.lpips_protocol))
        self.ours_fid.update(pred, gt)
        self.gut_fid.update(rendered, gt)

    def update_masked(
        self, scene_id: str, pred: torch.Tensor, rendered: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Record visibility-masked PSNR/SSIM/LPIPS and FID for one frame."""
        if not self.use_masks:
            raise ValueError("Masked metrics are disabled")
        pred_m, rendered_m, gt_m = pred * mask, rendered * mask, gt * mask
        self._add_masked(self.ours_masked, scene_id, pred_m, gt_m)
        self._add_masked(self.gut_masked, scene_id, rendered_m, gt_m)
        self.ours_fid_masked.update(pred_m, gt_m)
        self.gut_fid_masked.update(rendered_m, gt_m)

    def _add_masked(self, aggregator: MetricsAggregator, scene_id: str, image_m: torch.Tensor, gt_m: torch.Tensor):
        from torchmetrics.functional import structural_similarity_index_measure as masked_ssim

        aggregator.add(
            scene_id,
            psnr=self._masked_psnr(image_m, gt_m).cpu().item(),
            ssim=masked_ssim(image_m, gt_m, data_range=1.0).cpu().item(),
            lpips=self._masked_lpips(image_m, gt_m),
        )

    def report(self, path: Path, metrics_suffix: str = "") -> None:
        """Save scores under path/metrics_{protocol}, preserving historical files."""
        ours_fid_score = self.ours_fid.compute()
        gut_fid_score = self.gut_fid.compute()
        ours_fid_masked_score = self.ours_fid_masked.compute() if self.use_masks else None
        gut_fid_masked_score = self.gut_fid_masked.compute() if self.use_masks else None
        have_masked = self.use_masks and self.ours_masked.get_all_values("psnr")

        print("\n=== Summary ===")
        self.gut.print_overall_summary(prefix="3D-GUT ")
        if gut_fid_score is not None:
            print(f"3D-GUT fid: {gut_fid_score:.4f}")
        self.ours.print_overall_summary()
        if ours_fid_score is not None:
            print(f"fid: {ours_fid_score:.4f}")

        if have_masked:
            print()
            self.gut_masked.print_overall_summary(prefix="3D-GUT (masked) ")
            if gut_fid_masked_score is not None:
                print(f"3D-GUT (masked) fid: {gut_fid_masked_score:.4f}")
            self.ours_masked.print_overall_summary(prefix="(masked) ")
            if ours_fid_masked_score is not None:
                print(f"(masked) fid: {ours_fid_masked_score:.4f}")

        suffix = f"_{metrics_suffix}" if metrics_suffix else ""
        output_dir = path / f"metrics_{self.lpips_protocol}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.ours.save_to_yaml(
            output_dir / f"metrics{suffix}.yaml", include_per_frame=True, protocol=self.protocol, fid=ours_fid_score
        )
        self.gut.save_to_yaml(
            output_dir / f"metrics_3dgut{suffix}.yaml",
            include_per_frame=True,
            protocol=self.protocol,
            fid=gut_fid_score,
        )
        if have_masked:
            masked_protocol = {
                **self.protocol,
                "masked_lpips_backend": self.masked_lpips_backend,
                "lpips_backbone": "alex" if self.masked_lpips_backend == "torchmetrics" else self.masked_lpips_backend,
            }
            if self.masked_lpips_backend == "torchmetrics":
                masked_protocol["lpips_input_range"] = [-1, 1]
            self.ours_masked.save_to_yaml(
                output_dir / f"metrics_masked{suffix}.yaml", protocol=masked_protocol, fid=ours_fid_masked_score
            )
            self.gut_masked.save_to_yaml(
                output_dir / f"metrics_3dgut_masked{suffix}.yaml", protocol=masked_protocol, fid=gut_fid_masked_score
            )
