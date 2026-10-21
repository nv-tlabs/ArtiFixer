# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute MEt3R multi-view consistency on consecutive frame pairs.

Supports ArtiFixer evaluation outputs and comparison methods.
Each invocation evaluates one source (pred or rendered).

Usage:
    # ArtiFixer outputs on DL3DV
    python -m model_eval.compute_met3r \
        --dataset dl3dv --method ours \
        --results_dir /path/to/distilled_views_dl3dv_ours \
        --scene_list_file data_processing/dl3dv_benchmark_ours_test_scenes.txt \
        --output_json /path/to/distilled_views_dl3dv_ours/met3r_pred.json

    # Comparison method outputs (genfusion, difix3dplus, difix3d, gen3c)
    python -m model_eval.compute_met3r \
        --dataset dl3dv --method genfusion \
        --results_dir /path/to/GenFusion-outputs/E6_our_dl3dv \
        --scene_list_file data_processing/dl3dv_benchmark_ours_test_scenes.txt \
        --output_json /path/to/GenFusion-outputs/E6_our_dl3dv/met3r_pred.json

    # Multiple MEt3R variants in one run
    python -m model_eval.compute_met3r \
        --dataset dl3dv --method ours \
        --results_dir /path/to/distilled_views_dl3dv_ours \
        --scene_list_file data_processing/dl3dv_benchmark_ours_test_scenes.txt \
        --output_json /path/to/distilled_views_dl3dv_ours/met3r_pred.json \
        --metrics mast3r:cosine raft:cosine mast3r:lpips mast3r:ssim raft:lpips raft:ssim
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from data_processing.nerfbusters import NERFBUSTERS_SCENES
from data_processing.scene_utils import read_scene_list

VALID_BACKBONES = {"mast3r", "dust3r", "raft"}
VALID_DISTANCES = {"cosine", "lpips", "ssim", "psnr", "rmse", "mse"}
VALID_FEATURE_BACKBONES = {"dino16", "dinov2", "clip", "maskclip", "vit", "resnet50"}

DEFAULT_METRICS = ["mast3r:cosine"]

VALID_DATASETS = ["dl3dv", "nerfbusters"]
VALID_METHODS = ["ours", "difix", "genfusion", "difix3dplus", "difix3d", "gen3c"]

EVALSET_TO_SCENE_LIST = {
    "dl3dv_difix": "dl3dv_benchmark_difix_test_scenes.txt",
    "dl3dv_ours": "dl3dv_benchmark_ours_test_scenes.txt",
    "genfusion": "dl3dv_benchmark_ours_test_scenes.txt",
    "difix3dplus": "dl3dv_benchmark_ours_test_scenes.txt",
    "difix3d": "dl3dv_benchmark_ours_test_scenes.txt",
    "gen3c": "dl3dv_benchmark_ours_test_scenes.txt",
}


def resolve_evalset(dataset: str, method: str) -> str:
    """Map the (dataset, method) pair onto the output layout key."""
    if dataset == "nerfbusters":
        if method != "ours":
            raise ValueError(f"--method {method} is only supported with --dataset dl3dv")
        return dataset
    if method == "ours":
        return "dl3dv_ours"
    if method == "difix":
        return "dl3dv_difix"
    return method


def parse_metric_specs(specs: list[str]) -> list[tuple[str, str]]:
    """Parse 'backbone:distance' strings, validating each component."""
    parsed = []
    seen = set()
    for s in specs:
        parts = s.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid metric '{s}', expected 'backbone:distance' " f"(e.g. mast3r:cosine, raft:lpips)"
            )
        backbone, distance = parts
        if backbone not in VALID_BACKBONES:
            raise argparse.ArgumentTypeError(f"Unknown backbone '{backbone}', choices: {sorted(VALID_BACKBONES)}")
        if distance not in VALID_DISTANCES:
            raise argparse.ArgumentTypeError(f"Unknown distance '{distance}', choices: {sorted(VALID_DISTANCES)}")
        key = f"{backbone}:{distance}"
        if key not in seen:
            parsed.append((backbone, distance))
            seen.add(key)
    return parsed


def build_metrics(
    specs: list[tuple[str, str]],
    img_size: int,
    feature_backbone: str,
    device: torch.device,
) -> dict[str, Any]:
    """Instantiate one MEt3R model per unique (backbone, distance) pair."""
    from met3r import MEt3R

    metrics: dict[str, Any] = {}
    for backbone, distance in specs:
        key = f"{backbone}:{distance}"
        kwargs: dict[str, Any] = dict(
            img_size=img_size,
            backbone=backbone,
            distance=distance,
            freeze=True,
        )
        if feature_backbone != "dino16":
            kwargs["feature_backbone"] = feature_backbone
        print(f"  Initializing {key}...")
        metrics[key] = MEt3R(**kwargs).to(device)
    return metrics


def load_benchmark_scene_ids(evalset: str) -> list[str] | None:
    if evalset == "nerfbusters":
        return list(NERFBUSTERS_SCENES)
    scene_file = EVALSET_TO_SCENE_LIST.get(evalset)
    if scene_file is None:
        return None
    script_dir = Path(__file__).resolve().parent.parent
    return read_scene_list(script_dir / "data_processing" / scene_file)


def preprocess_tensor(t: torch.Tensor, img_size: int) -> torch.Tensor:
    """Resize and normalize a [0,1] tensor to [-1,1]."""
    t = TF.resize(t, [img_size, img_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return t * 2 - 1


def load_and_preprocess(image_paths: list[str], img_size: int, device: torch.device) -> torch.Tensor:
    """Load images from paths, resize to (img_size, img_size), normalize to [-1, 1]."""
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        t = TF.to_tensor(img)
        tensors.append(preprocess_tensor(t, img_size))
    return torch.stack(tensors).to(device)


def load_gen3c_video_frames(video_path: Path, img_size: int, device: torch.device) -> torch.Tensor | None:
    """Extract Gen3C frames: every 4th video frame, right 1/3 crop (the generated view)."""
    from model_training.utils.video_io import iter_video_frames

    tensors = []
    for i, frame in enumerate(iter_video_frames(video_path)):
        if i % 4 == 0:
            crop = frame[:, -frame.shape[1] // 3 :]
            img = Image.fromarray(crop).convert("RGB")
            t = TF.to_tensor(img)
            tensors.append(preprocess_tensor(t, img_size))
    return torch.stack(tensors).to(device) if len(tensors) >= 2 else None


def form_consecutive_pairs(images: torch.Tensor) -> torch.Tensor:
    """Given (N, C, H, W), return (N-1, 2, C, H, W) consecutive pairs."""
    return torch.stack([images[:-1], images[1:]], dim=1)


def compute_met3r_batched(metric, pairs: torch.Tensor, batch_size: int) -> list[float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        score, *_ = metric(images=batch)
        scores.extend(score.cpu().tolist())
    return scores


# ---------------------------------------------------------------------------
# Scene frame loading per evalset / source
# ---------------------------------------------------------------------------


def load_scene_frames(
    results_dir: Path,
    scene: str,
    evalset: str,
    source: str,
    img_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Load all frames for a scene as (N, C, H, W) tensor normalized to [-1,1].

    Handles different directory layouts per evalset and source.
    """
    scene_dir = results_dir / scene

    if evalset in ("dl3dv_difix", "dl3dv_ours", "nerfbusters"):
        img_dir = scene_dir / "frames" / "batch_0000" / source
        if not img_dir.exists():
            return None
        paths = sorted(glob.glob(str(img_dir / "*.png")))

    elif evalset == "genfusion":
        train_dir = scene_dir / "train" / "renders"
        test_dir = scene_dir / "test" / "renders"
        paths = []
        if train_dir.exists():
            paths.extend(glob.glob(str(train_dir / "*.png")))
        if test_dir.exists():
            paths.extend(glob.glob(str(test_dir / "*.png")))
        paths = sorted(paths, key=lambda p: os.path.basename(p))

    elif evalset == "difix3dplus":
        img_dir = scene_dir / "46000"
        if not img_dir.exists():
            return None
        paths = sorted(glob.glob(str(img_dir / "*.png")))

    elif evalset == "difix3d":
        img_dir = scene_dir / "images"
        if not img_dir.exists():
            return None
        paths = sorted(glob.glob(str(img_dir / "*.png")))

    elif evalset == "gen3c":
        video_path = scene_dir / "video.mp4"
        if not video_path.exists():
            return None
        return load_gen3c_video_frames(video_path, img_size, device)

    else:
        return None

    if len(paths) < 2:
        return None
    return load_and_preprocess(paths, img_size, device)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_eval(metrics_dict: dict[str, Any], args, device):
    entries = discover_scene_entries(args.results_dir, args.evalset, args.scene_list_file)

    if not entries:
        raise ValueError(f"No evaluation entries found in {args.results_dir}")
    if not metrics_dict:
        raise ValueError("At least one metric is required")

    metric_keys = list(metrics_dict.keys())
    print(f"Found {len(entries)} entries, evaluating {len(metric_keys)} metric(s): {metric_keys}")

    all_scores: dict[str, list[float]] = {k: [] for k in metric_keys}
    scene_scores: dict[str, dict[str, Any]] = {k: {} for k in metric_keys}
    frame_counts: dict[str, int] = {}

    for entry in tqdm(entries, desc=f"MEt3R ({args.evalset} {args.source})"):
        scene = entry["scene"]
        label = scene
        if label in frame_counts:
            raise ValueError(f"Duplicate evaluation entry: {label}")

        with torch.inference_mode():
            images = load_scene_frames(
                args.results_dir,
                scene,
                args.evalset,
                args.source,
                args.img_size,
                device,
            )
            if images is None or len(images) < 2:
                raise ValueError(f"Evaluation entry {label} must contain at least two frames for source {args.source}")

            pairs = form_consecutive_pairs(images)
            n_frames = len(images)
            del images
            parts = [f"  {label} ({n_frames} frames):"]

            for key, metric in metrics_dict.items():
                scores = compute_met3r_batched(metric, pairs, args.batch_size)
                if len(scores) != len(pairs) or not all(math.isfinite(score) for score in scores):
                    raise ValueError(f"Invalid scores from {key} for {label}: expected {len(pairs)} finite pair scores")
                mean_score = sum(scores) / len(scores)
                scene_scores[key][scene] = round(mean_score, 6)
                all_scores[key].append(mean_score)
                parts.append(f"{key}={mean_score:.4f}")

            frame_counts[label] = n_frames
            del pairs
            torch.cuda.empty_cache()

        print(" ".join(parts))

    results: dict[str, Any] = {
        "evalset": args.evalset,
        "source": args.source,
        "coverage": {
            "expected_entries": len(entries),
            "evaluated_entries": len(frame_counts),
            "frames_per_entry": frame_counts,
        },
        "metrics": {},
    }
    print(f"\n--- Overall ({args.evalset} {args.source}) ---")
    for key in metric_keys:
        scores = all_scores[key]
        overall = sum(scores) / len(scores)
        results["metrics"][key] = {
            "scenes": scene_scores[key],
            "overall": round(overall, 6),
        }
        print(f"  {key}: {overall:.4f}")

    output_path = args.output_json
    if output_path is None:
        output_path = args.results_dir / f"met3r_{args.source}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def discover_scene_entries(
    results_dir: Path, evalset: str, scene_list_file: Path | None = None
) -> list[dict[str, Any]]:
    if scene_list_file is not None:
        benchmark_scenes = read_scene_list(scene_list_file)
    else:
        benchmark_scenes = load_benchmark_scene_ids(evalset)
    if benchmark_scenes is None:
        print(f"Warning: no benchmark scene list for evalset={evalset}, using all subdirs")
        benchmark_scenes = sorted(d.name for d in results_dir.iterdir() if d.is_dir())

    missing = [scene for scene in benchmark_scenes if not (results_dir / scene).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing expected scenes in {results_dir}: {', '.join(missing)}")
    return [{"scene": scene} for scene in benchmark_scenes]


def main():
    parser = argparse.ArgumentParser(description="Compute MEt3R multi-view consistency")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=VALID_DATASETS, help="Dataset whose scenes are evaluated"
    )
    parser.add_argument(
        "--results_dir", type=Path, required=True, help="Directory holding per-scene outputs of the evaluated method"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ours",
        choices=VALID_METHODS,
        help="Method whose outputs populate --results_dir "
        "(selects the on-disk layout; comparison methods are dl3dv-only)",
    )
    parser.add_argument(
        "--scene_list_file",
        type=Path,
        default=None,
        help="File with one scene id per line restricting which scenes are "
        "evaluated (default: the benchmark list for the dataset/method)",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Where to write the metrics JSON " "(default: <results_dir>/met3r_<source>.json)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="pred",
        choices=["pred", "rendered"],
        help="Which image source to evaluate (ignored for comparison methods)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=DEFAULT_METRICS,
        help="backbone:distance pairs to evaluate " "(e.g. mast3r:cosine raft:lpips mast3r:ssim)",
    )
    parser.add_argument(
        "--feature_backbone",
        type=str,
        default="dino16",
        choices=sorted(VALID_FEATURE_BACKBONES),
        help="Feature backbone for cosine distance (default: dino16)",
    )
    args = parser.parse_args()

    try:
        args.evalset = resolve_evalset(args.dataset, args.method)
    except ValueError as exc:
        parser.error(str(exc))

    metric_specs = parse_metric_specs(args.metrics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing MEt3R metrics (img_size={args.img_size}, " f"feature_backbone={args.feature_backbone})...")
    metrics_dict = build_metrics(metric_specs, args.img_size, args.feature_backbone, device)

    run_eval(metrics_dict, args, device)


if __name__ == "__main__":
    main()
