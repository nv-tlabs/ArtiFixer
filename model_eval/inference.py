# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inference script for eval.
Distributes scenes across GPUs for parallel inference.

Usage:
    torchrun --nproc_per_node=4 --module model_eval.run_inference \
        --checkpoint /path/to/checkpoints/checkpoint_10000/pytorch_model_fsdp_0 \
        --save_dir /path/to/output \
        --render_name checkpoint_10000_pytorch_model_fsdp_0_chunk7_sink7 \
        --evalset 3dgrut_dl3dv_ours
"""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.distributed import DeviceMesh
from tqdm import tqdm

from model_eval.checkpoint_loading import eval_pipe_build_context, load_transformer_checkpoint
from model_eval.datasets.dl3dv_hdf5_eval import DL3DVHDF5EvalDataset
from model_eval.datasets.dl3dv_reconstruction_eval import create_dl3dv_reconstruction_dataset
from model_eval.datasets.nerfbusters_eval import NerfbustersEvalDataset
from model_eval.datasets.reconstructed_colmap_eval import create_reconstructed_colmap_dataset
from model_eval.dl3dv_reconstruction_evalsets import is_dl3dv_reconstruction_evalset
from model_eval.generation import SamplingConfig, generate, prepare_generation_inputs
from model_eval.reconstructed_colmap_evalsets import is_reconstructed_colmap_evalset, num_views_output_label
from model_training.data.utils import NeighborSelectionMode
from model_training.model_profiles import is_cosmos3_model, prompt_format_for_model, spatial_resize_multiple
from model_training.net.conditioning import validate_source_camera_conditioning_lengths
from model_training.prompt_formats import substitute_unconditional_prompt
from model_training.utils.precision import disable_tf32_matmul
from model_training.utils.train_utils import barrier_if_distributed, get_kv_cache_pipe, get_pipe
from model_training.utils.video_io import atomic_output_path, save_video, video_file_complete

DIAGNOSTIC_NAMES = ("no_rendered_rgb", "text_only", "no_text")

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"


def init_distributed(timeout_minutes: int = 60):
    """Initialize distributed evaluation if available."""
    if "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(minutes=timeout_minutes),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def compute_frame_padding(num_frames: int, frames_per_block: int, vae_temporal_scale: int = 4) -> int:
    """Compute how many frames to pad to satisfy VAE and frames_per_block requirements."""
    original = num_frames
    num_frames += (vae_temporal_scale - (num_frames - 1) % vae_temporal_scale) % vae_temporal_scale
    latent_frames = (num_frames - 1) // vae_temporal_scale + 1
    padded_latent = math.ceil(latent_frames / frames_per_block) * frames_per_block
    padded_num_frames = (padded_latent - 1) * vae_temporal_scale + 1
    return padded_num_frames - original


def latent_num_frames_from_rgb_num_frames(num_frames: int, vae_temporal_scale: int = 4) -> int:
    return math.ceil((num_frames - 1) / vae_temporal_scale) + 1


def pad_temporal(tensor: torch.Tensor, pad_frames: int, dim: int = 1) -> torch.Tensor:
    """Pad tensor along temporal dimension by repeating last frame."""
    if pad_frames <= 0:
        return tensor
    last_frame = tensor.select(dim, -1).unsqueeze(dim)
    padding = last_frame.expand(*[-1 if i != dim else pad_frames for i in range(tensor.dim())])
    return torch.cat([tensor, padding], dim=dim)


def save_image(frame: torch.Tensor, path: Path):
    """Save a single CHW tensor image."""
    image = (frame.cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    save_png(Image.fromarray(image), path)


def save_png(image: Image.Image, path: Path) -> None:
    """Keep a partial PNG out of the resumable output set."""
    with atomic_output_path(path) as temporary:
        image.save(temporary, format="PNG")


def png_complete(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return False
            image.verify()
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def _to_list(value) -> list:
    if torch.is_tensor(value):
        return value.cpu().tolist()
    return list(value)


def _valid_mask_list(item: dict) -> list[bool]:
    valid_mask = item.get("valid_frames_mask")
    if valid_mask is None:
        return [True] * int(item["rgb_rendered"].shape[0])
    return [bool(value) for value in _to_list(valid_mask)]


def _values_for_valid_frames(value, valid_mask: list[bool], name: str) -> list[int]:
    valid_count = sum(valid_mask)
    if value is None:
        return list(range(valid_count))

    values = [int(entry) for entry in _to_list(value)]
    if len(values) == len(valid_mask):
        return [entry for entry, keep in zip(values, valid_mask) if keep]
    assert len(values) == valid_count, f"{name} has {len(values)} entries; expected {len(valid_mask)} or {valid_count}"
    return values


def _eval_frame_metadata(item: dict) -> tuple[list[bool], list[int], list[int]]:
    valid_mask = _valid_mask_list(item)
    frame_indices = _values_for_valid_frames(item.get("frame_indices"), valid_mask, "frame_indices")
    output_source = item.get("gt_index")
    if output_source is None:
        output_source = item.get("frame_indices")
    output_indices = _values_for_valid_frames(output_source, valid_mask, "output_indices")
    return valid_mask, output_indices, frame_indices


def expected_output_frame_names(item: dict) -> list[str]:
    _, output_indices, _ = _eval_frame_metadata(item)
    return [f"{idx:05d}.png" for idx in output_indices if idx != -1]


def expected_video_frame_names(item: dict) -> list[str]:
    valid_mask = _valid_mask_list(item)
    frame_indices = _values_for_valid_frames(
        item.get("video_frame_indices", item.get("frame_indices")), valid_mask, "video_frame_indices"
    )
    return [f"{idx:05d}.png" for idx in frame_indices]


def output_kinds(item: dict) -> tuple[str, ...]:
    return ("pred", "gt", "rendered") if "rgb_gt" in item else ("pred", "rendered")


def output_frames_complete(scene_dir: Path, item: dict) -> bool:
    names = expected_output_frame_names(item)
    frames_dir = scene_dir / "frames" / "batch_0000"
    return all(png_complete(frames_dir / kind / name) for kind in output_kinds(item) for name in names)


def output_video_frames_complete(scene_dir: Path, item: dict) -> bool:
    names = expected_video_frame_names(item)
    video_frames_dir = scene_dir / "video_frames" / "batch_0000"
    return all(png_complete(video_frames_dir / kind / name) for kind in output_kinds(item) for name in names)


def diagnostic_output_key(name: str, chunk_idx: int | None) -> str:
    return name if chunk_idx is None else f"{name}_chunk{int(chunk_idx):04d}"


def diagnostics_complete(scene_dir: Path, item: dict) -> bool:
    frame_count = sum(_valid_mask_list(item))
    return all(
        video_file_complete(scene_dir / f"{diagnostic_output_key(name, item.get('chunk_idx'))}.mp4", frame_count)
        for name in DIAGNOSTIC_NAMES
    )


def create_dataset(args, rank: int):
    """Create the appropriate dataset based on evalset argument."""
    max_test_frames = args.bidirectional_chunk_size if args.inference_pipeline == "bidirectional" else None
    include_all_frames = args.render_trajectory == "all_frames"

    if args.evalset == "nerfbusters":
        selection_mode = NeighborSelectionMode(args.neighbor_selection_mode)
        return NerfbustersEvalDataset(
            nerfbusters_dir=args.nerfbusters_dir,
            recon_results_dir=args.nerfbusters_recon_results_dir,
            num_views=args.num_views,
            dataset_scaling_factor=args.dataset_scaling_factor,
            neighbor_selection_mode=selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            filter_scene_id=args.scene_id,
            checkpoint=args.nerfbusters_checkpoint,
            recon_experiment_name=args.nerfbusters_recon_experiment_name,
            image_folder=args.nerfbusters_image_folder,
            resize_multiple=spatial_resize_multiple(args.model_id),
            prompt_format=prompt_format_for_model(args.model_id),
            camera_pose_alignment=args.camera_pose_alignment,
            captions_dir=args.nerfbusters_captions_dir,
            verbose=(rank == 0),
        )
    elif args.evalset == "3dgrut_dl3dv_ours":
        selection_mode = NeighborSelectionMode(args.neighbor_selection_mode)
        return DL3DVHDF5EvalDataset(
            split="test",
            split_path=args.split_path,
            dl3dv_dir=args.dl3dv_dir,
            prompt_dir=args.prompt_dir,
            num_views=args.num_views,
            dataset_scaling_factor=args.dataset_scaling_factor,
            frames_per_block=args.frames_per_block,
            neighbor_selection_mode=selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            filter_scene_id=args.scene_id,
            resize_multiple=spatial_resize_multiple(args.model_id),
            prompt_format=prompt_format_for_model(args.model_id),
            camera_pose_alignment=args.camera_pose_alignment,
            verbose=(rank == 0),
        )
    elif is_dl3dv_reconstruction_evalset(args.evalset):
        selection_mode = NeighborSelectionMode(args.neighbor_selection_mode)
        return create_dl3dv_reconstruction_dataset(
            args=args,
            selection_mode=selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            rank=rank,
        )
    elif is_reconstructed_colmap_evalset(args.evalset):
        selection_mode = NeighborSelectionMode(args.neighbor_selection_mode)
        return create_reconstructed_colmap_dataset(
            args=args,
            selection_mode=selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            rank=rank,
        )
    assert False, f"Invalid evalset: {args.evalset}"


def trajectory_suffix(render_trajectory: str) -> str:
    return "" if render_trajectory == "val_frames" else f"_{render_trajectory}"


def output_suffix_part(output_suffix: str) -> str:
    return f"_{output_suffix}" if output_suffix else ""


def get_output_dir(args) -> Path:
    """--save_dir / --render_name / a per-evalset subtree; the subtree encodes only the dataset side of the run."""
    num_views = (
        num_views_output_label(args.num_views)
        if is_reconstructed_colmap_evalset(args.evalset) or is_dl3dv_reconstruction_evalset(args.evalset)
        else str(args.num_views)
    )
    if (
        args.evalset == "3dgrut_dl3dv_ours"
        or is_dl3dv_reconstruction_evalset(args.evalset)
        or is_reconstructed_colmap_evalset(args.evalset)
    ):
        mode_name = (
            f"distilled_views_{args.evalset}_{num_views}_{args.neighbor_selection_mode}"
            f"{trajectory_suffix(args.render_trajectory)}{output_suffix_part(args.output_suffix)}"
        )
        return args.save_dir / args.render_name / mode_name
    if args.evalset == "nerfbusters":
        mode_name = (
            f"nerfbusters_{num_views}_{args.neighbor_selection_mode}"
            f"{trajectory_suffix(args.render_trajectory)}{output_suffix_part(args.output_suffix)}"
        )
        return args.save_dir / args.render_name / mode_name
    assert False, f"Invalid evalset: {args.evalset}"


def get_eval_pipe(args: argparse.Namespace, device: torch.device):
    """Build the eval pipeline, load its transformer checkpoint and return it
    cast to bf16 and placed on ``device``.

    With a .pt checkpoint the transformer's parameters are created on the meta
    device (nothing is allocated on the host) and load_transformer_checkpoint
    materializes them straight onto ``device``; the final cast/move then only
    places the transformer's buffers (the VAE is built in bf16 on the device).
    A DCP checkpoint loads into the cast and placed model, so that happens
    before the load and the final cast/move is a no-op.
    """
    with eval_pipe_build_context(args) as transformer_on_meta:
        if args.inference_pipeline == "bidirectional":
            pipe = get_pipe(
                args,
                load_pretrained_transformer_weights=False,
                frames_per_block=None,
                device=device,
            )
        else:
            # The causal rollout of a distilled checkpoint; the dispatcher picks the family pipeline.
            pipe = get_kv_cache_pipe(args, device)
    if args.vae_tiling:
        # Only sets the VAE's tiling flags; the tiles are cut at encode/decode time.
        pipe.vae.enable_tiling()
    if not transformer_on_meta:
        pipe = pipe.to(torch.bfloat16).to(device)
    load_transformer_checkpoint(pipe.transformer, args, device=device, dtype=torch.bfloat16)
    return pipe.to(torch.bfloat16).to(device)


def create_context_parallel_meshes(
    rank: int,
    world_size: int,
    cp_group_size: int,
    device_type: str,
) -> tuple[DeviceMesh | None, int, int, int]:
    if cp_group_size <= 1:
        return None, world_size, 0, 0

    assert world_size % cp_group_size == 0, (
        f"world_size={world_size} must be divisible by context_parallel_size={cp_group_size}. "
        "This eval path does not allow leftover idle ranks."
    )

    num_groups = world_size // cp_group_size
    all_group_ranks = [list(range(g * cp_group_size, (g + 1) * cp_group_size)) for g in range(num_groups)]

    my_mesh = None
    # All ranks must construct the same sequence of meshes.
    for group_ranks in all_group_ranks:
        mesh = DeviceMesh(device_type, group_ranks)
        if rank in group_ranks:
            my_mesh = mesh

    group_index = rank // cp_group_size
    group_rank = rank % cp_group_size

    return my_mesh, num_groups, group_index, group_rank


def save_comparison_output(
    output: torch.Tensor,
    rgb_gt: torch.Tensor | None,
    rgb_rendered: torch.Tensor,
    rgb_neighbors: torch.Tensor,
    save_dir: Path,
    key: str = "comparison",
    fps: int = 15,
    save_frames: bool = True,
    output_indices: torch.Tensor | None = None,
):
    """
    Save comparison video/images with pred, optional gt, rendered side by side and neighbors below.

    Args:
        output: (T, C, H, W) predicted frames
        rgb_gt: optional (T, C, H, W) ground truth frames
        rgb_rendered: (T, C, H, W) rendered frames
        rgb_neighbors: (N, C, H, W) neighbor frames
        save_dir: directory to save outputs
        key: filename prefix
        fps: video framerate
        save_frames: whether to save individual PNG frames
        output_indices: optional (T,) tensor of dataset frame indices for metric PNG names;
            -1 means the frame stays in videos but has no metric PNG output
    """
    output = output.cpu().float().clamp(0, 1)
    if rgb_gt is not None:
        rgb_gt = rgb_gt.cpu().float().clamp(0, 1)
    rgb_rendered = rgb_rendered.cpu().float().clamp(0, 1)
    rgb_neighbors = rgb_neighbors.cpu().float().clamp(0, 1)

    output_size = output.shape[-2:]
    if rgb_gt is not None and rgb_gt.shape[-2:] != output_size:
        rgb_gt = F.interpolate(rgb_gt, size=output_size, mode="bilinear", align_corners=False)
    if rgb_rendered.shape[-2:] != output_size:
        rgb_rendered = F.interpolate(rgb_rendered, size=output_size, mode="bilinear", align_corners=False)
    if rgb_neighbors.shape[-2:] != output_size:
        rgb_neighbors = F.interpolate(
            rgb_neighbors,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

    first_row = [output]
    if rgb_gt is not None:
        first_row.append(rgb_gt)
    first_row.append(rgb_rendered)
    columns = len(first_row)

    def to_uint8_hwc(frame: torch.Tensor) -> np.ndarray:
        return (frame.permute(1, 2, 0) * 255).round().byte().cpu().numpy()

    # The neighbor rows are identical for every frame; build them once as uint8.
    neighbor_rows = []
    for neighbor_index in range(0, rgb_neighbors.shape[0], columns):
        neighbors_in_row = rgb_neighbors[neighbor_index : neighbor_index + columns]
        if neighbors_in_row.shape[0] < columns:
            padding = torch.zeros(
                columns - neighbors_in_row.shape[0], *neighbors_in_row.shape[1:], dtype=rgb_neighbors.dtype
            )
            neighbors_in_row = torch.cat([neighbors_in_row, padding], dim=0)
        neighbor_rows.append(torch.cat([n for n in neighbors_in_row], dim=-1))
    neighbor_block = to_uint8_hwc(torch.cat(neighbor_rows, dim=-2)) if neighbor_rows else None

    num_frames = output.shape[0]

    def grid_frame(frame_idx: int) -> np.ndarray:
        frame = to_uint8_hwc(torch.cat([row[frame_idx] for row in first_row], dim=-1))
        if neighbor_block is not None:
            frame = np.concatenate([frame, neighbor_block], axis=0)
        return frame

    save_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = None
    if save_frames:
        frames_dir = save_dir / f"{key}_all"
        frames_dir.mkdir(parents=True, exist_ok=True)

    def grid_frames():
        # Yield one uint8 grid frame at a time (saving PNGs along the way)
        # instead of building the full (T, H, W, 3) grid video in memory.
        for frame_idx in range(num_frames):
            frame = grid_frame(frame_idx)
            if frames_dir is not None:
                if output_indices is not None:
                    output_idx = int(output_indices[frame_idx])
                    fname = None if output_idx == -1 else f"{output_idx:05d}.png"
                else:
                    fname = f"{frame_idx:05d}.png"
                if fname is not None:
                    save_png(Image.fromarray(frame), frames_dir / fname)
            yield frame

    save_video(grid_frames(), save_dir / f"{key}.mp4", fps=fps)


def _item_seed(base_seed: int, item: dict) -> int:
    frame_indices = _values_for_valid_frames(item.get("frame_indices"), _valid_mask_list(item), "frame_indices")
    identity = (
        int(base_seed),
        str(item["scene_id"]),
        None if item.get("split") is None else str(item["split"]),
        frame_indices,
        None if item.get("chunk_idx") is None else int(item["chunk_idx"]),
    )
    return int.from_bytes(hashlib.sha256(json.dumps(identity).encode("utf-8")).digest()[:8], "big")


@contextmanager
def _inference_rng(seed: int, device: torch.device):
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(seed)
        yield


def process_item(pipe, item, args, output_dir, rank, device, vae_temporal_scale, save_outputs: bool = True):
    """
    Unified processing function for all evalsets.

    Item dict expected keys:
        - rgb_rendered: (T, C, H, W) rendered frames
        - rgb_gt (optional): (T, C, H, W) ground truth frames
        - rgb_neighbors: (N, C, H, W) neighbor frames
        - encoded_prompt: encoded text prompt
        - scene_id: scene identifier
        - split (optional): nested output group
        - frame_indices (optional): non-padding frame indices in output video order
        - valid_frames_mask (optional): mask selecting non-padding frames
        - gt_index (optional): output frame indices; -1 skips frame PNG output
        - target_h, target_w (optional): output PNG/video frame size
        - opacity, camera_rays, w2cs, Ks, etc.
    """
    scene_id = item["scene_id"]
    split = item.get("split")
    has_gt = "rgb_gt" in item

    if split is not None:
        scene_dir = output_dir / scene_id / str(split)
    else:
        scene_dir = output_dir / scene_id

    if not args.replace_if_exists and output_frames_complete(scene_dir, item):
        if args.save_frame_outputs_only or output_video_frames_complete(scene_dir, item):
            if args.save_frame_outputs_only or not args.render_diagnostics or diagnostics_complete(scene_dir, item):
                return

    if is_cosmos3_model(args.model_id) and item["rgb_neighbors"].shape[0] > 0:
        target_hw = tuple(item["rgb_rendered"].shape[-2:])
        reference_hw = tuple(item["rgb_neighbors"].shape[-2:])
        if target_hw != reference_hw:
            raise ValueError(
                f"Item {scene_id!r} (split={split!r}, chunk={item.get('chunk_idx')!r}): Cosmos3 "
                f"requires matching latent grids; target spatial shape is {target_hw}, reference shape is {reference_hw}."
            )

    rgb_neighbors_cpu = item["rgb_neighbors"].cpu() if save_outputs else None
    inputs = prepare_generation_inputs(
        item,
        device=device,
        dtype=torch.bfloat16,
        batched=False,
        stage_conditioning_on_cpu=pipe.stages_conditioning_per_chunk,
        stage_neighbors_on_cpu=args.max_neighbors_per_encode is not None,
    )
    if not is_cosmos3_model(args.model_id):
        inputs.pop("neighbor_camera_rays", None)
    rgb_rendered = inputs["rendered_rgb"]
    rgb_gt = item["rgb_gt"].unsqueeze(0) if has_gt else None
    rgb_neighbors = inputs["neighbors"]

    original_num_frames = rgb_rendered.shape[1]
    # Two padding purposes, kept separate: compute_frame_padding satisfies
    # the VAE stride and the rollout chunk size (needed at every CP size);
    # the pipeline's cp_raw_frame_padding adds CP divisibility only when its
    # transformer cannot pad its own frame axis.
    pad_frames = compute_frame_padding(original_num_frames, pipe.latent_frame_multiple(), vae_temporal_scale)
    pad_frames += pipe.cp_raw_frame_padding(
        original_num_frames + pad_frames,
        getattr(pipe.transformer, "_cp_world_size", 1),
        vae_temporal_scale,
    )
    target_num_frames = original_num_frames + pad_frames
    source_latent_num_frames = latent_num_frames_from_rgb_num_frames(original_num_frames, vae_temporal_scale)
    target_latent_num_frames = latent_num_frames_from_rgb_num_frames(target_num_frames, vae_temporal_scale)
    validate_source_camera_conditioning_lengths(
        camera_rays=item["camera_rays"],
        w2cs=item["w2cs"],
        Ks=item["Ks"],
        latent_num_frames=source_latent_num_frames,
    )
    latent_pad_frames = target_latent_num_frames - source_latent_num_frames

    if pad_frames > 0:
        rgb_rendered = pad_temporal(rgb_rendered, pad_frames, dim=1)

    opacity = inputs["rendered_opacity"]
    if pad_frames > 0:
        opacity = pad_temporal(opacity, pad_frames, dim=1)

    camera_rays = inputs["camera_rays"]
    if camera_rays.shape[2:4] != rgb_rendered.shape[-2:]:
        raise ValueError(
            f"camera_rays spatial shape {tuple(camera_rays.shape[2:4])} must match "
            f"rgb_rendered shape {tuple(rgb_rendered.shape[-2:])}"
        )
    if latent_pad_frames > 0:
        camera_rays = pad_temporal(camera_rays, latent_pad_frames, dim=1)

    w2cs = inputs["w2cs"]
    if latent_pad_frames > 0:
        w2cs = pad_temporal(w2cs, latent_pad_frames, dim=1)

    Ks = inputs["Ks"]
    if latent_pad_frames > 0:
        Ks = pad_temporal(Ks, latent_pad_frames, dim=1)

    kwargs = dict(
        inputs, rendered_rgb=rgb_rendered, rendered_opacity=opacity, camera_rays=camera_rays, w2cs=w2cs, Ks=Ks
    )
    sampling = SamplingConfig(
        num_inference_steps=args.num_inference_steps,
        text_guidance_scale=args.text_guidance_scale if args.inference_pipeline != "kv_cache" else None,
        max_neighbors_per_encode=args.max_neighbors_per_encode,
        progress_bar_leave=None,
    )
    seed = _item_seed(args.seed, item)
    with _inference_rng(seed, device):
        if args.inference_pipeline == "kv_cache":
            # Denoise and decode separately so non-saving ranks can drop inference caches and skip the decode.
            latents = generate(pipe, kwargs, sampling, output_latents=True, show_progress=rank == 0)
            if not save_outputs:
                pipe.clear_inference_caches()
                return
            out = pipe.decode_latents_to_video(latents)
            # Free the GPU latents before output processing and any diagnostics forwards.
            del latents
        else:
            out = generate(pipe, kwargs, sampling, show_progress=rank == 0)

    if not save_outputs:
        return

    out = out[:, :original_num_frames].cpu()
    rgb_gt_out = rgb_gt[:, :original_num_frames] if rgb_gt is not None else None
    rgb_rendered_out = rgb_rendered[:, :original_num_frames].cpu()

    valid_mask_list, output_indices_list, frame_indices = _eval_frame_metadata(item)
    video_frame_indices = _values_for_valid_frames(
        item.get("video_frame_indices", item.get("frame_indices")), valid_mask_list, "video_frame_indices"
    )
    assert (
        len(valid_mask_list) == original_num_frames
    ), f"valid_frames_mask has {len(valid_mask_list)} entries, expected {original_num_frames}"
    valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool)

    pred_all = out[0, valid_mask].float().clamp(0, 1)
    gt_all = rgb_gt_out[0, valid_mask].float().clamp(0, 1) if rgb_gt_out is not None else None
    rendered_all = rgb_rendered_out[0, valid_mask].float().clamp(0, 1)

    scene_dir.mkdir(parents=True, exist_ok=True)

    output_indices = torch.tensor(output_indices_list, dtype=torch.long)
    if not args.save_frame_outputs_only and args.inference_pipeline != "bidirectional":
        comparison_key = "default" if split is not None else "comparison"
        chunk_idx = item.get("chunk_idx")
        if chunk_idx is not None:
            # A scene can produce several chunk items, and different ranks may process
            # chunks of the same scene, so each chunk needs its own output name.
            comparison_key = f"{comparison_key}_chunk{int(chunk_idx):04d}"
        save_comparison_output(
            output=pred_all,
            rgb_gt=gt_all,
            rgb_rendered=rendered_all,
            rgb_neighbors=rgb_neighbors_cpu,
            save_dir=scene_dir,
            key=comparison_key,
            fps=args.output_fps,
            save_frames=True,
            output_indices=output_indices,
        )

    # Save individual frames
    frames_dir = scene_dir / "frames" / "batch_0000"
    video_frames_dir = scene_dir / "video_frames" / "batch_0000"
    for subdir in output_kinds(item):
        (frames_dir / subdir).mkdir(parents=True, exist_ok=True)
        if not args.save_frame_outputs_only:
            (video_frames_dir / subdir).mkdir(parents=True, exist_ok=True)

    target_h = item.get("target_h")
    target_w = item.get("target_w")
    assert (target_h is None) == (target_w is None), "target_h and target_w must be provided together"

    for frame_idx in range(pred_all.shape[0]):
        pred_frame = pred_all[frame_idx]
        gt_frame = gt_all[frame_idx] if has_gt else None
        rendered_frame = rendered_all[frame_idx]

        if target_h is not None:
            th = target_h.item() if torch.is_tensor(target_h) else target_h
            tw = target_w.item() if torch.is_tensor(target_w) else target_w
            if pred_frame.shape[-2:] != (th, tw):
                pred_frame = F.interpolate(
                    pred_frame.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
                ).squeeze(0)
                if gt_frame is not None:
                    gt_frame = F.interpolate(
                        gt_frame.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
                    ).squeeze(0)
                rendered_frame = F.interpolate(
                    rendered_frame.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
                ).squeeze(0)

        output_idx = int(output_indices[frame_idx])
        for kind, frame in (("pred", pred_frame), ("gt", gt_frame), ("rendered", rendered_frame)):
            if frame is None:
                continue
            paths = []
            if output_idx != -1:
                paths.append(frames_dir / kind / f"{output_idx:05d}.png")
            if not args.save_frame_outputs_only:
                paths.append(video_frames_dir / kind / f"{video_frame_indices[frame_idx]:05d}.png")
            if not paths:
                continue
            # Encode each frame to PNG once; copy the file for any other destination.
            save_image(frame, paths[0])
            for extra_path in paths[1:]:
                with atomic_output_path(extra_path) as temporary:
                    shutil.copyfile(paths[0], temporary)

    if args.render_diagnostics and not args.save_frame_outputs_only:
        _render_diagnostics(
            pipe,
            kwargs,
            args,
            scene_dir,
            original_num_frames,
            valid_mask,
            gt_all,
            rendered_all,
            rgb_neighbors_cpu,
            output_indices,
            sampling=sampling,
            seed=seed,
            device=device,
            chunk_idx=item.get("chunk_idx"),
        )


def _render_diagnostics(
    pipe,
    kwargs: dict,
    args,
    scene_dir: Path,
    original_num_frames: int,
    valid_mask: torch.Tensor,
    gt_all: torch.Tensor | None,
    rendered_all: torch.Tensor,
    rgb_neighbors_cpu: torch.Tensor,
    output_indices: torch.Tensor,
    *,
    sampling: SamplingConfig,
    seed: int,
    device: torch.device,
    chunk_idx: int | None = None,
) -> None:
    """Run extra forwards with parts of the conditioning zeroed out and save the results."""
    # Swap in the stored unconditional when the payload has one (zeroed token ids would be a
    # garbage prompt); zero the embedding otherwise. Guidance scale 1 keeps the pass unguided —
    # Wan would otherwise guide against its default negative prompt.
    no_text_prompt, no_text_prompt_mask = substitute_unconditional_prompt(
        kwargs["prompt"],
        kwargs["prompt_attention_mask"],
        kwargs["negative_prompt"],
        kwargs["negative_prompt_attention_mask"],
    )
    no_text_kwargs = {
        "prompt": no_text_prompt,
        "prompt_attention_mask": no_text_prompt_mask,
        "negative_prompt": None,
        "negative_prompt_attention_mask": None,
        "text_guidance_scale": 1.0,
    }
    for diag_name, diag_kwargs in [
        (
            "no_rendered_rgb",
            {
                "rendered_rgb": torch.zeros_like(kwargs["rendered_rgb"]),
                "rendered_opacity": (
                    torch.zeros_like(kwargs["rendered_opacity"]) if kwargs["rendered_opacity"] is not None else None
                ),
            },
        ),
        (
            "text_only",
            {
                "rendered_rgb": torch.zeros_like(kwargs["rendered_rgb"]),
                "rendered_opacity": (
                    torch.zeros_like(kwargs["rendered_opacity"]) if kwargs["rendered_opacity"] is not None else None
                ),
                "neighbors": None,
            },
        ),
        ("no_text", no_text_kwargs),
    ]:
        output_key = diagnostic_output_key(diag_name, chunk_idx)
        diag_output_path = scene_dir / f"{output_key}.mp4"
        if args.replace_if_exists or not video_file_complete(diag_output_path, int(valid_mask.sum())):
            diag_full_kwargs = dict(kwargs)
            diag_full_kwargs.update(diag_kwargs)
            diag_sampling = replace(
                sampling, text_guidance_scale=diag_full_kwargs.pop("text_guidance_scale", sampling.text_guidance_scale)
            )
            with _inference_rng(seed, device):
                diag_out = generate(pipe, diag_full_kwargs, diag_sampling)
            diag_out = diag_out[0, :original_num_frames].cpu()[valid_mask].float().clamp(0, 1)
            save_comparison_output(
                output=diag_out,
                rgb_gt=gt_all,
                rgb_rendered=rendered_all,
                rgb_neighbors=rgb_neighbors_cpu,
                save_dir=scene_dir,
                key=output_key,
                fps=args.output_fps,
                save_frames=False,
                output_indices=output_indices,
            )
            del diag_out


def finalize_video_outputs(output_dir: Path, fps: int, replace: bool = False) -> None:
    for batch_dir in sorted(output_dir.glob("**/video_frames/batch_*")):
        scene_dir = batch_dir.parent.parent
        videos_dir = scene_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        for kind in ("pred", "gt", "rendered"):
            paths = sorted(path for path in (batch_dir / kind).glob("*.png") if not path.name.startswith("."))
            output_path = videos_dir / f"{batch_dir.name}_{kind}.mp4"
            if paths and (replace or not video_file_complete(output_path, len(paths))):
                save_video(paths, output_path, fps=fps)


def dataset_scene_ids(dataset) -> list[str]:
    scene_ids = getattr(dataset, "scene_ids", None)
    if scene_ids is not None:
        return sorted(str(scene_id) for scene_id in scene_ids)

    inference_items = getattr(dataset, "inference_items", None)
    assert inference_items is not None, f"{type(dataset).__name__} does not expose scene ids for replacement cleanup"
    scene_ids = {str(pair.scene_id) for _, pair in inference_items}
    return sorted(scene_ids)


def clear_existing_scene_outputs(output_dir: Path, dataset) -> None:
    for scene_id in dataset_scene_ids(dataset):
        scene_dir = output_dir / scene_id
        if scene_dir.exists():
            assert scene_dir.is_dir() and not scene_dir.is_symlink(), f"Expected eval scene directory, got {scene_dir}"
            shutil.rmtree(scene_dir)


def iter_prefetched_items(dataset, item_indices: Sequence[int]):
    """Yield dataset items in index order, loading the next item in a background
    thread so the CPU-side data prep overlaps GPU inference on the current item."""
    item_indices = list(item_indices)
    if not item_indices:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(dataset.__getitem__, item_indices[0])
        for next_idx in item_indices[1:]:
            item = future.result()
            future = executor.submit(dataset.__getitem__, next_idx)
            yield item
        yield future.result()


def first_item_num_frames(dataset) -> int:
    """Frame count of item 0, read from metadata when available so the full item is not loaded."""
    inference_items = getattr(dataset, "inference_items", None)
    if inference_items:
        _, pair = inference_items[0]
        return len(pair.test_indices)
    return int(dataset[0]["rgb_rendered"].shape[0])


def process_items_with_context_parallel(
    pipe,
    dataset,
    args,
    output_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
):
    vae_temporal_scale = pipe.vae.config.scale_factor_temporal

    if len(dataset) == 0:
        return

    if args.context_parallel_size <= 1:
        item_indices = range(rank, len(dataset), world_size)
        for item in tqdm(
            iter_prefetched_items(dataset, item_indices),
            total=len(item_indices),
            disable=(rank != 0),
            desc=f"GPU {rank}",
        ):
            torch.cuda.empty_cache()
            gc.collect()
            process_item(pipe, item, args, output_dir, rank, device, vae_temporal_scale)
        return

    total_num_frames = first_item_num_frames(dataset)
    pad_frames = compute_frame_padding(total_num_frames, pipe.latent_frame_multiple(), vae_temporal_scale)
    pad_frames += pipe.cp_raw_frame_padding(
        total_num_frames + pad_frames,
        args.context_parallel_size,
        vae_temporal_scale,
    )
    latent_num_frames = latent_num_frames_from_rgb_num_frames(total_num_frames + pad_frames, vae_temporal_scale)
    pipe.validate_cp_size(args.context_parallel_size, latent_num_frames)

    my_mesh, num_groups, group_index, group_rank = create_context_parallel_meshes(
        rank, world_size, args.context_parallel_size, device.type
    )

    if rank == 0:
        print(f"Eval CP: cp_size={args.context_parallel_size}, groups={num_groups}, items={len(dataset)}")

    if my_mesh is not None:
        pipe.transformer.enable_context_parallel(my_mesh)

    try:
        item_indices = range(group_index, len(dataset), num_groups)
        for item in tqdm(
            iter_prefetched_items(dataset, item_indices),
            total=len(item_indices),
            disable=(rank != 0),
            desc=f"GPU {rank}",
        ):
            torch.cuda.empty_cache()
            gc.collect()
            process_item(
                pipe,
                item,
                args,
                output_dir,
                rank,
                device,
                vae_temporal_scale,
                save_outputs=(group_rank == 0),
            )
    finally:
        if my_mesh is not None:
            pipe.transformer.disable_context_parallel()


@torch.inference_mode()
def main(args: argparse.Namespace):
    rank, world_size, local_rank = init_distributed(args.distributed_timeout_minutes)
    device = torch.device(f"cuda:{local_rank}")

    disable_tf32_matmul(verbose=rank == 0)

    try:
        if rank == 0:
            print(f"Running on {world_size} GPUs, evalset: {args.evalset}")

        pipe = get_eval_pipe(args, device)
        pipe.transformer.eval()
        barrier_if_distributed()

        if rank == 0:
            print("Loaded transformer checkpoint")

        dataset = create_dataset(args, rank)
        if rank == 0:
            print(f"Dataset has {len(dataset)} items")

        output_dir = get_output_dir(args)
        if rank == 0:
            if args.replace_if_exists:
                clear_existing_scene_outputs(output_dir, dataset)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Writing outputs to {output_dir}")

        barrier_if_distributed()

        process_items_with_context_parallel(pipe, dataset, args, output_dir, rank, world_size, device)

        barrier_if_distributed()

        if rank == 0 and not args.save_frame_outputs_only:
            finalize_video_outputs(output_dir, fps=args.output_fps, replace=args.replace_if_exists)

        if rank == 0:
            print("Done!")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
