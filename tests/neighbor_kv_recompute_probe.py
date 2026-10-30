# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU known answer for --recompute_neighbor_kv: one Wan KV-cache rollout, cached vs recomputed.

Run as a module on one GPU. The eval pipeline is built and loaded exactly as
run_inference builds it (get_eval_pipe: VAE, transformer, .pt or DCP
checkpoint, bf16, eval mode, TF32 off). One deterministic synthetic clip
(smooth RGB, an opacity field, posed reference views, the repository's
default T5 prompt embedding) then goes through process_item's kv_cache
denoise and decode twice under the same per-item seed: once with the
reference K/V cached (the default) and once recomputed. Only the pipe's flag
changes between the two runs; nothing is reloaded. A one-chunk warm-up runs
first so kernel selection is not timed.

The report records whether the latents and the decoded frames are
bit-identical, the peak torch.cuda.max_memory_allocated of each denoise and
decode, the reference K/V bytes each rollout stored, and wall clock. The exit
status is 0 only when both are bit-identical; 2 means a run ran out of memory.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import model_training.pipeline.wan_kv_cache_pipeline as wan_kv_module
from model_eval import inference_args
from model_eval.generation import SamplingConfig, generate, prepare_generation_inputs
from model_eval.inference import _inference_rng, get_eval_pipe, save_image
from model_training.data.utils import compute_camera_conditioning
from model_training.net.rollout_kv_cache import RolloutKvCache
from model_training.utils.precision import disable_tf32_matmul

_STORED_NEIGHBOR_KV_BYTES = [0]


class _CountingRolloutKvCache(RolloutKvCache):
    """Counts the reference K/V bytes a rollout stores (held until it ends)."""

    def store_neighbor_kv(self, layer_index, k_neighbor, v_neighbor):
        super().store_neighbor_kv(layer_index, k_neighbor, v_neighbor)
        _STORED_NEIGHBOR_KV_BYTES[0] += sum(t.numel() * t.element_size() for t in (k_neighbor, v_neighbor))


def _synthetic_item(args, prompt_path: Path) -> dict:
    """A deterministic clip on the CPU in the dataset's item layout."""
    generator = torch.Generator().manual_seed(args.seed)
    frames, views, height, width = args.num_frames, args.num_views, args.height, args.width
    y = torch.linspace(0, 1, height).view(height, 1)
    x = torch.linspace(0, 1, width).view(1, width)

    def image(phase: float) -> torch.Tensor:
        channels = [0.5 + 0.4 * torch.sin(2 * torch.pi * (x * (1 + c) + y * (2 - c) + phase)) for c in range(3)]
        noise = 0.03 * torch.randn(3, height, width, generator=generator)
        return (torch.stack(channels) + noise).clamp(0, 1)

    rgb = torch.stack([image(t / frames) for t in range(frames)])
    neighbors = torch.stack([image(0.37 * (v + 1)) for v in range(views)])
    radius = torch.stack([((x - 0.3 - 0.4 * t / frames) ** 2 + (y - 0.5) ** 2).sqrt() for t in range(frames)])
    opacity = torch.sigmoid((radius - 0.25) * 40)  # a moving hole of low opacity

    c2ws = torch.eye(4).repeat(frames, 1, 1)
    c2ws[:, 0, 3] = torch.linspace(-0.5, 0.5, frames)
    neighbor_c2ws = torch.eye(4).repeat(views, 1, 1)
    neighbor_c2ws[:, 0, 3] = torch.linspace(-1.0, 1.0, views)
    neighbor_c2ws[:, 2, 3] = -0.3
    focal = 0.6
    intrinsics = torch.tensor([focal, focal * width / height, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    cameras = compute_camera_conditioning(
        intrinsics={},
        c2ws_world=c2ws,
        neighbor_c2ws_world=neighbor_c2ws,
        scale=1.0,
        image_height=height,
        image_width=width,
        target_intrinsics=intrinsics.repeat(frames, 1),
        neighbor_intrinsics=intrinsics.repeat(views, 1),
    )
    prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)
    return dict(
        cameras,
        rgb_rendered=rgb,
        opacity=opacity,
        rgb_neighbors=neighbors,
        encoded_prompt=prompt.reshape(prompt.shape[-2:]),
    )


def _first_chunk(inputs: dict, latent_frames: int) -> dict:
    """The inputs cut to their first ``latent_frames`` latent frames."""
    chunk = dict(inputs)
    for key in ("rendered_rgb", "rendered_opacity"):
        chunk[key] = inputs[key][:, : (latent_frames - 1) * 4 + 1]
    for key in ("camera_rays", "w2cs", "Ks"):
        chunk[key] = inputs[key][:, :latent_frames]
    return chunk


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _cuda_stat(device: torch.device, name: str) -> float:
    """A torch.cuda memory statistic in GiB (0 on CPU, which the probe never measures)."""
    return getattr(torch.cuda, name)(device) / 2**30 if device.type == "cuda" else 0.0


def _rollout(pipe, inputs: dict, sampling: SamplingConfig, seed: int, device, recompute: bool) -> dict:
    """process_item's kv_cache denoise and decode with the flag set as given."""
    cuda = device.type == "cuda"
    pipe.recompute_neighbor_kv = recompute
    _STORED_NEIGHBOR_KV_BYTES[0] = 0
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    resident = _cuda_stat(device, "memory_allocated")
    start = time.perf_counter()
    with _inference_rng(seed, device):
        latents = generate(pipe, inputs, sampling, output_latents=True, show_progress=True)
        if cuda:
            torch.cuda.synchronize(device)
        denoised = time.perf_counter()
        denoise_peak = _cuda_stat(device, "max_memory_allocated")
        stored_neighbor_kv = _STORED_NEIGHBOR_KV_BYTES[0]
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        frames = pipe.decode_latents_to_video(latents)
        if cuda:
            torch.cuda.synchronize(device)
    decoded = time.perf_counter()
    return dict(
        latents=latents.cpu(),
        frames=frames.cpu(),
        stats=dict(
            recompute_neighbor_kv=recompute,
            resident_before_gib=resident,
            denoise_peak_allocated_gib=denoise_peak,
            decode_peak_allocated_gib=_cuda_stat(device, "max_memory_allocated"),
            peak_reserved_gib=_cuda_stat(device, "max_memory_reserved"),
            stored_neighbor_kv_gib=stored_neighbor_kv / 2**30,
            denoise_seconds=denoised - start,
            decode_seconds=decoded - denoised,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Wan .pt export or DCP checkpoint directory.")
    parser.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--attention_backend", default=None)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path.")
    parser.add_argument("--save_frames_dir", type=Path, default=None, help="Also write both runs' frames as PNGs.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--num_frames", type=int, default=109, help="Raw frames; (n - 1) / 4 + 1 latent frames.")
    parser.add_argument("--num_views", type=int, default=12, help="Reference views.")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--frames_per_block", type=int, default=7)
    parser.add_argument("--local_attn_size", type=int, default=21)
    parser.add_argument("--sink_size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", choices=["cached_first", "recomputed_first"], default="cached_first")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new evidence path")
    latent_frames = (args.num_frames - 1) // 4 + 1
    if (args.num_frames - 1) % 4 or latent_frames % args.frames_per_block:
        parser.error("--num_frames must give (n - 1) / 4 + 1 latent frames divisible by --frames_per_block")

    device = torch.device("cuda", torch.cuda.current_device())
    disable_tf32_matmul(verbose=True)
    eval_argv = [
        *("--checkpoint", str(args.checkpoint), "--model_id", args.model_id),
        *("--save_dir", str(args.output.parent), "--render_name", "neighbor_kv_recompute_probe"),
        *("--inference_pipeline", "kv_cache", "--frames_per_block", str(args.frames_per_block)),
        *("--local_attn_size", str(args.local_attn_size), "--sink_size", str(args.sink_size)),
    ]
    if args.model_revision is not None:
        eval_argv += ["--model_revision", args.model_revision]
    if args.attention_backend is not None:
        eval_argv += ["--attention_backend", args.attention_backend]
    eval_args = inference_args.build_parser().parse_args(eval_argv)
    pipe = get_eval_pipe(eval_args, device)
    pipe.transformer.eval()
    wan_kv_module.RolloutKvCache = _CountingRolloutKvCache

    item = _synthetic_item(args, eval_args.default_negative_prompt_path)
    inputs = prepare_generation_inputs(
        item,
        device=device,
        dtype=torch.bfloat16,
        batched=False,
        stage_conditioning_on_cpu=pipe.stages_conditioning_per_chunk,
    )
    inputs.pop("neighbor_camera_rays", None)
    sampling = SamplingConfig(
        num_inference_steps=args.num_inference_steps, max_neighbors_per_encode=None, progress_bar_leave=None
    )

    # Warm-up on the first chunk only, so kernel selection is not timed.
    _rollout(pipe, _first_chunk(inputs, args.frames_per_block), sampling, args.seed, device, recompute=False)

    runs, oom = {}, None
    order = [False, True] if args.order == "cached_first" else [True, False]
    for recompute in order:
        name = "recomputed" if recompute else "cached"
        try:
            runs[name] = _rollout(pipe, inputs, sampling, args.seed, device, recompute)
        except torch.cuda.OutOfMemoryError as error:
            oom = f"{name}: {error}".splitlines()[0]
            torch.cuda.empty_cache()
    report = dict(
        device=torch.cuda.get_device_name(device),
        torch=torch.__version__,
        checkpoint=str(args.checkpoint),
        model_id=args.model_id,
        clip=dict(
            width=args.width,
            height=args.height,
            raw_frames=args.num_frames,
            latent_frames=latent_frames,
            reference_views=args.num_views,
            steps=args.num_inference_steps,
            window=[args.frames_per_block, args.local_attn_size, args.sink_size],
        ),
        order=args.order,
        out_of_memory=oom,
        runs={name: run["stats"] for name, run in runs.items()},
    )
    if len(runs) == 2:
        cached, recomputed = runs["cached"], runs["recomputed"]
        report.update(
            latents_bit_identical=torch.equal(cached["latents"], recomputed["latents"]),
            frames_bit_identical=torch.equal(cached["frames"], recomputed["frames"]),
            frames_max_abs_diff=(cached["frames"].float() - recomputed["frames"].float()).abs().max().item(),
            latents_sha256={name: _sha256(run["latents"]) for name, run in runs.items()},
            denoise_peak_saving_gib=cached["stats"]["denoise_peak_allocated_gib"]
            - recomputed["stats"]["denoise_peak_allocated_gib"],
            denoise_time_ratio=recomputed["stats"]["denoise_seconds"] / cached["stats"]["denoise_seconds"],
        )
    if args.save_frames_dir is not None:
        for name, run in runs.items():
            for index, frame in enumerate(run["frames"][0].float().clamp(0, 1)):
                path = args.save_frames_dir / name / f"{index:05d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_image(frame, path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if oom is not None:
        raise SystemExit(2)
    if not (report["latents_bit_identical"] and report["frames_bit_identical"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
