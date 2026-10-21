# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate ArtiFixer prompt HDF5 files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from data_processing.captioning.caption_plan import caption_output_path, caption_variants, scene_limited_num_frames
from data_processing.captioning.generate_captions import (
    generate_caption_hdf5,
    get_caption_model_and_processor,
    get_text_encoder_and_tokenizer,
)
from data_processing.captioning.prompt_hdf5 import default_prompt_model_id, get_cosmos3_tokenizer
from data_processing.scene_utils import (
    discover_scene_zips,
    read_scene_list,
    resolve_scene_path,
    scene_name,
    scene_relative_parent,
)
from model_training.constants import DL3DV_DATASET_FPS
from model_training.prompt_formats import PROMPT_FORMAT_CHOICES, WAN_T5_PROMPT_FORMAT, caption_filename_for_format


def load_models(args: argparse.Namespace) -> dict:
    download_kwargs = {}
    if getattr(args, "hf_cache_dir", None) is not None:
        download_kwargs["cache_dir"] = args.hf_cache_dir
    if getattr(args, "hf_hub_offline", False):
        download_kwargs["local_files_only"] = True
    caption_model, processor = get_caption_model_and_processor(
        args.captioning_model_id, args.captioning_attn_implementation, **download_kwargs
    )
    models = {"caption_model": caption_model, "processor": processor}
    if args.prompt_format == WAN_T5_PROMPT_FORMAT:
        text_encoder_model_id = args.text_encoder_model_id or default_prompt_model_id(args.prompt_format)
        text_encoder, tokenizer = get_text_encoder_and_tokenizer(text_encoder_model_id, **download_kwargs)
        models.update({"text_encoder": text_encoder, "tokenizer": tokenizer})
    else:
        # Token-id prompt formats tokenize with the pinned Cosmos3 tokenizer;
        # load it once here so the scene/variant loop reuses it instead of
        # re-resolving the snapshot on every generate_caption_hdf5 call.
        models["cosmos3_tokenizer"] = get_cosmos3_tokenizer(
            default_prompt_model_id(args.prompt_format), **download_kwargs
        )
    return models


def generate_for_scene(args: argparse.Namespace, scene_path: Path, models: dict) -> None:
    assert (
        scene_path.suffix == ".zip" or scene_path.is_dir()
    ), f"DL3DV captioning expects scene zip or directory inputs: {scene_path}"
    scene_output_dir = args.output_dir / scene_relative_parent(args.dl3dv_dir, scene_path) / scene_name(scene_path)

    effective_num_frames = scene_limited_num_frames(args.num_frames, scene_path)
    # Suppress reverse/revisit only for the short-scene collapse: each variant
    # costs a full captioning pass over the same short clip. Explicit full-clip
    # requests (--num_frames <= 0) keep their requested variants.
    short_scene = effective_num_frames is None and args.num_frames is not None and args.num_frames > 0
    for frame_stride, revisit, reverse in caption_variants(
        args.max_frame_stride,
        include_reverse=args.include_reverse and not short_scene,
        include_revisit=args.include_revisit and not short_scene,
    ):
        output_path = caption_output_path(
            scene_output_dir, effective_num_frames, frame_stride, revisit=revisit, reverse=reverse
        )
        if args.output_role == "evaluation":
            output_path = output_path.with_name(caption_filename_for_format(output_path.name, args.prompt_format))
        if not models and not output_path.exists():
            models.update(load_models(args))
        generate_caption_hdf5(
            scene_path,
            output_path,
            num_frames=effective_num_frames,
            frame_stride=frame_stride,
            revisit=revisit,
            reverse_frames=reverse,
            captioning_model_id=args.captioning_model_id,
            captioning_attn_implementation=args.captioning_attn_implementation,
            prompt_format=args.prompt_format,
            dataset_fps=args.dataset_fps,
            check_if_exists=True,
            **models,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dl3dv_dir", type=Path, required=True, help="Root containing DL3DV scene zips")
    parser.add_argument("--output_dir", type=Path, required=True, help="Root for generated prompt HDF5 files")
    parser.add_argument("--scene_id", action="append", default=[], help="Only process this scene id/path")
    parser.add_argument("--scene_list", type=Path, help="Only process scenes listed in this file")
    parser.add_argument("--num_frames", type=int, default=81, help="Prompt window length; <=0 means all frames")
    parser.add_argument("--max_frame_stride", type=int, default=1)
    parser.add_argument("--include_reverse", action="store_true", help="Also generate *_reverse prompt files")
    parser.add_argument("--include_revisit", action="store_true", help="Also generate *_revisit prompt files")
    parser.add_argument("--hf_cache_dir", type=Path)
    parser.add_argument("--hf_hub_offline", action="store_true", help="Use only already-cached Hugging Face assets")
    parser.add_argument("--captioning_model_id", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--captioning_attn_implementation")
    parser.add_argument("--prompt_format", choices=PROMPT_FORMAT_CHOICES, default=WAN_T5_PROMPT_FORMAT)
    parser.add_argument(
        "--output_role",
        choices=("training", "evaluation"),
        default="training",
        help="Evaluation uses one full-scene caption with the model-format filename.",
    )
    parser.add_argument("--dataset_fps", type=float, default=DL3DV_DATASET_FPS)
    parser.add_argument(
        "--text_encoder_model_id",
        help=(
            "Override the Wan T5 text encoder location, e.g. a local snapshot directory for offline "
            "nodes. Defaults to the prompt format's hub model id. Ignored for token-id prompt formats."
        ),
    )
    args = parser.parse_args()

    if args.max_frame_stride < 1:
        parser.error("--max_frame_stride must be >= 1")
    if not math.isfinite(args.dataset_fps) or args.dataset_fps <= 0:
        parser.error("--dataset_fps must be finite and positive")
    if args.output_role == "evaluation" and (
        args.num_frames > 0 or args.max_frame_stride != 1 or args.include_reverse or args.include_revisit
    ):
        parser.error("--output_role evaluation requires --num_frames 0, stride 1 and no augmentation variants")

    scenes = list(args.scene_id)
    if args.scene_list is not None:
        scenes.extend(read_scene_list(args.scene_list))
    if scenes:
        args.scene_paths = [resolve_scene_path(args.dl3dv_dir, scene) for scene in scenes]
    else:
        args.scene_paths = discover_scene_zips(args.dl3dv_dir)
        if not args.scene_paths:
            parser.error(f"No scene zips found under {args.dl3dv_dir}")
    return args


def main() -> None:
    args = parse_args()
    # Load the captioner and text encoder on the first missing output and reuse
    # them across all scenes and variants. Idempotent re-runs over already
    # complete outputs must touch no models (they may run on offline/CPU nodes).
    models: dict = {}
    for scene_path in args.scene_paths:
        generate_for_scene(args, scene_path, models)


if __name__ == "__main__":
    main()
