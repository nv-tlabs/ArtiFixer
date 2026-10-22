# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retokenize existing caption HDF5 files into Cosmos3 prompt-token HDF5 files.

This is the on-cluster corpus (re)generation tool: the stored raw scene
captions are re-tokenized into fresh prompt-token files (template or
tokenizer changes run through this same tool). The output filename follows
the output tree:

  * separate output_dir (the TRAINING corpus layout): outputs keep the input
    filename — dl3dv_base's prompt globbing reads the plain
    ``frames_*_stride_1*.h5`` stems from a cosmos3-only tokenized tree.
  * output_dir == input_dir (the EVAL layout): outputs land beside the Wan
    caption files under the format-derived name from
    prompt_formats.caption_filename_for_format (``frames_*.cosmos3.h5``),
    exactly the name the eval datasets resolve for the cosmos3 format.

The camera-sentence scrub runs on every caption BEFORE tokenization
(idempotent — already-clean captions pass through unchanged; see
camera_scrub.py). The output ``caption`` attr stores the SCRUBBED text, so
the regenerated corpus stays self-consistent (attr matches tokens).
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import h5py
from PIL import Image

from data_processing.captioning.camera_scrub import ScrubStats, apply_camera_scrub, walk_corpus_tree
from data_processing.captioning.caption_plan import caption_frame_stride
from data_processing.captioning.prompt_hdf5 import (
    cosmos3_training_frame_size,
    get_cosmos3_tokenizer,
    write_cosmos3_prompt_dataset,
)
from data_processing.scene_utils import downsampled_image_path, load_scene_transforms_from_zip, scene_zip_member
from model_training.constants import DL3DV_DATASET_FPS
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, caption_filename_for_format

# The suffix the cosmos3 naming rule appends: "<base>.h5" -> "<base>.cosmos3.h5".
_COSMOS3_NAME_SUFFIX = caption_filename_for_format(".h5", COSMOS3_PROMPT_FORMAT)


def _source_zip_path(prompt_path: Path, input_dir: Path, dl3dv_dir: Path) -> Path:
    rel_path = prompt_path.relative_to(input_dir)
    assert len(rel_path.parts) >= 2, f"Prompt path must be <scene>/<file>.h5 or <subdir>/<scene>/<file>.h5: {rel_path}"
    scene_dir = rel_path.parent
    return dl3dv_dir / scene_dir.parent / f"{scene_dir.name}.zip"


@lru_cache(maxsize=None)
def _scene_frame_size(source_zip_path: Path) -> tuple[int, int]:
    # Cached per scene: a scene directory holds several stride/reverse window
    # files, and re-opening the zip + parsing transforms + decoding an image
    # for each one dominated retokenization time. DL3DV frames within a scene
    # share one resolution, so frame 0 stands for all.
    assert source_zip_path.is_file(), f"Source scene zip missing: {source_zip_path}"
    with ZipFile(source_zip_path, "r") as zf:
        transforms, scene_root = load_scene_transforms_from_zip(zf, source_zip_path.stem)
        # 4x downsampled images, matching the training data's resolution tier.
        rel_image_path = downsampled_image_path(transforms["frames"][0]["file_path"], 4)
        with zf.open(scene_zip_member(scene_root, rel_image_path), "r") as f:
            with Image.open(f) as image:
                width, height = image.size
    return height, width


def _source_frame_size(prompt_path: Path, input_dir: Path, dl3dv_dir: Path) -> tuple[int, int]:
    return _scene_frame_size(_source_zip_path(prompt_path, input_dir, dl3dv_dir))


def retokenize_file(
    input_path: Path,
    input_dir: Path,
    output_path: Path,
    *,
    tokenizer,
    dl3dv_dir: Path,
    replace: bool,
) -> ScrubStats | None:
    """Retokenize one prompt file; returns ``None`` when the output already
    exists and ``replace`` is off (skipped, not scanned — a skip must not read
    as a file with nothing to scrub)."""
    if output_path.exists() and not replace:
        return None

    stats = ScrubStats()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with h5py.File(input_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        dataset_names = list(src.keys())
        assert dataset_names, f"No prompt datasets found in {input_path}"
        source_height, source_width = _source_frame_size(input_path, input_dir, dl3dv_dir)
        height, width = cosmos3_training_frame_size(source_height, source_width)
        for dataset_name in dataset_names:
            src_dataset = src[dataset_name]
            # Recorded FPS already includes temporal subsampling. Legacy Wan
            # captions have no FPS attribute and use the filename's stride.
            fps = (
                float(src_dataset.attrs["fps"])
                if "fps" in src_dataset.attrs
                else DL3DV_DATASET_FPS / caption_frame_stride(input_path)
            )
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"{input_path}:{dataset_name} has invalid FPS {fps}")
            caption = src_dataset.attrs["caption"]
            # The scrubbed text is what gets tokenized AND what the output
            # `caption` attr stores below (write_cosmos3_prompt_dataset
            # stores the caption it tokenizes), so the regenerated corpus
            # is self-consistent. The guardrail inside apply_camera_scrub
            # keeps the ORIGINAL caption when a scrub empties it or
            # removes more than half of it (reported, never fatal).
            caption, dropped, suspicious = apply_camera_scrub(caption)
            stats.sentences_dropped += dropped
            if dropped:
                stats.captions_changed += 1
            if suspicious:
                stats.skipped_suspicious += 1
            image_indices = src_dataset.attrs["image_indices"]
            write_cosmos3_prompt_dataset(
                dst,
                dataset_name,
                caption,
                image_indices,
                tokenizer,
                num_frames=len(image_indices),
                height=height,
                width=width,
                fps=fps,
            )
    tmp_path.rename(output_path)
    return stats


def retokenize_tree(args: argparse.Namespace) -> None:
    tokenizer = get_cosmos3_tokenizer(COSMOS3_NANO_MODEL_ID)
    # Output naming per tree layout: see the module docstring.
    co_located = args.output_dir.resolve() == args.input_dir.resolve()
    input_files = sorted(path for path in args.input_dir.rglob("*.h5") if not path.name.endswith(_COSMOS3_NAME_SUFFIX))
    assert input_files, f"No HDF5 prompt files found under {args.input_dir}"

    def process_file(input_path):
        rel_path = input_path.relative_to(args.input_dir)
        if co_located:
            rel_path = rel_path.with_name(caption_filename_for_format(rel_path.name, COSMOS3_PROMPT_FORMAT))
        return retokenize_file(
            input_path,
            args.input_dir,
            args.output_dir / rel_path,
            tokenizer=tokenizer,
            dl3dv_dir=args.dl3dv_dir,
            replace=args.replace,
        )

    walk_corpus_tree(
        input_files,
        process_file,
        shard_label="retokenize+scrub",
        tool_label="retokenize",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--dl3dv_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    retokenize_tree(parse_args())
