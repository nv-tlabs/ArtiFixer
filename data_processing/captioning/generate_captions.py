# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import json
import math
import os
import warnings
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen3VLProcessor,
    UMT5EncoderModel,
)

from data_processing.captioning.camera_scrub import ScrubStats, apply_camera_scrub
from data_processing.captioning.caption_plan import caption_output_path
from data_processing.captioning.prompt_hdf5 import (
    cosmos3_training_frame_size,
    default_prompt_model_id,
    get_cosmos3_tokenizer,
    write_cosmos3_prompt_dataset,
)
from data_processing.scene_utils import (
    downsampled_image_path,
    load_scene_transforms_from_dir,
    load_scene_transforms_from_zip,
    scene_zip_member,
)
from model_training.constants import DL3DV_DATASET_FPS
from model_training.prompt_formats import PROMPT_FORMAT_CHOICES, WAN_T5_PROMPT_FORMAT, validate_prompt_file_format

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"

IMAGE_PROMPT = """
    You are an image captioning specialist whose goal is to generate high-quality English prompts by referring to the details of the user's input images. Your task is to carefully analyze the content, context, and visual elements within the image, and produce a complete, expressive, and natural-sounding caption that accurately conveys the scene. The caption should preserve the original intent and meaning of the image while enhancing its clarity and descriptive richness. Strictly adhere to the formatting of the examples provided.

    Task Requirements:
    1. You need to describe the main subject of the image in detail, including their appearance, actions, expressions, and the surrounding environment.
    2. You should never describe any details about the camera movement or camera angles.
    3. Your output should convey natural visual attributes, incorporating natural details related to the described subject category, using simple and direct descriptive language as much as possible.
    4. You should reference the detailed information in the image, such as character actions, clothing, backgrounds, and emphasize the details in the photo.
    5. Control the output prompt to around 80-100 words.
    6. No matter what language the user inputs, you must always output in English.

    Example of the English prompt:
    1. A Japanese fresh film-style photo of a young East Asian girl with double braids sitting by the boat. The girl wears a white square collar puff sleeve dress, decorated with pleats and buttons. She has fair skin, delicate features, and slightly melancholic eyes, staring directly at the camera. Her hair falls naturally, with bangs covering part of her forehead. She rests her hands on the boat, appearing natural and relaxed. The background features a blurred outdoor scene, with hints of blue sky, mountains, and some dry plants. The photo has a vintage film texture. A medium shot of a seated portrait.
    2. An anime illustration in vibrant thick painting style of a white girl with cat ears holding a folder, showing a slightly dissatisfied expression. She has long dark purple hair and red eyes, wearing a dark gray skirt and a light gray top with a white waist tie and a name tag in bold Chinese characters that says "紫阳" (Ziyang). The background has a light yellow indoor tone, with faint outlines of some furniture visible. A pink halo hovers above her head, in a smooth Japanese cel-shading style. A close-up shot from a slightly elevated perspective.
    3. CG game concept digital art featuring a huge crocodile with its mouth wide open, with trees and thorns growing on its back. The crocodile's skin is rough and grayish-white, resembling stone or wood texture. Its back is lush with trees, shrubs, and thorny protrusions. With its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The background features a dusk sky with some distant trees, giving the overall scene a dark and cold atmosphere. A close-up from a low angle.
    4. In the style of an American drama promotional poster, Walter White sits in a metal folding chair wearing a yellow protective suit, with the words "Breaking Bad" written in sans-serif English above him, surrounded by piles of dollar bills and blue plastic storage boxes. He wears glasses, staring forward, dressed in a yellow jumpsuit, with his hands resting on his knees, exuding a calm and confident demeanor. The background shows an abandoned, dim factory with light filtering through the windows. There's a noticeable grainy texture. A medium shot with a straight-on close-up of the character.

    Directly output the English text.
"""

VIDEO_PROMPT = """
    You are a video captioning specialist whose goal is to generate high-quality English prompts by referring to the details of the user's input videos. Your task is to carefully analyze the content, context, and actions within the video, and produce a complete, expressive, and natural-sounding caption that accurately conveys the scene. The caption should preserve the original intent and meaning of the video while enhancing its clarity and descriptive richness. Strictly adhere to the formatting of the examples provided.

    Task Requirements:
    1. You need to describe the main subject of the video in detail, including their appearance, actions, expressions, and the surrounding environment.
    2. You should never describe any details about the camera movement or camera angles.
    3. Your output should convey natural movement attributes, incorporating natural actions related to the described subject category, using simple and direct verbs as much as possible.
    4. You should reference the detailed information in the video, such as character actions, clothing, backgrounds, and emphasize the details in the photo.
    5. Control the output prompt to around 80-100 words.
    6. No matter what language the user inputs, you must always output in English.

    Example of the English prompt:
    1. A Japanese fresh film-style photo of a young East Asian girl with double braids sitting by the boat. The girl wears a white square collar puff sleeve dress, decorated with pleats and buttons. She has fair skin, delicate features, and slightly melancholic eyes, staring directly at the camera. Her hair falls naturally, with bangs covering part of her forehead. She rests her hands on the boat, appearing natural and relaxed. The background features a blurred outdoor scene, with hints of blue sky, mountains, and some dry plants. The photo has a vintage film texture. A medium shot of a seated portrait.
    2. An anime illustration in vibrant thick painting style of a white girl with cat ears holding a folder, showing a slightly dissatisfied expression. She has long dark purple hair and red eyes, wearing a dark gray skirt and a light gray top with a white waist tie and a name tag in bold Chinese characters that says "紫阳" (Ziyang). The background has a light yellow indoor tone, with faint outlines of some furniture visible. A pink halo hovers above her head, in a smooth Japanese cel-shading style. A close-up shot from a slightly elevated perspective.
    3. CG game concept digital art featuring a huge crocodile with its mouth wide open, with trees and thorns growing on its back. The crocodile's skin is rough and grayish-white, resembling stone or wood texture. Its back is lush with trees, shrubs, and thorny protrusions. With its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The background features a dusk sky with some distant trees, giving the overall scene a dark and cold atmosphere. A close-up from a low angle.
    4. In the style of an American drama promotional poster, Walter White sits in a metal folding chair wearing a yellow protective suit, with the words "Breaking Bad" written in sans-serif English above him, surrounded by piles of dollar bills and blue plastic storage boxes. He wears glasses, staring forward, dressed in a yellow jumpsuit, with his hands resting on his knees, exuding a calm and confident demeanor. The background shows an abandoned, dim factory with light filtering through the windows. There's a noticeable grainy texture. A medium shot with a straight-on close-up of the character.

    Directly output the English text.
"""


def generate_caption(
    image_paths: np.ndarray,
    fps: float,
    model: PreTrainedModel,
    processor: Qwen3VLProcessor,
) -> str:
    if len(image_paths) == 1:
        content = {"type": "image", "image": image_paths[0]}
    else:
        content = {"type": "video", "video": image_paths}

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": IMAGE_PROMPT if len(image_paths) == 1 else VIDEO_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [content],
        },
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        video_metadata={"fps": fps, "total_num_frames": len(image_paths)},
    )
    inputs = inputs.to(model.device).to(model.dtype)
    generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Trailing whitespace/newlines from the decode would survive into the
    # stored caption and (on the Cosmos3 path) malform the prompt's sentence
    # separators; the Wan path's prompt_clean strips anyway.
    return output_text[0].strip()


def generate_text_embedding(
    caption: str, text_encoder: UMT5EncoderModel, tokenizer: AutoTokenizer, max_sequence_length: int
) -> np.ndarray:
    prompt = prompt_clean(caption)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), mask.to(text_encoder.device)).last_hidden_state

    # Numpy doesn't have bfloat16 data type, so convert to uint16 for storage
    return prompt_embeds.squeeze(0)[:seq_lens].view(torch.uint16).cpu().numpy()


def get_caption_model_and_processor(
    model_id: str, attn_implementation: str | None = None, **download_kwargs
) -> tuple[PreTrainedModel, Qwen3VLProcessor]:
    caption_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        attn_implementation=attn_implementation,
        **download_kwargs,
    )
    processor = Qwen3VLProcessor.from_pretrained(model_id, **download_kwargs)
    return caption_model, processor


def get_text_encoder_and_tokenizer(model_id: str, **download_kwargs) -> tuple[UMT5EncoderModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", **download_kwargs)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", device_map="auto", torch_dtype=torch.bfloat16, **download_kwargs
    )
    return text_encoder, tokenizer


def _write_wan_prompt_dataset(
    hf: h5py.File,
    dataset_name: str,
    caption: str,
    selected_image_indices: np.ndarray,
    text_encoder: UMT5EncoderModel,
    tokenizer: PreTrainedTokenizerBase,
    text_encoder_max_sequence_length: int,
    *,
    fps: float,
) -> None:
    text_embedding = generate_text_embedding(caption, text_encoder, tokenizer, text_encoder_max_sequence_length)
    dataset = hf.create_dataset(dataset_name, data=text_embedding)
    dataset.attrs["caption"] = caption
    dataset.attrs["image_indices"] = selected_image_indices
    dataset.attrs["fps"] = float(fps)


@torch.inference_mode()
def generate_caption_hdf5(
    input_path: Path,
    output_path: Path,
    *,
    num_frames: int | None = None,
    trajectory_path: Path | None = None,
    frame_stride: int = 1,
    dataset_fps: float = DL3DV_DATASET_FPS,
    dataset_downsample_factor: int = 4,
    captioning_model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    captioning_attn_implementation: str | None = None,
    text_encoder_max_sequence_length: int = 512,
    prompt_format: str = WAN_T5_PROMPT_FORMAT,
    prompt_model_id: str | None = None,
    check_if_exists: bool = False,
    throw_error_if_exists: bool = False,
    reverse_frames: bool = False,
    revisit: bool = False,
    caption_model: PreTrainedModel | None = None,
    processor: Qwen3VLProcessor | None = None,
    text_encoder: UMT5EncoderModel | None = None,
    tokenizer: AutoTokenizer | None = None,
    cosmos3_tokenizer: PreTrainedTokenizerBase | None = None,
) -> Path | None:
    """Generate prompt-caption embeddings and return the actual HDF5 path.

    When ``num_frames`` requests more frames than the input contains, caption the
    full available video and write ``frames_all_stride_1.h5`` instead of a
    misleading ``frames_<N>`` file.  Training expands those short-video prompts
    with pingpong frame indices.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    assert (
        prompt_format in PROMPT_FORMAT_CHOICES
    ), f"prompt_format must be one of {PROMPT_FORMAT_CHOICES}, got {prompt_format!r}"
    if prompt_model_id is None:
        prompt_model_id = default_prompt_model_id(prompt_format)
    assert frame_stride >= 1, "frame_stride must be >= 1"
    if not math.isfinite(dataset_fps) or dataset_fps <= 0:
        raise ValueError("dataset_fps must be finite and positive")
    if num_frames is not None and num_frames <= 0:
        num_frames = None

    if check_if_exists and output_path.exists():
        validate_prompt_file_format(output_path, prompt_format, expected_fps=dataset_fps / frame_stride)
        message = f"Output path {output_path} already exists"
        if throw_error_if_exists:
            raise RuntimeError(message)
        print(message)
        return output_path

    image_names: list[str] = []
    images: list[np.ndarray] = []
    # Caption an image directory in the order specified by an explicit trajectory.
    if trajectory_path is not None:
        with trajectory_path.open() as f:
            trajectory_indices = json.load(f)["permutation"]

        image_paths = sorted(input_path.iterdir())
        for index in tqdm(trajectory_indices):
            image_names.append(image_paths[index].name)
            with Image.open(image_paths[index]) as image:
                images.append(np.array(image.convert("RGB")))
    elif input_path.is_dir():
        json_data = load_scene_transforms_from_dir(input_path)
        for frame in json_data["frames"]:
            rel_image_path = downsampled_image_path(frame["file_path"], dataset_downsample_factor)
            image_name = Path(frame["file_path"]).name
            image_names.append(image_name)
            with Image.open(input_path / rel_image_path) as image:
                images.append(np.array(image.convert("RGB")))
    elif input_path.suffix == ".zip":
        with ZipFile(input_path, "r") as zf:
            json_data, scene_root = load_scene_transforms_from_zip(zf, input_path.stem)
            for frame in json_data["frames"]:
                rel_image_path = downsampled_image_path(frame["file_path"], dataset_downsample_factor)
                image_name = Path(frame["file_path"]).name
                image_names.append(image_name)
                with zf.open(scene_zip_member(scene_root, rel_image_path), "r") as f:
                    with Image.open(BytesIO(f.read())) as image:
                        images.append(np.array(image.convert("RGB")))
    else:
        raise ValueError(f"Unsupported input path: {input_path}")

    image_indices = np.array(list(range(len(images))))
    if reverse_frames:
        images = images[::-1]
        image_names = image_names[::-1]
        image_indices = image_indices[::-1]

    if not images:
        raise ValueError(f"No images found for captioning input {input_path}")
    effective_num_frames = num_frames
    if num_frames is not None:
        required_frames = 1 + (num_frames - 1) * frame_stride
        if len(images) < required_frames:
            canonical_windowed = caption_output_path(
                output_path.parent, num_frames, frame_stride, revisit=revisit, reverse=reverse_frames
            )
            if output_path != canonical_windowed:
                # A caller-chosen filename cannot be renamed to the frames_all
                # convention without colliding with other inputs captioned into
                # the same directory.
                raise ValueError(
                    f"Not enough frames in {input_path}: need at least {required_frames} "
                    f"for num_frames={num_frames} and frame_stride={frame_stride}, got {len(images)}"
                )
            warnings.warn(
                f"Captioning all available frames for {input_path}: requested num_frames={num_frames} "
                f"with frame_stride={frame_stride} requires {required_frames} frames, got {len(images)}",
                stacklevel=2,
            )
            effective_num_frames = None
            output_path = caption_output_path(
                output_path.parent,
                None,
                frame_stride,
                revisit=revisit,
                reverse=reverse_frames,
            )
            if check_if_exists and output_path.exists():
                validate_prompt_file_format(output_path, prompt_format, expected_fps=dataset_fps / frame_stride)
                message = f"Output path {output_path} already exists"
                if throw_error_if_exists:
                    raise RuntimeError(message)
                print(message)
                return output_path

    images = np.stack(images)

    if prompt_format != WAN_T5_PROMPT_FORMAT and (len(images) < 2 or num_frames == 1):
        raise ValueError(
            "Cosmos3 prompts describe a video clip (duration/fps sentences), so "
            f"single-frame inputs cannot be captioned (got {len(images)} usable frame(s) "
            f"from {input_path}, num_frames={num_frames}). Use the Wan prompt format for "
            "single-image captioning."
        )

    # Callers that process many scenes/variants pass preloaded models to avoid
    # reloading the captioner and text encoder on every invocation.
    owns_caption_model = caption_model is None
    if owns_caption_model:
        caption_model, processor = get_caption_model_and_processor(captioning_model_id, captioning_attn_implementation)

    caption_records: list[tuple[str, str, np.ndarray]] = []
    scrub_stats = ScrubStats()

    def add_caption_record(dataset_name: str, indices_to_use: Iterable[int]) -> None:
        indices = np.asarray(list(indices_to_use), dtype=np.int64)
        if len(indices) == 0:
            return
        caption = generate_caption(
            images[indices],
            # True division, matching the fps write_cosmos3_prompt_dataset
            # stamps into the h5 attr and prompt metadata below — floor
            # division told the captioner a different clip speed whenever
            # frame_stride does not divide dataset_fps.
            dataset_fps / frame_stride,
            caption_model,
            processor,
        )
        # Enforce the captioning instruction (requirement 2: never describe
        # camera movement) at the source: the model still leaks a dedicated
        # "The camera ..." sentence into a few percent of captions. The
        # guardrail inside apply_camera_scrub keeps the original when
        # scrubbing would remove more than half of it.
        caption, dropped, suspicious = apply_camera_scrub(caption)
        scrub_stats.sentences_dropped += dropped
        if dropped:
            scrub_stats.captions_changed += 1
        if suspicious:
            # The guardrail KEPT this caption (ambiguous or would lose more
            # than half its text); it still contains camera narration and
            # needs re-captioning upstream.
            scrub_stats.skipped_suspicious += 1
        caption_records.append((dataset_name, caption, image_indices[indices]))

    if effective_num_frames is None:
        indices_to_use = list(range(0, len(images), frame_stride))
        if revisit:
            indices_to_use = indices_to_use + indices_to_use[:-1][::-1]
        add_caption_record(image_names[0], indices_to_use)
    else:
        window_span = 1 + (effective_num_frames - 1) * frame_stride
        for i in tqdm(range(len(images) - window_span + 1)):
            indices_to_use = list(range(i, i + (effective_num_frames * frame_stride), frame_stride))
            if revisit:
                pivot_index = math.ceil(len(indices_to_use) / 2)
                indices_to_use = (
                    indices_to_use[:pivot_index]
                    + indices_to_use[: (pivot_index - (1 if len(indices_to_use) % 2 != 0 else 0))][::-1]
                )

            if len(indices_to_use) != effective_num_frames:
                warnings.warn(
                    f"Skipping caption window for {input_path} at frame {i}: "
                    f"expected {effective_num_frames} frames, got {len(indices_to_use)}",
                    stacklevel=2,
                )
                continue
            add_caption_record(image_names[i], indices_to_use)

    if scrub_stats.captions_changed or scrub_stats.skipped_suspicious:
        print(
            f"camera-sentence scrub: dropped {scrub_stats.sentences_dropped} sentences from "
            f"{scrub_stats.captions_changed} captions, kept {scrub_stats.skipped_suspicious} suspicious "
            f"captions (still contain camera narration; re-caption upstream) for {output_path}",
            flush=True,
        )

    if owns_caption_model:
        del caption_model, processor
        torch.cuda.empty_cache()
        gc.collect()

    if not caption_records:
        return None

    if prompt_format == WAN_T5_PROMPT_FORMAT:
        if text_encoder is None:
            text_encoder, tokenizer = get_text_encoder_and_tokenizer(prompt_model_id)
    else:
        text_encoder, tokenizer = None, None
        if cosmos3_tokenizer is None:
            # Same reuse contract as the Wan text encoder above: scene-loop
            # callers pass a preloaded tokenizer so the snapshot resolve +
            # AutoTokenizer load does not repeat for every scene/variant.
            cosmos3_tokenizer = get_cosmos3_tokenizer(prompt_model_id)
        prompt_height, prompt_width = cosmos3_training_frame_size(images.shape[1], images.shape[2], prompt_model_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file_path = output_path.with_suffix(".tmp")
    with h5py.File(tmp_file_path, "w") as hf:
        hf.attrs["prompt_format"] = prompt_format
        for dataset_name, caption, selected_image_indices in caption_records:
            if prompt_format == WAN_T5_PROMPT_FORMAT:
                _write_wan_prompt_dataset(
                    hf,
                    dataset_name,
                    caption,
                    selected_image_indices,
                    text_encoder,
                    tokenizer,
                    text_encoder_max_sequence_length,
                    fps=dataset_fps / frame_stride,
                )
            else:
                write_cosmos3_prompt_dataset(
                    hf,
                    dataset_name,
                    caption,
                    selected_image_indices,
                    cosmos3_tokenizer,
                    num_frames=len(selected_image_indices),
                    height=prompt_height,
                    width=prompt_width,
                    fps=dataset_fps / frame_stride,
                )

    tmp_file_path.rename(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caption generation script")
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=None)
    # Optional ordering for captions generated from an image directory.
    parser.add_argument("--trajectory_path", type=Path, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--dataset_fps", type=float, default=DL3DV_DATASET_FPS)
    parser.add_argument("--dataset_downsample_factor", type=int, default=4)
    parser.add_argument("--captioning_model_id", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--captioning_attn_implementation", type=str, default=None)
    parser.add_argument("--text_encoder_max_sequence_length", default=512, type=int)
    parser.add_argument("--prompt_format", choices=PROMPT_FORMAT_CHOICES, default=WAN_T5_PROMPT_FORMAT)

    parser.add_argument("--check_if_exists", action="store_true")
    parser.add_argument("--throw_error_if_exists", action="store_true")
    parser.add_argument("--reverse_frames", action="store_true")
    parser.add_argument("--revisit", action="store_true")
    parser.add_argument(
        "--prompt_model_id",
        default=None,
        help="Override the model the prompts are encoded/tokenized with (a hub id or a local "
        "snapshot path for offline nodes); defaults to the prompt format's standard model.",
    )

    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame_stride must be >= 1")
    if args.num_frames is not None and args.num_frames <= 0:
        args.num_frames = None

    generate_caption_hdf5(
        args.input_path,
        args.output_path,
        num_frames=args.num_frames,
        trajectory_path=args.trajectory_path,
        frame_stride=args.frame_stride,
        dataset_fps=args.dataset_fps,
        dataset_downsample_factor=args.dataset_downsample_factor,
        captioning_model_id=args.captioning_model_id,
        captioning_attn_implementation=args.captioning_attn_implementation,
        text_encoder_max_sequence_length=args.text_encoder_max_sequence_length,
        prompt_format=args.prompt_format,
        prompt_model_id=args.prompt_model_id,
        check_if_exists=args.check_if_exists,
        throw_error_if_exists=args.throw_error_if_exists,
        reverse_frames=args.reverse_frames,
        revisit=args.revisit,
    )
