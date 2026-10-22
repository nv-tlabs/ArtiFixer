# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import h5py

from model_training.cosmos3_text import COSMOS3_DEFAULT_MAX_TEXT_TOKENS
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID, spatial_resize_multiple
from model_training.pins import tokenizer_revision_for_model
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, PROMPT_FORMAT_CHOICES

WAN_TEXT_ENCODER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Prompt text templates. The conditional wraps the caption in
# duration/fps/resolution metadata sentences under the chat system prompt;
# the unconditional (CFG negative) is the EMPTY caption wrapped in the
# inverse-metadata sentences — the exact text interface the base model was
# trained with. Template bytes are pinned by the token-parity tests
# (tests/test_cosmos3_prompt_flat.py); changing them means regenerating the
# prompt corpus into a NEW directory.
#
# NOTE on fps: the metadata sentences state the clip's TRUE fps (60/stride,
# same value as the h5 `fps` attr). Rope spacing ignores fps — it is pinned
# separately (model_profiles.COSMOS3_ROPE_FPS) — do not "fix" the sentences
# to 24.
COSMOS3_VIDEO_SYSTEM_PROMPT = "You are a helpful assistant who will generate videos from a give prompt."


def _append_sentence(base: str, addition: str) -> str:
    # Trailing whitespace would block the period strip below and leave a
    # malformed "...text.\n. addition" prompt, so it goes first. Stripping
    # ALL trailing periods (an ellipsis collapses to nothing) is part of the
    # pinned template behavior — the token-parity tests hold it in place.
    base = base.rstrip().rstrip(".")
    return f"{base}. {addition}" if base else addition


def _format_cosmos3_prompt_text(
    text: str, *, is_negative: bool, num_frames: int, height: int, width: int, fps: float
) -> str:
    assert num_frames > 1, "flat-format video prompt requires num_frames > 1"
    if is_negative:
        duration = f"The video is not {num_frames / fps:.1f} seconds long and is not of {fps:.0f} FPS."
    else:
        duration = f"The video is {num_frames / fps:.1f} seconds long and is of {fps:.0f} FPS."
    text = _append_sentence(text, duration)
    if is_negative:
        resolution = f"This video is not of {height}x{width} resolution."
    else:
        resolution = f"This video is of {height}x{width} resolution."
    return _append_sentence(text, resolution)


def _tokenize_chat_prompt(text: str, tokenizer: Any) -> list[int]:
    conversations = [
        {"role": "system", "content": COSMOS3_VIDEO_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    encodings = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=True,
    )
    return list(encodings.input_ids)


@dataclass(frozen=True)
class Cosmos3PromptTokens:
    input_ids: np.ndarray
    # The CFG unconditional: the EMPTY caption wrapped in the inverse-metadata
    # sentences (the base model's tokenizer implements CFG dropout by
    # replacing the caption with ""). Consumed by training text dropout and
    # validation CFG alike.
    negative_input_ids: np.ndarray
    pad_token_id: int


def default_prompt_model_id(prompt_format: str) -> str:
    assert (
        prompt_format in PROMPT_FORMAT_CHOICES
    ), f"prompt_format must be one of {PROMPT_FORMAT_CHOICES}, got {prompt_format!r}"
    return COSMOS3_NANO_MODEL_ID if prompt_format == COSMOS3_PROMPT_FORMAT else WAN_TEXT_ENCODER_MODEL_ID


def get_cosmos3_tokenizer(model_id: str, *, cache_dir=None, local_files_only: bool = False) -> Any:
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # A local snapshot directory (offline nodes) loads straight from the
    # filesystem; snapshot_download only accepts hub repo ids.
    if Path(model_id).is_dir():
        return AutoTokenizer.from_pretrained(Path(model_id) / "text_tokenizer")

    # Prompt HDF5s are tokenized at the pin-registry revision, not whatever
    # `main` resolves to at generation time.
    #
    # Resolve the pinned snapshot to a local directory and load from the
    # filesystem: the repo-id path makes AutoTokenizer probe optional files
    # (tokenizer.model, chat templates, ...) through the hub, and under
    # HF_HUB_OFFLINE=1 a hand-populated cache carries no `.no_exist` records,
    # so those probes raise LocalEntryNotFoundError instead of returning None.
    # A local-directory load only stats the filesystem.
    from huggingface_hub.utils import LocalEntryNotFoundError

    download_kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
    try:
        snapshot_dir = snapshot_download(
            model_id,
            revision=tokenizer_revision_for_model(model_id),
            allow_patterns=["text_tokenizer/*"],
            local_files_only=True,
            **download_kwargs,
        )
        # A cached snapshot can contain only transformer/VAE files. Hub's
        # local lookup does not check allow_patterns. The pinned fast tokenizer
        # needs its configuration, serialized vocabulary and chat template.
        tokenizer_dir = Path(snapshot_dir) / "text_tokenizer"
        required_files = ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja")
        missing = [name for name in required_files if not (tokenizer_dir / name).is_file()]
        if missing:
            raise LocalEntryNotFoundError(f"Incomplete pinned tokenizer in {tokenizer_dir}: missing {missing}")
    except LocalEntryNotFoundError:
        if local_files_only:
            raise
        # Not fully cached yet: fetch the pinned revision. Under HF_HUB_OFFLINE=1
        # the hub client refuses network access with a clear error, so the
        # offline contract still holds.
        snapshot_dir = snapshot_download(
            model_id,
            revision=tokenizer_revision_for_model(model_id),
            allow_patterns=["text_tokenizer/*"],
            **download_kwargs,
        )
    return AutoTokenizer.from_pretrained(Path(snapshot_dir) / "text_tokenizer")


def cosmos3_training_frame_size(height: int, width: int, model_id: str = COSMOS3_NANO_MODEL_ID) -> tuple[int, int]:
    multiple = spatial_resize_multiple(model_id)
    assert multiple > 0, f"spatial resize multiple must be positive, got {multiple}"
    return round(height / multiple) * multiple, round(width / multiple) * multiple


def tokenize_cosmos3_prompt(
    caption: str,
    tokenizer: Any,
    *,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
) -> Cosmos3PromptTokens:
    """Tokenize one caption into the stored (cond, uncond) token payload.

    Conditional = system prompt + caption + duration/fps/resolution metadata
    sentences; unconditional = system prompt + the inverse-metadata sentences
    of the EMPTY caption. Both streams get ``[eos, <|vision_start|>]``
    appended after chat templating.
    """
    assert num_frames >= 1, f"num_frames must be >= 1, got {num_frames}"
    assert math.isfinite(fps) and fps > 0, f"fps must be finite and positive, got {fps}"

    def _tokenize(caption_text: str) -> tuple[list[int], list[int]]:
        cond_text = _format_cosmos3_prompt_text(
            caption_text, is_negative=False, num_frames=num_frames, height=height, width=width, fps=fps
        )
        # The unconditional is the EMPTY caption wrapped in the
        # inverse-metadata sentences.
        uncond_text = _format_cosmos3_prompt_text(
            "",
            is_negative=True,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )
        eos_token_id = int(tokenizer.eos_token_id)
        vision_start_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        cond_ids = _tokenize_chat_prompt(cond_text, tokenizer)
        uncond_ids = _tokenize_chat_prompt(uncond_text, tokenizer)
        cond_ids.extend([eos_token_id, vision_start_id])
        uncond_ids.extend([eos_token_id, vision_start_id])
        return cond_ids, uncond_ids

    cond_ids, uncond_ids = _tokenize(caption)

    # Length bound at GENERATION time: the training loader pads to
    # COSMOS3_DEFAULT_MAX_TEXT_TOKENS and hard-asserts inside the DataLoader
    # worker (pad_cosmos3_input_ids), so an over-length caption written here
    # would crash a training run mid-stream at data-load time. Healthy DL3DV
    # captions (~80-100 words + the metadata sentences) leave large headroom, but
    # degenerate captioner outputs exist in the corpus (observed during
    # corpus regeneration: repetition loops like a single CJK glyph repeated
    # hundreds of times). For those, clip the CAPTION TEXT (never the token
    # stream — that would sever the metadata sentences and special-token
    # tail) until the tokenized prompt fits, and warn with the scene context
    # so the offending captions can be re-captioned upstream.
    if len(cond_ids) > COSMOS3_DEFAULT_MAX_TEXT_TOKENS:
        original_tokens = len(cond_ids)
        low, high = 0, len(caption)
        while low < high:
            mid = (low + high + 1) // 2
            if len(_tokenize(caption[:mid])[0]) <= COSMOS3_DEFAULT_MAX_TEXT_TOKENS:
                low = mid
            else:
                high = mid - 1
        clipped_caption = caption[:low]
        cond_ids, uncond_ids = _tokenize(clipped_caption)
        warnings.warn(
            f"Clipped a degenerate caption from {original_tokens} tokens ({len(caption)} chars) to "
            f"{len(cond_ids)} tokens ({len(clipped_caption)} chars) to fit the training loader's cap of "
            f"{COSMOS3_DEFAULT_MAX_TEXT_TOKENS} (COSMOS3_DEFAULT_MAX_TEXT_TOKENS). Re-caption this scene "
            f"upstream. Caption head: {caption[:120]!r}",
            stacklevel=2,
        )

    for token_kind, token_ids in (("conditional", cond_ids), ("unconditional", uncond_ids)):
        assert len(token_ids) <= COSMOS3_DEFAULT_MAX_TEXT_TOKENS, (
            f"Tokenized {token_kind} prompt is {len(token_ids)} tokens, above the training loader's "
            f"cap of {COSMOS3_DEFAULT_MAX_TEXT_TOKENS} (COSMOS3_DEFAULT_MAX_TEXT_TOKENS) even after "
            f"caption clipping. Caption ({len(caption)} chars): {caption[:200]!r}..."
        )

    # Pad id falls back to the EOS id when the tokenizer defines none.
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    return Cosmos3PromptTokens(
        input_ids=np.asarray(cond_ids, dtype=np.int32),
        negative_input_ids=np.asarray(uncond_ids, dtype=np.int32),
        pad_token_id=int(pad_token_id),
    )


def write_cosmos3_prompt_dataset(
    hf: h5py.File,
    dataset_name: str,
    caption: str,
    selected_image_indices: np.ndarray,
    tokenizer: Any,
    *,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
) -> None:
    tokens = tokenize_cosmos3_prompt(
        caption,
        tokenizer,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
    )
    if "negative_input_ids" not in hf.attrs:
        hf.attrs["negative_input_ids"] = tokens.negative_input_ids
        hf.attrs["pad_token_id"] = tokens.pad_token_id
    else:
        # The unconditional payload is stored once per file; a mismatch means
        # the file's datasets were tokenized inconsistently (e.g. mixed
        # tokenizer revisions across a partial regeneration). The flat
        # negative varies with (num_frames, H, W, fps) ACROSS files, but those
        # are computed once per file (retokenize_file), so the per-file
        # consistency assert holds — a future multi-geometry
        # file would trip it, which is the right guard.
        assert np.array_equal(hf.attrs["negative_input_ids"], tokens.negative_input_ids), (
            f"negative_input_ids for dataset {dataset_name!r} disagree with the file-level attr: "
            f"{tokens.negative_input_ids.tolist()} vs {np.asarray(hf.attrs['negative_input_ids']).tolist()}"
        )
        assert int(hf.attrs["pad_token_id"]) == tokens.pad_token_id, (
            f"pad_token_id for dataset {dataset_name!r} disagrees with the file-level attr: "
            f"{tokens.pad_token_id} vs {int(hf.attrs['pad_token_id'])}"
        )

    dataset = hf.create_dataset(dataset_name, data=tokens.input_ids)
    # The scene caption TEXT that was tokenized (post-scrub) is stored — not
    # the formatted prompt — so files stay retokenizable and the attr always
    # matches the tokens. fps is the clip's TRUE fps (60/stride; the metadata
    # sentences state the same value — rope spacing is pinned separately via
    # COSMOS3_ROPE_FPS).
    dataset.attrs["caption"] = caption
    dataset.attrs["image_indices"] = selected_image_indices
    dataset.attrs["fps"] = float(fps)
