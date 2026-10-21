# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-neutral prompt-format registry and prompt-payload policy.

Both model families flow through here: the prompt-format constants name the
on-disk payload encodings (Wan pre-encoded T5 embeddings vs Cosmos3 token
ids), the eval caption filename is derived from the format by
caption_filename_for_format, and the text-dropout/unconditional substitution
policy is shared by training dropout, validation zero-prompt tasks, and the
eval no-text diagnostic. Cosmos3-only tokenizer mechanics stay in
cosmos3_text.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

from model_training.constants import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT

PROMPT_FORMAT_CHOICES = (WAN_T5_PROMPT_FORMAT, COSMOS3_PROMPT_FORMAT)


def validate_prompt_file_format(path: Path, expected: str, *, expected_fps: float | None = None) -> None:
    """Reject a reused caption file from another family before loading models.

    Legacy files without a format attribute are admitted by their payload
    layout. This checks representation, not source-frame or tokenizer parity.
    """
    import h5py
    import numpy as np

    from model_training.cosmos3_text import COSMOS3_DEFAULT_MAX_TEXT_TOKENS

    def validate_ids(value, label):
        ids = np.asarray(value)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or not 0 < ids.size <= COSMOS3_DEFAULT_MAX_TEXT_TOKENS
        ):
            raise ValueError(
                f"{path}:{label} must be a nonempty 1D integer array of at most {COSMOS3_DEFAULT_MAX_TEXT_TOKENS} tokens"
            )

    if expected not in PROMPT_FORMAT_CHOICES:
        raise ValueError(f"Unsupported prompt format: {expected}")
    with h5py.File(path, "r") as source:
        recorded = source.attrs.get("prompt_format")
        if recorded is not None and recorded != expected:
            raise ValueError(f"{path} records prompt_format={recorded}, expected {expected}")
        if not len(source):
            raise ValueError(f"{path} has no prompt records")
        if expected == COSMOS3_PROMPT_FORMAT:
            for attr in ("negative_input_ids", "pad_token_id"):
                if attr not in source.attrs:
                    raise ValueError(f"{path} is missing Cosmos3 {attr}")
            validate_ids(source.attrs["negative_input_ids"], "negative_input_ids")
            pad = np.asarray(source.attrs["pad_token_id"])
            if pad.ndim != 0 or not np.issubdtype(pad.dtype, np.integer):
                raise ValueError(f"{path}:pad_token_id must be a scalar integer")
        for name, payload in source.items():
            if not isinstance(payload, h5py.Dataset):
                raise ValueError(f"{path}:{name} is not a prompt dataset")
            if expected == WAN_T5_PROMPT_FORMAT:
                valid = payload.ndim == 2 and payload.dtype == np.uint16
            else:
                valid = payload.ndim == 1 and np.issubdtype(payload.dtype, np.integer)
            if not valid or payload.shape[0] == 0:
                raise ValueError(f"{path}:{name} payload does not match {expected}")
            if expected == COSMOS3_PROMPT_FORMAT and payload.size > COSMOS3_DEFAULT_MAX_TEXT_TOKENS:
                raise ValueError(f"{path}:{name} exceeds {COSMOS3_DEFAULT_MAX_TEXT_TOKENS} tokens")
            if expected == COSMOS3_PROMPT_FORMAT and expected_fps is not None:
                recorded_fps = float(payload.attrs.get("fps", float("nan")))
                if not np.isfinite(recorded_fps) or not np.isclose(recorded_fps, expected_fps, rtol=0, atol=1e-9):
                    raise ValueError(f"{path}:{name} records FPS {recorded_fps}, expected {expected_fps}")


def caption_filename_for_format(base_filename: str, prompt_format: str) -> str:
    """Both families' eval captions share one tree per dataset; the prompt
    format picks the filename inside it.

    The Wan T5 format returns the base name unchanged: the caption trees on
    the clusters were written under exactly these names, so this branch must
    stay byte-identical to keep them loadable without any data migration.
    The Cosmos3 token-id format inserts a "cosmos3" segment before the
    extension, placing its files beside the Wan ones. An already-tagged
    filename is retained, including both .h5 and .hdf5 files.
    """
    if prompt_format == WAN_T5_PROMPT_FORMAT:
        return base_filename
    if prompt_format == COSMOS3_PROMPT_FORMAT:
        stem, dot, extension = base_filename.rpartition(".")
        assert dot, f"Caption filename {base_filename!r} has no extension to place the format segment before"
        if stem.endswith(".cosmos3"):
            return base_filename
        return f"{stem}.cosmos3.{extension}"
    raise ValueError(f"Unknown prompt format {prompt_format!r}; choices: {PROMPT_FORMAT_CHOICES}")


def substitute_unconditional_prompt(
    prompt: torch.Tensor,
    prompt_attention_mask: torch.Tensor | None,
    stored_unconditional: torch.Tensor | None,
    stored_unconditional_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The single unconditional-prompt policy: substitute the stored
    unconditional (plus its mask) when the payload carries one, otherwise
    zero the embedding.

    Payloads with a stored unconditional (Cosmos3 token ids, whose
    prompt-writer stores the empty-caption unconditional) swap it in —
    zeroing token ids would yield a garbage prompt of L copies of token id 0.
    Payloads without one (Wan T5 embeddings) drop to zeros, the Wan no-text
    convention.
    """
    if stored_unconditional is not None:
        return stored_unconditional, stored_unconditional_mask
    return torch.zeros_like(prompt), prompt_attention_mask


def prompt_payload_for_text_dropout(data: dict[str, torch.Tensor], *, drop_text: bool) -> dict[str, torch.Tensor]:
    """Assemble the training prompt payload, substituting the unconditional on dropout.

    For Cosmos3 (token-id prompts) the dropout target is ``encoded_negative_prompt``
    — the empty-caption unconditional stored at prompt-generation time (the
    reference tokenizer implements CFG-dropout and the CFG unconditional by
    replacing the caption with the empty string; see prompt_hdf5.py). The
    validation unconditional branch reads the same tensor, keeping train-time
    and eval-time unconditioning identical by construction. The dropout RATE
    is owned by the caller (``args.dropout_rate``, pinned at 0.1). Wan
    T5-embedding prompts have no stored unconditional and drop to zeros.
    """
    encoded_prompt = data["encoded_prompt"]
    prompt_attention_mask = data.get("prompt_attention_mask")
    encoded_negative_prompt = data.get("encoded_negative_prompt")
    negative_prompt_attention_mask = data.get("negative_prompt_attention_mask")

    if drop_text:
        encoded_prompt, prompt_attention_mask = substitute_unconditional_prompt(
            encoded_prompt,
            prompt_attention_mask,
            encoded_negative_prompt,
            negative_prompt_attention_mask,
        )

    payload = {"encoded_prompt": encoded_prompt}
    if prompt_attention_mask is not None:
        payload["prompt_attention_mask"] = prompt_attention_mask
    if encoded_negative_prompt is not None:
        payload["encoded_negative_prompt"] = encoded_negative_prompt
    if negative_prompt_attention_mask is not None:
        payload["negative_prompt_attention_mask"] = negative_prompt_attention_mask
    return payload
