# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Cosmos3 prompt interface: system prompt + caption + duration/fps/
resolution metadata sentences, inverse-metadata negative.

The template bytes are load-bearing — the production corpus stores tokens
produced from exactly these templates at the pinned tokenizer revision, so
any template change silently desyncs writer and corpus. These tests pin:

(a) the template STRINGS themselves;
(b) the tokenized structure on a tiny lossless chat tokenizer;
(c) checked-in expected token ids from the real tokenizer for one verified
    window (169-token cond / 63-token neg), skipped unless the pinned
    tokenizer is in the local HF cache;
(d) the writer -> load_prompt_payload roundtrip, including the text-dropout
    substitution of the inverse-metadata negative.
"""

import h5py
import numpy as np
import pytest
import torch

from data_processing.captioning.prompt_hdf5 import (
    COSMOS3_VIDEO_SYSTEM_PROMPT,
    _format_cosmos3_prompt_text,
    tokenize_cosmos3_prompt,
    write_cosmos3_prompt_dataset,
)
from model_training.data.utils import load_prompt_payload
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, prompt_payload_for_text_dropout

# A real transformers apply_chat_template jinja path over a lossless
# byte-level vocab, no network (the retokenize tests carry the same local
# fixture).
_QWEN2_STYLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_chat_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {char: index for index, char in enumerate(sorted(alphabet))}
    tokenizer_object = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer_object.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_object.decoder = decoders.ByteLevel()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        eos_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|vision_start|>"],
    )
    tokenizer.chat_template = _QWEN2_STYLE_CHAT_TEMPLATE
    return tokenizer


# ---------------------------------------------------------------------------
# (a) Template-string equality against the recovered code.
# Provenance: GitLab blob 419369a8ccf26e55e67d250bef244eed351a4cee,
# content_sha256 7c950c8852d887311e68203a8bf3a723dfbdf0b21aa41a6a6c35ed4f9d3f7d66.
# ---------------------------------------------------------------------------


def test_flat_system_prompt_is_the_recovered_string():
    # Including the original's "a give prompt" typo: byte parity is the contract.
    assert COSMOS3_VIDEO_SYSTEM_PROMPT == ("You are a helpful assistant who will generate videos from a give prompt.")


def test_flat_metadata_sentences_match_the_recovered_templates():
    assert _format_cosmos3_prompt_text(
        "a bright training scene.", is_negative=False, num_frames=81, height=544, width=960, fps=60.0
    ) == (
        "a bright training scene. The video is 1.4 seconds long and is of 60 FPS. "
        "This video is of 544x960 resolution."
    )
    # The recovered uncond: inverse-metadata sentences of the EMPTY caption.
    assert _format_cosmos3_prompt_text("", is_negative=True, num_frames=81, height=544, width=960, fps=60.0) == (
        "The video is not 1.4 seconds long and is not of 60 FPS. " "This video is not of 544x960 resolution."
    )


def test_flat_format_requires_multi_frame_video():
    # Recovered code asserted num_frames > 1 for the video templates.
    with pytest.raises(AssertionError, match="num_frames > 1"):
        _format_cosmos3_prompt_text("a caption", is_negative=False, num_frames=1, height=544, width=960, fps=60.0)


def test_trailing_whitespace_does_not_malform_the_sentence_separator():
    """A caption ending in whitespace/newline used to block the trailing-period
    strip and produce "caption.\n. The video is ..." — a malformed prompt the
    corpus writer would happily tokenize. Clean captions are byte-unchanged."""
    assert _format_cosmos3_prompt_text(
        "a bright training scene.\n", is_negative=False, num_frames=81, height=544, width=960, fps=60.0
    ) == (
        "a bright training scene. The video is 1.4 seconds long and is of 60 FPS. "
        "This video is of 544x960 resolution."
    )


# ---------------------------------------------------------------------------
# (b) Structural asserts on the tiny lossless tokenizer (the same checks the
# verification probe ran).
# ---------------------------------------------------------------------------


def test_flat_tokenization_carries_system_prompt_metadata_and_vision_start(tiny_chat_tokenizer):
    tokens = tokenize_cosmos3_prompt(
        "a bright training scene",
        tiny_chat_tokenizer,
        num_frames=81,
        height=544,
        width=960,
        fps=60.0,
    )

    decoded_cond = tiny_chat_tokenizer.decode(tokens.input_ids.tolist())
    decoded_uncond = tiny_chat_tokenizer.decode(tokens.negative_input_ids.tolist())

    expected_prefix = f"<|im_start|>system\n{COSMOS3_VIDEO_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    assert decoded_cond.startswith(expected_prefix)
    assert decoded_uncond.startswith(expected_prefix)
    assert decoded_cond.endswith("<|vision_start|>")
    assert decoded_uncond.endswith("<|vision_start|>")
    # Manual [eos, <|vision_start|>] append after chat templating.
    expected_suffix = [
        tiny_chat_tokenizer.eos_token_id,
        tiny_chat_tokenizer.convert_tokens_to_ids("<|vision_start|>"),
    ]
    assert tokens.input_ids[-2:].tolist() == expected_suffix
    assert tokens.negative_input_ids[-2:].tolist() == expected_suffix
    # Metadata + inverse-metadata sentences present (TRUE fps 60 in the
    # prose; rope ignores fps via the COSMOS3_ROPE_FPS pin).
    assert "The video is 1.4 seconds long and is of 60 FPS. This video is of 544x960 resolution." in decoded_cond
    assert (
        "The video is not 1.4 seconds long and is not of 60 FPS. This video is not of 544x960 resolution."
        in decoded_uncond
    )
    assert "a bright training scene" in decoded_cond
    assert "a bright training scene" not in decoded_uncond


def test_flat_tokenization_clips_degenerate_captions_to_the_training_token_cap(tiny_chat_tokenizer):
    """Degenerate
    repetition-loop captions exist in the corpus (regen jobs 29857122/29857129)
    and must be clipped by the binary-search caption clip at generation time."""
    over_length_caption = "a scene with far too many words to ever fit " * 60

    with pytest.warns(UserWarning, match="Clipped a degenerate caption"):
        tokens = tokenize_cosmos3_prompt(
            over_length_caption,
            tiny_chat_tokenizer,
            num_frames=81,
            height=544,
            width=960,
            fps=60.0,
        )

    assert len(tokens.input_ids) <= 1024
    decoded = tiny_chat_tokenizer.decode(tokens.input_ids.tolist())
    # The metadata sentences survive the clip (built AROUND the clipped text).
    assert "This video is of 544x960 resolution." in decoded


def test_pad_token_falls_back_to_eos_like_reference_processor(tiny_chat_tokenizer):
    """Reference processors/base.py _resolve_pad_id: pad_token_id or the EOS id."""
    assert tiny_chat_tokenizer.pad_token_id is None

    tokens = tokenize_cosmos3_prompt(
        "a bright training scene",
        tiny_chat_tokenizer,
        num_frames=81,
        height=544,
        width=960,
        fps=60.0,
    )

    assert tokens.pad_token_id == tiny_chat_tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# (c) Checked-in expected token ids from the real tokenizer for one verified window:
# scene 2385549d* "shark carousel", 81 frames / 544x960 / fps 60, at the
# pinned tokenizer revision 23314034b82f46b45339035dba67c3ee9bbcb8ba
# (model_training/pins.py COSMOS3_NANO_HF_REVISION). These ids were
# cross-checked byte-identical against the production corpus's stored
# tokens for the same window (169-token cond / 63-token neg).
# ---------------------------------------------------------------------------

_PROBE_FRAME_00115_CAPTION = (
    "A colorful children's amusement ride at an outdoor park, featuring cartoonish animal-shaped seats in "
    "bright green, yellow, red, and blue, arranged in a circular pattern. The ride is enclosed by a metal "
    "railing and filled with small multicolored balls. At the center, a large blue shark sculpture with a "
    "wide-open mouth serves as the main structure, surrounded by smaller animal figures. The background "
    "shows green trees under a cloudy sky, with a small building and a sign visible in the distance. The "
    "ride is stationary, with no people present."
)

# fmt: off
_PROBE_EXPECTED_COND_IDS = [
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 879, 686, 6923, 6803, 504, 264, 2968, 9934, 13,
    151645, 198, 151644, 872, 198, 32, 33866, 2841, 594, 61368, 11877, 518, 458, 15789, 6118, 11, 16445,
    30349, 812, 9864, 34731, 16312, 304, 9906, 6176, 11, 13753, 11, 2518, 11, 323, 6303, 11, 27802, 304,
    264, 26931, 5383, 13, 576, 11877, 374, 43810, 553, 264, 9317, 91830, 323, 10199, 448, 2613, 2745,
    26463, 3018, 20332, 13, 2411, 279, 4126, 11, 264, 3460, 6303, 43792, 49967, 448, 264, 6884, 25686,
    10780, 17045, 438, 279, 1887, 5944, 11, 22865, 553, 9155, 9864, 12396, 13, 576, 4004, 4933, 6176,
    12408, 1212, 264, 73549, 12884, 11, 448, 264, 2613, 4752, 323, 264, 1841, 9434, 304, 279, 6010, 13,
    576, 11877, 374, 52635, 11, 448, 902, 1251, 3042, 13, 576, 2766, 374, 220, 16, 13, 19, 6486, 1293,
    323, 374, 315, 220, 21, 15, 43628, 13, 1096, 2766, 374, 315, 220, 20, 19, 19, 87, 24, 21, 15, 10935,
    13, 151645, 198, 151644, 77091, 198, 151645, 151652,
]
_PROBE_EXPECTED_NEG_IDS = [
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 879, 686, 6923, 6803, 504, 264, 2968, 9934, 13,
    151645, 198, 151644, 872, 198, 785, 2766, 374, 537, 220, 16, 13, 19, 6486, 1293, 323, 374, 537, 315,
    220, 21, 15, 43628, 13, 1096, 2766, 374, 537, 315, 220, 20, 19, 19, 87, 24, 21, 15, 10935, 13,
    151645, 198, 151644, 77091, 198, 151645, 151652,
]
# fmt: on


def test_probe_expected_token_ids_on_the_real_pinned_tokenizer():
    """Byte-for-byte agreement with the production corpus, transitively:
    these expected ids match the corpus's stored tokens for this window, so
    the writer reproducing them reproduces the corpus. Runs wherever the
    pinned tokenizer snapshot is in the local HF cache (e.g. the training
    container); skips otherwise so the default suite stays network-free."""
    from data_processing.captioning.prompt_hdf5 import get_cosmos3_tokenizer

    try:
        tokenizer = get_cosmos3_tokenizer(COSMOS3_NANO_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - any load failure means "unavailable here"
        pytest.skip(f"pinned Cosmos3-Nano text tokenizer not available locally: {exc}")

    tokens = tokenize_cosmos3_prompt(
        _PROBE_FRAME_00115_CAPTION,
        tokenizer,
        num_frames=81,
        height=544,
        width=960,
        fps=60.0,
    )

    assert len(tokens.input_ids) == 169
    assert len(tokens.negative_input_ids) == 63
    assert tokens.input_ids.tolist() == _PROBE_EXPECTED_COND_IDS
    assert tokens.negative_input_ids.tolist() == _PROBE_EXPECTED_NEG_IDS
    # The old always-on system prompt IS the flat interface (unlike the JSON format).
    decoded_cond = tokenizer.decode(tokens.input_ids.tolist())
    assert decoded_cond.startswith(f"<|im_start|>system\n{COSMOS3_VIDEO_SYSTEM_PROMPT}")


# ---------------------------------------------------------------------------
# (d) Writer -> loader roundtrip under expected=flat.
# ---------------------------------------------------------------------------


def test_flat_prompt_hdf5_roundtrip_and_dropout_use_the_inverse_metadata_negative(tiny_chat_tokenizer, tmp_path):
    expected = tokenize_cosmos3_prompt(
        "a bright training scene",
        tiny_chat_tokenizer,
        num_frames=81,
        height=544,
        width=960,
        fps=60.0,
    )

    prompt_path = tmp_path / "frames_81_stride_1.h5"
    with h5py.File(prompt_path, "w") as hf:
        for dataset_name, caption, start in (
            ("frame_0", "a bright training scene", 0),
            ("frame_1", "another sweep of the same scene", 1),
        ):
            write_cosmos3_prompt_dataset(
                hf,
                dataset_name,
                caption,
                np.arange(start, start + 81, dtype=np.int64),
                tiny_chat_tokenizer,
                num_frames=81,
                height=544,
                width=960,
                fps=60.0,
            )

    with h5py.File(prompt_path, "r") as hf:
        # The writer stores the TRUE fps (60/stride).
        assert float(hf["frame_0"].attrs["fps"]) == 60.0
        payload = load_prompt_payload(
            hf,
            hf["frame_0"],
            prompt_format=COSMOS3_PROMPT_FORMAT,
        )

    cond_length = len(expected.input_ids)
    assert payload["encoded_prompt"][:cond_length].tolist() == expected.input_ids.tolist()
    assert payload["prompt_attention_mask"][:cond_length].all()
    assert not payload["prompt_attention_mask"][cond_length:].any()
    neg_length = len(expected.negative_input_ids)
    assert payload["encoded_negative_prompt"][:neg_length].tolist() == expected.negative_input_ids.tolist()
    assert not payload["negative_prompt_attention_mask"][neg_length:].any()

    # Text dropout substitutes the stored INVERSE-METADATA negative (the flat
    # regime's uncond), through the same prompt_payload_for_text_dropout the
    # trainer uses — train dropout, validation CFG, and DMD score CFG all read
    # this stored tensor.
    data = {
        key: payload[key].unsqueeze(0)
        for key in (
            "encoded_prompt",
            "prompt_attention_mask",
            "encoded_negative_prompt",
            "negative_prompt_attention_mask",
        )
    }
    dropped = prompt_payload_for_text_dropout(data, drop_text=True)
    assert torch.equal(dropped["encoded_prompt"], data["encoded_negative_prompt"])
    assert torch.equal(dropped["prompt_attention_mask"], data["negative_prompt_attention_mask"])
    decoded_dropped = tiny_chat_tokenizer.decode(
        dropped["encoded_prompt"][0][dropped["prompt_attention_mask"][0].bool()].tolist()
    )
    assert "The video is not 1.4 seconds long and is not of 60 FPS." in decoded_dropped


def test_flat_negative_is_shared_per_file_and_varies_with_geometry(tiny_chat_tokenizer, tmp_path):
    """The flat negative depends on (num_frames, H, W, fps). Within one file
    those are computed
    once (retokenize_file), so the writer's per-file consistency assert holds;
    a mixed-geometry write into the same file must trip it."""
    prompt_path = tmp_path / "frames_81_stride_1.h5"
    with h5py.File(prompt_path, "w") as hf:
        write_cosmos3_prompt_dataset(
            hf,
            "frame_0",
            "a bright training scene",
            np.arange(81, dtype=np.int64),
            tiny_chat_tokenizer,
            num_frames=81,
            height=544,
            width=960,
            fps=60.0,
        )
        with pytest.raises(AssertionError, match="negative_input_ids"):
            write_cosmos3_prompt_dataset(
                hf,
                "frame_1",
                "a bright training scene",
                np.arange(81, dtype=np.int64),
                tiny_chat_tokenizer,
                num_frames=81,
                height=704,
                width=1280,
                fps=60.0,
            )


def test_get_cosmos3_tokenizer_loads_local_snapshot_directories_without_the_hub(tmp_path, monkeypatch):
    """--prompt_model_id promises local snapshot paths work: a directory
    argument loads AutoTokenizer straight from <dir>/text_tokenizer and must
    never call snapshot_download (offline nodes have no hub)."""
    from pathlib import Path

    import huggingface_hub
    import transformers

    from data_processing.captioning.prompt_hdf5 import get_cosmos3_tokenizer

    snapshot = tmp_path / "snapshot"
    (snapshot / "text_tokenizer").mkdir(parents=True)
    sentinel = object()
    recorded = []

    def fake_from_pretrained(path, *args, **kwargs):
        recorded.append(Path(path))
        return sentinel

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("local directories must not resolve through the hub")),
    )

    tokenizer = get_cosmos3_tokenizer(str(snapshot))

    assert tokenizer is sentinel
    assert recorded == [snapshot / "text_tokenizer"]
