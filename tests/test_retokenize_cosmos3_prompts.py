# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem-fixture tests for the corpus regeneration tool
data_processing/captioning/retokenize_cosmos3_prompts.py.

The tool is the sole producer of the regenerated prompt corpora, and its
value-add over the writer (tested separately against checked-in expected
outputs) is pure derivation:
mapping a caption HDF5 back to its source scene zip, deriving the training
height/width from the downsampled source frame, carrying the recorded fps
(legacy files without one derive DL3DV_DATASET_FPS / stride from the
filename), the ``--replace`` skip semantics, and the .tmp atomic-rename. A
silent error in any of these would poison every metadata sentence of a
regenerated corpus, and the launch preflight machine-checks only
stamps/attrs, not content correctness — so these tests drive the real
``retokenize_file`` against a tiny on-disk scene (style of
test_pose_jump_window_filtering) for both caption formats.
"""

import argparse
import json
from io import BytesIO
from zipfile import ZipFile

import h5py
import numpy as np
import pytest

from data_processing.captioning import retokenize_cosmos3_prompts
from data_processing.captioning.camera_scrub import ScrubStats
from data_processing.captioning.prompt_hdf5 import tokenize_cosmos3_prompt
from data_processing.captioning.retokenize_cosmos3_prompts import _source_zip_path, retokenize_file
from data_processing.captioning.scrub_dl3dv_captions import scrub_caption_file
from model_training.constants import DL3DV_DATASET_FPS
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT, caption_filename_for_format

SUBDIR = "960P"
SCENE = "scene_a"
NUM_SOURCE_FRAMES = 6
# Downsampled (images_4) source frame size, deliberately NOT a multiple of 32
# so the test proves the derivation chain _source_frame_size ->
# cosmos3_training_frame_size: 60 -> 64, 100 -> 96.
SOURCE_HEIGHT = 60
SOURCE_WIDTH = 100
TRAIN_HEIGHT = 64
TRAIN_WIDTH = 96
FRAME_STRIDE = 2
CAPTION_FILENAME = f"frames_3_stride_{FRAME_STRIDE}.h5"
EXPECTED_FPS = DL3DV_DATASET_FPS / FRAME_STRIDE  # 30.0
# Two stride-2 windows over the 6 source frames; num_frames = 3 for both (the
# flat negative varies with the window geometry, computed once per file).
CAPTION_WINDOWS = {
    "frame_00000": ("a tiny synthetic scene", [0, 2, 4]),
    "frame_00001": ("another sweep of the same scene", [1, 3, 5]),
}
# Camera-sentence scrub fixtures: frame_00000 carries the observed trailing
# "The camera ..." sentence shape; frame_00001 is camera-free and must pass
# through untouched even with the scrub on. The kept text is long enough that
# the drop stays under the more-than-half guardrail.
SCRUB_KEPT_CAPTION = (
    "A stone bridge with three mossy arches crosses a quiet river, while low golden light "
    "glints off the wet cobblestones along the far bank."
)
SCRUB_CAMERA_SENTENCE = "The camera slowly pans across the scene, capturing the arches."
SCRUB_WINDOWS = {
    "frame_00000": (f"{SCRUB_KEPT_CAPTION} {SCRUB_CAMERA_SENTENCE}", [0, 2, 4]),
    "frame_00001": ("another sweep of the same scene", [1, 3, 5]),
}

# Same construction as the tiny_chat_tokenizer in
# tests/test_cosmos3_prompt_flat.py (kept local like that module does). Real
# transformers apply_chat_template jinja path, lossless byte-level vocab, no
# network.
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


def _write_scene_zip(dl3dv_dir, scene_id):
    from PIL import Image

    frames = [{"file_path": f"images/frame_{i:05d}.png"} for i in range(NUM_SOURCE_FRAMES)]
    (dl3dv_dir / SUBDIR).mkdir(parents=True, exist_ok=True)
    with ZipFile(dl3dv_dir / SUBDIR / f"{scene_id}.zip", "w") as zf:
        zf.writestr(f"{scene_id}/transforms.json", json.dumps({"frames": frames}))
        for frame in frames:
            png = BytesIO()
            Image.new("RGB", (SOURCE_WIDTH, SOURCE_HEIGHT)).save(png, format="PNG")
            zf.writestr(f"{scene_id}/{frame['file_path'].replace('images/', 'images_4/')}", png.getvalue())


def _write_caption_h5(input_dir, scene_id, windows=CAPTION_WINDOWS):
    caption_dir = input_dir / SUBDIR / scene_id
    caption_dir.mkdir(parents=True, exist_ok=True)
    caption_path = caption_dir / CAPTION_FILENAME
    with h5py.File(caption_path, "w") as f:
        for dataset_name, (caption, image_indices) in windows.items():
            # Placeholder ids as data; retokenize reads only the attrs.
            dataset = f.create_dataset(dataset_name, data=np.zeros(8, dtype=np.int32))
            dataset.attrs["caption"] = caption
            dataset.attrs["image_indices"] = np.asarray(image_indices, dtype=np.int64)
    return caption_path


@pytest.fixture()
def corpus(tmp_path):
    dl3dv_dir = tmp_path / "dl3dv"
    input_dir = tmp_path / "captions"
    output_dir = tmp_path / "prompts"
    _write_scene_zip(dl3dv_dir, SCENE)
    input_path = _write_caption_h5(input_dir, SCENE)
    output_path = output_dir / SUBDIR / SCENE / CAPTION_FILENAME
    return dl3dv_dir, input_dir, input_path, output_path


def _retokenize(corpus, tokenizer, *, replace=False):
    dl3dv_dir, input_dir, input_path, output_path = corpus
    retokenize_file(
        input_path,
        input_dir,
        output_path,
        tokenizer=tokenizer,
        dl3dv_dir=dl3dv_dir,
        replace=replace,
    )
    return output_path


def test_retokenize_file_writes_tokens_and_derived_geometry(corpus, tiny_chat_tokenizer):
    """The output h5 carries the stride-derived fps and token ids identical to
    tokenizing the stored caption at the training frame size derived from the
    source zip (60x100 -> 64x96), and exactly the data attrs — no format
    stamps."""
    output_path = _retokenize(corpus, tiny_chat_tokenizer)

    with h5py.File(output_path, "r") as hf:
        assert sorted(hf) == sorted(CAPTION_WINDOWS)
        for dataset_name, (caption, image_indices) in CAPTION_WINDOWS.items():
            expected = tokenize_cosmos3_prompt(
                caption,
                tiny_chat_tokenizer,
                num_frames=len(image_indices),
                height=TRAIN_HEIGHT,
                width=TRAIN_WIDTH,
                fps=EXPECTED_FPS,
            )
            dataset = hf[dataset_name]
            assert sorted(dataset.attrs) == ["caption", "fps", "image_indices"]
            assert float(dataset.attrs["fps"]) == EXPECTED_FPS
            # The caption and window stay retokenizable.
            assert dataset.attrs["caption"] == caption
            assert dataset.attrs["image_indices"].tolist() == image_indices
            assert dataset[()].tolist() == expected.input_ids.tolist()
            assert hf.attrs["negative_input_ids"].tolist() == expected.negative_input_ids.tolist()
    # Atomic rename: no .tmp residue after a successful write.
    assert list(output_path.parent.glob("*.tmp")) == []


def test_retokenize_file_flat_metadata_sentences_use_derived_fps_and_resolution(corpus, tiny_chat_tokenizer):
    """Observable end-to-end derivation on the flat arm: the metadata
    sentences literally embed fps = DL3DV_DATASET_FPS / stride and the
    rounded training resolution read back from the scene zip's downsampled
    frame."""
    output_path = _retokenize(corpus, tiny_chat_tokenizer)

    with h5py.File(output_path, "r") as hf:
        decoded_cond = tiny_chat_tokenizer.decode(hf["frame_00000"][()].tolist())
        decoded_neg = tiny_chat_tokenizer.decode(hf.attrs["negative_input_ids"].tolist())
    # 3 frames / 30 fps -> 0.1 seconds; 60x100 source -> 64x96 training size.
    assert "The video is 0.1 seconds long and is of 30 FPS." in decoded_cond
    assert "This video is of 64x96 resolution." in decoded_cond
    assert "The video is not 0.1 seconds long and is not of 30 FPS." in decoded_neg
    assert "This video is not of 64x96 resolution." in decoded_neg


@pytest.mark.parametrize("stride_in_name", [True, False], ids=["stride_filename", "personal_filename"])
@pytest.mark.parametrize("prompt_format", [WAN_T5_PROMPT_FORMAT, COSMOS3_PROMPT_FORMAT], ids=["wan", "cosmos3"])
def test_generated_fractional_fps_survives_retokenization(
    corpus, tiny_chat_tokenizer, monkeypatch, prompt_format, stride_in_name
):
    """Both prompt families record the FPS they were captioned at, so a
    generated 25 FPS / stride-2 file retokenizes at 12.5 FPS: the stride
    filename would derive 30 FPS (the legacy fallback) and a personal-scene
    filename has no stride at all."""
    from data_processing.captioning import generate_captions

    dl3dv_dir, input_dir, input_path, output_path = corpus
    caption = "A quiet stone courtyard."
    monkeypatch.setattr(generate_captions, "generate_caption", lambda *a: caption)
    monkeypatch.setattr(generate_captions, "generate_text_embedding", lambda *a: np.zeros((4, 8), dtype=np.uint16))
    generate_captions.generate_caption_hdf5(
        dl3dv_dir / SUBDIR / f"{SCENE}.zip",
        input_path,
        num_frames=3,
        frame_stride=2,
        dataset_fps=25.0,
        prompt_format=prompt_format,
        caption_model=object(),
        processor=object(),
        text_encoder=object(),
        tokenizer=object(),
        cosmos3_tokenizer=tiny_chat_tokenizer,
    )
    if not stride_in_name:
        input_path = input_path.rename(input_path.with_name(caption_filename_for_format("caption.h5", prompt_format)))
    retokenize_file(
        input_path, input_dir, output_path, tokenizer=tiny_chat_tokenizer, dl3dv_dir=dl3dv_dir, replace=False
    )

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "r") as dst:
        assert sorted(dst) == sorted(src)
        for name in src:
            expected = tokenize_cosmos3_prompt(
                caption,
                tiny_chat_tokenizer,
                num_frames=len(src[name].attrs["image_indices"]),
                height=TRAIN_HEIGHT,
                width=TRAIN_WIDTH,
                fps=12.5,
            )
            assert src[name].attrs["fps"] == dst[name].attrs["fps"] == 12.5
            assert dst[name][()].tolist() == expected.input_ids.tolist()
            if prompt_format == COSMOS3_PROMPT_FORMAT:
                np.testing.assert_array_equal(src[name][:], dst[name][:])
            assert dst.attrs["negative_input_ids"].tolist() == expected.negative_input_ids.tolist()
        decoded_cond = tiny_chat_tokenizer.decode(dst[sorted(dst)[0]][()].tolist())
        decoded_neg = tiny_chat_tokenizer.decode(dst.attrs["negative_input_ids"].tolist())
    assert "The video is 0.2 seconds long and is of 12 FPS." in decoded_cond
    assert "The video is not 0.2 seconds long and is not of 12 FPS." in decoded_neg


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_retokenization_rejects_invalid_recorded_fps(corpus, tiny_chat_tokenizer, fps):
    _, _, input_path, output_path = corpus
    with h5py.File(input_path, "a") as source:
        for payload in source.values():
            payload.attrs["fps"] = fps
    with pytest.raises(ValueError, match="invalid FPS"):
        _retokenize(corpus, tiny_chat_tokenizer)
    assert not output_path.exists()


def test_retokenize_file_replace_flag_controls_overwrite(corpus, tiny_chat_tokenizer):
    dl3dv_dir, input_dir, input_path, output_path = corpus
    _retokenize(corpus, tiny_chat_tokenizer)

    # A changed source caption is NOT picked up without --replace...
    updated = {name: ("an updated caption", indices) for name, (_, indices) in CAPTION_WINDOWS.items()}
    _write_caption_h5(input_dir, SCENE, windows=updated)
    _retokenize(corpus, tiny_chat_tokenizer, replace=False)
    with h5py.File(output_path, "r") as hf:
        assert hf["frame_00000"].attrs["caption"] == CAPTION_WINDOWS["frame_00000"][0]

    # ...and is picked up with it.
    _retokenize(corpus, tiny_chat_tokenizer, replace=True)
    with h5py.File(output_path, "r") as hf:
        assert hf["frame_00000"].attrs["caption"] == "an updated caption"


def test_retokenize_file_crash_mid_write_leaves_no_output(corpus, tiny_chat_tokenizer, monkeypatch):
    """The .tmp + rename scheme must never leave a partial file under the
    output name a training run would load."""
    _, _, _, output_path = corpus

    def _crash(*args, **kwargs):
        raise RuntimeError("simulated mid-write crash")

    monkeypatch.setattr(retokenize_cosmos3_prompts, "write_cosmos3_prompt_dataset", _crash)
    with pytest.raises(RuntimeError, match="simulated mid-write crash"):
        _retokenize(corpus, tiny_chat_tokenizer)

    assert not output_path.exists()
    assert list(output_path.parent.glob("*.h5")) == []


def test_source_zip_path_arithmetic(tmp_path):
    """<subdir>/<scene>/<file>.h5 maps to <dl3dv>/<subdir>/<scene>.zip, and a
    subdir-less <scene>/<file>.h5 maps to <dl3dv>/<scene>.zip."""
    input_dir = tmp_path / "captions"
    dl3dv_dir = tmp_path / "dl3dv"

    nested = input_dir / SUBDIR / SCENE / CAPTION_FILENAME
    assert _source_zip_path(nested, input_dir, dl3dv_dir) == dl3dv_dir / SUBDIR / f"{SCENE}.zip"

    top_level = input_dir / SCENE / CAPTION_FILENAME
    assert _source_zip_path(top_level, input_dir, dl3dv_dir) == dl3dv_dir / f"{SCENE}.zip"

    with pytest.raises(AssertionError, match="Prompt path must be"):
        _source_zip_path(input_dir / "orphan.h5", input_dir, dl3dv_dir)


# ---------------------------------------------------------------------------
# The camera-sentence scrub at retokenization time. The scrub itself
# (sentence shapes, guardrail, byte preservation) is pinned in
# test_camera_sentence_scrub.py; these tests pin the tool-level contract.
# ---------------------------------------------------------------------------


def test_retokenize_file_scrub_stores_scrubbed_caption_and_matching_tokens(corpus, tiny_chat_tokenizer):
    """The output `caption` attr is the SCRUBBED text and the stored ids are
    exactly what tokenizing that scrubbed text directly produces (no double
    transform), so the regenerated corpus is self-consistent."""
    dl3dv_dir, input_dir, input_path, output_path = corpus
    _write_caption_h5(input_dir, SCENE, windows=SCRUB_WINDOWS)

    stats = retokenize_file(
        input_path,
        input_dir,
        output_path,
        tokenizer=tiny_chat_tokenizer,
        dl3dv_dir=dl3dv_dir,
        replace=False,
    )

    assert stats.captions_changed == 1
    assert stats.sentences_dropped == 1
    assert stats.skipped_suspicious == 0
    with h5py.File(output_path, "r") as hf:
        scrubbed_dataset = hf["frame_00000"]
        assert scrubbed_dataset.attrs["caption"] == SCRUB_KEPT_CAPTION
        expected = tokenize_cosmos3_prompt(
            SCRUB_KEPT_CAPTION,
            tiny_chat_tokenizer,
            num_frames=3,
            height=TRAIN_HEIGHT,
            width=TRAIN_WIDTH,
            fps=EXPECTED_FPS,
        )
        assert scrubbed_dataset[()].tolist() == expected.input_ids.tolist()
        # The camera-free window passes through untouched.
        assert hf["frame_00001"].attrs["caption"] == SCRUB_WINDOWS["frame_00001"][0]


def test_retokenize_file_scrub_guardrail_keeps_suspicious_captions(corpus, tiny_chat_tokenizer):
    """A scrub that would empty a caption or remove more than half of it keeps
    the ORIGINAL caption and is reported as skipped_suspicious; the run never
    fails."""
    dl3dv_dir, input_dir, input_path, output_path = corpus
    windows = {
        # Scrubbing would empty the caption entirely.
        "frame_00000": ("The camera pans across the scene, capturing the skyline.", [0, 2, 4]),
        # Scrubbing would remove more than half of the text.
        "frame_00001": (
            "A door. The camera moves forward along the hallway, capturing a long row of identical doors.",
            [1, 3, 5],
        ),
    }
    _write_caption_h5(input_dir, SCENE, windows=windows)

    stats = retokenize_file(
        input_path,
        input_dir,
        output_path,
        tokenizer=tiny_chat_tokenizer,
        dl3dv_dir=dl3dv_dir,
        replace=False,
    )

    assert stats.captions_changed == 0
    assert stats.sentences_dropped == 0
    assert stats.skipped_suspicious == 2
    with h5py.File(output_path, "r") as hf:
        for dataset_name, (caption, _) in windows.items():
            assert hf[dataset_name].attrs["caption"] == caption


def test_retokenize_tree_prints_scrub_counts_for_the_ops_marker(corpus, tiny_chat_tokenizer, monkeypatch, capsys):
    dl3dv_dir, input_dir, input_path, output_path = corpus
    _write_caption_h5(input_dir, SCENE, windows=SCRUB_WINDOWS)
    monkeypatch.setattr(retokenize_cosmos3_prompts, "get_cosmos3_tokenizer", lambda model_id: tiny_chat_tokenizer)

    retokenize_cosmos3_prompts.retokenize_tree(
        argparse.Namespace(
            input_dir=input_dir,
            dl3dv_dir=dl3dv_dir,
            # The corpus fixture's output root (output_path is <root>/<subdir>/<scene>/<file>).
            output_dir=output_path.parents[2],
            replace=False,
        )
    )

    summary = capsys.readouterr().out
    assert "retokenize summary:" in summary
    assert "files_scanned=1" in summary
    assert "files_skipped=0" in summary
    assert "retokenize+scrub" in summary
    assert "files_changed=1" in summary
    assert "sentences_dropped=1" in summary
    assert "skipped_suspicious=0" in summary

    # A rerun without --replace skips the existing output: the summary reports
    # the skip instead of presenting zero drop counters as a clean corpus.
    retokenize_cosmos3_prompts.retokenize_tree(
        argparse.Namespace(
            input_dir=input_dir,
            dl3dv_dir=dl3dv_dir,
            output_dir=output_path.parents[2],
            replace=False,
        )
    )
    summary = capsys.readouterr().out
    assert "files_scanned=0" in summary
    assert "files_skipped=1" in summary
    assert "files_changed=0" in summary


def test_retokenize_tree_into_the_input_tree_names_outputs_by_format(corpus, tiny_chat_tokenizer, monkeypatch, capsys):
    """output_dir == input_dir is the eval layout: outputs land
    beside the Wan caption files under the caption_filename_for_format name,
    the source files stay untouched, and a rerun neither re-reads the
    generated files as inputs nor stacks a second format segment."""
    dl3dv_dir, input_dir, input_path, _ = corpus
    monkeypatch.setattr(retokenize_cosmos3_prompts, "get_cosmos3_tokenizer", lambda model_id: tiny_chat_tokenizer)
    args = argparse.Namespace(input_dir=input_dir, dl3dv_dir=dl3dv_dir, output_dir=input_dir, replace=False)

    retokenize_cosmos3_prompts.retokenize_tree(args)

    expected_name = caption_filename_for_format(CAPTION_FILENAME, COSMOS3_PROMPT_FORMAT)
    output_path = input_path.with_name(expected_name)
    with h5py.File(output_path, "r") as hf:
        assert sorted(hf) == sorted(CAPTION_WINDOWS)
    # The Wan-named source is untouched (still the raw caption file).
    with h5py.File(input_path, "r") as hf:
        assert hf["frame_00000"].attrs["caption"] == CAPTION_WINDOWS["frame_00000"][0]
    assert "files_scanned=1" in capsys.readouterr().out

    retokenize_cosmos3_prompts.retokenize_tree(args)

    # The rerun skips the existing output instead of retokenizing it.
    summary = capsys.readouterr().out
    assert "files_scanned=0" in summary
    assert "files_skipped=1" in summary
    assert sorted(path.name for path in input_path.parent.iterdir()) == sorted([CAPTION_FILENAME, expected_name])


def test_retokenize_tree_separate_output_tree_keeps_the_input_filename(corpus, tiny_chat_tokenizer, monkeypatch):
    """A separate output_dir is the training-corpus layout: dl3dv_base globs
    the plain frames_* stems there, so the output keeps the input name."""
    dl3dv_dir, input_dir, _, output_path = corpus
    monkeypatch.setattr(retokenize_cosmos3_prompts, "get_cosmos3_tokenizer", lambda model_id: tiny_chat_tokenizer)

    retokenize_cosmos3_prompts.retokenize_tree(
        argparse.Namespace(input_dir=input_dir, dl3dv_dir=dl3dv_dir, output_dir=output_path.parents[2], replace=False)
    )

    assert sorted(path.name for path in output_path.parent.iterdir()) == [CAPTION_FILENAME]


def test_retokenize_scrub_is_a_no_op_on_a_pre_scrubbed_raw_corpus(corpus, tiny_chat_tokenizer, tmp_path):
    """Idempotence across the two scrub tools: after scrub_dl3dv_captions.py
    stages a scrubbed raw corpus, retokenizing THAT corpus reports zero
    additional drops."""
    dl3dv_dir, input_dir, input_path, _ = corpus
    _write_caption_h5(input_dir, SCENE, windows=SCRUB_WINDOWS)

    scrubbed_dir = tmp_path / "captions_scrubbed"
    scrubbed_path = scrubbed_dir / SUBDIR / SCENE / CAPTION_FILENAME
    raw_stats = scrub_caption_file(input_path, scrubbed_path, replace=False)
    assert raw_stats.captions_changed == 1
    assert raw_stats.sentences_dropped == 1

    output_path = tmp_path / "prompts_from_scrubbed" / SUBDIR / SCENE / CAPTION_FILENAME
    stats = retokenize_file(
        scrubbed_path,
        scrubbed_dir,
        output_path,
        tokenizer=tiny_chat_tokenizer,
        dl3dv_dir=dl3dv_dir,
        replace=False,
    )

    assert stats == ScrubStats()
    with h5py.File(output_path, "r") as hf:
        assert hf["frame_00000"].attrs["caption"] == SCRUB_KEPT_CAPTION
