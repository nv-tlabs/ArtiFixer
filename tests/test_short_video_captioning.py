# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from data_processing import run_captioning
from data_processing.captioning import caption_plan, generate_captions
from data_processing.captioning.caption_plan import scene_limited_num_frames
from model_training.data.dl3dv_base import _default_prompt_paths_for_scene, _pingpong_frame_indices
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT, caption_filename_for_format


def _write_scene(scene_dir: Path, frame_count: int, image_size: tuple[int, int] = (4, 4)) -> None:
    image_dir = scene_dir / "images_4"
    image_dir.mkdir(parents=True)
    frames = []
    for index in range(frame_count):
        image_name = f"{index:03d}.png"
        Image.fromarray(np.full((*image_size, 3), index, dtype=np.uint8)).save(image_dir / image_name)
        frames.append(
            {
                "file_path": f"images/{image_name}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    (scene_dir / "transforms.json").write_text(json.dumps({"frames": frames}))


def test_short_caption_request_writes_truthful_frames_all_file(tmp_path: Path, monkeypatch) -> None:
    scene_dir = tmp_path / "scene"
    _write_scene(scene_dir, frame_count=3)

    caption_lengths = []

    def fake_generate_caption(images, fps, model, processor):
        caption_lengths.append(len(images))
        return f"caption over {len(images)} frames"

    monkeypatch.setattr(generate_captions, "generate_caption", fake_generate_caption)
    monkeypatch.setattr(
        generate_captions.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        generate_captions.Qwen3VLProcessor,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(generate_captions, "get_text_encoder_and_tokenizer", lambda _model_id: (object(), object()))
    monkeypatch.setattr(
        generate_captions,
        "generate_text_embedding",
        lambda _caption, _text_encoder, _tokenizer, _max_sequence_length: np.array([[1, 2]], dtype=np.uint16),
    )

    requested_output = tmp_path / "captions" / "frames_1533_stride_1.h5"
    actual_output = generate_captions.generate_caption_hdf5(
        scene_dir,
        requested_output,
        num_frames=1533,
        frame_stride=1,
        check_if_exists=True,
    )

    fallback_output = requested_output.parent / "frames_all_stride_1.h5"
    assert not requested_output.exists()
    assert actual_output == fallback_output
    assert fallback_output.exists()
    assert caption_lengths == [3]

    with h5py.File(fallback_output, "r") as h5_file:
        [dataset_name] = list(h5_file.keys())
        assert dataset_name == "000.png"
        assert h5_file[dataset_name].attrs["caption"] == "caption over 3 frames"
        np.testing.assert_array_equal(h5_file[dataset_name].attrs["image_indices"], np.array([0, 1, 2]))


def test_short_input_with_caller_chosen_filename_raises(tmp_path: Path, monkeypatch) -> None:
    # Renaming a caller-chosen filename to the frames_all convention could
    # collide with another input captioned into the same directory, so only
    # canonical windowed names fall back; everything else fails loudly.
    scene_dir = tmp_path / "scene"
    _write_scene(scene_dir, frame_count=3)
    monkeypatch.setattr(generate_captions, "generate_caption", lambda *a, **k: "caption")
    monkeypatch.setattr(generate_captions.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(generate_captions.Qwen3VLProcessor, "from_pretrained", lambda *a, **k: object())

    with pytest.raises(ValueError, match="Not enough frames"):
        generate_captions.generate_caption_hdf5(
            scene_dir,
            tmp_path / "captions" / "sceneA.h5",
            num_frames=81,
            frame_stride=1,
        )


def test_prompt_paths_fall_back_to_frames_all_only_for_short_scenes(tmp_path: Path) -> None:
    scene_prompt_dir = tmp_path / "prompts"
    scene_prompt_dir.mkdir()
    frames_all = scene_prompt_dir / "frames_all_stride_1.h5"
    frames_all.touch()

    assert _default_prompt_paths_for_scene(scene_prompt_dir, num_frames=1533, scene_num_frames=3) == [frames_all]
    assert _default_prompt_paths_for_scene(scene_prompt_dir, num_frames=1533, scene_num_frames=2000) == []

    frames_1533 = scene_prompt_dir / "frames_1533_stride_1.h5"
    frames_1533.touch()
    assert _default_prompt_paths_for_scene(scene_prompt_dir, num_frames=1533, scene_num_frames=3) == [frames_all]
    assert _default_prompt_paths_for_scene(scene_prompt_dir, num_frames=1533, scene_num_frames=2000) == [frames_1533]
    assert _default_prompt_paths_for_scene(scene_prompt_dir, num_frames=None, scene_num_frames=2000) == [frames_all]
    # A scene exactly window-length uses windowed prompts, not the fallback.
    assert _default_prompt_paths_for_scene(scene_prompt_dir, num_frames=1533, scene_num_frames=1533) == [frames_1533]


def test_prompt_paths_exclude_format_tagged_eval_captions(tmp_path: Path) -> None:
    """The eval layout writes "<base>.cosmos3.h5" beside the training captions, which
    the "{stem}_*.h5" glob matches; those must stay out of the training prompt pool
    while the augmented captions it exists for stay in."""
    scene_prompt_dir = tmp_path / "prompts"
    scene_prompt_dir.mkdir()
    base = scene_prompt_dir / "frames_81_stride_1.h5"
    augmented = scene_prompt_dir / "frames_81_stride_1_aug.h5"
    tagged = scene_prompt_dir / caption_filename_for_format(augmented.name, COSMOS3_PROMPT_FORMAT)
    assert tagged.name == "frames_81_stride_1_aug.cosmos3.h5"
    for path in (base, augmented, tagged):
        path.touch()

    assert _default_prompt_paths_for_scene(scene_prompt_dir, num_frames=81, scene_num_frames=300) == [base, augmented]


def test_scene_limited_num_frames_falls_back_for_short_scenes(monkeypatch) -> None:
    scene = Path("scene.zip")
    monkeypatch.setattr(caption_plan, "scene_frame_count", lambda _p: 335)
    assert scene_limited_num_frames(1533, scene) is None
    monkeypatch.setattr(caption_plan, "scene_frame_count", lambda _p: 1533)
    assert scene_limited_num_frames(1533, scene) == 1533
    monkeypatch.setattr(caption_plan, "scene_frame_count", lambda _p: 2000)
    assert scene_limited_num_frames(1533, scene) == 1533
    # None / <= 0 mean full clip and must not read the zip.
    monkeypatch.setattr(
        caption_plan,
        "scene_frame_count",
        lambda _p: (_ for _ in ()).throw(AssertionError("full-clip requests must not read the zip")),
    )
    assert scene_limited_num_frames(None, scene) is None
    assert scene_limited_num_frames(0, scene) is None


def test_pingpong_frame_indices_repeats_single_frame() -> None:
    expanded = _pingpong_frame_indices(
        np.array([7], dtype=np.int64),
        num_frames=4,
        generator=torch.Generator().manual_seed(0),
    )

    np.testing.assert_array_equal(expanded, np.array([7, 7, 7, 7]))


def _reference_reflection_walk(frame_indices: np.ndarray, num_frames: int, generator: torch.Generator) -> np.ndarray:
    """Independent reimplementation of the pingpong walk, kept as a test oracle."""
    index = torch.randint(0, len(frame_indices), size=(1,), generator=generator)
    direction = torch.randint(0, 2, size=(1,), generator=generator) * 2 - 1
    walked = []
    while len(walked) < num_frames:
        walked.append(frame_indices[index])
        if index == 0:
            direction = 1
        elif index == len(frame_indices) - 1:
            direction = -1
        index += direction
    return np.asarray([int(x) for x in walked], dtype=np.int64)


def test_pingpong_matches_the_reference_reflection_walk_exactly() -> None:
    for length, num_frames in ((5, 5), (3, 8), (2, 7), (5, 12)):
        # Non-contiguous values so a walk over positions cannot masquerade as
        # a walk over the index values themselves.
        indices = np.array([7, 11, 23, 40, 52][:length], dtype=np.int64) * 3
        for seed in range(20):
            expected = _reference_reflection_walk(indices, num_frames, torch.Generator().manual_seed(seed))
            actual = _pingpong_frame_indices(indices, num_frames, generator=torch.Generator().manual_seed(seed))
            np.testing.assert_array_equal(actual, expected, err_msg=f"length={length}, seed={seed}")


def test_generated_captions_are_scrubbed_of_camera_sentences_at_the_source(tmp_path: Path, monkeypatch) -> None:
    """The captioning instruction forbids camera-movement narration, but the
    model leaks a dedicated "The camera ..." sentence into a few percent of
    captions; generate_caption_hdf5 enforces the instruction by scrubbing its
    own output."""
    scene_dir = tmp_path / "scene"
    _write_scene(scene_dir, frame_count=3)

    scene_text = (
        "A weathered wooden pier extends into a calm mountain lake at sunrise, its planks "
        "streaked with dew and its pilings wrapped in soft mist rising off the water."
    )
    leaked = f"{scene_text} The camera slowly pans across the scene, capturing the still water."
    monkeypatch.setattr(generate_captions, "generate_caption", lambda *a, **k: leaked)
    monkeypatch.setattr(generate_captions.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(generate_captions.Qwen3VLProcessor, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(generate_captions, "get_text_encoder_and_tokenizer", lambda _model_id: (object(), object()))
    encoded_texts: list[str] = []

    def fake_embedding(caption, _text_encoder, _tokenizer, _max_sequence_length):
        encoded_texts.append(caption)
        return np.array([[1, 2]], dtype=np.uint16)

    monkeypatch.setattr(generate_captions, "generate_text_embedding", fake_embedding)

    output_path = generate_captions.generate_caption_hdf5(
        scene_dir,
        tmp_path / "captions" / "frames_3_stride_1.h5",
        num_frames=3,
        frame_stride=1,
    )

    expected = scene_text
    with h5py.File(output_path, "r") as h5_file:
        [dataset_name] = list(h5_file.keys())
        assert h5_file[dataset_name].attrs["caption"] == expected
    # The stored attr and the encoded text see the SAME caption.
    assert encoded_texts == [expected]


def test_generation_scrub_reports_guardrail_kept_captions(tmp_path: Path, monkeypatch, capsys) -> None:
    """A caption that is mostly camera narration is KEPT by the guardrail and
    still contains camera motion; the per-file summary must say so — a
    clean-looking drop count alone would misreport the corpus as scrubbed."""
    scene_dir = tmp_path / "scene"
    _write_scene(scene_dir, frame_count=3)

    leaked = "The camera pans across the scene, capturing the skyline."
    monkeypatch.setattr(generate_captions, "generate_caption", lambda *a, **k: leaked)
    monkeypatch.setattr(generate_captions.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(generate_captions.Qwen3VLProcessor, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(generate_captions, "get_text_encoder_and_tokenizer", lambda _model_id: (object(), object()))
    monkeypatch.setattr(
        generate_captions,
        "generate_text_embedding",
        lambda *_args: np.array([[1, 2]], dtype=np.uint16),
    )

    output_path = generate_captions.generate_caption_hdf5(
        scene_dir,
        tmp_path / "captions" / "frames_3_stride_1.h5",
        num_frames=3,
        frame_stride=1,
    )

    with h5py.File(output_path, "r") as h5_file:
        [dataset_name] = list(h5_file.keys())
        assert h5_file[dataset_name].attrs["caption"] == leaked
    summary = capsys.readouterr().out
    assert "kept 1 suspicious captions" in summary


# --- cosmos3 (token-id) corpus writing, end-to-end with a faked captioner ---

# Same construction as the tiny_chat_tokenizer in
# tests/test_cosmos3_prompt_flat.py (kept local like that module does).
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


def test_cosmos3_caption_hdf5_writes_token_ids_negatives_and_scrubbed_captions(
    tmp_path: Path, monkeypatch, tiny_chat_tokenizer
) -> None:
    """The cosmos3 corpus writer end-to-end (faked captioner, real writer):
    token-id datasets (not T5 embeddings), the per-file negative payload, the
    camera-sentence scrub, the true-division fps, and the training-frame-size
    prompt geometry all land in the written file."""
    from data_processing.captioning.prompt_hdf5 import cosmos3_training_frame_size, tokenize_cosmos3_prompt
    from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT

    scene_dir = tmp_path / "scene"
    # 11 source frames at 60x100: stride 5 windows of 3 frames ([0, 5, 10]),
    # and a fps the floor-division bug would corrupt (24 / 5 = 4.8).
    _write_scene(scene_dir, frame_count=11, image_size=(60, 100))

    scene_text = (
        "A stone bridge with three mossy arches crosses a quiet river, while low golden light "
        "glints off the wet cobblestones along the far bank."
    )
    leaked = f"{scene_text} The camera slowly pans across the scene, capturing the arches."
    captioner_calls = []

    def fake_generate_caption(images, fps, model, processor):
        captioner_calls.append((len(images), fps))
        return leaked

    monkeypatch.setattr(generate_captions, "generate_caption", fake_generate_caption)
    monkeypatch.setattr(generate_captions.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(generate_captions.Qwen3VLProcessor, "from_pretrained", lambda *a, **k: object())
    # Token-id corpora must never touch the Wan T5 stack.
    monkeypatch.setattr(
        generate_captions,
        "get_text_encoder_and_tokenizer",
        lambda _model_id: (_ for _ in ()).throw(AssertionError("cosmos3 prompts must not load the T5 encoder")),
    )

    output_path = generate_captions.generate_caption_hdf5(
        scene_dir,
        tmp_path / "captions" / "frames_3_stride_5.h5",
        num_frames=3,
        frame_stride=5,
        dataset_fps=24,
        prompt_format=COSMOS3_PROMPT_FORMAT,
        cosmos3_tokenizer=tiny_chat_tokenizer,
    )

    # The captioner saw the true clip speed, not floor(24 / 5).
    assert captioner_calls == [(3, 24 / 5)]

    prompt_height, prompt_width = cosmos3_training_frame_size(60, 100)
    expected = tokenize_cosmos3_prompt(
        scene_text,  # scrubbed: the trailing camera sentence must be gone
        tiny_chat_tokenizer,
        num_frames=3,
        height=prompt_height,
        width=prompt_width,
        fps=24 / 5,
    )
    with h5py.File(output_path, "r") as h5_file:
        [dataset_name] = list(h5_file.keys())
        assert dataset_name == "000.png"
        dataset = h5_file[dataset_name]
        assert np.issubdtype(dataset.dtype, np.integer)
        np.testing.assert_array_equal(dataset[()], expected.input_ids)
        assert dataset.attrs["caption"] == scene_text
        np.testing.assert_array_equal(dataset.attrs["image_indices"], np.array([0, 5, 10]))
        assert dataset.attrs["fps"] == 24 / 5
        # The CFG unconditional is stored once per file.
        np.testing.assert_array_equal(h5_file.attrs["negative_input_ids"], expected.negative_input_ids)
        assert int(h5_file.attrs["pad_token_id"]) == expected.pad_token_id


def test_cosmos3_caption_hdf5_refuses_single_frame_inputs(tmp_path: Path) -> None:
    """Cosmos3 prompts state a clip duration/fps, so single images cannot be
    captioned into that format — refuse before any model loads."""
    from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT

    scene_dir = tmp_path / "scene"
    _write_scene(scene_dir, frame_count=1)

    with pytest.raises(ValueError, match="single-frame inputs cannot be captioned"):
        generate_captions.generate_caption_hdf5(
            scene_dir,
            tmp_path / "captions" / "frames_all_stride_1.h5",
            prompt_format=COSMOS3_PROMPT_FORMAT,
        )


# --- run_captioning.generate_for_scene: worker-side caption planning ---


def _worker_args(tmp_path: Path, num_frames: int | None) -> argparse.Namespace:
    return argparse.Namespace(
        dl3dv_dir=tmp_path / "dataset",
        output_dir=tmp_path / "captions",
        num_frames=num_frames,
        max_frame_stride=1,
        include_reverse=True,
        include_revisit=True,
        captioning_model_id=None,
        captioning_attn_implementation=None,
        text_encoder_model_id=None,
        prompt_format=WAN_T5_PROMPT_FORMAT,
        output_role="training",
        dataset_fps=60,
    )


def _stub_lazy_model_load(monkeypatch) -> None:
    # The outputs don't exist, so generate_for_scene's lazy first-missing-output
    # model load would hit the Hugging Face hub; these tests pin the caption
    # plan only, so stub it out.
    monkeypatch.setattr(run_captioning, "load_models", lambda args: {})


def test_generate_for_scene_collapses_short_scene_to_one_frames_all_pass(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_generate(scene_path, output_path, **kwargs):
        calls.append((output_path, kwargs))

    monkeypatch.setattr(run_captioning, "generate_caption_hdf5", fake_generate)
    monkeypatch.setattr(caption_plan, "scene_frame_count", lambda scene_path: 335)
    _stub_lazy_model_load(monkeypatch)

    args = _worker_args(tmp_path, num_frames=1533)
    run_captioning.generate_for_scene(args, args.dl3dv_dir / "scene.zip", models={})

    # Training reads only the plain frames_all file for short scenes, so the
    # reverse/revisit variants must not each spend a full captioning pass.
    assert len(calls) == 1
    output_path, kwargs = calls[0]
    assert kwargs["num_frames"] is None
    assert not kwargs["revisit"]
    assert not kwargs["reverse_frames"]
    assert output_path == tmp_path / "captions" / "scene" / "frames_all_stride_1.h5"


def test_generate_for_scene_treats_nonpositive_num_frames_as_full_clip(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_generate(scene_path, output_path, **kwargs):
        calls.append((output_path, kwargs))

    monkeypatch.setattr(run_captioning, "generate_caption_hdf5", fake_generate)
    # <= 0 must not read the scene zip at all.
    monkeypatch.setattr(
        caption_plan,
        "scene_frame_count",
        lambda _scene_path: (_ for _ in ()).throw(AssertionError("full-clip requests must not read the zip")),
    )
    _stub_lazy_model_load(monkeypatch)

    args = _worker_args(tmp_path, num_frames=0)
    # Explicit full-clip requests keep their requested variants (only the
    # short-scene collapse suppresses them), so pin the full-clip handling
    # itself with the variant flags off.
    args.include_reverse = False
    args.include_revisit = False
    run_captioning.generate_for_scene(args, args.dl3dv_dir / "scene.zip", models={})

    assert len(calls) == 1
    output_path, kwargs = calls[0]
    assert kwargs["num_frames"] is None
    assert not kwargs["revisit"]
    assert not kwargs["reverse_frames"]
    assert output_path == tmp_path / "captions" / "scene" / "frames_all_stride_1.h5"


def test_generate_for_scene_keeps_windowed_variants_for_long_scenes(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_generate(scene_path, output_path, **kwargs):
        calls.append((output_path, kwargs))

    monkeypatch.setattr(run_captioning, "generate_caption_hdf5", fake_generate)
    monkeypatch.setattr(caption_plan, "scene_frame_count", lambda scene_path: 2000)
    _stub_lazy_model_load(monkeypatch)

    args = _worker_args(tmp_path, num_frames=1533)
    run_captioning.generate_for_scene(args, args.dl3dv_dir / "scene.zip", models={})

    assert len(calls) == 4
    for _output_path, kwargs in calls:
        assert kwargs["num_frames"] == 1533
