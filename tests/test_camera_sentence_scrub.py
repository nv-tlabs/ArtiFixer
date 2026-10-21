# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera-sentence scrub: camera_scrub.py and the raw-corpus staging tool
data_processing/captioning/scrub_dl3dv_captions.py.

Roughly 10% of the raw DL3DV captions carry a dedicated sentence narrating
camera movement ("The camera slowly pans across the scene, capturing...")
that slipped past the captioning instruction to never describe camera
movement; the validated training recipe requires camera-motion-free caption
text. These tests pin the observed sentence shapes, the subject-position-only
match rule, byte-for-byte preservation of kept text, the suspicious-scrub
guardrail, and the staging tool's bit-exact copy contract (the retokenizer's
scrub pass is covered in test_retokenize_cosmos3_prompts).
"""

import argparse
import hashlib

import h5py
import numpy as np
import pytest

from data_processing.captioning.camera_scrub import ScrubStats, apply_camera_scrub, strip_camera_sentences
from data_processing.captioning.scrub_dl3dv_captions import scrub_caption_file, scrub_tree

# Long enough that dropping one camera sentence never trips the more-than-half
# guardrail (real DL3DV captions are ~80-100 words).
RAW_KEPT_CAPTION = (
    "A weathered wooden pier extends into a calm mountain lake at sunrise, its planks "
    "streaked with dew and its pilings wrapped in soft mist rising off the water."
)
RAW_CAMERA_SENTENCE = "The camera slowly pans across the scene, capturing the still water."
CLEAN_CAPTION = "A quiet reading room with tall oak shelves and a brass lamp on every table."


# ---------------------------------------------------------------------------
# strip_camera_sentences: the pure scrub.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "camera_sentence",
    [
        "The camera slowly pans across the scene, capturing the fine detail of the stonework.",
        "The camera moves forward along the bridge, capturing the railings and the water below.",
    ],
    ids=["pans-across", "moves-forward"],
)
def test_observed_trailing_camera_sentences_are_removed(camera_sentence):
    """The exact sentence shapes verified in the raw corpus: a standalone
    trailing sentence whose subject is "The camera"."""
    caption = f"A stone bridge crosses a quiet river at dusk. {camera_sentence}"
    assert strip_camera_sentences(caption) == ("A stone bridge crosses a quiet river at dusk.", 1, 0)


def test_mid_caption_camera_sentence_is_removed_and_neighbors_rejoin_with_one_space():
    caption = (
        "A stone bridge crosses a quiet river. "
        "The camera moves forward along the bridge, capturing the railings. "
        "Green trees line both banks."
    )
    assert strip_camera_sentences(caption) == (
        "A stone bridge crosses a quiet river. Green trees line both banks.",
        1,
        0,
    )


def test_match_allows_a_lowercase_sentence_start_and_leading_whitespace():
    caption = "A quiet plaza at noon.  the camera pans left across the fountains."
    assert strip_camera_sentences(caption) == ("A quiet plaza at noon.", 1, 0)


def test_a_proper_noun_subject_beginning_the_camera_survives():
    """ "camera" is matched lowercase exactly: a capitalized "Camera" is part
    of a proper-noun subject, not camera-movement narration."""
    caption = (
        "The Camera Obscura building stands at the corner, its white tower rising above "
        "the rooftops. A row of gift shops lines the street below."
    )
    assert strip_camera_sentences(caption) == (caption, 0, 0)


def test_an_abbreviation_inside_a_camera_sentence_is_kept_and_reported():
    """An abbreviation dot is not a sentence boundary, so a camera span
    containing one MAY be two merged sentences. It is kept whole and counted
    as ambiguous — never cut at "St." (fragment) and never dropped whole
    (which could silently delete merged scene text)."""
    caption = (
        f"{RAW_KEPT_CAPTION} " "The camera pans across the St. Mary church facade, capturing the carved stone doorway."
    )
    assert strip_camera_sentences(caption) == (caption, 0, 1)
    stored, dropped, suspicious = apply_camera_scrub(caption)
    assert stored == caption and dropped == 0 and suspicious is True


def test_exclamation_terminated_sentences_are_boundaries():
    """'!' and '?' end sentences: a camera sentence after an exclamation is
    still scrubbed, and an exclamation-terminated camera sentence cannot
    swallow the legitimate sentence after it."""
    caption = "It's a grand old bridge! The camera pans across it slowly. Boats pass underneath."
    assert strip_camera_sentences(caption) == ("It's a grand old bridge! Boats pass underneath.", 1, 0)

    caption = (
        "A grand stone bridge spans the wide river in warm afternoon light. "
        "The camera moves forward along the bridge! Boats pass underneath between its arches."
    )
    scrubbed, dropped, ambiguous = strip_camera_sentences(caption)
    assert "Boats pass underneath" in scrubbed and dropped == 1 and ambiguous == 0


def test_camera_text_hiding_behind_a_suppressed_abbreviation_boundary_is_reported():
    """A kept scene sentence ending in an abbreviation can have a camera
    sentence merged in behind the suppressed boundary; it must never be kept
    silently — it is reported suspicious. A non-camera continuation after the
    same abbreviation stays kept and unsuspicious."""
    leak = f"{RAW_KEPT_CAPTION} A row of shops stands near Main St. The camera pans across the scene."
    assert strip_camera_sentences(leak) == (leak, 0, 1)
    stored, dropped, suspicious = apply_camera_scrub(leak)
    assert stored == leak and dropped == 0 and suspicious is True

    control = f"{RAW_KEPT_CAPTION} A row of shops stands near Main St. The wide shot shows the rooftops."
    assert strip_camera_sentences(control) == (control, 0, 0)
    assert apply_camera_scrub(control) == (control, 0, False)


def test_multiple_camera_sentences_are_all_removed_and_counted():
    caption = (
        "The camera starts at the gate. A gravel path leads to a wooden cabin. "
        "The camera moves forward along the path, capturing the door. "
        "Snow covers the roof. The camera tilts up toward the chimney."
    )
    assert strip_camera_sentences(caption) == (
        "A gravel path leads to a wooden cabin. Snow covers the roof.",
        3,
        0,
    )


def test_sentences_with_camera_in_non_subject_position_survive():
    """A caption can describe a camera IN the scene; only subject-position
    "The camera" sentences are camera-movement narration."""
    caption = (
        "A camera store display fills the shop window with lenses and tripods. "
        "An old film camera sits on a wooden shelf beside the register."
    )
    assert strip_camera_sentences(caption) == (caption, 0, 0)


def test_a_different_subject_sharing_the_prefix_survives():
    """Word boundary after "camera": "The cameraman ..." is a different subject."""
    caption = "The cameraman's silhouette is reflected in the shop window."
    assert strip_camera_sentences(caption) == (caption, 0, 0)


def test_kept_text_is_preserved_byte_for_byte_including_weird_spacing():
    """Only the dropped sentence and its boundary whitespace go: weird spacing
    inside AND between kept sentences survives, and a decimal point ("3.5") is
    not a sentence boundary."""
    caption = (
        "A  narrow   alley, about 3.5 m wide, runs between brick walls."
        "  Laundry lines cross overhead. "
        "The camera pans across the scene, capturing the walls."
    )
    assert strip_camera_sentences(caption) == (
        "A  narrow   alley, about 3.5 m wide, runs between brick walls.  Laundry lines cross overhead.",
        1,
        0,
    )


def test_caption_without_camera_sentences_is_returned_unchanged():
    # Leading/trailing/internal weird whitespace and no final period: a no-op
    # scrub is byte-identical.
    caption = "  A bright courtyard.   Two windows face   the street  "
    assert strip_camera_sentences(caption) == (caption, 0, 0)


def test_empty_caption_is_a_no_op():
    assert strip_camera_sentences("") == ("", 0, 0)


def test_caption_of_only_camera_sentences_scrubs_to_empty():
    assert strip_camera_sentences(
        "The camera slowly pans across the scene. The camera tilts down toward the water."
    ) == ("", 2, 0)


def test_trailing_camera_sentence_without_a_final_period_is_removed():
    caption = "A rooftop garden with potted plants lines the terrace. The camera moves forward along the walkway"
    assert strip_camera_sentences(caption) == ("A rooftop garden with potted plants lines the terrace.", 1, 0)


def test_trailing_whitespace_after_a_dropped_final_sentence_goes_with_it():
    caption = "A rooftop garden with potted plants lines the terrace. The camera tilts down. "
    assert strip_camera_sentences(caption) == ("A rooftop garden with potted plants lines the terrace.", 1, 0)


# ---------------------------------------------------------------------------
# apply_camera_scrub: the shared suspicious-scrub guardrail both tools use.
# ---------------------------------------------------------------------------


def test_guardrail_keeps_a_caption_that_scrubs_to_empty():
    caption = "The camera pans across the scene, capturing the skyline."
    assert apply_camera_scrub(caption) == (caption, 0, True)


def test_guardrail_keeps_a_caption_that_loses_more_than_half_its_text():
    caption = "A door. The camera moves forward along the hallway, capturing a long row of identical doors."
    assert apply_camera_scrub(caption) == (caption, 0, True)


def test_guardrail_passes_a_healthy_scrub_through():
    caption = f"{RAW_KEPT_CAPTION} {RAW_CAMERA_SENTENCE}"
    assert apply_camera_scrub(caption) == (RAW_KEPT_CAPTION, 1, False)


def test_guardrail_is_inert_when_nothing_is_dropped():
    caption = "A camera store display fills the shop window."
    assert apply_camera_scrub(caption) == (caption, 0, False)


# ---------------------------------------------------------------------------
# scrub_dl3dv_captions.py: staging a scrubbed copy of a raw caption corpus.
# ---------------------------------------------------------------------------


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raw_caption_file(path, captions):
    """Mimic a generate_captions.py raw file: per-window datasets carrying an
    embedding-like payload plus caption/image_indices/prompt_format attrs, and
    a root attr, so the bit-exact copy contract has content to preserve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["fixture_root_attr"] = "raw caption corpus fixture"
        for index, (dataset_name, caption) in enumerate(captions.items()):
            data = np.arange(12, dtype=np.float16).reshape(3, 4) + index
            dataset = f.create_dataset(dataset_name, data=data)
            dataset.attrs["prompt_format"] = "wan_t5"
            dataset.attrs["caption"] = caption
            dataset.attrs["image_indices"] = np.arange(index, index + 3, dtype=np.int64)
    return path


def test_unchanged_file_is_staged_byte_identical(tmp_path):
    input_path = _write_raw_caption_file(
        tmp_path / "in" / "1K" / "scene_a" / "frames_3_stride_1.h5",
        {"frame_00000": CLEAN_CAPTION},
    )
    output_path = tmp_path / "out" / "1K" / "scene_a" / "frames_3_stride_1.h5"

    stats = scrub_caption_file(input_path, output_path, replace=False)

    assert stats == ScrubStats()
    assert output_path.read_bytes() == input_path.read_bytes()


def test_changed_file_scrubs_only_the_caption_and_copies_everything_else(tmp_path):
    captions = {
        "frame_00000": f"{RAW_KEPT_CAPTION} {RAW_CAMERA_SENTENCE}",
        "frame_00001": CLEAN_CAPTION,
    }
    input_path = _write_raw_caption_file(tmp_path / "in" / "1K" / "scene_a" / "frames_3_stride_1.h5", captions)
    input_digest_before = _sha256(input_path)
    output_path = tmp_path / "out" / "1K" / "scene_a" / "frames_3_stride_1.h5"

    stats = scrub_caption_file(input_path, output_path, replace=False)

    assert stats.captions_changed == 1
    assert stats.sentences_dropped == 1
    assert stats.skipped_suspicious == 0
    # The input corpus is never modified.
    assert _sha256(input_path) == input_digest_before
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "r") as dst:
        assert dst.attrs["fixture_root_attr"] == src.attrs["fixture_root_attr"]
        assert sorted(dst) == sorted(src)
        for name in src:
            # Dataset payloads, dtypes, and non-caption attrs are copied exactly.
            assert dst[name].dtype == src[name].dtype
            np.testing.assert_array_equal(dst[name][()], src[name][()])
            assert dst[name].attrs["prompt_format"] == src[name].attrs["prompt_format"]
            np.testing.assert_array_equal(dst[name].attrs["image_indices"], src[name].attrs["image_indices"])
        assert dst["frame_00000"].attrs["caption"] == RAW_KEPT_CAPTION
        assert dst["frame_00001"].attrs["caption"] == CLEAN_CAPTION
    # No .tmp residue after the atomic rename.
    assert list(output_path.parent.glob("*.tmp")) == []


def test_raw_tool_guardrail_keeps_suspicious_captions(tmp_path):
    input_path = _write_raw_caption_file(
        tmp_path / "in" / "1K" / "scene_a" / "frames_3_stride_1.h5",
        {"frame_00000": "The camera pans across the scene, capturing the skyline."},
    )
    output_path = tmp_path / "out" / "1K" / "scene_a" / "frames_3_stride_1.h5"

    stats = scrub_caption_file(input_path, output_path, replace=False)

    assert stats.captions_changed == 0
    assert stats.sentences_dropped == 0
    assert stats.skipped_suspicious == 1
    # The guarded caption is kept, so the staged file is a byte-identical copy.
    assert output_path.read_bytes() == input_path.read_bytes()


def test_existing_output_is_skipped_without_replace(tmp_path):
    input_path = _write_raw_caption_file(
        tmp_path / "in" / "1K" / "scene_a" / "frames_3_stride_1.h5",
        {"frame_00000": f"{RAW_KEPT_CAPTION} {RAW_CAMERA_SENTENCE}"},
    )
    output_path = tmp_path / "out" / "1K" / "scene_a" / "frames_3_stride_1.h5"

    assert scrub_caption_file(input_path, output_path, replace=False) is not None
    staged_digest = _sha256(output_path)
    # Second run: the output exists and --replace is off -> skipped untouched.
    assert scrub_caption_file(input_path, output_path, replace=False) is None
    assert _sha256(output_path) == staged_digest
    # --replace regenerates it.
    assert scrub_caption_file(input_path, output_path, replace=True) is not None


def test_scrubbing_a_scrubbed_file_is_a_byte_identical_no_op(tmp_path):
    """Idempotence: a staged scrubbed file has nothing left to drop, so a
    second pass stages a byte-identical copy and reports zero drops."""
    input_path = _write_raw_caption_file(
        tmp_path / "in" / "1K" / "scene_a" / "frames_3_stride_1.h5",
        {"frame_00000": f"{RAW_KEPT_CAPTION} {RAW_CAMERA_SENTENCE}"},
    )
    first_path = tmp_path / "out1" / "1K" / "scene_a" / "frames_3_stride_1.h5"
    second_path = tmp_path / "out2" / "1K" / "scene_a" / "frames_3_stride_1.h5"

    assert scrub_caption_file(input_path, first_path, replace=False).captions_changed == 1
    stats = scrub_caption_file(first_path, second_path, replace=False)

    assert stats == ScrubStats()
    assert second_path.read_bytes() == first_path.read_bytes()


def test_scrub_tree_shards_the_walk_and_prints_the_run_stats(tmp_path, capsys):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _write_raw_caption_file(
        input_dir / "1K" / "scene_a" / "frames_3_stride_1.h5",
        {"frame_00000": f"{RAW_KEPT_CAPTION} {RAW_CAMERA_SENTENCE}"},
    )
    _write_raw_caption_file(
        input_dir / "1K" / "scene_b" / "frames_3_stride_1.h5",
        {"frame_00000": CLEAN_CAPTION},
    )
    _write_raw_caption_file(
        input_dir / "2K" / "scene_c" / "frames_3_stride_1.h5",
        {"frame_00000": f"{RAW_KEPT_CAPTION} {RAW_CAMERA_SENTENCE}"},
    )

    scrub_tree(argparse.Namespace(input_dir=input_dir, output_dir=output_dir, shard="1K", replace=False))

    summary = capsys.readouterr().out
    assert "shard=1K" in summary
    assert "files_scanned=2" in summary
    assert "files_changed=1" in summary
    assert "sentences_dropped=1" in summary
    assert "skipped_suspicious=0" in summary
    # Only the shard was staged, at the identical <shard>/<scene>/<file> layout.
    assert (output_dir / "1K" / "scene_a" / "frames_3_stride_1.h5").exists()
    assert (output_dir / "1K" / "scene_b" / "frames_3_stride_1.h5").exists()
    assert not (output_dir / "2K").exists()

    # A follow-up whole-tree run picks up the remaining shard and skips the
    # already-staged files without --replace.
    scrub_tree(argparse.Namespace(input_dir=input_dir, output_dir=output_dir, shard=None, replace=False))
    summary = capsys.readouterr().out
    assert "shard=all" in summary
    assert "files_scanned=1" in summary
    assert (output_dir / "2K" / "scene_c" / "frames_3_stride_1.h5").exists()


def test_scrub_tree_refuses_in_place_operation(tmp_path):
    input_dir = tmp_path / "in"
    _write_raw_caption_file(input_dir / "1K" / "scene_a" / "frames_3_stride_1.h5", {"frame_00000": CLEAN_CAPTION})
    with pytest.raises(AssertionError, match="never modified in place"):
        scrub_tree(argparse.Namespace(input_dir=input_dir, output_dir=input_dir, shard=None, replace=False))


def test_scrub_tree_refuses_output_nested_in_input_dir(tmp_path):
    """Staging inside the input corpus would make every follow-up run re-scrub
    its own output and recursively duplicate the tree."""
    input_dir = tmp_path / "in"
    _write_raw_caption_file(input_dir / "1K" / "scene_a" / "frames_3_stride_1.h5", {"frame_00000": CLEAN_CAPTION})
    with pytest.raises(AssertionError, match="must not nest"):
        scrub_tree(
            argparse.Namespace(input_dir=input_dir, output_dir=input_dir / "scrubbed", shard=None, replace=False)
        )
