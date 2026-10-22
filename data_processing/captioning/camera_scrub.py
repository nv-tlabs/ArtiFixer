# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera-sentence scrub for DL3DV captions.

Roughly 10% of the raw DL3DV captions carry a dedicated sentence narrating
camera movement ("The camera slowly pans across the scene, capturing...",
"The camera moves forward along the bridge, capturing..."), even though the
captioning instruction (generate_captions.py IMAGE_PROMPT/VIDEO_PROMPT
requirement 2) says to never describe camera movement — these slipped
through. The training recipe requires camera-motion-free caption text, so the scrub
runs at three points:

- ``generate_captions.py`` applies it to every model-generated caption, so
  new corpora are clean at the source;
- ``retokenize_cosmos3_prompts.py`` scrubs at tokenization time while
  regenerating a prompt-token corpus (the output ``caption`` attr stores
  the scrubbed text, so attr and tokens stay consistent);
- ``scrub_dl3dv_captions.py`` stages a scrubbed copy of an existing raw
  caption corpus.

Every verified offending sample is a standalone sentence whose subject is
"The camera". Only sentences in that shape are dropped; sentences merely
containing the word "camera" elsewhere (a caption can describe a camera IN
the scene) are never touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Subject-position "the camera" only: the sentence must START with the words
# "the camera" (initial capital optional, leading whitespace allowed). The
# second word is matched lowercase exactly, so proper-noun subjects such as
# "The Camera Obscura building ..." survive; the trailing word boundary keeps
# different subjects such as "The cameraman ..." alive.
_CAMERA_SENTENCE_PREFIX = re.compile(r"\s*[Tt]he camera\b")

# Plain-word tokens whose trailing period is an abbreviation dot, not a
# sentence end ("St. Mary", "Mr. Alvarez"). Single-letter initials
# ("J. Smith") and dotted tokens ("U.S.", "e.g.") are recognized structurally
# in _is_sentence_boundary. Admission rule: a token is listed only if it is
# BOTH plausible mid-sentence in a scene caption AND implausible as a
# sentence-final word — every admitted token silently suppresses a real
# boundary whenever a sentence happens to end with it.
_ABBREVIATION_TOKENS = {"dr", "mr", "mrs", "ms", "mt", "st"}


@dataclass
class ScrubStats:
    """Per-file scrub counters (all zero when the scrub changed nothing)."""

    captions_changed: int = 0
    sentences_dropped: int = 0
    skipped_suspicious: int = 0


def _is_sentence_boundary(caption: str, index: int) -> bool:
    """True when the terminator at ``index`` ends a sentence.

    A terminator ('.', '!', '?') followed by whitespace or the end of the
    string ends a sentence — except a period that is the dot of an
    abbreviation: a single-letter initial ("J. Smith"), a token that already
    contains a dot ("U.S.", "e.g.", an ellipsis), or a known plain-word
    abbreviation ("St. Mary"). Misreading an abbreviation dot as a boundary
    would cut a sentence in two and let the tail of a dropped camera sentence
    survive as a garbled fragment. Merging in the other direction is handled
    by strip_camera_sentences: a DROPPED span that contains a suppressed
    boundary is kept and reported instead of silently deleting merged text.
    """
    if index + 1 < len(caption) and not caption[index + 1].isspace():
        return False
    if caption[index] in "!?":
        return True
    token_start = index
    while token_start > 0 and not caption[token_start - 1].isspace():
        token_start -= 1
    token = caption[token_start:index].lstrip("\"'([{")
    if len(token) == 1 and token.isalpha():
        return False
    if "." in token:
        return False
    return token.lower() not in _ABBREVIATION_TOKENS


def _sentence_spans(caption: str) -> list[tuple[int, int]]:
    """Split a single-paragraph prose caption into conservative sentence spans.

    A sentence ends at a sentence-boundary period (see _is_sentence_boundary);
    any other period — decimals like "3.5", abbreviations like "St.",
    mid-token dots — stays inside its sentence. Each span starts right after
    the previous sentence's period, so a span carries its own leading
    (inter-sentence) whitespace; a final whitespace-only span can occur when
    the caption ends with trailing whitespace.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(caption):
        if char in ".!?" and _is_sentence_boundary(caption, index):
            spans.append((start, index + 1))
            start = index + 1
    if start < len(caption):
        spans.append((start, len(caption)))
    return spans


def _span_contains_suppressed_boundary(caption: str, start: int, end: int) -> bool:
    """True when the span holds an internal terminator+whitespace that the
    abbreviation rule suppressed — i.e. the span MAY be two merged sentences.
    Dropping such a span could silently delete legitimate scene text, so the
    caller keeps it and reports it instead."""
    for index in range(start, end - 1):
        if caption[index] == "." and caption[index + 1].isspace() and not _is_sentence_boundary(caption, index):
            return True
    return False


def _suppressed_boundary_hides_camera_text(caption: str, start: int, end: int) -> bool:
    """True when a suppressed (abbreviation) boundary inside the span is
    immediately followed by text matching the camera-sentence prefix — the
    span may be a kept scene sentence with a merged camera sentence hiding
    behind it ("... near Main St. The camera pans ..."). Keeping such a span
    silently would leak the camera narration, so the caller reports it as
    suspicious instead."""
    for index in range(start, end - 1):
        if (
            caption[index] == "."
            and caption[index + 1].isspace()
            and not _is_sentence_boundary(caption, index)
            and _CAMERA_SENTENCE_PREFIX.match(caption[index + 1 : end])
        ):
            return True
    return False


def strip_camera_sentences(caption: str) -> tuple[str, int, int]:
    """Drop every sentence that starts with "The camera"; return
    ``(scrubbed, dropped, kept_ambiguous)``.

    Kept text is preserved byte-for-byte — including any weird spacing inside
    and between kept sentences; only the dropped sentences and their boundary
    whitespace go, with a single space joining the kept neighbors of a removal
    point. A caption with nothing to drop is returned unchanged. A camera
    span that contains an abbreviation-suppressed internal boundary MAY be
    two merged sentences, so it is KEPT and counted in ``kept_ambiguous``
    rather than risking silent deletion of merged scene text.
    """
    kept_parts: list[str] = []
    dropped = 0
    kept_ambiguous = 0
    previous_dropped = False
    for start, end in _sentence_spans(caption):
        raw = caption[start:end]
        if not raw.strip():
            # Whitespace-only span: trailing whitespace after the caption's
            # final period. It is boundary whitespace of the sentence before
            # it, so it goes if that sentence was dropped.
            if not previous_dropped:
                kept_parts.append(raw)
            continue
        if _CAMERA_SENTENCE_PREFIX.match(raw):
            if _span_contains_suppressed_boundary(caption, start, end):
                kept_ambiguous += 1
            else:
                dropped += 1
                previous_dropped = True
                continue
        elif _suppressed_boundary_hides_camera_text(caption, start, end):
            # A kept scene sentence with camera narration merged in behind a
            # suppressed abbreviation boundary: keep it, but never silently.
            kept_ambiguous += 1
        if previous_dropped:
            # The whitespace at the head of this span bounded the dropped
            # sentence; it goes with it. Kept neighbors rejoin with ONE space.
            raw = raw.lstrip()
            if kept_parts:
                kept_parts.append(" ")
        kept_parts.append(raw)
        previous_dropped = False
    if dropped == 0:
        return caption, 0, kept_ambiguous
    return "".join(kept_parts), dropped, kept_ambiguous


def apply_camera_scrub(caption: str) -> tuple[str, int, bool]:
    """strip_camera_sentences plus the suspicious-scrub guardrail.

    Returns ``(caption_to_store, sentences_dropped, skipped_suspicious)``.
    Guardrail: when the scrub empties the caption or removes more than half
    of its text, the ORIGINAL caption is returned with sentences_dropped=0
    and skipped_suspicious=True — the run reports it and keeps going, it
    never fails. Camera spans holding an abbreviation-suppressed internal
    boundary (possible merged sentences) also count as suspicious: they are
    kept, never guessed at.
    """
    scrubbed, dropped, kept_ambiguous = strip_camera_sentences(caption)
    if dropped == 0:
        return caption, 0, kept_ambiguous > 0
    if not scrubbed.strip() or len(scrubbed) < len(caption) / 2:
        return caption, 0, True
    return scrubbed, dropped, kept_ambiguous > 0


def walk_corpus_tree(input_files, process_file, *, shard_label: str, tool_label: str) -> None:
    """Shared walk/accumulate/summarize loop for the corpus tree tools.

    ``process_file(path)`` returns per-file ``ScrubStats`` or ``None`` for a
    skipped (already-staged) file; the summary separates scanned from skipped
    so an all-skipped rerun cannot read as a clean corpus.
    """
    files_scanned = 0
    files_changed = 0
    totals = ScrubStats()
    for input_path in input_files:
        stats = process_file(input_path)
        if stats is None:
            continue
        files_scanned += 1
        if stats.captions_changed:
            files_changed += 1
        totals.sentences_dropped += stats.sentences_dropped
        totals.skipped_suspicious += stats.skipped_suspicious
    print(
        f"{tool_label} summary: shard={shard_label} files_scanned={files_scanned} "
        f"files_skipped={len(input_files) - files_scanned} files_changed={files_changed} "
        f"sentences_dropped={totals.sentences_dropped} skipped_suspicious={totals.skipped_suspicious}"
    )
