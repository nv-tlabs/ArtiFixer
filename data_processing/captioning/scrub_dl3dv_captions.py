# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a camera-motion-free copy of a raw DL3DV caption corpus.

Walks an input raw-caption tree (<shard>/<scene>/<window>.h5 — the same
layout retokenize_cosmos3_prompts.py reads), passes every dataset's
``caption`` attr through the camera-sentence scrub (camera_scrub.py), and
writes a STAGED output tree with identical structure. Everything except the
scrubbed caption attrs is copied bit-exact — datasets, dtypes, every other
attr — and a file with nothing to scrub is staged as a byte-identical copy
of the input. The input tree is never modified.

Per-shard parallelizable: pass ``--shard 1K`` .. ``--shard 11K`` so cluster
jobs can split the corpus; each job walks only <input_dir>/<shard> and the
output keeps the same <shard>/ prefix.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py

from data_processing.captioning.camera_scrub import ScrubStats, apply_camera_scrub, walk_corpus_tree


def scrub_caption_file(input_path: Path, output_path: Path, *, replace: bool) -> ScrubStats | None:
    """Stage one raw caption HDF5 with camera sentences scrubbed.

    Returns the per-file counters, or ``None`` when the output already exists
    and ``replace`` is off (the file is skipped, not scanned).
    """
    if output_path.exists() and not replace:
        return None

    stats = ScrubStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    scrubbed_captions: dict[str, str] = {}
    with h5py.File(input_path, "r") as src:
        dataset_names = list(src.keys())
        assert dataset_names, f"No caption datasets found in {input_path}"
        for dataset_name in dataset_names:
            caption = src[dataset_name].attrs["caption"]
            new_caption, dropped, suspicious = apply_camera_scrub(caption)
            stats.sentences_dropped += dropped
            if dropped:
                stats.captions_changed += 1
                scrubbed_captions[dataset_name] = new_caption
            if suspicious:
                stats.skipped_suspicious += 1
        if scrubbed_captions:
            with h5py.File(tmp_path, "w") as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                for dataset_name in dataset_names:
                    # h5py Group.copy (H5Ocopy) carries the data, dtype,
                    # chunking/filters, and every attr of untouched content
                    # bit-exact; only the scrubbed caption attrs are rewritten.
                    src.copy(src[dataset_name], dst, name=dataset_name)
                    if dataset_name in scrubbed_captions:
                        dst[dataset_name].attrs["caption"] = scrubbed_captions[dataset_name]
    if not scrubbed_captions:
        # Nothing to scrub: stage a byte-identical copy of the input file.
        shutil.copyfile(input_path, tmp_path)
    tmp_path.rename(output_path)
    return stats


def scrub_tree(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    assert output_dir != input_dir, "output_dir must differ from input_dir: the input corpus is never modified in place"
    assert input_dir not in output_dir.parents, (
        "output_dir and input_dir must not nest: staging inside the input corpus would make "
        "every follow-up run re-scrub its own output"
    )
    input_root = args.input_dir / args.shard if args.shard else args.input_dir
    input_files = sorted(input_root.rglob("*.h5"))
    assert input_files, f"No caption HDF5 files found under {input_root}"

    def process_file(input_path):
        # Relative to input_dir (not the shard root) so the staged tree keeps
        # the identical <shard>/<scene>/<window>.h5 layout.
        rel_path = input_path.relative_to(args.input_dir)
        return scrub_caption_file(input_path, args.output_dir / rel_path, replace=args.replace)

    walk_corpus_tree(input_files, process_file, shard_label=args.shard or "all", tool_label="caption scrub")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Only process <input_dir>/<shard> (e.g. 1K .. 11K) so cluster jobs can split "
        "the corpus; the output keeps the same <shard>/ prefix.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    scrub_tree(parse_args())
