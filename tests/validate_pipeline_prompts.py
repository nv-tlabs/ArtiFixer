"""Prompt-file preflight checks for the pipeline smoke test."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def num_frames_from_extra_args(extra_args: str) -> int:
    tokens = shlex.split(extra_args)
    # 81 is the pipeline smoke test's num_frames default when --num_frames is
    # not passed through --extra-args.
    num_frames = 81
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--num_frames":
            if index + 1 >= len(tokens):
                raise ValueError("--num_frames requires a value")
            num_frames = int(tokens[index + 1])
            index += 2
            continue
        if token.startswith("--num_frames="):
            num_frames = int(token.split("=", 1)[1])
        index += 1
    return num_frames


def missing_prompt_files(split_path: Path, prompt_dir: Path, num_frames: int) -> list[str]:
    with split_path.open() as f:
        split_data = json.load(f)

    missing = []
    for split in ("trainval", "test"):
        for scene_id, reconstructions in split_data.get(split, {}).items():
            if not reconstructions:
                missing.append(f"{split}/{scene_id}: no reconstructions in split file")
                continue
            subdir = reconstructions[0]["subdir"]
            # Match the stride-1 file and its _reverse/_revisit variants ONLY
            # (a bare "stride_1*" glob would also accept stride_10 files),
            # plus the short-scene fallback frames_all_stride_1.h5 — the
            # training loader falls back to it exactly like this.
            patterns = (
                f"frames_{num_frames}_stride_1.h5",
                f"frames_{num_frames}_stride_1_*.h5",
                "frames_all_stride_1.h5",
            )
            scene_prompt_dir = prompt_dir / subdir / scene_id
            if not any(True for pattern in patterns for _ in scene_prompt_dir.glob(pattern)):
                missing.append(f"{subdir}/{scene_id}/{patterns[0]}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", required=True, type=Path)
    parser.add_argument("--prompt-dir", required=True, type=Path)
    parser.add_argument("--extra-args", default="")
    args = parser.parse_args()

    num_frames = num_frames_from_extra_args(args.extra_args)

    missing = missing_prompt_files(args.split_path, args.prompt_dir, num_frames)
    if missing:
        print(f"ERROR: prompt preflight failed for num_frames={num_frames}")
        for item in missing:
            print(f"  missing: {item}")
        return 1

    print(f"Prompt preflight passed for num_frames={num_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
