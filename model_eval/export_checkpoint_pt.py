# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import subprocess
from argparse import Namespace
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from model_training.checkpoint_metadata import checkpoint_metadata_path, resolve_camera_pose_alignment
from model_training.model_profiles import CAMERA_POSE_ALIGNMENTS

DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
EXPORT_TRANSFORMER_DEFAULTS = {
    "frames_per_block": 7,
    "local_attn_size": 21,
    "sink_size": 7,
    "gradient_checkpointing": True,
    "checkpoint_every_n_blocks": 1,
    "attention_backend": None,
}


def resolve_export_pose_alignment(args: Namespace) -> None:
    args.camera_pose_alignment = resolve_camera_pose_alignment(
        args.checkpoint_dir, args.model_id, camera_pose_alignment=args.camera_pose_alignment
    )


def _git_value(repo_dir: Path, *args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=repo_dir, check=False, capture_output=True, text=True)
    return result.stdout.strip() or None


def _ema_export_metadata(args: Namespace) -> dict[str, str | int | float]:
    from model_training.utils.ema import ema_export_metadata

    return ema_export_metadata(args.checkpoint_dir, args.checkpoint, args.slot)


def build_metadata(
    args: Namespace,
    *,
    branch: str | None,
    commit: str | None,
    export_date: datetime,
    weight_metadata: dict[str, str | int | float] | None = None,
) -> dict[str, str | int | float]:
    source_path = args.source_path if args.source_path is not None else args.checkpoint_dir
    return {
        "run_id": args.run_id,
        "checkpoint": int(args.checkpoint),
        "slot": args.slot,
        "source_path": str(source_path),
        "output_pt": str(args.output_pt),
        "model_id": args.model_id,
        "camera_pose_alignment": args.camera_pose_alignment,
        "branch": branch or "",
        "commit": commit or "",
        "export_date": export_date.isoformat(),
    } | (_ema_export_metadata(args) if weight_metadata is None else weight_metadata)


def _state_dict_to_cpu(state_dict: dict) -> dict:
    import torch

    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in state_dict.items()}


def _export_transformer_args(args: Namespace) -> Namespace:
    return Namespace(**(vars(args) | EXPORT_TRANSFORMER_DEFAULTS))


def create_export_transformer(args: Namespace):
    from model_training.utils.train_utils import get_kv_cache_pipe

    return get_kv_cache_pipe(_export_transformer_args(args), device="cpu").transformer


def export_checkpoint(args: Namespace) -> tuple[Path, Path]:
    weight_metadata = _ema_export_metadata(args)
    metadata_path = checkpoint_metadata_path(args.output_pt)
    marker = args.output_pt.with_name(args.output_pt.name + ".incomplete")
    if (args.output_pt.exists() or marker.exists()) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_pt}")

    import torch

    from model_training.utils.train_utils import load_model_weights_from_dcp
    from model_training.utils.video_io import atomic_output_path

    transformer = create_export_transformer(args)
    load_model_weights_from_dcp(transformer, args.checkpoint_dir)

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(__file__).resolve().parents[1]
    metadata = build_metadata(
        args,
        branch=args.branch or _git_value(repo_dir, "branch", "--show-current"),
        commit=args.commit or _git_value(repo_dir, "rev-parse", "HEAD"),
        export_date=datetime.now(timezone.utc),
        weight_metadata=weight_metadata,
    )
    with (
        atomic_output_path(args.output_pt) as temporary_weights,
        atomic_output_path(metadata_path) as temporary_metadata,
    ):
        torch.save(_state_dict_to_cpu(transformer.state_dict()), temporary_weights)
        temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        # Both files are ready. Refuse loading if publication stops between renames.
        marker.touch()
    marker.unlink()
    return args.output_pt, metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a DCP/FSDP transformer checkpoint to a single .pt state dict.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument(
        "--output_pt", type=Path, required=True, help="Output state dict; its .metadata.json is written beside it."
    )
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, type=str)
    parser.add_argument(
        "--camera_pose_alignment", default=None, choices=CAMERA_POSE_ALIGNMENTS, help="Must match the training record."
    )
    parser.add_argument("--run_id", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=int)
    parser.add_argument("--slot", required=True, type=str)
    parser.add_argument("--source_path", type=Path, default=None)
    parser.add_argument("--branch", default=None, type=str)
    parser.add_argument("--commit", default=None, type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    args = build_parser().parse_args(argv)
    resolve_export_pose_alignment(args)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    output_pt, metadata_path = export_checkpoint(parse_args(argv))
    print(f"Wrote {output_pt}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
