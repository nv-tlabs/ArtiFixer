# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from argparse import Namespace
from pathlib import Path


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint_pt",
        type=Path,
        default=None,
        help="Single-file transformer state dict exported from a DCP/FSDP checkpoint.",
    )


def validate_checkpoint_args(parser: argparse.ArgumentParser, args: Namespace) -> None:
    has_checkpoint_dir = args.checkpoint_dir is not None
    has_checkpoint_pt = args.checkpoint_pt is not None
    if has_checkpoint_dir == has_checkpoint_pt:
        parser.error(
            "exactly one of --checkpoint_dir or --checkpoint_pt is required")


def checkpoint_output_name(args: Namespace) -> str:
    if args.checkpoint_pt is not None:
        return args.checkpoint_pt.stem

    ckpt_parent = args.checkpoint_dir.parent.name
    ckpt_name = args.checkpoint_dir.name
    return f"{ckpt_parent}_{ckpt_name}" if ckpt_parent.startswith("checkpoint") else ckpt_name


def _torch_load_state_dict(checkpoint_pt: Path):
    import torch

    # Keep the checkpoint state dict off CUDA; the model is already on GPU.
    try:
        return torch.load(checkpoint_pt, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(checkpoint_pt, map_location="cpu")


def load_model_weights_from_pt(model, checkpoint_pt: Path | str) -> None:
    import gc

    import torch

    state_dict = _torch_load_state_dict(Path(checkpoint_pt))
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state dict in {checkpoint_pt}, got {type(state_dict).__name__}")

    model_state = model.state_dict()
    missing_keys = []
    shape_mismatches = []

    with torch.no_grad():
        for name, target_tensor in model_state.items():
            source_tensor = state_dict.pop(name, None)
            if source_tensor is None:
                missing_keys.append(name)
                continue

            if source_tensor.shape != target_tensor.shape:
                shape_mismatches.append(
                    f"{name}: checkpoint {tuple(source_tensor.shape)} != model {tuple(target_tensor.shape)}"
                )
                del source_tensor
                continue

            target_tensor.copy_(source_tensor, non_blocking=False)
            del source_tensor

    unexpected_keys = list(state_dict.keys())
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if missing_keys or unexpected_keys or shape_mismatches:
        messages = []
        if missing_keys:
            messages.append(f"Missing keys: {missing_keys[:20]}")
        if unexpected_keys:
            messages.append(f"Unexpected keys: {unexpected_keys[:20]}")
        if shape_mismatches:
            messages.append(f"Shape mismatches: {shape_mismatches[:20]}")
        raise RuntimeError("Error loading checkpoint:\n" + "\n".join(messages))


def load_transformer_checkpoint(
    model,
    args: Namespace,
    *,
    dcp_loader=None,
    pt_loader=None,
) -> None:
    if args.checkpoint_pt is not None:
        loader = pt_loader if pt_loader is not None else load_model_weights_from_pt
        loader(model, args.checkpoint_pt)
        return

    if dcp_loader is None:
        from model_training.utils.train_utils import load_model_weights_from_dcp

        dcp_loader = load_model_weights_from_dcp
    dcp_loader(model, args.checkpoint_dir)
