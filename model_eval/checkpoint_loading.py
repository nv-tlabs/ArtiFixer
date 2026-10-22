# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a DCP/FSDP checkpoint directory or a single-file .pt transformer state dict.",
    )


def _is_pt_checkpoint(checkpoint: Path) -> bool:
    return checkpoint.suffix.lower() == ".pt"


def _torch_load_state_dict(checkpoint_pt: Path):
    import torch

    from model_training.checkpoint_metadata import require_finished_export

    require_finished_export(checkpoint_pt)
    return torch.load(checkpoint_pt, map_location="cpu", weights_only=True, mmap=True)


@contextmanager
def eval_pipe_build_context(args: Namespace):
    """Build .pt transformers on meta to avoid allocating a full CPU copy.

    The VAE loads its own pretrained weights. Buffers remain on CPU.
    """
    if not _is_pt_checkpoint(Path(args.checkpoint)):
        yield False
        return

    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        yield True


def load_model_weights_from_pt(model, checkpoint_pt: Path | str, *, device=None, dtype=None) -> None:
    """Validate keys and shapes, then materialize each parameter on the requested device."""
    from accelerate.utils import set_module_tensor_to_device

    checkpoint_pt = Path(checkpoint_pt)
    state_dict = _torch_load_state_dict(checkpoint_pt)
    expected = model.state_dict()
    missing = sorted(expected.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected.keys())
    if missing or unexpected:
        raise RuntimeError(
            f"{checkpoint_pt} does not match the model state dict: "
            f"{len(missing)} missing key(s) {missing[:3]}, "
            f"{len(unexpected)} unexpected key(s) {unexpected[:3]}"
        )
    mismatched = [name for name, tensor in state_dict.items() if tensor.shape != expected[name].shape]
    if mismatched:
        name = mismatched[0]
        raise RuntimeError(
            f"{checkpoint_pt} does not match the model state dict: {len(mismatched)} shape mismatch(es), "
            f"e.g. {name}: checkpoint {tuple(state_dict[name].shape)} vs model {tuple(expected[name].shape)}"
        )
    for name, tensor in state_dict.items():
        set_module_tensor_to_device(model, name, device or "cpu", value=tensor, dtype=dtype)


def load_transformer_checkpoint(model, args: Namespace, *, device=None, dtype=None) -> None:
    """Load .pt weights onto the requested device, or DCP weights into the existing model."""
    checkpoint = Path(args.checkpoint)
    if _is_pt_checkpoint(checkpoint):
        load_model_weights_from_pt(model, checkpoint, device=device, dtype=dtype)
        return

    from model_training.utils.train_utils import load_model_weights_from_dcp

    load_model_weights_from_dcp(model, checkpoint)
