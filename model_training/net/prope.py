# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

# How to use PRoPE attention for cross-attention:
#
#    attn_src = PropeDotProductAttention(...)
#    attn_tgt = PropeDotProductAttention(...)
#    src_fns = attn_src.precompute_apply_fns(viewmats_src, Ks_src)
#    tgt_fns = attn_tgt.precompute_apply_fns(viewmats_tgt, Ks_tgt)
#    q_src = src_fns.apply_to_q(q_src)
#    k_tgt = tgt_fns.apply_to_kv(k_tgt)
#    v_tgt = tgt_fns.apply_to_kv(v_tgt)
#    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs)
#    o_src = src_fns.apply_to_o(o_src)

from functools import partial
from typing import Callable, List, NamedTuple, Tuple

import torch
from torch import nn

from model_training.utils.pose_utils import invert_SE3


class PropeDotProductAttention(torch.nn.Module):
    """PRoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int = 0,
        patches_y: int = 0,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.freq_base = freq_base
        self.freq_scale = freq_scale

    def update_coeffs(self, patches_x: int, patches_y: int, device: str | torch.device):
        self.patches_x = patches_x
        self.patches_y = patches_y
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y,)),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.repeat_interleave(torch.arange(patches_y, device=device), patches_x),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        # Do not save coeffs to checkpoint as `cameras` might change during testing.
        self.coeffs_x_0 = nn.Buffer(coeffs_x[0], persistent=False)
        self.coeffs_x_1 = nn.Buffer(coeffs_x[1], persistent=False)
        self.coeffs_y_0 = nn.Buffer(coeffs_y[0], persistent=False)
        self.coeffs_y_1 = nn.Buffer(coeffs_y[1], persistent=False)

    def precompute_apply_fns(self, viewmats: torch.Tensor, Ks_norm: torch.Tensor) -> "PropeApplyFns":
        """Build the PRoPE transforms for one set of cameras.

        The transforms are returned rather than stored on this (shared) module.
        Gradient checkpointing re-runs each transformer block at backward time
        with the same non-tensor args it was given, so transforms stored on the
        module could be overwritten by a later forward (e.g. the next chunk of
        an autoregressive rollout) before backward runs.
        """
        batch, cameras, _, _ = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks_norm.shape == (batch, cameras, 3, 3)

        return PropeApplyFns(
            *_prepare_apply_fns(
                head_dim=self.head_dim,
                viewmats=viewmats,
                Ks_norm=Ks_norm,
                coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
                coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
            )
        )


class PropeApplyFns(NamedTuple):
    """The PRoPE Q/KV/O transforms for one set of cameras."""

    apply_to_q: Callable[[torch.Tensor], torch.Tensor]
    apply_to_kv: Callable[[torch.Tensor], torch.Tensor]
    apply_to_o: Callable[[torch.Tensor], torch.Tensor]


def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks_norm: torch.Tensor,  # (batch, cameras, 3, 3)
    coeffs_x: Tuple[torch.Tensor, torch.Tensor],
    coeffs_y: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    batch, cameras, _, _ = viewmats.shape

    # Compute the camera projection matrices we use in PRoPE.
    # - K is an `image<-camera` transform.
    # - viewmats is a `camera<-world` transform.
    # - P = lift(K) @ viewmats is an `image<-world` transform.
    P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
    P_T = P.transpose(-1, -2)
    P_inv = torch.einsum(
        "...ij,...jk->...ik",
        invert_SE3(viewmats),
        _lift_K(_invert_K(Ks_norm)),
    )

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    batch, num_heads, seqlen, feat_dim = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base ** (-torch.arange(num_freqs, device=positions.device)[None, None, None, :] / num_freqs)
    )
    angles = positions[None, None, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.empty_like(feats)
    start = 0
    for func, x_block, block_size in zip(funcs, x_blocks, block_sizes):
        end = start + block_size
        out[..., start:end] = func(x_block)
        start = end
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out
