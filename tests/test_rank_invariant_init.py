# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""New weights must start with identical values on every worker.

Training splits each worker's weights without first copying one worker's values
onto the others. Different starting values would therefore make the model depend
on how many workers are used. Compare the values directly, not just their sizes.
"""

import copy
import random

import numpy as np
import pytest
import torch
from diffusers import WanTransformer3DModel
from torch import nn

from model_training.net.cosmos3_transformer import ArtifixerCosmos3Transformer
from model_training.net.rng import rank_invariant_init
from model_training.net.scope import ScopeQKInjection
from model_training.net.wan_transformer import ArtifixerTransformer
from tests._helpers import _TinyCosmosBaseTransformer

# Small Wan model with the same input channels and patch shape as production.
_REDUCED_WAN_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=16,
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=32,
    ffn_dim=48,
    num_layers=2,
    rope_max_seq_len=256,
)


class _Wrapped(nn.Module):
    @rank_invariant_init
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(7, 64)


def _seed_as_rank(rank: int, base: int = 42) -> None:
    """Give each simulated worker its own random seed."""
    random.seed(base + rank)
    np.random.seed(base + rank)
    torch.manual_seed(base + rank)


def _wrapper_parameters(base_transformer: nn.Module, rank: int) -> dict[str, torch.Tensor]:
    """Build the added layers for one simulated worker, keeping the base model fixed."""
    _seed_as_rank(rank)
    wrapper = ArtifixerCosmos3Transformer(
        copy.deepcopy(base_transformer),
        vae_scale_factor_spatial=2,
        vae_scale_factor_temporal=4,
        gradient_checkpointing=False,
    )
    state = wrapper.state_dict()
    return {name: tensor.clone() for name, tensor in state.items() if not name.startswith("base_transformer.")}


def _wan_wrapper_state(
    base_transformer: WanTransformer3DModel, rank: int, *, decorated: bool
) -> dict[str, torch.Tensor]:
    """Build Wan from a fixed base model, optionally disabling the shared initialization seed."""
    _seed_as_rank(rank)
    kwargs = dict(
        frames_per_block=None,
        use_kv_cache_attention=False,
        vae_scale_factor_spatial=8,
        vae_scale_factor_temporal=4,
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
    )
    if decorated:
        wrapper = ArtifixerTransformer(copy.deepcopy(base_transformer), **kwargs)
    else:
        wrapper = ArtifixerTransformer.__new__(ArtifixerTransformer)
        ArtifixerTransformer.__init__.__wrapped__(wrapper, copy.deepcopy(base_transformer), **kwargs)
    return {name: tensor.clone() for name, tensor in wrapper.state_dict().items()}


@pytest.mark.parametrize("ranks", [(0, 1), (1, 7), (3, 11)])
def test_decorated_init_agrees_across_ranks(ranks):
    built = []
    for rank in ranks:
        _seed_as_rank(rank)
        built.append(_Wrapped().projection.weight)
    assert torch.equal(*built)


def test_undecorated_init_disagrees_across_ranks():
    """Without the shared seed, different workers must start with different random weights."""
    built = []
    for rank in (1, 7):
        _seed_as_rank(rank)
        built.append(nn.Linear(7, 64).weight)
    assert not torch.equal(*built)


def test_cosmos3_wrapper_builds_the_same_fresh_parameters_on_every_rank():
    """Every worker must build identical added weights, including SCoPE's random strength layers."""
    torch.manual_seed(0)
    base_transformer = _TinyCosmosBaseTransformer()

    built = [_wrapper_parameters(base_transformer, rank) for rank in (1, 7)]
    assert built[0].keys() == built[1].keys()
    for name, tensor in built[0].items():
        assert torch.equal(tensor, built[1][name]), name


def test_wan_wrapper_builds_the_same_fresh_parameters_on_every_rank():
    """Wan's new key-projection weights start at random and must also match across workers."""
    torch.manual_seed(0)
    base_transformer = WanTransformer3DModel(**_REDUCED_WAN_CONFIG)

    built = [_wan_wrapper_state(base_transformer, rank, decorated=True) for rank in (1, 7)]
    assert built[0].keys() == built[1].keys()
    assert "blocks.0.attn2.add_k_proj.weight" in built[0]
    for name, tensor in built[0].items():
        assert torch.equal(tensor, built[1][name]), name


def test_undecorated_wan_wrapper_draws_add_k_proj_per_rank():
    """Disabling the shared seed must expose different key-projection weights across workers."""
    torch.manual_seed(0)
    base_transformer = WanTransformer3DModel(**_REDUCED_WAN_CONFIG)

    built = [_wan_wrapper_state(base_transformer, rank, decorated=False) for rank in (1, 7)]
    fresh_and_random = {n for n in built[0] if not torch.equal(built[0][n], built[1][n])}
    assert fresh_and_random == {
        f"blocks.{block}.attn2.add_k_proj.{p}"
        for block in range(_REDUCED_WAN_CONFIG["num_layers"])
        for p in ("weight", "bias")
    }


@pytest.mark.parametrize("rank", [1, 7])
def test_ambient_rng_is_restored(rank):
    """Building the model must not change the worker's later random draws."""
    _seed_as_rank(rank)
    expected = torch.randn(4)
    _seed_as_rank(rank)
    _Wrapped()
    assert torch.equal(torch.randn(4), expected)


def test_the_scope_gate_is_what_the_pin_protects():
    """The random strength-layer weights need a shared seed; the zero projections already match."""
    built = []
    for rank in (1, 7):
        _seed_as_rank(rank)
        built.append(ScopeQKInjection(32, 8, 128))
    assert not torch.equal(built[0].query_gate[0].weight, built[1].query_gate[0].weight)
    assert torch.equal(built[0].query_projection.weight, built[1].query_projection.weight)
