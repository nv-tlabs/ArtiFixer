# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rope-fps single-sourcing tests.

The Cosmos3 MRoPE temporal-scale fps is pinned to 24 (the ctor default every
prior good-generation lineage trained with; sampling the same checkpoint at
both settings showed the 60-spacing prior is visibly degenerate),
deliberately decoupled from the clip's true fps: the h5 ``fps`` attr, the
caption text, and DL3DV_DATASET_FPS stay 60. The wrapper reads the pin from
the model_profiles single source at exactly one line inside
``_prepare_vision_segment`` when building the MRoPE ids; there is no
per-sample fps input to the wrapper.

Modeled on tests/test_flow_shift_single_source.py.
"""

import torch

import model_training.net.cosmos3_transformer as cosmos3_transformer_module
from model_training.model_profiles import COSMOS3_ROPE_FPS
from tests._helpers import LATENT_CHANNELS, _TinyCosmosBaseTransformer
from tests.test_cosmos3_transformer import _forward_kwargs, _tiny_transformer


def _base_call() -> dict:
    base = _TinyCosmosBaseTransformer()
    base.config.enable_fps_modulation = True
    transformer = _tiny_transformer(base_transformer=base)
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 2, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)

    transformer(**kwargs)

    return base.calls[-1]


def _vision_temporal_step(call: dict) -> torch.Tensor:
    vision_t = call["position_ids"][0, call["vision_sequence_indexes"]].reshape(2, -1)[:, 0]
    return vision_t[1] - vision_t[0]


def test_wrapper_rope_consumes_the_profile_constant(monkeypatch):
    """COSMOS3_ROPE_FPS is the wrapper's ONLY fps source: retargeting the
    constant moves the vision temporal spacing (base_fps / rope_fps), proving
    forward reads the profile pin rather than a stale copy. This is the single reopen point for any future fps
    sweep. There is no per-sample fps input to the wrapper at all."""
    torch.testing.assert_close(
        _vision_temporal_step(_base_call()),
        torch.tensor(24.0 / COSMOS3_ROPE_FPS),
        atol=1e-3,
        rtol=0,
    )

    monkeypatch.setattr(cosmos3_transformer_module, "COSMOS3_ROPE_FPS", 48.0)
    torch.testing.assert_close(
        _vision_temporal_step(_base_call()),
        torch.tensor(24.0 / 48.0),
        atol=1e-3,
        rtol=0,
    )
