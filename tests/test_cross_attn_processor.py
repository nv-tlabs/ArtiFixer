# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the reference-view (PRoPE) branch of ArtifixerCrossAttnProcessor.

When neighbor (reference-view) hidden states are present, the processor runs a
second attention pass: the PRoPE target transforms wrap the query and the
attention output, the PRoPE reference transform wraps the neighbor K/V, all in
float32 with results cast back to the query dtype, and the neighbor output is
added to the main cross-attention output while both are still head-split
(batch, seq, heads, head_dim) -- only then are heads flattened and the output
projection applied.  These tests compare that composition against a manually
computed reference built from the same fakes.  Each branch uses distinct
constant tensors, and the query path is float16 while the neighbor projections
are float32, so a dropped ``.to(query_dtype)`` cast changes the result.

Uses the same fake attention and monkeypatched dispatch_attention_fn as
tests/test_kv_cache_pipeline.py.  No GPU required -- run with:
    pytest tests/test_cross_attn_processor.py -v
"""

import unittest
from unittest.mock import patch

import torch
from torch import nn

import model_training.net.wan_transformer as transformer_mod
from model_training.net.prope import PropeApplyFns
from model_training.net.wan_transformer import ArtifixerCrossAttnProcessor

_BATCH = 2
_HEADS = 2
_HEAD_DIM = 4
_DIM = _HEADS * _HEAD_DIM
_QUERY_TOKENS = 3
_TEXT_TOKENS = 4
_NEIGHBOR_TOKENS = 5


def _ramp(*shape, dtype, low=-1.0, high=1.0):
    """Deterministic, per-element-distinct constant tensor."""
    numel = 1
    for size in shape:
        numel *= size
    return torch.linspace(low, high, numel, dtype=torch.float32).reshape(shape).to(dtype)


class _FakeWanAttention(nn.Module):
    """tests/test_kv_cache_pipeline.py fake, extended with the neighbor (added-KV) modules."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.add_k_proj = nn.Linear(dim, dim, bias=False)
        self.add_v_proj = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.norm_added_k = nn.Identity()
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Identity()])


def _fake_attention_math(query, value):
    """Stand-in for dispatch_attention_fn: query-shaped output that mixes both inputs."""
    return query * 2 + value.mean(dim=1, keepdim=True)


def _target_q_transform(tensor):
    return tensor * 2.0 + 0.25


def _reference_kv_transform(tensor):
    return tensor * 3.0


def _target_o_transform(tensor):
    return tensor * 0.5 - 0.125


def _recording(log, tag, fn):
    def wrapper(tensor):
        log.append((tag, tuple(tensor.shape), tensor.dtype))
        return fn(tensor)

    return wrapper


def _must_not_run(tag):
    def wrapper(tensor):
        raise AssertionError(f"{tag} must not be called by the cross-attention neighbor branch")

    return wrapper


class CrossAttnProcessorNeighborBranchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.attn = _FakeWanAttention(_DIM, _HEADS).to(torch.float16)
        with torch.no_grad():
            self.attn.to_out[0].weight.copy_(_ramp(_DIM, _DIM, dtype=torch.float16, low=-0.5, high=0.5))

        self.hidden_states = _ramp(_BATCH, _QUERY_TOKENS, _DIM, dtype=torch.float16)
        self.encoder_hidden_states = _ramp(_BATCH, _TEXT_TOKENS, _DIM, dtype=torch.float16, low=0.0, high=2.0)
        self.neighbor_hidden_states = _ramp(_BATCH, _NEIGHBOR_TOKENS, _DIM, dtype=torch.float16, low=-2.0, high=2.0)

        # Distinct constants per projection so every branch is distinguishable in the output.
        # The main path is float16 (the query dtype); the neighbor projections are float32
        # values that round under a float16 cast, so a missing .to(query_dtype) is caught.
        self.query_proj = _ramp(_BATCH, _QUERY_TOKENS, _DIM, dtype=torch.float16, low=-1.0, high=1.0)
        self.key_proj = _ramp(_BATCH, _TEXT_TOKENS, _DIM, dtype=torch.float16, low=0.0, high=2.0)
        self.value_proj = _ramp(_BATCH, _TEXT_TOKENS, _DIM, dtype=torch.float16, low=-2.0, high=0.0)
        self.neighbor_key_proj = _ramp(_BATCH, _NEIGHBOR_TOKENS, _DIM, dtype=torch.float32, low=-1 / 3, high=2 / 3)
        self.neighbor_value_proj = _ramp(_BATCH, _NEIGHBOR_TOKENS, _DIM, dtype=torch.float32, low=1 / 7, high=5 / 7)

    def _run(self, *, neighbor_hidden_states, ignore_neighbors=False, prope_target=None, prope_reference=None):
        prope_log = []
        dispatch_calls = []

        def fake_get_qkv_projections(attn, hidden_states, encoder_hidden_states):
            self.assertIs(attn, self.attn)
            self.assertIs(encoder_hidden_states, self.encoder_hidden_states)
            return self.query_proj, self.key_proj, self.value_proj

        def fake_get_added_kv_projections(attn, neighbor_states):
            self.assertIs(attn, self.attn)
            self.assertIs(neighbor_states, self.neighbor_hidden_states)
            return self.neighbor_key_proj, self.neighbor_value_proj

        def fake_dispatch_attention_fn(query, key, value, **kwargs):
            dispatch_calls.append({"query": query, "key": key, "value": value, **kwargs})
            return _fake_attention_math(query, value)

        if prope_target is None:
            prope_target = PropeApplyFns(
                apply_to_q=_recording(prope_log, "target.apply_to_q", _target_q_transform),
                apply_to_kv=_must_not_run("prope_target.apply_to_kv"),
                apply_to_o=_recording(prope_log, "target.apply_to_o", _target_o_transform),
            )
        if prope_reference is None:
            prope_reference = PropeApplyFns(
                apply_to_q=_must_not_run("prope_reference.apply_to_q"),
                apply_to_kv=_recording(prope_log, "reference.apply_to_kv", _reference_kv_transform),
                apply_to_o=_must_not_run("prope_reference.apply_to_o"),
            )

        with (
            patch.object(transformer_mod, "_get_qkv_projections", fake_get_qkv_projections),
            patch.object(transformer_mod, "_get_added_kv_projections", fake_get_added_kv_projections),
            patch.object(transformer_mod, "dispatch_attention_fn", fake_dispatch_attention_fn),
        ):
            output = ArtifixerCrossAttnProcessor()(
                self.attn,
                self.hidden_states,
                encoder_hidden_states=self.encoder_hidden_states,
                neighbor_hidden_states=neighbor_hidden_states,
                ignore_neighbors=ignore_neighbors,
                prope_target=prope_target,
                prope_reference=prope_reference,
            )
        return output, prope_log, dispatch_calls

    def _expected(self, *, with_neighbors=True, ignore_neighbors=False):
        """Manual reference composition mirroring the processor, built from the same fakes."""
        heads = (_HEADS, -1)
        query = self.query_proj.unflatten(2, heads)
        value = self.value_proj.unflatten(2, heads)
        main_out = _fake_attention_math(query, value)

        composed = main_out
        if with_neighbors:
            key_neighbor = self.neighbor_key_proj.unflatten(2, heads)
            value_neighbor = self.neighbor_value_proj.unflatten(2, heads)
            value_neighbor = (
                _reference_kv_transform(value_neighbor.transpose(2, 1).float()).transpose(2, 1).to(torch.float16)
            )
            query_neighbor = _target_q_transform(query.transpose(2, 1).float()).transpose(2, 1).to(torch.float16)

            neighbor_out = _fake_attention_math(query_neighbor, value_neighbor)
            neighbor_out = _target_o_transform(neighbor_out.transpose(2, 1).float()).transpose(2, 1).to(torch.float16)
            composed = main_out + neighbor_out * (0 if ignore_neighbors else 1)

        return self.attn.to_out[1](self.attn.to_out[0](composed.flatten(2, 3)))

    def test_neighbor_output_composed_head_split_and_cast_to_query_dtype(self):
        # Confirm the float32 neighbor values change under a float16 round-trip;
        # otherwise the exact-equality check below could not catch a missing .to(query_dtype).
        transformed = _reference_kv_transform(self.neighbor_value_proj)
        assert not torch.equal(transformed, transformed.to(torch.float16).float())

        output, prope_log, dispatch_calls = self._run(neighbor_hidden_states=self.neighbor_hidden_states)

        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(output.shape, (_BATCH, _QUERY_TOKENS, _DIM))
        self.assertTrue(torch.equal(output, self._expected()))

        # PRoPE transforms see head-major (batch, heads, seq, head_dim) float32 tensors,
        # in the order K-neighbor, V-neighbor, Q, then attention output.
        self.assertEqual(
            prope_log,
            [
                ("reference.apply_to_kv", (_BATCH, _HEADS, _NEIGHBOR_TOKENS, _HEAD_DIM), torch.float32),
                ("reference.apply_to_kv", (_BATCH, _HEADS, _NEIGHBOR_TOKENS, _HEAD_DIM), torch.float32),
                ("target.apply_to_q", (_BATCH, _HEADS, _QUERY_TOKENS, _HEAD_DIM), torch.float32),
                ("target.apply_to_o", (_BATCH, _HEADS, _QUERY_TOKENS, _HEAD_DIM), torch.float32),
            ],
        )

        # Both attention passes run on head-split tensors already cast to the query dtype.
        self.assertEqual(len(dispatch_calls), 2)
        neighbor_call = next(call for call in dispatch_calls if call["key"].shape[1] == _NEIGHBOR_TOKENS)
        main_call = next(call for call in dispatch_calls if call["key"].shape[1] == _TEXT_TOKENS)
        for call in (neighbor_call, main_call):
            self.assertEqual(call["query"].shape, (_BATCH, _QUERY_TOKENS, _HEADS, _HEAD_DIM))
            self.assertEqual(call["key"].shape[2:], (_HEADS, _HEAD_DIM))
            self.assertEqual(call["value"].shape, call["key"].shape)
            self.assertEqual(call["query"].dtype, torch.float16)
            self.assertEqual(call["key"].dtype, torch.float16)
            self.assertEqual(call["value"].dtype, torch.float16)
            self.assertIsNone(call["attn_mask"])
            self.assertFalse(call["is_causal"])

    def test_ignore_neighbors_zeroes_neighbor_contribution(self):
        output, _, dispatch_calls = self._run(neighbor_hidden_states=self.neighbor_hidden_states, ignore_neighbors=True)

        # The neighbor pass still runs (it feeds the neighbor KV cache) but contributes zero.
        self.assertEqual(len(dispatch_calls), 2)
        self.assertTrue(torch.equal(output, self._expected(ignore_neighbors=True)))
        self.assertFalse(torch.equal(self._expected(ignore_neighbors=True), self._expected()))

    def test_no_neighbor_states_skips_prope_and_neighbor_attention(self):
        forbidden = PropeApplyFns(
            apply_to_q=_must_not_run("prope.apply_to_q"),
            apply_to_kv=_must_not_run("prope.apply_to_kv"),
            apply_to_o=_must_not_run("prope.apply_to_o"),
        )
        output, prope_log, dispatch_calls = self._run(
            neighbor_hidden_states=None, prope_target=forbidden, prope_reference=forbidden
        )

        self.assertEqual(len(dispatch_calls), 1)
        self.assertEqual(prope_log, [])
        self.assertTrue(torch.equal(output, self._expected(with_neighbors=False)))


if __name__ == "__main__":
    unittest.main()
