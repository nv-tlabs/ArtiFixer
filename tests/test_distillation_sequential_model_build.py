# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural guard for the DMD host-memory invariant.

Building the student, fake-score critic, and frozen teacher before sharding
would materialize three full FP32 16B transformers (about 64 GB each) per
rank. Four ranks can therefore need about 768 GB of host memory before any
FSDP2 sharding releases a full model copy.

The fix serializes construction: every transformer build must be followed by
that model's FSDP2 shard (accelerator.prepare / accelerator._prepare_fsdp2 /
fsdp2_prepare_model, which free the full host copy) before the next build
fires, capping the per-rank peak at ONE unsharded transformer — the same
footprint stage-1 is proven to fit with. These tests pin that ordering
structurally (no RSS measurement): main runs against fakes that log a
build/shard event stream, and the walker below rejects any overlap.
"""

from __future__ import annotations

import pytest

from tests._helpers import _run_main_with_event_log


def _assert_at_most_one_unsharded(events: list[tuple[str, str]]) -> None:
    unsharded: list[str] = []
    for kind, role in events:
        if kind == "build":
            assert not unsharded, (
                f"transformer {role!r} was built while {unsharded} still held full unsharded "
                f"host weights (the host-OOM failure class this guard exists for); event log: {events}"
            )
            unsharded.append(role)
        elif kind == "shard":
            assert (
                unsharded and unsharded[-1] == role
            ), f"shard of {role!r} does not match the most recently built transformer; event log: {events}"
            unsharded.pop()
    assert not unsharded, f"transformers built but never sharded: {unsharded}; event log: {events}"


@pytest.mark.parametrize(
    ("ema_weight", "expected_builds"),
    [
        # Production default (--ema_weight 0.99): student + two critics; EMA clones parameter shards.
        (0.99, {"student", "score_0", "score_1"}),
        # EMA disabled: student + two critics.
        (0.0, {"student", "score_0", "score_1"}),
    ],
)
def test_distillation_builds_each_transformer_only_after_previous_is_sharded(
    monkeypatch, tmp_path, ema_weight, expected_builds
):
    events = _run_main_with_event_log(monkeypatch, tmp_path, ema_weight=ema_weight)

    built = {role for kind, role in events if kind == "build"}
    assert built == expected_builds, f"unexpected transformer builds; event log: {events}"
    _assert_at_most_one_unsharded(events)
