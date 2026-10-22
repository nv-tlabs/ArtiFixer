# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests._helpers import _run_main_with_event_log


@pytest.mark.parametrize("ema_weight", [0.0, 0.99])
def test_dmd_setup_does_not_move_unsharded_score_models_to_gpu(monkeypatch, tmp_path, ema_weight):
    events = _run_main_with_event_log(monkeypatch, tmp_path, ema_weight=ema_weight)
    built, sharded, moved = set(), set(), set()
    for kind, role in events:
        if kind == "build":
            built.add(role)
        elif kind == "shard":
            assert role in built, f"shard before build: {events}"
            sharded.add(role)
        elif kind == "move":
            assert role in sharded, f"{role} moved to the accelerator before sharding: {events}"
            moved.add(role)

    assert {"score_0", "score_1"} <= built
    assert built == sharded == moved
