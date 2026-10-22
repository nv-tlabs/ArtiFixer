# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed wiring tests (no real process group).

Pin two easy-to-regress contracts:

- the scene-metadata broadcast runs on a dedicated gloo group with a
  60-minute timeout (rank 0 can scan shared storage far longer than the
  default NCCL timeout) and destroys that group even when the broadcast
  fails;
- the replicated-DCP save/load wrappers pass ``no_dist=True`` (every rank
  holds the full replicated state dict; a collective save/load would hang or
  shard it).
"""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import model_training.utils.train_utils as train_utils
from model_training.data.dl3dv_base import DL3DVPairedDatasetBase, dist


def _mock_dist(monkeypatch, *, broadcast):
    calls = SimpleNamespace(new_group_kwargs=None, group=None, destroyed=[])

    def fake_new_group(*args, **kwargs):
        calls.new_group_kwargs = kwargs
        calls.group = object()
        return calls.group

    def fake_broadcast_object_list(values, src, group=None):
        calls.broadcast_group = group
        broadcast(values)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "new_group", fake_new_group)
    monkeypatch.setattr(dist, "broadcast_object_list", fake_broadcast_object_list)
    monkeypatch.setattr(dist, "destroy_process_group", calls.destroyed.append)
    return calls


def _load_scene_metadata(monkeypatch, *, broadcast):
    calls = _mock_dist(monkeypatch, broadcast=broadcast)
    dataset = object.__new__(DL3DVPairedDatasetBase)
    result = DL3DVPairedDatasetBase._load_scene_metadata(dataset, {}, Path("/nonexistent/prompts"), None, verbose=False)
    return calls, result


def test_scene_metadata_broadcast_uses_a_long_timeout_gloo_group(monkeypatch):
    metadata = {"scene": ("1K", {"frames": []}, False, [])}

    def broadcast(values):
        values[0] = metadata
        values[1] = None

    calls, result = _load_scene_metadata(monkeypatch, broadcast=broadcast)

    assert calls.new_group_kwargs == {"backend": "gloo", "timeout": timedelta(minutes=60)}
    # The broadcast must run on the dedicated group, not the default one.
    assert calls.broadcast_group is calls.group
    # The group is destroyed after use.
    assert calls.destroyed == [calls.group]
    assert result == metadata


def test_scene_metadata_group_is_destroyed_even_when_the_broadcast_fails(monkeypatch):
    def broadcast(values):
        raise RuntimeError("collective failed")

    calls = _mock_dist(monkeypatch, broadcast=broadcast)
    dataset = object.__new__(DL3DVPairedDatasetBase)
    with pytest.raises(RuntimeError, match="collective failed"):
        DL3DVPairedDatasetBase._load_scene_metadata(dataset, {}, Path("/nonexistent/prompts"), None, verbose=False)

    assert calls.destroyed == [calls.group]


def test_replicated_dcp_wrappers_save_and_load_without_collectives(monkeypatch, tmp_path):
    recorded = {}

    def fake_save(state_dict, **kwargs):
        recorded["save"] = kwargs

    def fake_load(state_dict, **kwargs):
        recorded["load"] = kwargs

    monkeypatch.setattr(train_utils.dcp, "save", fake_save)
    monkeypatch.setattr(train_utils.dcp, "load", fake_load)

    state_dict = {"weight": torch.zeros(1)}
    train_utils._save_replicated_state_dict_to_dcp(state_dict, tmp_path / "cache")
    train_utils._load_replicated_state_dict_from_dcp(state_dict, tmp_path / "cache")

    assert recorded["save"]["no_dist"] is True
    assert recorded["load"]["no_dist"] is True
