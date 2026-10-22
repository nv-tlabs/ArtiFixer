# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType

import torch

from model_training.data.dl3dv_test import DL3DVPairedDatasetTest


def _fake_get_item_inner(self, data_idx, num_train_views, num_neighbors, generator=None):
    assert generator is not None
    return {
        "data_idx": data_idx,
        "num_train_views": num_train_views,
        "num_neighbors": num_neighbors,
        "draws": torch.randint(0, 1_000_000, size=(8,), generator=generator),
    }


def _make_validation_probe(num_views: int = 3) -> DL3DVPairedDatasetTest:
    dataset = object.__new__(DL3DVPairedDatasetTest)
    dataset.data = [object() for _ in range(5)]
    dataset.num_views = num_views
    dataset.validation_seed = 42
    dataset.manifest_indices = None
    dataset.get_item_inner = MethodType(_fake_get_item_inner, dataset)
    return dataset


def test_validation_sampling_is_a_pure_function_of_the_index():
    first = _make_validation_probe()[2]
    second = _make_validation_probe()[2]

    assert first["data_idx"] == second["data_idx"]
    assert first["num_train_views"] == second["num_train_views"]
    assert first["num_neighbors"] == second["num_neighbors"]
    assert torch.equal(first["draws"], second["draws"])


def test_validation_sampling_changes_across_indices():
    index_2 = _make_validation_probe()[2]
    index_3 = _make_validation_probe()[3]

    assert index_2["data_idx"] != index_3["data_idx"]
    assert not torch.equal(index_2["draws"], index_3["draws"])
