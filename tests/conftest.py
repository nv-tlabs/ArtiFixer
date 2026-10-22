# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest setup for the test suite."""

import pytest

# transformer.py requires the torch >= 2.11 flash-attention impl API. Add
# no-op shims for the two functions when they are missing so the module
# imports on older CPU-only torch; never override a real implementation.
try:
    import torch.nn.attention as _torch_attention
except ImportError:  # torch-less environments; torch-dependent tests skip themselves
    pass
else:
    if not hasattr(_torch_attention, "activate_flash_attention_impl"):
        _torch_attention.activate_flash_attention_impl = lambda *args, **kwargs: None
    if not hasattr(_torch_attention, "list_flash_attention_impls"):
        _torch_attention.list_flash_attention_impls = lambda: []


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: slower but CPU-runnable tests (e.g. the real-model contract smoke "
        "tests); deselect with -m 'not slow'",
    )
    config.addinivalue_line(
        "markers",
        "requires_host_ffmpeg: encodes or decodes video, so it needs an ffmpeg from the host "
        "(IMAGEIO_FFMPEG_EXE or PATH); the release images ship none and these tests skip without one",
    )


def pytest_collection_modifyitems(config, items):
    marked = [item for item in items if item.get_closest_marker("requires_host_ffmpeg")]
    if not marked:
        return
    from model_training.utils.video_io import resolve_ffmpeg_exe

    try:
        resolve_ffmpeg_exe()
    except RuntimeError:
        skip = pytest.mark.skip(reason="no host ffmpeg (IMAGEIO_FFMPEG_EXE or PATH); the codec-free image ships none")
        for item in marked:
            item.add_marker(skip)


@pytest.fixture
def isolated_accelerator_state():
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    try:
        yield
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()


@pytest.fixture
def recording_metric_suite():
    from model_eval.metrics_utils import MetricsAggregator

    class RecordingMetricSuite:
        instances = []

        def __init__(self, *args, **kwargs):
            self.options = kwargs
            self.ours = MetricsAggregator(["psnr"])
            self.gut = MetricsAggregator(["psnr"])
            self.ours_masked = MetricsAggregator(["psnr"])
            self.gut_masked = MetricsAggregator(["psnr"])
            self.reports = []
            self.instances.append(self)

        def update(self, scene_id, pred, rendered, gt):
            self.ours.add(scene_id, psnr=1.0)
            self.gut.add(scene_id, psnr=1.0)

        def update_masked(self, scene_id, pred, rendered, gt, mask):
            self.ours_masked.add(scene_id, psnr=1.0)
            self.gut_masked.add(scene_id, psnr=1.0)

        def report(self, path, **kwargs):
            self.reports.append(path)

    return RecordingMetricSuite
