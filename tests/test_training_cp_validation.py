# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""run_supervised_training admits every model family under context parallelism;
the startup check is the pipeline's own validate_cp_size, called with the
run's latent frame count."""

import argparse
from types import SimpleNamespace

import pytest
from torch import nn

import model_training.utils.train_utils as train_utils
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline
from model_training.utils.train_utils import run_supervised_training


class _ReachedTraining(Exception):
    """Raised by the stubbed first call inside run_supervised_training."""


def test_cp_training_admits_cosmos3_at_the_entry_point(monkeypatch):
    """Both model families share the CP training path: a cosmos3 model_id with
    context_parallel=True reaches training setup instead of being refused."""

    def _sentinel(args):
        raise _ReachedTraining

    monkeypatch.setattr(train_utils, "get_run_id_and_should_resume", _sentinel)
    with pytest.raises(_ReachedTraining):
        run_supervised_training(
            argparse.Namespace(model_id=COSMOS3_NANO_MODEL_ID),
            frames_per_block=None,
            base_checkpoint_dir=None,
            context_parallel=True,
        )


class _SizeCheckRecorded(Exception):
    """Raised by the recording pipe once validate_cp_size has run."""


def test_cp_training_startup_runs_the_pipelines_size_check(monkeypatch):
    """After the frame-layout check, startup hands the requested CP size and
    the run's latent frame count to pipe.validate_cp_size — the same check
    eval and the rollout entry run."""
    recorded = []

    def _record_size_check(cp_size, num_latent_frames):
        recorded.append((cp_size, num_latent_frames))
        raise _SizeCheckRecorded

    pipe = SimpleNamespace(
        vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)),
        validate_cp_size=_record_size_check,
    )
    monkeypatch.setattr(train_utils, "get_run_id_and_should_resume", lambda args: ("run", False))
    monkeypatch.setattr(train_utils, "configure_context_parallel_shard_size", lambda args: None)
    monkeypatch.setattr(train_utils, "get_accelerator", lambda args, run_id: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(train_utils, "get_pipe", lambda *call_args, **call_kwargs: pipe)

    with pytest.raises(_SizeCheckRecorded):
        run_supervised_training(
            argparse.Namespace(model_id=COSMOS3_NANO_MODEL_ID, num_frames=85, context_parallel_size=2),
            frames_per_block=None,
            base_checkpoint_dir=None,
            context_parallel=True,
        )

    assert recorded == [(2, 22)]  # 85 raw frames = 22 latent frames


class _GqaAttentionStub(nn.Module):
    """Grouped-query attention shape of the reduced cosmos3 config: the KV
    head count (2) is the binding CP constraint."""

    def __init__(self):
        super().__init__()
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.processor = SimpleNamespace(_cp_mesh=None)


class _GqaPipe:
    """The base pipeline's size check, bound onto a stub whose transformer
    carries the GQA head counts."""

    cp_head_divisor = ArtifixerPipelineBase.cp_head_divisor
    validate_cp_size = ArtifixerPipelineBase.validate_cp_size

    def __init__(self):
        transformer = nn.Module()
        transformer.attn = _GqaAttentionStub()
        self.transformer = transformer

    def _cp_frame_divisor(self, total_latent_frames):
        return total_latent_frames


def test_cp_size_must_divide_the_head_divisor():
    _GqaPipe().validate_cp_size(2, 8)
    with pytest.raises(AssertionError, match=r"CP size 4 must divide .*head divisor 2"):
        _GqaPipe().validate_cp_size(4, 8)


class _RolloutPipe:
    """The Wan rollout shape on the base size check: chunk-divisor frame
    rule, and cp_head_divisor is None at its owner — the cache path keeps
    all heads per rank."""

    ar_frames_per_chunk = 7
    _cp_frame_divisor = ArtifixerKvCachePipeline._cp_frame_divisor
    cp_head_divisor = ArtifixerKvCachePipeline.cp_head_divisor
    validate_cp_size = ArtifixerPipelineBase.validate_cp_size


def test_rollout_pipes_stay_exempt_from_the_head_rule():
    _RolloutPipe().validate_cp_size(7, 21)
