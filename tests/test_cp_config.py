# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-GPU tests for artifixer context-parallel group-size selection:
pipelines list the CP sizes they can run (``cp_size_candidates``) and the
evaluation runner's ``_choose_cp_group_size`` ranks those candidates by validation
wall-clock cost.

No distributed environment required -- run with:
    pytest tests/test_cp_config.py -v
"""

from types import SimpleNamespace

import pytest

from model_eval.validation import EvaluationRunner
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase

# The real staticmethod, so these tests fail if its selection logic drifts.
_choose_cp_group_size = EvaluationRunner._choose_cp_group_size


def _base_candidates(
    frame_divisor: int,
    world_size: int,
    *,
    head_divisor: int | None = None,
    num_latent_frames: int | None = None,
    causal_block_frames: int | None = None,
) -> list[int]:
    """The real base-pipeline candidate rule on a minimal duck-typed pipe."""
    pipe = SimpleNamespace(
        _cp_frame_divisor=lambda total_latent_frames: frame_divisor,
        causal_block_frames=causal_block_frames,
        cp_head_divisor=lambda required=False: head_divisor,
    )
    return ArtifixerPipelineBase.cp_size_candidates(
        pipe, num_latent_frames if num_latent_frames is not None else frame_divisor, world_size
    )


class TestChooseCPGroupSize:
    """Base-rule candidates (frame divisors, block counts, head divisors)
    ranked by ``rounds * ceil(pass_latent_frames / cp_size)``."""

    def test_block_causal_21_frames_7_fpb(self):
        """21 latent frames, 7 frames/block, 20 GPUs, 4 tasks.

        num_blocks = 3, so candidates = divisors of gcd(21, 3) = {1, 3}.
        CP=3: 20//3=6 groups, ceil(4/6)=1 round, cost=1*ceil(21/3)=7.
        CP=1: 20 groups, 1 round, cost=21.
        Best: CP=3.
        """
        candidates = _base_candidates(21, world_size=20, num_latent_frames=21, causal_block_frames=7)
        assert candidates == [1, 3]
        assert _choose_cp_group_size(candidates, world_size=20, num_tasks=4, pass_latent_frames=21) == 3

    def test_no_block_causal_21_frames(self):
        """21 latent frames, no block-causal: candidates {1, 3, 7} at 20 GPUs.

        CP=7: 20//7=2 groups, ceil(4/2)=2 rounds, cost=2*3=6.
        CP=3: 6 groups, 1 round, cost=7.
        Best: CP=7.
        """
        candidates = _base_candidates(21, world_size=20)
        assert candidates == [1, 3, 7]
        assert _choose_cp_group_size(candidates, world_size=20, num_tasks=4, pass_latent_frames=21) == 7

    def test_many_tasks_favors_data_parallel(self):
        """12 latent frames, 4 frames/block, 8 GPUs, 8 tasks.

        Candidates = divisors of gcd(12, 3) = {1, 3}.
        CP=3: 8//3=2 groups, ceil(8/2)=4 rounds, cost=4*4=16.
        CP=1: 8 groups, 1 round, cost=12.
        Best: CP=1 (data-parallel wins with many tasks).
        """
        candidates = _base_candidates(12, world_size=8, num_latent_frames=12, causal_block_frames=4)
        assert candidates == [1, 3]
        assert _choose_cp_group_size(candidates, world_size=8, num_tasks=8, pass_latent_frames=12) == 1

    def test_tie_breaks_to_smaller_cp(self):
        """8 latent frames, 2 frames/block, 4 GPUs, 2 tasks.

        Candidates = divisors of gcd(8, 4) = {1, 2, 4}.
        CP=4: 1 group, 2 rounds, cost=2*2=4.  CP=2: 2 groups, 1 round,
        cost=4.  Tie at 4; ascending iteration with strict < means CP=2 wins.
        """
        candidates = _base_candidates(8, world_size=4, num_latent_frames=8, causal_block_frames=2)
        assert candidates == [1, 2, 4]
        assert _choose_cp_group_size(candidates, world_size=4, num_tasks=2, pass_latent_frames=8) == 2

    def test_single_gpu_returns_1(self):
        """A single GPU offers only CP=1."""
        candidates = _base_candidates(21, world_size=1)
        assert candidates == [1]
        assert _choose_cp_group_size(candidates, world_size=1, num_tasks=4, pass_latent_frames=21) == 1

    def test_few_tasks_favors_cp(self):
        """6 latent frames, 3 frames/block, 6 GPUs, 1 task.

        Candidates = divisors of gcd(6, 2) = {1, 2}.
        CP=2: 3 groups, 1 round, cost=3.  CP=1: cost=6.  Best: CP=2.
        """
        candidates = _base_candidates(6, world_size=6, num_latent_frames=6, causal_block_frames=3)
        assert candidates == [1, 2]
        assert _choose_cp_group_size(candidates, world_size=6, num_tasks=1, pass_latent_frames=6) == 2

    def test_frame_divisor_equals_world_size(self):
        """The frame divisor equals world_size -- maximum CP should be chosen
        when there are few tasks (8 frames, 8 GPUs, 1 task: cost 1*1=1)."""
        assert _choose_cp_group_size([1, 2, 4, 8], world_size=8, num_tasks=1, pass_latent_frames=8) == 8

    def test_attention_heads_constrain_cp_size(self):
        """The base rule drops frame divisors that split the head gcd (40
        heads): 3 and 6 divide the frames but not the heads."""
        assert _base_candidates(6, world_size=6, head_divisor=40) == [1, 2]
        assert _choose_cp_group_size([1, 2], world_size=6, num_tasks=1, pass_latent_frames=6) == 2
        assert _base_candidates(3, world_size=6, head_divisor=40) == [1]
        assert _choose_cp_group_size([1], world_size=6, num_tasks=1, pass_latent_frames=3) == 1


class TestPaddedShardingSelection:
    """Cosmos3 pipelines offer every divisor of the attention-head gcd as a
    candidate — their transformer pads the frame axis, so no frame rule
    applies (the family candidate rules are pinned on real pipelines in
    test_causal_training.py and test_cosmos3_ar_rollout.py). The chooser weighs
    every candidate by ``rounds * ceil(pass_latent_frames / cp_size)``, so
    pad overhead counts against a candidate."""

    def test_production_geometry_unlocks_cp(self):
        """21 latent frames x 32/8 heads (gcd 8): frame divisors {1, 3, 7,
        21} share no divisor > 1 with the heads, so the base rule is stuck
        at CP=1; the head-gcd candidates {1, 2, 4, 8} unlock CP=2 at
        8 GPUs / 12 tasks:
          d=1: groups=8, rounds=2, cost=2*21=42
          d=2: groups=4, rounds=3, cost=3*ceil(21/2)=33  <-- best
          d=4: groups=2, rounds=6, cost=6*ceil(21/4)=36
          d=8: groups=1, rounds=12, cost=12*ceil(21/8)=36
        """
        assert _base_candidates(21, world_size=8, head_divisor=8) == [1]
        assert _choose_cp_group_size([1], world_size=8, num_tasks=12, pass_latent_frames=21) == 1
        assert _choose_cp_group_size([1, 2, 4, 8], world_size=8, num_tasks=12, pass_latent_frames=21) == 2

    def test_single_task_maximizes_cp(self):
        """One task, 8 GPUs: cost = ceil(21/d), minimized at d=8 (3 frames
        per rank, 3 of them padding)."""
        assert _choose_cp_group_size([1, 2, 4, 8], world_size=8, num_tasks=1, pass_latent_frames=21) == 8

    def test_exact_frame_divisor_wins_cost_ties(self):
        """7-frame KV-cache chunks (candidates = chunk divisors plus head
        divisors): d=7 (pad-free, cost ceil(7/7)=1) ties d=8 (cost
        ceil(7/8)=1); ascending iteration with strict < keeps the pad-free
        divisor."""
        assert _choose_cp_group_size([1, 2, 4, 7, 8], world_size=8, num_tasks=1, pass_latent_frames=7) == 7

    def test_chooser_requires_positive_pass_frames(self):
        with pytest.raises(AssertionError, match="pass_latent_frames"):
            _choose_cp_group_size([1, 2], world_size=8, num_tasks=1, pass_latent_frames=0)


def test_cp_raw_frame_padding_per_family():
    """Eval-side raw CP padding is pipeline-owned: cosmos3 transformers pad
    their own frame axis (no raw padding — the model sees the same frames at
    every CP size), while the base/Wan rule pads the raw clip until the
    latent count divides the CP size (81 frames = 21 latents -> 85 = 22 at
    CP=2)."""
    from model_training.pipeline.cosmos3_kv_cache_pipeline import ArtifixerCosmos3KvCachePipeline
    from model_training.pipeline.cosmos3_pipeline import ArtifixerCosmos3Pipeline
    from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline
    from model_training.pipeline.wan_pipeline import ArtifixerPipeline

    assert ArtifixerCosmos3Pipeline.cp_raw_frame_padding(None, 81, 2, 4) == 0
    assert ArtifixerCosmos3KvCachePipeline.cp_raw_frame_padding(None, 81, 2, 4) == 0
    for pipeline_cls in (ArtifixerPipelineBase, ArtifixerPipeline, ArtifixerKvCachePipeline):
        assert pipeline_cls.cp_raw_frame_padding(None, 81, 2, 4) == 4
        assert pipeline_cls.cp_raw_frame_padding(None, 81, 1, 4) == 0, "CP=1 needs no padding"
