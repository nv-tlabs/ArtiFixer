# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow-shift single-sourcing tests.

Every scheduler in a run — train FlowMatchScheduler, validation UniPC, the
KV-cache rollout scheduler, and the DMD noising warp — must read the shift
from the model profile, so the noising warp and the step schedule cannot
desynchronize.
"""

import inspect
from types import SimpleNamespace

import torch

from model_training.model_profiles import (
    COSMOS3_FLOW_SHIFT,
    COSMOS3_NANO_MODEL_ID,
    WAN22_TI2V_5B_MODEL_ID,
    WAN_FLOW_SHIFT,
    flow_shift_for_model,
)
from model_training.pipeline.cosmos3_kv_cache_pipeline import ArtifixerCosmos3KvCachePipeline
from model_training.trainers.dmd import DMD
from model_training.utils import train_utils
from tests._helpers import _TinyCosmosBaseTransformer, _TinyCosmosVae, patch_kv_cache_pipeline_factories

_NUM_TRAIN_TIMESTEPS = 1000
_MIN_STEP = 20
_MAX_STEP = 980
_SEED = 1234


def test_flow_shift_accessor_resolves_the_cosmos3_profile_constant():
    assert flow_shift_for_model(COSMOS3_NANO_MODEL_ID) == COSMOS3_FLOW_SHIFT


def test_validation_unipc_scheduler_reads_profile_flow_shift():
    scheduler = train_utils._scheduler_for_model(COSMOS3_NANO_MODEL_ID)

    assert scheduler.config.flow_shift == COSMOS3_FLOW_SHIFT
    assert scheduler.config.prediction_type == "flow_prediction"
    assert scheduler.config.use_flow_sigmas is True


def test_cosmos3_kv_cache_rollout_scheduler_defaults_to_profile_flow_shift():
    signature = inspect.signature(ArtifixerCosmos3KvCachePipeline.__init__)

    assert signature.parameters["flow_shift"].default == COSMOS3_FLOW_SHIFT


def test_cosmos3_kv_cache_pipeline_scheduler_is_the_flow_match_rollout_scheduler():
    """The pipeline's scheduler is the profile-shift FlowMatchScheduler, not
    a throwaway scheduler that is later overwritten."""
    from model_training.schedulers.flow_match import FlowMatchScheduler

    pipe = ArtifixerCosmos3KvCachePipeline(
        vae=_TinyCosmosVae(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
        ar_frames_per_chunk=7,
        local_attn_size=21,
        sink_size=7,
    )

    assert isinstance(pipe.scheduler, FlowMatchScheduler)
    assert pipe.scheduler.shift == COSMOS3_FLOW_SHIFT


def _trainer_with_scheduler_shift(shift: float) -> DMD:
    trainer = object.__new__(DMD)
    trainer.pipe = SimpleNamespace(
        scheduler=SimpleNamespace(shift=shift, num_train_timesteps=_NUM_TRAIN_TIMESTEPS),
        vae=SimpleNamespace(device=torch.device("cpu")),
    )
    trainer.min_step = _MIN_STEP
    trainer.max_step = _MAX_STEP
    return trainer


def _warped_timesteps(shift: float, batch_size: int = 256) -> torch.Tensor:
    torch.manual_seed(_SEED)
    return _trainer_with_scheduler_shift(shift).get_timestep(batch_size)


def test_dmd_noising_warp_follows_the_rollout_scheduler_shift():
    batch_size = 256
    torch.manual_seed(_SEED)
    base = torch.randint(0, _NUM_TRAIN_TIMESTEPS, (batch_size,), dtype=torch.long)

    for shift in (5.0, 10.0):
        u = base / _NUM_TRAIN_TIMESTEPS
        expected = (shift * u / (1 + (shift - 1) * u) * _NUM_TRAIN_TIMESTEPS).clamp(_MIN_STEP, _MAX_STEP)
        assert torch.equal(_warped_timesteps(shift, batch_size), expected)

    # Two different scheduler shifts warp the same base draw differently:
    # the warp really is a function of the pipeline scheduler's shift.
    assert not torch.equal(_warped_timesteps(5.0, batch_size), _warped_timesteps(10.0, batch_size))


def test_dmd_noising_warp_is_identity_at_shift_one():
    batch_size = 256
    torch.manual_seed(_SEED)
    base = torch.randint(0, _NUM_TRAIN_TIMESTEPS, (batch_size,), dtype=torch.long)

    assert torch.equal(_warped_timesteps(1.0, batch_size), base.clamp(_MIN_STEP, _MAX_STEP))


def test_get_kv_cache_pipe_forwards_the_resolved_profile_shift_to_both_families(monkeypatch):
    """get_kv_cache_pipe resolves the shift through flow_shift_for_model(args.model_id)
    and forwards it into the constructed pipeline for cosmos3 AND Wan."""
    constructed = []
    # The same list for both families records the four constructions in call order.
    patch_kv_cache_pipeline_factories(monkeypatch, constructed, constructed)

    cosmos3_args = SimpleNamespace(
        model_id=COSMOS3_NANO_MODEL_ID,
        gradient_checkpointing=False,
        attention_backend=None,
        frames_per_block=7,
        local_attn_size=21,
        sink_size=7,
    )
    wan_args = SimpleNamespace(
        model_id=WAN22_TI2V_5B_MODEL_ID,
        gradient_checkpointing=False,
        attention_backend=None,
        frames_per_block=7,
        local_attn_size=21,
        sink_size=7,
        checkpoint_every_n_blocks=1,
    )

    train_utils.get_kv_cache_pipe(cosmos3_args, "cpu")
    train_utils.get_kv_cache_pipe(wan_args, "cpu")

    # Sentinel resolver: proves the constructed shift comes from
    # flow_shift_for_model, not from a hardcoded copy of the profile value.
    sentinels = {COSMOS3_NANO_MODEL_ID: 7.25, WAN22_TI2V_5B_MODEL_ID: 9.5}
    monkeypatch.setattr(train_utils, "flow_shift_for_model", lambda model_id: sentinels[model_id])
    train_utils.get_kv_cache_pipe(cosmos3_args, "cpu")
    train_utils.get_kv_cache_pipe(wan_args, "cpu")

    assert [kwargs["flow_shift"] for kwargs in constructed] == [COSMOS3_FLOW_SHIFT, WAN_FLOW_SHIFT, 7.25, 9.5]


def test_wan_kv_cache_pipeline_flow_shift_defaults_to_profile_shift():
    """The Wan pipeline's flow_shift defaults to the profile value."""
    from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline

    signature = inspect.signature(ArtifixerKvCachePipeline.__init__)

    assert signature.parameters["flow_shift"].default == WAN_FLOW_SHIFT == 5.0


def test_train_flow_match_scheduler_sigma_profile_matches_the_profile_shift():
    """The training scheduler built from the profile shift matches the shifted sigma curve."""
    from model_training.schedulers.flow_match import FlowMatchScheduler

    shift = flow_shift_for_model(COSMOS3_NANO_MODEL_ID)
    scheduler = FlowMatchScheduler(
        shift=shift,
        sigma_min=0.0,
        extra_one_step=True,
        device="cpu",
    )

    assert scheduler.shift == shift
    linear = torch.linspace(1.0, 0.0, 1001)[:-1]
    torch.testing.assert_close(scheduler.sigmas, shift * linear / (1 + (shift - 1) * linear))
