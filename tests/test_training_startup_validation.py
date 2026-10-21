# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from model_training import stages
from model_training.checkpoint_metadata import write_checkpoint_metadata
from model_training.common.context_parallel import validate_latent_frame_layout
from model_training.distillation import build_parser


def _args(*extra):
    return build_parser().parse_args(
        [
            "--project_dir",
            "/unused",
            "--base_checkpoint_dir",
            "/unused",
            "--base_checkpoint_dir_critic",
            "/unused",
            "--split_path",
            "/unused",
            "--dl3dv_dir",
            "/unused",
            "--prompt_dir",
            "/unused",
            "--log_with",
            "none",
            *extra,
        ]
    )


@pytest.mark.parametrize("cp_size", [0, 2, 8])
def test_diffusion_forcing_rejects_unsupported_cp_before_setup(monkeypatch, cp_size):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported CP reached training setup")

    monkeypatch.setattr(stages, "run_supervised_training", unexpected)
    with pytest.raises(ValueError, match="context_parallel_size 1"):
        stages.run_diffusion_forcing(SimpleNamespace(context_parallel_size=cp_size))


def test_dmd_rejects_training_window_before_score_models_are_built(monkeypatch):
    args = _args("--num_frames", "1533", "--frames_per_block", "8")
    monkeypatch.setattr(stages, "resolve_camera_pose_alignment", lambda *args, **kwargs: "mean")
    monkeypatch.setattr(stages, "get_run_id_and_should_resume", lambda args: (None, False))
    monkeypatch.setattr(stages, "get_accelerator", lambda *args: SimpleNamespace(device="cpu"))
    pipe = SimpleNamespace(
        vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)),
        ar_window_chunks=4,
        ar_frames_per_chunk=8,
    )
    monkeypatch.setattr(stages, "get_kv_cache_pipe", lambda *args, **kwargs: pipe)
    with pytest.raises(AssertionError, match="384 latent frames exceed"):
        stages.run_distillation(args)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("teacher_pose", ["mean", "last", None])
def test_dmd_checks_teacher_pose_before_setup(tmp_path, monkeypatch, resume, teacher_pose):
    teacher = tmp_path / "teacher"
    if teacher_pose is not None:
        write_checkpoint_metadata(teacher, {"camera_pose_alignment": teacher_pose})
    args = _args(
        "--model_id",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "--base_checkpoint_dir_critic",
        str(teacher),
        *(["--resume_from_checkpoint", "auto"] if resume else []),
    )

    class ReachedSetup(Exception):
        pass

    def setup(*args):
        raise ReachedSetup

    monkeypatch.setattr(stages, "get_run_id_and_should_resume", setup)
    if teacher_pose == "last":
        with pytest.raises(ValueError, match="uses camera_pose_alignment"):
            stages.run_distillation(args)
    else:
        with pytest.raises(ReachedSetup):
            stages.run_distillation(args)


@pytest.mark.parametrize("limit", [0, -1])
def test_checkpoint_retention_limit_must_be_positive(limit):
    with pytest.raises(SystemExit):
        _args("--max_checkpoints", str(limit))


def test_pipeline_stage2_smoke_fits_the_short_training_window():
    source = (Path(__file__).resolve().parents[1] / "tests/test_pipeline.sh").read_text()
    stage2 = source.split("# STAGE 2:", 1)[1].split("# STAGE 3:", 1)[0]
    assert "--context_parallel_size 1 --shard_size $SHARD_SIZE" in stage2
    assert (
        validate_latent_frame_layout(
            num_frames=125,
            temporal_scale=4,
            frames_per_block=8,
            context_parallel_size=1,
            require_cp_rank_frames_per_block=True,
        )
        == 32
    )
