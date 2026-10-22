from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("family", ["wan", "cosmos3"])
def test_stage3_smoke_reaches_training_entrypoint(monkeypatch, tmp_path, family):
    from tests import stage3_smoke

    def reached_training(args):
        raise RuntimeError("stage3 smoke reached training")

    monkeypatch.setattr(stage3_smoke.entry, "main", reached_training)
    with pytest.raises(RuntimeError, match="stage3 smoke reached training"):
        stage3_smoke.main(tmp_path, family, prepare=False, cp_size=2)


def test_distillation_validation_builds_full_clip_val_datasets(monkeypatch, tmp_path):
    """Stage-3 validation must request num_frames=None (full clips) when
    building the validation datasets, while keeping the run's
    frames_per_block. Behavioral rewrite of the old AST scan over
    distillation.py: distillation.main
    runs against fakes and the real get_val_datasets call is captured."""
    from model_training import distillation, stages
    from model_training.checkpoint_metadata import write_checkpoint_metadata

    teacher_checkpoint = tmp_path / "teacher"
    write_checkpoint_metadata(teacher_checkpoint, {"camera_pose_alignment": "mean"})

    val_dataset_calls = []
    kv_cache_pipe_calls = []
    dataloader_calls = []

    class _FakeAccelerator:
        # A real device object: main() reads accelerator.device.type when
        # building the (here trivial) context-parallel mesh.
        device = torch.device("cpu")

        def prepare(self, *objects):
            assert all(not isinstance(value, str) for value in objects)
            return objects

        def prepare_data_loader(self, dataloader, device_placement=True):
            # Stage 3 prepares the training dataloader without device
            # placement so raw batches stay on CPU.
            assert device_placement is False
            dataloader_calls.append(dataloader)
            return dataloader

    class _FakeDMD:
        def __init__(self, *args, **kwargs):
            self.ema = SimpleNamespace(restore_after_model_load=lambda: None)

        def train(self):
            pass

    pipe = SimpleNamespace(
        vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)),
        transformer=torch.nn.Linear(1, 1),
        ar_window_chunks=None,
        ar_frames_per_chunk=7,
    )

    monkeypatch.setattr(stages, "get_run_id_and_should_resume", lambda args: ("run", False))
    monkeypatch.setattr(stages, "get_accelerator", lambda args, run_id: _FakeAccelerator())
    # main must forward its RESOLVED timestep shift into the pipeline so the
    # noising warp and the rollout scheduler share one shift.
    monkeypatch.setattr(
        stages,
        "get_kv_cache_pipe",
        lambda args, device, *, truncate_rollout_gradients=False: kv_cache_pipe_calls.append(device) or pipe,
    )
    monkeypatch.setattr(
        stages,
        "build_score_transformer",
        lambda args, model_id, pipe, *, gradient_checkpointing: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(stages, "get_train_dataloader", lambda args, accelerator, fpb: "dataloader")
    monkeypatch.setattr(stages, "barrier_if_distributed", lambda: None)
    monkeypatch.setattr(stages, "load_model_weights_from_dcp", lambda model, path: None)
    monkeypatch.setattr(stages, "fsdp2_prepare_model", lambda accelerator, model: model)
    monkeypatch.setattr(stages, "maybe_write_run_id", lambda *call_args: None)
    monkeypatch.setattr(
        stages,
        "get_val_datasets",
        lambda *call_args: val_dataset_calls.append(call_args) or {},
    )
    monkeypatch.setattr(stages, "DMD", _FakeDMD)

    args = SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        model_id_critic=None,
        num_frames=81,  # 21 latent frames, divisible by frames_per_block=7
        frames_per_block=7,
        # Single-rank, full-attention run: main() validates these before
        # building anything, and CP-specific checks short-circuit at size 1.
        context_parallel_size=1,
        local_attn_size=-1,
        sink_size=0,
        gradient_checkpointing=False,
        learning_rate=1e-4,
        learning_rate_critic=1e-4,
        weight_decay=0.0,
        ema_weight=0,
        truncate_rollout_gradients=False,
        project_dir=None,
        base_checkpoint_dir=None,
        base_checkpoint_dir_critic=teacher_checkpoint,
        camera_pose_alignment="mean",
        log_with="none",
    )

    distillation.main(args)

    assert dataloader_calls == ["dataloader"]
    assert len(val_dataset_calls) == 1
    _, _, num_frames, frames_per_block = val_dataset_calls[0]
    # Stage 3 validation must use all frames for full-clip validation.
    assert num_frames is None
    assert frames_per_block == args.frames_per_block
    # The resolved timestep shift reached the pipeline builder.
    assert len(kv_cache_pipe_calls) == 1
