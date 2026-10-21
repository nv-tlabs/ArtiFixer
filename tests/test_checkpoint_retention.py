# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from model_training.utils import train_utils
from tests.test_timeout_checkpoint_ack import _Trainer
from tests.test_training_startup_validation import _args


def _trainer(tmp_path, limit=1):
    accelerator = Accelerator(
        cpu=True,
        project_config=ProjectConfiguration(project_dir=tmp_path, automatic_checkpoint_naming=True),
    )
    model = accelerator.prepare(torch.nn.Linear(2, 2))
    trainer = _Trainer.__new__(_Trainer)
    trainer.args = SimpleNamespace(max_checkpoints=limit)
    trainer.accelerator = accelerator
    return trainer, model


def test_failed_save_keeps_the_previous_loadable_checkpoint(tmp_path, isolated_accelerator_state):
    trainer, model = _trainer(tmp_path)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    trainer._save_state_with_commit_marker(1)

    def fail_save(*args):
        raise OSError("injected write failure")

    handle = trainer.accelerator.register_save_state_pre_hook(fail_save)
    with pytest.raises(OSError, match="injected write failure"):
        trainer._save_state_with_commit_marker(2)
    handle.remove()
    previous = tmp_path / "checkpoints/checkpoint_1"
    assert (previous / "checkpoint_complete").is_file()
    assert not (tmp_path / "checkpoints/checkpoint_2").exists()
    with torch.no_grad():
        model.weight.zero_()
    trainer.accelerator.load_state(previous)
    torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)

    trainer._save_state_with_commit_marker(2)
    assert not previous.exists()
    assert (tmp_path / "checkpoints/checkpoint_2/checkpoint_complete").is_file()


def test_accelerator_does_not_apply_retention_before_saving(monkeypatch, tmp_path):
    captured = {}

    class StopBeforeModelSetup(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise StopBeforeModelSetup

    monkeypatch.setattr(train_utils, "Accelerator", capture)
    monkeypatch.setattr(train_utils, "FullyShardedDataParallelPlugin", lambda **kwargs: None)
    monkeypatch.setattr(train_utils, "ParallelismConfig", lambda **kwargs: None)
    args = SimpleNamespace(
        gradient_accumulation_steps=1,
        model_id="nvidia/Cosmos3-Nano",
        shard_size=1,
        log_with="none",
        project_dir=tmp_path,
        max_checkpoints=1,
        process_group_timeout_minutes=5,
    )
    with pytest.raises(StopBeforeModelSetup):
        train_utils.get_accelerator(args, None)
    assert captured["project_config"].total_limit is None


def test_retention_orders_steps_numerically_and_preserves_other_entries(tmp_path, isolated_accelerator_state):
    trainer, _ = _trainer(tmp_path, limit=2)
    root = tmp_path / "checkpoints"
    for step in (2, 9, 10):
        path = root / f"checkpoint_{step}"
        path.mkdir(parents=True)
        (path / "checkpoint_complete").touch()
    (root / "checkpoint_3").mkdir()  # incomplete save
    (root / "checkpoint_notes").mkdir()
    (root / "checkpoint_notes/checkpoint_complete").touch()
    external = tmp_path / "external"
    external.mkdir()
    (external / "checkpoint_complete").touch()
    (root / "checkpoint_1").symlink_to(external, target_is_directory=True)
    trainer._prune_completed_checkpoints(10)
    assert not (root / "checkpoint_2").exists()
    assert (root / "checkpoint_9/checkpoint_complete").is_file()
    assert (root / "checkpoint_10/checkpoint_complete").is_file()
    assert (root / "checkpoint_3").is_dir()
    assert (root / "checkpoint_notes").is_dir()
    assert (root / "checkpoint_1").is_symlink()
    assert external.is_dir()


def test_saving_an_older_step_does_not_delete_the_new_save_or_future_history(tmp_path, isolated_accelerator_state):
    trainer, _ = _trainer(tmp_path)
    trainer._save_state_with_commit_marker(10)
    trainer._save_state_with_commit_marker(2)
    for step in (2, 10):
        assert (tmp_path / f"checkpoints/checkpoint_{step}/checkpoint_complete").is_file()


def test_default_dmd_run_retains_every_scheduled_checkpoint(tmp_path, isolated_accelerator_state):
    args = _args()
    trainer, model = _trainer(tmp_path, limit=args.max_checkpoints)
    steps = list(range(args.save_steps, args.max_iterations + 1, args.save_steps))
    assert steps == list(range(100, 1001, 100))
    for step in steps:
        with torch.no_grad():
            model.weight.fill_(step)
        trainer._save_state_with_commit_marker(step)

    # The final save repeats the last periodic step, without consuming another slot.
    trainer._save_state_with_commit_marker(args.max_iterations)
    for step in steps:
        checkpoint = tmp_path / f"checkpoints/checkpoint_{step}"
        assert (checkpoint / "checkpoint_complete").is_file()
        trainer.accelerator.load_state(checkpoint)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, step), rtol=0, atol=0)
