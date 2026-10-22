from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model_training.trainers.trainer_base import TrainerBase


class _Trainer(TrainerBase):
    def training_batch(self, data):
        raise NotImplementedError

    def to_accumulate(self):
        return []

    def on_before_optimizer_step(self):
        pass

    def on_before_zero_grad(self):
        pass


class _CheckpointAccelerator:
    def __init__(self, project_dir: Path, ack_path: Path, *, fail_save: bool = False):
        self.project_dir = project_dir
        self.project_configuration = SimpleNamespace(iteration=0)
        self.is_main_process = True
        self.process_index = 0
        self.device = torch.device("cpu")
        self._ack_path = ack_path
        self._fail_save = fail_save

    def save_state(self):
        assert not self._ack_path.exists(), "timeout acknowledgment must follow the completed save"
        if self._fail_save:
            raise RuntimeError("injected save failure")
        checkpoint_dir = self.project_dir / "checkpoints" / f"checkpoint_{self.project_configuration.iteration}"
        metadata = checkpoint_dir / "pytorch_model_fsdp_0" / ".metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("complete")

    def print(self, *_args, **_kwargs):
        pass


def _trainer(tmp_path: Path, *, fail_save: bool = False) -> tuple[_Trainer, Path, Path]:
    request_flag = tmp_path / "should_checkpoint"
    request_flag.write_text("request-1460966-0\n")
    ack_path = Path(f"{request_flag}.complete")

    trainer = _Trainer.__new__(_Trainer)
    trainer.args = SimpleNamespace(should_checkpoint_flag=request_flag)
    trainer.accelerator = _CheckpointAccelerator(tmp_path, ack_path, fail_save=fail_save)
    return trainer, request_flag, ack_path


def test_timeout_checkpoint_ack_binds_request_token_to_completed_step(tmp_path):
    trainer, _request_flag, ack_path = _trainer(tmp_path)

    trainer._save_timeout_checkpoint_and_ack(17)

    assert (tmp_path / "checkpoints/checkpoint_17/checkpoint_complete").is_file()
    assert (tmp_path / "checkpoints/checkpoint_17/pytorch_model_fsdp_0/.metadata").is_file()
    assert ack_path.read_text() == "request=request-1460966-0\nstep=17\n"


def test_timeout_checkpoint_failure_does_not_acknowledge_request(tmp_path):
    trainer, _request_flag, ack_path = _trainer(tmp_path, fail_save=True)

    with pytest.raises(RuntimeError, match="injected save failure"):
        trainer._save_timeout_checkpoint_and_ack(17)

    assert not ack_path.exists()


@pytest.mark.parametrize("flag_contents", ["", "\n", "  \n\n"], ids=["empty", "newline", "whitespace"])
def test_timeout_checkpoint_rejects_empty_request_token(tmp_path, flag_contents):
    # A bare `touch`-style flag (the abandoned pre-hardening launcher protocol)
    # carries no token to bind the ack to the requeue authorization; the trainer
    # must fail loudly instead of writing an unbindable "request=" ack.
    trainer, request_flag, ack_path = _trainer(tmp_path)
    request_flag.write_text(flag_contents)

    with pytest.raises(RuntimeError, match="carries no request token"):
        trainer._save_timeout_checkpoint_and_ack(17)

    assert not ack_path.exists()


def test_timeout_checkpoint_does_not_acknowledge_an_already_complete_step(tmp_path):
    trainer, _request_flag, ack_path = _trainer(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints/checkpoint_17"
    (checkpoint_dir / "pytorch_model_fsdp_0").mkdir(parents=True)
    (checkpoint_dir / "pytorch_model_fsdp_0/.metadata").write_text("complete")
    (checkpoint_dir / "checkpoint_complete").touch()

    with pytest.raises(RuntimeError, match="already complete"):
        trainer._save_timeout_checkpoint_and_ack(17)

    assert not ack_path.exists()
