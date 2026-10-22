import pytest

from model_training.utils.train_utils import wandb_resume_mode


@pytest.mark.parametrize("resume_mode", ["never", "must"])
def test_wandb_resume_policy_honors_explicit_environment(monkeypatch, resume_mode):
    monkeypatch.setenv("WANDB_RESUME", resume_mode)

    assert wandb_resume_mode() == resume_mode


def test_wandb_resume_policy_defaults_to_allow(monkeypatch):
    monkeypatch.delenv("WANDB_RESUME", raising=False)

    assert wandb_resume_mode() == "allow"


def test_wandb_resume_policy_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("WANDB_RESUME", "sometimes")

    with pytest.raises(ValueError, match="WANDB_RESUME"):
        wandb_resume_mode()
