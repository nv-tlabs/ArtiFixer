from types import SimpleNamespace

import pytest
import torch

from model_training.net.context_parallel import clear_context_parallel_state, set_context_parallel_state
from model_training.pipeline.pipeline_base import (
    create_denoising_step_list,
    generate_and_sync_exit_flag,
    prepare_transformer_for_training,
)


class _BackendModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.backend = None

    def set_attention_backend(self, backend: str) -> None:
        self.backend = backend


def test_prepare_transformer_for_training_sets_mode_and_backend_without_changing_grad_flags():
    module = _BackendModule()
    module.eval()
    module.requires_grad_(False)

    prepare_transformer_for_training(module, "_flash_3")

    assert module.training
    assert not any(param.requires_grad for param in module.parameters())
    assert module.backend == "_flash_3"


def test_prepare_transformer_for_training_skips_backend_setter_when_backend_is_none():
    module = _BackendModule()
    module.eval()

    prepare_transformer_for_training(module, None)

    assert module.training
    assert module.backend is None


def test_prepare_transformer_for_training_always_requires_backend_setter():
    # A requested backend on a transformer without set_attention_backend
    # fails loudly instead of silently training on the default backend.
    # Test fakes provide a no-op setter.
    module = torch.nn.Linear(1, 1)

    with pytest.raises(AttributeError, match="set_attention_backend"):
        prepare_transformer_for_training(module, "_flash_3")


def test_context_parallel_helpers_set_and_clear_state():
    module = SimpleNamespace(_cp_mesh=None, _cp_rank=0, _cp_world_size=1)
    mesh = SimpleNamespace(get_local_rank=lambda: 2, size=lambda: 8)

    set_context_parallel_state(module, mesh)

    assert module._cp_mesh is mesh
    assert module._cp_rank == 2
    assert module._cp_world_size == 8

    clear_context_parallel_state(module)

    assert module._cp_mesh is None
    assert module._cp_rank == 0
    assert module._cp_world_size == 1


def test_generate_and_sync_exit_flag_single_process_draws_a_valid_step_index():
    """The single-process (torch.distributed uninitialized) path both
    pipelines' training rollouts rely on: rank 0 draws the early-exit step as
    randint(0, num_inference_steps), returned as a plain Python int."""
    import torch.distributed as dist

    assert not dist.is_initialized()

    num_inference_steps = 4
    flags = [generate_and_sync_exit_flag(num_inference_steps, device="cpu") for _ in range(64)]

    assert all(type(flag) is int for flag in flags)
    assert all(0 <= flag < num_inference_steps for flag in flags)
    # The draw comes from torch's generator, so a pinned seed pins the flags.
    torch.manual_seed(1234)
    first = [generate_and_sync_exit_flag(num_inference_steps, device="cpu") for _ in range(8)]
    torch.manual_seed(1234)
    second = [generate_and_sync_exit_flag(num_inference_steps, device="cpu") for _ in range(8)]
    assert first == second


def test_create_denoising_step_list_uses_scheduler_timestep_indexing():
    scheduler = SimpleNamespace(
        num_train_timesteps=1000,
        timesteps=torch.linspace(999, 0, 1000),
    )

    steps = create_denoising_step_list(scheduler, num_inference_steps=4, device=torch.device("cpu"))

    assert steps.tolist() == [999.0, 749.0, 499.0, 249.0]


def test_create_denoising_step_list_uses_scheduler_train_timestep_count():
    scheduler = SimpleNamespace(
        num_train_timesteps=10,
        timesteps=torch.linspace(9, 0, 10),
    )

    steps = create_denoising_step_list(scheduler, num_inference_steps=2, device=torch.device("cpu"))

    assert steps.tolist() == [9.0, 4.0]
