# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from model_training.net.scope import (
    GATE_MOMENT_OFFSET,
    PLUCKER_DIM,
    PLUCKER_SCALE,
    SCOPE_FEATURE_DIM,
    ScopeQKInjection,
    decompose_plucker,
    scope_statistics,
    token_plucker_rays,
)

HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8


def _injection() -> ScopeQKInjection:
    return ScopeQKInjection(HEADS, KV_HEADS, HEAD_DIM)


def _camera_tokens(origins: torch.Tensor, directions: torch.Tensor, window: int) -> torch.Tensor:
    """Repeat each ray over its pixels, storing (position x direction, direction) as the dataset does."""
    rays = torch.cat([torch.linalg.cross(origins, directions), directions], dim=-1)
    return rays.unsqueeze(-1).expand(*rays.shape, window).reshape(*rays.shape[:-1], PLUCKER_DIM * window)


def _reciprocal_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left[..., :3] * right[..., 3:]).sum(-1) + (right[..., :3] * left[..., 3:]).sum(-1)


def test_token_rays_recover_a_unit_direction():
    """Reading the input channels in the wrong order would give a direction whose length is not one."""
    directions = torch.nn.functional.normalize(torch.randn(1, 5, 3), dim=-1)
    rays = token_plucker_rays(_camera_tokens(torch.randn(1, 5, 3) * 3, directions, window=9))
    assert rays[..., :3].norm(dim=-1) == pytest.approx(torch.ones(1, 5), abs=1e-5)


def test_token_rays_keep_direction_orthogonal_to_moment():
    directions = torch.nn.functional.normalize(torch.randn(1, 6, 3), dim=-1)
    rays = token_plucker_rays(_camera_tokens(torch.randn(1, 6, 3) * 5, directions, window=4))
    assert (rays[..., :3] * rays[..., 3:]).sum(-1).abs().max() < 1e-5


def test_identity_projections_recover_the_reciprocal_product():
    """Swapping the two ray parts for keys must match the score calculated directly below."""
    rays = torch.randn(5, PLUCKER_DIM)
    rays[:, :3] = torch.nn.functional.normalize(rays[:, :3], dim=-1)
    query_features, key_features, _ = decompose_plucker(rays)

    identity = torch.eye(SCOPE_FEATURE_DIM)[:PLUCKER_DIM]
    scored = (query_features @ identity.T) @ (key_features @ identity.T).T

    normalised = torch.cat([rays[:, :3], torch.nn.functional.normalize(rays[:, 3:], dim=-1)], dim=-1)
    expected = torch.stack([_reciprocal_product(normalised[i], normalised) for i in range(len(rays))])
    assert scored == pytest.approx(expected, abs=1e-5)


def test_injection_is_exactly_zero_at_initialisation():
    injection = _injection()
    query, key = torch.randn(6, HEADS, HEAD_DIM), torch.randn(6, KV_HEADS, HEAD_DIM)
    new_query, new_key = injection(query, key, torch.randn(6, PLUCKER_DIM))
    assert torch.equal(new_query, query)
    assert torch.equal(new_key, key)


def test_projections_receive_gradient_at_initialisation():
    """The first step must be able to change projection weights, while alpha stays unchanged.

    Starting both at zero would prevent the camera branch from learning.
    """
    injection = _injection()
    new_query, new_key = injection(
        torch.randn(6, HEADS, HEAD_DIM), torch.randn(6, KV_HEADS, HEAD_DIM), torch.randn(6, PLUCKER_DIM)
    )
    (new_query.square().sum() + new_key.square().sum()).backward()
    assert injection.query_projection.weight.grad.abs().sum() > 0
    assert injection.key_projection.weight.grad.abs().sum() > 0
    assert injection.alpha.grad.abs().sum() == 0


def test_initialisation_is_finite_through_the_norm():
    """All-zero projections must still produce finite outputs and gradients."""
    injection = _injection()
    new_query, new_key = injection(
        torch.randn(6, HEADS, HEAD_DIM), torch.randn(6, KV_HEADS, HEAD_DIM), torch.randn(6, PLUCKER_DIM)
    )
    assert torch.isfinite(new_query).all()
    assert torch.isfinite(new_key).all()
    (new_query.square().sum() + new_key.square().sum()).backward()
    assert torch.isfinite(injection.query_projection.weight.grad).all()


def test_injection_moves_qk_once_the_projections_are_non_zero():
    injection = _injection()
    with torch.no_grad():
        injection.query_projection.weight.normal_(0.0, 0.02)
        injection.key_projection.weight.normal_(0.0, 0.02)

    query, key = torch.randn(6, HEADS, HEAD_DIM), torch.randn(6, KV_HEADS, HEAD_DIM)
    new_query, new_key = injection(query, key, torch.randn(6, PLUCKER_DIM))
    assert not torch.equal(new_query, query)
    assert not torch.equal(new_key, key)


# Use the model's head counts and several seeds, rather than one small random sample.
REAL_HEADS, REAL_KV_HEADS, REAL_HEAD_DIM = 32, 8, 128
GATE_SEEDS = (0, 1, 2, 3)


def _real_geometry_injection(seed: int) -> ScopeQKInjection:
    torch.manual_seed(seed)
    return ScopeQKInjection(REAL_HEADS, REAL_KV_HEADS, REAL_HEAD_DIM)


def _saturated_fraction(gate: torch.Tensor) -> float:
    return ((gate < 0.001) | (gate > 0.999)).float().mean().item()


@pytest.mark.parametrize("seed", GATE_SEEDS)
def test_gate_is_neutral_on_average_and_unsaturated_on_real_magnitudes(seed):
    """Initial strength values should stay around 0.5 rather than near 0 or 1."""
    injection = _real_geometry_injection(seed)
    assert injection.query_gate(torch.zeros(4, 1)).detach() == pytest.approx(
        torch.full((4, REAL_HEADS * REAL_HEAD_DIM), 0.5), abs=1e-6
    )

    for log_moment in (-8.0, -4.5, -2.0):
        gate_input = math.log(math.exp(log_moment) + GATE_MOMENT_OFFSET)
        gate = injection.query_gate(torch.full((1, 1), gate_input)).detach()
        assert gate.mean() == pytest.approx(0.5, abs=0.05)
        assert _saturated_fraction(gate) < 0.01, f"gate saturates at |m| = e^{log_moment}"


@pytest.mark.parametrize("seed", GATE_SEEDS)
def test_the_anchor_frames_zero_moment_keeps_the_gate_off_the_rails(seed):
    """Zero position-related ray values must not push the strength values toward 0 or 1."""
    injection = _real_geometry_injection(seed)
    _, _, log_magnitude = decompose_plucker(torch.zeros(1, 1, PLUCKER_DIM))
    assert log_magnitude.item() == pytest.approx(math.log(GATE_MOMENT_OFFSET), abs=1e-6)

    at_eta = injection.query_gate(log_magnitude.reshape(1, 1)).detach()
    assert at_eta.mean() == pytest.approx(0.5, abs=0.05)
    assert _saturated_fraction(at_eta) < 0.01


def test_the_offset_moves_the_zero_input_point_off_unit_magnitude():
    """Adding the offset moves the log's zero from 1 to 1 - GATE_MOMENT_OFFSET."""
    directions = torch.tensor([[[0.0, 0.0, 1.0]]])
    origins = torch.tensor([[[0.0, 1.0 - GATE_MOMENT_OFFSET, 0.0]]])
    _, _, log_magnitude = decompose_plucker(token_plucker_rays(_camera_tokens(origins, directions, window=1)))
    assert log_magnitude.item() == pytest.approx(0.0, abs=1e-6)


def test_every_parameter_is_shardable_by_fsdp():
    """The distributed training code cannot split a zero-dimensional parameter."""
    for name, parameter in _injection().named_parameters():
        assert parameter.dim() >= 1, f"{name} is 0-dimensional and cannot be sharded"


def test_rays_must_line_up_with_the_tokens():
    with pytest.raises(AssertionError, match="one ray per gen token"):
        _injection()(torch.randn(6, HEADS, HEAD_DIM), torch.randn(6, KV_HEADS, HEAD_DIM), torch.randn(5, PLUCKER_DIM))


def test_statistics_report_the_initial_alpha_before_training():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scope = torch.nn.ModuleList([_injection(), _injection()])

    logs = scope_statistics(_Model())
    assert logs["scope/alpha_mean"] == PLUCKER_SCALE
    assert logs["scope/alpha_00"] == PLUCKER_SCALE
    assert scope_statistics(torch.nn.Linear(2, 2)) == {}


def test_gate_input_is_bounded_at_degenerate_tokens():
    """The log input must stay finite for zero position-related ray values and grow for larger values."""
    directions = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    # Camera at the origin, moved along its viewing direction, and moved sideways.
    origins = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0217, 0.0]]])
    rays = token_plucker_rays(_camera_tokens(origins, directions, window=4))
    _, _, log_magnitude = decompose_plucker(rays)

    floor = torch.tensor(GATE_MOMENT_OFFSET).log().item()
    assert log_magnitude.min().item() >= floor - 1e-6
    assert log_magnitude[0, 0, 0].item() == pytest.approx(floor, abs=1e-6)
    assert log_magnitude[0, 1, 0].item() == pytest.approx(floor, abs=1e-6)
    assert log_magnitude[0, 2, 0].item() > floor + 0.5


def test_gate_input_keeps_ordering_above_the_offset():
    """Larger inputs must still produce larger log values."""
    directions = torch.tensor([[[0.0, 0.0, 1.0]] * 4])
    origins = torch.tensor([[[0.0, m, 0.0] for m in (0.01, 0.023, 0.074, 0.154)]])
    _, _, log_magnitude = decompose_plucker(token_plucker_rays(_camera_tokens(origins, directions, window=4)))
    values = log_magnitude.reshape(-1)
    assert torch.all(values[1:] > values[:-1])
