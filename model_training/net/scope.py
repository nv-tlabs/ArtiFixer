# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use camera rays to guide Cosmos3 (SCoPE, arXiv:2606.27345).

Add each token's camera information to its query and key after the base model's
position encoding. Handle the ray's direction and size separately so the model
can control how much camera information it uses.
"""

from __future__ import annotations

import torch
from torch import nn

PLUCKER_DIM = 6
SCOPE_FEATURE_DIM = 7

# Keep the extra parameter count small: each input is a single number.
GATE_HIDDEN_DIM = 128

_EPS = 1e-6

# Prevent log(0) when the position-dependent part of a ray is zero.
GATE_MOMENT_OFFSET = 1e-3

# Start alpha at the value used by the reference implementation.
PLUCKER_SCALE = 1.0


def full_tensor(data: torch.Tensor) -> torch.Tensor:
    """Return the whole tensor, gathering pieces from other workers if needed."""
    return data.full_tensor() if hasattr(data, "full_tensor") else data


def token_plucker_rays(camera_tokens: torch.Tensor) -> torch.Tensor:
    """Average each token's pixels into one ray, returning direction first.

    The input stores ``(o x d, d)``, where ``o`` is camera position and ``d`` is
    viewing direction. Each of the six channels contains its whole pixel window.
    Keep this input order to match existing checkpoints.

    The pixels share a camera position, so averaging preserves ``o x mean(d)``.
    Use the same scale for both parts, avoiding division by zero.
    """
    channels = camera_tokens.shape[-1]
    assert channels % PLUCKER_DIM == 0, f"packed raymap channels {channels} must be a multiple of {PLUCKER_DIM}"

    # Keep the ray calculations in float32 to reduce rounding error.
    packed = camera_tokens.float().reshape(*camera_tokens.shape[:-1], PLUCKER_DIM, -1).mean(dim=-1)
    direction, moment = packed[..., 3:], packed[..., :3]
    scale = direction.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cat([direction / scale, moment / scale], dim=-1)


def decompose_plucker(rays: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build query inputs from the ray direction, position-related values, and their size.

    Swap the first two groups for the keys.
    """
    direction, moment = rays[..., :3], rays[..., 3:]
    magnitude = moment.norm(dim=-1, keepdim=True)
    # Avoid division by zero; keep the original length for the log below.
    unit_moment = moment / magnitude.clamp_min(_EPS)

    log_magnitude = (magnitude + GATE_MOMENT_OFFSET).log()
    query_features = torch.cat([direction, unit_moment, log_magnitude], dim=-1)
    key_features = torch.cat([unit_moment, direction, log_magnitude], dim=-1)
    return query_features, key_features, log_magnitude


class ScopeQKInjection(nn.Module):
    """Add each token's camera information to its query and key.

    Queries and keys can have different head counts, so they use separate strength
    layers. Scale each head separately, matching the base model's grouping.
    """

    def __init__(self, num_attention_heads: int, num_key_value_heads: int, head_dim: int):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        query_dim = num_attention_heads * head_dim
        key_dim = num_key_value_heads * head_dim

        # Start with no camera contribution, preserving the pretrained output.
        self.query_projection = nn.Linear(SCOPE_FEATURE_DIM, query_dim, bias=False)
        self.key_projection = nn.Linear(SCOPE_FEATURE_DIM, key_dim, bias=False)
        nn.init.zeros_(self.query_projection.weight)
        nn.init.zeros_(self.key_projection.weight)

        # Use a length-one tensor: the distributed training code cannot split a scalar parameter.
        self.alpha = nn.Parameter(torch.full((1,), PLUCKER_SCALE))

        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        # Queries and keys have different widths, so each needs its own strength values.
        self.query_gate = self._build_gate(GATE_HIDDEN_DIM, query_dim)
        self.key_gate = self._build_gate(GATE_HIDDEN_DIM, key_dim)

    @staticmethod
    def _build_gate(hidden_dim: int, output_dim: int) -> nn.Sequential:
        gate = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim), nn.Sigmoid())
        # Zero biases give a strength of 0.5 when the log input is zero.
        nn.init.zeros_(gate[0].bias)
        nn.init.zeros_(gate[2].bias)
        return gate

    def _delta(
        self,
        features: torch.Tensor,
        log_magnitude: torch.Tensor,
        projection: nn.Linear,
        norm: nn.RMSNorm,
        gate: nn.Sequential,
        num_heads: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        delta = norm(projection(features.to(dtype)).view(-1, num_heads, self.head_dim))
        return delta * gate(log_magnitude.to(dtype)).view(-1, num_heads, self.head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, rays: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add camera information without changing query or key shapes.

        Queries and keys are [tokens, heads, head_size]; rays are [tokens, 6].
        """
        assert rays.shape[0] == query.shape[0] == key.shape[0], (
            f"SCoPE needs one ray per gen token; got {rays.shape[0]} rays for "
            f"{query.shape[0]} queries and {key.shape[0]} keys"
        )

        query_features, key_features, log_magnitude = decompose_plucker(rays.float())
        query_delta = self._delta(
            query_features,
            log_magnitude,
            self.query_projection,
            self.query_norm,
            self.query_gate,
            self.num_attention_heads,
            query.dtype,
        )
        key_delta = self._delta(
            key_features,
            log_magnitude,
            self.key_projection,
            self.key_norm,
            self.key_gate,
            self.num_key_value_heads,
            key.dtype,
        )
        return query + self.alpha * query_delta, key + self.alpha * key_delta


def scope_statistics(transformer: nn.Module) -> dict[str, float]:
    """Return each layer's alpha for logging, or {} for models without SCoPE.

    All workers must call this together when parameters are split between them.
    """
    injections = getattr(transformer, "scope", None)
    if injections is None:
        return {}

    alphas = [float(full_tensor(injection.alpha)) for injection in injections]
    logs = {f"scope/alpha_{index:02d}": alpha for index, alpha in enumerate(alphas)}
    logs["scope/alpha_mean"] = sum(alphas) / len(alphas)
    return logs
