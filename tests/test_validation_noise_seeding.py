# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The noise seed must actually reach ``prepare_latents``: every pipeline threads it through a
different chain, and a dropped hop leaves the noise on the global RNG with no visible symptom."""

from __future__ import annotations

import ast
import pathlib

import pytest

PIPELINE_MODULES = (
    "model_training.pipeline.pipeline_base",
    "model_training.pipeline.wan_pipeline",
    "model_training.pipeline.cosmos3_pipeline",
    "model_training.pipeline.wan_kv_cache_pipeline",
    "model_training.pipeline.cosmos3_kv_cache_pipeline",
)
GENERATION_MODULES = ("model_eval.validation", "model_eval.generation")

# Calls that continue the noise chain, and the keyword that carries it.
_CHAIN = {
    "denoise_to_latents": "noise_seed",
    "_denoise_to_latents": "noise_seed",
    "forward_inference_local_cp": "noise_seed",
    "generate_samples_from_batch": "noise_seed",
    "prepare_latents": "generator",
}
# Where evaluation starts and dispatches the chain.
_GENERATION_CALLS = ("generate", "forward_inference", "forward_inference_local_cp", "denoise_to_latents")


def _called_name(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _passes(call: ast.Call, keyword: str) -> bool:
    return any(k.arg == keyword or k.arg is None for k in call.keywords)  # explicit, or via **kwargs


def _module_tree(module_name: str) -> ast.Module:
    module = pytest.importorskip(module_name)
    return ast.parse(pathlib.Path(module.__file__).read_text())


@pytest.mark.parametrize("module_name", PIPELINE_MODULES)
def test_no_inference_path_drops_the_noise_seed(module_name):
    """Every chain call inside a seed-taking function must pass the seed onward."""
    dropped = []
    for function in ast.walk(_module_tree(module_name)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = function.args
        if not any(a.arg == "noise_seed" for a in args.args + args.kwonlyargs):
            continue
        for call in ast.walk(function):
            if not isinstance(call, ast.Call):
                continue
            keyword = _CHAIN.get(_called_name(call))
            if keyword and not _passes(call, keyword):
                dropped.append(f"{function.name} -> {_called_name(call)} (line {call.lineno})")

    assert not dropped, "noise seed dropped on: " + ", ".join(dropped)


@pytest.mark.parametrize("module_name", GENERATION_MODULES)
def test_every_validation_generation_call_is_seeded(module_name):
    """Validation generates from several call sites (plain, local-CP, DMD's KV-cache path)."""
    calls = [
        call
        for call in ast.walk(_module_tree(module_name))
        if isinstance(call, ast.Call) and _called_name(call) in _GENERATION_CALLS
    ]
    assert calls, "Do not silently stop checking a moved generation path"
    unseeded = [f"{_called_name(call)} (line {call.lineno})" for call in calls if not _passes(call, "noise_seed")]
    assert not unseeded, "unseeded validation generation: " + ", ".join(unseeded)


def test_seeded_and_unseeded_draws_differ_and_repeat():
    import torch

    from model_training.pipeline.pipeline_base import ArtifixerPipelineBase

    device = torch.device("cpu")
    first = torch.randn((2, 3, 4), generator=ArtifixerPipelineBase._noise_generator(7, device))
    second = torch.randn((2, 3, 4), generator=ArtifixerPipelineBase._noise_generator(7, device))
    other = torch.randn((2, 3, 4), generator=ArtifixerPipelineBase._noise_generator(8, device))

    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert ArtifixerPipelineBase._noise_generator(None, device) is None  # training keeps the global stream


def test_kv_cache_rollouts_build_the_generator_on_the_vae_device():
    """The Wan KV path hands ``generate_samples_from_batch`` a CPU-resident condition; the noise is
    drawn on the VAE device, and a generator on the wrong device makes ``torch.randn`` raise."""
    for module_name in (
        "model_training.pipeline.wan_kv_cache_pipeline",
        "model_training.pipeline.cosmos3_kv_cache_pipeline",
    ):
        source = pathlib.Path(pytest.importorskip(module_name).__file__).read_text()
        assert "self._noise_generator(noise_seed, self.vae.device)" in source, module_name
        assert "self._noise_generator(noise_seed, condition.device)" not in source, module_name
