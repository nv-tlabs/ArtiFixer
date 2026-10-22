# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Context-parallel attention tests for artifixer.

The CPU tests drive the production ``ArtifixerSelfAttnProcessor`` Ulysses CP
path (all-to-all over heads/sequence, global block mask) with a fake CP mesh
and a monkeypatched collective -- no GPU or torchrun required:
    pytest tests/test_cp_attention.py -v

They also pin the grad-carrying KV-cache CP gather that DMD rollouts backprop
through: ``all_gather_cat_with_grad`` itself and the grad-enabled CP branch of
the production ``KvCacheWanSelfAttnProcessor``.

The tests at the bottom of the file need REAL collectives and run only under
torchrun with >= 2 GPUs — tests/run_cp_attention_tests.sh launches exactly
this file:
    torchrun --nproc_per_node=2 -m pytest tests/test_cp_attention.py -v
They pin (i) the C-times reduce-scatter gradient factor behind the CP
training no-rescale decision, which single-process collective fakes CANNOT
pin (a second backward through the one shared fake graph no-ops), and (ii)
cosmos3's CP=2 transformer forward+backward equivalence against CP=1.
"""

import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from diffusers import WanTransformer3DModel
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
from torch import nn
from torch.distributed import DeviceMesh

import model_training.common.context_parallel as common_cp_mod
import model_training.net.attention_config as attention_config
import model_training.net.cosmos3_transformer as cosmos3_mod
import model_training.net.wan_transformer as transformer_mod
from model_training.common.context_parallel import all_gather_cat_with_grad, frame_block_causal_mask
from model_training.net.context_parallel import transformer_cp_head_counts
from model_training.net.rollout_kv_cache import RolloutKvCache
from model_training.net.wan_transformer import (
    ArtifixerSelfAttnProcessor,
    ArtifixerTransformer,
    KvCacheWanSelfAttnProcessor,
)
from model_training.utils import train_utils
from tests._helpers import (
    REPO_ROOT,
    VAE_SCALE_FACTOR_TEMPORAL,
    _FakeCpMesh,
    _growing_prefix_kwargs,
    _reduced_real_transformer,
    _TinyCosmosBaseTransformer,
    _TinyCosmosVae,
)


@pytest.fixture
def attention_runtime(monkeypatch):
    runtime = SimpleNamespace(
        major=9,
        cuda_available=True,
        impls=("FA4",),
        fa3_installed=True,
        device=None,
        capability_devices=[],
        activations=[],
    )

    @contextmanager
    def device_context(device):
        previous_device = runtime.device
        runtime.device = torch.device(device)
        try:
            yield
        finally:
            runtime.device = previous_device

    def capability(device):
        assert torch.device(device) == runtime.device
        runtime.capability_devices.append(torch.device(device))
        return runtime.major, 0

    def activate(impl):
        assert runtime.device is not None
        runtime.activations.append((impl, runtime.device))

    original_find_spec = attention_config.importlib.util.find_spec

    def find_spec(name, package=None):
        if name == "flash_attn_interface":
            return ModuleSpec(name, loader=None) if runtime.fa3_installed else None
        return original_find_spec(name, package)

    monkeypatch.setattr(attention_config, "_FLEX_KERNEL_OPTIONS", {})
    monkeypatch.setattr(attention_config, "_FLASH_3_PATCHED", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: runtime.cuda_available)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(attention_config, "list_flash_attention_impls", lambda: runtime.impls)
    monkeypatch.setattr(attention_config, "activate_flash_attention_impl", activate)
    monkeypatch.setattr(attention_config.importlib.util, "find_spec", find_spec)
    return runtime


@pytest.mark.parametrize("cuda_available", [False, True])
def test_attention_imports_leave_gpu_policy_unconfigured(cuda_available):
    script = textwrap.dedent(
        """
        import argparse
        import sys
        from importlib.machinery import ModuleSpec
        from types import ModuleType

        import torch
        import torch._dynamo
        import torch.nn.attention as torch_attention
        from diffusers import Cosmos3OmniTransformer, WanTransformer3DModel
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import get_3d_mrope_ids_text_tokens
        from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

        def forbidden(*args, **kwargs):
            raise AssertionError("model/parser import must not configure GPU attention")

        class ReadOnlyRegistry(dict):
            __setitem__ = forbidden
            update = forbidden

        registry_before = dict(_AttentionBackendRegistry._backends)
        registry = ReadOnlyRegistry(registry_before)
        _AttentionBackendRegistry._backends = registry
        torch.cuda.is_available = lambda: sys.argv[1] == "True"
        torch.cuda.get_device_capability = forbidden
        torch.cuda.set_device = forbidden
        torch.cuda._lazy_init = forbidden
        torch_attention.list_flash_attention_impls = forbidden
        torch_attention.activate_flash_attention_impl = forbidden
        fa3 = ModuleType("flash_attn_interface")
        fa3.__spec__ = ModuleSpec(fa3.__name__, loader=None)
        fa3.flash_attn_func = forbidden
        fa3._flash_attn_backward = forbidden
        sys.modules[fa3.__name__] = fa3

        assert "model_training.net.attention_config" not in sys.modules
        from model_training.net import attention_config
        assert attention_config.flex_block_size() == 128
        assert not attention_config._FLASH_3_PATCHED
        attention_config.configure_attention = forbidden

        import model_training.net.cosmos3_transformer
        assert "model_training.net.wan_transformer" not in sys.modules
        import model_training.net.wan_transformer
        from model_training.arguments import add_common_runtime_opts
        from model_training.utils import train_utils
        parser = argparse.ArgumentParser()
        add_common_runtime_opts(parser)
        assert parser.parse_args([]).attention_backend is None
        assert attention_config.flex_block_size() == 128
        assert not attention_config._FLASH_3_PATCHED
        assert _AttentionBackendRegistry._backends is registry
        assert registry == registry_before
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(cuda_available)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "major,impls,fa3_installed,fa3_importable,default_backend",
    [
        (7, ("FA4",), True, True, None),
        (8, ("FA4",), True, True, None),
        (9, ("FA4",), True, True, "_flash_3"),
        (9, (), True, True, "_flash_3"),
        (9, ("FA4",), False, False, None),
        (9, ("FA4",), True, False, None),
        (9, (), False, False, None),
        (10, ("FA4",), True, True, None),
        (12, (), False, False, None),
    ],
)
@pytest.mark.parametrize("override", [None, "native"])
def test_attention_configuration_selects_backend_and_matching_flex_mask(
    monkeypatch, attention_runtime, major, impls, fa3_installed, fa3_importable, default_backend, override
):
    runtime = attention_runtime
    runtime.major = major
    runtime.impls = impls
    runtime.fa3_installed = fa3_installed
    patch_backend = Mock(return_value=fa3_importable)
    monkeypatch.setattr(attention_config, "_patch_flash_3_backend", patch_backend)
    device = torch.device("cuda:3")

    backend = attention_config.configure_attention(device, attention_backend=override)

    assert backend == (default_backend if override is None else override)
    assert runtime.capability_devices == [device]
    assert runtime.device is None, "setup must restore the caller's active device context"
    use_flash = major >= 9 and "FA4" in impls
    assert runtime.activations == ([("FA4", device)] if use_flash else [])
    assert patch_backend.call_count == int(major == 9 and fa3_installed)
    assert attention_config.flex_block_size() == ((256, 128) if use_flash else 128)

    flex_kernel = Mock(return_value=object())
    monkeypatch.setattr(attention_config, "_flex_attention_orig", flex_kernel)
    mask = object()
    output = attention_config.flex_attention_with_options("q", "k", "v", block_mask=mask, enable_gqa=True)
    assert output is flex_kernel.return_value
    flex_kernel.assert_called_once_with(
        "q", "k", "v", block_mask=mask, enable_gqa=True, kernel_options={"BACKEND": "FLASH"} if use_flash else {}
    )


@pytest.mark.parametrize("device,cuda_available", [("cpu", False), ("cpu", True), ("cuda:3", False)])
@pytest.mark.parametrize("override", [None, "native"])
def test_attention_configuration_cpu_path_never_touches_gpu_setup(monkeypatch, device, cuda_available, override):
    unavailable_call = Mock(side_effect=AssertionError("CPU setup must not query or configure GPU attention"))
    available = Mock(return_value=cuda_available)
    monkeypatch.setattr(attention_config, "_FLEX_KERNEL_OPTIONS", {"BACKEND": "FLASH"})
    monkeypatch.setattr(torch.cuda, "is_available", available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unavailable_call)
    monkeypatch.setattr(torch.cuda, "device", unavailable_call)
    monkeypatch.setattr(attention_config, "list_flash_attention_impls", unavailable_call)
    monkeypatch.setattr(attention_config, "activate_flash_attention_impl", unavailable_call)
    monkeypatch.setattr(attention_config, "_patch_flash_3_backend", unavailable_call)

    assert attention_config.configure_attention(device, attention_backend=override) == override
    assert attention_config.flex_block_size() == 128
    unavailable_call.assert_not_called()
    assert available.call_count == int(torch.device(device).type == "cuda")


def test_attention_configuration_keeps_flex_flash_when_aten_activation_fails(monkeypatch, attention_runtime):
    monkeypatch.setattr(
        attention_config, "activate_flash_attention_impl", Mock(side_effect=RuntimeError("aten override unavailable"))
    )
    monkeypatch.setattr(attention_config, "_patch_flash_3_backend", Mock(return_value=True))

    assert attention_config.configure_attention("cuda:3") == "_flash_3"
    assert attention_config.flex_block_size() == (256, 128)


def test_attention_configuration_installs_idempotent_gradient_capable_fa3_patch(monkeypatch, attention_runtime):
    fa3 = ModuleType("flash_attn_interface")
    backward_calls = []

    def forward(q, k, v, softmax_scale, causal, return_attn_probs):
        assert return_attn_probs
        return q + 2 * k + 3 * v, q.new_zeros((q.shape[0], q.shape[2], q.shape[1]))

    def backward(*args, **kwargs):
        assert not kwargs, "FA3 backward buffers must be passed positionally"
        assert len(args) == 22
        grad_out = args[0]
        assert grad_out.is_contiguous()
        assert args[6:12] == (None,) * 6
        assert args[15:] == (0.25, True, -1, -1, 0.0, False, 0)
        for gradient, multiplier in zip(args[12:15], (1, 2, 3), strict=True):
            gradient.copy_(grad_out * multiplier)
        backward_calls.append(grad_out)

    fa3.flash_attn_func = forward
    fa3._flash_attn_backward = backward
    monkeypatch.setitem(sys.modules, fa3.__name__, fa3)
    original_backend = object()
    registry = {AttentionBackendName._FLASH_3: original_backend}
    monkeypatch.setattr(_AttentionBackendRegistry, "_backends", registry)
    allow_in_graph = Mock(side_effect=lambda function: function)
    monkeypatch.setattr(torch.compiler, "allow_in_graph", allow_in_graph)

    assert attention_config.configure_attention("cuda:3") == "_flash_3"
    backend = registry[AttentionBackendName._FLASH_3]
    assert backend is not original_backend
    assert attention_config.configure_attention("cuda:3") == "_flash_3"
    assert registry[AttentionBackendName._FLASH_3] is backend
    allow_in_graph.assert_called_once()

    q, k, v = [torch.randn(1, 3, 2, 4, requires_grad=True) for _ in range(3)]
    output, lse = backend(q, k, v, scale=0.25, is_causal=True, return_lse=True)
    torch.testing.assert_close(output, q + 2 * k + 3 * v)
    assert lse.shape == (1, 2, 3)
    output.sum().backward()
    assert len(backward_calls) == 1
    for tensor, multiplier in zip((q, k, v), (1, 2, 3), strict=True):
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, multiplier))
    assert isinstance(backend(q, k, v), torch.Tensor)
    with pytest.raises(ValueError, match="attn_mask.*not supported"):
        backend(q, k, v, attn_mask=torch.ones(1))


def test_attention_configuration_falls_back_if_fa3_cannot_be_imported(monkeypatch, attention_runtime):
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    registry_before = dict(_AttentionBackendRegistry._backends)

    assert attention_config.configure_attention("cuda:3") is None
    assert _AttentionBackendRegistry._backends == registry_before
    assert not attention_config._FLASH_3_PATCHED
    assert attention_config.flex_block_size() == (256, 128)


@pytest.mark.parametrize("cosmos3", [False, True])
@pytest.mark.parametrize("kv_cache", [False, True])
@pytest.mark.parametrize("override", [None, "native"])
def test_pipeline_factories_initialize_attention_for_runtime_device(
    monkeypatch, attention_runtime, cosmos3, kv_cache, override
):
    device = torch.device("cuda:3")
    monkeypatch.setattr(attention_config, "_patch_flash_3_backend", Mock(return_value=True))
    monkeypatch.setattr(train_utils, "is_cosmos3_model", lambda model_id: cosmos3)
    monkeypatch.setattr(train_utils, "_transformer_from_config", lambda *args: object())
    monkeypatch.setattr(train_utils, "_vae_for_model", lambda *args: object())
    monkeypatch.setattr(train_utils, "_scheduler_for_model", lambda *args: object())
    monkeypatch.setattr(train_utils, "uses_expanded_timesteps", lambda model_id: False)
    monkeypatch.setattr(train_utils, "flow_shift_for_model", lambda model_id: 5.0)
    monkeypatch.setattr(train_utils, "validate_model_profile", lambda *args: None)
    constructed = []
    result = object()

    def build_pipeline(**kwargs):
        assert attention_runtime.capability_devices == [device], "initialize once, before constructing the wrapper"
        assert attention_config.flex_block_size() == (256, 128)
        constructed.append(kwargs)
        return result

    pipeline_name = {
        (False, False): "ArtifixerPipeline",
        (False, True): "ArtifixerKvCachePipeline",
        (True, False): "ArtifixerCosmos3Pipeline",
        (True, True): "ArtifixerCosmos3KvCachePipeline",
    }[cosmos3, kv_cache]
    monkeypatch.setattr(train_utils, pipeline_name, build_pipeline)
    args = SimpleNamespace(
        model_id="test-model",
        attention_backend=override,
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
        default_negative_prompt_path=None,
        frames_per_block=1,
        local_attn_size=-1,
        sink_size=0,
    )

    if kv_cache:
        pipe = train_utils.get_kv_cache_pipe(args, device)
    else:
        pipe = train_utils.get_pipe(args, load_pretrained_transformer_weights=False, frames_per_block=1, device=device)

    assert pipe is result
    assert len(constructed) == 1
    assert constructed[0]["attention_backend"] == ("_flash_3" if override is None else override)


@pytest.mark.parametrize("cosmos3", [False, True])
@pytest.mark.parametrize("kv_cache", [False, True])
@pytest.mark.parametrize("backend", [None, "native"])
def test_pipeline_constructors_keep_resolved_attention_backend(monkeypatch, tmp_path, cosmos3, kv_cache, backend):
    model_id = "nvidia/Cosmos3-Nano" if cosmos3 else "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    transformer = _TinyCosmosBaseTransformer() if cosmos3 else WanTransformer3DModel.from_config(_REDUCED_WAN_CONFIG)
    vae = _TinyCosmosVae()
    vae.config = SimpleNamespace(**(vars(vae.config) | {"scale_factor_spatial": 16 if cosmos3 else 8}))
    monkeypatch.setattr(train_utils, "_transformer_from_config", lambda *args: transformer)
    monkeypatch.setattr(train_utils, "_vae_for_model", lambda *args: vae)
    monkeypatch.setattr(train_utils, "_scheduler_for_model", lambda *args: SimpleNamespace())
    monkeypatch.setattr(attention_config, "_FLEX_KERNEL_OPTIONS", {})
    negative_prompt = tmp_path / "negative.pt"
    torch.save(torch.zeros(1, 4, 32), negative_prompt)
    args = SimpleNamespace(
        model_id=model_id,
        attention_backend=backend,
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
        default_negative_prompt_path=negative_prompt,
        frames_per_block=1,
        local_attn_size=-1,
        sink_size=0,
    )

    pipe = (
        train_utils.get_kv_cache_pipe(args, "cpu")
        if kv_cache
        else train_utils.get_pipe(args, load_pretrained_transformer_weights=False, frames_per_block=None, device="cpu")
    )

    assert pipe.attention_backend == backend


# ---------------------------------------------------------------------------
# CPU fakes (same doubles as tests/test_kv_cache_pipeline.py)
# ---------------------------------------------------------------------------


class _GroupedFakeCpMesh(_FakeCpMesh):
    """tests/test_kv_cache_pipeline.py double: the Ulysses exchange keys its
    reverse pass on a non-None process group, so this mesh must hand one out."""

    _group = object()

    def get_group(self):
        return self._group


class _FakeWanAttention(nn.Module):
    """tests/test_kv_cache_pipeline.py fake WanAttention module."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Identity()])


def _patch_qkv_projections(monkeypatch):
    """Route diffusers' _get_qkv_projections through the fake module's projections."""

    def fake_get_qkv_projections(attn, hidden_states, encoder_hidden_states):
        return attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)

    monkeypatch.setattr(transformer_mod, "_get_qkv_projections", fake_get_qkv_projections)


def _sdpa_dispatch(query, key, value, **kwargs):
    """Real attention math in the (batch, seq, heads, head_dim) dispatch layout."""
    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)


def _batch_as_rank_all_to_all(world_size):
    """Exact single-process stand-in for the Ulysses all-to-all.

    The batch dimension indexes the CP rank: entry ``r`` of the input tensor is
    what rank ``r`` holds locally.  As in the real collective, rank ``r``
    receives scatter-chunk ``r`` from every peer, concatenated along the gather
    dimension, and the result is returned in the same batch-as-rank layout.
    """

    def fake_all_to_all_with_grad(tensor, scatter_dim, gather_dim, group=None):
        assert group is not None, "the production CP path always exchanges through the mesh group"
        chunks = tensor.chunk(world_size, dim=scatter_dim)
        per_rank = [
            torch.cat([chunks[rank][peer : peer + 1] for peer in range(world_size)], dim=gather_dim)
            for rank in range(world_size)
        ]
        return torch.cat(per_rank, dim=0)

    return fake_all_to_all_with_grad


# ---------------------------------------------------------------------------
# Production Ulysses CP self-attention path (CPU)
# ---------------------------------------------------------------------------


def test_ulysses_cp_matches_non_cp_attention_on_the_same_sequence(monkeypatch):
    """The head-shard/sequence-gather round trip of the production
    ArtifixerSelfAttnProcessor must reproduce, per rank, exactly the local
    slice of the non-CP output over the full sequence, given identical weights
    and inputs.  Rank shards differ, so chunk-routing mistakes in either
    all-to-all (wrong scatter/gather dim, wrong peer order) change values."""
    world_size, tokens_per_rank, dim, heads = 2, 6, 32, 4

    _patch_qkv_projections(monkeypatch)
    monkeypatch.setattr(transformer_mod, "dispatch_attention_fn", _sdpa_dispatch)
    monkeypatch.setattr(transformer_mod, "all_to_all_with_grad", _batch_as_rank_all_to_all(world_size))

    torch.manual_seed(0)
    attn = _FakeWanAttention(dim, heads)
    hidden_global = torch.randn(1, world_size * tokens_per_rank, dim)
    local_shards = torch.cat(hidden_global.chunk(world_size, dim=1), dim=0)

    processor = ArtifixerSelfAttnProcessor()
    processor.set_context_parallel_mesh(_GroupedFakeCpMesh(size=world_size, rank=0))
    cp_out = processor(attn, local_shards, None, None, None, None)

    processor.set_context_parallel_mesh(None)
    reference = processor(attn, hidden_global, None, None, None, None)

    assert cp_out.shape == (world_size, tokens_per_rank, dim)
    torch.testing.assert_close(cp_out.flatten(0, 1).unsqueeze(0), reference)


def test_ulysses_cp_rejects_heads_not_divisible_by_cp_size(monkeypatch):
    """The Ulysses exchange shards attention heads across the CP group, so a
    group size that does not divide the head count must be rejected up front."""
    _patch_qkv_projections(monkeypatch)

    processor = ArtifixerSelfAttnProcessor()
    processor.set_context_parallel_mesh(_GroupedFakeCpMesh(size=3, rank=0))

    with pytest.raises(AssertionError, match="must be divisible by CP size"):
        processor(_FakeWanAttention(8, heads=4), torch.randn(1, 5, 8), None, None, None, None)


def test_ulysses_cp_forwards_the_global_block_mask_unchanged(monkeypatch):
    """Under CP the caller builds one mask for the gathered (global) sequence
    and the processor hands that exact object to FlexAttention: the old
    per-rank mask rewrite is gone from the shipped design.  Output stays a
    local sequence shard with all heads."""
    world_size, frames_per_rank, frame_seqlen, dim, heads = 2, 4, 2, 32, 4

    _patch_qkv_projections(monkeypatch)
    monkeypatch.setattr(transformer_mod, "all_to_all_with_grad", _batch_as_rank_all_to_all(world_size))
    seen_masks = []

    def recording_flex_attention(query, key, value, block_mask=None):
        seen_masks.append(block_mask)
        return query

    monkeypatch.setattr(transformer_mod, "flex_attention", recording_flex_attention)

    # The global mask exactly as ArtifixerTransformer.forward builds it under CP.
    block_mask = frame_block_causal_mask(
        device=torch.device("cpu"),
        num_frames=world_size * frames_per_rank,
        frame_seqlen=frame_seqlen,
        frames_per_block=frames_per_rank,
        block_size=attention_config.flex_block_size(),
        attention_heads=heads // world_size,
    )

    torch.manual_seed(0)
    attn = _FakeWanAttention(dim, heads)
    hidden = torch.randn(world_size, frames_per_rank * frame_seqlen, dim)

    processor = ArtifixerSelfAttnProcessor()
    processor.set_context_parallel_mesh(_GroupedFakeCpMesh(size=world_size, rank=0))
    out = processor(attn, hidden, None, None, None, block_mask)

    assert len(seen_masks) == 1
    assert seen_masks[0] is block_mask
    assert out.shape == hidden.shape


# ---------------------------------------------------------------------------
# Grad-carrying KV-cache CP gather (CPU)
# ---------------------------------------------------------------------------


def test_all_gather_cat_with_grad_routes_upstream_grad_slices(monkeypatch):
    """all_gather_cat_with_grad must stay autograd-aware: with the collective
    faked as an in-process concatenation, one backward through the gathered
    tensor must route each upstream-grad slice to the tensor occupying that
    rank slot — the caller's leaf gets exactly its own slice, the peer leaves
    theirs.  A detach, or a switch to the non-differentiable dist.all_gather,
    drops exactly that credit.  What this fake CANNOT pin is the cross-rank
    reduce-scatter SUMMATION (each rank backwards its own loss copy; a second
    backward through this one shared graph would no-op) — that C-times factor
    is pinned on real collectives by
    test_all_gather_cat_with_grad_real_backward_gives_world_size_times_slice_grad
    below, which runs under torchrun only."""
    world_size, tokens, channels = 2, 3, 4
    caller_rank = 0
    cp_group = object()

    monkeypatch.setattr(common_cp_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(common_cp_mod.dist, "get_world_size", lambda group=None: world_size)

    torch.manual_seed(0)
    rank_inputs = [torch.randn(1, tokens, channels, requires_grad=True) for _ in range(world_size)]
    upstream_grad = torch.randn(1, world_size * tokens, channels)

    def fake_all_gather(tensor, group=None):
        # In-process stand-in for torch.distributed.nn's all_gather: the
        # caller's slot is filled from the tensor it actually handed the
        # collective (so a detach upstream of the exchange breaks the graph
        # exactly as it would for real), peer slots from the peer leaves.
        assert group is cp_group, "the gather must run on the caller's CP group"
        slots = list(rank_inputs)
        slots[caller_rank] = tensor
        return slots

    monkeypatch.setattr(common_cp_mod.dist_nn_func, "all_gather", fake_all_gather)

    gathered = all_gather_cat_with_grad(rank_inputs[caller_rank], dim=1, group=cp_group)
    torch.testing.assert_close(gathered, torch.cat(rank_inputs, dim=1))
    assert gathered.requires_grad, "the gathered sequence must stay on the autograd graph"

    gathered.backward(upstream_grad)
    for peer, leaf in enumerate(rank_inputs):
        torch.testing.assert_close(leaf.grad, upstream_grad[:, peer * tokens : (peer + 1) * tokens])


def test_kv_cache_cp_gather_carries_grad_to_kv_sources(monkeypatch):
    """A grad-enabled CP forward of the real KvCacheWanSelfAttnProcessor is
    the DMD rollout path: gradient from the attention output must flow through
    the gathered K/V into both the current pass's K/V projections and the
    cached history chunks.  Falling back to the inference-only all_gather_cat
    raises on grad-carrying tensors, and a detach in the gather would leave
    these .grad fields empty."""
    world_size, hist_tokens, own_tokens, dim, heads = 2, 4, 6, 32, 4

    _patch_qkv_projections(monkeypatch)
    monkeypatch.setattr(transformer_mod, "dispatch_attention_fn", _sdpa_dispatch)
    gather_groups = []

    def fake_all_gather_cat_with_grad(tensor, dim, group=None):
        # Batch-as-rank stand-in for the grad-carrying gather: every rank ends
        # up with the rank-order concatenation, and pure chunk/cat keeps the
        # exchange autograd-aware like the real collective.
        gather_groups.append(group)
        gathered = torch.cat(tensor.chunk(world_size, dim=0), dim=dim)
        return torch.cat([gathered] * world_size, dim=0)

    # gather_kv_cat resolves the collectives from its own module, so the fake
    # patches common_cp_mod (the grad-mode dispatch still runs).
    monkeypatch.setattr(common_cp_mod, "all_gather_cat_with_grad", fake_all_gather_cat_with_grad)

    torch.manual_seed(0)
    attn = _FakeWanAttention(dim, heads)
    # One finished history chunk per rank, stored with its autograd graph
    # (truncate_rollout_gradients=False), exactly what the exit denoising
    # step's gather must keep connected.
    k_history = torch.randn(world_size, hist_tokens, heads, dim // heads, requires_grad=True)
    v_history = torch.randn(world_size, hist_tokens, heads, dim // heads, requires_grad=True)
    cache = RolloutKvCache(1, truncate_rollout_gradients=False)
    cache.store_gen_chunk(0, 0, 0, 1, k_history, v_history)

    processor = KvCacheWanSelfAttnProcessor()
    processor.set_context_parallel_mesh(_GroupedFakeCpMesh(size=world_size, rank=0))
    out = processor(
        attn,
        torch.randn(world_size, own_tokens, dim),
        None,
        None,
        None,
        kv_cache=cache,
        layer_index=0,
        num_history_chunks=1,
        kv_cache_stage=None,
    )

    assert out.shape == (world_size, own_tokens, dim)
    # One grad-carrying gather each for K and V, both on the mesh group.
    assert gather_groups == [processor._cp_mesh.get_group()] * 2

    out.sum().backward()

    for source in (k_history, v_history, attn.to_k.weight, attn.to_v.weight):
        assert source.grad is not None
        assert torch.any(source.grad != 0)


# ---------------------------------------------------------------------------
# transformer_cp_head_counts (CPU): the single source of truth for the CP head
# constraint, read by both entry points (run_inference's CP-eval assert and
# the trainer's validation CP group sizing)
# ---------------------------------------------------------------------------


# Reduced Wan config (tests/test_wan_pipeline_cp_inference.py): production
# latent contract (16 channels, patch (1, 2, 2)), everything else shrunk.
_REDUCED_WAN_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=16,
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=32,
    ffn_dim=48,
    num_layers=2,
    rope_max_seq_len=256,
)


def _attention_test_wan_transformer():
    return ArtifixerTransformer(
        WanTransformer3DModel(**_REDUCED_WAN_CONFIG),
        frames_per_block=1,
        use_kv_cache_attention=False,
        vae_scale_factor_spatial=2,
        vae_scale_factor_temporal=4,
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
    )


@pytest.mark.parametrize("cosmos3", [False, True])
def test_tiny_cpu_wrappers_do_not_initialize_attention(monkeypatch, cosmos3):
    forbidden = Mock(side_effect=AssertionError("direct CPU construction must not initialize GPU attention"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    monkeypatch.setattr(attention_config, "configure_attention", forbidden)
    monkeypatch.setattr(attention_config, "_patch_flash_3_backend", forbidden)
    monkeypatch.setattr(attention_config, "activate_flash_attention_impl", forbidden)
    monkeypatch.setattr(attention_config, "list_flash_attention_impls", forbidden)

    wrapper = _reduced_real_transformer() if cosmos3 else _attention_test_wan_transformer()

    assert all(parameter.device.type == "cpu" for parameter in wrapper.parameters())
    forbidden.assert_not_called()


@pytest.mark.parametrize("context_parallel", [False, True])
def test_wan_masks_follow_late_attention_configuration(monkeypatch, attention_runtime, context_parallel):
    runtime = attention_runtime
    runtime.cuda_available = False
    wrapper = _attention_test_wan_transformer()
    if context_parallel:
        wrapper.enable_context_parallel(_FakeCpMesh(size=2, rank=0))
    for block in wrapper.blocks:
        monkeypatch.setattr(block, "forward", lambda hidden_states, *args: hidden_states)
    builder_name = "frame_block_causal_mask"
    builder = Mock(wraps=getattr(transformer_mod, builder_name))
    monkeypatch.setattr(transformer_mod, builder_name, builder)
    monkeypatch.setattr(attention_config, "_patch_flash_3_backend", Mock(return_value=True))
    hidden_states = torch.zeros(1, 16, 2, 4, 4)
    kwargs = dict(
        hidden_states=hidden_states,
        timestep=torch.ones(1),
        encoder_hidden_states=torch.ones(1, 3, 32),
        neighbor_hidden_states=None,
        opacity=torch.ones(1, 5, 8, 8),
        camera_rays=torch.zeros(1, 2, 8, 8, 6),
        w2cs=torch.eye(4).view(1, 1, 4, 4).expand(1, 2, -1, -1),
        neighbor_w2cs=None,
        Ks=torch.eye(3).view(1, 1, 3, 3).expand(1, 2, -1, -1),
        neighbor_Ks=None,
        local_cp_input=context_parallel,
    )

    for major, builds in ((8, 1), (9, 2), (8, 2)):
        runtime.major = major
        runtime.cuda_available = True
        attention_config.configure_attention("cuda:3")
        runtime.cuda_available = False
        assert wrapper(**kwargs)[0].shape == hidden_states.shape
        assert builder.call_count == builds

    calls = [call.kwargs for call in builder.call_args_list]
    assert [call["block_size"] for call in calls] == [128, (256, 128)]
    for mask in wrapper.block_masks.values():
        assert mask.kv_num_blocks.shape[:2] == (1, 1)
        assert mask.kv_indices.shape[:2] == (1, 1)


def test_cosmos3_masks_follow_late_attention_configuration(monkeypatch, attention_runtime):
    runtime = attention_runtime
    runtime.cuda_available = False
    wrapper = cosmos3_mod.ArtifixerCosmos3Transformer(
        _TinyCosmosBaseTransformer(),
        vae_scale_factor_spatial=2,
        vae_scale_factor_temporal=4,
        gradient_checkpointing=False,
        causal_block_frames=1,
    )
    builder_name = "create_prefix_block_causal_block_mask"
    builder = Mock(wraps=getattr(cosmos3_mod, builder_name))
    monkeypatch.setattr(cosmos3_mod, builder_name, builder)
    monkeypatch.setattr(attention_config, "_patch_flash_3_backend", Mock(return_value=True))
    make_mask = wrapper._armed_block_causal_block_mask
    kwargs = dict(
        und_len=3,
        block_frames=1,
        spatial_tokens_per_frame=4,
        device="cpu",
    )
    kwargs["global_latent_frames"] = 4
    masks = []

    for major in (8, 9, 8):
        runtime.major = major
        runtime.cuda_available = True
        attention_config.configure_attention("cuda:3")
        masks.append(make_mask(**kwargs))

    assert masks[0] is masks[2] and masks[0] is not masks[1]
    calls = [call.kwargs for call in builder.call_args_list]
    assert [call["block_size"] for call in calls] == [128, (256, 128)]
    for mask in masks:
        assert mask.kv_num_blocks.shape[:2] == (1, 1)
        assert mask.kv_indices.shape[:2] == (1, 1)


def _stub_transformer_without_cp_processors() -> nn.Module:
    """A transformer shape outside the CP design: its attention-like submodule
    carries a processor, but not one with the ``_cp_mesh`` slot that
    ``enable_context_parallel`` sets — so it must not count as CP-capable."""
    attention = nn.Module()
    attention.heads = 8
    attention.processor = object()
    transformer = nn.Module()
    transformer.attn = attention
    return transformer


def test_cp_head_counts_report_cosmos3_real_gqa_shape():
    """The REAL (reduced-config) cosmos3 wrapper reports (4 query heads, 2 KV
    heads): grouped-query attention keeps the two counts distinct, and the
    smaller KV count is the binding CP-size constraint. Only a transformer
    whose attention processors carry the ``_cp_mesh`` slot (which the real
    wrappers install at construction, before CP is ever enabled) drives the
    helper's counting loop at all — stubs skip it entirely."""
    assert transformer_cp_head_counts(_reduced_real_transformer()) == (4, 2)


def test_cp_head_counts_report_equal_counts_for_wan_mha():
    """The REAL (reduced-config) Wan wrapper reports equal counts: Wan
    attention is multi-head (one KV head per query head) and exposes only
    ``heads``, so the helper fills both slots from it and the query and KV
    constraints coincide."""
    base = WanTransformer3DModel(**_REDUCED_WAN_CONFIG)
    # What prepare_transformer_for_training applies before the pipelines wrap
    # the base model.
    base.set_attention_backend("native")
    wrapper = ArtifixerTransformer(
        base,
        frames_per_block=None,
        use_kv_cache_attention=False,
        vae_scale_factor_spatial=8,
        vae_scale_factor_temporal=4,
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
    )
    assert transformer_cp_head_counts(wrapper) == (2, 2)


def test_cp_head_counts_yield_none_for_tolerant_callers_on_cp_less_transformers():
    """``required=False`` is the tolerant-caller contract (trainer
    validation): a transformer with no CP-slotted attention processor — test
    stubs, foreign backbones — yields None, meaning "no head constraint",
    instead of an error."""
    assert transformer_cp_head_counts(_stub_transformer_without_cp_processors(), required=False) is None


def test_cp_head_counts_raise_for_strict_callers_on_cp_less_transformers():
    """``required=True`` (the default — run_inference's CP-eval contract): the
    same CP-less transformer is a loud RuntimeError, because a CP run whose
    head limits cannot be validated must not be waved through silently."""
    with pytest.raises(RuntimeError, match="no attention module with a context-parallel"):
        transformer_cp_head_counts(_stub_transformer_without_cp_processors())


# ---------------------------------------------------------------------------
# Real-collective (torchrun) tests — tests/run_cp_attention_tests.sh launches
# this file under torchrun, so they need no CI change; under plain pytest they
# skip. Every rank runs the same pytest collection, so the collective schedule
# stays rank-symmetric.
# ---------------------------------------------------------------------------


def _under_torchrun() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


requires_torchrun = pytest.mark.skipif(
    not _under_torchrun(),
    reason="requires torchrun with >= 2 GPUs (tests/run_cp_attention_tests.sh)",
)


@pytest.fixture(scope="module")
def distributed_env():
    """Init the NCCL process group from torchrun's env once for this module;
    torn down after the last test. xdist workers collect different subsets
    per process, which would desync the rank-symmetric collective schedule."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("torch.distributed process groups are incompatible with xdist workers")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not dist.is_initialized():
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    yield rank, world_size
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@requires_torchrun
def test_all_gather_cat_with_grad_real_backward_gives_world_size_times_slice_grad(distributed_env):
    """THE premise of the CP-training no-rescale decision, pinned on real
    collectives: when every rank forwards through all_gather_cat_with_grad
    and computes the IDENTICAL full-tensor loss, torch.distributed.nn's
    backward reduce-scatter SUMS the world_size identical upstream grads, so
    each rank's leaf gradient is exactly world_size TIMES its slice of the
    single-loss gradient — the factor that FSDP's gradient mean over the
    full world divides back out, which is why the trainer applies no loss
    rescaling. A single-process fake cannot pin this factor (its second
    backward through the one shared graph no-ops), and this is also the only
    real-collective coverage of all_gather_cat_with_grad's forward values."""
    rank, world_size = distributed_env
    device = torch.device("cuda", rank)
    torch.manual_seed(0)  # identical draw on every rank
    full = torch.randn(world_size * 3, 2, 4, device=device)
    local = full.chunk(world_size, dim=0)[rank].detach().clone().requires_grad_(True)

    gathered = all_gather_cat_with_grad(local, dim=0, group=dist.group.WORLD)
    # Forward: rank-order concatenation reassembles the full tensor exactly.
    torch.testing.assert_close(gathered, full, rtol=0, atol=0)

    weights = torch.arange(1, gathered.numel() + 1, device=device, dtype=gathered.dtype).view_as(gathered)
    loss = (gathered * weights).sum()  # identical on every rank: dL/d(gathered) = weights
    loss.backward()

    # Backward: the reduce-scatter sums world_size identical weight slices —
    # summing bit-equal fp32 values is exact, so the pin is bitwise.
    expected = world_size * weights.chunk(world_size, dim=0)[rank]
    torch.testing.assert_close(local.grad, expected, rtol=0, atol=0)
    dist.barrier()


def _cosmos3_reduced_transformer_with_live_adapters(device: torch.device):
    """Build a small real model with active camera and opacity inputs.

    Fixed seeds give every worker identical weights.
    """
    transformer = _reduced_real_transformer()
    generator = torch.Generator().manual_seed(23)
    for adapter in transformer.opacity_embeddings:
        adapter.weight.data.normal_(0.0, 0.02, generator=generator)
        adapter.bias.data.normal_(0.0, 0.02, generator=generator)
    # Nonzero weights make the output checks sensitive to incorrect camera slices.
    for injection in transformer.scope:
        injection.query_projection.weight.data.normal_(0.0, 0.02, generator=generator)
        injection.key_projection.weight.data.normal_(0.0, 0.02, generator=generator)
        injection.alpha.data.fill_(0.25)
    return transformer.to(device)


def test_cosmos3_cp_fixture_has_live_camera_conditioning():
    inputs = _growing_prefix_kwargs(4)
    inputs["neighbor_hidden_states"] = None
    transformer = _cosmos3_reduced_transformer_with_live_adapters(torch.device("cpu"))
    with torch.no_grad():
        reference = transformer(**inputs)[0]
        changed = transformer(**dict(inputs, camera_rays=inputs["camera_rays"].roll(1, dims=1)))[0]
    assert not torch.allclose(reference, changed)


# Splitting work across GPUs changes the order of floating-point calculations.
# Compare errors with each parameter's gradient size, allowing small rounding
# differences but not incorrect scaling between workers.
_CP_GRAD_EQUIVALENCE_MAX_DEVIATION = 2e-3


@requires_torchrun
def test_cosmos3_cp_transformer_forward_backward_matches_single_rank(distributed_env):
    """Compare one-worker and two-worker outputs and gradients on a small real model.

    Both workers receive the full clip and compute the same loss. Averaging their
    gradients must match the one-worker result, allowing for rounding differences.
    """
    rank, world_size = distributed_env
    if world_size != 2:
        pytest.skip("the reduced cosmos3 config has 4 query / 2 KV heads: its Ulysses path supports exactly CP=2")
    device = torch.device("cuda", rank)

    # True-fp32 island: the NGC base images start torch with TF32 matmuls
    # force-enabled (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1); TF32's 10-bit
    # mantissa dominates the CP-vs-CP=1 backward noise, so pin it off for
    # both runs and restore afterwards.
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        inputs = _growing_prefix_kwargs(4)  # 4 latent frames: 2 per rank at CP=2
        inputs["neighbor_hidden_states"] = None
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

        reference = _cosmos3_reduced_transformer_with_live_adapters(device)
        reference_output = reference(**inputs)[0]
        reference_output.square().sum().backward()
        reference_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
            if parameter.grad is not None
        }

        cp_transformer = _cosmos3_reduced_transformer_with_live_adapters(device)
        cp_transformer.enable_context_parallel(DeviceMesh("cuda", torch.arange(world_size)))
        cp_output = cp_transformer(**inputs)[0]
        assert cp_output.shape == reference_output.shape, "the CP forward must return the gathered full-clip output"
        torch.testing.assert_close(cp_output, reference_output, rtol=1e-4, atol=1e-5)

        # The identical full-clip loss on every rank — the trainer's contract.
        cp_output.square().sum().backward()
        for name, parameter in cp_transformer.named_parameters():
            if parameter.grad is None:
                assert name not in reference_grads, f"{name}: CP dropped a gradient the single-rank run produced"
                continue
            averaged_grad = parameter.grad.detach().clone()
            dist.all_reduce(averaged_grad, op=dist.ReduceOp.AVG)  # FSDP's mean over the full world
            reference_grad = reference_grads[name]
            # Inf-norm deviation, relative to this parameter's own gradient
            # scale (clamped so an all-zero reference grad compares against
            # a small absolute floor instead of dividing by zero).
            grad_scale = float(reference_grad.abs().max().clamp_min(1e-3))
            max_deviation = float((averaged_grad - reference_grad).abs().max())
            assert max_deviation <= _CP_GRAD_EQUIVALENCE_MAX_DEVIATION * grad_scale, (
                f"{name}: world-averaged CP grad diverged from CP=1: max |avg - ref| = {max_deviation:.3e} "
                f"exceeds {_CP_GRAD_EQUIVALENCE_MAX_DEVIATION:.0e} of the gradient scale {grad_scale:.3e} "
                f"(ratio {max_deviation / grad_scale:.3e}; a mis-scaled or misrouted CP path lands at ~0.5..1.0)"
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    dist.barrier()


@requires_torchrun
def test_cosmos3_cp_local_shard_training_step_matches_single_rank(distributed_env):
    """Check training when each worker receives only its own frames.

    Each output must match its slice of the one-worker output. Averaging the
    workers' gradients must match training on the whole clip at once.
    """
    rank, world_size = distributed_env
    if world_size != 2:
        pytest.skip("the reduced cosmos3 config has 4 query / 2 KV heads: its Ulysses path supports exactly CP=2")
    device = torch.device("cuda", rank)

    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        inputs = _growing_prefix_kwargs(4)  # 4 latent frames: 2 per rank at CP=2
        inputs["neighbor_hidden_states"] = None
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

        reference = _cosmos3_reduced_transformer_with_live_adapters(device)
        reference_output = reference(**inputs)[0]
        # CPU draw so the target is bit-identical on every rank.
        torch.manual_seed(11)
        target = torch.randn(reference_output.shape).to(device=device, dtype=reference_output.dtype)
        torch.nn.functional.mse_loss(reference_output.float(), target.float()).backward()
        reference_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
            if parameter.grad is not None
        }

        local_frames = inputs["hidden_states"].shape[2] // world_size
        start, end = rank * local_frames, (rank + 1) * local_frames
        # The causal VAE folds pixel frames [4k-3, 4k] into latent frame
        # k >= 1; latent frame 0 is pixel frame 0 alone.
        pixel_start = 0 if start == 0 else start * VAE_SCALE_FACTOR_TEMPORAL - (VAE_SCALE_FACTOR_TEMPORAL - 1)
        pixel_end = end * VAE_SCALE_FACTOR_TEMPORAL - (VAE_SCALE_FACTOR_TEMPORAL - 1)
        local_inputs = dict(
            inputs,
            hidden_states=inputs["hidden_states"][:, :, start:end].contiguous(),
            opacity=inputs["opacity"][:, pixel_start:pixel_end].contiguous(),
            camera_rays=inputs["camera_rays"][:, start:end].contiguous(),
            w2cs=inputs["w2cs"][:, start:end].contiguous(),
            Ks=inputs["Ks"][:, start:end].contiguous(),
            frame_offset=start,
            local_cp_input=True,
        )

        cp_transformer = _cosmos3_reduced_transformer_with_live_adapters(device)
        cp_transformer.enable_context_parallel(DeviceMesh("cuda", torch.arange(world_size)))
        local_output = cp_transformer(**local_inputs)[0]
        assert local_output.shape[2] == local_frames, "the local contract must return only this rank's frames"
        torch.testing.assert_close(local_output, reference_output[:, :, start:end], rtol=1e-4, atol=1e-5)

        torch.nn.functional.mse_loss(local_output.float(), target[:, :, start:end].float()).backward()
        for name, parameter in cp_transformer.named_parameters():
            if parameter.grad is None:
                assert name not in reference_grads, f"{name}: CP dropped a gradient the single-rank run produced"
                continue
            averaged_grad = parameter.grad.detach().clone()
            dist.all_reduce(averaged_grad, op=dist.ReduceOp.AVG)  # FSDP's mean over the full world
            reference_grad = reference_grads[name]
            grad_scale = float(reference_grad.abs().max().clamp_min(1e-3))
            max_deviation = float((averaged_grad - reference_grad).abs().max())
            assert max_deviation <= _CP_GRAD_EQUIVALENCE_MAX_DEVIATION * grad_scale, (
                f"{name}: world-averaged CP grad diverged from CP=1: max |avg - ref| = {max_deviation:.3e} "
                f"exceeds {_CP_GRAD_EQUIVALENCE_MAX_DEVIATION:.0e} of the gradient scale {grad_scale:.3e} "
                f"(ratio {max_deviation / grad_scale:.3e}; a mis-scaled or misrouted CP path lands at ~0.5..1.0)"
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    dist.barrier()


@pytest.mark.parametrize("cosmos3", [False, True])
def test_production_transformers_reject_teacher_forcing_inputs(cosmos3):
    wrapper = _reduced_real_transformer() if cosmos3 else _attention_test_wan_transformer()
    with pytest.raises(TypeError, match="unexpected keyword argument 'clean_x'"):
        wrapper(clean_x=torch.zeros(1, 16, 2, 4, 4))
