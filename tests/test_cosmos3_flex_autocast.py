# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real CUDA regressions for mixed FP32 Q/K and BF16 V at the flex boundary."""

import json
import sys
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.attention._registry as flash_registry
import torch.nn.functional as F
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn import attention as torch_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_mask

import model_training.net.cosmos3_transformer as cosmos
from model_training.net import attention_config
from model_training.utils.precision import disable_tf32_matmul
from tests._helpers import _activate_conditioning_adapters, _joint_reference_kwargs, _reduced_real_transformer
from tests._teacher_forcing_reference import cosmos3_teacher_forcing_reference, teacher_forcing_block_mask


class _ProbeError(Exception):
    pass


def _matmul_precision_state():
    return (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.fp32_precision,
        torch.backends.mkldnn.matmul.fp32_precision,
    )


def _restore_matmul_precision_state(state):
    mode, cuda, mkldnn = state
    torch.set_float32_matmul_precision(mode)
    torch.backends.cuda.matmul.fp32_precision = cuda
    torch.backends.mkldnn.matmul.fp32_precision = mkldnn


@contextmanager
def _native_matmul_precision():
    previous = _matmul_precision_state()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        _restore_matmul_precision_state(previous)


@pytest.fixture(autouse=True)
def native_matmul_precision():
    with _native_matmul_precision():
        yield


def _enter_high_through_legacy_cuda_setter():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = True


def _enter_medium_with_unset_mkldnn():
    torch.set_float32_matmul_precision("medium")
    torch.backends.mkldnn.matmul.fp32_precision = "none"


_INCOMING_PRECISION_STATES = {
    "default": lambda: _restore_matmul_precision_state(("highest", "none", "none")),
    "high_legacy_cuda_setter": _enter_high_through_legacy_cuda_setter,
    "medium": lambda: torch.set_float32_matmul_precision("medium"),
    "medium_unset_mkldnn": _enter_medium_with_unset_mkldnn,
}


@pytest.fixture
def restored_matmul_precision():
    previous = _matmul_precision_state()
    yield
    _restore_matmul_precision_state(previous)


@pytest.mark.parametrize("raise_error", [False, True])
@pytest.mark.parametrize("incoming", list(_INCOMING_PRECISION_STATES))
def test_precision_context_restores_the_complete_state(restored_matmul_precision, incoming, raise_error):
    _INCOMING_PRECISION_STATES[incoming]()
    before = _matmul_precision_state()
    with pytest.raises(_ProbeError) if raise_error else nullcontext():
        with _native_matmul_precision():
            assert _matmul_precision_state() == ("highest", "ieee", "ieee")
            if raise_error:
                raise _ProbeError("exercise cleanup")
    assert _matmul_precision_state() == before


def test_precision_context_keeps_the_legacy_cuda_setter_usable(restored_matmul_precision):
    _enter_high_through_legacy_cuda_setter()
    with _native_matmul_precision():
        pass
    disable_tf32_matmul(verbose=True)
    assert _matmul_precision_state() == ("highest", "ieee", "ieee")
    assert torch.backends.cuda.matmul.allow_tf32 is False


def _configure_production_attention(device):
    # FP32-master fixtures deliberately produce mixed Q/K/V. Unmasked FA3
    # requires half inputs, so use math SDPA there while retaining the real
    # architecture-selected FA4 flex backend at the boundary under test.
    backend = attention_config.configure_attention(device, attention_backend="_native_math")
    assert backend == "_native_math"
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 9:
        assert attention_config._FLEX_KERNEL_OPTIONS.get("BACKEND") == "FLASH", "FA4 is required on Hopper/Blackwell"
        assert attention_config.flex_block_size() == (256, 128)
    else:
        assert attention_config.flex_block_size() == 128
    print(
        f"attention_backend={backend} flex_options={attention_config._FLEX_KERNEL_OPTIONS} "
        f"block_size={attention_config.flex_block_size()} device={torch.cuda.get_device_name(device)}",
        flush=True,
    )
    return backend


def _attention_state():
    return (
        dict(attention_config._FLEX_KERNEL_OPTIONS),
        attention_config._FLASH_3_PATCHED,
        _AttentionBackendRegistry._backends.get(AttentionBackendName._FLASH_3),
        torch_attention.current_flash_attention_impl(),
    )


@contextmanager
def _isolated_attention_configuration():
    registry = _AttentionBackendRegistry._backends
    unset = object()
    flex_options = dict(attention_config._FLEX_KERNEL_OPTIONS)
    flash_3_patched = attention_config._FLASH_3_PATCHED
    flash_3_backend = registry.get(AttentionBackendName._FLASH_3, unset)
    flash_impl = torch_attention.current_flash_attention_impl()
    try:
        yield
    finally:
        attention_config._FLEX_KERNEL_OPTIONS.clear()
        attention_config._FLEX_KERNEL_OPTIONS.update(flex_options)
        attention_config._FLASH_3_PATCHED = flash_3_patched
        if flash_3_backend is unset:
            registry.pop(AttentionBackendName._FLASH_3, None)
        else:
            registry[AttentionBackendName._FLASH_3] = flash_3_backend
        if torch_attention.current_flash_attention_impl() != flash_impl:
            if flash_impl is None:
                torch_attention.restore_flash_attention_impl()
            else:
                torch_attention.activate_flash_attention_impl(flash_impl)


@contextmanager
def _configured_attention(device):
    with _isolated_attention_configuration():
        yield _configure_production_attention(device)


@pytest.fixture
def configured_attention():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA flex attention")
    with _configured_attention(torch.device("cuda")) as backend:
        yield backend


class _FakeFlashImpl:
    def __init__(self):
        self.handles = []

    def register(self):
        handle = SimpleNamespace(removed=False)
        handle.remove = lambda: setattr(handle, "removed", True)
        self.handles.append(handle)
        return handle

    def live(self):
        return [handle for handle in self.handles if not handle.removed]


@pytest.fixture
def fake_attention_runtime(monkeypatch):
    """Drive the real configure_attention on CPU against observable FA4/FA3 stand-ins."""
    runtime = SimpleNamespace(major=9, fa4=_FakeFlashImpl(), setup_states=[])

    def capability(device=None):
        runtime.setup_states.append(_attention_state())
        return runtime.major, 0

    fa3 = ModuleType("flash_attn_interface")
    fa3.__spec__ = ModuleSpec(fa3.__name__, loader=None)
    fa3.flash_attn_func = fa3._flash_attn_backward = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, fa3.__name__, fa3)
    monkeypatch.setitem(flash_registry._FLASH_ATTENTION_IMPLS, "FA4", runtime.fa4.register)
    monkeypatch.setattr(flash_registry, "_FLASH_ATTENTION_ACTIVE", None)
    monkeypatch.setattr(_AttentionBackendRegistry, "_backends", dict(_AttentionBackendRegistry._backends))
    monkeypatch.setattr(attention_config, "_FLEX_KERNEL_OPTIONS", {})
    monkeypatch.setattr(attention_config, "_FLASH_3_PATCHED", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: f"fake sm_{runtime.major}0")
    return runtime


_PRIOR_ATTENTION_STATES = {"default": None, "fa3": 9, "fa4": 10}


@pytest.mark.parametrize("outcome", ["normal", "body_failure", "setup_failure"])
@pytest.mark.parametrize("prior", list(_PRIOR_ATTENTION_STATES))
def test_configured_attention_restores_backend_state_on_every_exit(monkeypatch, fake_attention_runtime, prior, outcome):
    runtime = fake_attention_runtime
    if _PRIOR_ATTENTION_STATES[prior] is not None:
        runtime.major = _PRIOR_ATTENTION_STATES[prior]
        attention_config.configure_attention("cuda")
    before = _attention_state()
    assert before[1] == (prior == "fa3")
    assert before[3] == (None if prior == "default" else "FA4")
    runtime.major = 9
    runtime.setup_states.clear()
    if outcome == "setup_failure":
        monkeypatch.delitem(flash_registry._FLASH_ATTENTION_IMPLS, "FA4")
    expected_error = {"normal": None, "body_failure": _ProbeError, "setup_failure": AssertionError}[outcome]
    with pytest.raises(expected_error) if expected_error else nullcontext():
        with _configured_attention(torch.device("cuda")) as backend:
            assert backend == "_native_math"
            flex_options, flash_3_patched, flash_3_backend, flash_impl = _attention_state()
            assert flex_options == {"BACKEND": "FLASH"} and flash_3_patched and flash_impl == "FA4"
            assert (flash_3_backend is before[2]) == (prior == "fa3")
            assert len(runtime.fa4.live()) == 1
            if outcome == "body_failure":
                raise _ProbeError("exercise cleanup")
    if outcome == "setup_failure":
        flex_options, flash_3_patched, flash_3_backend, flash_impl = runtime.setup_states[-1]
        assert flex_options == {} and flash_3_patched and flash_impl == before[3]
        assert (flash_3_backend is before[2]) == (prior == "fa3")
    assert _attention_state() == before
    assert len(runtime.fa4.live()) == (0 if prior == "default" else 1)


def _teacher_inputs():
    kwargs = {
        name: value.cuda() if torch.is_tensor(value) else value
        for name, value in _joint_reference_kwargs(total_frames=4).items()
    }
    clean = torch.randn_like(kwargs["hidden_states"])
    kwargs["timestep"] = torch.tensor([[800.0, 400.0]], device="cuda").repeat_interleave(8, dim=1)
    return kwargs, clean


def _teacher(backend=None):
    model = _reduced_real_transformer(causal_block_frames=2)
    _activate_conditioning_adapters(model)
    model = model.cuda()
    model.set_attention_backend(backend)
    return model


@pytest.mark.parametrize("execution_context", [torch.no_grad, torch.inference_mode], ids=["no_grad", "inference_mode"])
def test_cuda_bf16_no_grad_teacher_forcing_matches_dense_oracle(monkeypatch, execution_context, configured_attention):
    """Compare actual main-model masked forward using identical QKV/mask semantics."""
    teacher = _teacher(configured_attention)
    kwargs, clean = _teacher_inputs()

    def dense_attention(self, q, k, v, block_mask):
        dense_mask = create_mask(block_mask.mask_mod, None, None, q.shape[0], k.shape[0], q.device)
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(
                q.unsqueeze(0).transpose(1, 2),
                k.unsqueeze(0).transpose(1, 2),
                v.unsqueeze(0).transpose(1, 2),
                attn_mask=dense_mask,
                enable_gqa=True,
            ).transpose(1, 2)

    with execution_context(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = cosmos3_teacher_forcing_reference(teacher, **kwargs, clean_x=clean)[0]
        monkeypatch.setattr(cosmos.ArtifixerCosmos3AttnProcessor, "_masked_gen_attention", dense_attention)
        expected = cosmos3_teacher_forcing_reference(teacher, **kwargs, clean_x=clean)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_cuda_mixed_qkv_attention_matches_dense_oracle(grad_enabled, configured_attention):
    """The same active mask/QKV must agree in inference and differentiable calls."""
    torch.manual_seed(13)
    q = torch.randn(32, 4, 32, device="cuda", requires_grad=grad_enabled)
    k = torch.randn(48, 2, 32, device="cuda", requires_grad=grad_enabled)
    v = torch.randn(48, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=grad_enabled)
    mask = teacher_forcing_block_mask(
        device=q.device,
        num_frames=4,
        frame_seqlen=4,
        frames_per_block=2,
        block_size=attention_config.flex_block_size(),
        prefix_len=16,
    )
    dense_mask = create_mask(mask.mask_mod, None, None, 32, 48, q.device)
    with torch.set_grad_enabled(grad_enabled), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = cosmos.ArtifixerCosmos3AttnProcessor()._masked_gen_attention(q, k, v, mask)
        with sdpa_kernel([SDPBackend.MATH]):
            expected = F.scaled_dot_product_attention(
                q.unsqueeze(0).transpose(1, 2),
                k.unsqueeze(0).transpose(1, 2),
                v.unsqueeze(0).transpose(1, 2),
                attn_mask=dense_mask,
                enable_gqa=True,
            ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-2)
    if grad_enabled:
        weights = torch.randn_like(actual)
        actual_grad = torch.autograd.grad((actual * weights).float().sum(), (q, k, v))
        expected_grad = torch.autograd.grad((expected * weights).float().sum(), (q, k, v))
        for observed, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(observed, reference, rtol=2e-2, atol=1e-2)


def _implicit_autocast_attention(self, q, k, v, block_mask):
    """Pre-fix CUDA call; retained only as the differentiable training control."""
    return cosmos.flex_attention(
        q.unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        v.unsqueeze(0).transpose(1, 2),
        block_mask=block_mask,
        enable_gqa=True,
    ).transpose(1, 2)


@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_boundary_preserves_real_model_parameter_gradients(monkeypatch, autocast, configured_attention):
    """Same native parameters, inputs and loss retain training and FP32 behavior."""
    if not autocast:
        # FA4/FA3 accept half precision. The FP32 control uses Triton + math SDPA.
        attention_config._FLEX_KERNEL_OPTIONS.clear()
    teacher = _teacher(configured_attention if autocast else "_native_math")
    kwargs, clean = _teacher_inputs()
    fixed = cosmos.ArtifixerCosmos3AttnProcessor._masked_gen_attention

    def run(method):
        monkeypatch.setattr(cosmos.ArtifixerCosmos3AttnProcessor, "_masked_gen_attention", method)
        teacher.zero_grad(set_to_none=True)
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            result = cosmos3_teacher_forcing_reference(teacher, **kwargs, clean_x=clean)[0]
            loss = result.float().square().mean()
        loss.backward()
        gradients = {name: p.grad.detach().clone() for name, p in teacher.named_parameters() if p.grad is not None}
        assert len(gradients) > 10 and all(torch.isfinite(g).all() for g in gradients.values())
        assert any(torch.count_nonzero(g) for g in gradients.values())
        return result.detach(), gradients

    reference, reference_grad = run(_implicit_autocast_attention)
    observed, observed_grad = run(fixed)
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert observed_grad.keys() == reference_grad.keys()
    for name in reference_grad:
        torch.testing.assert_close(observed_grad[name], reference_grad[name], rtol=1e-5, atol=1e-6, msg=name)


def _bf16_cp_worker(rank, init_file, mode):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=240)
    )
    original = cosmos.ArtifixerCosmos3AttnProcessor._masked_gen_attention
    boundary = original
    calls = []

    def observed_boundary(self, q, k, v, block_mask):
        assert torch.is_autocast_enabled("cuda")
        if mode == "fsdp2":
            assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        else:
            assert q.dtype == k.dtype == torch.float32 and v.dtype == torch.bfloat16
        calls.append((q.shape[1], torch.is_grad_enabled()))
        return boundary(self, q, k, v, block_mask)

    cosmos.ArtifixerCosmos3AttnProcessor._masked_gen_attention = observed_boundary
    try:
        with _native_matmul_precision():
            backend = _configure_production_attention(device)
            reference = _teacher(backend)
            if mode == "fsdp2":
                # FSDP's policy casts parameters, not the FP32 RoPE buffers.
                for parameter in reference.parameters():
                    parameter.data = parameter.data.to(torch.bfloat16)
            kwargs, _ = _teacher_inputs()
            cp = _reduced_real_transformer(
                causal_block_frames=2, gradient_checkpointing=mode in ("checkpointed", "fsdp2")
            ).to(device)
            cp.load_state_dict(reference.state_dict(), strict=True)
            cp.set_attention_backend(backend)
            mesh = DeviceMesh("cuda", torch.arange(2))
            cp.enable_context_parallel(mesh)
            if mode == "fsdp2":
                policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
                )
                for layer in cp.base_transformer.layers:
                    fully_shard(layer, mesh=mesh, mp_policy=policy)
                fully_shard(cp, mesh=mesh, mp_policy=policy)
            grad_enabled = mode != "no_grad"

            def run(model):
                nonlocal boundary
                model.zero_grad(set_to_none=True)
                inputs = dict(kwargs)
                for name in ("hidden_states", "neighbor_hidden_states"):
                    inputs[name] = inputs[name].detach().clone().requires_grad_(grad_enabled)
                with torch.set_grad_enabled(grad_enabled), torch.autocast("cuda", dtype=torch.bfloat16):
                    # CP supports the block-causal path; clean_x teacher forcing
                    # explicitly rejects CP and is covered above at CP=1.
                    result = model(**inputs)[0]
                    if grad_enabled:
                        result.float().square().sum().backward()
                if grad_enabled:
                    # Compare each layout with the old differentiable call
                    # exactly, independently of BF16 CP reduction rounding.
                    saved = {}
                    for name, parameter in model.named_parameters():
                        gradient = parameter.grad
                        if gradient is not None:
                            local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
                            saved[name] = local.detach().clone()
                    model.zero_grad(set_to_none=True)
                    control_inputs = dict(inputs)
                    for name in ("hidden_states", "neighbor_hidden_states"):
                        control_inputs[name] = inputs[name].detach().clone().requires_grad_(True)
                    boundary = _implicit_autocast_attention
                    try:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            control = model(**control_inputs)[0]
                            control.float().square().sum().backward()
                    finally:
                        boundary = original
                    torch.testing.assert_close(result, control, rtol=0, atol=0)
                    for name, parameter in model.named_parameters():
                        gradient = parameter.grad
                        assert (gradient is not None) == (name in saved), name
                        if gradient is not None:
                            local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
                            torch.testing.assert_close(local, saved[name], rtol=0, atol=0, msg=name)
                    for name in ("hidden_states", "neighbor_hidden_states"):
                        torch.testing.assert_close(
                            inputs[name].grad, control_inputs[name].grad, rtol=0, atol=0, msg=name
                        )
                return result.detach(), inputs

            reference_output, reference_inputs = run(reference)
            assert calls and all(heads == 4 for heads, _ in calls)
            reference_call_count = len(calls)
            calls.clear()
            output, cp_inputs = run(cp)
            assert calls and all(heads == 2 for heads, _ in calls)
            if mode in ("checkpointed", "fsdp2"):
                assert len(calls) > reference_call_count, "checkpointed backward must recompute the masked boundary"
            torch.testing.assert_close(output, reference_output, rtol=2e-2, atol=1e-2)
            maximum_grad_difference = 0.0
            maximum_grad_relative_max = 0.0
            maximum_grad_relative_l2 = 0.0
            if grad_enabled:
                checked = 0

                def compare_grad(observed, expected, name):
                    nonlocal maximum_grad_difference, maximum_grad_relative_max, maximum_grad_relative_l2
                    if mode == "fsdp2" and observed is not None and hasattr(observed, "full_tensor"):
                        observed = observed.full_tensor()
                    if expected is None:
                        assert observed is None, name
                        return
                    assert observed is not None, name
                    assert torch.isfinite(observed).all(), name
                    assert torch.isfinite(expected).all(), name
                    averaged = observed.detach().float().clone()
                    dist.all_reduce(averaged)
                    averaged /= 2
                    # CP changes the order of BF16 partial-gradient rounding.
                    # Bound both tensor-scale maximum error and relative L2
                    # by two BF16 epsilons; pointwise relative error is unstable
                    # around zero. The old-call control above must still be exact.
                    expected = expected.float()
                    difference = averaged - expected
                    scale = expected.abs().max().clamp_min(1e-20)
                    relative_max = (difference.abs().max() / scale).item()
                    relative_l2 = (difference.norm() / expected.norm().clamp_min(1e-20)).item()
                    tolerance = 2 * torch.finfo(torch.bfloat16).eps
                    assert relative_max <= tolerance, (name, "relative_max", relative_max, tolerance)
                    assert relative_l2 <= tolerance, (name, "relative_l2", relative_l2, tolerance)
                    maximum_grad_difference = max(maximum_grad_difference, difference.abs().max().item())
                    maximum_grad_relative_max = max(maximum_grad_relative_max, relative_max)
                    maximum_grad_relative_l2 = max(maximum_grad_relative_l2, relative_l2)

                expected_parameters = dict(reference.named_parameters())
                assert dict(cp.named_parameters()).keys() == expected_parameters.keys()
                for name, parameter in cp.named_parameters():
                    expected_parameter = expected_parameters[name]
                    compare_grad(parameter.grad, expected_parameter.grad, name)
                    if expected_parameter.grad is not None and torch.count_nonzero(expected_parameter.grad):
                        checked += 1
                assert checked > 10
                for name in ("hidden_states", "neighbor_hidden_states"):
                    assert reference_inputs[name].grad is not None
                    compare_grad(cp_inputs[name].grad, reference_inputs[name].grad, name)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "mode": mode,
                        "backend": backend,
                        "flex_options": attention_config._FLEX_KERNEL_OPTIONS,
                        "output_max_abs": (output - reference_output).abs().max().item(),
                        "gradient_max_abs": maximum_grad_difference if grad_enabled else None,
                        "gradient_relative_max": maximum_grad_relative_max if grad_enabled else None,
                        "gradient_relative_l2": maximum_grad_relative_l2 if grad_enabled else None,
                        "old_call_exact_parity": grad_enabled,
                        "mixed_qkv_asserted": mode != "fsdp2",
                        "fsdp2_bf16_compute": mode == "fsdp2",
                        "status": "pass",
                    }
                ),
                flush=True,
            )
    finally:
        cosmos.ArtifixerCosmos3AttnProcessor._masked_gen_attention = original
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices for configured BF16 NCCL parity")
@pytest.mark.parametrize("mode", ["no_grad", "backward", "checkpointed", "fsdp2"])
def test_configured_bf16_masked_cp1_cp2_outputs_and_gradients(tmp_path, mode):
    mp.spawn(_bf16_cp_worker, args=(str(tmp_path / "nccl-init"), mode), nprocs=2, join=True)
