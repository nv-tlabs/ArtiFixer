# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention backend setup for the selected runtime device."""

import importlib.util

import torch
from torch.nn.attention import activate_flash_attention_impl, list_flash_attention_impls
from torch.nn.attention.flex_attention import flex_attention as _flex_attention_orig

# flex_attention kernel_options are set based on GPU arch after configure_attention runs.
_FLEX_KERNEL_OPTIONS: dict[str, str] = {}
_FLASH_3_PATCHED = False


def flex_attention_with_options(query, key, value, block_mask=None, enable_gqa=False):
    return _flex_attention_orig(
        query, key, value, block_mask=block_mask, enable_gqa=enable_gqa, kernel_options=_FLEX_KERNEL_OPTIONS
    )


def flex_block_size() -> int | tuple[int, int]:
    """Return the block size required by the configured flex backend."""
    # FA4 requires Q blocks divisible by 256 and KV blocks of 128; other backends use 128.
    return (256, 128) if _FLEX_KERNEL_OPTIONS.get("BACKEND") == "FLASH" else 128


# ---------------------------------------------------------------------------
# Fix: diffusers >= 0.37 wraps flash_attn_3_func in a _custom_op without
# autograd backward, breaking training.  We replace the _flash_3 backend with
# a torch.autograd.Function that supports both torch.compile (via
# allow_in_graph) and training (proper backward).  This is the same pattern
# FA2 uses (FlashAttnFunc) and should be upstreamed to diffusers.
# See: https://github.com/huggingface/diffusers/issues/12022
# ---------------------------------------------------------------------------
def _patch_flash_3_backend() -> bool:
    global _FLASH_3_PATCHED
    if _FLASH_3_PATCHED:
        return True
    try:
        from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
        from flash_attn_interface import _flash_attn_backward as _fa3_bwd
        from flash_attn_interface import flash_attn_func as _fa3_func

        @torch.compiler.allow_in_graph
        class _FlashAttn3Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v, softmax_scale, causal):
                out, lse, *_ = _fa3_func(
                    q=q,
                    k=k,
                    v=v,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    return_attn_probs=True,
                )
                ctx.save_for_backward(q, k, v, out, lse)
                ctx.softmax_scale = softmax_scale
                ctx.causal = causal
                return out, lse

            @staticmethod
            def backward(ctx, grad_out, grad_lse):
                q, k, v, out, lse = ctx.saved_tensors
                dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
                # Call positionally to match FA3's own backward and avoid
                # custom op kwarg dispatch issues.
                _fa3_bwd(
                    grad_out.contiguous(),
                    q,
                    k,
                    v,
                    out,
                    lse,
                    None,
                    None,  # cu_seqlens_q, cu_seqlens_k
                    None,
                    None,  # sequed_q, sequed_k
                    None,
                    None,  # max_seqlen_q, max_seqlen_k
                    dq,
                    dk,
                    dv,
                    ctx.softmax_scale,
                    ctx.causal,
                    -1,
                    -1,  # window_size_left, window_size_right
                    0.0,  # softcap
                    False,  # deterministic
                    0,  # sm_margin
                )
                return dq, dk, dv, None, None

        def _flash_attention_3(
            query,
            key,
            value,
            attn_mask=None,
            scale=None,
            is_causal=False,
            return_lse=False,
            _parallel_config=None,
            **kwargs,
        ):
            if attn_mask is not None:
                raise ValueError("`attn_mask` is not supported for flash-attn 3.")
            out, lse = _FlashAttn3Func.apply(query, key, value, scale, is_causal)
            return (out, lse) if return_lse else out

        _AttentionBackendRegistry._backends[AttentionBackendName._FLASH_3] = _flash_attention_3
        _FLASH_3_PATCHED = True
        print("Patched _flash_3 backend with autograd.Function + allow_in_graph")
        return True
    except ImportError:
        return False  # FA3 not installed


def _select_attention_config(device: torch.device) -> str | None:
    """Select FA implementation and return the default diffusers attention backend
    based on the GPU architecture.

    Returns the recommended ``dispatch_attention_fn`` backend string, or *None*
    to let diffusers auto-detect (cuDNN SDPA — best for A100 / sm_80).

    Three mechanisms work together:

    * ``activate_flash_attention_impl("FA4")`` — registers FA4 kernels into
      ``aten::_scaled_dot_product_flash_attention`` (used by SDPA ``_native_flash``).
    * ``_FLEX_KERNEL_OPTIONS['BACKEND'] = 'FLASH'`` — tells ``flex_attention`` to
      use FA4's CuTeDSL kernels instead of Triton codegen.
    * ``dispatch_attention_fn(backend=…)`` — diffusers' own dispatch for cross-attn,
      non-block self-attn, and KV-cache attn.

    ======== ==================================== ==========================================
    GPU      ``flex_attention``                    ``dispatch_attention_fn`` backend
    ======== ==================================== ==========================================
    A100     Triton (AUTO)                        ``None`` → auto SDPA (cuDNN)
    H100     FA4 CuTeDSL (BACKEND=FLASH)          ``"_flash_3"`` → Dao-AILab FA3 Hopper
    GB200    FA4 CuTeDSL (BACKEND=FLASH)          ``None`` → auto SDPA (FA4 via aten)
    ======== ==================================== ==========================================

    FA3 is Hopper-only and does not run on Blackwell. Images built with
    ``BUILD_FLASH_ATTN_3=0`` ship no FA3; on H100 they use auto SDPA instead.
    """
    major, _ = torch.cuda.get_device_capability(device)
    impls = list_flash_attention_impls()

    if major >= 9 and "FA4" in impls:
        # Activate FA4 for aten SDPA flash override (used by _native_flash on GB200).
        try:
            activate_flash_attention_impl("FA4")
            print(f"Activated FA4 on sm_{major}0")
        except Exception as e:
            print(f"WARNING: FA4 activation failed on sm_{major}0: {e}")

        # Use FA4's CuTeDSL kernels for flex_attention (requires flash_attn.cute).
        _FLEX_KERNEL_OPTIONS["BACKEND"] = "FLASH"
        print("flex_attention will use FA4 CuTeDSL backend")

    if major >= 10:
        # Blackwell (sm_100+) — auto SDPA picks FA4-flash or cuDNN as appropriate.
        print(f"Attention config: sm_{major}0 (Blackwell) — auto SDPA, dispatch=auto")
        return None

    if major >= 9:
        # Hopper (sm_90) — FA3 (Dao-AILab Hopper kernels) for dispatch when the
        # image was built with it (BUILD_FLASH_ATTN_3=1); otherwise auto SDPA.
        if importlib.util.find_spec("flash_attn_interface") is None or not _patch_flash_3_backend():
            print(f"Attention config: sm_{major}0 (Hopper) — FA3 not installed, auto SDPA, dispatch=auto")
            return None
        print(f"Attention config: sm_{major}0 (Hopper) — FA4 aten + FA3 dispatch")
        return "_flash_3"

    # A100 (sm_80) or older — auto SDPA (picks cuDNN).
    print(f"Attention config: sm_{major}0 — auto SDPA (cuDNN), dispatch=auto")
    return None


def configure_attention(device: torch.device | str, attention_backend: str | None = None) -> str | None:
    """Configure attention for a device and return the pipeline's dispatch backend.

    Call after selecting the device, before building a pipeline or running a forward.
    Flex settings are shared by all models in the process. An explicit backend overrides
    dispatch, not the architecture-specific flex settings. CPU wrappers can use the
    initial defaults without calling this function.
    """
    device = torch.device(device)
    _FLEX_KERNEL_OPTIONS.clear()
    if device.type != "cuda" or not torch.cuda.is_available():
        # CPU-only host (unit tests, data prep, tooling) — auto SDPA.
        return attention_backend

    with torch.cuda.device(device):
        # Resolved for the runtime device; used as the default when --attention_backend is
        # not supplied (see train_utils.py).
        default_backend = _select_attention_config(device)
    return attention_backend if attention_backend is not None else default_backend
