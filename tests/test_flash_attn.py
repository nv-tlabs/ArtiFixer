# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validate FlashAttention 3 and FlashAttention 4 installations.

Run from a source checkout mounted inside the built container:
  python tests/test_flash_attn.py

On a CUDA 12 image (sm_90/H100 for FA3, sm_100/B200 for FA4 base):
  docker run --rm --gpus all -v "$PWD:/workspace/artifixer:ro" \
    -w /workspace/artifixer artifixer:cuda12 python tests/test_flash_attn.py

On a CUDA 13 image (full sm_121/GB200 FA4 support):
  docker run --rm --gpus all -v "$PWD:/workspace/artifixer:ro" \
    -w /workspace/artifixer artifixer:cuda13 python tests/test_flash_attn.py

Images contain the environment, not the application. Run the diagnostic from
the mounted checkout so it can test that revision's production attention policy.
"""

import importlib.util
import platform
import sys
from pathlib import Path

# This file is a container diagnostic, not a pytest suite: it needs a GPU,
# changes global torch attention state, and defines no test items. Skip it
# under pytest so collection works everywhere, including CPU login nodes.
if "pytest" in sys.modules:
    import pytest

    pytest.skip(
        "standalone container diagnostic; run as `python tests/test_flash_attn.py`",
        allow_module_level=True,
    )

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def main() -> int:
    errors = []

    def check(label, fn):
        try:
            result = fn()
            if result is False:
                raise AssertionError("returned False")
            print(f"  [{PASS}] {label}" + (f": {result}" if result is not None else ""))
        except Exception as e:
            print(f"  [{FAIL}] {label}: {e}")
            errors.append(label)

    def skip(label, reason):
        print(f"  [{SKIP}] {label}: {reason}")

    # -----------------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------------
    print("\n=== Environment ===")
    import torch

    check("torch version", lambda: torch.__version__)
    check(
        "CUDA available",
        lambda: torch.cuda.is_available() and f"{torch.cuda.is_available()} — {torch.cuda.get_device_name(0)}",
    )
    check("CUDA version (torch)", lambda: torch.version.cuda)
    check("cuDNN version", lambda: torch.backends.cudnn.version())

    try:
        import nvidia.cudnn_frontend

        check("cuDNN frontend version", lambda: nvidia.cudnn_frontend.__version__)
    except ImportError:
        try:
            import cudnn

            check("cuDNN frontend version", lambda: cudnn.__version__)
        except ImportError:
            skip("cuDNN frontend", "not installed")

    print("\n=== Repository Attention Policy ===")
    if torch.cuda.is_available():
        try:
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            from model_training.common.context_parallel import frame_block_causal_mask
            from model_training.net.attention_config import configure_attention, flex_block_size
            from model_training.net.cosmos3_transformer import flex_attention as cosmos3_flex_attention
            from model_training.net.wan_transformer import flex_attention as wan_flex_attention

            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            backend = configure_attention(device)
            check("configured dispatch and block size", lambda: f"backend={backend!r}, block_size={flex_block_size()}")

            def check_gradients(output, inputs):
                assert bool(torch.isfinite(output).all())
                output.float().square().mean().backward()
                torch.cuda.synchronize(device)
                for tensor in inputs:
                    assert tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
                    assert bool(tensor.grad.ne(0).any())
                return f"output shape {tuple(output.shape)}; finite nonzero Q/K/V gradients"

            def dispatch_backward(compiled):
                inputs = [
                    torch.randn(1, 512, heads, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
                    for heads in (4, 2, 2)
                ]

                def attention(query, key, value):
                    return dispatch_attention_fn(query, key, value, backend=backend, enable_gqa=True)

                attention_fn = torch.compile(attention, fullgraph=True) if compiled else attention
                return check_gradients(attention_fn(*inputs), inputs)

            def flex_backward(attention_fn):
                block_size = flex_block_size()
                block_mask = frame_block_causal_mask(
                    device=device,
                    num_frames=4,
                    frame_seqlen=128,
                    frames_per_block=1,
                    block_size=block_size,
                )
                inputs = [
                    torch.randn(1, heads, 512, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
                    for heads in (4, 2, 2)
                ]
                return check_gradients(attention_fn(*inputs, block_mask=block_mask, enable_gqa=True), inputs)

            check("configured dispatch forward/backward (bf16 GQA)", lambda: dispatch_backward(False))
            check("compiled configured dispatch forward/backward (bf16 GQA)", lambda: dispatch_backward(True))
            check("Wan configured flex forward/backward (bf16 GQA)", lambda: flex_backward(wan_flex_attention))
            check("Cosmos3 configured flex forward/backward (bf16 GQA)", lambda: flex_backward(cosmos3_flex_attention))
        except Exception as e:
            check("repository attention policy", lambda e=e: (_ for _ in ()).throw(e))
    else:
        skip("repository attention policy", "no CUDA device")

    # -----------------------------------------------------------------------
    # PyTorch native FA backends (PyTorch 2.11+)
    # -----------------------------------------------------------------------
    print("\n=== PyTorch Native Flash Attention Backends ===")
    try:
        from torch.nn.attention import (
            activate_flash_attention_impl,
            current_flash_attention_impl,
            list_flash_attention_impls,
        )

        impls = list_flash_attention_impls()
        check("available FA implementations", lambda: impls)
        if platform.machine() != "x86_64":
            # FA3's Hopper kernels are x86-only; the aarch64 (GB200/GB300)
            # images deliberately ship FA4 only.
            skip("FA3 backend registered", f"FA3 is x86/Hopper-only (machine: {platform.machine()})")
        elif importlib.util.find_spec("flash_attn_interface") is None:
            skip("FA3 backend registered", "image built with BUILD_FLASH_ATTN_3=0 (Hopper-only kernels; SDPA is used)")
        else:
            check("FA3 backend registered", lambda: "FA3" in impls)

        if torch.cuda.is_available():
            # Test FA3 activation
            if "FA3" in impls:
                try:
                    activate_flash_attention_impl("FA3")
                    check("FA3 backend activation", lambda: f"active impl = {current_flash_attention_impl()}")
                except Exception as e:
                    skip("FA3 backend activation", str(e))

            # Test FA4 activation
            if "FA4" in impls:
                try:
                    activate_flash_attention_impl("FA4")
                    check("FA4 backend activation", lambda: f"active impl = {current_flash_attention_impl()}")
                except Exception as e:
                    skip("FA4 backend activation", f"{e} (expected on non-Blackwell GPUs)")
        else:
            skip("FA backend activation", "no CUDA device")
    except ImportError as e:
        check("torch.nn.attention FA API", lambda e=e: (_ for _ in ()).throw(e))

    # -----------------------------------------------------------------------
    # FlexAttention with FA backends
    # -----------------------------------------------------------------------
    print("\n=== FlexAttention Forward Pass ===")
    if torch.cuda.is_available():
        try:
            from torch.nn.attention.flex_attention import flex_attention

            B, H, S, D = 1, 4, 64, 64
            q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
            out = flex_attention(q, k, v)
            check("flex_attention forward pass (bf16)", lambda: f"output shape {tuple(out.shape)}")
            active = current_flash_attention_impl() if "current_flash_attention_impl" in dir() else "unknown"
            check("flex_attention using backend", lambda: active)
        except Exception as e:
            check("flex_attention forward pass", lambda e=e: (_ for _ in ()).throw(e))
    else:
        skip("flex_attention forward pass", "no CUDA device")

    # -----------------------------------------------------------------------
    # FlashAttention 4 package (flash_attn.cute)
    # -----------------------------------------------------------------------
    print("\n=== FlashAttention 4 Package (flash_attn.cute) ===")

    try:
        spec = importlib.util.find_spec("flash_attn.cute") or importlib.util.find_spec("flash_attn_4")
        check("flash-attn-4 package installed", lambda: f"found at {spec.origin}" if spec else "NOT FOUND")
        if spec is None:
            errors.append("flash-attn-4 package not installed")
    except Exception as e:
        check("flash-attn-4 package check", lambda e=e: (_ for _ in ()).throw(e))

    if torch.cuda.is_available():
        try:
            import flash_attn.cute as fa4

            check("flash_attn.cute importable", lambda: "ok")

            if hasattr(fa4, "_flash_attn_fwd"):
                check("FA4 _flash_attn_fwd available", lambda: "ok")
            elif hasattr(fa4, "flash_attn_func"):
                check("FA4 flash_attn_func available", lambda: "ok")
            else:
                available = [x for x in dir(fa4) if not x.startswith("_")]
                skip("FA4 forward function", f"exports: {available[:10]}")
        except ImportError as e:
            skip("flash_attn.cute import", f"{e} (requires GPU context)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    if errors:
        print(f"  [{FAIL}] {len(errors)} check(s) failed:")
        for e in errors:
            print(f"         - {e}")
        return 1
    print(f"  [{PASS}] All checks passed.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
