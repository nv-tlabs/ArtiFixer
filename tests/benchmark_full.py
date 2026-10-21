# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Full attention benchmark: all backends x {eager, torch.compile} x {fwd, fwd+bwd}
Plus end-to-end transformer block benchmark.

Usage:
  docker run --gpus all ... -v ...:/workspace/artifixer:ro \
    -w /workspace/artifixer artifixer:cuda12 python tests/benchmark_full.py
"""

import time

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
    list_flash_attention_impls,
    restore_flash_attention_impl,
)
from torch.nn.attention.flex_attention import flex_attention as _flex_orig

from model_training.net.attention_config import configure_attention, flex_block_size

# -------------------------------------------------------------------------
WARMUP = 10
ITERS = 50
SHAPES = [
    (2, 40, 4096, 128),
]
DTYPE = torch.bfloat16

device = torch.device("cuda:0")
torch.cuda.set_device(device)
attention_backend = configure_attention(device)
gpu_name = torch.cuda.get_device_name(device)
cc = torch.cuda.get_device_capability(device)
print(f"GPU: {gpu_name}  cc: {cc[0]}.{cc[1]}")
print(f"FA impls: {list_flash_attention_impls()}")
print(f"Active FA: {current_flash_attention_impl()}")
print(f"Default dispatch backend: {attention_backend!r}")
print(f"Flex block size: {flex_block_size()}")
print()


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def try_bench(fn):
    try:
        return f"{bench(fn):.2f}"
    except TypeError:
        # A TypeError usually means a call in this file no longer matches the
        # current forward signature. Re-raise it; an ERR row would hide that.
        raise
    except Exception as e:
        return f"ERR: {str(e)[:40]}"


# =========================================================================
# Part 1: dispatch_attention_fn — all backends x eager
# =========================================================================
print("=" * 90)
print("PART 1: dispatch_attention_fn (eager)")
print("=" * 90)

dispatch_backends = [
    ("None (auto, no FA)", None, None),
    ("None (auto, FA4 active)", None, "FA4"),
    ("_native_flash (no FA)", "_native_flash", None),
    ("_native_flash (FA4 active)", "_native_flash", "FA4"),
    ("_native_cudnn", "_native_cudnn", None),
    ("_native_math", "_native_math", None),
]

# Add _flash_3 if available
try:
    from diffusers.utils.import_utils import is_flash_attn_3_available

    if is_flash_attn_3_available():
        dispatch_backends.append(("_flash_3 (FA3 Hopper)", "_flash_3", None))
except Exception:
    pass

for B, H, S, D in SHAPES:
    print(f"\n--- B={B} H={H} S={S} D={D} ---")
    print(f"{'Backend':<35} {'Fwd (ms)':>10} {'Fwd+Bwd (ms)':>12}")
    print("-" * 60)

    for label, backend, fa_impl in dispatch_backends:
        try:
            restore_flash_attention_impl(_raise_warn=False)
        except Exception:
            pass
        if fa_impl:
            try:
                activate_flash_attention_impl(fa_impl)
            except Exception:
                print(f"{label:<35} {'SKIP':>10}")
                continue

        q = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")
        k = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")
        v = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")

        fwd = try_bench(
            lambda: dispatch_attention_fn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, backend=backend)
        )

        def make_fwd_bwd(q, k, v, backend):
            def fn():
                q_ = q.detach().requires_grad_(True)
                k_ = k.detach().requires_grad_(True)
                v_ = v.detach().requires_grad_(True)
                dispatch_attention_fn(
                    q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False, backend=backend
                ).sum().backward()

            return fn

        fwdbwd = try_bench(make_fwd_bwd(q, k, v, backend))
        print(f"{label:<35} {fwd:>10} {fwdbwd:>12}")

# Restore FA4 for subsequent tests
try:
    restore_flash_attention_impl(_raise_warn=False)
except Exception:
    pass
try:
    activate_flash_attention_impl("FA4")
except Exception as e:
    print(f"FA4 activation SKIP: {e} (expected on non-Blackwell GPUs)")
print(f"Active FA impl for Parts 2-4: {current_flash_attention_impl()}")


# =========================================================================
# Part 2: dispatch_attention_fn — torch.compile
# =========================================================================
print()
print("=" * 90)
print("PART 2: dispatch_attention_fn (torch.compile)")
print("=" * 90)

compile_backends = [
    ("None (auto, FA4 active)", None),
    ("_flash_3 (FA3 Hopper)", "_flash_3"),
    ("_native_flash (FA4 active)", "_native_flash"),
]

for B, H, S, D in SHAPES:
    print(f"\n--- B={B} H={H} S={S} D={D} ---")
    print(f"{'Backend':<35} {'Fwd (ms)':>10} {'Fwd+Bwd (ms)':>12}")
    print("-" * 60)

    for label, backend in compile_backends:
        q = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")
        k = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")
        v = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")

        try:
            compiled_fn = torch.compile(
                lambda q, k, v: dispatch_attention_fn(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, backend=backend
                ),
                dynamic=False,
            )
            # Warm up compile
            for _ in range(3):
                compiled_fn(q, k, v)
            torch.cuda.synchronize()

            fwd = try_bench(lambda: compiled_fn(q, k, v))

            def make_compiled_fwd_bwd(q, k, v, compiled_fn):
                def fn():
                    q_ = q.detach().requires_grad_(True)
                    k_ = k.detach().requires_grad_(True)
                    v_ = v.detach().requires_grad_(True)
                    compiled_fn(q_, k_, v_).sum().backward()

                return fn

            fwdbwd = try_bench(make_compiled_fwd_bwd(q, k, v, compiled_fn))
        except Exception as e:
            fwd = f"ERR: {str(e)[:35]}"
            fwdbwd = ""

        print(f"{label:<35} {fwd:>10} {fwdbwd:>12}")


# =========================================================================
# Part 3: flex_attention — all backends x eager vs compile
# =========================================================================
print()
print("=" * 90)
print("PART 3: flex_attention")
print("=" * 90)

for B, H, S, D in SHAPES:
    print(f"\n--- B={B} H={H} S={S} D={D} ---")
    print(f"{'Backend':<45} {'Fwd (ms)':>10} {'Fwd+Bwd (ms)':>12}")
    print("-" * 70)

    qf = torch.randn(B, H, S, D, dtype=DTYPE, device="cuda")
    kf = torch.randn(B, H, S, D, dtype=DTYPE, device="cuda")
    vf = torch.randn(B, H, S, D, dtype=DTYPE, device="cuda")

    flex_configs = [
        ("Triton AUTO (compiled)", {}, True),
        ("FA4 CuTeDSL FLASH (compiled)", {"BACKEND": "FLASH"}, True),
        ("Triton AUTO (eager)", {}, False),
        ("FA4 CuTeDSL FLASH (eager)", {"BACKEND": "FLASH"}, False),
    ]

    for label, kopts, do_compile in flex_configs:
        try:
            if do_compile:
                fn = torch.compile(
                    lambda q, k, v: _flex_orig(q, k, v, kernel_options=kopts),
                    dynamic=False,
                )
            else:
                fn = lambda q, k, v: _flex_orig(q, k, v, kernel_options=kopts)

            # Warm up
            for _ in range(3):
                fn(qf, kf, vf)
            torch.cuda.synchronize()

            fwd = try_bench(lambda: fn(qf, kf, vf))

            def make_flex_bwd(qf, kf, vf, fn):
                def bwd():
                    q_ = qf.detach().requires_grad_(True)
                    k_ = kf.detach().requires_grad_(True)
                    v_ = vf.detach().requires_grad_(True)
                    fn(q_, k_, v_).sum().backward()

                return bwd

            fwdbwd = try_bench(make_flex_bwd(qf, kf, vf, fn))
        except Exception as e:
            fwd = f"ERR: {str(e)[:40]}"
            fwdbwd = ""

        print(f"{label:<45} {fwd:>10} {fwdbwd:>12}")


# =========================================================================
# Part 4: End-to-end ArtifixerTransformer block
# =========================================================================
print()
print("=" * 90)
print("PART 4: End-to-end ArtifixerTransformerBlock")
print("=" * 90)

from diffusers import WanTransformer3DModel

from model_training.net.wan_transformer import ArtifixerTransformerBlock

# Build a single block
transformer = WanTransformer3DModel.from_config(
    {
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
        "num_layers": 1,
        "cross_attention_dim": 4096,
        "patch_size": [1, 2, 2],
        "norm_type": "layer_norm",
    }
)

# Set the attention backend
attention_backend = configure_attention(device)
if attention_backend is not None:
    transformer.set_attention_backend(attention_backend)

base_block = transformer.blocks[0]
block = (
    ArtifixerTransformerBlock(
        base_block,
        opacity_embedding_dim=0,
        camera_embedding_dim=0,
        use_kv_cache_attention=False,
    )
    .cuda()
    .to(DTYPE)
)

# Create dummy inputs matching the block's forward signature
B_e2e = 2
S_e2e = 4096
inner_dim = 40 * 128  # heads * head_dim
cross_dim = 4096

hidden = torch.randn(B_e2e, S_e2e, inner_dim, dtype=DTYPE, device="cuda")
encoder_hidden = torch.randn(B_e2e, 256, cross_dim, dtype=DTYPE, device="cuda")
temb = torch.randn(B_e2e, 6, inner_dim, dtype=DTYPE, device="cuda")

# RoPE: (1, S, 1, head_dim)
rope_cos = torch.randn(1, S_e2e, 1, 128, dtype=DTYPE, device="cuda")
rope_sin = torch.randn(1, S_e2e, 1, 128, dtype=DTYPE, device="cuda")
rotary_emb = (rope_cos, rope_sin)

print(f"\nBlock config: B={B_e2e} S={S_e2e} heads=40 head_dim=128")
print(f"Attention backend: {attention_backend!r}")
print(f"Flex block size: {flex_block_size()}")
print()


# Call with keyword arguments so a change to ArtifixerTransformerBlock.forward's
# signature raises a TypeError here instead of binding values to the wrong
# parameters.
def run_block(hidden_states, block_mask=None):
    return block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden,
        neighbor_hidden_states=None,
        ignore_neighbors=True,
        temb=temb,
        rotary_emb=rotary_emb,
        opacity_extra_patches=None,
        camera_extra_patches=None,
        kv_cache=None,
        layer_index=0,
        kv_cache_num_history_chunks=0,
        kv_cache_stage=None,
        context_kv_cache=None,
        prope_target=None,
        prope_reference=None,
        block_mask=block_mask,
    )


# Without block_mask (dispatch_attention_fn path only)
print("--- Without block_mask (dispatch_attention_fn only) ---")
print(f"{'Mode':<35} {'Fwd (ms)':>10} {'Fwd+Bwd (ms)':>12}")
print("-" * 60)


def block_fwd():
    run_block(hidden)


def block_fwd_bwd():
    h = hidden.detach().requires_grad_(True)
    run_block(h).sum().backward()


fwd = try_bench(block_fwd)
fwdbwd = try_bench(block_fwd_bwd)
print(f"{'Eager':<35} {fwd:>10} {fwdbwd:>12}")

# With block_mask (flex_attention path)
from torch.nn.attention.flex_attention import create_block_mask


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


bmask = create_block_mask(causal_mask, B=None, H=None, Q_LEN=S_e2e, KV_LEN=S_e2e, device="cuda")

print()
print("--- With block_mask (flex_attention + dispatch_attention_fn) ---")
print(f"{'Mode':<35} {'Fwd (ms)':>10} {'Fwd+Bwd (ms)':>12}")
print("-" * 60)


def block_fwd_mask():
    run_block(hidden, block_mask=bmask)


def block_fwd_bwd_mask():
    h = hidden.detach().requires_grad_(True)
    run_block(h, block_mask=bmask).sum().backward()


fwd = try_bench(block_fwd_mask)
fwdbwd = try_bench(block_fwd_bwd_mask)
print(f"{'Eager (flex=FA4 CuTeDSL)':<35} {fwd:>10} {fwdbwd:>12}")

print()
print("=" * 90)
print("Done.")
