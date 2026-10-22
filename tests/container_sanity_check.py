#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanity checks for the artifixer container."""

import importlib
import importlib.metadata
import importlib.util
import platform
import sys


def check(label, fn):
    try:
        result = fn()
        if result is False:
            print(f"  [FAIL] {label}: False")
            return False
        print(f"  [OK] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


failures = 0

# ── Core framework ──
print("\n=== Core Framework ===")
if not check("torch", lambda: f"{__import__('torch').__version__} (CUDA {__import__('torch').version.cuda})"):
    failures += 1
if not check("torchvision", lambda: __import__("torchvision").__version__):
    failures += 1

# ── GPU availability ──
print("\n=== GPU ===")
import torch

if not check("CUDA available", lambda: torch.cuda.is_available()):
    failures += 1
if torch.cuda.is_available():
    if not check("GPU name", lambda: torch.cuda.get_device_name(0)):
        failures += 1
    if not check(
        "GPU compute capability",
        lambda: f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}",
    ):
        failures += 1

# ── FlashAttention ──
print("\n=== FlashAttention ===")
# flash_attn namespace exists (created by FA4's flash_attn.cute) but has no __version__
# when FA2 wheel is not separately installed — that's fine for H100/GB200 targets.
check("flash_attn namespace", lambda: (__import__("flash_attn"), "importable (FA4 provides flash_attn.cute)")[1])

# FA3 - check both the package and the interface module.  FA3's Hopper kernels
# are x86/sm_90-only: the aarch64 (GB200/GB300) images deliberately ship FA4
# only, so FA3 is required only on x86_64.
if platform.machine() != "x86_64":
    print(f"  [SKIP] FA3 checks: FA3 is x86/Hopper-only (FA4-only image on {platform.machine()})")
elif importlib.util.find_spec("flash_attn_interface") is None:
    # Images built with BUILD_FLASH_ATTN_3=0 ship no FA3; Hopper falls back to SDPA.
    print("  [SKIP] FA3 checks: image built with BUILD_FLASH_ATTN_3=0 (Hopper-only kernels; SDPA is used)")
else:
    if not check("flash_attn_3 package", lambda: __import__("flash_attn_3").__name__):
        failures += 1
    if not check("flash_attn_interface (FA3 API)", lambda: (__import__("flash_attn_interface"), "importable")[1]):
        failures += 1

# FA4
fa4_ok = check("flash_attn.cute (FA4)", lambda: (importlib.import_module("flash_attn.cute"), "importable")[1])
if not fa4_ok:
    # FA4 might fail on non-Blackwell GPUs at import time - check if package is at least installed
    fa4_installed = check("flash-attn-4 installed (pip)", lambda: (importlib.metadata.version("flash-attn-4")))
    if platform.machine() != "x86_64" and not fa4_installed:
        # Non-x86 images ship no FA3, so a missing FA4 leaves no flash attention at all.
        failures += 1

# PyTorch FA backends
try:
    from torch.nn.attention import list_flash_attention_impls

    check("PyTorch FA backends", lambda: list_flash_attention_impls())
except ImportError:
    check("PyTorch FA backends", lambda: "list_flash_attention_impls not available")

# ── HuggingFace Libraries ──
print("\n=== HuggingFace Libraries ===")
if not check("diffusers", lambda: __import__("diffusers").__version__):
    failures += 1
if not check("transformers", lambda: __import__("transformers").__version__):
    failures += 1
if not check("accelerate", lambda: __import__("accelerate").__version__):
    failures += 1

# Check diffusers FA detection.  Informational: FA2 is never installed in
# these images, and FA3 is x86-only (its absence is already fatal above where
# it matters).
try:
    from diffusers.utils.import_utils import is_flash_attn_3_available, is_flash_attn_available

    check("diffusers FA detection", lambda: f"FA2={is_flash_attn_available()}, FA3={is_flash_attn_3_available()}")
except ImportError as e:
    print(f"  [WARN] diffusers FA detection: {e}")

# ── Diffusers internals used by our code ──
print("\n=== Diffusers Internals (used by artifixer) ===")
if not check(
    "dispatch_attention_fn",
    lambda: (importlib.import_module("diffusers.models.attention_dispatch").dispatch_attention_fn.__name__),
):
    failures += 1
if not check(
    "WanTransformer3DModel", lambda: (getattr(importlib.import_module("diffusers"), "WanTransformer3DModel").__name__)
):
    failures += 1
if not check("AutoencoderKLWan", lambda: (getattr(importlib.import_module("diffusers"), "AutoencoderKLWan").__name__)):
    failures += 1
if not check(
    "prompt_clean", lambda: (importlib.import_module("diffusers.pipelines.wan.pipeline_wan").prompt_clean.__name__)
):
    failures += 1
if not check(
    "retrieve_latents",
    lambda: (importlib.import_module("diffusers.pipelines.wan.pipeline_wan_i2v").retrieve_latents.__name__),
):
    failures += 1
if not check("VideoProcessor", lambda: (importlib.import_module("diffusers.video_processor").VideoProcessor.__name__)):
    failures += 1
if not check(
    "FP32LayerNorm", lambda: (importlib.import_module("diffusers.models.normalization").FP32LayerNorm.__name__)
):
    failures += 1
if not check(
    "WanAttnProcessor2_0",
    lambda: (
        getattr(
            importlib.import_module("diffusers.models.transformers.transformer_wan"), "WanAttnProcessor2_0"
        ).__name__
    ),
):
    failures += 1

# ── Training dependencies ──
print("\n=== Training Dependencies ===")
for pkg in [
    "einops",
    "scipy",
    "wandb",
    "tqdm",
    "PIL",
    "matplotlib",
    "cv2",
    "yaml",
    "torchmetrics",
    "imageio",
    "imageio_ffmpeg",
    "h5py",
    "torch_fidelity",
    "ftfy",
    "numpy",
]:
    if not check(pkg, lambda p=pkg: (importlib.import_module(p), "importable")[1]):
        failures += 1

# ── Thirdparty: 3dgrut ──
print("\n=== 3dgrut ===")
if not check("threedgrut", lambda: (importlib.import_module("threedgrut"), "importable")[1]):
    failures += 1
if not check(
    "threedgrut.datasets.camera_models",
    lambda: (importlib.import_module("threedgrut.datasets.camera_models"), "importable")[1],
):
    failures += 1

# ── Model training code imports ──
print("\n=== Model Training Code Imports ===")
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mt_path = os.path.join(repo_root, "model_training")
if os.path.exists(mt_path):
    # `python tests/container_sanity_check.py` puts tests/ on sys.path, not the repo root.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    for mod in [
        "model_training.utils.pose_utils",
        "model_training.utils.train_utils",
        "model_training.data.utils",
        "model_training.net.prope",
        "model_training.schedulers.flow_match",
    ]:
        if not check(mod, lambda m=mod: (importlib.import_module(m), "importable")[1]):
            failures += 1
else:
    print(f"  [SKIP] model_training not found at {mt_path}")

# ── FA3 training dispatch (artifixer autograd patch) ──
# Importing model_training.net.wan_transformer replaces diffusers' _flash_3
# backend with an autograd-capable version. Run a forward+backward through
# the same dispatch path training uses, so an FA3 API change fails here
# instead of on the first training step.
print("\n=== FA3 Training Dispatch ===")


def _fa3_dispatch_smoke():
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    importlib.import_module("model_training.net.wan_transformer")

    query = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    out = dispatch_attention_fn(query, key, value, attn_mask=None, is_causal=False, backend="_flash_3")
    out.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all(), "non-finite FA3 gradients"
    return "forward+backward through dispatch_attention_fn OK"


fa3_smoke_applicable = (
    platform.machine() == "x86_64"
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] == 9
    and os.path.exists(mt_path)
    and importlib.util.find_spec("flash_attn_interface") is not None
)
if fa3_smoke_applicable:
    if not check("FA3 dispatch forward+backward", _fa3_dispatch_smoke):
        failures += 1
else:
    print("  [SKIP] FA3 dispatch smoke: needs x86_64 + sm_90 GPU + model_training on path + an FA3 build")

# ── Summary ──
print(f"\n{'='*40}")
if failures:
    print(f"RESULT: {failures} check(s) FAILED")
    sys.exit(1)
else:
    print("RESULT: All checks passed!")
    sys.exit(0)
