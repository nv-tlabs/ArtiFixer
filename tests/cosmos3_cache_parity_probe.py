# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare teacher-forced output/clean K with a two-chunk cached replay.

Run as a module. A nonzero exit status means the recorded tolerance was missed;
this diagnostic is separate from the attention regression suite.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import model_training.net.cosmos3_transformer as cosmos
from model_training.net import attention_config
from tests._helpers import _activate_conditioning_adapters, _joint_reference_kwargs, _reduced_real_transformer
from tests._teacher_forcing_reference import cosmos3_teacher_forcing_reference
from tests.test_cosmos3_kv_cache_rollout import _cache_chunk_forward, _chunk_slice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["production", "triton"], default="production")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attention-backend", choices=["_native_math", "_native_flash"])
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new evidence path")
    if args.dtype == "fp32" and args.backend != "triton":
        parser.error("the FP32 control requires --backend triton")
    if args.dtype == "fp32" and args.attention_backend not in (None, "_native_math"):
        parser.error("the FP32 control requires math SDPA for unmasked attention")
    torch.set_float32_matmul_precision("highest")
    assert not torch.backends.cuda.matmul.allow_tf32
    device = torch.device("cuda")
    backend = attention_config.configure_attention(device, attention_backend=args.attention_backend)
    if args.backend == "triton":
        attention_config._FLEX_KERNEL_OPTIONS.clear()
        if args.dtype == "fp32":
            backend = "_native_math"
    elif torch.cuda.get_device_capability(device)[0] >= 9:
        assert attention_config._FLEX_KERNEL_OPTIONS.get("BACKEND") == "FLASH"
    teacher = _reduced_real_transformer(causal_block_frames=2)
    _activate_conditioning_adapters(teacher)
    teacher = teacher.cuda()
    teacher.set_attention_backend(backend)
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in _joint_reference_kwargs(total_frames=4).items()}
    clean = torch.randn_like(inputs["hidden_states"])
    inputs["timestep"] = torch.tensor([[800.0, 400.0]], device=device).repeat_interleave(8, dim=1)
    replay = _reduced_real_transformer().cuda()
    replay.load_state_dict(teacher.state_dict())
    replay.set_attention_backend(backend)
    keys = {}
    handles = [
        layer.register_forward_hook(lambda m, a, out, i=i: keys.__setitem__(i, out[1].detach().clone()))
        for i, layer in enumerate(teacher.scope)
    ]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.dtype == "bf16"):
        expected = cosmos3_teacher_forcing_reference(teacher, **inputs, clean_x=clean)[0]
        for handle in handles:
            handle.remove()
        cache = replay.new_rollout_kv_cache(truncate_rollout_gradients=False)
        outputs = []
        for chunk in range(2):
            start = chunk * 2
            outputs.append(
                _cache_chunk_forward(replay, inputs, cache, inputs["hidden_states"][:, :, start : start + 2], chunk)
            )
            if chunk == 0:
                seed = _chunk_slice(inputs, 0, 2)
                seed["timestep"] = None
                replay(
                    **seed, hidden_states=clean[:, :, :2], kv_cache=cache, kv_cache_seed=True, kv_cache_start_frame=0
                )

    def compare(observed, reference):
        assert torch.isfinite(observed).all() and torch.isfinite(reference).all(), "nonfinite cache comparison"
        difference = (observed - reference).abs().float()
        return {
            "shape": list(observed.shape),
            "dtype": str(observed.dtype),
            "count": observed.numel(),
            "max_abs": difference.max().item(),
            "outside_tolerance": int((difference > 0.01 + 0.02 * reference.abs()).sum()),
        }

    references = inputs["neighbor_hidden_states"].shape[2] * 4
    comparisons = {"output": compare(torch.cat(outputs, dim=2), expected)}
    for index, key in keys.items():
        comparisons[f"layer_{index}_clean_K"] = compare(
            cache.read_gen_history(index, 1)[0][0], key[references : references + 8]
        )
    report = {
        "source_revision": args.source_revision,
        "transformer_source_sha256": hashlib.sha256(Path(cosmos.__file__).read_bytes()).hexdigest(),
        "reference_source_sha256": hashlib.sha256(
            Path(__file__).with_name("_teacher_forcing_reference.py").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "dispatch_backend": backend,
        "flex_options": dict(attention_config._FLEX_KERNEL_OPTIONS),
        "block_size": attention_config.flex_block_size(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "rtol": 0.02,
        "atol": 0.01,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return int(any(row["outside_tolerance"] for row in comparisons.values()))


if __name__ == "__main__":
    raise SystemExit(main())
