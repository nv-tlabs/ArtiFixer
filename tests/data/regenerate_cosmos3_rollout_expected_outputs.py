# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate tests/data/cosmos3_rollout_expected_outputs.pt.

Run from the repo root, inside the pinned training container, after an
INTENTIONAL semantic change moves rollout values (the original generator —
the truncated-BPTT full-prefix oracle the KV cache replaced — no longer
exists, so the stored outputs are refreshed from the current implementation):

    PYTHONPATH=. python3 tests/data/regenerate_cosmos3_rollout_expected_outputs.py

The stored outputs only pin drift; they do not define correctness. Before
regenerating, the live equivalence tests in test_cosmos3_kv_cache_rollout.py
(cache hygiene, checkpointing parity, CP contract) must already be green.
Reuses the test module's own fixtures so the generated values are exactly
what the oracle tests recompute.
"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import tests.test_cosmos3_kv_cache_rollout as rollout


def main() -> None:
    transformer = rollout._reduced_real_transformer()
    rollout._activate_conditioning_adapters(transformer)
    kwargs = rollout._rollout_kwargs()
    noisy_chunks, clean_chunks = rollout._chunk_tensors(seed=7)

    wrapper_chunk_x0 = []
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        cache = transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
        for chunk_index in range(rollout.NUM_CHUNKS):
            wrapper_chunk_x0.append(
                rollout._cache_chunk_forward(transformer, kwargs, cache, noisy_chunks[chunk_index], chunk_index)
            )
            if chunk_index < rollout.NUM_CHUNKS - 1:
                rollout._seed_chunk(transformer, kwargs, cache, clean_chunks[chunk_index], chunk_index)

    inputs = rollout._pipeline_inputs()
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        pipeline_video_steps3 = rollout._run_pipeline_rollout(rollout._real_kv_cache_pipe(False), inputs)
    with torch.enable_grad(), sdpa_kernel([SDPBackend.MATH]):
        pipeline_video_steps2_grad = rollout._run_pipeline_rollout(
            rollout._real_kv_cache_pipe(False), inputs, num_inference_steps=2
        ).detach()
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        pipeline_video_steps3_exit_flag = rollout._run_pipeline_rollout(
            rollout._real_kv_cache_pipe(False), inputs, use_exit_flag=True
        )

    torch.save(
        {
            "wrapper_chunk_x0": wrapper_chunk_x0,
            "pipeline_video_steps3": pipeline_video_steps3,
            "pipeline_video_steps2_grad": pipeline_video_steps2_grad,
            "pipeline_video_steps3_exit_flag": pipeline_video_steps3_exit_flag,
        },
        rollout.EXPECTED_OUTPUTS_PATH,
    )
    print(f"wrote {rollout.EXPECTED_OUTPUTS_PATH}")


if __name__ == "__main__":
    main()
