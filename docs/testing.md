<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Testing reference

Run commands from the repository root and record passes, failures, and skips.
Use a [project Docker environment](setup.md), or an environment matching
its PyTorch and [Python dependency versions](../docker/requirements.txt), with
pytest installed. Install `zstandard` for the zstd cases in
`tests/test_saved_image.py`. Run the full suite with:

```bash
python -m pytest tests -q
```

For a faster subset, use `python -m pytest tests -q -m 'not slow'`. The full
suite includes [reduced Cosmos3 model tests](../tests/test_cosmos3_real_model_contract.py)
marked `slow`.

Complete validation also requires ShellCheck on `PATH` and the Cosmos3-Nano
`text_tokenizer` assets in the local Hugging Face cache at the revision in
[the pin registry](../model_training/pins.py).

The [pretrained LPIPS integration check](../tests/test_lpips_integration.py)
requires the test-only reference package `lpips==0.1.4` and pretrained VGG16
and LPIPS v0.1 weights. The first run downloads missing weights into the
PyTorch cache; set `TORCH_HOME` to reuse an existing cache. Run it on CPU with:

```bash
python -m pip install lpips==0.1.4
python -m pytest tests/test_lpips_integration.py -q -s
```

The check compares fixed RGB PNG pairs without resizing or cropping, verifies
that both implementations loaded identical weights, and accepts standard
LPIPS differences within `atol=1e-6, rtol=1e-5`. It also checks that legacy
mode reproduces the `[0, 1]`-input scores and that PSNR/SSIM are unchanged. It prints
the path to a JSON scoring receipt in pytest's temporary directory.

Tests marked `requires_host_ffmpeg` need an FFmpeg executable on `PATH` or
at `IMAGEIO_FFMPEG_EXE`; otherwise they skip. See [runtime setup](setup.md#ffmpeg).

Run the DMD normalization checks separately with four CPU processes and Gloo.
An explicit loopback rendezvous also works in containers without hostname DNS;
choose another `--master_port` if 29571 is already in use:

```bash
python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
    --master_addr 127.0.0.1 --master_port 29571 -m pytest \
    tests/test_dmd_training_batch.py -q -k normalization_matches_full_video
```

Distributed GPU attention tests need at least two GPUs in the project container:

```bash
bash tests/run_cp_attention_tests.sh 2
```

Also exercise Cosmos3 block-causal forward/backward on CUDA, including
bfloat16 autocast and gradient checkpointing. This reaches compiled
`flex_attention`; the CPU cases use dense SDPA instead:

```bash
python -m pytest tests/test_cosmos3_flex_autocast.py -q -s
python -m pytest tests/test_cosmos3_real_model_contract.py -q -k 'block_causal_gradients and cuda'
```

The flex-autocast module configures the repository's production backend and
checks its block geometry. Hopper/Blackwell runs require FA4; the FP32 control
uses Triton and math SDPA. The mixed-Q/K/V fixtures use math SDPA for unmasked
attention, since FA3 requires homogeneous half inputs; the native FSDP2 smoke
below exercises the default complete pipeline. Run on both architectures for
backend qualification.
With two CUDA devices, the same command also runs masked BF16 CP1/CP2 output
and gradient comparisons, including no-grad and checkpointed backward, plus
FSDP2 with the native BF16 parameter-compute policy. The first three CP cases
assert mixed FP32 Q/K and BF16 V; the FSDP2 case compares matching BF16 compute.
Each differentiable layout must preserve the pre-cast call's outputs and
gradients exactly. CP1/CP2 gradients additionally use tensor-scale maximum
error and relative L2 bounds of two BF16 epsilons (`0.015625`), accounting for
different partial-gradient rounding; missing and nonfinite gradients fail.
Run plain pytest so those tests can spawn their own NCCL workers.

For the full-sequence versus cached-replay diagnostic, use a new output file
for each source revision and backend:

```bash
python -m tests.cosmos3_cache_parity_probe --source-revision "$(git rev-parse HEAD)" \
    --backend production --dtype bf16 --attention-backend _native_math \
    --output /path/to/new-cache-comparison.json
```

The probe sets FP32 matmul precision to `highest` (TF32 off), compares the
concatenated replay output and each layer's first clean-chunk K tensor with
a joint clean/noisy reference calculation in
[`tests/_teacher_forcing_reference.py`](../tests/_teacher_forcing_reference.py).
The probe records every mismatch at `rtol=0.02`, `atol=0.01` and exits nonzero
on a mismatch. Use `--backend triton --dtype fp32` for the FP32 control.
`--attention-backend` controls unmasked dispatch separately
from `--backend`, which selects the masked implementation. Use `_native_flash`
for a BF16 unmasked flash comparison; omitting the option selects the runtime
default, which may reject FP32-master fixture inputs before masked attention.

Run the shared-gather and Cosmos3 CP1/CP2 parity cases with at least two CUDA
devices. These tests spawn their own NCCL groups: run plain pytest, not
`torchrun`, which would make them skip their nested process groups. They cover
non-contiguous attention gradients, reference head slicing, padded/local and
checkpointed forwards, and both cached layouts and history-detach modes. They
also compare DMD input routing from the training dataset's per-rank shards
against full-clip preprocessing using a fixture VAE:

```bash
python -m pytest tests/test_causal_training.py tests/test_cosmos3_real_model_contract.py \
    tests/test_cosmos3_kv_cache_rollout.py tests/test_dmd_training_batch.py -q -k nccl
```

The single-tensor SDPA-backward regression also runs without GPUs:

```bash
python -m pytest tests/test_causal_training.py -q -k 'all_gather_cat_with_grad_sdpa and gloo'
```

Run the standalone attention diagnostic in the GPU container:

```bash
python tests/test_flash_attn.py
```

The [stage-3 smoke test](../tests/stage3_smoke.py) runs six updates per model
family through the normal training entrypoint on two GPUs (bf16, FSDP2, CP=2).
It uses reduced real models and synthetic data to check student/critic gradients,
EMA, and checkpoint loading/saving. Use a fresh scratch directory for each run:

```bash
set -e
SMOKE_ROOT=/path/to/scratch/stage3-smoke
for family in wan cosmos3; do
    python -m tests.stage3_smoke "$SMOKE_ROOT/$family" --family "$family" --prepare
    accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
        --machine_rank 0 --mixed_precision bf16 --dynamo_backend no \
        --main_process_port 29571 --module tests.stage3_smoke \
        "$SMOKE_ROOT/$family" --family "$family"
done
```

The [validation-only smoke](../tests/validation_smoke.py) exercises actual DCP
loading, FSDP2 sharding, CP inference, complete reports and fixed-panel replay
for reduced Wan and Cosmos3 models. It reuses the stage-3 models, synthetic VAE,
and dataset fixtures. Use two GPUs, host FFmpeg, and a fresh output root:

```bash
set -e
SMOKE_ROOT=/path/to/scratch/validation-smoke
for family in wan cosmos3; do
    python -m tests.validation_smoke "$SMOKE_ROOT/$family" --family "$family" --prepare
    for path in standard kv; do
        for replay in "" --replay; do
            accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
                --machine_rank 0 --mixed_precision bf16 --dynamo_backend no \
                --main_process_port 29641 --module tests.validation_smoke \
                "$SMOKE_ROOT/$family" --family "$family" --path "$path" $replay
        done
    done
done
```

Formatting follows [pyproject.toml](../pyproject.toml), using **Black 24.8.0** and
**isort 5.13.2**, pinned in `docker/requirements.txt`:

```bash
python -m black --check .
python -m isort --check-only .
```
