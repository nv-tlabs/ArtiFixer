<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Environment

The [quick start](../README.md#quick-start) uses a prebuilt CUDA 12 x86 image
or builds it locally.
`Dockerfile.cuda13` is the other x86 option; use `Dockerfile.cuda13-aarch64`
for ARM64 systems such as GB200. All images require a compatible host NVIDIA
driver and GPU-enabled container runtime. Initialize recursive Git submodules
before building.

## Prebuilt images

The following environments are published to the GitHub Container Registry and
can be pulled without authentication. You can also build from the Dockerfiles
instead.

All tags below use `ghcr.io/nv-tlabs/artifixer:` as their prefix.

| Platform | Tag | Registry digest |
| --- | --- | --- |
| CUDA 12, x86 (`linux/amd64`) | `cuda12-ffmpeg-free-20261021-5b1bb939` | `sha256:9c5aa877cbc2ff1be3c06d58296768b88d7ee5ac3ee5b244afc729a1f79a5d23` |
| CUDA 13, x86 (`linux/amd64`) | `cuda13-ffmpeg-free-20261021-5b1bb939` | `sha256:2c5348ec87a5dd83cc345ca84945591c2252abfd768558ee3fc320d407856d32` |
| CUDA 13, ARM64 (`linux/arm64`) | `cuda13-aarch64-ffmpeg-free-20261021-5b1bb939` | `sha256:cfe05bf0a9ce3aa9753cc42442c8c30670db117f517665902b795e55c223ed08` |

For an immutable CUDA 12 reference, use:

```bash
export ARTIFIXER_IMAGE=ghcr.io/nv-tlabs/artifixer@sha256:9c5aa877cbc2ff1be3c06d58296768b88d7ee5ac3ee5b244afc729a1f79a5d23
docker pull --platform linux/amd64 "$ARTIFIXER_IMAGE"
```

Use the selected image in the [quick-start container command](../README.md#1-set-up-the-environment).
Mount this source checkout and provide [FFmpeg](#ffmpeg). The images contain
the dependency environment and 3DGRUT; ArtiFixer source and model weights are
loaded separately. Matching [component notices](../docker/notices/README.md)
are available for each image.

## Hardware

These image builds were tested on H100 (x86), GB200 and GB300 (ARM64). A100 uses PyTorch SDPA
instead of FA3. GPU requirements depend on model size, image resolution,
reference-view count, and the workflow.

The Wan 14B checkpoint needs roughly 34 GB for BF16 transformer weights. Approximate
single-GPU KV-cache allocations, excluding activations and VAE buffers, are:

| Frame size (W × H) | Window of 21 latent frames | Each reference view |
| --- | --- | --- |
| 960 × 544 | 35 GB | 1.7 GB |
| 1024 × 576 | 40 GB | 1.9 GB |
| 1920 × 1080 | 140 GB | 6.7 GB |

At 960 × 544, 14B weights plus caches total about 79 GB with six references and
89 GB with the default 12 for `reconstructed_colmap`. Allow additional memory
for activations and VAE buffers. The 1.3B model uses about 3.4 GB
for BF16 transformer weights and roughly 4.5 times less cache memory.

The Cosmos3-Nano model has 15.8B transformer parameters: approximately
31.6 GB of BF16 weights and a 63.2 GB FP32 checkpoint file. Those figures exclude
the VAE, activations, references, and KV cache; a 24 GB GPU cannot hold the
transformer alone. The Wan cache estimates above do not describe Cosmos.
Both exported Cosmos3-Nano S1 and S3 checkpoints were evaluated on H100 80 GB
using the DL3DV test set. Minimum-VRAM requirements have not been measured.

Released Wan2.1 inference keeps native frame sizes, snapped to multiples of 16.
Cosmos3 and Wan2.2 use multiples of 32. Downscale large
captures before scene preparation or reduce `--num_views`. Defaults are up to
12 references for `reconstructed_colmap` and `3dgrut_dl3dv_difix`, and six for
`3dgrut_dl3dv_ours` and `nerfbusters`. Match geometry to training; the released
models use `--frames_per_block 7 --local_attn_size 21 --sink_size 7`.

`--vae_tiling` reduces VAE memory at the cost of speed and small output
differences. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can reduce
allocator peaks. Host conditioning and output memory still grow with clip
length. Single-file checkpoints are memory-mapped and loaded tensor by tensor;
their FP32 files are about 67.6 GB for 14B and 6.7 GB for 1.3B. Allow roughly
160 GB of disk for an image, checkpoint, and captioner; datasets are additional.

The default Qwen3-VL-30B-A3B captioner needs about 60 GB of GPU memory in BF16.
Use `--captioning_model_id Qwen/Qwen3-VL-4B-Instruct` for the smaller captioner.
Wan caption preparation also downloads a shared T5 text encoder (about 23 GB),
including when inference uses the 1.3B checkpoint. Keep the model cache mounted
to reuse that download on later runs.

## Build options

The x86 Dockerfiles compile FA3 for Hopper. Skip the FA3 build for A100 and Blackwell:

```bash
docker build -f Dockerfile.cuda12 \
    --build-arg BUILD_FLASH_ATTN_3=0 -t artifixer:cuda12 .
```

H100 falls back to SDPA if FA3 is absent.

FA3 compilation is CPU- and memory-intensive. For `docker build`, pass
`--build-arg FLASH_ATTN_MAX_JOBS=N` (default 16); each job can use roughly two
threads and 5–6 GB of RAM. The [release wrapper](../docker/README.md) reads
`FLASH_ATTN_MAX_JOBS` from the environment or derives a bounded default.

## FFmpeg

Images omit the FFmpeg executable and bundled codecs. Supply FFmpeg 4.x or
later at runtime with the codecs your inputs need and `libx264` for the default
outputs. Either install it inside the running consumer container:

```bash
apt-get update && apt-get install -y ffmpeg
```

or, on the host, reuse `ARTIFIXER_IMAGE` and `ARTIFIXER_DATA` from the quick
start and mount a compatible self-contained installation:

```bash
docker run --gpus all --ipc=host --rm -it \
    -v "$PWD":/workspace/artifixer \
    -v "$ARTIFIXER_DATA":/data \
    -v /path/to/ffmpeg-env:/opt/ffmpeg-env:ro \
    -e IMAGEIO_FFMPEG_EXE=/opt/ffmpeg-env/bin/ffmpeg \
    -e HF_HOME=/data/.cache/huggingface \
    -w /workspace/artifixer "$ARTIFIXER_IMAGE"
```

An `ffmpeg` executable on `PATH` is also detected. With Pyxis/Enroot, mount the
installation into the job and set `IMAGEIO_FFMPEG_EXE` there.

For even-sized `yuv420p` video, decoding uses fixed BT.601 limited-range
conversion with nearest-neighbor chroma. It ignores color tags and is unsuitable
for BT.709 or full-range input. Other formats and odd dimensions use FFmpeg's
conversion, which can differ across builds. Keep the same FFmpeg build for
metric comparisons using that path; its version is logged on first video I/O.

Before distributing a container, follow the [container release workflow](../docker/README.md).
Deleting codecs from the final filesystem leaves their bytes in ancestor layers;
release images must be flattened and checked.
