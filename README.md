<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models

[Riccardo de Lutio](https://riccardodelutio.github.io/),
[Tobias Fischer](https://tobiasfshr.github.io/),
[Yen-Yu Chang](https://yuyuchang.github.io/),
[Yuxuan Zhang](https://scholar.google.com/citations?user=Jt5VvNgAAAAJ&hl=en),
[Jay Zhangjie Wu](https://zhangjiewu.github.io/),
[Xuanchi Ren](https://xuanchiren.com/),
[Tianchang Shen](https://www.cs.toronto.edu/~shenti11/),
[Katarina Tothova](https://www.linkedin.com/in/katarina-tothova/),
[Zan Gojcic](https://zgojcic.github.io/),
[Haithem Turki](https://haithemturki.com/)

[Project](https://research.nvidia.com/labs/sil/projects/artifixer/) ·
[Paper](https://research.nvidia.com/labs/sil/projects/artifixer/assets/paper.pdf) ·
[Checkpoints](https://huggingface.co/nvidia/ArtiFixer)

![Base 3DGRUT and ArtiFixer3D+ comparison](https://github.com/user-attachments/assets/3d1af206-9818-4482-bb71-b801badffc08)

ArtiFixer is an autoregressive video diffusion model that can correct and extend
rendered views of a scene reconstructed with 3D Gaussians. This repository
contains the official implementation, scene preparation, inference, training
code, and benchmark metrics. ArtiFixer3D distills the corrected views into a
new 3D Gaussian reconstruction, and ArtiFixer3D+ applies ArtiFixer again to that
reconstruction's renders.

| Task | Guide |
| --- | --- |
| Correct a COLMAP scene | [Quick start](#quick-start) |
| Set up the environment | [Environment](docs/setup.md) |
| Run inference for ArtiFixer or ArtiFixer3D/+ | [Inference](docs/inference.md) |
| Prepare DL3DV training data | [Dataset preparation](docs/data.md) |
| Train Cosmos3-Nano or Wan through all three stages | [Training recipes](docs/training.md) |
| Run DL3DV or NerfBusters benchmarks | [Evaluation](docs/evaluation.md) |
| Validate training checkpoints | [Automated validation](model_training/VALIDATION.md) |
| Contribute or run checks | [Contributing](CONTRIBUTING.md) |
| Build and release containers | [Container release workflow](docker/README.md) |

## Quick start

The fastest way to try ArtiFixer: correct the source views of one COLMAP scene
with the smaller released model. You need an NVIDIA GPU, Docker with GPU
support, and a COLMAP reconstruction. Preparation also runs 3DGRUT, MoGe, and a
captioning model. For larger models or image sizes, see
[hardware requirements](docs/setup.md#hardware).

### 1. Set up the environment

Clone the source:

```bash
git clone --recurse-submodules https://github.com/nv-tlabs/ArtiFixer.git
cd ArtiFixer
```

Existing clones need `git submodule update --init --recursive`.

Prebuilt images are published to the GitHub Container Registry. For CUDA 12 on
x86:

```bash
export ARTIFIXER_IMAGE=ghcr.io/nv-tlabs/artifixer:cuda12-ffmpeg-free-20261008-eaf4b814
docker pull "$ARTIFIXER_IMAGE"
```

See [prebuilt images](docs/setup.md#prebuilt-images) for CUDA 13, ARM64, and
immutable digests. To build the image yourself instead:

```bash
docker build -f Dockerfile.cuda12 -t artifixer:cuda12 .
export ARTIFIXER_IMAGE=artifixer:cuda12
```

When building for GPUs other than H100, add `--build-arg BUILD_FLASH_ATTN_3=0`. The default
FA3 build uses 16 compiler jobs, each needing roughly 5–6 GB of RAM; see
[build options](docs/setup.md#build-options). ARM64 systems such as GB200 use
`Dockerfile.cuda13-aarch64` and a matching image tag.

Choose a host data directory with enough space for checkpoints, model caches,
and your scenes. This example keeps it beside the checkout:

```bash
mkdir -p ../artifixer-data
export ARTIFIXER_DATA="$(cd ../artifixer-data && pwd)"
docker run --gpus all --ipc=host --rm -it \
    -v "$PWD":/workspace/artifixer \
    -v "$ARTIFIXER_DATA":/data \
    -e HF_HOME=/data/.cache/huggingface \
    -w /workspace/artifixer "$ARTIFIXER_IMAGE"
```

The image provides dependencies; the mounted checkout supplies the application. Model
downloads persist under the mounted data directory across container sessions.

Inside the container, install FFmpeg. Run the remaining Python and Hugging Face
commands here, from `/workspace/artifixer`:

```bash
apt-get update && apt-get install -y ffmpeg
```

The distributed images omit FFmpeg. Install it in the running container, as above,
or [mount an executable from the host](docs/setup.md#ffmpeg).

### 2. Download a checkpoint

```bash
hf download nvidia/ArtiFixer artifixer-1.3b.pt \
    --local-dir /data/artifixer-checkpoints
export CHECKPOINT=/data/artifixer-checkpoints/artifixer-1.3b.pt
export MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

Use `artifixer-14b.pt` with `Wan-AI/Wan2.1-T2V-14B-Diffusers` for the larger
Wan model. A model based on Cosmos3-Nano, `artifixer-cosmos3-nano.pt`, is also
available; it needs different preparation and inference flags, so follow the
[Cosmos3-Nano recipe](docs/inference.md#cosmos3-nano). Always match `--model_id`
to the checkpoint. The weights contain the transformer only; other components
are loaded from the base model. See the
[checkpoint reference](docs/inference.md#checkpoints) for teachers and DCP files.

### 3. Prepare a scene

Use an existing COLMAP reconstruction with this layout, or follow the
[single-scene DL3DV example](docs/inference.md#download-a-dl3dv-example):

```text
my_capture/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

```bash
export COLMAP_SCENE=/data/my_capture
export SCENE_ROOT=/data/artifixer-prep/my_capture

python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir "$COLMAP_SCENE" \
    --output_root "$SCENE_ROOT" \
    --captioning_model_id Qwen/Qwen3-VL-4B-Instruct
```

Use a new output root for each scene or view selection. This command trains
3DGRUT for 10,000 iterations, renders the source cameras, estimates metric
scale, and writes captions and `split.json`. The 4B captioner uses less GPU
memory than the default 30B model.

All source images are anchors by default. To use ArtiFixer3D afterward, prepare
a [subset of anchors or a novel trajectory](docs/inference.md#choose-anchors-and-target-views)
from the start; ArtiFixer3D requires at least one generated target view.

### 4. Correct the views

```bash
export SAVE_DIR=/data/artifixer-corrected
export RENDER_NAME=my-capture-1p3b

python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint "$CHECKPOINT" \
    --model_id "$MODEL_ID" \
    --save_dir "$SAVE_DIR" \
    --render_name "$RENDER_NAME" \
    --split_path "$SCENE_ROOT/split.json" \
    --render_trajectory all_frames
```

The command prints `Writing outputs to ...`. Under that directory, each scene
is named after its `SCENE_ROOT` folder (`my_capture` here). Corrected
videos are in `my_capture/videos/` and full-clip PNGs are in
`my_capture/video_frames/batch_0000/pred/`. Held-out or novel-target PNGs also
appear in `my_capture/frames/batch_0000/pred/`; this directory is empty when
every source frame is an anchor. `--save_frame_outputs_only` omits videos and
full-clip PNGs, keeping only held-out or novel-target frames.

Choose a new `--render_name` when changing the checkpoint, inputs, seed, or
sampling settings. Existing complete frame sets are skipped without comparing
those settings. `--replace_if_exists` deletes the selected scenes' old outputs
before regenerating them.

## Training code

Training has three stages, each with its own entry point:

1. **S1:** conditioned diffusion finetuning (`model_training.train`). Released
   S1 checkpoints serve as teachers.
2. **S2:** autoregressive finetuning with diffusion forcing
   (`model_training.diffusion_forcing`).
3. **S3:** self forcing DMD distillation (`model_training.distillation`).

Follow the [training guide](docs/training.md) for complete Cosmos3-Nano commands,
the release configuration, Wan variants, and checkpoint handoffs. DMD defaults
to 1,000 iterations, saving every 100 and retaining the ten latest completed
checkpoints. Each entry point also lists its options with `--help`.

See [automated validation](model_training/VALIDATION.md) for checkpoint panels
and saved-report comparisons, and [dataset preparation](docs/data.md) for inputs.

## License and contributions

ArtiFixer is released under [Apache-2.0](LICENSE). See
[third-party notices](THIRD-PARTY-NOTICES.md) and
[contribution requirements](CONTRIBUTING.md).

### Automatically retrieved materials

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or
interacts with external materials. Those retrieved materials are not
distributed with this software and are governed solely by separate terms,
conditions and licenses. You are solely responsible for finding, reviewing and
complying with all applicable terms, conditions, and licenses, and for
verifying the security, integrity and suitability of any retrieved materials
for your specific use case. This software is provided "AS IS", without
warranty of any kind. The author makes no representations or warranties
regarding any retrieved materials, and assumes no liability for any losses,
damages, liabilities or legal consequences from your use or inability to use
this software or any retrieved materials. Use this software and the retrieved
materials at your own risk.

## Citation

```bibtex
@inproceedings{delutio2026artifixer,
    title={ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models},
    author={de Lutio, Riccardo and Fischer, Tobias and Chang, Yen-Yu and Zhang, Yuxuan and
            Wu, Jay Zhangjie and Ren, Xuanchi and Shen, Tianchang and Tothova, Katarina and
            Gojcic, Zan and Turki, Haithem},
    booktitle={SIGGRAPH},
    year={2026}
}
```
