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

[Project Page](https://research.nvidia.com/labs/sil/projects/artifixer/) / [Paper](https://research.nvidia.com/labs/sil/projects/artifixer/assets/paper.pdf)

![Base 3DGRUT vs ArtiFixer3D+ slider](https://github.com/user-attachments/assets/3d1af206-9818-4482-bb71-b801badffc08)

This repository provides the official implementation of ArtiFixer.

## License and Contributions

This project is released under the Apache License, Version 2.0. See [LICENSE](LICENSE).
Third-party notices and additional license texts are listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

This project will only accept contributions under Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms.

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

## Repository Layout

- `model_training/`: model definition, data loaders, training loop, and diffusion pipelines.
- `model_eval/`: inference entry point and metric computation for DL3DV and Nerfbusters evaluations.
- `data_processing/`: public data-preparation wrappers, split generation, captioning helpers, and sparse-reconstruction data conversion.
- `thirdparty/`: external reconstruction dependencies used by the data-preparation pipeline.

## Setup

Clone the repository with its ArtiFixer-compatible 3DGRUT submodule:

```bash
git clone --recurse-submodules https://github.com/nv-tlabs/ArtiFixer.git
cd ArtiFixer
```

If you already cloned the repository without submodules, initialize the 3DGRUT
dependency before building Docker images or running sparse reconstruction and
ArtiFixer3D:

```bash
git submodule update --init --recursive
```

The recommended environment is one of the provided CUDA Dockerfiles:

```bash
docker build -f Dockerfile.cuda12 -t artifixer:cuda12 .
docker build -f Dockerfile.cuda13 -t artifixer:cuda13 .
docker build -f Dockerfile.cuda13-aarch64 -t artifixer:cuda13-aarch64 .
```

Use `Dockerfile.cuda13-aarch64` for ARM64 systems such as GB200 nodes. Use the CUDA 12 or CUDA 13 Dockerfiles for standard x86_64 CUDA environments.

Run the image with the repository and datasets mounted:

```bash
docker run --gpus all --ipc=host --rm -it \
    -v "$PWD":/workspace/artifixer \
    -v /path/to/data:/data \
    artifixer:cuda12
cd /workspace/artifixer
```

Download the release checkpoint from the [ArtiFixer Hugging Face repo](https://huggingface.co/nvidia/ArtiFixer):

```bash
mkdir -p /data/artifixer-checkpoints
huggingface-cli download nvidia/ArtiFixer \
    artifixer-14b.pt \
    --local-dir /data/artifixer-checkpoints

export CHECKPOINT_PT=/data/artifixer-checkpoints/artifixer-14b.pt
```

## Inference

To try out the workflow on one scene, download this DL3DV archive:

```bash
export DL3DV_ROOT=/data/DL3DV-ALL-960P

python scripts/download_dl3dv_scene.py \
    --local-dir "$DL3DV_ROOT" \
    --scene-id 15ff83e2531668d27c92091c97d31401ce323e24ee7c844cb32d5109ab9335f7 \
    --subdir 8K
```

For an arbitrary image collection, first run COLMAP and organize the result as:

```text
<COLMAP_SCENE>/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

Then prepare the scene for ArtiFixer inference:

```bash
python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir /path/to/COLMAP_SCENE \
    --output_root /path/to/artifixer-prep/my_scene
```

By default, every COLMAP image is used as a 3DGRUT training view. To select a subset of images, pass a
newline-delimited file of selected training image names:

```bash
python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir /path/to/COLMAP_SCENE \
    --output_root /path/to/artifixer-prep/my_scene \
    --selected_image_names_file /path/to/selected_train_images.txt
```

Each prepared `split.json` describes one render path. To prepare a novel camera path, use a separate output
root and pass a transforms-style JSON file with camera intrinsics and 4x4 camera-to-world matrices. Frame entries
may override the top-level focal length, principal point, and distortion, but keep one fixed resolution across the
trajectory. The preparation command renders the 3DGRUT reconstruction along that path and writes a new
`split.json` that points to those renders.

```json
{
  "camera_model": "OPENCV",
  "w": 1024,
  "h": 576,
  "fl_x": 640.0,
  "fl_y": 640.0,
  "cx": 512.0,
  "cy": 288.0,
  "frames": [
    {"transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
  ]
}
```

```bash
python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir /path/to/COLMAP_SCENE \
    --output_root /path/to/artifixer-prep/my_scene_orbit_360 \
    --selected_image_names_file /path/to/selected_train_images.txt \
    --trajectory_path /path/to/orbit_360.json
```

The command trains a 3DGRUT COLMAP MCMC reconstruction for 10,000 iterations by default, renders the source
cameras or the requested trajectory, estimates metric scale with MoGe, and writes caption embeddings. It prepares
these inputs for `model_eval.run_inference`:

```text
/path/to/artifixer-prep/my_scene/
  split.json
  selected_indices.json
  selected_images.txt
  3dgrut_input/
  recon_results/
  captions/
  metric_alignment/scale_info.txt
```

Run ArtiFixer on the prepared full clip with the generated paths. Release
checkpoints are single-file transformer state dicts; DCP/FSDP checkpoint
directories are also supported by replacing `--checkpoint_pt` with
`--checkpoint_dir`.

```bash
export SCENE_ROOT=/path/to/artifixer-prep/my_scene
export SAVE_DIR=/path/to/artifixer-corrected

python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint_pt "$CHECKPOINT_PT" \
    --save_dir "$SAVE_DIR" \
    --split_path "$SCENE_ROOT/split.json" \
    --render_trajectory all_frames
```

To run on a prepared novel trajectory:

```bash
export SCENE_ROOT=/path/to/artifixer-prep/my_scene_orbit_360

python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint_pt "$CHECKPOINT_PT" \
    --save_dir "$SAVE_DIR" \
    --split_path "$SCENE_ROOT/split.json" \
    --render_trajectory trajectory
```

### ArtiFixer3D and ArtiFixer3D+

`model_eval.run_inference` can correct held-out validation frames, the full source trajectory, or the prepared trajectory described by the split:

```bash
# Default: held-out validation frames.
--render_trajectory val_frames

# Full source clip.
--render_trajectory all_frames

# Prepared novel trajectory.
--render_trajectory trajectory
```

ArtiFixer3D trains a fresh 3DGRUT optimization by default on the union of real anchor views and
ArtiFixer-generated target views. The split defines those roles: selected source images are real anchors,
and non-selected source or trajectory frames are targets whose RGB comes from the ArtiFixer prediction directory.
Use `--selected_image_names_file` to create source-camera targets, or `--trajectory_path` to create novel-trajectory targets.

After the ArtiFixer run completes, pass its predicted frames into the ArtiFixer3D stage. Use the output directory
printed by `model_eval.run_inference`:

```bash
export ARTIFIXER_OUTPUT_DIR=/path/to/artifixer-corrected/<checkpoint_name>/<run_name>
export SCENE_ID=$(basename "$SCENE_ROOT")
export ARTIFIXER_FRAMES_DIR="$ARTIFIXER_OUTPUT_DIR/$SCENE_ID/frames/batch_0000/pred"

python -m data_processing.run_artifixer3d \
    --scene_root "$SCENE_ROOT" \
    --artifixer_frames_dir "$ARTIFIXER_FRAMES_DIR"
```

The ArtiFixer3D stage renders the updated reconstruction and writes the metadata used by ArtiFixer3D+ inference:

```text
$SCENE_ROOT/artifixer3d/
  distillation_input/
  runs/
  recon_results/
$SCENE_ROOT/split_artifixer3d_plus.json
```

Run ArtiFixer3D+ by applying ArtiFixer again with that generated inference metadata.

```bash
export ARTIFIXER3D_PLUS_SAVE_DIR=/path/to/artifixer3d-plus
export RENDER_TRAJECTORY=all_frames  # use trajectory for a prepared novel-trajectory split

python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint_pt "$CHECKPOINT_PT" \
    --save_dir "$ARTIFIXER3D_PLUS_SAVE_DIR" \
    --split_path "$SCENE_ROOT/split_artifixer3d_plus.json" \
    --render_trajectory "$RENDER_TRAJECTORY"
```

## Training Data Preparation

Training expects three prepared inputs:

1. DL3DV scene archives from the [DL3DV-ALL-960P Hugging Face dataset](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P), arranged under a root such as `<DL3DV_ROOT>/<split_or_subdir>/<scene_id>.zip`.
2. Reconstruction HDF5 files referenced by the split JSON. Each reconstruction file must include selected indices, render/opacity payloads, and a valid scale.
3. Prompt HDF5 files under `<PROMPT_ROOT>/<split_or_subdir>/<scene_id>/frames_<num_frames>_stride_1*.h5`.


The workflow below runs the required data-preparation tasks directly. The captioning and reconstruction commands process every scene zip under `--dl3dv_dir` by default; use `--scene_id` or `--scene_list` only when intentionally restricting a run.

### 1. Download DL3DV

Download the DL3DV scene zips from the [DL3DV-ALL-960P Hugging Face dataset](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P):

```bash
huggingface-cli download DL3DV/DL3DV-ALL-960P \
    --repo-type dataset \
    --local-dir /path/to/DL3DV-ALL-960P
```

### 2. Generate Prompt HDF5 Files

Generate the text-conditioning HDF5 files used during training:

```bash
python -m data_processing.run_captioning \
    --dl3dv_dir /path/to/DL3DV-ALL-960P \
    --output_dir /path/to/artifixer-data/DL3DV-ALL-960P-captions
```

### 3. Generate Reconstruction HDF5 Files

Generate sparse 3D reconstructions and convert their renders, opacity, depth, selected indices, and metric scale into ArtiFixer HDF5 files:

```bash
python -m data_processing.run_sparse_reconstruction \
    --dl3dv_dir /path/to/DL3DV-ALL-960P \
    --output_root /path/to/artifixer-data/reconstructions \
    --work_root /path/to/artifixer-work/reconstructions \
    --num_selected_indices 2 3 6 12
```

This wrapper runs the required per-scene operations in order:

1. Half-covisibility camera split generation.
2. 3DGRUT sparse reconstruction training for each requested scene half and view count.
3. Metric-scale alignment.
4. Conversion to HDF5.
5. Copying final `data_*.h5`, `parsed_*.yaml`, and `ckpt_last_*.pt` files into the reconstruction root.

The split builder expects reconstruction subdirectories named `dl3dv_<dl3dv_subdir>`, which the wrapper creates by default. Final files are written as:

```text
<RECON_ROOT>/dl3dv_<dl3dv_subdir>/<scene_id>/
  data_<scene_id>_<scene_half>_<num_views>.h5
  parsed_<scene_id>_<scene_half>_<num_views>.yaml
  ckpt_last_<scene_id>_<scene_half>_<num_views>.pt
```

Metric alignment uses MoGe for monocular depth. If the run environment cannot download MoGe weights, download the checkpoint ahead of time and set `MOGE_MODEL_PATH` to that local checkpoint directory before launching reconstruction.

### 4. Build the Train/Test Split

Generate the split JSON after prompt and reconstruction files are available:

```bash
python -m data_processing.trainval_test_split \
    --data_path /path/to/artifixer-data/reconstructions \
    --dl3dv_dir /path/to/DL3DV-ALL-960P \
    --output_root /path/to/artifixer-data
```

This writes `/path/to/artifixer-data/trainval_test_split.json`. The script validates source archives, required reconstruction splits, duplicate reconstruction files, and known bad scenes before writing the split.

### 5. Use Prepared Paths

Use the prepared paths consistently for training and evaluation:

```bash
export SPLIT_PATH=/path/to/artifixer-data/trainval_test_split.json
export DL3DV_ROOT=/path/to/DL3DV-ALL-960P
export PROMPT_ROOT=/path/to/artifixer-data/DL3DV-ALL-960P-captions
```

## Training

ArtiFixer training has three stages:

1. Stage 1 supervised finetuning on reconstruction-conditioned DL3DV clips.
2. Stage 2 diffusion-forcing finetuning from the stage 1 checkpoint.
3. Stage 3 DMD distillation, using the stage 2 checkpoint as the student/generator initialization and the stage 1
   checkpoint as the critic initialization.

The default model is Wan2.1 14B. Set `num_processes * gradient_accumulation_steps` to 128 for the default recipe; for example, use `--gradient_accumulation_steps 16` with 8 processes.

Launch stage 1 with `accelerate`:

```bash
export PROJECT_DIR=/path/to/runs/artifixer-s1-14b
export SPLIT_PATH=/path/to/artifixer-data/trainval_test_split.json
export DL3DV_ROOT=/path/to/DL3DV-ALL-960P
export PROMPT_ROOT=/path/to/artifixer-data/DL3DV-ALL-960P-captions
export NUM_PROCESSES=8
export GRADIENT_ACCUMULATION_STEPS=16

accelerate launch \
    --multi_gpu \
    --num_processes "$NUM_PROCESSES" \
    --module model_training.train \
    --project_dir "$PROJECT_DIR" \
    --split_path "$SPLIT_PATH" \
    --dl3dv_dir "$DL3DV_ROOT" \
    --prompt_dir "$PROMPT_ROOT" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --tracker_run_name artifixer-s1-14b \
    --resume_from_checkpoint auto
```

For multi-node Slurm jobs, start from `model_training/slurm/sample-slurm-submit.sh`; it is a template that expects you to provide your cluster account, partition, paths, and optional container image through standard Slurm flags or environment variables.

Stage 2 finetunes a stage 1 checkpoint with block-causal diffusion-forcing training:

```bash
export STAGE1_CHECKPOINT=/path/to/runs/artifixer-s1-14b/checkpoints/checkpoint_25000/pytorch_model_fsdp_0
export STAGE2_PROJECT_DIR=/path/to/runs/artifixer-s2-14b-from-s1-25000

accelerate launch \
    --multi_gpu \
    --num_processes "$NUM_PROCESSES" \
    --module model_training.diffusion_forcing \
    --project_dir "$STAGE2_PROJECT_DIR" \
    --base_checkpoint_dir "$STAGE1_CHECKPOINT" \
    --split_path "$SPLIT_PATH" \
    --dl3dv_dir "$DL3DV_ROOT" \
    --prompt_dir "$PROMPT_ROOT" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --tracker_run_name artifixer-s2-14b-from-s1-25000 \
    --resume_from_checkpoint auto
```

Stage 3 runs DMD distillation. `--base_checkpoint_dir` initializes the student/generator; `--base_checkpoint_dir_critic` initializes the fixed real-score critic and trainable fake-score critic. The critic model config defaults to `--model_id`; pass `--model_id_critic` only when the critic checkpoint uses a different base model.

```bash
export STAGE2_CHECKPOINT=/path/to/runs/artifixer-s2-14b-from-s1-25000/checkpoints/checkpoint_10000/pytorch_model_fsdp_0
export CRITIC_CHECKPOINT=/path/to/runs/artifixer-s1-14b/checkpoints/checkpoint_25000/pytorch_model_fsdp_0
export STAGE3_PROJECT_DIR=/path/to/runs/artifixer-s3-14b-s2-10000-s1-25000

accelerate launch \
    --multi_gpu \
    --num_processes "$NUM_PROCESSES" \
    --module model_training.distillation \
    --project_dir "$STAGE3_PROJECT_DIR" \
    --base_checkpoint_dir "$STAGE2_CHECKPOINT" \
    --base_checkpoint_dir_critic "$CRITIC_CHECKPOINT" \
    --split_path "$SPLIT_PATH" \
    --dl3dv_dir "$DL3DV_ROOT" \
    --prompt_dir "$PROMPT_ROOT" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --tracker_run_name artifixer-s3-14b-s2-10000-s1-25000 \
    --resume_from_checkpoint auto
```

## Evaluation

Release evaluation reports four rows:

1. `3DGUT`: the base sparse 3D reconstruction renders.
2. `ArtiFixer`: direct frame output from `model_eval.run_inference`.
3. `ArtiFixer3D`: a fresh 3DGRUT optimization distilled from the direct ArtiFixer frames.
4. `ArtiFixer3D+`: ArtiFixer run again on the ArtiFixer3D renders and generated inference metadata.

DL3DV evaluation uses the same prepared DL3DV dataset flow as training. Use the split JSON to select the evaluation scenes and keep `--dl3dv_dir` pointed at the DL3DV-ALL-960P root used to prepare captions and reconstructions.

NerfBusters uses each scene's `transforms.json` plus the scene-specific image folder selected by the shared resolution helper. `aloe`, `car`, `garbage`, and `table` use `images_2`; the remaining scenes use `images`. NerfBusters visibility masks are eval-only and are not passed to 3DGRUT distillation training.

Run direct ArtiFixer inference from a checkpoint. These commands write PNG frames needed by the metric scripts. The examples use one process and therefore one GPU; to distribute scenes across multiple GPUs, run the same module through `torchrun --nproc_per_node <num-gpus>`.

DL3DV (our split):

```bash
export CHECKPOINT_PT=/data/artifixer-checkpoints/artifixer-14b.pt
export SAVE_DIR=/path/to/artifixer-eval
export SPLIT_PATH=/path/to/artifixer-data/trainval_test_split.json
export DL3DV_ROOT=/path/to/DL3DV-ALL-960P
export PROMPT_ROOT=/path/to/artifixer-data/DL3DV-ALL-960P-captions

python -m model_eval.run_inference \
    --evalset 3dgrut_dl3dv_ours \
    --checkpoint_pt "$CHECKPOINT_PT" \
    --save_dir "$SAVE_DIR" \
    --split_path "$SPLIT_PATH" \
    --dl3dv_dir "$DL3DV_ROOT" \
    --prompt_dir "$PROMPT_ROOT" \
    --save_frame_outputs_only
```

```bash
export EVAL_OUTPUT_NAME=artifixer-14b

python -m model_eval.compute_metrics_dl3dv \
    --evalset 3dgrut_dl3dv_ours \
    --eval_output_name "$EVAL_OUTPUT_NAME" \
    --sink_size 7 \
    --split_path "$SPLIT_PATH" \
    --dl3dv_dir "$DL3DV_ROOT" \
    --eval_base_path "$SAVE_DIR" \
    --no_masks
```

DL3DV (DiFix split):

```bash
export DIFIX_RECON_RESULTS_DIR=/path/to/difix-reconstruction-results
export DIFIX_TRAIN_IDS_DIR=/path/to/difix-train-ids
export DIFIX_VISIBILITY_MASKS_DIR=/path/to/difix-visibility-masks

python -m model_eval.run_inference \
    --evalset 3dgrut_dl3dv_difix \
    --checkpoint_pt "$CHECKPOINT_PT" \
    --save_dir "$SAVE_DIR" \
    --split_path "$SPLIT_PATH" \
    --dl3dv_dir "$DL3DV_ROOT" \
    --prompt_dir "$PROMPT_ROOT" \
    --recon_results_dir "$DIFIX_RECON_RESULTS_DIR" \
    --save_frame_outputs_only
```

```bash
python -m model_eval.compute_metrics_dl3dv \
    --evalset 3dgrut_dl3dv_difix \
    --eval_output_name "$EVAL_OUTPUT_NAME" \
    --sink_size 7 \
    --split_path "$SPLIT_PATH" \
    --dl3dv_dir "$DL3DV_ROOT" \
    --eval_base_path "$SAVE_DIR" \
    --difix_train_ids_dir "$DIFIX_TRAIN_IDS_DIR" \
    --visibility_masks_dir "$DIFIX_VISIBILITY_MASKS_DIR"
```

The DiFix comparison uses the masked metric YAMLs as the paper-style numbers. The mask convention is to black out pixels outside the visibility mask and then compute full-image metrics.

NerfBusters:

```bash
export NERFBUSTERS_DIR=/path/to/nerfbusters
export NERFBUSTERS_RECON_RESULTS_DIR=/path/to/nerfbusters-reconstruction-results
export NERFBUSTERS_CAPTIONS_DIR=/path/to/nerfbusters-captions
export NERFBUSTERS_VISIBILITY_MASKS_DIR=/path/to/nerfbusters-visibility-masks

python -m model_eval.run_inference \
    --evalset nerfbusters \
    --checkpoint_pt "$CHECKPOINT_PT" \
    --save_dir "$SAVE_DIR" \
    --nerfbusters_dir "$NERFBUSTERS_DIR" \
    --nerfbusters_recon_results_dir "$NERFBUSTERS_RECON_RESULTS_DIR" \
    --nerfbusters_captions_dir "$NERFBUSTERS_CAPTIONS_DIR" \
    --save_frame_outputs_only
```

```bash
python -m model_eval.compute_metrics_nerfbusters \
    --eval_output_name "$EVAL_OUTPUT_NAME" \
    --eval_base_path "$SAVE_DIR" \
    --nerfbusters_dir "$NERFBUSTERS_DIR" \
    --visibility_masks_dir "$NERFBUSTERS_VISIBILITY_MASKS_DIR"
```

For ArtiFixer3D, direct ArtiFixer output frames are passed to 3DGRUT through
`image_path_override`. Source/selected frames remain the original GT anchors.
NerfBusters visibility masks are applied only by the metric script.

Use `--replace_if_exists` to regenerate existing outputs. Use `--scene_id <id>`
for a single DL3DV or NerfBusters scene. By default, the DL3DV metric scripts
evaluate every scene in the configured evaluation split.
