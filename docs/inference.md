<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference reference

Run the commands inside the [project environment](../README.md#1-set-up-the-environment).
Set `COLMAP_SCENE` to your calibrated scene using the
[input-layout example](../README.md#3-prepare-a-scene), then follow either the
[Wan 2.1](#wan-21) or [Cosmos3-Nano](#cosmos3-nano) recipe. Each recipe prepares
the inputs shared by its distilled models and S1 teachers.

## Checkpoints

Download released weights from [nvidia/ArtiFixer](https://huggingface.co/nvidia/ArtiFixer).
Always pass the matching `--model_id`:

| Checkpoint | Base model | Use |
| --- | --- | --- |
| `artifixer-14b.pt` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Distilled autoregressive model; see [Wan 2.1](#wan-21) |
| `artifixer-1.3b.pt` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Smaller distilled model; see [Wan 2.1](#wan-21) |
| `artifixer-cosmos3-nano.pt` | `nvidia/Cosmos3-Nano` | Distilled autoregressive model; see [Cosmos3-Nano](#cosmos3-nano) |
| `artifixer-s1-14b.pt` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Stage-1 bidirectional teacher; see [Wan S1](#wan-stage-1-teachers) |
| `artifixer-s1-1.3b.pt` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Smaller stage-1 bidirectional teacher; see [Wan S1](#wan-stage-1-teachers) |
| `artifixer-s1-cosmos3-nano.pt` | `nvidia/Cosmos3-Nano` | Stage-1 bidirectional teacher; see [Cosmos3-Nano S1](#cosmos3-nano-stage-1-teacher) |

Teachers use `--inference_pipeline bidirectional --num_inference_steps 50`.
The default pipeline, `kv_cache`, uses four steps for the distilled models.
Bidirectional chunks default to 81 raw frames; adjust `--bidirectional_chunk_size`
if needed. Its text-guidance default is 5; `--text_guidance_scale 1` disables
classifier-free text guidance. That flag is rejected by `kv_cache`. Use a new
`--render_name` for each sampling configuration.

Both Wan and Cosmos3 accept a `.pt` export or the exact DCP/FSDP model-slot
directory through `--checkpoint`. Keep pose metadata with the weights:
`NAME.metadata.json` beside `NAME.pt`, or `artifixer.metadata.json` inside the
DCP slot. Inference uses the recorded `camera_pose_alignment`; an override
must agree. Wan2.1 checkpoints without a record use `mean`. Other models
require the record. Both model families use the first target camera as the
fixed origin throughout the rollout.
An interrupted `.pt` export leaves a `NAME.pt.incomplete` marker; inference
refuses it until the export is rerun successfully with `--overwrite`.

The released KV-cache geometry is `--frames_per_block 7 --local_attn_size 21
--sink_size 7`, in latent frames. Match geometry to the checkpoint's training
settings. `--seed` defaults to 42 and is applied independently to each item.
`--render_diagnostics` adds three conditioning-ablation videos per chunk.
Retries regenerate missing or incomplete diagnostics. `--save_frame_outputs_only`
skips diagnostics and video encoding.
Interrupted PNG writes are regenerated, and incomplete videos are rebuilt from
retained frames. New image and video files replace their destinations only after
writing completes.

## Download a DL3DV example

Accept the terms for both the [scene archives](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P)
and [COLMAP cache](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-ColmapCache),
then run `hf auth login` first. This downloads one scene and its matching model:

```bash
export DL3DV_ROOT=/data/DL3DV-ALL-960P
export DL3DV_COLMAP_ROOT=/data/DL3DV-ALL-ColmapCache
export DL3DV_SCENE=15ff83e2531668d27c92091c97d31401ce323e24ee7c844cb32d5109ab9335f7

hf download DL3DV/DL3DV-ALL-960P "8K/$DL3DV_SCENE.zip" \
    --repo-type dataset --local-dir "$DL3DV_ROOT"
hf download DL3DV/DL3DV-ALL-ColmapCache "8K/$DL3DV_SCENE.zip" \
    --repo-type dataset --local-dir "$DL3DV_COLMAP_ROOT"

mkdir -p /data/colmap-scenes
unzip -qo "$DL3DV_ROOT/8K/$DL3DV_SCENE.zip" -d /data/colmap-scenes
unzip -qo "$DL3DV_COLMAP_ROOT/8K/$DL3DV_SCENE.zip" \
    "$DL3DV_SCENE/colmap/sparse/0/*.bin" -d /data/colmap-scenes
export COLMAP_SCENE=/data/colmap-scenes/$DL3DV_SCENE
```

Scene preparation accepts `sparse/0/` or `colmap/sparse/0/` and an `images/`
directory or a single `images_<factor>/` directory. It rescales intrinsics
to the chosen images. For downsampled frames stored in plain `images/`, pass
`--pyramid_factor` if independent dimension rounding needs the known integer
factor. With no factor, both dimensions must share the same scale. A known
factor permits at most one output pixel of rounding per axis.

## Choose anchors and target views

The selected source images are real anchors. For a source-camera split, all
other frames are targets. The default selects every source image and therefore
leaves no targets for ArtiFixer3D.

To reserve target views, write a nonempty proper subset of image filenames,
one per line, and prepare a new output root:

```bash
export SCENE_ROOT=/data/artifixer-prep/my_capture_subset
python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir "$COLMAP_SCENE" \
    --output_root "$SCENE_ROOT" \
    --selected_image_names_file /data/selected_train_images.txt \
    --captioning_model_id Qwen/Qwen3-VL-4B-Instruct
```

For a novel camera path, also pass `--trajectory_path /data/trajectory.json`.
Use transforms-style JSON with camera intrinsics and OpenGL camera-to-world
matrices in the COLMAP model's world coordinates. An `applied_transform`, if
present, is removed first. Per-frame intrinsics and distortion may vary, but resolution must
remain fixed. For example, one camera frame is represented as:

```json
{
  "camera_model": "OPENCV",
  "w": 1024, "h": 576,
  "fl_x": 640.0, "fl_y": 640.0, "cx": 512.0, "cy": 288.0,
  "frames": [
    {"transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
  ]
}
```

Use a new output root for a new scene or selection. Phase-specific reruns keep
the saved anchors and frame order. A different image set, camera geometry, or
explicit selection is rejected. Repeat the original prompt-format and FPS options,
which otherwise use their defaults. `--replace` overwrites the selected phases.
Replacing prepared inputs requires all phases and clears the old reconstruction,
renders, scale, captions, and split before rebuilding.

`--reconstruction_checkpoint` rerenders the supplied checkpoint and records its
actual render directory. A later scale or caption retry reuses those renders
to finish the split.

Inference selects the requested frames with `--render_trajectory`:

| Value | Frames |
| --- | --- |
| `val_frames` (default) | Held-out source frames |
| `all_frames` | Full source clip |
| `trajectory` | Prepared novel camera path |

For the novel path, use the [quick-start inference command](../README.md#4-correct-the-views)
with `--render_trajectory trajectory` and the new root's `split.json`.
When preparing inputs with a model-family recipe below, include the same
`--selected_image_names_file` or `--trajectory_path` options chosen here.

## Wan 2.1

Wan 2.1 provides 14B and 1.3B distilled models and their S1 bidirectional
teachers, listed in the [checkpoint table](#checkpoints). All four use Wan T5
captions and `mean` camera-pose alignment. The 1.3B models use less GPU memory;
see [hardware requirements](setup.md#hardware).

Prepare a fresh scene root with the source sequence's actual frame rate (60
below for DL3DV). If you already followed the [quick start](../README.md#quick-start),
retain its `SCENE_ROOT` and `SAVE_DIR` and skip this preparation block.

```bash
export SCENE_ROOT=/data/artifixer-prep/my_capture_wan
export DATASET_FPS=60
export SAVE_DIR=/data/artifixer-corrected
python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir "$COLMAP_SCENE" --output_root "$SCENE_ROOT" \
    --captioning_model_id Qwen/Qwen3-VL-4B-Instruct \
    --prompt_format wan_t5_embedding --dataset_fps "$DATASET_FPS"
```

### Wan distilled models

The example uses the 1.3B distilled model. For 14B, select its checkpoint and
matching base model from the table above when setting the first two variables:

```bash
export WAN_CHECKPOINT_FILE=artifixer-1.3b.pt
export MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
hf download nvidia/ArtiFixer "$WAN_CHECKPOINT_FILE" \
    --local-dir /data/artifixer-checkpoints
export CHECKPOINT=/data/artifixer-checkpoints/$WAN_CHECKPOINT_FILE
export RENDER_NAME="my-capture-${WAN_CHECKPOINT_FILE%.pt}"

python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint "$CHECKPOINT" --model_id "$MODEL_ID" \
    --save_dir "$SAVE_DIR" --render_name "$RENDER_NAME" \
    --split_path "$SCENE_ROOT/split.json" --render_trajectory all_frames \
    --inference_pipeline kv_cache --num_inference_steps 4 \
    --frames_per_block 7 --local_attn_size 21 --sink_size 7 \
    --camera_pose_alignment mean --seed 42
```

### Wan stage-1 teachers

Reuse the Wan preparation, `SCENE_ROOT`, and `SAVE_DIR` above. The example uses
the 1.3B teacher; for 14B, select its S1 checkpoint and matching base model from
the [checkpoint table](#checkpoints) when setting the first two variables:

```bash
export WAN_S1_FILE=artifixer-s1-1.3b.pt
export WAN_S1_MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
hf download nvidia/ArtiFixer "$WAN_S1_FILE" \
    --local-dir /data/artifixer-checkpoints

python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint "/data/artifixer-checkpoints/$WAN_S1_FILE" \
    --model_id "$WAN_S1_MODEL_ID" \
    --save_dir "$SAVE_DIR" --render_name "my-capture-${WAN_S1_FILE%.pt}" \
    --split_path "$SCENE_ROOT/split.json" --render_trajectory all_frames \
    --inference_pipeline bidirectional --num_inference_steps 50 \
    --text_guidance_scale 5 --bidirectional_chunk_size 81 \
    --camera_pose_alignment mean --seed 42
```

## Cosmos3-Nano

The Cosmos3-Nano S3 distilled model is `artifixer-cosmos3-nano.pt`; its S1
bidirectional teacher is `artifixer-s1-cosmos3-nano.pt`. Download each checkpoint
with its matching `.metadata.json` sidecar. A Wan checkpoint
cannot be used by changing only `MODEL_ID`. See [Cosmos memory requirements](setup.md#hardware).

Prepare a fresh scene root with Cosmos token-ID prompts and the source sequence's
actual frame rate (60 below for DL3DV). Both S3 and S1 use these prepared inputs,
the same base revision, and `last` camera-pose alignment:

```bash
export MODEL_ID=nvidia/Cosmos3-Nano
export MODEL_REVISION=23314034b82f46b45339035dba67c3ee9bbcb8ba
export SCENE_ROOT=/data/artifixer-prep/my_capture_cosmos3
export DATASET_FPS=60
export SAVE_DIR=/data/artifixer-corrected
python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir "$COLMAP_SCENE" --output_root "$SCENE_ROOT" \
    --captioning_model_id Qwen/Qwen3-VL-4B-Instruct \
    --prompt_format cosmos3_omni_token_ids --dataset_fps "$DATASET_FPS"
```

Preparation writes `caption.cosmos3.h5` and records it in `split.json`, including
the ArtiFixer3D+ handoff. The base configuration, VAE, and tokenizer use the
pinned revision above. Camera rays use the first target camera as their fixed
origin; the model profile sets flow shift 5 and RoPE FPS 24 independently of
the source FPS.

### Cosmos3-Nano distilled model

Download the S3 weights and metadata, then run four-step KV-cache inference with
7/21/7 geometry. This pipeline does not use text guidance:

```bash
hf download nvidia/ArtiFixer \
    artifixer-cosmos3-nano.pt artifixer-cosmos3-nano.metadata.json \
    --local-dir /data/artifixer-checkpoints
export CHECKPOINT=/data/artifixer-checkpoints/artifixer-cosmos3-nano.pt
export RENDER_NAME=my-capture-cosmos3
test -f "$CHECKPOINT" && test -f "${CHECKPOINT%.pt}.metadata.json"

python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint "$CHECKPOINT" --model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" \
    --save_dir "$SAVE_DIR" --render_name "$RENDER_NAME" \
    --split_path "$SCENE_ROOT/split.json" --render_trajectory all_frames \
    --inference_pipeline kv_cache --num_inference_steps 4 \
    --frames_per_block 7 --local_attn_size 21 --sink_size 7 \
    --camera_pose_alignment last --seed 42
```

For DL3DV benchmarks, prepare [full-scene Cosmos evaluation captions](data.md#2-generate-captions)
and use this checkpoint, model, and prompt root in the
[evaluation commands](evaluation.md#shared-settings).

### Cosmos3-Nano stage-1 teacher

Download the S1 weights and metadata, then reuse the Cosmos preparation,
`MODEL_ID`, `MODEL_REVISION`, `SCENE_ROOT`, and `SAVE_DIR` above. Use a separate
render name and the bidirectional pipeline:

```bash
hf download nvidia/ArtiFixer \
    artifixer-s1-cosmos3-nano.pt artifixer-s1-cosmos3-nano.metadata.json \
    --local-dir /data/artifixer-checkpoints
export S1_CHECKPOINT=/data/artifixer-checkpoints/artifixer-s1-cosmos3-nano.pt
test -f "$S1_CHECKPOINT" && test -f "${S1_CHECKPOINT%.pt}.metadata.json"

python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint "$S1_CHECKPOINT" --model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" \
    --save_dir "$SAVE_DIR" --render_name my-capture-cosmos3-s1 \
    --split_path "$SCENE_ROOT/split.json" --render_trajectory all_frames \
    --inference_pipeline bidirectional --num_inference_steps 50 \
    --text_guidance_scale 5 --bidirectional_chunk_size 81 \
    --camera_pose_alignment last --seed 42
```

## Refresh captions

To change only prompt format or FPS in an existing prepared scene, run only
the caption phase. Set `COLMAP_SCENE` and `SCENE_ROOT` to that scene, choose
`wan_t5_embedding` for Wan or `cosmos3_omni_token_ids` for Cosmos, and use the
source sequence's actual frame rate:

```bash
export PROMPT_FORMAT=wan_t5_embedding  # cosmos3_omni_token_ids for Cosmos
export DATASET_FPS=60
python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir "$COLMAP_SCENE" --output_root "$SCENE_ROOT" \
    --phases caption --replace \
    --captioning_model_id Qwen/Qwen3-VL-4B-Instruct \
    --prompt_format "$PROMPT_FORMAT" --dataset_fps "$DATASET_FPS"
```

This retains the reconstruction, renders, anchors, and target trajectory.
Omit a previous `--trajectory_path` on this caption-only rerun; the saved split
retains it. Using `--replace` without `--phases caption` reruns all phases and
rebuilds the reconstruction. Use a new inference `--render_name` after changing captions.

## ArtiFixer3D and ArtiFixer3D+

Prepare anchors and targets as described above, then run ArtiFixer for every
target frame. ArtiFixer3D trains a fresh 3DGRUT reconstruction using original
anchor images and ArtiFixer-generated targets.

Set `ARTIFIXER_OUTPUT_DIR` to the directory printed by inference:

```bash
export ARTIFIXER_OUTPUT_DIR="/path/printed/by/inference"
export SCENE_ID="$(basename "$SCENE_ROOT")"
export ARTIFIXER_FRAMES_DIR="$ARTIFIXER_OUTPUT_DIR/$SCENE_ID/frames/batch_0000/pred"

python -m data_processing.run_artifixer3d \
    --scene_root "$SCENE_ROOT" \
    --artifixer_frames_dir "$ARTIFIXER_FRAMES_DIR"
```

This writes the new reconstruction under `$SCENE_ROOT/artifixer3d/` and an
inference split at `$SCENE_ROOT/split_artifixer3d_plus.json`. Distillation with
`--replace` clears the old checkpoint, renders, and split before rebuilding.
Use a new output root to retain them. ArtiFixer3D+ applies ArtiFixer to those
new renders. The example below reuses `CHECKPOINT`, `MODEL_ID`, and
`RENDER_NAME` from the selected family's distilled-model recipe.

```bash
export RENDER_TRAJECTORY=all_frames  # trajectory for a novel path
python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint "$CHECKPOINT" \
    --model_id "$MODEL_ID" \
    --save_dir /data/artifixer3d-plus \
    --render_name "$RENDER_NAME" \
    --split_path "$SCENE_ROOT/split_artifixer3d_plus.json" \
    --render_trajectory "$RENDER_TRAJECTORY"
```
