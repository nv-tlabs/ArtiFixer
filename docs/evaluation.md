<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Benchmark evaluation

Benchmark evaluation uses PNG predictions from `model_eval.run_inference`.
See [dataset preparation](data.md) for DL3DV archives, reconstruction HDF5s,
and full-scene captions. Match data, camera splits, metric protocols, and
reference-view settings across methods.

The commands below score base renders and direct ArtiFixer predictions. To score
distilled ArtiFixer3D/+ benchmark reconstructions, use
`--evalset artifixer3d_dl3dv_ours` or `artifixer3d_dl3dv_difix`. The
[ArtiFixer3D guide](inference.md#artifixer3d-and-artifixer3d) covers single prepared scenes.

## Shared settings

These defaults select Wan. For [Cosmos3](inference.md#cosmos3-nano),
keep that recipe's `CHECKPOINT` and `MODEL_ID`, choose a new `RENDER_NAME`, and
set `PROMPT_ROOT` to your full-scene Cosmos evaluation captions instead.

```bash
export CHECKPOINT=/data/artifixer-checkpoints/artifixer-14b.pt
export MODEL_ID=Wan-AI/Wan2.1-T2V-14B-Diffusers
export SAVE_DIR=/data/artifixer-eval
export RENDER_NAME=artifixer-14b_chunk7_sink7
export SPLIT_PATH=/data/trainval_test_split.json
export DL3DV_ROOT=/data/DL3DV-ALL-960P
export PROMPT_ROOT=/data/captions
```

The base revision is pinned in [pins.py](../model_training/pins.py).
Match checkpoint and model as listed in the [checkpoint reference](inference.md#checkpoints).
The default KV-cache geometry `7/21/7` matches released weights. Use the same
`--render_name`, view-count policy, neighbor-selection mode, trajectory, and
output suffix in inference and scoring. Choose a new render name for changed
inputs or settings, or use inference's `--replace_if_exists` to regenerate.

Examples use one GPU. To distribute independent scenes, run inference through
`torchrun --nproc_per_node N --module model_eval.run_inference` with the same
arguments. `--scene_id` limits inference, but the metric commands require the
complete configured benchmark. Missing predictions, renders, ground truth,
or required masks are errors. `--no_masks` omits masked metrics and their models.

## DL3DV: our split

```bash
python -m model_eval.run_inference \
    --evalset 3dgrut_dl3dv_ours \
    --checkpoint "$CHECKPOINT" --model_id "$MODEL_ID" \
    --save_dir "$SAVE_DIR" --render_name "$RENDER_NAME" \
    --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" \
    --prompt_dir "$PROMPT_ROOT" --num_views 6 --save_frame_outputs_only

python -m model_eval.compute_metrics_dl3dv \
    --evalset 3dgrut_dl3dv_ours --num_views 6 \
    --render_name "$RENDER_NAME" --eval_base_path "$SAVE_DIR" \
    --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" --no_masks
```

## DL3DV: DiFix split

Download the published training-view IDs and visibility masks:

```bash
hf download nvidia/ArtiFixer --include "eval-assets/dl3dv_difix/*" \
    --local-dir /data/artifixer-eval-assets
export DIFIX_TRAIN_IDS_DIR=/data/artifixer-eval-assets/eval-assets/dl3dv_difix/train_ids
export DIFIX_VISIBILITY_MASKS_DIR=/data/artifixer-eval-assets/eval-assets/dl3dv_difix/visibility_masks
export DIFIX_RECON_RESULTS_DIR=/data/difix-reconstruction-results
```

For each scene in
[`dl3dv_benchmark_difix_test_scenes.txt`](../data_processing/dl3dv_benchmark_difix_test_scenes.txt),
stage the images and COLMAP model, then prepare the compact frame table used by
DiFix's training IDs and masks. This removes the extra source frames listed in
[`dl3dv_difix_frame_remap.json`](../data_processing/dl3dv_difix_frame_remap.json).

```bash
export SCENE_ID=your-scene-id
export DIFIX_STAGED_ROOT=/data/difix-staged
export DIFIX_RUNTIME_ROOT=/data/difix-runtime
export DIFIX_TRAIN_ROOT=/data/difix-training
python -m model_eval.prepare_dl3dv_distill_dataset \
    --dl3dv_dir "$DL3DV_ROOT" --split_path "$SPLIT_PATH" \
    --scene_list_file data_processing/dl3dv_benchmark_difix_test_scenes.txt \
    --colmap_zip_root /data/DL3DV-ALL-ColmapCache --output_root "$DIFIX_STAGED_ROOT"
python -m model_eval.dl3dv_distill_validation "$DIFIX_STAGED_ROOT/$SCENE_ID" \
    --prepare_only --runtime_scene_dataset_path "$DIFIX_RUNTIME_ROOT/$SCENE_ID" \
    --selected_indices_file "$DIFIX_TRAIN_IDS_DIR/$SCENE_ID/difix_train_ids.json"
python thirdparty/3DGRUT-ArtiFixer/train.py --config-name apps/colmap_3dgut_sparse_mcmc \
    path="$DIFIX_RUNTIME_ROOT/$SCENE_ID" out_dir="$DIFIX_TRAIN_ROOT" \
    selected_indices_file="$DIFIX_RUNTIME_ROOT/$SCENE_ID/selected_indices.json" \
    dataset.downsample_factor=4 experiment_name=difix_split_reconstruction \
    n_iterations=30000 'checkpoint.iterations=[30000]' test_last=False use_wandb=False
python -m data_processing.render_3dgrut_colmap \
    --checkpoint "$DIFIX_TRAIN_ROOT/difix_split_reconstruction/$SCENE_ID/ours_30000/ckpt_30000.pt" \
    --colmap_dir "$DIFIX_RUNTIME_ROOT/$SCENE_ID" --downsample_factor 4 \
    --selected_indices "$DIFIX_RUNTIME_ROOT/$SCENE_ID/selected_indices.json" \
    --trajectory_path "$DIFIX_RUNTIME_ROOT/$SCENE_ID/transforms.json" --trajectory_output_subdir '' \
    --experiment_name difix_split_reconstruction \
    --output_root "$DIFIX_RECON_RESULTS_DIR/difix_split_reconstruction_eval/$SCENE_ID"
```

The scene directory must be named `SCENE_ID`; output naming uses that basename
and the checkpoint's stored step. The required layout is:

```text
<DIFIX_RECON_RESULTS_DIR>/difix_split_reconstruction_eval/<scene_id>/
  difix_split_reconstruction/<scene_id>/ours_30000/
    renders/
    opacity/
    selected_indices.json
```

Frames use contiguous compact indices from zero. Inference and scoring default to up to
12 references and label that policy `auto12`. The metric example sets
`--num_views auto` explicitly; when overriding the view count, match both commands.

Scoring needs `--recon_results_dir` for every directory-based evalset, including
when the conditioning-view count is explicit. The reconstruction's anchors define
held-out frames.

```bash
python -m model_eval.run_inference \
    --evalset 3dgrut_dl3dv_difix \
    --checkpoint "$CHECKPOINT" --model_id "$MODEL_ID" \
    --save_dir "$SAVE_DIR" --render_name "$RENDER_NAME" \
    --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" \
    --prompt_dir "$PROMPT_ROOT" --recon_results_dir "$DIFIX_RECON_RESULTS_DIR" \
    --save_frame_outputs_only

python -m model_eval.compute_metrics_dl3dv \
    --evalset 3dgrut_dl3dv_difix --num_views auto \
    --render_name "$RENDER_NAME" --eval_base_path "$SAVE_DIR" \
    --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" \
    --recon_results_dir "$DIFIX_RECON_RESULTS_DIR" \
    --difix_train_ids_dir "$DIFIX_TRAIN_IDS_DIR" \
    --visibility_masks_dir "$DIFIX_VISIBILITY_MASKS_DIR"
```

DiFix's paper-style results use the masked YAMLs: pixels outside visibility
masks are blacked out before full-image scoring.

## NerfBusters

Prepare NerfBusters reconstructions, captions, and visibility masks before running
these commands.
`aloe`, `car`, `garbage`, and `table` use `images_2`; other scenes use `images`.
Reconstruction output must follow:

```text
<reconstruction_root>/<scene>/<experiment>/<scene>/ours_30000/
  renders/
  opacity/
```

Captions are one prompt HDF5 per scene at `<NERFBUSTERS_CAPTIONS_DIR>/<scene>.h5`
(Cosmos3 uses the format-tagged filename). Masks are
`<NERFBUSTERS_VISIBILITY_MASKS_DIR>/<scene>/<index>.png`, with five-digit indices
in sorted test-filename order.

Set the experiment name to the directory created by your reconstruction run:

```bash
export NERFBUSTERS_DIR=/data/nerfbusters
export NERFBUSTERS_RECON_RESULTS_DIR=/data/nerfbusters-reconstruction-results
export NERFBUSTERS_RECON_EXPERIMENT=your-reconstruction-experiment
export NERFBUSTERS_CAPTIONS_DIR=/data/nerfbusters-captions
export NERFBUSTERS_VISIBILITY_MASKS_DIR=/data/nerfbusters-visibility-masks

python -m model_eval.run_inference \
    --evalset nerfbusters \
    --checkpoint "$CHECKPOINT" --model_id "$MODEL_ID" \
    --save_dir "$SAVE_DIR" --render_name "$RENDER_NAME" \
    --nerfbusters_dir "$NERFBUSTERS_DIR" \
    --nerfbusters_recon_results_dir "$NERFBUSTERS_RECON_RESULTS_DIR" \
    --nerfbusters_recon_experiment_name "$NERFBUSTERS_RECON_EXPERIMENT" \
    --nerfbusters_captions_dir "$NERFBUSTERS_CAPTIONS_DIR" \
    --save_frame_outputs_only

python -m model_eval.compute_metrics_nerfbusters \
    --render_name "$RENDER_NAME" --eval_base_path "$SAVE_DIR" \
    --nerfbusters_dir "$NERFBUSTERS_DIR" \
    --visibility_masks_dir "$NERFBUSTERS_VISIBILITY_MASKS_DIR"
```

Visibility masks are used only for evaluation, including when comparing
ArtiFixer3D. Real anchors retain their original images during distillation.

## Metric protocols

Full-frame LPIPS uses VGG v0.1 with `--lpips_protocol standard`: RGB is converted
once from `[0, 1]` to `[-1, 1]`. Results go in `metrics_standard/`. Historical
`--lpips_protocol legacy` uses `[0, 1]` inputs and writes `metrics_legacy/`.
Use the same protocol for every method.

Masked LPIPS is separate. Its default TorchMetrics AlexNet implementation
normalizes correctly with either protocol. Optional `alex`, `squeeze`, and
`vgg` masked backends use the selected protocol. Reports record the protocol,
backbone, and input range. Rerunning scoring overwrites files within its
protocol directory.
