<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DL3DV preparation

Training and DL3DV evaluation use scene archives, reconstruction HDF5 files,
captions, and a generated train/test split. For a single COLMAP scene, use
the [quick start](../README.md#3-prepare-a-scene).

The captioning and reconstruction wrappers process every scene zip under
`--dl3dv_dir` by default. For benchmark evaluation, use the union of the shipped
test-scene lists and pass it to both wrappers:

```bash
sort -u data_processing/dl3dv_benchmark_{ours,difix}_test_scenes.txt \
    > /data/benchmark_scenes.txt
```

For benchmark evaluation, download those scenes and add
`--scene_list /data/benchmark_scenes.txt` to the captioning and reconstruction
commands below. For one scene, use `--scene_id`. Building the benchmark split
requires every scene in the combined list.

## 1. Download scenes and COLMAP models

Both the [scene archives](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P)
and [COLMAP cache](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-ColmapCache)
are gated: accept each dataset's terms and run `hf auth login` first.
For benchmark scenes only:

```bash
while IFS= read -r scene; do
    hf download DL3DV/DL3DV-ALL-960P --repo-type dataset \
        --include "*/$scene.zip" --local-dir /data/DL3DV-ALL-960P
    hf download DL3DV/DL3DV-ALL-ColmapCache --repo-type dataset \
        --include "*/$scene.zip" --local-dir /data/DL3DV-ALL-ColmapCache
done < /data/benchmark_scenes.txt
```

To download the full datasets for training, run each `hf download` command once
without `--include`.

Both datasets use `<subdir>/<scene_id>.zip`. Reconstruction reads
`<scene_id>/colmap/sparse/0/{cameras,images,points3D}.bin` from each cache zip;
the full cache is about 4 TB because it also includes databases and features.
You may retain only those three files if you preserve their paths and zip layout.

## 2. Generate captions

Choose the prompt length for the workflow:

| Workflow | `--num_frames` | Wan output filename |
| --- | --- | --- |
| Standalone benchmark evaluation | `0` | `frames_all_stride_1.h5` |
| Default 81-frame training and training validation | `81` | `frames_81_stride_1.h5` |

```bash
export PROMPT_ROOT=/data/captions
python -m data_processing.run_captioning \
    --dl3dv_dir /data/DL3DV-ALL-960P \
    --output_dir "$PROMPT_ROOT" \
    --scene_list /data/benchmark_scenes.txt --num_frames 0
```

For 81-frame training windows, run separately:

```bash
python -m data_processing.run_captioning \
    --dl3dv_dir /data/DL3DV-ALL-960P \
    --output_dir /data/captions \
    --num_frames 81
```

For other training lengths, repeat the command with the matching `--num_frames`.

The files coexist under `<output_dir>/<subdir>/<scene_id>/`. Scenes shorter
than the training window use `frames_all_stride_1.h5`, matching the loader's
fallback. Add `--include_reverse` or `--include_revisit` for training variants
you intend to use. Inference chunk length is configured separately.

For Cosmos3, add `--prompt_format cosmos3_omni_token_ids --dataset_fps FPS`
using the source frame rate and a separate output root. Training retains the
filenames above. Standalone evaluation also needs `--output_role evaluation`
with `--num_frames 0`; it writes `frames_all_stride_1.cosmos3.h5`. Evaluation
requires stride 1 without reverse or revisit variants.

For example, generate full-scene Cosmos benchmark prompts with:

```bash
export PROMPT_ROOT=/data/captions-cosmos3-evaluation
export DATASET_FPS=60  # set to the source sequence's actual frame rate
python -m data_processing.run_captioning \
    --dl3dv_dir /data/DL3DV-ALL-960P --output_dir "$PROMPT_ROOT" \
    --scene_list /data/benchmark_scenes.txt \
    --captioning_model_id Qwen/Qwen3-VL-4B-Instruct \
    --num_frames 0 --output_role evaluation \
    --prompt_format cosmos3_omni_token_ids --dataset_fps "$DATASET_FPS"
```

Existing captions are checked for representation and, for Cosmos3, FPS. After
changing source images, text, or caption settings, regenerate into a new
`--output_dir` and update `--prompt_dir` in later commands.

### Cosmos3 training captions

For the [Cosmos3-Nano training recipe](training.md), generate both 81-frame
windows and full-scene captions into a dedicated Cosmos root. The full-scene
files are used by DMD's training validation and the short-scene fallback:

```bash
export PROMPT_ROOT=/data/captions-cosmos3-training
export DATASET_FPS=60  # native DL3DV frame rate at stride 1
for NUM_FRAMES in 81 0; do
    python -m data_processing.run_captioning \
        --dl3dv_dir /data/DL3DV-ALL-960P \
        --output_dir "$PROMPT_ROOT" \
        --captioning_model_id Qwen/Qwen3-VL-4B-Instruct \
        --num_frames "$NUM_FRAMES" --output_role training \
        --prompt_format cosmos3_omni_token_ids --dataset_fps "$DATASET_FPS"
done
```

Omit benchmark-only scene filters when preparing the training corpus. The 4B
captioner is a lower-memory choice; changing the captioner changes the training
text. If the intended corpus includes reverse or revisit variants, add
`--include_reverse` or `--include_revisit`. The loader samples matching stride-1
variants already present in the root, so keep the variant set fixed across a
run and its resumes.

Cosmos stores integer prompt token IDs and unconditional token IDs, rather
than Wan's T5 embeddings. Prompt construction includes frame count, image
geometry, and FPS and uses the tokenizer revision in
[pins.py](../model_training/pins.py). Do not mix Wan embeddings and Cosmos token
IDs under the same training filenames. Training and standalone evaluation use
different naming conventions:

| Consumer | Caption command | File under `<PROMPT_ROOT>/<subdir>/<scene_id>/` |
| --- | --- | --- |
| 81-frame training and S1/S2 validation | `--num_frames 81 --output_role training` | `frames_81_stride_1.h5` |
| DMD training validation and short-scene fallback | `--num_frames 0 --output_role training` | `frames_all_stride_1.h5` |
| Standalone Cosmos benchmark inference | `--num_frames 0 --output_role evaluation` | `frames_all_stride_1.cosmos3.h5` |

Short scenes produce `frames_all_stride_1.h5` even when 81 frames are requested.
For custom frame rates, set `--dataset_fps` to the source rate. Temporal stride
is incorporated into the stored prompt FPS; the model's RoPE spacing remains
24 FPS independently. A full-scene file bearing the `.cosmos3.h5` suffix does
not replace the unsuffixed file required by training validation.

### Reuse existing Wan captions

If you already have caption HDF5 files, reuse their stored caption text and
frame indices instead of running the captioner again:

```bash
python -m data_processing.captioning.retokenize_cosmos3_prompts \
    --input_dir /data/captions-wan \
    --output_dir /data/captions-cosmos3-training \
    --dl3dv_dir /data/DL3DV-ALL-960P
```

This tool reads scene dimensions from the DL3DV archives, applies the existing
camera-sentence scrub to caption text, and tokenizes with the pinned Cosmos
tokenizer. It does not load the captioner or the full Cosmos transformer.
With separate input/output roots, filenames are preserved for training. The
source tree must contain both windowed and full-scene captions if both are
needed; retokenization does not generate missing caption windows or new text.

The tool preserves recorded effective FPS. Legacy captions without FPS use
the DL3DV source rate of 60 divided by the filename's stride; use fresh
caption generation with the correct `--dataset_fps` for other source rates
without recorded FPS. Existing destinations are skipped unless `--replace`
is passed. Prefer a new root when changing tokenization or source assets.
Using the same input and output root instead writes format-suffixed
`.cosmos3.h5` siblings for evaluation; it does not create a Cosmos training tree.

## 3. Build sparse reconstructions

Before this step, create a COLMAP model for DiFix test scene
`fb3b73f1d3fe9d192f21f55f5100fd258887aef345f778e0a64fc0587930a6f9`.
The cache has none, and an explicitly listed scene without a model stops the
wrapper. Put its three model files in `8K/<scene_id>.zip` using the layout above.

```bash
python -m data_processing.run_sparse_reconstruction \
    --dl3dv_dir /data/DL3DV-ALL-960P \
    --colmap_zip_root /data/DL3DV-ALL-ColmapCache \
    --output_root /data/reconstructions \
    --work_root /data/artifixer-work/reconstructions \
    --scene_list /data/benchmark_scenes.txt \
    --num_selected_indices 2 3 6 12
```

The wrapper creates [half-covisibility splits](../data_processing/sparse_recon/half_covisibility_sampling.md),
trains each scene-half/view-count reconstruction, estimates metric scale with MoGe, and converts the results to
HDF5. Omit `--scene_list` for all scenes when preparing training data. It copies
`data_*.h5`, `parsed_*.yaml`, and `ckpt_last_*.pt` into:

```text
<output_root>/dl3dv_<subdir>/<scene_id>/
```

`--colmap_zip_root` is needed unless the scene zips already include
`colmap/sparse/0/` or `sparse/0/`. Scenes with missing or invalid COLMAP models
are skipped during a directory scan and listed in
`dl3dv_<subdir>/skipped_scenes.txt`. An explicitly requested missing scene fails.

For offline MoGe use, download its checkpoint in advance and set
`MOGE_MODEL_PATH=/path/to/moge/model.pt` to the checkpoint file.

## 4. Write the split

```bash
python -m data_processing.trainval_test_split \
    --data_path /data/reconstructions \
    --dl3dv_dir /data/DL3DV-ALL-960P \
    --output_root /data
```

The command validates scene archives, reconstruction coverage, duplicate files,
and known bad scenes, then writes `trainval_test_split.json`. Keep `PROMPT_ROOT`
set to the caption directory chosen in step 2, and use these roots for training
and evaluation:

```bash
export SPLIT_PATH=/data/trainval_test_split.json
export DL3DV_ROOT=/data/DL3DV-ALL-960P
```

## Training settings

See [Training](training.md) for complete S1/S2/S3 commands, the Cosmos release
configuration, effective batch size, checkpoint selection, and exports.

All stages default to `--model_id Wan-AI/Wan2.1-T2V-14B-Diffusers` and 81 raw
frames (21 latent frames). Diffusion forcing and DMD use `--frames_per_block 7`;
DMD also uses `--local_attn_size 21 --sink_size 7`, matching the released
checkpoints. Diffusion-forcing training requires `--context_parallel_size 1`.
Longer clips need a larger attention window or `--local_attn_size -1` to retain
history for backward recomputation. For inference, use the checkpoint's
training geometry.
Pass `--model_id` explicitly when using another backbone, matching its checkpoints.

Frame, prompt, and reference sampling is seeded per logical fetch. With the same
configuration and source revision, resume preserves the data sequence across
DataLoader worker schedules.

`--max_checkpoints` removes older checkpoints only after a replacement has its
completion marker. DMD defaults to 1,000 iterations, a save every 100, and
retention of the ten latest completed checkpoints; S1/S2 retain five by
default. Extra timeout saves count toward the same limit. Allow space for
one extra checkpoint during saving.
