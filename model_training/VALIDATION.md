<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Automated validation

Use this workflow for DCP checkpoints produced by training. It scores validation
clips and replays fixed panels on saved checkpoints. For released `.pt` weights,
use [benchmark inference](../docs/evaluation.md).

| Workflow | Entry points |
| --- | --- |
| Training validation and DCP checkpoint panels | `model_training.train`, `diffusion_forcing`, `distillation` |
| Compare saved panel reports | `model_training.compare_validation_results` |
| Standalone benchmark inference | `model_eval.run_inference` |
| Corpus metrics on generated benchmark outputs | `model_eval.compute_metrics_dl3dv`, `compute_metrics_nerfbusters` |
| Export checkpoints and their pose-alignment metadata | `model_eval.export_checkpoint_pt` |

All three training entrypoints (`train`, `diffusion_forcing`, and
`distillation`) score regular validation clips. `--test_panel_scenes N` also
evaluates a fixed panel in the full-conditioning and zero-rendered configurations.
DMD uses its KV-cache sampler for both passes, including context parallelism,
and the same active EMA weights for its regular clips and fixed panels. The
original weights and mode are restored even if sampling fails.
EMA media filenames keep the `_ema` suffix; metric task keys and noise seeds stay
fixed, and `results.json` records whether the pass used raw or EMA weights.

The panel adds two generations per sample. The samples are **scene-halves** from
the held-out split; N does not necessarily mean N distinct source scenes. Regular
`val/*` clips rotate through samples; `test/*` compares the fixed panel. Seeds are
fixed per task and independent of training RNG. Different samplers consume noise
differently, so a shared seed does not imply identical latent noise across stages.

Scoring is enabled by default. Use `--no_validation_metrics` for ordinary training
validation clips without numeric scoring, JSON reports, panel generations, or source
fingerprinting. The panel is off by default (`--test_panel_scenes 0`); disabling metrics
also suppresses it when a nonzero count is supplied. `--validate_only` and
`--validation_strict` require metrics and reject that opt-out. An explicitly supplied
`--validation_manifest` still verifies its source files and conditioning even when
metrics are disabled, because replay must enforce the captured input contract.

## Evaluate a checkpoint

`--validate_only` loads the selected model and runs evaluation without creating
training dataloaders, optimizers, critics, or a separate EMA model.

Pass the exact DCP model subdirectory to `--base_checkpoint_dir`, including
the EMA model's subdirectory if evaluating saved EMA weights. The artifact records
this path as `loaded_checkpoint`; it does not silently select another weight set.
`--resume_from_checkpoint` is incompatible with this mode.
Evaluation resolves `camera_pose_alignment` from that model slot, not the
checkpoint's parent directory. An explicit `--camera_pose_alignment` must match
the resolved value; conflicts are rejected before dataset construction. See
[Checkpoint reference](../docs/inference.md#checkpoints) for pose-record requirements and camera coordinates.

Set `MODEL_ID`, `EVAL_ROOT`, `SPLIT_PATH`, `DL3DV_ROOT`, and `PROMPT_ROOT` first.
Set `S1_CHECKPOINT` and `S1_CANDIDATE_CHECKPOINT` for S1 comparisons, or
`S2_CHECKPOINT` and `S2_CANDIDATE_CHECKPOINT` for S2. These examples use
381-frame validation; generate matching
381-frame captions for long scenes before running it, using
`data_processing.run_captioning --num_frames 381`:

```bash
accelerate launch --num_processes 8 --module model_training.train \
  --model_id "$MODEL_ID" \
  --base_checkpoint_dir "$S1_CHECKPOINT" \
  --project_dir "$EVAL_ROOT/s1" \
  --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" --prompt_dir "$PROMPT_ROOT" \
  --num_frames 381 --context_parallel_size 4 \
  --validate_only --test_panel_scenes 8 --log_with none
```

Choose a frame count and CP size compatible with every stage/model being compared.
For block size 8, 381 raw frames give 96 latent frames; context-parallel
validation can split these over four CP ranks with 24 latent frames per rank.
For S2 diffusion-forcing capture and replay, explicitly pass
`--frames_per_block 8 --context_parallel_size 1`: 96 latent frames divide into
blocks of 8, but not the default blocks of 7. W&B remains
optional; JSON artifacts are always written when metrics are enabled.

Every validation writes `project_dir/val/step_NNNNNN/results.json`; a completed
panel input capture also writes `manifest.json`. Standalone evaluation uses step 0.
The version-3 manifest pins scene/half, raw frame indices, neighbor indices, prompt
filename and window, prompt content hash, output resolution, pose alignment,
dataset scale, and the fixed noise-seed policy. Each sample also records the
per-reconstruction metric scale and content checksum for every available view-count
variant, the effective unconditional tensor shapes/dtypes/content hashes (or null
for a sampler without CFG), a checksum of the camera geometry, and a checksum of the source scene ZIP
(including its images). Replay rejects changed scales, reconstruction assets,
geometry, source images, or effective negative conditioning before generation. These identities contain no host paths,
so unchanged assets can be relocated.

The first access reads each source file to compute its SHA-256; subsequent accesses
cache fingerprints by file identity and timestamps. Hashing streams the files and
does not load whole archives into memory. Source assets must stay fixed during an
evaluation pass.

Replay those inputs with another checkpoint through the same pipeline:

```bash
accelerate launch --num_processes 8 --module model_training.train \
  --model_id "$MODEL_ID" \
  --base_checkpoint_dir "$S1_CANDIDATE_CHECKPOINT" \
  --project_dir "$EVAL_ROOT/s1-candidate" \
  --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" --prompt_dir "$PROMPT_ROOT" \
  --validation_manifest "$EVAL_ROOT/s1/val/step_000000/manifest.json" \
  --context_parallel_size 4 \
  --validate_only --log_with none
```

The manifest selects the evaluation length before layout checks and supplies the
panel size. It also replays the same frames/prompts for regular validation; only
the three-neighbor panel has its neighbor selections pinned. For S2, capture and replay
with `model_training.diffusion_forcing --frames_per_block 8 --context_parallel_size 1`
for the 381-frame example above, replacing the S1 CP=4 option on both commands.
For S3, use `model_training.distillation`. Each
pipeline enforces the effective negative conditioning recorded in its manifest.

For example, capture the S2 panel with the explicit block layout:

```bash
accelerate launch --num_processes 8 --module model_training.diffusion_forcing \
  --model_id "$MODEL_ID" --base_checkpoint_dir "$S2_CHECKPOINT" \
  --project_dir "$EVAL_ROOT/s2" \
  --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" --prompt_dir "$PROMPT_ROOT" \
  --num_frames 381 --frames_per_block 8 --context_parallel_size 1 \
  --validate_only --test_panel_scenes 8 --log_with none
```

Replay with the same block layout; the manifest supplies the 381-frame length:

```bash
accelerate launch --num_processes 8 --module model_training.diffusion_forcing \
  --model_id "$MODEL_ID" --base_checkpoint_dir "$S2_CANDIDATE_CHECKPOINT" \
  --project_dir "$EVAL_ROOT/s2-candidate" \
  --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" --prompt_dir "$PROMPT_ROOT" \
  --validation_manifest "$EVAL_ROOT/s2/val/step_000000/manifest.json" \
  --frames_per_block 8 --context_parallel_size 1 \
  --validate_only --log_with none
```

Without a manifest, standalone DMD evaluation honors `--num_frames`, like S1/S2.
During training, DMD retains its full-clip validation default. Set
`--validation_num_frames` to use a common validation length during any stage's
training; training clip length is unaffected. A manifest overrides validation
length and rejects a conflicting `--validation_num_frames`.

## Completeness and comparisons

`results.json` records checkpoint/run identity, code revision when available,
sampler settings, device/world size, per-task CP size, seeds, per-task metrics,
errors, missing metrics, and expected/scored/complete task counts. Metric counts
make the denominator of each aggregate explicit. No repeated camera poses is
recorded as `revisit_status: not_applicable` with zero pairs, rather than a missing
or perfect score.

`--validate_only` always requires complete, finite results. During training,
`--validation_strict` enables the same behavior; otherwise metric failures are
recorded while training continues. Results are written before strict mode raises,
and every distributed rank receives the failure. Capping the panel with
`--max_validation_tasks`, or requesting more samples than exist, leaves it
incomplete and fails strict evaluation. A data-loading or generation failure
still fails the job directly; the JSON completeness report covers scoring failures,
not recovery from failed distributed inference.

To compare a student against an independently evaluated teacher, evaluate both with
the same validation pipeline and effective negative conditioning. For example, use
`model_training.distillation --validate_only` for both teacher and student checkpoints
when comparing KV-cache rollouts. CFG and non-CFG runs have different effective
conditioning identities and are not accepted as paired inputs:

```bash
python -m model_training.compare_validation_results \
  --reference "$EVAL_ROOT/s1/val/step_000000/results.json" \
  --candidate "$EVAL_ROOT/s1-candidate/val/step_000000/results.json" \
  --output "$EVAL_ROOT/s1-comparison.json"
```

The command requires complete version-3 panels, matching manifest identifiers, matching
task coverage, the same metric protocol, and the same chunk-metric grid. It reports candidate-minus-reference
quality metrics. Generation speedups are included only for matching GPU model,
world size, and per-task CP size. Timings include encoding, denoising, decoding,
and CPU output transfer, and exclude metric scoring and media writing. They are
diagnostics rather than warmed throughput benchmarks.

## Interpreting the metrics

PSNR and SSIM use the same lightweight tensor kernels as offline benchmark
scoring. Both reduce each frame first and then average frames; uneven device
chunks do not change their weighting. Validation explicitly floors per-frame
MSE at `1e-12` for finite PSNR (maximum 120 dB), while benchmark PSNR retains
infinity for an exact match and has no floor. SSIM uses the benchmark's 11-pixel
Gaussian window (sigma 1.5), zero padding, and per-frame spatial/channel mean.
These policies are recorded in `results.json` as `metric_protocol`; the comparison
command rejects differing policies. No LPIPS or FID model is initialized
or imported by validation scoring.

Ablations are diagnostic interventions and may improve or worsen a score. The
aggregate over unmodified tasks excludes them. Rendered-baseline scores and gains
are omitted for full or partial rendered-input dropout; partial dropout still
provides rendered conditioning, but it no longer has an unmodified baseline.

- **PSNR/SSIM and gain over the render** measure reconstruction fidelity. Gain
  provides a per-clip baseline; it does not remove all scene difficulty.
- **Revisit consistency** measures repeatability at approximately repeated poses,
  including rotation-only returns. A frozen video can score perfectly. It is not
  an estimate of camera-pose accuracy and must not be a standalone quality gate.
- **Temporal delta L1** compares generated consecutive-frame differences with GT;
  lower is better. `motion_generated`, `motion_gt`, and `motion_ratio` expose a
  frozen prediction when GT moves. These are pixel-space diagnostics, affected by
  appearance changes as well as camera motion. `motion_ratio` is omitted when GT
  has effectively no pixel motion.
- **Chunk PSNR and last-minus-first PSNR** expose deterioration over time.
  **Boundary delta L1** measures temporal-difference error at a fixed raw-frame
  grid, set with `--validation_chunk_frames` (default 32). This grid supports
  common comparisons. `sampler_boundary_delta_l1` additionally measures errors
  at the stage's actual latent block boundaries, accounting for the first latent
  owning only one raw frame (e.g. the first boundary is frame 29 for block size 8
  and temporal scale 4).

Use fidelity, motion, and consistency together. Thresholds need calibration on
accepted/rejected checkpoints; the strict flag checks evaluation completeness,
not model quality. This path adds no learned metric models or metric downloads.
