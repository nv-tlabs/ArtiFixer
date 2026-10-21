<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Training

ArtiFixer training has three stages. Each stage writes a separate run directory:

| Stage | Entry point | Initialization | Result |
| --- | --- | --- | --- |
| S1: conditioned diffusion finetuning | `model_training.train` | Pretrained base model | Bidirectional teacher |
| S2: autoregressive diffusion forcing | `model_training.diffusion_forcing` | S1 raw transformer | Autoregressive student initialization |
| S3: self forcing DMD | `model_training.distillation` | Student from S2; real and fake score critics from S1 | Four-step autoregressive model and EMA weights |

The commands below use Cosmos3-Nano. See [Wan variants](#wan-variants) to change
the backbone. Run inside the [project environment](setup.md), with FFmpeg
available for validation videos and persistent mounts for data, caches, and runs.
All commands run from the repository root.

## Cosmos3-Nano release configuration

The selected Cosmos3-Nano checkpoint comes from S1 at iteration 25,000, S2
diffusion forcing at iteration 10,000, and a 1,000-iteration DMD run whose
EMA checkpoint at iteration 700 was selected. The recorded settings are:

| Setting | S1 | S2 | S3 DMD |
| --- | --- | --- | --- |
| Raw frames / latent frames | 81 / 21 | 81 / 21 | 81 / 21 |
| Iterations | 25,000 | 10,000 | 1,000 |
| Learning rate | `1e-5` | `1e-5` | Student `2e-6`; critic `4e-7` |
| Weight decay | `0.01` | `0.01` | `0.01` |
| Maximum gradient norm | 1 | 1 | 10 |
| Latent frames per block | Bidirectional | 7 | 7 |
| Local attention window / sink | Not applicable | Not applicable | 21 / 7 |
| Context parallel size | 1 | 1 | 1 |
| Effective batch size | 128 clips | 128 clips | 128 clips |
| Checkpoint interval | 100 | 100 | 100 |
| Validation interval in the recorded run | 1,000 | 1,000 | 1,000 |

DMD uses four denoising steps, teacher text guidance 3, five critic updates per
generator update, EMA decay 0.99, and EMA start step 200. Gradients flow through
cached K/V across chunks; leave `--truncate_rollout_gradients` off. Iterations
count critic-only steps too, so 1,000 iterations do not mean 1,000 generator
updates. All stages use seed 42, conditioning dropout 0.1, dataset scaling
factor 0.01, BF16 compute, and gradient checkpointing.

The base model is `nvidia/Cosmos3-Nano` at revision
`23314034b82f46b45339035dba67c3ee9bbcb8ba`. The loaders resolve the model,
VAE, and tokenizer revision through [pins.py](../model_training/pins.py), which
also pins Diffusers 0.39.0. Training has no `--model_revision` override.
The [model profile](../model_training/model_profiles.py) sets camera pose
alignment to `last`, flow shift to 5, and RoPE FPS to 24. Camera rays use the
first target camera as the fixed origin. The prompt FPS describes the source
clip, independently of RoPE FPS; DL3DV uses 60 FPS at stride 1.

This is the 81-frame release recipe. A 381-frame experiment encodes to 96
latent frames and needs a different block/window layout; changing only
`--num_frames` does not reproduce the release configuration. The commands here
apply the recorded hyperparameters to current code and locally prepared data.
Reproducing identical weights also requires the original source revision,
scene selection, reconstruction assets, captions, and training state.

## Prepare data and choose resources

Follow [DL3DV preparation](data.md) to build scene reconstructions and the split.
Generate the [Cosmos training captions](data.md#cosmos3-training-captions),
including both 81-frame and full-scene files in a separate Cosmos prompt root.
S1/S2 validation uses the training window; DMD training validation uses full
scenes by default. Existing captions can instead be
[retokenized](data.md#reuse-existing-wan-captions).

```bash
export MODEL_ID=nvidia/Cosmos3-Nano
export SPLIT_PATH=/data/trainval_test_split.json
export DL3DV_ROOT=/data/DL3DV-ALL-960P
export PROMPT_ROOT=/data/captions-cosmos3-training
export DCP_CACHE_DIR=/data/.cache/artifixer/dcp_weights
export RUN_ROOT=/data/artifixer-training/cosmos3-run-001
export S1_DIR="$RUN_ROOT/s1"
export S2_DIR="$RUN_ROOT/s2"
export S3_DIR="$RUN_ROOT/s3"
```

Use a new `RUN_ROOT` for a new experiment. The split contains reconstruction
paths; those paths, the scene archives, and prompt files must be readable from
every worker. Keep the Hugging Face cache mounted as described in
[environment setup](setup.md). S1 converts the pretrained transformer into a
fingerprinted DCP cache on first use; place `DCP_CACHE_DIR` on shared persistent
storage and budget for both the original weights and that converted copy.

The release runs used 128 GPUs with one clip per rank and no accumulation.
S1/S2 used the default per-node FSDP shard group; DMD used `--shard_size 128`.
With CP=1, effective batch size is `NUM_PROCESSES * gradient_accumulation_steps`.
Accumulation can preserve that batch size on fewer GPUs, but does not reduce
per-clip activation memory or replace model sharding.

The example launch settings below describe one eight-GPU node with accumulation
16. This is a topology example, not a measured minimum hardware requirement.
Training must fit model shards, activations, optimizer state, and, for DMD,
student, critics, and EMA. [Inference memory estimates](setup.md#hardware) do
not establish training capacity.

```bash
export NUM_MACHINES=1
export NUM_PROCESSES=8  # total GPUs across all machines
export MACHINE_RANK=0
export MAIN_PROCESS_IP=127.0.0.1
export MAIN_PROCESS_PORT=29500
export GPUS_PER_NODE=8
export GRADIENT_ACCUMULATION_STEPS=16

LAUNCH_ARGS=(
    --multi_gpu --mixed_precision bf16
    --num_machines "$NUM_MACHINES" --num_processes "$NUM_PROCESSES"
    --machine_rank "$MACHINE_RANK"
    --main_process_ip "$MAIN_PROCESS_IP" --main_process_port "$MAIN_PROCESS_PORT"
)
TRAIN_ARGS=(
    --model_id "$MODEL_ID"
    --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" --prompt_dir "$PROMPT_ROOT"
    --num_frames 81 --context_parallel_size 1
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --dataset_scaling_factor 0.01 --dropout_rate 0.1 --seed 42
    --dataloader_num_workers 2 --dataloader_prefetch_factor 2
    --checkpoint_every_n_blocks 1 --log_with none
)
```

For the original 128-GPU topology, use 16 eight-GPU machines,
`NUM_PROCESSES=128`, and `GRADIENT_ACCUMULATION_STEPS=1`. Run each stage's
command once per allocated node, with machine ranks 0–15 and a shared,
reachable rank-0 address and port. `NUM_PROCESSES` is the global count.
Use your scheduler's allocation and process-launch mechanism. The existing
[public Slurm example](../model_training/slurm/sample-slurm-submit.sh) launches
S1 and supplies checkpoint/requeue handling; it is not an S2/S3 launcher.

The Bash arrays above are shared by the following commands; keep them in the
same shell or your launch script. `--log_with none` writes local checkpoints
and validation reports without W&B. Set `--log_with wandb` and tracker options
if desired. Do not launch overlapping writers into the same stage directory.

## Stage 1: train the bidirectional teacher

```bash
accelerate launch "${LAUNCH_ARGS[@]}" --module model_training.train \
    "${TRAIN_ARGS[@]}" \
    --project_dir "$S1_DIR" --dcp_cache_dir "$DCP_CACHE_DIR" \
    --shard_size "$GPUS_PER_NODE" \
    --max_iterations 25000 --learning_rate 1e-5 \
    --weight_decay 0.01 --max_grad_norm 1.0 \
    --save_steps 100 --validation_steps 1000 --max_checkpoints 5
```

After S1 completes, use its raw model slot for both the S2 initialization and
the later DMD critics:

```bash
export S1_CHECKPOINT="$S1_DIR/checkpoints/checkpoint_25000/pytorch_model_fsdp_0"
```

## Stage 2: train with diffusion forcing

```bash
accelerate launch "${LAUNCH_ARGS[@]}" --module model_training.diffusion_forcing \
    "${TRAIN_ARGS[@]}" \
    --project_dir "$S2_DIR" --base_checkpoint_dir "$S1_CHECKPOINT" \
    --shard_size "$GPUS_PER_NODE" --frames_per_block 7 \
    --max_iterations 10000 --learning_rate 1e-5 \
    --weight_decay 0.01 --max_grad_norm 1.0 \
    --save_steps 100 --validation_steps 1000 --max_checkpoints 5

export S2_CHECKPOINT="$S2_DIR/checkpoints/checkpoint_10000/pytorch_model_fsdp_0"
```

Diffusion-forcing training requires `--context_parallel_size 1`. Its
`--base_checkpoint_dir` is the exact DCP model slot, not the checkpoint parent
directory or a `.pt` export.

## Stage 3: distill the autoregressive model

```bash
accelerate launch "${LAUNCH_ARGS[@]}" --module model_training.distillation \
    "${TRAIN_ARGS[@]}" \
    --project_dir "$S3_DIR" \
    --base_checkpoint_dir "$S2_CHECKPOINT" \
    --base_checkpoint_dir_critic "$S1_CHECKPOINT" \
    --shard_size "$NUM_PROCESSES" \
    --frames_per_block 7 --local_attn_size 21 --sink_size 7 \
    --num_inference_steps 4 --text_guidance_scale 3 --dfake_gen_update_ratio 5 \
    --max_iterations 1000 --learning_rate 2e-6 --learning_rate_critic 4e-7 \
    --weight_decay 0.01 --max_grad_norm 10.0 \
    --ema_weight 0.99 --ema_start_step 200 \
    --save_steps 100 --validation_steps 1000 --max_checkpoints 10
```

The student and critics must use the same model family. `--model_id_critic`
defaults to `--model_id`. Both critics load the S1 weights; only the fake score
critic is optimized. `--text_guidance_scale` controls the teacher in training;
the distilled `kv_cache` inference pipeline does not use text guidance.

Current DMD defaults are 1,000 iterations, saves and validation every 100,
and retention of the ten latest completed checkpoints. The command above
uses the recorded run's validation interval of 1,000; omit that override for
more frequent validation. With periodic saves only, checkpoints 100 through
1,000 remain available. Extra timeout saves also consume retention slots;
increase `--max_checkpoints` if retaining every periodic checkpoint across
preemptions or a longer run. Retention applies after the replacement save
completes, so budget temporary room for one extra checkpoint. A training
checkpoint includes optimizer and critic state and is much larger than an
inference export.

## Resume, select, and export checkpoints

To resume a stage, repeat its command with the same project directory and add
`--resume_from_checkpoint auto`. This restores the latest completed training
state, including optimizers and RNG. Keep model/data settings and the iteration
horizon fixed for the same sampling sequence. A weights-only initialization
with `--base_checkpoint_dir` starts a new training run and does not restore
optimizer or sampling state. Scheduler-managed runs must also use their
launcher's resume protocol.

Current checkpoint layouts include:

| Path below `checkpoints/checkpoint_N/` | Purpose |
| --- | --- |
| `checkpoint_complete` | Whole-checkpoint completion marker |
| `pytorch_model_fsdp_0/` | Raw teacher (S1), raw autoregressive model (S2), or raw student (S3) |
| `pytorch_model_fsdp_1/` | DMD fake score critic |
| `pytorch_model_fsdp_ema/` | Current DMD EMA student |
| `ema.json` | Current DMD EMA progress and update counts |

Keep `artifixer.metadata.json` inside each model slot. Full resume needs the
whole checkpoint, including optimizer, scheduler, RNG, and EMA progress files.
The historical Cosmos release used EMA slot `pytorch_model_fsdp_2`; newly
trained models use `pytorch_model_fsdp_ema`. Select the slot that actually
exists in the checkpoint you are exporting.

Use [automated validation](../model_training/VALIDATION.md) to score and compare
candidate slots. EMA700 is the release selection, not a guarantee that step
700 is best for a new dataset or run. To export that step from a new run:

```bash
python -m model_eval.export_checkpoint_pt \
    --checkpoint_dir "$S3_DIR/checkpoints/checkpoint_700/pytorch_model_fsdp_ema" \
    --output_pt /data/artifixer-checkpoints/my-cosmos3-dmd-ema700.pt \
    --model_id "$MODEL_ID" --run_id my-cosmos3-dmd \
    --checkpoint 700 --slot pytorch_model_fsdp_ema
```

Export writes transformer weights and a matching `.metadata.json` sidecar.
Keep both files together and use the [Cosmos inference recipe](inference.md#cosmos3-nano).
An S1 teacher can be exported with the same command using its raw slot and
step; inference uses `--inference_pipeline bidirectional --num_inference_steps 50`.
Full-state checkpoints remain necessary for training resumes. Training
initialization currently accepts DCP model slots, not the exported `.pt` files.

## Wan variants

Set `MODEL_ID` before constructing `TRAIN_ARGS`, and use a separate run root,
Wan T5 caption directory, and matching S1/S2 parents throughout:

| Backbone | `MODEL_ID` | Prompt format | Default pose alignment |
| --- | --- | --- | --- |
| Cosmos3-Nano | `nvidia/Cosmos3-Nano` | `cosmos3_omni_token_ids` | `last` |
| Wan2.1 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `wan_t5_embedding` | `mean` |
| Wan2.1 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `wan_t5_embedding` | `mean` |

The same entry points support these backbones. The table and commands above
describe the Cosmos release settings; changing `MODEL_ID` does not establish
reproduction of a released Wan checkpoint. Each model's pose alignment is
resolved automatically and must agree with its parent checkpoint metadata.

The released Wan S1 bidirectional teachers are `artifixer-s1-14b.pt` and
`artifixer-s1-1.3b.pt` in [nvidia/ArtiFixer](https://huggingface.co/nvidia/ArtiFixer).
They are usable for inference and evaluation as described in the
[checkpoint reference](inference.md#checkpoints). A `.pt` teacher cannot be
passed directly as `--base_checkpoint_dir`; use the DCP S1 output for the
three-stage training workflow. A Cosmos S1 download is not required to run
this guide: stage 1 produces that teacher from the pinned base model.
