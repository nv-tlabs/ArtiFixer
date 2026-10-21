# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MAX_SEQUENCE_LENGTH = 512
COSMOS3_PROMPT_FORMAT = "cosmos3_omni_token_ids"
WAN_T5_PROMPT_FORMAT = "wan_t5_embedding"

# DL3DV-10K source clips are 60 fps at unit frame stride. Every consumer of
# the dataset's native frame rate — caption/prompt-HDF5 generation
# (generate_captions.py's dataset_fps default,
# retokenize_cosmos3_prompts.py's stride-scaled fps stamp) — shares this one
# constant instead of re-stating its own 60.
DL3DV_DATASET_FPS = 60
