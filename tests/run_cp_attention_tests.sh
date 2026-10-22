#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrapper script for multi-GPU CP attention tests.
# Run inside the artifixer container with torchrun + pytest.
set -euo pipefail

export PATH=/usr/local/bin:/usr/bin:/bin
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${REPO_DIR}"

NUM_GPUS="${1:-2}"

echo "================================================================"
echo "CP Attention Tests ($(hostname) @ $(date))"
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================================================"

python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, {torch.cuda.get_device_name()}')"

torchrun --nproc_per_node="${NUM_GPUS}" -m pytest tests/test_cp_attention.py -v -x 2>&1

echo "================================================================"
echo "CP Attention Tests COMPLETE"
echo "================================================================"
