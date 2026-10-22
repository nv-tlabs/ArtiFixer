#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Flatten, verify the exact flattened artifact, then assign a new local tag.
# Usage: flatten_release_image.sh SOURCE DESTINATION [--platform OS/ARCH] [--base-image BASE --report receipt.json]
# No registry writes. Existing destination tags and receipts are preserved.
set -euo pipefail
exec python3 "$(dirname "$0")/release_image.py" flatten "$@"
