#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# MoGe's image operations work with headless OpenCV. Build an explicitly
# versioned downstream wheel whose dependency names the codec-free source
# build, rather than leaving an unsatisfied opencv-python requirement that a
# later pip install could satisfy with a bundled-FFmpeg wheel.
revision=07444410f1e33f402353b99d6ccd26bd31e469e8
version=2.0.0+artifixer.1
build_root=$(mktemp -d)
trap 'rm -rf "$build_root"' EXIT
git -C "$build_root" init upstream
git -C "$build_root/upstream" remote add origin https://github.com/microsoft/MoGe.git
git -C "$build_root/upstream" fetch --depth 1 origin "$revision"
git -C "$build_root/upstream" checkout --detach FETCH_HEAD
test "$(git -C "$build_root/upstream" rev-parse HEAD)" = "$revision"
python - "$build_root/upstream/pyproject.toml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
for old, new in (
    ('version = "2.0.0"', 'version = "2.0.0+artifixer.1"'),
    ('"opencv-python"', '"opencv-python-headless==4.12.0.88"'),
):
    assert text.count(old) == 1, f"Unexpected MoGe metadata: {old}"
    text = text.replace(old, new)
path.write_text(text)
PY
mkdir -p /opt/artifixer/sources "$build_root/source"
git -C "$build_root/upstream" diff > "/opt/artifixer/sources/moge-${version}.patch"
git -C "$build_root/upstream" archive HEAD | tar -x -C "$build_root/source"
cp "$build_root/upstream/pyproject.toml" "$build_root/source/pyproject.toml"
cat > "$build_root/source/ARTIFIXER-SOURCE.txt" <<EOF
Upstream: https://github.com/microsoft/MoGe
Revision: $revision
Downstream version: $version
Change: require the source-built, FFmpeg-disabled opencv-python-headless distribution.
No MoGe model code changes. Upstream MIT license preserved in LICENSE.
EOF
tar -czf "/opt/artifixer/sources/moge-${version}.tar.gz" -C "$build_root/source" .
python -m pip wheel --no-deps "$build_root/source" -w "$build_root/wheels"
python -m pip install "$build_root/wheels/"moge-*.whl
