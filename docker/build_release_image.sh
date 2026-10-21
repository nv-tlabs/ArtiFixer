#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build an ArtiFixer release image the way the published images are built,
# verify the codec-free contract at runtime, and optionally flatten it for
# registry publication.
#
# Usage:
#   docker/build_release_image.sh <cuda12|cuda13|cuda13-aarch64> [options]
#
# Options:
#   --no-fa3               Skip the FlashAttention-3 build (BUILD_FLASH_ATTN_3=0).
#                          FA3 is Hopper-only: A100, B200 and RTX PRO 6000 never
#                          load it, and the build drops from ~35 min (20 nvcc jobs
#                          on a 64-core, 125 GB host) to minutes.
#                          No effect on the aarch64 image, which never builds it.
#   --flatten              Also produce <tag>-flat with docker/flatten_release_image.sh.
#                          A registry-published image must be the flattened one:
#                          the NGC base layers still carry libav bytes that the
#                          Dockerfile only deletes in a later layer.
#   --tag NAME             Image tag (default: artifixer:<variant>-<YYYYMMDDTHHMMSSZ>-<git sha>).
#   --check-only           Skip the build; only run the runtime checks against --tag
#                          (e.g. on a pulled image).
#   --report-dir PATH      New directory for build and final image JSON receipts.
#                          Defaults to a unique directory under /tmp.
#   --base-image IMAGE     Base for the build and inventories (default: Dockerfile ARG).
#                          Builds pull a missing or wrong-platform base;
#                          --check-only requires the correct base locally.
#   --repo-host-path PATH  Deprecated compatibility option; checks use stdin and
#                          do not mount the checkout over the image filesystem.
#
# Environment:
#   FLASH_ATTN_MAX_JOBS      parallel nvcc jobs for FA3 (default: half the cores,
#                            capped at RAM/6 GB: each job takes ~5-6 GB)
#   FLASH_ATTN_NVCC_THREADS  threads per nvcc job (default: 2)
#   OPENCV_BUILD_JOBS        parallel jobs for the opencv source build (default: cores)
#   ARTIFIXER_GIT_REVISION   value for the org.opencontainers.image.revision label
#                            (default: `git rev-parse HEAD`, or "unknown")
#   ARTIFIXER_SOURCE_URL     source repository URL for the image provenance label
#                            (default: public ArtiFixer repository)
set -euo pipefail

usage() {
    sed -n '4,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 2
}

variant=${1:-}
case "$variant" in
    cuda12 | cuda13 | cuda13-aarch64) shift ;;
    *) usage ;;
esac

fa3=1
flatten=0
check_only=0
tag=""
report_dir=""
base_image=""
while [ $# -gt 0 ]; do
    case "$1" in
        --no-fa3) fa3=0 ;;
        --flatten) flatten=1 ;;
        --check-only) check_only=1 ;;
        --tag) tag=${2:?--tag needs a value}; shift ;;
        --repo-host-path) : "${2:?--repo-host-path needs a value}"; shift ;;
        --report-dir) report_dir=${2:?--report-dir needs a value}; shift ;;
        --base-image) base_image=${2:?--base-image needs a value}; shift ;;
        -h | --help) usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
    shift
done

cd "$(dirname "$0")/.."
[ -f "Dockerfile.$variant" ] || { echo "Dockerfile.$variant not found (run from the repository)" >&2; exit 2; }
if [ "$check_only" = "0" ]; then
    [ -f thirdparty/3DGRUT-ArtiFixer/setup.py ] || [ -f thirdparty/3DGRUT-ArtiFixer/pyproject.toml ] \
        || { echo "thirdparty/3DGRUT-ArtiFixer is empty: run 'git submodule update --init --recursive' first" >&2; exit 2; }
    # The 3DGUT tracer is JIT-compiled at first use against the nested submodules
    # (tiny-cuda-nn and friends). An image built from a checkout without them
    # builds fine and fails at the first reconstruction, so refuse early.
    [ -f thirdparty/3DGRUT-ArtiFixer/thirdparty/tiny-cuda-nn/include/tiny-cuda-nn/common.h ] \
        || { echo "thirdparty/3DGRUT-ArtiFixer/thirdparty/* is empty: run 'git submodule update --init --recursive' (nested submodules)" >&2; exit 2; }
fi

cores=$(nproc)
mem_gb=$(awk '/MemTotal/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
cross_build=0
[ "$variant" = "cuda13-aarch64" ] && [ "$(uname -m)" != "aarch64" ] && cross_build=1
# Each FA3 nvcc job takes ~5-6 GB of RAM: 32 jobs on a 125 GB host got the
# build OOM-killed, 24 succeeded. Bound the default by memory, not just cores.
by_cores=$(( cores / 2 > 0 ? cores / 2 : 1 ))
by_memory=$(( mem_gb / 6 > 0 ? mem_gb / 6 : 1 ))
: "${FLASH_ATTN_MAX_JOBS:=$(( by_cores < by_memory ? by_cores : by_memory ))}"
: "${FLASH_ATTN_NVCC_THREADS:=2}"
# Under QEMU every compiler is an emulated process; keep the opencv build at
# the Dockerfile's default parallelism instead of one job per core.
if [ "$cross_build" = "1" ]; then
    : "${OPENCV_BUILD_JOBS:=4}"
else
    : "${OPENCV_BUILD_JOBS:=$(( cores < by_memory ? cores : by_memory ))}"
fi
revision=${ARTIFIXER_GIT_REVISION:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}
created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source_url=${ARTIFIXER_SOURCE_URL:-https://github.com/nv-tlabs/ArtiFixer}
source_dirty=0
[ -n "$(git status --porcelain --untracked-files=normal)" ] && source_dirty=1
if [ "$check_only" = "1" ] && [ -z "$tag" ]; then
    echo "--check-only requires --tag" >&2; exit 2
fi
tag=${tag:-artifixer:${variant}-$(date -u +%Y%m%dT%H%M%SZ)-${revision:0:7}}
if [ -n "$report_dir" ]; then
    mkdir -p "$(dirname "$report_dir")"
    mkdir "$report_dir"  # a retry must use a new receipt directory
else
    report_dir=$(mktemp -d "${TMPDIR:-/tmp}/artifixer-image-check.XXXXXXXX")
fi
report_dir=$(cd "$report_dir" && pwd)
expected_platform=linux/amd64
[ "$variant" = "cuda13-aarch64" ] && expected_platform=linux/arm64
base_image=${base_image:-$(sed -n 's/^ARG ARTIFIXER_BASE_IMAGE=//p' "Dockerfile.$variant")}
[[ -n "$base_image" ]] || { echo "Missing ARTIFIXER_BASE_IMAGE default" >&2; exit 2; }

resolve_base_image() {
    local base_record
    local format='{{.Id}} {{.Os}}/{{.Architecture}}'
    if [ "$check_only" = "0" ]; then
        base_root_id=$(docker image inspect --format '{{.Id}}' "$base_image" 2>/dev/null) || base_root_id=""
        base_record=""
        if [ -n "$base_root_id" ]; then
            base_record=$(docker image inspect --platform "$expected_platform" --format "$format" "$base_root_id" 2>/dev/null) || base_record=""
        fi
        if [ "${base_record#* }" != "$expected_platform" ]; then
            docker pull --platform "$expected_platform" "$base_image"
            base_root_id=$(docker image inspect --format '{{.Id}}' "$base_image")
            base_record=$(docker image inspect --platform "$expected_platform" --format "$format" "$base_root_id")
        fi
    else
        base_root_id=$(docker image inspect --format '{{.Id}}' "$base_image")
        base_record=$(docker image inspect --platform "$expected_platform" --format "$format" "$base_root_id")
    fi
    base_id=${base_record%% *}
    if [ "${base_record#* }" != "$expected_platform" ]; then
        echo "Base image $base_image has platform ${base_record#* }; expected $expected_platform" >&2
        return 1
    fi
    printf '%s\n' "$base_id" > "$report_dir/base-image.id"
    printf '%s\n' "$base_root_id" > "$report_dir/base-image.root-id"
}

for value in "$FLASH_ATTN_MAX_JOBS" "$FLASH_ATTN_NVCC_THREADS" "$OPENCV_BUILD_JOBS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Build concurrency must be positive integers" >&2; exit 2; }
done

if [ "$check_only" = "1" ]; then
    resolve_base_image
    # Freeze the addressable root once; verification selects the requested
    # platform from that immutable root, including containerd image indexes.
    source_id=$(docker image inspect --format '{{.Id}}' "$tag")
fi

build_args=(
    --platform "$expected_platform"
    --build-arg "FLASH_ATTN_MAX_JOBS=$FLASH_ATTN_MAX_JOBS"
    --build-arg "FLASH_ATTN_NVCC_THREADS=$FLASH_ATTN_NVCC_THREADS"
    --build-arg "OPENCV_BUILD_JOBS=$OPENCV_BUILD_JOBS"
)
if [ "$variant" = "cuda13-aarch64" ]; then
    # The aarch64 image can be cross-built on x86 under QEMU (about 3 hours
    # for the opencv source build alone; a native GB200/Grace host is faster).
    # This needs the arm64 binfmt handler registered on the daemon's host:
    #   docker run --privileged --rm tonistiigi/binfmt --install arm64
    if [ "$cross_build" = "1" ] && [ ! -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]; then
        echo ">> note: no qemu-aarch64 binfmt handler visible here; the cross-build fails unless the daemon host has one registered" >&2
    fi
else
    build_args+=(--build-arg "BUILD_FLASH_ATTN_3=$fa3")
fi

if [ "$check_only" = "0" ]; then
    fa3_note="FA3=$fa3, FA3 jobs=$FLASH_ATTN_MAX_JOBS x $FLASH_ATTN_NVCC_THREADS threads"; [ "$variant" = "cuda13-aarch64" ] && fa3_note="FA3=n/a (FA4-only image)"
    echo ">> building $tag from Dockerfile.$variant ($fa3_note, opencv jobs=$OPENCV_BUILD_JOBS, revision=$revision)"
    # Never replace an existing build tag during release preparation.
    # Serialize cooperating release writers on this daemon/tag through the
    # absence check and build. All writers must share this lock directory.
    lock_file=$(python3 docker/release_image.py lock-path "$tag")
    exec {tag_lock}>"$lock_file"
    flock "$tag_lock"
    python3 docker/release_image.py assert-absent "$tag"
    resolve_base_image
    base_digest=$(docker image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "$base_root_id")
    [[ "$base_digest" == *@sha256:* ]] || {
        echo "Release builds require a base with a registry digest; pull the base image first" >&2
        exit 2
    }
    printf '%s\n' "$base_digest" > "$report_dir/base-image.ref"
    build_args+=(--build-arg "ARTIFIXER_BASE_IMAGE=$base_digest")
    docker build --progress=plain -f "Dockerfile.$variant" "${build_args[@]}" \
        --label "org.opencontainers.image.revision=$revision" \
        --label "org.opencontainers.image.created=$created" \
        --label "com.nvidia.artifixer.source_dirty=$source_dirty" \
        --label "org.opencontainers.image.source=$source_url" \
        --label "com.nvidia.artifixer.variant=$variant" \
        --iidfile "$report_dir/build-image.id" -t "$tag" .
    source_id=$(cat "$report_dir/build-image.id")
    flock -u "$tag_lock"
fi

# Always scan the immutable image ID with no bind mounts, inherited entrypoint,
# host codec or GPU. A build-image pass covers the final filesystem only.
python3 docker/release_image.py verify "$source_id" --build-image --platform "$expected_platform" --base-image "$base_root_id" \
    --report "$report_dir/build-image.json"

if [ "$flatten" = "1" ]; then
    echo ">> flattening $tag -> $tag-flat"
    docker/flatten_release_image.sh "$source_id" "$tag-flat" --platform "$expected_platform" --base-image "$base_root_id" --report "$report_dir/release-image.json"
    echo ">> flattened artifact verified locally: $tag-flat"
else
    echo ">> build filesystem verified; registry distribution still requires --flatten"
fi

echo ">> receipts: $report_dir (GPU qualification and release approval are separate)"
