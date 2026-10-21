#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Collect the source code of every OSS component added on top of the NGC base
# image, and upgraded components for release alongside the container images.
# This collects technical evidence; it does not establish release clearance.
#
# Usage: docker/collect_container_sources.sh <base-image> <final-image> <output-dir> [linux/amd64|linux/arm64]
# Produces <output-dir>/artifixer-container-sources.tar plus MANIFEST.txt and,
# if anything could not be fetched automatically, NEEDS-ATTENTION.txt (unresolved
# collection entries; an empty file is not release approval).
set -euo pipefail

BASE=${1:?usage: collect_container_sources.sh <base-image> <final-image> <output-dir>}
FINAL=${2:?usage: collect_container_sources.sh <base-image> <final-image> <output-dir>}
OUT=${3:?usage: collect_container_sources.sh <base-image> <final-image> <output-dir>}
PLATFORM=${4:-}
# Never combine a previous attempt's downloads or success state with this one.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
python3 - "$SCRIPT_DIR" "$BASE" "$FINAL" "$PLATFORM" > "$OUT/image-identities.txt" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from release_image import inspect_image, platform, platform_matches
final = inspect_image(sys.argv[3], expected_platform=sys.argv[4] or None)
selected = sys.argv[4] or platform(final)
base = inspect_image(sys.argv[2], expected_platform=selected)
if not platform_matches(selected, platform(base)) or not platform_matches(selected, platform(final)):
    raise ValueError(f"Both images must provide {selected}")
print(base["Id"])
print(final["Id"])
print(selected)
PYEOF
mapfile -t IDENTITIES < "$OUT/image-identities.txt"
BASE=${IDENTITIES[0]}
FINAL=${IDENTITIES[1]}
PLATFORM=${IDENTITIES[2]}
OUT=$(cd "$OUT" && pwd)  # docker -v needs an absolute path

inventory() {
    docker run --rm --platform "$PLATFORM" --network none --read-only --user 0 \
        --env NVIDIA_VISIBLE_DEVICES=void --env PYTHONDONTWRITEBYTECODE=1 \
        --entrypoint python -i "$1" - < "$SCRIPT_DIR/image_inventory.py"
}

inventory "$BASE" > "$OUT/base.json"
inventory "$FINAL" > "$OUT/final.json"

# Include upgrades/downgrades, not just names absent from the base image.
python3 "$SCRIPT_DIR/container_inventory.py" "$OUT/base.json" "$OUT/final.json" pip > "$OUT/pip.added"
python3 "$SCRIPT_DIR/container_inventory.py" "$OUT/base.json" "$OUT/final.json" apt > "$OUT/apt.added"

echo "pip components: $(wc -l < "$OUT/pip.added"), apt components: $(wc -l < "$OUT/apt.added")"

# Fetch logic is written to helper files and executed inside the final image
# (its package versions are authoritative); no shell-quote nesting.
cat > "$OUT/fetch_pip.py" <<'PYEOF'
# Fetch sdists via the PyPI JSON API: pip download would try to build each
# sdist's metadata, which fails for build backends not installed in the image.
import json
import os
import re
import shutil
import subprocess
import urllib.request


def note(msg):
    with open("/out/NEEDS-ATTENTION.txt", "a") as fh:
        fh.write(msg + "\n")


def info(msg):
    with open("/out/SOURCE-NOTES.txt", "a") as fh:
        fh.write(msg + "\n")


def normalized_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


# NVIDIA first-party wheels distributed under NVIDIA proprietary licenses:
# not third-party OSS, so no source-release obligation. Documented, not fetched.
PROPRIETARY_PREFIXES = (
    "nvidia-cublas", "nvidia-cuda-", "nvidia-cudnn", "nvidia-cufft", "nvidia-cufile",
    "nvidia-curand", "nvidia-cusolver", "nvidia-cusparse", "nvidia-nvjitlink",
    "nvidia-nvshmem", "nvidia-nvtx", "cuda-toolkit",
    # The CUTLASS Python DSL wheels ship under the NVIDIA proprietary license
    # (their metadata says "Other/Proprietary License"); they are not the
    # BSD-licensed C++ CUTLASS repository.
    "nvidia-cutlass-dsl",
)

# Wheel-only PyPI releases whose source lives in a git repository: the exact
# tag/commit for the installed version (every ref here must be immutable; a
# floating branch would not reproduce the shipped binary). Packages installed
# straight from git carry their commit in direct_url.json and are resolved
# from that instead (see git_ref_from_install below).
GIT_SOURCES = {
    "flash_attn_3": ("Dao-AILab/flash-attention", "77aacb68d194ba9af1010eda5eac3e7c0df8e6f6"),
    "triton": ("triton-lang/triton", "v{version}"),
    # OpenUSD zero-pads the minor in its tags (26.5 -> v26.05).
    "usd-core": ("PixarAnimationStudios/OpenUSD", "v{version}"),
    "nvidia-nccl-cu12": ("NVIDIA/nccl", "v{version}-1"),
    "cuda-bindings": ("NVIDIA/cuda-python", "v{version}"),
    "cuda-pathfinder": ("NVIDIA/cuda-python", "cuda-pathfinder-v{version}"),
}
# Match the inventory's normalized names, including overrides declared with
# underscores/dots and legacy freeze-format collection inputs.
GIT_SOURCES = {normalized_name(name): source for name, source in GIT_SOURCES.items()}

# Our own code, distributed in source form in the released repository itself.
FIRST_PARTY = {"threedgrut"}


def fetch_github(name, version, repo, ref):
    ref_version = version
    if name == "usd-core":
        parts = version.split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"Unknown OpenUSD release version: {version}")
        ref_version = f"{int(parts[0]):02d}.{int(parts[1]):02d}"
    ref = ref.format(version=ref_version)
    url = f"https://codeload.github.com/{repo}/tar.gz/{ref}"
    dest = f"/out/sources/pip/{name}-{version}-{ref.replace('/', '_')}.tar.gz"
    urllib.request.urlretrieve(url, dest)
    info(f"pip {name}=={version}: source from github.com/{repo} @ {ref}")


def git_ref_from_install(name):
    """(repo, commit) recorded by pip for a package installed from a git URL, else None."""
    import importlib.metadata

    try:
        raw = importlib.metadata.distribution(name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        raw = None
    if not raw:
        return None
    data = json.loads(raw)
    commit = data.get("vcs_info", {}).get("commit_id")
    match = re.match(r"https://github\.com/([^/]+/[^/.]+)(?:\.git)?/?$", data.get("url", ""))
    if commit and match:
        return match.group(1), commit
    return None


for line in open("/out/pip.added"):
    line = line.strip()
    if not line or "==" not in line:
        continue
    name, version = line.split("==", 1)
    name = normalized_name(name)
    if name in FIRST_PARTY:
        info(f"pip {name}=={version}: NVIDIA first-party, source is the released repository itself")
        continue
    if name.startswith(PROPRIETARY_PREFIXES):
        info(f"pip {name}=={version}: NVIDIA proprietary wheel (not third-party OSS, no source obligation)")
        continue
    try:
        if name == "moge" and version == "2.0.0+artifixer.1":
            # Include the exact downstream source, upstream MIT license and
            # packaging patch retained by install_moge.sh, not an upstream-only
            # archive that cannot reproduce the installed wheel metadata.
            for suffix in ("tar.gz", "patch"):
                source = f"/opt/artifixer/sources/moge-{version}.{suffix}"
                shutil.copyfile(source, f"/out/sources/pip/moge-{version}.{suffix}")
            info(f"pip moge=={version}: exact patched source and patch from the image; upstream revision in ARTIFIXER-SOURCE.txt")
            continue
        installed_from_git = git_ref_from_install(name)
        if installed_from_git is not None:
            fetch_github(name, version, *installed_from_git)
            continue
        if name in GIT_SOURCES:
            fetch_github(name, version, *GIT_SOURCES[name])
            continue
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as r:
            data = json.load(r)
        sdists = [u for u in data["urls"] if u["packagetype"] == "sdist"]
        if not sdists:
            raise LookupError("no sdist in release")
        url = sdists[0]["url"]
        urllib.request.urlretrieve(url, os.path.join("/out/sources/pip", url.rsplit("/", 1)[-1]))
    except Exception as exc:
        home = ""
        try:
            out = subprocess.run(["pip", "show", name], capture_output=True, text=True).stdout
            home = next((l.split(": ", 1)[1] for l in out.splitlines() if l.startswith("Home-page: ")), "")
        except Exception:
            pass
        note(f"pip {name}=={version} sdist unavailable ({exc}); source: {home or 'UNKNOWN - resolve manually'}")
PYEOF

cat > "$OUT/fetch_apt.sh" <<'SHEOF'
#!/bin/bash
# Enable deb-src entries (disabled by default in the base image).
sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/*.sources 2>/dev/null || true
sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list 2>/dev/null || true
apt-get update -qq >/dev/null 2>&1 || true
cd /out/sources/apt
while IFS='=' read -r name version; do
    [ -z "$name" ] && continue
    if ! apt-get source --download-only "${name}=${version}" >/dev/null 2>&1; then
        if ! apt-get source --download-only "$name" >/dev/null 2>&1; then
            echo "apt ${name}=${version} source fetch failed - resolve manually" >> /out/NEEDS-ATTENTION.txt
        else
            echo "apt ${name}: exact version ${version} unavailable, fetched current source instead" >> /out/NEEDS-ATTENTION.txt
        fi
    fi
done < /out/apt.added
SHEOF

docker run --rm --platform "$PLATFORM" --network host -v "$OUT":/out "$FINAL" bash -c \
    'mkdir -p /out/sources/pip /out/sources/apt && : > /out/NEEDS-ATTENTION.txt && python3 /out/fetch_pip.py && bash /out/fetch_apt.sh'

{
    echo "ArtiFixer container source collection — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "base image: $BASE"
    echo "final image: $FINAL"
    echo "platform: $PLATFORM"
    echo
    echo "== pip components =="; cat "$OUT/pip.added"
    echo
    echo "== apt components =="; cat "$OUT/apt.added"
    echo
    echo "== source notes (first-party, proprietary, git-sourced) =="
    cat "$OUT/SOURCE-NOTES.txt" 2>/dev/null || true
} > "$OUT/MANIFEST.txt"

if [ -s "$OUT/NEEDS-ATTENTION.txt" ]; then
    echo "SOURCE COLLECTION INCOMPLETE — resolve the entries in $OUT/NEEDS-ATTENTION.txt:"
    cat "$OUT/NEEDS-ATTENTION.txt"
    exit 1
fi

# Package only complete collections. Hash the retrieved bytes and the archive.
(cd "$OUT" && find sources -type f -print0 | sort -z | xargs -0 -r sha256sum) > "$OUT/SHA256SUMS"
(cd "$OUT" && sha256sum MANIFEST.txt base.json final.json) >> "$OUT/SHA256SUMS"
tar -cf "$OUT/artifixer-container-sources.tar" -C "$OUT" sources MANIFEST.txt base.json final.json SHA256SUMS
(cd "$OUT" && sha256sum artifixer-container-sources.tar) > "$OUT/artifixer-container-sources.tar.sha256"
echo "wrote $OUT/artifixer-container-sources.tar (technical source collection; no publication performed)"
