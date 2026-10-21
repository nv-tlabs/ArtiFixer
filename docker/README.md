<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Container build and release

ArtiFixer distributes an environment without bundled FFmpeg. Consumers supply
FFmpeg at runtime for video I/O. Build images may retain deleted codec bytes in
their NGC ancestor layers, so the distributable image must be flattened and
checked separately. The scripts here create local artifacts and receipts; they
never push images or establish release approval.

Use Docker Engine/CLI 28 or later (API 1.49+) for explicit platform inspection.
The verifier supports classic and containerd image stores. Install the Python
`zstandard` module if saved images use zstd-compressed layers; plain and gzip
layers use the standard library. Missing decoding support fails verification.

## Build a candidate

Initialize all submodules, including 3DGRUT's nested dependencies. Build on a
host with adequate memory and disk for the image, build intermediates, and an
additional exported/imported root filesystem. Prefer native ARM hardware for
the aarch64 build; QEMU CPU checks do not qualify a Grace/Blackwell GPU runtime.
Allow room for another temporary full layer in Docker's storage during
saved-payload verification, even though the verifier reads the archive as a stream.

```bash
git submodule update --init --recursive
FLASH_ATTN_MAX_JOBS=4 OPENCV_BUILD_JOBS=4 \
    docker/build_release_image.sh cuda12 \
    --tag artifixer:cuda12-candidate-001 \
    --flatten --report-dir /path/to/new-release-receipts
```

Other variants are `cuda13` and `cuda13-aarch64`. Use `--no-fa3` on an x86
build when Hopper FA3 is not needed. The ARM image never builds FA3. Thread
counts are positive integers; defaults are bounded by host resources, and ARM
cross-builds use four OpenCV jobs.

The wrapper uses `linux/amd64` for both x86 variants and `linux/arm64` for
the ARM variant. Before building, it pulls the base for that platform if the
base is missing locally or the cached tag selects a different platform. It
then selects the requested platform from a frozen root ID. `base-image.id`
records the selected platform ID; `base-image.root-id` records the addressable
root used for Docker operations. This also handles coexisting architectures in
containerd without replacing another cached variant.
The build uses that base's registry digest through `ARTIFIXER_BASE_IMAGE`;
`base-image.ref` records it. `--base-image` changes both the build and inventory
base. Release builds require a locally available registry digest.
`--check-only` does not pull: the image and its correct-platform base must
already be available locally.

All builds run `pip check`. NumPy and the QuACK attention dependency are pinned
across architectures; CUDA Python and its compiled bindings are installed as a
matching pair. Unused NGC applications that require an incompatible older NumPy
are removed together with their dependents. The removal script refuses to
remove any declared project dependency or its transitive dependencies.
Selected extras propagate through that closure, and package metadata is resolved
from the active installation even when the base exposes shadowed distributions.

MoGe uses the pinned upstream revision with a packaging-only patch to require
the source-built headless OpenCV distribution. Its model code is unchanged.
The downstream wheel is versioned `2.0.0+artifixer.1`; its modified source,
upstream license, revision and patch remain in `/opt/artifixer/sources` and are
included by the source collector.

The build receipt checks the **final filesystem only**. The release receipt
checks the final image's layer structure as well. Image IDs, platform, size,
layer content hashes, scanner hash, and a hash of the runtime configuration are
recorded. Uncommitted builds carry `com.nvidia.artifixer.source_dirty=1`; freeze
and review a clean source revision before publishing a release candidate.
For a fork, set `ARTIFIXER_SOURCE_URL` to its repository URL so the provenance
label identifies the source actually built.
Receipt directories and destination tags must be new, preserving earlier
attempts. Default receipt directories are unique temporary directories; use an
explicit durable destination for a release.

Every successful receipt references complete pip/dpkg inventories for its
immutable base and final images. For `release-image.json`, these live alongside
it as `release-image.base-inventory.json` and `release-image.final-inventory.json`;
the receipt records each relative path, SHA256 and image ID. Pip entries include
installed VCS commit IDs. Use `--base-image` if the original Dockerfile `FROM`
reference is available locally under a different name; it must identify that
same base, not an arbitrary comparison image.

`image_id` is Docker's addressable immutable root: a configuration ID in the
classic store, or a manifest/index ID in containerd. `platform_image_id` is the
selected platform's identity. Both appear in receipts and retained inventories;
Docker operations use the root plus an explicit platform because a containerd
child manifest ID is not always directly runnable. The saved-payload receipt
also records the authenticated descriptor chain and `configuration_digest`.
These digest types are distinct from the normalized runtime `config_sha256`.

Saved-byte verification hashes manifest/index/configuration blobs and validates
descriptor sizes and layer media types. It hashes distributed layer blobs,
decodes gzip/zstd when needed, and checks the uncompressed diffIDs while scanning
filesystem members. The full root export preserves the parent index proof;
every included filesystem layer must belong to an authenticated configuration.
If other Linux platforms are included, their layer bytes are also validated and
scanned, while runtime checks apply to the selected platform.

Release writers use an advisory file lock per Docker daemon and normalized
destination tag. All writers on a shared daemon must use this workflow and the
same `ARTIFIXER_RELEASE_LOCK_DIR` (default: `/tmp/artifixer-release-locks-UID`).
Docker provides no tag reservation through this lock: an external `docker tag`
command or a writer on another host that ignores the shared lock can still
replace a tag. Do not run such writers against release destinations. Builds
capture `--iidfile`; verification and flattening use that single immutable ID,
even if the build tag is subsequently changed.

## Check the application attention runtime

The images provide the environment; mount the source checkout whose attention
policy you want to validate. From that checkout's repository root:

```bash
docker run --rm --gpus all -v "$PWD:/workspace/artifixer:ro" \
    -w /workspace/artifixer artifixer:cuda12 python tests/test_flash_attn.py
```

Use the matching image tag for CUDA 13 or ARM. The copied
`/workspace/test_flash_attn.py` alone cannot validate the application because
`model_training` is not bundled in the image. FFmpeg and model downloads are
not needed for this reduced attention diagnostic. A passing diagnostic does
not qualify a full checkpoint or distributed inference.

## Check an existing image

No submodule checkout or source bind mount is needed for this check:

```bash
python3 docker/release_image.py verify artifixer:cuda12-candidate-001-flat \
    --platform linux/amd64 --base-image nvcr.io/nvidia/pytorch:25.01-py3 \
    --report /path/to/new-verification.json
```

To inspect an intermediate build without claiming its layers are suitable for
distribution, add `--build-image`. A clean final filesystem is insufficient
when old layers contain removed libraries. To flatten an existing image:

```bash
docker/flatten_release_image.sh artifixer:cuda12-candidate-001 \
    artifixer:cuda12-candidate-002-flat --base-image nvcr.io/nvidia/pytorch:25.01-py3 \
    --report /path/to/new-flatten-receipt.json
```

Use `--platform linux/arm64` when flattening the ARM variant of a multi-platform
source on x86. The build wrapper passes its variant platform through both
verification and flattening automatically.

Flattening resolves the source tag to an immutable ID, verifies it, exports its
filesystem, and reapplies the runtime metadata without executing its entrypoint.
Environment values, labels, command, entrypoint, working directory, user,
exposed ports, stop signal, shell, and platform are preserved and compared.
The destination tag is assigned only after the resulting image passes.

The image has one filesystem payload. Some Docker versions append a canonical
empty tar when applying metadata; only that exact content hash is allowed after
the root layer. Other inherited layers cause rejection. Images with declared
volumes, ONBUILD hooks, or health checks are refused because this workflow does
not yet preserve and inspect those contracts. Multiline configuration is also
refused instead of being changed silently.

## What the FFmpeg check establishes

For a flat release, the verifier also streams `docker image save`, checks the
configuration digest and each layer's actual byte hash against its diffID,
and scans the tar members without extracting them. This includes files beneath
Docker's runtime mounts at `/dev`, `/proc` and `/sys`. Known codec binaries,
libraries, wheel members and standard package caches fail that scan. The
runtime check alone cannot inspect bytes hidden by those mounts.

The verifier runs as root with a read-only filesystem, no network, no GPU, no
host bind mounts, and an explicit Python entrypoint. It scans `/`, checks known
codec filenames and wheel members, package caches, PyAV/DALI/torchaudio
presence, OpenCV build information, and imageio-ffmpeg's resolver. The resolver
must fail without host FFmpeg. Partial roots, missing/unreadable scan inputs,
and a preconfigured `IMAGEIO_FFMPEG_EXE` cannot qualify a release image.

The Dockerfiles build OpenCV with `WITH_FFMPEG=OFF`, install imageio-ffmpeg from
its codec-free source distribution, remove known bundled carriers, and execute
the gate. Do not replace the source install with the imageio-ffmpeg platform
wheel. Do not install consumer FFmpeg in an image that will be redistributed
under this contract.

This is a bounded technical check, not an exhaustive analysis of arbitrary
archives or statically linked code. Changes to base images and native
dependencies require review of new potential carriers. GPU operation,
vulnerability findings, source/license obligations, and release approval remain
separate checks; a JSON `pass` does not establish them.

## Consumer video I/O and GPU qualification

After the image check, test video I/O in a **separate consumer container** with
an explicitly supplied, compatible FFmpeg executable or environment. Follow
the root README's runtime setup. Check both missing-FFmpeg failure and an actual
encode/decode round trip with the supplied executable. Record its version; do
not conflate that container's host-mounted codecs with the distributed bytes.

Run the required tests in [CONTRIBUTING.md](../CONTRIBUTING.md) against the final
image and the matching code revision. Record actual GPU architecture, driver,
attention implementation, and CUDA results separately. ARM emulation, imports,
or the CPU filesystem gate cannot establish CUDA kernel correctness. The
container is an environment; the README mounts the application source, which
must be pinned alongside the image ID/digest.

## Notices and source archive

Use the actual base and final image identities for each variant:

```bash
python3 docker/generate_container_notices.py BASE_IMAGE FINAL_IMAGE \
    --output /path/to/THIRD-PARTY-NOTICES-CONTAINER.md \
    --previous THIRD-PARTY-NOTICES-CONTAINER.md
docker/collect_container_sources.sh BASE_IMAGE FINAL_IMAGE /path/to/new-sources
```

Both tools freeze tags to IDs and select the final image's platform by default.
For a specific variant of a
multi-platform image, pass `--platform linux/arm64` to the notice generator and
`linux/arm64` as the source collector's fourth argument. The base inventory,
final inventory, provenance, and source fetching use that same platform.

Package differences include version replacements
as well as new names and recorded Git source revisions. The source collector
retains complete `base.json` and `final.json` inventories before selecting its
`pip.added` and `apt.added` lists. Metadata does not capture every vendored or locally patched component;
review those separately against the actual build inputs.

Unknown license metadata stops notice generation without overwriting its output.
Missing or nonmatching source versions remain in `NEEDS-ATTENTION.txt`; no
source archive is packaged until collection succeeds. Sources and the manifest
receive checksums, as does the resulting archive. A successful collection does
not establish legal clearance or a complete image SBOM. Keep per-variant
inventories and resolve all applicable release reviews before publication.

## Replay release integration checks

The committed fixture exercises real export/import, metadata restoration,
deleted-layer rejection, runtime-masked codec rejection, preserved destination
tags and useful failure receipts. It deliberately substitutes tiny OpenCV and
imageio modules; native package/GPU qualification remains a separate check.

```bash
python3 tests/release_image_integration.py /path/to/new-fixture-evidence --enroot
python3 tests/source_collection_integration.py /path/to/new-source-evidence
```

Provide a local `python:3.12-slim` image (or `--base-image IMAGE`) and Docker.
`--enroot` additionally requires Enroot, `sqfstar` and `unsquashfs`; it compares
standard and streaming imports by every path, file SHA256, type, mode and link
target. Missing tools fail an explicitly requested integration run. Commands,
stdout/stderr, JSON receipts, inventories and the comparison are retained in
the output directory. Fixture images remain available for inspection.

The source fixture runs the complete shell collector against two real Docker
images with the same package version and different installed VCS revisions.
It also adds `flash_attn_3==3.0.0` without VCS metadata, matching the published
x86 wheels, and verifies that the normalized name selects the pinned Git
source override. It checks inventory production, replacement selection, fetched
revisions, and the complete archive/checksum outputs. Only network retrieval is
replaced by a deterministic transport fixture, so replay does not require those
synthetic Git commits to exist on a server.

`tests/test_release_build_wrapper.py` exercises the actual shell wrapper with
controlled Docker/verifier commands. It covers cold and warm caches, wrong
platforms, pull failures and check-only behavior without downloading an NGC
image or compiling the application.

For live storage-backend coverage, run `release_image_integration.py` against
separate classic and containerd daemons using an explicit `DOCKER_HOST`. Its
summary records the daemon ID, version and storage driver. The fixture accepts
`--platform linux/arm64 --base-image IMAGE` for ARM CPU execution (native or with
an already configured binfmt emulator). This does not qualify native GPU code.
After producing amd64 and arm64 flat fixtures in the same containerd daemon:

```bash
python3 tests/release_platform_integration.py /path/to/new-platform-evidence \
    --amd64-image AMD64_FLAT_ID --arm64-image ARM64_FLAT_ID
```

This creates a real multi-platform index, verifies both descriptor chains, and
executes the actual build/check-only/flatten paths with both platforms cached.
It verifies selected base IDs in the receipts and rejects any unexpected pull.
Docker and the verifier are real; the small Python/OpenCV fixture remains the
application stand-in. Use isolated test daemons; these fixtures need no change
to a shared daemon's storage configuration.

Docker references: [image import](https://docs.docker.com/reference/cli/docker/image/import/)
[container export and volume behavior](https://docs.docker.com/reference/cli/docker/container/export/),
[platform inspection](https://docs.docker.com/reference/cli/docker/image/inspect/),
and [containerd image storage](https://docs.docker.com/engine/storage/containerd/).
