# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the final image filesystem for known FFmpeg and codec carriers.

Released ArtiFixer images must not distribute ffmpeg or ffmpeg-derived codecs
(legal requirement). This script is run as the last step of every Dockerfile and
scans the image for:

  * ffmpeg/ffprobe/ffplay executables (including the binary bundled by
    imageio-ffmpeg platform wheels),
  * libav*/libsw*/libpostproc shared libraries,
  * H.264/H.265 encoder libraries (libx264/libx265/libopenh264),
  * a PyAV installation,
  * an opencv build compiled with ffmpeg support,
  * a populated pip download cache (uninstalled wheels keep their bytes there).

Residual risk this scan cannot see: codec code statically linked inside some
other ELF. The Dockerfiles remove the known carriers (opencv rebuilt without
ffmpeg, DALI and torchaudio purged); re-check that list when bumping the base
image.

Video I/O still works at runtime through model_training/utils/video_io.py,
which uses an ffmpeg executable supplied by the host system.
"""

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Alternate roots are for diagnostic callers of scan_filesystem only. The
# release entrypoint requires the full root and no host FFmpeg/mounts.
SCAN_ROOTS = tuple(os.environ.get("FFMPEG_SCAN_ROOTS", "/").split(":"))
SKIP_DIRS = {"/proc", "/sys", "/dev"}

EXACT_BINARY_NAMES = {"ffmpeg", "ffprobe", "ffplay"}
LIBRARY_PREFIXES = (
    "libavcodec",
    "libavformat",
    "libavutil",
    "libavfilter",
    "libavdevice",
    "libswscale",
    "libswresample",
    "libpostproc",
    "libavresample",
    "libx264",
    "libx265",
    "libopenh264",
    # Video encoders/decoders PyAV-style wheels vendor alongside libav*.
    # (Audio codecs like libopus/libmp3lame are deliberately not listed: they
    # ship as ordinary system libraries in some bases and only ever carry
    # ffmpeg content next to libav* files the scan already catches.)
    "libsvtav1",
    "libdav1d",
    "libde265",
    "libkvazaar",
    "libxvid",
)
BUNDLED_BINARY_PREFIXES = ("ffmpeg-linux", "ffmpeg-macos", "ffmpeg-osx", "ffmpeg-win")


def is_native_code(path: str) -> bool:
    """ELF shared object/executable, or a static archive (`ar` magic; `.a`
    files are not ELF, so an ELF-only check would never flag them)."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
    except FileNotFoundError:
        return False
    return head[:4] == b"\x7fELF" or head == b"!<arch>\n"


def matches(name: str) -> bool:
    name = name.lower()
    if name in EXACT_BINARY_NAMES:
        return True
    if name.startswith(BUNDLED_BINARY_PREFIXES):
        return True
    if name.startswith(LIBRARY_PREFIXES) and (".so" in name or name.endswith(".a")):
        return True
    return False


def scan_filesystem() -> list[str]:
    offenders = []
    seen_roots = set()
    for root in SCAN_ROOTS:
        real_root = os.path.realpath(root)
        if not root or not os.path.isdir(real_root):
            raise ValueError(f"Scan root is not an existing directory: {root!r}")
        if real_root in seen_roots:
            continue
        seen_roots.add(real_root)

        def fail_on_walk_error(error: OSError) -> None:
            raise error

        for dirpath, dirnames, filenames in os.walk(real_root, onerror=fail_on_walk_error):
            dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in SKIP_DIRS]
            for filename in filenames:
                if matches(filename):
                    path = os.path.join(dirpath, filename)
                    if (
                        os.path.islink(path)
                        or is_native_code(path)
                        or (filename.lower() in EXACT_BINARY_NAMES and os.stat(path).st_mode & 0o111)
                    ):
                        offenders.append(path)
                elif filename.endswith(".whl"):
                    # A removed package can survive as an installable wheel outside pip's cache.
                    path = os.path.join(dirpath, filename)
                    with zipfile.ZipFile(path) as wheel:
                        for member in wheel.namelist():
                            if matches(member.rsplit("/", 1)[-1]):
                                offenders.append(f"{path}!{member}")
    return offenders


def cache_directories() -> set[Path]:
    import pwd

    caches = {Path(entry.pw_dir) / ".cache/pip" for entry in pwd.getpwall() if entry.pw_dir.startswith("/")}
    for variable in ("PIP_CACHE_DIR", "XDG_CACHE_HOME"):
        if value := os.environ.get(variable):
            caches.add(Path(value) / "pip" if variable == "XDG_CACHE_HOME" else Path(value))
    return caches | {Path("/var/cache/apt/archives")}


def check_image() -> dict:
    failures = []
    if SCAN_ROOTS != ("/",):
        failures.append("Release checks require FFMPEG_SCAN_ROOTS=/; partial scans cannot qualify an image.")
    if os.environ.get("IMAGEIO_FFMPEG_EXE"):
        failures.append("IMAGEIO_FFMPEG_EXE must be unset when checking a distributable image.")
    try:
        offenders = scan_filesystem()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        offenders = []
        failures.append(f"Filesystem scan incomplete: {exc}")
    if offenders:
        failures.append("Known codec files found: " + ", ".join(sorted(offenders)))

    def walk_error(error):
        raise error

    try:
        for cache in sorted(cache_directories()):
            if cache.is_dir() and any(
                files for _, _, files in os.walk(cache, onerror=walk_error) if any(name != "lock" for name in files)
            ):
                failures.append(f"Package download cache is not empty: {cache}")
    except OSError as exc:
        failures.append(f"Cache scan incomplete: {exc}")

    for module in ("av", "torchaudio", "nvidia.dali"):
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            spec = None  # e.g. the nvidia namespace is absent
        except (ImportError, ValueError) as exc:
            failures.append(f"Cannot inspect known codec carrier {module}: {exc}")
            continue
        if spec is not None:
            failures.append(f"Known codec carrier is installed: {module}")

    if shutil.which("ffmpeg") is not None:
        failures.append("ffmpeg is available on PATH.")
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    except ImportError as exc:
        failures.append(f"imageio_ffmpeg is unavailable; install its codec-free sdist: {exc}")
    else:
        binaries = Path(imageio_ffmpeg.__file__).parent / "binaries"
        bundled = sorted(p.name for p in binaries.glob("ffmpeg*"))
        if bundled:
            failures.append(f"imageio_ffmpeg bundles FFmpeg binaries: {bundled}")
        try:
            resolved = imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError:
            pass
        else:
            failures.append(f"imageio_ffmpeg unexpectedly resolved FFmpeg: {resolved}")

    opencv = None
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        failures.append(f"cv2 is unavailable; expected OpenCV built with WITH_FFMPEG=OFF: {exc}")
    else:
        lines = [line.strip() for line in cv2.getBuildInformation().splitlines() if line.strip().startswith("FFMPEG:")]
        opencv = {"version": cv2.__version__, "ffmpeg_build_lines": lines}
        # OpenCV can omit this line entirely when the backend was disabled.
        if any(not line.endswith("NO") for line in lines):
            failures.append(f"OpenCV was built with FFmpeg support: {lines}")

    return {
        "status": "fail" if failures else "pass",
        "scope": "final filesystem, known codec filenames/carriers, wheel members and package caches",
        "scan_roots": list(SCAN_ROOTS),
        "limitations": [
            "not an image-layer scan",
            "not an exhaustive static-link or arbitrary-archive analysis",
            "not legal clearance or GPU qualification",
        ],
        "opencv": opencv,
        "failures": failures,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="Print a machine-readable check result")
    args = parser.parse_args(argv)
    report = check_image()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    elif report["failures"]:
        print("FFMPEG IMAGE CHECK FAILED", file=sys.stderr)
        for failure in report["failures"]:
            print(f"- {failure}", file=sys.stderr)
    else:
        print("FFmpeg image check passed: no known codec carriers found in the final filesystem")
    return int(report["status"] != "pass")


if __name__ == "__main__":
    sys.exit(main())
