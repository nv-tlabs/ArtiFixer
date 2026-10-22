# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for the compliance gate's matcher (tests/assert_no_ffmpeg_in_image.py).

The gate is a build step, so a matcher gap only shows up when someone builds a
release image; the `.a` archive case slipped past an earlier ELF-only check
that way. These pin the filename and file-header rules directly.
"""

import importlib.util
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

GATE = Path(__file__).with_name("assert_no_ffmpeg_in_image.py")


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("assert_no_ffmpeg_in_image", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ELF = b"\x7fELF" + b"\0" * 12
AR = b"!<arch>\n" + b"x" * 8


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ffmpeg", True),
        ("ffprobe", True),
        ("ffmpeg-linux-x86_64-v7.0.2", True),  # imageio-ffmpeg's bundled binary
        ("libavcodec.so.58", True),
        ("libavcodec.so", True),
        ("libswscale.so.7", True),
        ("libx264.so.164", True),
        ("libSvtAv1Enc.so.1", True),  # PyAV vendors mixed-case names
        ("libavcodec.a", True),  # static archive
        ("libavutil.txt", False),  # name prefix alone is not enough
        ("libavcodec.h", False),
        ("libc.so.6", False),
        ("libopus.so.0", False),  # audio codecs are deliberately not matched
        ("README.md", False),
    ],
)
def test_matches_codec_names_only(gate, name, expected):
    assert gate.matches(name) is expected


def test_is_native_code_accepts_elf_and_ar_archives_only(gate, tmp_path):
    (tmp_path / "elf.so").write_bytes(ELF)
    (tmp_path / "static.a").write_bytes(AR)
    (tmp_path / "text.so").write_text("not a binary")
    (tmp_path / "empty.a").write_bytes(b"")

    assert gate.is_native_code(str(tmp_path / "elf.so"))
    assert gate.is_native_code(str(tmp_path / "static.a"))
    assert not gate.is_native_code(str(tmp_path / "text.so"))
    assert not gate.is_native_code(str(tmp_path / "empty.a"))
    assert not gate.is_native_code(str(tmp_path / "does-not-exist"))


def test_scan_flags_codec_binaries_archives_and_dangling_symlinks(gate, tmp_path, monkeypatch):
    (tmp_path / "libavcodec.so.60").write_bytes(ELF)  # shared object
    (tmp_path / "libavcodec.a").write_bytes(AR)  # static archive (the case that slipped past ELF-only)
    (tmp_path / "libavformat.so.txt").write_text("text")  # matching name, not native code
    (tmp_path / "libjpeg.a").write_bytes(AR)  # native code, not a codec name
    os.symlink(tmp_path / "gone" / "libavutil.so.58", tmp_path / "libavutil.so.58")  # dangling symlink
    (tmp_path / "ffmpeg").write_bytes(ELF)

    monkeypatch.setattr(gate, "SCAN_ROOTS", (str(tmp_path),))
    flagged = sorted(os.path.basename(path) for path in gate.scan_filesystem())

    assert flagged == ["ffmpeg", "libavcodec.a", "libavcodec.so.60", "libavutil.so.58"]


@pytest.mark.parametrize("root", ["", "missing"])
def test_invalid_scan_scope_is_not_a_clean_scan(gate, tmp_path, monkeypatch, root):
    monkeypatch.setattr(gate, "SCAN_ROOTS", (str(tmp_path / root) if root else "",))
    with pytest.raises(ValueError, match="Scan root"):
        gate.scan_filesystem()


def test_executable_wrapper_and_codec_wheel_are_detected(gate, tmp_path, monkeypatch):
    wrapper = tmp_path / "ffmpeg"
    wrapper.write_text("#!/bin/sh\nexit 0\n")
    wrapper.chmod(0o755)
    wheel = tmp_path / "cached-package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package.libs/libavcodec.so.60", ELF)
    monkeypatch.setattr(gate, "SCAN_ROOTS", (str(tmp_path),))
    assert sorted(gate.scan_filesystem()) == sorted([str(wrapper), f"{wheel}!package.libs/libavcodec.so.60"])


@pytest.fixture
def clean_image(gate, tmp_path, monkeypatch):
    def unavailable():
        raise RuntimeError("no FFmpeg")

    imageio = SimpleNamespace(__file__=str(tmp_path / "imageio_ffmpeg/__init__.py"), get_ffmpeg_exe=unavailable)
    cv2 = SimpleNamespace(__version__="fixture", getBuildInformation=lambda: "FFMPEG: NO")
    gate.importlib = SimpleNamespace(
        import_module=lambda name: {"imageio_ffmpeg": imageio, "cv2": cv2}[name],
        util=SimpleNamespace(find_spec=lambda name: None),
    )
    gate.shutil = SimpleNamespace(which=lambda name: None)
    monkeypatch.setattr(gate, "SCAN_ROOTS", ("/",))
    monkeypatch.setattr(gate, "scan_filesystem", lambda: [])
    monkeypatch.setattr(gate, "cache_directories", lambda: {tmp_path / "cache"})
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    return gate


def test_release_gate_requires_full_scope_and_no_codec_override(clean_image, monkeypatch):
    assert clean_image.check_image()["status"] == "pass"
    clean_image.SCAN_ROOTS = ("/empty",)
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", "/some/host/ffmpeg")
    report = clean_image.check_image()
    assert report["status"] == "fail"
    assert any("partial scans" in error for error in report["failures"])
    assert any("IMAGEIO_FFMPEG_EXE" in error for error in report["failures"])


def test_unreadable_scan_is_a_failure(clean_image):
    def denied():
        raise PermissionError("unreadable subtree")

    clean_image.scan_filesystem = denied
    assert "incomplete" in clean_image.check_image()["failures"][0]


def test_known_carrier_is_rejected_without_importing_it(clean_image):
    clean_image.importlib.util.find_spec = lambda name: object() if name == "av" else None
    assert "Known codec carrier is installed: av" in clean_image.check_image()["failures"]


def test_cached_wheel_bytes_are_not_hidden_by_uninstall(clean_image, tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/opaque-download").write_bytes(b"codec wheel")
    assert any("cache is not empty" in error for error in clean_image.check_image()["failures"])
