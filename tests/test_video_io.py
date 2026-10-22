# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
import torch
from PIL import Image

from model_training.utils import video_io
from model_training.utils.video_io import (
    FFMPEG_HINT,
    _is_even_yuv420p,
    _yuv420p_to_rgb24,
    decode_video_frame_count,
    decode_video_to_frames,
    encode_frames_to_video,
    iter_video_frames,
    probe_video_frame_count,
    resolve_ffmpeg_exe,
    save_video,
)


def iter_video_frames_to_tmp(video: bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        Path(tmp.name).write_bytes(video)
        yield from iter_video_frames(tmp.name)


def test_resolve_ffmpeg_exe_fails_closed_with_the_setup_hint(monkeypatch, tmp_path):
    """The headline contract of the codec-free images: with no usable ffmpeg,
    video I/O refuses with guidance instead of a bare lookup error, and an
    explicit IMAGEIO_FFMPEG_EXE is validated (imageio-ffmpeg returns it as-is)."""
    # No host ffmpeg anywhere imageio-ffmpeg looks.
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    monkeypatch.setattr(
        video_io.imageio_ffmpeg, "get_ffmpeg_exe", lambda: (_ for _ in ()).throw(RuntimeError("no ffmpeg"))
    )
    with pytest.raises(RuntimeError, match="IMAGEIO_FFMPEG_EXE") as excinfo:
        resolve_ffmpeg_exe()
    assert FFMPEG_HINT in str(excinfo.value)

    # An explicit path that does not exist.
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(tmp_path / "missing-ffmpeg"))
    with pytest.raises(RuntimeError, match="is not an executable that exists"):
        resolve_ffmpeg_exe()

    # An explicit path that exists but is not executable.
    not_executable = tmp_path / "ffmpeg"
    not_executable.write_text("#!/bin/sh\n")
    not_executable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(not_executable))
    with pytest.raises(RuntimeError, match="is not an executable that exists"):
        resolve_ffmpeg_exe()

    # A bare name that is not on PATH.
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", "definitely-not-an-ffmpeg-binary")
    with pytest.raises(RuntimeError, match="is not an executable that exists"):
        resolve_ffmpeg_exe()


def test_resolve_ffmpeg_exe_accepts_an_explicit_executable(monkeypatch, tmp_path):
    exe = tmp_path / "ffmpeg"
    exe.write_text("#!/bin/sh\necho ffmpeg version test\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(exe))
    monkeypatch.setattr(video_io.imageio_ffmpeg, "get_ffmpeg_exe", lambda: os.environ["IMAGEIO_FFMPEG_EXE"])

    assert resolve_ffmpeg_exe() == str(exe)


@pytest.mark.requires_host_ffmpeg
def test_encode_frames_to_video_stream_round_trips_frame_count():
    frames = np.zeros((3, 6, 8, 3), dtype=np.float32)
    frames[0, :, :, 0] = 1.0
    frames[1, :, :, 1] = 1.0
    frames[2, :, :, 2] = 1.0

    video = encode_frames_to_video(frames, fps=5, crf=28)

    assert decode_video_frame_count(video) == 3
    assert probe_video_frame_count(video) == 3
    decoded = decode_video_to_frames(video)
    assert decoded.shape == (3, 6, 8, 3)
    assert decoded.dtype == np.uint8
    # Each frame keeps its dominant channel: catches channel-order mixups in
    # either direction of the round trip.
    for i in range(3):
        assert decoded[i, ..., i].mean() > 180
        assert decoded[i, ..., (i + 1) % 3].mean() < 60
        assert decoded[i, ..., (i + 2) % 3].mean() < 60


@pytest.mark.requires_host_ffmpeg
def test_decodes_videos_encoded_by_the_previous_pyav_implementation():
    # All existing HDF5 render/opacity videos were written by the PyAV-based
    # encoder this module replaced; decoding them must keep working.
    video = (Path(__file__).parent / "data" / "pyav_encoded_rgb.mp4").read_bytes()

    assert probe_video_frame_count(video) == 3
    decoded = decode_video_to_frames(video)
    assert decoded.shape == (3, 12, 16, 3)
    for i in range(3):
        assert decoded[i, ..., i].mean() > 180
        assert decoded[i, ..., (i + 1) % 3].mean() < 60

    iterated = list(iter_video_frames_to_tmp(video))
    assert len(iterated) == 3
    np.testing.assert_array_equal(np.stack(iterated), decoded)


@pytest.mark.requires_host_ffmpeg
def test_save_video_accepts_tchw_tensors(tmp_path):
    frames = torch.zeros(2, 3, 6, 8)
    frames[0, 0] = 1.0
    frames[1, 1] = 1.0
    output = tmp_path / "tensor.mp4"

    save_video(frames, output, fps=4, crf=28)

    assert output.stat().st_size > 0
    assert decode_video_frame_count(output.read_bytes()) == 2


@pytest.mark.requires_host_ffmpeg
def test_save_video_accepts_path_iterables_and_odd_dimensions(tmp_path):
    paths = []
    for idx, color in enumerate(((255, 0, 0), (0, 255, 0))):
        path = tmp_path / f"frame_{idx}.png"
        Image.new("RGB", (7, 5), color).save(path)
        paths.append(path)

    output = tmp_path / "paths.mp4"
    save_video(paths, output, fps=4, crf=28)

    assert output.stat().st_size > 0
    assert decode_video_frame_count(output.read_bytes()) == 2


def test_save_video_rejects_empty_iterables(tmp_path):
    with pytest.raises(ValueError, match="zero frames"):
        save_video([], tmp_path / "empty.mp4")


@pytest.mark.requires_host_ffmpeg
def test_failed_encode_preserves_previous_video_and_cleans_temporary_file(tmp_path):
    path = tmp_path / "clip.mp4"
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    save_video([frame, frame], path)
    previous = path.read_bytes()

    def interrupted_frames():
        yield frame
        raise RuntimeError("input interrupted")

    with pytest.raises(RuntimeError, match="input interrupted"):
        save_video(interrupted_frames(), path)
    assert path.read_bytes() == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]
    assert video_io.video_file_complete(path, 2)


@pytest.mark.requires_host_ffmpeg
def test_missing_codec_cannot_publish_a_video(tmp_path):
    path = tmp_path / "clip.mp4"
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    with pytest.raises((RuntimeError, OSError)):
        video_io._write_video(frames, path, fps=15, codec="not_a_real_codec", crf=23, pix_fmt="yuv420p")
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def _bt601_limited_rgb(y: int, u: int, v: int) -> tuple[int, int, int]:
    """Float BT.601 limited-range equations, independent of the integer kernel under test."""
    c, d, e = int(y) - 16, int(u) - 128, int(v) - 128
    channels = (1.164 * c + 1.596 * e, 1.164 * c - 0.392 * d - 0.813 * e, 1.164 * c + 2.017 * d)
    return tuple(int(min(max(round(channel), 0), 255)) for channel in channels)


def _bt601_reference(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = y.shape
    return np.array(
        [
            [_bt601_limited_rgb(y[i, j], u[i // 2, j // 2], v[i // 2, j // 2]) for j in range(width)]
            for i in range(height)
        ]
    )


def _planar_yuv420p(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> bytes:
    """One frame as ffmpeg's rawvideo yuv420p: the Y plane, then U, then V, each row-major."""
    return np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).astype(np.uint8).tobytes()


def _swscale_rgb24(path, input_params=None) -> np.ndarray:
    """ffmpeg's own conversion of ``path`` to rgb24: the swscale reference the kernel replaces."""
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", input_params=input_params)
    try:
        width, height = next(reader)["size"]
        return np.stack([np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3) for frame in reader])
    finally:
        reader.close()


def test_yuv420p_conversion_matches_the_bt601_limited_range_equations():
    height, width = 6, 8
    rng = np.random.default_rng(0)
    y = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    u = rng.integers(0, 256, size=(height // 2, width // 2), dtype=np.uint8)
    v = rng.integers(0, 256, size=(height // 2, width // 2), dtype=np.uint8)
    frame = _planar_yuv420p(y, u, v)

    rgb = _yuv420p_to_rgb24(frame, height, width)

    assert rgb.shape == (height, width, 3) and rgb.dtype == np.uint8
    assert np.abs(rgb.astype(int) - _bt601_reference(y, u, v)).max() <= 1
    scratch = np.full((2, height // 2, 2, width // 2, 2), -1, dtype=np.int32)
    np.testing.assert_array_equal(_yuv420p_to_rgb24(frame, height, width, scratch), rgb)


@pytest.mark.parametrize(
    "yuv, rgb",
    [((16, 128, 128), (0, 0, 0)), ((235, 128, 128), (255, 255, 255)), ((81, 90, 240), (255, 0, 0))],
)
def test_yuv420p_conversion_hits_the_limited_range_anchors_exactly(yuv, rgb):
    y, u, v = yuv
    frame = _planar_yuv420p(np.full((2, 2), y), np.full((1, 1), u), np.full((1, 1), v))
    assert _yuv420p_to_rgb24(frame, 2, 2).tolist() == [[list(rgb)] * 2] * 2


@pytest.mark.parametrize(
    "pix_fmt, size, expected",
    [
        ("yuv420p", (960, 540), True),
        ("yuv420p(tv, bt709, progressive)", (16, 12), True),
        ("yuvj420p", (16, 12), False),
        ("yuv444p(progressive)", (7, 5), False),
        ("yuv420p", (7, 5), False),
    ],
)
def test_native_yuv420p_gate_reads_the_stream_tags_imageio_parses(pix_fmt, size, expected):
    assert _is_even_yuv420p({"pix_fmt": pix_fmt, "size": size}) is expected


@pytest.mark.requires_host_ffmpeg
def test_kernel_reads_the_plane_layout_ffmpeg_emits(tmp_path):
    """Independent check of the planar layout the kernel assumes: ffmpeg reads the same raw bytes
    and converts them itself, so a misplaced or swapped plane shows up as a scrambled frame (tens
    of LSB), not a rounding difference. Flat chroma keeps swscale's chroma filter out of it."""
    height, width = 6, 8
    y = np.random.default_rng(1).integers(16, 236, size=(height, width), dtype=np.uint8)
    u = np.full((height // 2, width // 2), 90, dtype=np.uint8)
    v = np.full((height // 2, width // 2), 240, dtype=np.uint8)
    frame = _planar_yuv420p(y, u, v)
    raw = tmp_path / "frame.yuv"
    raw.write_bytes(frame)

    ours = _yuv420p_to_rgb24(frame, height, width)
    swscale = _swscale_rgb24(raw, ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-video_size", f"{width}x{height}"])

    assert swscale.shape == (1, height, width, 3)
    assert np.abs(ours.astype(int) - swscale[0].astype(int)).max() <= 3
    assert ours[..., 0].mean() > ours[..., 2].mean() + 100


@pytest.mark.requires_host_ffmpeg
def test_decoded_frames_agree_with_swscale_on_a_real_clip(tmp_path):
    """Gross agreement with swscale's converter (a wrong matrix, range or plane order is
    tens of LSB off); the tolerance covers swscale's own architecture spread of 3 LSB."""
    frames = np.zeros((2, 6, 8, 3), dtype=np.uint8)
    frames[0] = np.linspace(0, 255, 8, dtype=np.uint8)[None, :, None]  # grey ramp: chroma is flat
    frames[1, :, :4] = (200, 40, 40)
    frames[1, :, 4:] = (40, 40, 200)
    video = encode_frames_to_video(frames, fps=5, crf=18)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(video)

    ours = decode_video_to_frames(video)
    swscale = _swscale_rgb24(clip)

    assert ours.shape == swscale.shape == frames.shape
    assert np.abs(ours.astype(int) - swscale.astype(int)).max() <= 5


@pytest.mark.requires_host_ffmpeg
def test_non_yuv420p_streams_fall_back_to_swscale(tmp_path, monkeypatch):
    """Odd dimensions make the writer pick yuv444p, which the integer kernel does not
    handle; those frames must still decode, through ffmpeg's converter."""
    frames = np.zeros((2, 5, 7, 3), dtype=np.uint8)
    frames[1] = 255
    video = encode_frames_to_video(frames, fps=5, crf=18)
    clip = tmp_path / "odd.mp4"
    clip.write_bytes(video)
    reader = imageio_ffmpeg.read_frames(str(clip))
    stream_pix_fmt = next(reader)["pix_fmt"]
    reader.close()
    assert stream_pix_fmt.split("(")[0] == "yuv444p"

    def kernel_must_not_run(*args, **kwargs):
        raise AssertionError("the integer kernel was handed a yuv444p stream")

    monkeypatch.setattr(video_io, "_yuv420p_to_rgb24", kernel_must_not_run)

    decoded = decode_video_to_frames(video)

    assert decoded.shape == (2, 5, 7, 3)
    assert decoded[0].max() < 8 and decoded[1].min() > 247
    assert decode_video_frame_count(video) == 2
