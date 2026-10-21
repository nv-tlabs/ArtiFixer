# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video encode/decode helpers backed by an external ffmpeg executable.

The released ArtiFixer containers intentionally ship no ffmpeg codecs: no
ffmpeg binary, no libav* libraries, no PyAV, and no opencv/imageio wheels with
bundled ffmpeg. All video I/O in this repository funnels through this module,
which locates an ffmpeg executable provided by the host at runtime — via the
``IMAGEIO_FFMPEG_EXE`` environment variable, a conda installation, or ``ffmpeg``
on ``PATH``.
"""

import io
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

import imageio_ffmpeg
import numpy as np

logger = logging.getLogger(__name__)
_logged_ffmpeg: set[str] = set()

FrameSource = Any

FFMPEG_HINT = (
    "ArtiFixer images do not bundle ffmpeg. Provide an ffmpeg build that includes the codecs "
    "you need (libx264 for the default outputs) from the host system: set IMAGEIO_FFMPEG_EXE "
    "to an ffmpeg executable, or put `ffmpeg` on PATH (for example `apt-get install ffmpeg`, "
    "`conda install -c conda-forge ffmpeg`, or a mounted static build)."
)


def resolve_ffmpeg_exe() -> str:
    """Return the external ffmpeg executable, failing with setup guidance if none is usable.

    The encode/decode helpers call this before handing work to imageio-ffmpeg
    (which resolves the same way) so a missing ffmpeg fails with the setup
    guidance instead of a bare lookup error.
    """
    explicit = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if explicit:
        # imageio-ffmpeg returns IMAGEIO_FFMPEG_EXE unvalidated; check it here so a
        # missing or non-executable path fails with the setup hint, not deep in a decode.
        is_bare_name = explicit == os.path.basename(explicit)
        resolvable = (
            shutil.which(explicit) if is_bare_name else (Path(explicit).is_file() and os.access(explicit, os.X_OK))
        )
        if not resolvable:
            raise RuntimeError(f"IMAGEIO_FFMPEG_EXE={explicit} is not an executable that exists. {FFMPEG_HINT}")
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as exc:
        raise RuntimeError(f"No usable ffmpeg executable found. {FFMPEG_HINT}") from exc
    if exe not in _logged_ffmpeg:
        # Decoded pixel values can differ by up to ~3/255 between ffmpeg
        # builds (version and architecture change swscale rounding), so record
        # which build served this process for run-to-run comparability.
        _logged_ffmpeg.add(exe)
        try:
            version = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10).stdout.splitlines()[
                0
            ]
        except Exception:
            version = "unknown version"
        logger.info("Using external ffmpeg: %s (%s)", exe, version)
    return exe


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if np.issubdtype(frame.dtype, np.floating):
        frame = (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)
    elif frame.dtype != np.uint8:
        raise TypeError(f"Expected uint8 or float video frames, got {frame.dtype}")
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {frame.shape}")
    return np.ascontiguousarray(frame)


def _is_torch_tensor(value: object) -> bool:
    if type(value).__module__.split(".", 1)[0] != "torch":
        return False
    import torch

    return torch.is_tensor(value)


def _iter_frame_arrays(frames: FrameSource):
    if _is_torch_tensor(frames):
        if frames.ndim != 4:
            raise ValueError(f"Expected TCHW tensor, got shape {tuple(frames.shape)}")
        for frame in frames:
            yield _as_uint8_rgb(frame.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy())
        return

    if isinstance(frames, np.ndarray):
        for frame in frames:
            yield _as_uint8_rgb(frame)
        return

    # Generic iterable: per-frame ndarrays or image paths.
    from PIL import Image

    for item in frames:
        if isinstance(item, np.ndarray):
            yield _as_uint8_rgb(item)
        else:
            with Image.open(item) as image:
                yield _as_uint8_rgb(np.asarray(image.convert("RGB")))


def _encode_to_file(
    path: str,
    first: np.ndarray,
    frame_iter,
    fps: int,
    codec: str,
    crf: int,
    pix_fmt: str,
) -> None:
    resolve_ffmpeg_exe()
    height, width = first.shape[:2]
    writer = imageio_ffmpeg.write_frames(
        path,
        (width, height),
        pix_fmt_in="rgb24",
        pix_fmt_out=pix_fmt,
        fps=fps,
        codec=codec,
        quality=None,
        output_params=["-crf", str(crf)],
        macro_block_size=1,
    )
    writer.send(None)
    frame_count = 1
    try:
        writer.send(first.tobytes())
        for frame_array in frame_iter:
            frame_array = _as_uint8_rgb(frame_array)
            if frame_array.shape[:2] != (height, width):
                raise ValueError(f"Video frame size changed from {(height, width)} to {frame_array.shape[:2]}")
            writer.send(frame_array.tobytes())
            frame_count += 1
    finally:
        writer.close()
    # imageio-ffmpeg can log encoder failures without raising. Count the frames
    # actually readable before publishing the result.
    output = Path(path)
    if not output.exists() or not output.stat().st_size:
        raise RuntimeError(f"ffmpeg produced no output for {path} (codec={codec}). {FFMPEG_HINT}")
    if not video_file_complete(output, frame_count):
        raise RuntimeError(f"ffmpeg produced an incomplete video for {path} (expected {frame_count} frames).")


def video_file_complete(path: Path, expected_frames: int) -> bool:
    """Check decoded frame count without copying video bytes or pixels into Python."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    resolve_ffmpeg_exe()
    try:
        frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    except (OSError, RuntimeError):
        return False
    return frames == expected_frames


@contextmanager
def atomic_output_path(path: Path):
    """Publish a finished sibling file with one rename; preserve old output on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    with temporary.open("xb"):
        pass
    try:
        if path.exists():
            shutil.copymode(path, temporary)
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_video(
    frames: FrameSource,
    output: str | Path | BinaryIO,
    fps: int,
    codec: str,
    crf: int,
    pix_fmt: str,
) -> None:
    frame_iter = iter(_iter_frame_arrays(frames))
    try:
        first = next(frame_iter)
    except StopIteration as exc:
        raise ValueError("Cannot write a video with zero frames") from exc

    height, width = first.shape[:2]
    if pix_fmt == "yuv420p" and (width % 2 or height % 2):
        pix_fmt = "yuv444p"

    if isinstance(output, (str, Path)):
        path = Path(output)
        with atomic_output_path(path) as temporary:
            _encode_to_file(str(temporary), first, frame_iter, fps=fps, codec=codec, crf=crf, pix_fmt=pix_fmt)
        return

    # ffmpeg needs a seekable output for mp4; encode to a temporary file for stream targets.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        _encode_to_file(tmp.name, first, frame_iter, fps=fps, codec=codec, crf=crf, pix_fmt=pix_fmt)
        output.write(Path(tmp.name).read_bytes())


def save_video(frames: FrameSource, path: Path, fps: int = 15, crf: int = 23) -> None:
    _write_video(frames, path, fps=fps, codec="libx264", crf=crf, pix_fmt="yuv420p")


def encode_frames_to_video(
    frames: np.ndarray, codec: str = "libx264", crf: int = 23, pix_fmt: str = "yuv420p", fps: int = 30
) -> bytes:
    output = io.BytesIO()
    _write_video(frames, output, fps=fps, codec=codec, crf=crf, pix_fmt=pix_fmt)
    return output.getvalue()


_BT601_LIMITED_Y_GAIN = 298
_BT601_LIMITED_R_V = 409
_BT601_LIMITED_G_U = -100
_BT601_LIMITED_G_V = -208
_BT601_LIMITED_B_U = 516


def _yuv420p_to_rgb24(frame: bytes, height: int, width: int, scratch: np.ndarray | None = None) -> np.ndarray:
    """Deterministic BT.601 limited-range yuv420p -> rgb24 for one planar frame (Y, then U, then V).

    Integer numpy only, so the result is identical on every CPU and ffmpeg build; nearest-neighbour
    chroma, since a scaler filter would vary by architecture. Applied unconditionally, ignoring
    colorspace/color_range tags -- swscale's default for the untagged corpus this reads, wrong for
    tagged BT.709/full-range content (as the swscale path also was). ``scratch`` is two int32 planes
    of shape ``(2, height // 2, 2, width // 2, 2)`` that a caller decoding many frames can reuse.
    """
    chroma_height, chroma_width = height // 2, width // 2
    luma_samples, chroma_samples = height * width, chroma_height * chroma_width
    planes = np.frombuffer(frame, dtype=np.uint8)
    y = planes[:luma_samples].reshape(chroma_height, 2, chroma_width, 2)
    u = planes[luma_samples : luma_samples + chroma_samples]
    v = planes[luma_samples + chroma_samples : luma_samples + 2 * chroma_samples]
    u = u.reshape(chroma_height, 1, chroma_width, 1).astype(np.int32) - 128
    v = v.reshape(chroma_height, 1, chroma_width, 1).astype(np.int32) - 128
    if scratch is None:
        scratch = np.empty((2, chroma_height, 2, chroma_width, 2), dtype=np.int32)
    luma, acc = scratch
    np.multiply(y, _BT601_LIMITED_Y_GAIN, out=luma, dtype=np.int32)
    luma += 128 - 16 * _BT601_LIMITED_Y_GAIN
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    channels = rgb.reshape(chroma_height, 2, chroma_width, 2, 3)
    chroma_terms = (
        _BT601_LIMITED_R_V * v,
        _BT601_LIMITED_G_U * u + _BT601_LIMITED_G_V * v,
        _BT601_LIMITED_B_U * u,
    )
    for channel, chroma in enumerate(chroma_terms):
        np.add(luma, chroma, out=acc)
        acc >>= 8
        np.clip(acc, 0, 255, out=channels[..., channel], casting="unsafe")
    return rgb


def _is_even_yuv420p(meta: dict) -> bool:
    width, height = meta["size"]
    return meta["pix_fmt"].split("(")[0] == "yuv420p" and width % 2 == 0 and height % 2 == 0


def _start_decoder(path: str):
    """Start ffmpeg on ``path``; return the raw-frame reader (metadata already consumed) and the
    function turning one raw frame into an (H, W, 3) uint8 RGB array.

    yuv420p streams at even sizes -- everything this repository's encoder writes -- are read as
    their native planes (a byte-exact pass-through) and converted by ``_yuv420p_to_rgb24``; any
    other pixel format is converted to rgb24 by ffmpeg's swscale, whose output differs by
    architecture.
    """
    resolve_ffmpeg_exe()
    reader = imageio_ffmpeg.read_frames(path, pix_fmt="yuv420p", bits_per_pixel=12)
    meta = next(reader)
    width, height = meta["size"]
    if _is_even_yuv420p(meta):
        scratch = np.empty((2, height // 2, 2, width // 2, 2), dtype=np.int32)
        return reader, lambda frame: _yuv420p_to_rgb24(frame, height, width, scratch)
    reader.close()
    reader = imageio_ffmpeg.read_frames(path, pix_fmt="rgb24")
    width, height = next(reader)["size"]
    return reader, lambda frame: np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)


def iter_video_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Decode a video file to RGB uint8 frames, one ndarray at a time.

    Treat the yielded arrays as read-only (the swscale fallback yields views over
    the decoder's byte buffers); copy before writing to them. Closing the
    iterator early (a ``break`` or ``.close()``) terminates the ffmpeg process.
    """
    reader, to_rgb = _start_decoder(str(path))
    try:
        for frame in reader:
            yield to_rgb(frame)
    finally:
        reader.close()


def _select_frames(
    decoded_frames: Iterator[bytes], frame_indices: np.ndarray | list[int], to_rgb: Callable[[bytes], np.ndarray]
) -> np.ndarray:
    """Pick ``frame_indices`` (any order, repeats allowed) out of a decode
    stream, stopping after the highest requested frame; only the picked
    frames go through ``to_rgb``."""
    requested = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if requested.size == 0:
        raise ValueError("Cannot decode an empty frame index list")
    if np.any(requested < 0):
        raise IndexError(f"Frame indices must be non-negative, got {requested.tolist()}")

    positions_by_frame: dict[int, list[int]] = {}
    for output_pos, frame_index in enumerate(requested.tolist()):
        positions_by_frame.setdefault(int(frame_index), []).append(output_pos)

    frames: list[np.ndarray | None] = [None] * len(requested)
    remaining = len(positions_by_frame)
    max_requested = int(requested.max())
    for decoded_index, frame in enumerate(decoded_frames):
        positions = positions_by_frame.get(decoded_index)
        if positions is not None:
            decoded = to_rgb(frame)
            for output_pos in positions:
                frames[output_pos] = decoded
            remaining -= 1
            if remaining == 0:
                break

    if remaining != 0:
        missing = [index for index, positions in positions_by_frame.items() if frames[positions[0]] is None]
        raise IndexError(f"Requested frame indices are not present in video: {missing}")
    return np.stack(frames)


def decode_video_to_frames(video_bytes: bytes, frame_indices: np.ndarray | list[int] | None = None) -> np.ndarray:
    """Decode ``video_bytes`` to a (T, H, W, 3) uint8 array; with
    ``frame_indices`` only those frames, in the requested order."""
    # ffmpeg needs seekable input for mp4 (the index lives at the end of what
    # encode_frames_to_video writes), so the bytes go through a temp file.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        Path(tmp.name).write_bytes(video_bytes)
        reader, to_rgb = _start_decoder(tmp.name)
        try:
            if frame_indices is None:
                return np.stack([to_rgb(frame) for frame in reader])
            return _select_frames(reader, frame_indices, to_rgb)
        finally:
            reader.close()  # stops ffmpeg before the temp file goes away


def decode_video_frame_count(video_bytes: bytes) -> int:
    # Full decode with the frames piped into this process: errors on partially
    # delivered frames, which is what validation callers want. Both count
    # functions decode every frame; this one is the stricter, slower check.
    frame_count = 0
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        Path(tmp.name).write_bytes(video_bytes)
        reader, _ = _start_decoder(tmp.name)
        try:
            for _ in reader:
                frame_count += 1
        finally:
            reader.close()
    return frame_count


def probe_video_frame_count(video_bytes: bytes) -> int:
    """Count frames by letting ffmpeg decode the stream (no pixels piped back).

    This is a full ffmpeg-side decode — roughly 100 ms per 300-frame video,
    not a metadata read — but it counts what actually decodes instead of
    trusting container metadata.
    """
    resolve_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        Path(tmp.name).write_bytes(video_bytes)
        frame_count, _ = imageio_ffmpeg.count_frames_and_secs(tmp.name)
    return frame_count
