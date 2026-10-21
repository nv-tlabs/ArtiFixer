# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from model_eval import inference
from model_training.utils.video_io import decode_video_to_frames, video_file_complete


@pytest.mark.requires_host_ffmpeg
@pytest.mark.parametrize("partial", [b"", b"truncated mp4"])
def test_finalize_repairs_incomplete_video_from_retained_pngs(tmp_path, monkeypatch, partial):
    scene = tmp_path / "scene"
    frames = scene / "video_frames/batch_0000/pred"
    frames.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (16, 16), (index * 30, 60, 90)).save(frames / f"{index:05d}.png")
    movie = scene / "videos/batch_0000_pred.mp4"
    movie.parent.mkdir()
    movie.write_bytes(partial)
    inference.finalize_video_outputs(tmp_path, fps=15)
    assert video_file_complete(movie, 3)
    before = movie.read_bytes()
    monkeypatch.setattr(inference, "save_video", lambda *a, **k: pytest.fail("complete video was re-encoded"))
    inference.finalize_video_outputs(tmp_path, fps=15)
    assert movie.read_bytes() == before


@pytest.mark.requires_host_ffmpeg
def test_finalize_ignores_abandoned_temporary_frames(tmp_path):
    scene = tmp_path / "scene"
    frames = scene / "video_frames/batch_0000/pred"
    frames.mkdir(parents=True)
    colors = [(240, 0, 0), (0, 240, 0), (0, 0, 240)]
    for index, color in enumerate(colors):
        Image.new("RGB", (16, 16), color).save(frames / f"{index:05d}.png")
    # A hard stop bypasses the writer's finally block, at any point before rename.
    (frames / ".00000.truncated.png").write_bytes(b"partial")
    Image.new("RGB", (16, 16), (255, 255, 255)).save(frames / ".00001.finished.png")

    inference.finalize_video_outputs(tmp_path, fps=15)
    movie = scene / "videos/batch_0000_pred.mp4"
    assert video_file_complete(movie, len(colors))
    decoded = decode_video_to_frames(movie.read_bytes())
    np.testing.assert_allclose(decoded.mean(axis=(1, 2)), colors, atol=8)


@pytest.mark.requires_host_ffmpeg
def test_video_order_is_independent_of_metric_frame_ids(tmp_path):
    from tests.test_run_inference_pipeline_selection import _eval_args, _eval_item, _NoiseEvalPipe

    item = _eval_item(2, 4, cosmos3=False)
    item.update(
        scene_id="scene",
        frame_indices=torch.tensor([4, 0, 3, 1, 2]),
        gt_index=torch.tensor([4, 0, 3, 1, 2]),
        video_frame_indices=torch.arange(5),
    )
    for index in range(5):
        item["rgb_rendered"][index].fill_(index / 5)
    args = _eval_args("Wan-AI/Wan2.1-T2V-14B-Diffusers", save_frame_outputs_only=False)
    args.inference_pipeline = "bidirectional"
    inference.process_item(_NoiseEvalPipe(), item, args, tmp_path, 0, torch.device("cpu"), 4)
    assert inference.output_video_frames_complete(tmp_path / "scene", item)
    inference.finalize_video_outputs(tmp_path, 15)
    movie = tmp_path / "scene/videos/batch_0000_rendered.mp4"
    decoded = decode_video_to_frames(movie.read_bytes())
    np.testing.assert_allclose(decoded.mean(axis=(1, 2, 3)), np.arange(5) * 51, atol=4)
    metric_frame = tmp_path / "scene/frames/batch_0000/rendered/00000.png"
    assert np.asarray(Image.open(metric_frame)).mean() == 51


def test_incomplete_png_is_not_a_completed_item(tmp_path):
    item = {"rgb_rendered": torch.zeros(1, 3, 16, 16), "frame_indices": [0]}
    for kind in ("pred", "rendered"):
        path = tmp_path / "frames/batch_0000" / kind / "00000.png"
        path.parent.mkdir(parents=True)
        inference.save_image(torch.zeros(3, 16, 16), path)
    assert inference.output_frames_complete(tmp_path, item)
    (tmp_path / "frames/batch_0000/pred/00000.png").write_bytes(b"partial")
    assert not inference.output_frames_complete(tmp_path, item)


def test_failed_png_write_preserves_previous_frame(tmp_path, monkeypatch):
    path = tmp_path / "frame.png"
    inference.save_image(torch.zeros(3, 16, 16), path)
    previous = path.read_bytes()

    def fail(image, filename, *args, **kwargs):
        Path(filename).write_bytes(b"partial")
        raise OSError("write interrupted")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OSError, match="write interrupted"):
        inference.save_image(torch.ones(3, 16, 16), path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
