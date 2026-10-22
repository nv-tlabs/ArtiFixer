# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from model_eval import inference


def test_resume_requires_complete_pngs_in_both_output_trees(tmp_path):
    item = {"rgb_rendered": torch.zeros(1, 3, 8, 8), "rgb_gt": torch.zeros(1, 3, 8, 8)}
    for tree, check in [
        ("frames", inference.output_frames_complete),
        ("video_frames", inference.output_video_frames_complete),
    ]:
        for kind in ("pred", "gt", "rendered"):
            path = tmp_path / tree / "batch_0000" / kind / "00000.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(path)
        assert check(tmp_path, item)
        path.write_bytes(b"\x89PNG\r\n\x1a\n")  # interrupted last write
        assert not check(tmp_path, item)


def test_partial_output_reaches_generation_instead_of_skipping(tmp_path, monkeypatch):
    item = {"scene_id": "scene", "rgb_rendered": torch.zeros(1, 3, 8, 8), "rgb_gt": torch.zeros(1, 3, 8, 8)}
    for kind in ("pred", "gt", "rendered"):
        path = tmp_path / "scene/frames/batch_0000" / kind / "00000.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(path)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    args = SimpleNamespace(
        replace_if_exists=False, save_frame_outputs_only=True, model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers"
    )

    class ReachedGeneration(Exception):
        pass

    def reached(*args):
        raise ReachedGeneration

    monkeypatch.setattr(inference, "is_cosmos3_model", reached)
    with pytest.raises(ReachedGeneration):
        inference.process_item(None, item, args, tmp_path, 0, torch.device("cpu"), 4)


def test_failed_image_write_leaves_previous_image_intact(tmp_path, monkeypatch):
    path = tmp_path / "frame.png"
    Image.new("RGB", (8, 8)).save(path)
    original = path.read_bytes()

    def fail_save(self, path, **kwargs):
        path.write_bytes(b"partial")
        raise OSError("injected write failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(OSError, match="injected write failure"):
        inference.save_image(torch.ones(3, 8, 8), path)
    assert path.read_bytes() == original
    assert sorted(tmp_path.iterdir()) == [path]


def test_atomic_image_save_preserves_exact_pixel_conversion(tmp_path):
    frame = torch.linspace(-0.1, 1.1, 3 * 8 * 8).view(3, 8, 8)
    path = tmp_path / "frame.png"
    expected = (frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    inference.save_image(frame, path)
    with Image.open(path) as image:
        import numpy as np

        np.testing.assert_array_equal(np.array(image), expected)
    assert sorted(tmp_path.iterdir()) == [path]


def test_atomic_image_save_preserves_normal_file_permissions(tmp_path):
    control = tmp_path / "control.png"
    Image.new("RGB", (8, 8)).save(control)
    path = tmp_path / "frame.png"
    inference.save_image(torch.zeros(3, 8, 8), path)
    assert path.stat().st_mode == control.stat().st_mode
    path.chmod(0o640)
    inference.save_image(torch.ones(3, 8, 8), path)
    assert path.stat().st_mode & 0o777 == 0o640


def test_video_finalization_ignores_interrupted_atomic_writes(tmp_path, monkeypatch):
    pred = tmp_path / "scene/video_frames/batch_0000/pred"
    pred.mkdir(parents=True)
    frame = pred / "00000.png"
    Image.new("RGB", (8, 8)).save(frame)
    (pred / ".00001.png.interrupted.tmp").write_bytes(b"partial")
    videos = []
    monkeypatch.setattr(inference, "save_video", lambda paths, *args, **kwargs: videos.append(paths))
    inference.finalize_video_outputs(tmp_path, fps=15)
    assert videos == [[frame]]
