# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer caption -> split -> inference/ArtiFixer3D handoffs on real files."""

import json
import struct
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from data_processing import artifixer3d
from data_processing import prepare_colmap_artifixer_inputs as prep
from data_processing import run_captioning
from data_processing.captioning import generate_captions
from data_processing.captioning.prompt_hdf5 import get_cosmos3_tokenizer
from data_processing.prep_utils import write_colmap_cameras, write_colmap_images
from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset
from model_training.prompt_formats import (
    COSMOS3_PROMPT_FORMAT,
    WAN_T5_PROMPT_FORMAT,
    caption_filename_for_format,
    validate_prompt_file_format,
)


def _scene(tmp_path):
    paths = prep.prepared_paths(tmp_path / "capture", "capture", 10000, COSMOS3_PROMPT_FORMAT)
    frames = []
    for i in range(5):
        pose = np.eye(4)
        pose[0, 3] = i * 0.02
        name = f"images/{i:05d}.png"
        for path, mode in (
            (paths.eval_scene_dir / name, "RGB"),
            (paths.render_checkpoint_dir / "renders" / f"{i:05d}.png", "RGB"),
            (paths.render_checkpoint_dir / "opacity" / f"{i:05d}.png", "L"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(mode, (64, 64), 100).save(path)
        frames.append({"file_path": name, "transform_matrix": pose.tolist()})
    transforms = dict(w=64, h=64, fl_x=64.0, fl_y=64.0, cx=32.0, cy=32.0, frames=frames)
    (paths.eval_scene_dir / "nerfstudio").mkdir()
    (paths.eval_scene_dir / "nerfstudio/transforms.json").write_text(json.dumps(transforms))
    paths.selected_indices_path.write_text("[0]")
    (paths.render_checkpoint_dir / "selected_indices.json").write_text("[0]")
    checkpoint = paths.scene_root / "reconstruction.pt"
    checkpoint.write_bytes(b"reconstruction placeholder; metadata handoff only")
    return paths, checkpoint


def test_real_cosmos_caption_to_dataset_and_artifixer3d_plus(tmp_path, monkeypatch):
    paths, checkpoint = _scene(tmp_path)
    tokenizer = get_cosmos3_tokenizer("nvidia/Cosmos3-Nano")
    caption_calls = []

    def caption(images, fps, model, processor):
        caption_calls.append((len(images), fps))
        return "A red cube on a wooden table."

    monkeypatch.setattr(generate_captions, "generate_caption", caption)
    writer = generate_captions.generate_caption_hdf5

    def supplied_models(**kwargs):
        return writer(**kwargs, caption_model=object(), processor=object(), cosmos3_tokenizer=tokenizer)

    monkeypatch.setattr(prep, "generate_caption_hdf5", supplied_models)
    args = Namespace(
        replace=False,
        prompt_format=COSMOS3_PROMPT_FORMAT,
        dataset_fps=12.5,
        captioning_model_id="not-loaded",
        text_encoder_model_id="unused-for-cosmos",
    )
    prep.generate_caption(args, paths)
    caption_path = paths.prompt_dir / paths.scene_id / paths.prompt_filename
    validate_prompt_file_format(caption_path, COSMOS3_PROMPT_FORMAT)
    assert caption_path.name == "caption.cosmos3.h5"
    assert caption_calls == [(5, 12.5)]
    with h5py.File(caption_path) as source:
        payload = next(iter(source.values()))
        assert payload.attrs["fps"] == 12.5
        np.testing.assert_array_equal(payload.attrs["image_indices"], np.arange(5))
        decoded = tokenizer.decode(payload[:])
        assert "0.4 seconds" in decoded
        assert "64x64 resolution" in decoded
        assert "negative_input_ids" in source.attrs
    prep.generate_caption(args, paths)
    assert len(caption_calls) == 1  # valid offline reuse
    args.dataset_fps = 24.0
    with pytest.raises(ValueError, match="FPS"):
        prep.generate_caption(args, paths)
    args.dataset_fps = 12.5
    prep.write_eval_split(paths, 2.5, None, checkpoint)
    saved = json.loads(paths.split_path.read_text())["test"]["capture"]
    assert saved["prompt_path"].endswith("caption.cosmos3.h5")

    def read_item(split):
        return ReconstructedColmapEvalDataset(
            split="test",
            split_path=split,
            num_views=1,
            include_all_frames=True,
            prompt_format=COSMOS3_PROMPT_FORMAT,
            resize_multiple=32,
            camera_pose_alignment="last",
        )[0]

    before = read_item(paths.split_path)
    assert before["rgb_rendered"].shape == (5, 3, 64, 64)
    assert before["encoded_prompt"].shape == (1024,)
    assert before["encoded_prompt"].dtype == torch.long
    assert before["prompt_attention_mask"].dtype == torch.bool
    assert torch.isfinite(before["camera_rays"]).all()
    assert before["neighbor_camera_rays"].shape == (1, 64, 64, 6)
    scene = artifixer3d.load_prepared_scene(paths.scene_root, None, None)
    assert scene.prompt_path == caption_path
    plus_paths = SimpleNamespace(
        artifixer3d_plus_inference_split_path=paths.scene_root / "split_artifixer3d_plus.json",
        distillation_selected_indices_path=paths.selected_indices_path,
    )
    plus = artifixer3d.write_artifixer3d_plus_inference_split(
        scene, plus_paths, paths.render_checkpoint_dir, checkpoint
    )
    after = read_item(plus)
    for key in ("encoded_prompt", "encoded_negative_prompt", "prompt_attention_mask", "camera_rays"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


@pytest.mark.parametrize("filename", ["caption.h5", "caption.hdf5", "frames_all_stride_1.h5", "garden.hdf5"])
def test_caption_name_is_idempotent_for_explicit_cosmos_paths(filename):
    tagged = caption_filename_for_format(filename, COSMOS3_PROMPT_FORMAT)
    assert caption_filename_for_format(tagged, COSMOS3_PROMPT_FORMAT) == tagged
    assert caption_filename_for_format(filename, WAN_T5_PROMPT_FORMAT) == filename


def test_reusing_wan_payload_for_cosmos_fails_before_model_loading(tmp_path, monkeypatch):
    path = tmp_path / "frames_all_stride_1.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("frame", data=np.zeros((2, 4096), dtype=np.uint16))
    validate_prompt_file_format(path, WAN_T5_PROMPT_FORMAT)  # legacy Wan accepted
    monkeypatch.setattr(
        generate_captions,
        "get_caption_model_and_processor",
        lambda *args: pytest.fail("model loading must not be reached"),
    )
    with pytest.raises(ValueError, match="Cosmos3"):
        generate_captions.generate_caption_hdf5(
            tmp_path / "absent-scene", path, prompt_format=COSMOS3_PROMPT_FORMAT, check_if_exists=True
        )


def test_standalone_cosmos_caption_name_and_fps_reach_writer(tmp_path, monkeypatch):
    paths, _ = _scene(tmp_path)
    args = Namespace(
        dl3dv_dir=paths.eval_scene_dir.parent,
        output_dir=tmp_path / "prompts",
        num_frames=0,
        max_frame_stride=1,
        include_reverse=False,
        include_revisit=False,
        output_role="evaluation",
        prompt_format=COSMOS3_PROMPT_FORMAT,
        dataset_fps=29.97,
        captioning_model_id="not-loaded",
        captioning_attn_implementation=None,
    )
    calls = []
    monkeypatch.setattr(run_captioning, "load_models", lambda args: {})
    monkeypatch.setattr(run_captioning, "generate_caption_hdf5", lambda *a, **kw: calls.append((a, kw)))
    run_captioning.generate_for_scene(args, paths.eval_scene_dir, {})
    assert len(calls) == 1
    assert calls[0][0][1].name == "frames_all_stride_1.cosmos3.h5"
    assert calls[0][1]["dataset_fps"] == 29.97


def _entrypoint_scene(tmp_path, trajectory, monkeypatch):
    paths, checkpoint = _scene(tmp_path)
    sparse = paths.eval_scene_dir / "sparse/0"
    sparse.mkdir(parents=True)
    camera = prep.Camera(id=1, model="PINHOLE", width=64, height=64, params=np.array([64.0, 64.0, 32.0, 32.0]))
    images = [
        prep.Image(
            id=i + 1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([-i * 0.02, 0.0, 0.0]),
            camera_id=1,
            name=f"{i:05d}.png",
            xys=np.empty((0, 2)),
            point3D_ids=np.empty(0, dtype=np.int64),
        )
        for i in range(5)
    ]
    write_colmap_cameras(sparse / "cameras.bin", [camera])
    write_colmap_images(sparse / "images.bin", images)
    (sparse / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    tokenizer = get_cosmos3_tokenizer("nvidia/Cosmos3-Nano")
    writer = generate_captions.generate_caption_hdf5
    monkeypatch.setattr(generate_captions, "generate_caption", lambda *a: "A red cube on a wooden table.")
    monkeypatch.setattr(generate_captions, "generate_text_embedding", lambda *a: np.zeros((4, 4096), dtype=np.uint16))

    def supplied_models(**kwargs):
        return writer(
            **kwargs,
            caption_model=object(),
            processor=object(),
            cosmos3_tokenizer=tokenizer,
            text_encoder=object(),
            tokenizer=object(),
        )

    monkeypatch.setattr(prep, "generate_caption_hdf5", supplied_models)
    prep.write_eval_split(paths, 2.5, None, checkpoint)
    saved = json.loads(paths.split_path.read_text())
    if trajectory:
        # No source-camera renders: the old entrypoint skipped its split update.
        novel = paths.scene_root / "novel"
        paths.render_checkpoint_dir.rename(novel)
        targets = paths.scene_root / "novel_target_indices.json"
        targets.write_text("[1, 2, 3, 4]")
        saved["test"][paths.scene_id].update(
            render_dir="novel/renders",
            opacity_dir="novel/opacity",
            target_indices_path=targets.name,
            selected_indices_path="novel/selected_indices.json",
            has_gt=False,
        )
        paths.split_path.write_text(json.dumps(saved))
    return paths, checkpoint


def _entrypoint_args(paths, checkpoint, prompt_format, phase, fps=12.5):
    return prep.build_parser().parse_args(
        [
            "--colmap_dir",
            str(paths.eval_scene_dir),
            "--output_root",
            str(paths.scene_root),
            "--phases",
            phase,
            "--replace",
            "--prompt_format",
            prompt_format,
            "--dataset_fps",
            str(fps),
            "--metric_scale",
            "2.5",
            "--reconstruction_checkpoint",
            str(checkpoint),
        ]
    )


@pytest.mark.parametrize("trajectory", [False, True], ids=["source", "novel"])
@pytest.mark.parametrize("first_format", [COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT])
def test_caption_only_format_switch_preserves_rendered_split(tmp_path, monkeypatch, trajectory, first_format):
    paths, checkpoint = _entrypoint_scene(tmp_path, trajectory, monkeypatch)
    before = json.loads(paths.split_path.read_text())["test"][paths.scene_id]
    other = WAN_T5_PROMPT_FORMAT if first_format == COSMOS3_PROMPT_FORMAT else COSMOS3_PROMPT_FORMAT
    for prompt_format in (first_format, other):
        prep.prepare_colmap_scene(_entrypoint_args(paths, checkpoint, prompt_format, "caption"))
        after = json.loads(paths.split_path.read_text())["test"][paths.scene_id]
        filename = caption_filename_for_format("caption.h5", prompt_format)
        assert after["prompt_path"].endswith(filename)
        validate_prompt_file_format(paths.scene_root / after["prompt_path"], prompt_format, expected_fps=12.5)
        assert {k: v for k, v in after.items() if k != "prompt_path"} == {
            k: v for k, v in before.items() if k != "prompt_path"
        }
        item = ReconstructedColmapEvalDataset(
            split="test",
            split_path=paths.split_path,
            num_views=1,
            prompt_format=prompt_format,
            include_all_frames=True,
            use_target_indices=trajectory,
            resize_multiple=32,
            camera_pose_alignment="last",
        )[0]
        assert torch.isfinite(item["encoded_prompt"]).all()


@pytest.mark.parametrize("trajectory", [False, True], ids=["source", "novel"])
@pytest.mark.parametrize("invalid", ["fps", "missing_format", "malformed"])
def test_non_caption_phase_rejects_bad_prompt_without_changing_split(tmp_path, monkeypatch, trajectory, invalid):
    paths, checkpoint = _entrypoint_scene(tmp_path, trajectory, monkeypatch)
    prep.prepare_colmap_scene(_entrypoint_args(paths, checkpoint, COSMOS3_PROMPT_FORMAT, "caption"))
    before = paths.split_path.read_bytes()
    args = _entrypoint_args(paths, checkpoint, COSMOS3_PROMPT_FORMAT, "scale")
    if invalid == "fps":
        args.dataset_fps = 24.0
    elif invalid == "missing_format":
        args.prompt_format = WAN_T5_PROMPT_FORMAT
    else:
        with h5py.File(paths.prompt_dir / paths.scene_id / paths.prompt_filename, "a") as hf:
            del hf.attrs["negative_input_ids"]
            hf.attrs["negative_input_ids"] = [[1, 2]]
    with pytest.raises((ValueError, FileNotFoundError)):
        prep.prepare_colmap_scene(args)
    assert paths.split_path.read_bytes() == before
