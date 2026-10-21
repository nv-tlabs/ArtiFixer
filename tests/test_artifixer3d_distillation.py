# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_processing import artifixer3d, run_artifixer3d


def _run_distillation(monkeypatch, tmp_path, extra_args=()):
    reconstruction_checkpoint = tmp_path / "prepared checkpoint.pt"
    reconstruction_checkpoint.touch()
    scene = artifixer3d.PreparedScene(
        scene_id="scene",
        scene_root=tmp_path / "prepared scene",
        transforms_path=tmp_path / "transforms.json",
        colmap_dir=tmp_path / "colmap",
        prompt_path=tmp_path / "prompt.pt",
        camera_scale=1.0,
        has_gt=True,
        selected_indices=[0, 3],
        target_indices_path=None,
        reconstruction_checkpoint=reconstruction_checkpoint,
        frame_count=5,
    )
    args = run_artifixer3d.build_parser().parse_args(
        [
            "--scene_root",
            str(scene.scene_root),
            "--artifixer_frames_dir",
            str(tmp_path / "predicted frames"),
            "--phases",
            "distill",
            *extra_args,
        ]
    )
    paths = artifixer3d.artifixer3d_paths(
        scene, args.output_root, args.artifixer3d_plus_inference_split_path, args.artifixer3d_steps
    )
    args.artifixer_frames_dir.mkdir()
    for index in (1, 2, 4):
        Image.new("RGB", (8, 8)).save(args.artifixer_frames_dir / f"{index:05d}.png")
    scene.transforms_path.write_text(
        json.dumps(
            {
                "camera_model": "OPENCV",
                "w": 8,
                "h": 8,
                "fl_x": 8,
                "fl_y": 8,
                "cx": 4,
                "cy": 4,
                "frames": [{"transform_matrix": np.eye(4).tolist()} for _ in range(scene.frame_count)],
            }
        )
    )
    materialized, launches = [], []
    monkeypatch.setattr(artifixer3d, "load_prepared_scene", lambda *args: scene)
    monkeypatch.setattr(artifixer3d, "materialize_distillation_input", lambda *args: materialized.append(args))

    def capture_training(config_name, overrides, config_dir):
        launches.append((config_name, list(overrides), config_dir))
        checkpoint = artifixer3d.artifixer3d_checkpoint(scene, paths, args.artifixer3d_steps)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()

    monkeypatch.setattr(artifixer3d.threedgrut_training, "train_3dgrut", capture_training)

    artifixer3d.run_artifixer3d(args)

    assert materialized == [(scene, paths, args.artifixer_frames_dir.resolve())]
    assert len(launches) == 1
    config_name, overrides, config_dir = launches[0]
    assert config_name == args.config_name
    assert config_dir == artifixer3d.threedgrut_training.DEFAULT_THREEDGRUT_CONFIG_DIR
    options = dict(override.split("=", 1) for override in overrides)
    assert len(options) == len(overrides)
    return paths, options


def test_public_distillation_defaults_to_scratch30k(monkeypatch, tmp_path):
    _, options = _run_distillation(monkeypatch, tmp_path)

    assert options["n_iterations"] == "30000"
    assert options["checkpoint.iterations"] == "[30000]"
    assert "resume" not in options
    assert "resume_mode" not in options
    assert "image_path_override_fallback_to_original" not in options


def test_public_resume_requires_explicit_base_checkpoint(monkeypatch, tmp_path):
    base_checkpoint = tmp_path / "explicit checkpoint.pt"
    base_checkpoint.touch()

    _, options = _run_distillation(monkeypatch, tmp_path, ["--base_checkpoint", str(base_checkpoint)])

    assert options["resume"] == str(base_checkpoint.resolve())
    assert options["n_iterations"] == "30000"


def test_public_distillation_passes_selected_indices_and_image_override(monkeypatch, tmp_path):
    paths, options = _run_distillation(monkeypatch, tmp_path, ["--artifixer3d_steps", "125"])

    assert options["selected_indices_file"] == str(paths.distillation_selected_indices_path)
    assert Path(options["path"]) / options["image_path_override"] == paths.override_image_dir
    assert options["out_dir"] == str(paths.run_root)
    assert options["n_iterations"] == "125"
    assert options["checkpoint.iterations"] == "[125]"


@pytest.mark.parametrize("interrupted", [False, True])
def test_replaced_distillation_never_reuses_previous_results(tmp_path, monkeypatch, interrupted):
    paths, _ = _run_distillation(monkeypatch, tmp_path)
    checkpoint = paths.run_root / "scene/scene/ours_30000/ckpt_30000.pt"
    checkpoint.write_bytes(b"old checkpoint")
    paths.artifixer3d_plus_inference_split_path.parent.mkdir(parents=True, exist_ok=True)
    paths.artifixer3d_plus_inference_split_path.write_text("old split")
    for kind in ("renders", "opacity"):
        (paths.render_checkpoint_dir / kind).mkdir(parents=True)
        for index in range(5):
            Image.new("RGB", (8, 8)).save(paths.render_checkpoint_dir / kind / f"{index:05d}.png")
    (paths.render_checkpoint_dir / "selected_indices.json").write_text("[0, 3]")
    paths.distillation_selected_indices_path.parent.mkdir(parents=True, exist_ok=True)
    paths.distillation_selected_indices_path.write_text("[0, 3]")
    args = run_artifixer3d.build_parser().parse_args(
        [
            "--scene_root",
            str(tmp_path / "prepared scene"),
            "--artifixer_frames_dir",
            str(tmp_path / "predicted frames"),
            "--phases",
            "distill",
            "--replace",
        ]
    )
    calls = []

    def train(*_args):
        calls.append("train")
        if interrupted and len(calls) == 1:
            raise RuntimeError("training interrupted")
        checkpoint.write_bytes(b"new checkpoint")

    monkeypatch.setattr(artifixer3d.threedgrut_training, "train_3dgrut", train)
    if interrupted:
        with pytest.raises(RuntimeError, match="training interrupted"):
            artifixer3d.run_artifixer3d(args)
        assert not checkpoint.exists()
        args.replace = False
    artifixer3d.run_artifixer3d(args)
    assert not paths.artifixer3d_plus_inference_split_path.exists()
    assert not paths.render_checkpoint_dir.exists()

    def render(**kwargs):
        calls.append("render")
        assert kwargs["checkpoint"].read_bytes() == b"new checkpoint"
        return paths.render_checkpoint_dir

    monkeypatch.setattr(artifixer3d, "render_3dgrut_colmap", render)
    args.phases = "render"
    args.replace = False
    artifixer3d.run_artifixer3d(args)
    assert calls == (["train", "train", "render"] if interrupted else ["train", "render"])
