# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from data_processing import prepare_colmap_artifixer_inputs as prep
from data_processing import render_3dgrut_colmap as render_module
from data_processing.prep_utils import write_colmap_cameras, write_colmap_images
from data_processing.prepare_colmap_artifixer_inputs import (
    Camera,
    ColmapScene,
    Image,
    build_parser,
    prepared_paths,
    resolve_colmap_paths,
    resolve_pyramid_factor,
    scale_colmap_scene_to_images,
    threedgrut_train_overrides,
)


def write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", size).save(path)


def pyramid_scene(*, width: int = 3115, height: int = 2078, model: str = "PINHOLE") -> ColmapScene:
    params = [3200.0, 3190.0, 1540.5, 1030.0]
    if model == "OPENCV":
        params += [0.0, 0.0, 0.0, 0.0]
    return ColmapScene(
        cameras=[Camera(id=1, model=model, width=width, height=height, params=np.array(params))],
        images=[
            Image(
                id=1,
                qvec=np.array([np.cos(0.1), 0.0, 0.0, np.sin(0.1)]),
                tvec=np.array([0.2, -0.15, 0.8]),
                camera_id=1,
                name="kitchen.png",
                xys=np.array([[100.0, 200.0], [1800.0, 1100.0]]),
                point3D_ids=np.array([1, -1]),
            )
        ],
    )


def _small_source_scene(tmp_path):
    root = tmp_path / "input"
    sparse = root / "sparse/0"
    sparse.mkdir(parents=True)
    base = pyramid_scene(width=32, height=32)
    images = [
        base.images[0]._replace(id=i + 1, name=f"{i:05d}.png", tvec=np.array([float(i), 0.0, 0.0])) for i in range(3)
    ]
    write_colmap_cameras(sparse / "cameras.bin", base.cameras)
    write_colmap_images(sparse / "images.bin", images)
    (sparse / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    for image in images:
        write_image(root / "images" / image.name, (32, 32))
    selection = tmp_path / "anchors.txt"
    selection.write_text("00000.png\n")
    return root, selection, images


def _prepare_args(source, output, *extra):
    return build_parser().parse_args(
        [
            "--colmap_dir",
            str(source),
            "--output_root",
            str(output),
            *map(str, extra),
        ]
    )


@pytest.mark.parametrize("scale", ["nan", "inf", "0", "-1"])
def test_preparation_rejects_invalid_metric_scale_before_writing(tmp_path, scale):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="finite and positive"):
        prep.prepare_colmap_scene(_prepare_args(tmp_path / "missing", output, "--metric_scale", scale))
    assert not output.exists()


@pytest.mark.parametrize("scale", ["nan", "inf", "0", "-1"])
def test_saved_invalid_scale_is_rejected_before_render_or_caption(tmp_path, monkeypatch, scale):
    source, selection, _ = _small_source_scene(tmp_path)
    output = tmp_path / "prepared"
    prep.prepare_colmap_scene(
        _prepare_args(source, output, "--phases", "prepare", "--selected_image_names_file", selection)
    )
    args = _prepare_args(source, output, "--phases", "render,caption")
    paths = prep.prepared_paths(output, output.name, args.reconstruction_steps, args.prompt_format)
    paths.scale_dir.mkdir(parents=True)
    (paths.scale_dir / "scale_info.txt").write_text(f"Scale factor: {scale}\n")
    paths.split_path.write_text("original split")
    paths.render_result_path.write_text("original render receipt")
    monkeypatch.setattr(prep, "render_reconstruction", lambda *a: pytest.fail("render ran before scale validation"))
    monkeypatch.setattr(prep, "generate_caption", lambda *a: pytest.fail("caption ran before scale validation"))
    with pytest.raises(ValueError, match="finite and positive"):
        prep.prepare_colmap_scene(args)
    assert paths.split_path.read_text() == "original split"
    assert paths.render_result_path.read_text() == "original render receipt"
    repair = _prepare_args(source, output, "--phases", "scale", "--replace", "--metric_scale", "2")
    paths.split_path.unlink()
    paths.render_result_path.unlink()
    prep.prepare_colmap_scene(repair)
    assert prep.read_metric_scale(paths.scale_dir / "scale_info.txt") == 2.0


def test_reconstruction_phase_preserves_saved_anchors_and_frame_order(tmp_path, monkeypatch):
    source, selection, images = _small_source_scene(tmp_path)
    output = tmp_path / "prepared"
    prep.prepare_colmap_scene(
        _prepare_args(source, output, "--phases", "prepare", "--selected_image_names_file", selection)
    )
    # Source records may be enumerated differently later; the prepared order
    # still defines the saved index-to-image mapping.
    write_colmap_images(source / "sparse/0/images.bin", images[::-1])
    captured = []

    def train(config, overrides, config_dir):
        captured.extend(overrides)
        paths = prepared_paths(output, output.name, 10000)
        checkpoint = prep.reconstruction_checkpoint(paths, _prepare_args(source, output))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()

    monkeypatch.setattr(prep, "train_3dgrut", train)
    prep.prepare_colmap_scene(_prepare_args(source, output, "--phases", "reconstruct"))
    assert f"selected_indices_file={output}/selected_indices.json" in captured
    assert "dataset.test_split_interval=0" not in captured
    assert json.loads((output / "selected_indices.json").read_text()) == [0]


@pytest.mark.parametrize("change", ["selection", "camera"])
def test_reused_preparation_rejects_conflicting_inputs(tmp_path, change):
    source, selection, images = _small_source_scene(tmp_path)
    output = tmp_path / "prepared"
    initial = _prepare_args(source, output, "--phases", "prepare", "--selected_image_names_file", selection)
    prep.prepare_colmap_scene(initial)
    before = (output / "selected_indices.json").read_bytes()
    if change == "selection":
        selection.write_text("00001.png\n")
    else:
        images[0] = images[0]._replace(tvec=np.array([9.0, 0.0, 0.0]))
        write_colmap_images(source / "sparse/0/images.bin", images)
    with pytest.raises(ValueError, match="differ"):
        prep.prepare_colmap_scene(initial)
    assert (output / "selected_indices.json").read_bytes() == before


def test_partial_prepare_replacement_does_not_invalidate_saved_outputs(tmp_path):
    source, selection, _ = _small_source_scene(tmp_path)
    output = tmp_path / "prepared"
    prep.prepare_colmap_scene(_prepare_args(source, output, "--phases", "prepare"))
    selection.write_text("00001.png\n")
    before = (output / "selected_indices.json").read_bytes()
    with pytest.raises(ValueError, match="requires all phases"):
        prep.prepare_colmap_scene(
            _prepare_args(source, output, "--phases", "prepare", "--selected_image_names_file", selection, "--replace")
        )
    assert (output / "selected_indices.json").read_bytes() == before


@pytest.mark.parametrize("replace_inputs", [False, True], ids=["reconstruct", "all-phases"])
def test_interrupted_replacement_does_not_reuse_old_results(tmp_path, monkeypatch, replace_inputs):
    source, selection, _ = _small_source_scene(tmp_path)
    output = tmp_path / "prepared"
    prep.prepare_colmap_scene(
        _prepare_args(source, output, "--phases", "prepare", "--selected_image_names_file", selection)
    )
    paths = prepared_paths(output, output.name, 10000)
    checkpoint = prep.reconstruction_checkpoint(paths, _prepare_args(source, output))
    old_outputs = [
        checkpoint,
        paths.render_checkpoint_dir / "renders/00000.png",
        paths.scale_dir / "scale_info.txt",
        paths.prompt_dir / paths.scene_id / paths.prompt_filename,
        prep.trajectory_paths(paths).transforms_path,
        paths.split_path,
        paths.render_result_path,
    ]
    for path in old_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old result")
    (paths.scale_dir / "scale_info.txt").write_text("Scale factor: 1\n")
    phases = "prepare,reconstruct,render,scale,caption" if replace_inputs else "reconstruct"
    expected_indices = [1] if replace_inputs else [0]
    if replace_inputs:
        selection.write_text("00001.png\n")
    trained_selections = []

    def train(config, overrides, config_dir):
        trained_selections.append(json.loads(paths.selected_indices_path.read_text()))
        if len(trained_selections) == 1:
            raise RuntimeError("training interrupted")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("new reconstruction")

    monkeypatch.setattr(prep, "train_3dgrut", train)
    with pytest.raises(RuntimeError, match="training interrupted"):
        prep.prepare_colmap_scene(
            _prepare_args(source, output, "--phases", phases, "--selected_image_names_file", selection, "--replace")
        )
    invalidated = old_outputs if replace_inputs else [checkpoint, paths.split_path, paths.render_result_path]
    assert all(not path.exists() for path in invalidated)
    assert json.loads(paths.selected_indices_path.read_text()) == expected_indices

    prep.prepare_colmap_scene(_prepare_args(source, output, "--phases", "reconstruct"))
    assert trained_selections == [expected_indices, expected_indices]
    assert checkpoint.read_text() == "new reconstruction"
    assert not paths.split_path.exists()


@pytest.mark.parametrize("novel", [False, True])
@pytest.mark.parametrize("interrupted", [False, True], ids=["complete", "caption-retry"])
def test_external_checkpoint_split_uses_actual_render_path(tmp_path, monkeypatch, novel, interrupted):
    source, selection, _ = _small_source_scene(tmp_path)
    output = tmp_path / "prepared"
    checkpoint = tmp_path / "external-30000.pt"
    checkpoint.touch()
    paths = prepared_paths(output, output.name, 10000)
    prompt = paths.prompt_dir / paths.scene_id / paths.prompt_filename

    def write_prompt():
        prompt.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(prompt, "w") as hf:
            hf.create_dataset("0", data=np.zeros((1, 4096), dtype=np.uint16))

    if interrupted:

        def fail_caption(**kwargs):
            raise RuntimeError("caption interrupted")

        monkeypatch.setattr(prep, "generate_caption_hdf5", fail_caption)
    else:
        write_prompt()
    trajectory_args = []
    if novel:
        trajectory_path = tmp_path / "novel.json"
        trajectory_path.write_text(
            json.dumps(
                {
                    "camera_model": "OPENCV",
                    "w": 32,
                    "h": 32,
                    "fl_x": 30,
                    "fl_y": 30,
                    "cx": 16,
                    "cy": 16,
                    "frames": [{"transform_matrix": np.eye(4).tolist()}],
                }
            )
        )
        trajectory_args = ["--trajectory_path", trajectory_path]
    rendered = []

    def render(trajectory_path, output_subdir):
        actual = render_module.render_output_dir(
            paths.render_output_root, prep.DEFAULT_RECON_SUBDIR, paths.threedgrut_input_dir, 30000, output_subdir
        )
        frames = json.loads(trajectory_path.read_text())["frames"]
        for kind in ("renders", "opacity"):
            for index in range(len(frames)):
                write_image(actual / kind / f"{index:05d}.png", (32, 32))
        rendered.append(actual)

    monkeypatch.setattr(
        render_module.Renderer,
        "from_checkpoint",
        lambda **kwargs: argparse.Namespace(global_step=30000, render_from_file=render),
    )
    args = _prepare_args(
        source,
        output,
        "--phases",
        "prepare,reconstruct,render,caption",
        "--selected_image_names_file",
        selection,
        "--reconstruction_checkpoint",
        checkpoint,
        "--metric_scale",
        "1",
        *trajectory_args,
    )
    if interrupted:
        with pytest.raises(RuntimeError, match="caption interrupted"):
            prep.prepare_colmap_scene(args)
        assert not paths.split_path.exists()
        monkeypatch.setattr(prep, "generate_caption_hdf5", lambda **kwargs: write_prompt())
        # The render record supplies both checkpoint and trajectory identity.
        prep.prepare_colmap_scene(_prepare_args(source, output, "--phases", "caption", "--metric_scale", "1"))
        assert len(rendered) == 1
    else:
        prep.prepare_colmap_scene(args)
    split = json.loads(paths.split_path.read_text())["test"][paths.scene_id]
    assert (output / split["render_dir"]).resolve() == rendered[-1] / "renders"
    assert (output / split["reconstruction_checkpoint"]).resolve() == checkpoint
    # An explicit external checkpoint must not reuse arbitrary old default-step renders.
    # A changed novel path on the same root must also update targets and remove stale frames.
    if novel:
        data = json.loads(trajectory_path.read_text())
        pose = np.eye(4)
        pose[0, 3] = 3
        data["frames"].append({"transform_matrix": pose.tolist()})
        trajectory_path.write_text(json.dumps(data))
    write_image(rendered[-1] / "renders/99999.png", (32, 32))
    prep.prepare_colmap_scene(args)
    assert len(rendered) == 2
    assert not (rendered[-1] / "renders/99999.png").exists()
    if novel:
        split = json.loads(paths.split_path.read_text())["test"][paths.scene_id]
        transforms = json.loads((output / split["transforms_path"]).read_text())
        assert len(list((rendered[-1] / "renders").glob("*.png"))) == 2
        assert json.loads((output / split["target_indices_path"]).read_text()) == [0, 1]
        assert transforms["frames"][1]["transform_matrix"] == pose.tolist()


def test_interrupted_render_invalidates_old_completion_record(tmp_path, monkeypatch):
    source, selection, _ = _small_source_scene(tmp_path)
    output = tmp_path / "prepared"
    prep.prepare_colmap_scene(
        _prepare_args(source, output, "--phases", "prepare", "--selected_image_names_file", selection)
    )
    paths = prepared_paths(output, output.name, 10000)
    checkpoint = prep.reconstruction_checkpoint(paths, _prepare_args(source, output))
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    render_dir = paths.render_checkpoint_dir
    for kind in ("renders", "opacity"):
        for index in range(3):
            write_image(render_dir / kind / f"{index:05d}.png", (32, 32))
    (render_dir / "selected_indices.json").write_text("[0]")
    assert render_module.render_outputs_complete(render_dir, 3)
    # A preexisting receipt cannot make a truncated final PNG complete.
    (render_dir / "renders/00002.png").write_bytes(b"partial")
    assert not render_module.render_outputs_complete(render_dir, 3)

    def interrupted(*args, **kwargs):
        assert not (render_dir / "selected_indices.json").exists()
        raise RuntimeError("render interrupted")

    monkeypatch.setattr(
        render_module.Renderer,
        "from_checkpoint",
        lambda **kwargs: argparse.Namespace(global_step=10000, render_from_file=interrupted),
    )
    with pytest.raises(RuntimeError, match="render interrupted"):
        prep.prepare_colmap_scene(_prepare_args(source, output, "--phases", "render"))
    assert not (render_dir / "selected_indices.json").exists()
    assert not paths.render_result_path.exists()


class ColmapInputValidationTests(unittest.TestCase):
    def test_integer_pyramid_rounding_preserves_common_camera_scale(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_image(image_dir / "kitchen.png", (779, 520))
        scene = pyramid_scene()
        scaled = scale_colmap_scene_to_images(image_dir, scene, pyramid_factor=4)
        self.assertEqual((scaled.cameras[0].width, scaled.cameras[0].height), (779, 520))
        np.testing.assert_allclose(scaled.cameras[0].params, scene.cameras[0].params / 4)
        np.testing.assert_allclose(scaled.images[0].xys, scene.images[0].xys / 4)
        np.testing.assert_array_equal(scaled.images[0].point3D_ids, scene.images[0].point3D_ids)

    def test_rounding_requires_a_known_pyramid_factor(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene = pyramid_scene()
        for size in [(779, 520), (779, 519)]:
            with self.subTest(size=size):
                write_image(image_dir / "kitchen.png", size)
                with self.assertRaisesRegex(AssertionError, "camera/image size mismatch"):
                    scale_colmap_scene_to_images(image_dir, scene)

    def test_explicit_factor_accepts_one_pixel_rounding_on_either_axis(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene = pyramid_scene(width=3200, height=2080)
        for width in (799, 800, 801):
            for height in (519, 520, 521):
                with self.subTest(width=width, height=height):
                    write_image(image_dir / "kitchen.png", (width, height))
                    scaled = scale_colmap_scene_to_images(image_dir, scene, pyramid_factor=4)
                    np.testing.assert_array_equal(scaled.cameras[0].params, scene.cameras[0].params / 4)
                    self.assertEqual((scaled.cameras[0].width, scaled.cameras[0].height), (width, height))

    def test_explicit_factor_rejects_sizes_just_outside_rounding_tolerance(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene = pyramid_scene()
        # 3115/4=778.75 and 2078/4=519.5: these cross each side of the one-pixel limit.
        for size in [(777, 520), (780, 520), (779, 518), (779, 521)]:
            with self.subTest(size=size):
                write_image(image_dir / "kitchen.png", size)
                with self.assertRaisesRegex(AssertionError, "pyramid factor 4"):
                    scale_colmap_scene_to_images(image_dir, scene, pyramid_factor=4)

    def test_factor_one_does_not_accept_a_one_pixel_crop(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_image(image_dir / "kitchen.png", (3114, 2078))
        with self.assertRaisesRegex(AssertionError, "pyramid factor 1"):
            scale_colmap_scene_to_images(image_dir, pyramid_scene(), pyramid_factor=1)

    def test_rejects_invalid_pyramid_factor(self) -> None:
        for factor in [0, -2, 1.5, True]:
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    scale_colmap_scene_to_images(Path("unused"), ColmapScene([], []), pyramid_factor=factor)

    def test_preserves_calibration_already_matching_pyramid_images(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory())) / "images_4"
        write_image(image_dir / "kitchen.png", (779, 520))
        scene = pyramid_scene(width=779, height=520)
        scaled = scale_colmap_scene_to_images(image_dir, scene, pyramid_factor=4)
        np.testing.assert_array_equal(scaled.cameras[0].params, scene.cameras[0].params)
        np.testing.assert_array_equal(scaled.images[0].xys, scene.images[0].xys)

    def test_prepared_intrinsics_and_projections_match_native_loader(self) -> None:
        from threedgrut.datasets.dataset_colmap import ColmapDataset

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        world_points = np.array([[0.1, -0.2, 2.0], [-0.3, 0.1, 3.0], [0.4, 0.3, 1.0]])
        for model in ("PINHOLE", "OPENCV"):
            with self.subTest(model=model):
                scene = pyramid_scene(model=model)
                original_root = root / model / "original"
                prepared_root = root / model / "prepared"
                image_dir = original_root / "images_4"
                write_image(image_dir / "kitchen.png", (779, 520))
                scaled = scale_colmap_scene_to_images(image_dir, scene, pyramid_factor=4)
                write_image(prepared_root / "images" / "kitchen.png", (779, 520))
                for scene_root, colmap_scene in [(original_root, scene), (prepared_root, scaled)]:
                    sparse_dir = scene_root / "sparse" / "0"
                    sparse_dir.mkdir(parents=True)
                    write_colmap_cameras(sparse_dir / "cameras.bin", colmap_scene.cameras)
                    write_colmap_images(sparse_dir / "images.bin", colmap_scene.images)

                original = ColmapDataset(str(original_root), device="cpu", downsample_factor=4, test_split_interval=0)
                prepared = ColmapDataset(str(prepared_root), device="cpu", downsample_factor=1, test_split_interval=0)
                original_params, original_origins, original_rays, camera_type = original.intrinsics[1][:4]
                prepared_params, prepared_origins, prepared_rays, prepared_type = prepared.intrinsics[1][:4]
                self.assertEqual(camera_type, prepared_type)
                self.assertEqual(original_params.keys(), prepared_params.keys())
                for key in original_params:
                    np.testing.assert_array_equal(prepared_params[key], original_params[key], err_msg=key)
                torch.testing.assert_close(prepared_origins, original_origins, rtol=0, atol=0)
                torch.testing.assert_close(prepared_rays, original_rays, rtol=0, atol=0)
                np.testing.assert_array_equal(prepared.poses, original.poses)

                projections = []
                for dataset, params in [(original, original_params), (prepared, prepared_params)]:
                    world_to_camera = np.linalg.inv(dataset.poses[0])
                    camera_points = world_points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
                    pixels = camera_points[:, :2] / camera_points[:, 2:]
                    projections.append(pixels * params["focal_length"] + params["principal_point"])
                np.testing.assert_allclose(projections[0], projections[1], rtol=0, atol=1e-6)

    def test_scales_camera_intrinsics_to_match_cleanly_resized_image_file(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_image(image_dir / "frame_00001.png", (960, 540))
        scene = ColmapScene(
            cameras=[
                Camera(
                    id=1,
                    model="OPENCV",
                    width=3840,
                    height=2160,
                    params=np.array([1856.0, 1848.0, 1920.0, 1080.0, 0.1, 0.2, 0.3, 0.4]),
                )
            ],
            images=[
                Image(
                    id=1,
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.zeros(3),
                    camera_id=1,
                    name="frame_00001.png",
                    xys=np.array([[384.0, 216.0], [1920.0, 1080.0]]),
                    point3D_ids=np.array([10, -1]),
                )
            ],
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            scaled = scale_colmap_scene_to_images(image_dir, scene)

        camera = scaled.cameras[0]
        image = scaled.images[0]
        self.assertEqual(camera.width, 960)
        self.assertEqual(camera.height, 540)
        np.testing.assert_allclose(camera.params, [464.0, 462.0, 480.0, 270.0, 0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(image.xys, [[96.0, 54.0], [480.0, 270.0]])
        np.testing.assert_array_equal(image.point3D_ids, [10, -1])
        self.assertIn("Scaling COLMAP camera 1 from 3840x2160 to 960x540", stdout.getvalue())

    def test_rejects_camera_resolution_mismatch_that_changes_aspect_ratio(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_image(image_dir / "frame_00001.png", (32, 20))
        scene = ColmapScene(
            cameras=[
                Camera(
                    id=1,
                    model="OPENCV",
                    width=64,
                    height=32,
                    params=np.array([40.0, 40.0, 32.0, 16.0, 0.0, 0.0, 0.0, 0.0]),
                )
            ],
            images=[
                Image(
                    id=1,
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.zeros(3),
                    camera_id=1,
                    name="frame_00001.png",
                    xys=np.empty((0, 2)),
                    point3D_ids=np.empty((0,)),
                )
            ],
        )

        with self.assertRaisesRegex(AssertionError, "camera/image size mismatch"):
            scale_colmap_scene_to_images(image_dir, scene)

    def test_mismatch_error_reports_the_image_for_that_camera(self) -> None:
        image_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_image(image_dir / "camera_2.png", (32, 20))
        write_image(image_dir / "camera_1.png", (32, 16))
        scene = ColmapScene(
            cameras=[
                Camera(
                    id=1,
                    model="OPENCV",
                    width=64,
                    height=32,
                    params=np.array([40.0, 40.0, 32.0, 16.0, 0.0, 0.0, 0.0, 0.0]),
                ),
                Camera(
                    id=2,
                    model="OPENCV",
                    width=64,
                    height=32,
                    params=np.array([40.0, 40.0, 32.0, 16.0, 0.0, 0.0, 0.0, 0.0]),
                ),
            ],
            images=[
                Image(
                    id=2,
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.zeros(3),
                    camera_id=2,
                    name="camera_2.png",
                    xys=np.empty((0, 2)),
                    point3D_ids=np.empty((0,)),
                ),
                Image(
                    id=1,
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.zeros(3),
                    camera_id=1,
                    name="camera_1.png",
                    xys=np.empty((0, 2)),
                    point3D_ids=np.empty((0,)),
                ),
            ],
        )

        with self.assertRaisesRegex(AssertionError, "camera_2.png"):
            scale_colmap_scene_to_images(image_dir, scene)


class ThreedgrutTrainOverridesTests(unittest.TestCase):
    def test_all_images_selected_uses_interval_zero_instead_of_an_empty_validation_split(self) -> None:
        paths = prepared_paths(Path("/prep/scene"), "scene", 100)
        args = argparse.Namespace(reconstruction_steps=100)

        overrides = threedgrut_train_overrides(paths, args, num_selected_views=3, num_images=3)
        self.assertIn("dataset.test_split_interval=0", overrides)
        self.assertFalse(any(o.startswith("selected_indices_file=") for o in overrides))

        overrides = threedgrut_train_overrides(paths, args, num_selected_views=2, num_images=3)
        self.assertIn(f"selected_indices_file={paths.selected_indices_path}", overrides)
        self.assertNotIn("dataset.test_split_interval=0", overrides)


def write_sparse_model(
    sparse_dir: Path, names: tuple[str, ...] = ("cameras.bin", "images.bin", "points3D.bin")
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (sparse_dir / name).write_bytes(b"")


class ResolveColmapPathsTests(unittest.TestCase):
    def test_resolves_known_factor_from_selected_image_directory(self) -> None:
        self.assertEqual(resolve_pyramid_factor(Path("scene/images_4"), None), 4)
        self.assertEqual(resolve_pyramid_factor(Path("scene/images_4"), 4), 4)
        self.assertEqual(resolve_pyramid_factor(Path("scene/images"), 4), 4)
        self.assertIsNone(resolve_pyramid_factor(Path("scene/images"), None))

    def test_rejects_conflicting_or_invalid_factor_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts with image directory"):
            resolve_pyramid_factor(Path("scene/images_4"), 2)
        for path, factor in [(Path("scene/images_0"), None), (Path("scene/images"), 0)]:
            with self.subTest(path=path, factor=factor):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    resolve_pyramid_factor(path, factor)

    def test_cli_accepts_explicit_pyramid_factor(self) -> None:
        args = build_parser().parse_args(["--colmap_dir", "scene", "--output_root", "output", "--pyramid_factor", "4"])
        self.assertEqual(args.pyramid_factor, 4)
        defaults = build_parser().parse_args(["--colmap_dir", "scene", "--output_root", "output"])
        self.assertIsNone(defaults.pyramid_factor)

    def test_accepts_plain_colmap_export(self) -> None:
        scene_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (scene_dir / "images").mkdir()
        write_sparse_model(scene_dir / "sparse" / "0")

        self.assertEqual(resolve_colmap_paths(scene_dir), (scene_dir / "images", scene_dir / "sparse" / "0"))

    def test_accepts_extracted_dl3dv_scene_with_colmap_cache(self) -> None:
        scene_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (scene_dir / "images_4").mkdir()
        (scene_dir / "transforms.json").write_text("{}")
        write_sparse_model(scene_dir / "colmap" / "sparse" / "0")

        self.assertEqual(
            resolve_colmap_paths(scene_dir), (scene_dir / "images_4", scene_dir / "colmap" / "sparse" / "0")
        )

    def test_prefers_full_resolution_images_directory(self) -> None:
        scene_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (scene_dir / "images").mkdir()
        (scene_dir / "images_4").mkdir()
        write_sparse_model(scene_dir / "sparse" / "0")

        image_dir, _ = resolve_colmap_paths(scene_dir)

        self.assertEqual(image_dir, scene_dir / "images")

    def test_rejects_ambiguous_downsampled_image_directories(self) -> None:
        scene_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (scene_dir / "images_2").mkdir()
        (scene_dir / "images_4").mkdir()
        write_sparse_model(scene_dir / "sparse" / "0")

        with self.assertRaisesRegex(ValueError, "images_2, images_4"):
            resolve_colmap_paths(scene_dir)

    def test_requires_every_sparse_model_file(self) -> None:
        scene_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (scene_dir / "images_4").mkdir()
        write_sparse_model(scene_dir / "colmap" / "sparse" / "0", names=("cameras.bin", "images.bin"))

        with self.assertRaisesRegex(FileNotFoundError, "points3D.bin"):
            resolve_colmap_paths(scene_dir)

    def test_reports_both_sparse_locations_when_missing(self) -> None:
        scene_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (scene_dir / "images").mkdir()

        with self.assertRaisesRegex(FileNotFoundError, "colmap/sparse/0 or .*sparse/0"):
            resolve_colmap_paths(scene_dir)


if __name__ == "__main__":
    unittest.main()
