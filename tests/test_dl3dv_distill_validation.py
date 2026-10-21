# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from model_eval import dl3dv_difix_frame_remap
from model_eval.dl3dv_distill_validation import (
    main,
    prepare_3dgrut_scene_dataset,
    select_matching_scene_dataset_path,
    validate_override_frame_indices,
)


@pytest.fixture
def difix_preparation_scene(tmp_path, monkeypatch):
    source = tmp_path / "scene"
    (source / "images_4").mkdir(parents=True)
    (source / "colmap/sparse/0").mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (source / "colmap/sparse/0" / name).write_bytes(b"fixture")
    frames = [{"file_path": f"images/frame_{index:05d}.png"} for index in range(4)]
    (source / "transforms.json").write_text(json.dumps({"frames": frames}))
    for frame in frames:
        (source / "images_4" / Path(frame["file_path"]).name).touch()
    ids = tmp_path / "difix_train_ids.json"
    ids.write_text("[1]")
    monkeypatch.setattr(dl3dv_difix_frame_remap, "load_difix_frame_remap", lambda: {"scene": {"frame_00001.png"}})
    return source, ids


def test_prepare_only_cli_uses_compact_difix_frames(tmp_path, difix_preparation_scene):
    source, ids = difix_preparation_scene
    runtime = tmp_path / "runtime/scene"
    runtime.mkdir(parents=True)
    (runtime / "old-runtime-file").write_text("replace me")
    assert (
        main(
            [
                str(source),
                "--prepare_only",
                "--runtime_scene_dataset_path",
                str(runtime),
                "--selected_indices_file",
                str(ids),
            ]
        )
        == 0
    )
    prepared = json.loads((runtime / "transforms.json").read_text())
    assert len(prepared["frames"]) == 3
    assert prepared["frames"][1]["file_path"] == "images/frame_00002.png"
    assert json.loads((runtime / "selected_indices.json").read_text()) == [1]
    assert not (runtime / "colmap/sparse/0/images.bin").exists()
    assert not (runtime / "old-runtime-file").exists()


@pytest.mark.parametrize("destination", ["source", "ancestor", "alias"])
def test_prepare_only_rejects_source_destructive_destinations(tmp_path, difix_preparation_scene, destination):
    source, ids = difix_preparation_scene
    if destination == "source":
        runtime = source
    elif destination == "ancestor":
        runtime = source.parent
    else:
        runtime = tmp_path / "source-alias"
        runtime.symlink_to(source, target_is_directory=True)
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="must not be the source directory or an ancestor"):
        main(
            [
                str(source),
                "--prepare_only",
                "--runtime_scene_dataset_path",
                str(runtime),
                "--selected_indices_file",
                str(ids),
            ]
        )
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == before
    assert ids.read_text() == "[1]"


class DL3DVDistillValidationTests(unittest.TestCase):
    def _write_transforms(self, path: Path, frame_count: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"frames": [{"file_path": f"{idx}.png"} for idx in range(frame_count)]}))

    def _write_predictions(self, pred_dir: Path, frame_count: int) -> None:
        pred_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(frame_count):
            (pred_dir / f"{idx:05d}.png").write_text("pred")

    def test_accepts_override_indices_inside_scene_frame_table(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps({"frames": [{"file_path": "0.png"}, {"file_path": "1.png"}, {"file_path": "2.png"}]})
        )
        (pred_dir / "00000.png").write_text("pred")
        (pred_dir / "00002.png").write_text("pred")

        validate_override_frame_indices(scene_dir, pred_dir)

    def test_rejects_missing_non_source_override_predictions(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        selected_indices = tmp_dir / "selected.json"
        scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps(
                {
                    "frames": [
                        {"file_path": "0.png"},
                        {"file_path": "1.png"},
                        {"file_path": "2.png"},
                        {"file_path": "3.png"},
                    ]
                }
            )
        )
        selected_indices.write_text("[0]")
        (pred_dir / "00002.png").write_text("pred")
        (pred_dir / "00003.png").write_text("pred")

        with self.assertRaisesRegex(ValueError, "missing predictions for non-source"):
            validate_override_frame_indices(
                scene_dir,
                pred_dir,
                selected_indices_file=selected_indices,
                num_selected_indices=1,
            )

    def test_accepts_complete_video_frame_override_predictions(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "video_frames" / "batch_0000" / "pred"
        selected_indices = tmp_dir / "selected.json"
        scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps(
                {
                    "frames": [
                        {"file_path": "0.png"},
                        {"file_path": "1.png"},
                        {"file_path": "2.png"},
                    ]
                }
            )
        )
        selected_indices.write_text("[0]")
        for idx in range(3):
            (pred_dir / f"{idx:05d}.png").write_text("pred")

        validate_override_frame_indices(
            scene_dir,
            pred_dir,
            selected_indices_file=selected_indices,
            num_selected_indices=1,
        )

    def test_selects_prepared_root_scene_layout_only(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dataset_root = tmp_dir / "dataset"
        scene_root = dataset_root / "scene_a"
        pred_dir = tmp_dir / "direct" / "scene_a" / "video_frames" / "batch_0000" / "pred"
        selected_indices = tmp_dir / "selected.json"

        self._write_transforms(scene_root / "transforms.json", 3)
        self._write_predictions(pred_dir, 3)
        selected_indices.write_text("[0]")

        selected = select_matching_scene_dataset_path(
            dataset_root,
            "scene_a",
            pred_dir,
            selected_indices_file=selected_indices,
            num_selected_indices=1,
        )

        self.assertEqual(selected, scene_root)

    def test_rejects_nerfstudio_only_scene_layout(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dataset_root = tmp_dir / "dataset"
        scene_root = dataset_root / "scene_a"
        nerfstudio_scene = scene_root / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene_a" / "video_frames" / "batch_0000" / "pred"
        selected_indices = tmp_dir / "selected.json"

        self._write_transforms(nerfstudio_scene / "transforms.json", 3)
        self._write_predictions(pred_dir, 3)
        selected_indices.write_text("[0]")

        with self.assertRaisesRegex(FileNotFoundError, "scene_a/transforms.json"):
            select_matching_scene_dataset_path(
                dataset_root,
                "scene_a",
                pred_dir,
                selected_indices_file=selected_indices,
                num_selected_indices=1,
            )

    def test_can_require_colmap_points_when_selecting_scene_dataset_path(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dataset_root = tmp_dir / "dataset"
        scene_root = dataset_root / "scene_a"
        pred_dir = tmp_dir / "direct" / "scene_a" / "video_frames" / "batch_0000" / "pred"
        selected_indices = tmp_dir / "selected.json"

        self._write_transforms(scene_root / "transforms.json", 3)
        self._write_predictions(pred_dir, 3)
        selected_indices.write_text("[0]")
        (scene_root / "colmap" / "sparse" / "0").mkdir(parents=True)
        (scene_root / "colmap" / "sparse" / "0" / "points3D.bin").write_text("points")

        selected = select_matching_scene_dataset_path(
            dataset_root,
            "scene_a",
            pred_dir,
            selected_indices_file=selected_indices,
            num_selected_indices=1,
            require_3dgrut_assets=True,
        )

        self.assertEqual(selected, scene_root)

    def test_prepare_3dgrut_scene_dataset_rejects_missing_source_images(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "dataset" / "scene_a"
        runtime_dir = tmp_dir / "runtime" / "scene_a"
        selected_indices = tmp_dir / "selected.json"
        pred_dir = tmp_dir / "direct" / "scene_a" / "video_frames" / "batch_0000" / "pred"

        self._write_transforms(scene_dir / "transforms.json", 3)
        self._write_predictions(pred_dir, 3)
        selected_indices.write_text("[0, 1]")
        (scene_dir / "images_4").mkdir(parents=True)
        (scene_dir / "images_4" / "0.png").write_text("source")
        sparse_dir = scene_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True)
        (sparse_dir / "cameras.bin").write_text("cameras")
        (sparse_dir / "images.bin").write_text("images")
        (sparse_dir / "points3D.bin").write_text("points")

        with self.assertRaisesRegex(ValueError, "Selected source frames are missing source images"):
            prepare_3dgrut_scene_dataset(
                scene_dir,
                runtime_dir,
                selected_indices,
                2,
                downsample_factor=4,
            )

    def test_prepare_3dgrut_scene_dataset_omits_images_bin(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "dataset" / "scene_a"
        runtime_dir = tmp_dir / "runtime" / "scene_a"
        selected_indices = tmp_dir / "selected.json"

        self._write_transforms(scene_dir / "transforms.json", 3)
        selected_indices.write_text("[0, 1]")
        (scene_dir / "images_4").mkdir(parents=True)
        (scene_dir / "images_4" / "0.png").write_text("source")
        (scene_dir / "images_4" / "1.png").write_text("source")
        sparse_dir = scene_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True)
        (sparse_dir / "cameras.bin").write_text("cameras")
        (sparse_dir / "images.bin").write_text("images")
        (sparse_dir / "points3D.bin").write_text("points")

        prepared = prepare_3dgrut_scene_dataset(
            scene_dir,
            runtime_dir,
            selected_indices,
            2,
            downsample_factor=4,
        )

        self.assertEqual(prepared.scene_dataset_path, runtime_dir)
        self.assertEqual(json.loads(prepared.selected_indices_file.read_text()), [0, 1])
        self.assertEqual(prepared.num_selected_indices, 2)
        self.assertTrue((runtime_dir / "transforms.json").is_symlink())
        self.assertTrue((runtime_dir / "images_4").is_symlink())
        self.assertTrue((runtime_dir / "colmap" / "sparse" / "0" / "cameras.bin").is_symlink())
        self.assertTrue((runtime_dir / "colmap" / "sparse" / "0" / "points3D.bin").is_symlink())
        self.assertFalse((runtime_dir / "colmap" / "sparse" / "0" / "images.bin").exists())

    def test_prepare_3dgrut_scene_dataset_rejects_negative_selected_indices(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "dataset" / "scene_a"
        runtime_dir = tmp_dir / "runtime" / "scene_a"
        selected_indices = tmp_dir / "selected.json"

        self._write_transforms(scene_dir / "transforms.json", 3)
        selected_indices.write_text("[-1]")
        (scene_dir / "images_4").mkdir(parents=True)
        (scene_dir / "images_4" / "2.png").write_text("source")
        sparse_dir = scene_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True)
        (sparse_dir / "points3D.bin").write_text("points")

        with self.assertRaisesRegex(ValueError, "non-negative"):
            prepare_3dgrut_scene_dataset(
                scene_dir,
                runtime_dir,
                selected_indices,
                1,
                downsample_factor=4,
            )

    def test_cli_can_print_selected_scene_dataset_path(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dataset_root = tmp_dir / "dataset"
        scene_root = dataset_root / "scene_a"
        pred_dir = tmp_dir / "direct" / "scene_a" / "video_frames" / "batch_0000" / "pred"
        selected_indices = tmp_dir / "selected.json"
        stdout = io.StringIO()

        self._write_transforms(scene_root / "transforms.json", 3)
        self._write_predictions(pred_dir, 3)
        selected_indices.write_text("[0]")

        with redirect_stdout(stdout):
            status = main(
                [
                    str(dataset_root),
                    str(pred_dir),
                    "--scene_id",
                    "scene_a",
                    "--selected_indices_file",
                    str(selected_indices),
                    "--num_selected_indices",
                    "1",
                    "--print_scene_dataset_path",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(stdout.getvalue().strip(), str(scene_root))

    def test_rejects_override_indices_outside_scene_frame_table(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps({"frames": [{"file_path": "0.png"}, {"file_path": "1.png"}, {"file_path": "2.png"}]})
        )
        (pred_dir / "00003.png").write_text("pred")

        with self.assertRaisesRegex(ValueError, "outside the scene frame table"):
            validate_override_frame_indices(scene_dir, pred_dir)

    def test_rejects_override_indices_with_shifted_reference_frame_table(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        reference_scene_dir = tmp_dir / "reference" / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        reference_scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps({"frames": [{"file_path": "images/frame_00001.png"}, {"file_path": "images/frame_00003.png"}]})
        )
        (reference_scene_dir / "transforms.json").write_text(
            json.dumps({"frames": [{"file_path": "images/frame_00001.png"}, {"file_path": "images/frame_00002.png"}]})
        )
        (pred_dir / "00001.png").write_text("pred")

        with self.assertRaisesRegex(ValueError, "different frame at override index"):
            validate_override_frame_indices(
                scene_dir,
                pred_dir,
                reference_scene_dataset_path=reference_scene_dir,
            )

    def test_rejects_override_indices_with_matching_paths_but_different_pose(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        reference_scene_dir = tmp_dir / "reference" / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        reference_scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps(
                {
                    "frames": [
                        {"file_path": "images/frame_00001.png", "transform_matrix": [[1, 0, 0, 0]]},
                    ]
                }
            )
        )
        (reference_scene_dir / "transforms.json").write_text(
            json.dumps(
                {
                    "frames": [
                        {"file_path": "images/frame_00001.png", "transform_matrix": [[1, 0, 0, 1]]},
                    ]
                }
            )
        )
        (pred_dir / "00000.png").write_text("pred")

        with self.assertRaisesRegex(ValueError, "camera metadata differs"):
            validate_override_frame_indices(scene_dir, pred_dir, reference_scene_dataset_path=reference_scene_dir)

    def test_cli_reports_validation_errors_without_traceback(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "scene" / "nerfstudio"
        pred_dir = tmp_dir / "direct" / "scene" / "frames" / "batch_0000" / "pred"
        scene_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(
            json.dumps({"frames": [{"file_path": "0.png"}, {"file_path": "1.png"}, {"file_path": "2.png"}]})
        )
        (pred_dir / "00003.png").write_text("pred")
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            status = main([str(scene_dir), str(pred_dir)])

        self.assertEqual(status, 2)
        self.assertIn("outside the scene frame table", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
