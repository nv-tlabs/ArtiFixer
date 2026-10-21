# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from zipfile import ZipFile

from model_eval.dl3dv_distill_validation import main as validation_main
from model_eval.prepare_dl3dv_distill_dataset import main
from tests._helpers import DIFIX_REMAP_DROPPED_FRAMES, DIFIX_REMAP_SCENE_ID


class PrepareDL3DVDistillDatasetTests(unittest.TestCase):
    def test_extracts_only_scene_list_scenes_from_split_subdirs(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dl3dv_dir = tmp_dir / "dl3dv"
        output_root = tmp_dir / "staged"
        scene_list = tmp_dir / "scenes.txt"
        split_path = tmp_dir / "trainval_test_split.json"
        (dl3dv_dir / "subdir_a").mkdir(parents=True)
        (dl3dv_dir / "subdir_b").mkdir(parents=True)
        with ZipFile(dl3dv_dir / "subdir_a" / "scene_a.zip", "w") as zip_file:
            zip_file.writestr("scene_a/transforms.json", json.dumps({"frames": []}))
            zip_file.writestr("scene_a/colmap/sparse/0/cameras.bin", b"cameras")
            zip_file.writestr("scene_a/colmap/sparse/0/images.bin", b"images")
            zip_file.writestr("scene_a/colmap/sparse/0/points3D.bin", b"points")
        with ZipFile(dl3dv_dir / "subdir_b" / "scene_b.zip", "w") as zip_file:
            zip_file.writestr("scene_b/transforms.json", json.dumps({"frames": []}))
            zip_file.writestr("scene_b/colmap/sparse/0/cameras.bin", b"cameras")
            zip_file.writestr("scene_b/colmap/sparse/0/images.bin", b"images")
            zip_file.writestr("scene_b/colmap/sparse/0/points3D.bin", b"points")
        split_path.write_text(
            json.dumps(
                {
                    "test": {
                        "scene_a": [{"subdir": "subdir_a"}],
                        "scene_b": [{"subdir": "subdir_b"}],
                    }
                }
            )
        )
        scene_list.write_text("scene_a\n")

        main(
            [
                "--dl3dv_dir",
                str(dl3dv_dir),
                "--split_path",
                str(split_path),
                "--scene_list_file",
                str(scene_list),
                "--output_root",
                str(output_root),
            ]
        )

        self.assertTrue((output_root / "scene_a" / "transforms.json").is_file())
        self.assertTrue((output_root / "scene_a" / "colmap" / "sparse" / "0" / "points3D.bin").is_file())
        self.assertFalse((output_root / "scene_b").exists())

    def test_missing_scene_list_entry_fails_before_extracting(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dl3dv_dir = tmp_dir / "dl3dv"
        scene_list = tmp_dir / "scenes.txt"
        split_path = tmp_dir / "trainval_test_split.json"
        split_path.write_text(json.dumps({"test": {"scene_a": [{"subdir": "subdir_a"}]}}))
        scene_list.write_text("scene_missing\n")

        with self.assertRaisesRegex(ValueError, "scene_missing"):
            main(
                [
                    "--dl3dv_dir",
                    str(dl3dv_dir),
                    "--split_path",
                    str(split_path),
                    "--scene_list_file",
                    str(scene_list),
                    "--output_root",
                    str(tmp_dir / "staged"),
                ]
            )

    def test_transform_only_zip_is_rejected_for_distillation(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dl3dv_dir = tmp_dir / "dl3dv"
        scene_list = tmp_dir / "scenes.txt"
        split_path = tmp_dir / "trainval_test_split.json"
        (dl3dv_dir / "subdir_a").mkdir(parents=True)
        with ZipFile(dl3dv_dir / "subdir_a" / "scene_a.zip", "w") as zip_file:
            zip_file.writestr("scene_a/transforms.json", json.dumps({"frames": []}))
        split_path.write_text(json.dumps({"test": {"scene_a": [{"subdir": "subdir_a"}]}}))
        scene_list.write_text("scene_a\n")

        with self.assertRaisesRegex(FileNotFoundError, "COLMAP sparse"):
            main(
                [
                    "--dl3dv_dir",
                    str(dl3dv_dir),
                    "--split_path",
                    str(split_path),
                    "--scene_list_file",
                    str(scene_list),
                    "--output_root",
                    str(tmp_dir / "staged"),
                ]
            )

    def test_transform_only_zip_can_be_staged_with_colmap_zip_cache(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dl3dv_dir = tmp_dir / "dl3dv"
        colmap_zip_root = tmp_dir / "colmap_cache"
        output_root = tmp_dir / "staged"
        scene_list = tmp_dir / "scenes.txt"
        split_path = tmp_dir / "trainval_test_split.json"
        (dl3dv_dir / "subdir_a").mkdir(parents=True)
        colmap_zip_root.mkdir(parents=True)
        with ZipFile(dl3dv_dir / "subdir_a" / "scene_a.zip", "w") as zip_file:
            zip_file.writestr("scene_a/nerfstudio/transforms.json", json.dumps({"frames": []}))
            zip_file.writestr("scene_a/images/000.png", b"image")
        with ZipFile(colmap_zip_root / "scene_a.zip", "w") as zip_file:
            zip_file.writestr("scene_a/colmap/sparse/0/cameras.bin", b"cameras")
            zip_file.writestr("scene_a/colmap/sparse/0/images.bin", b"images")
            zip_file.writestr("scene_a/colmap/sparse/0/points3D.bin", b"points")
        split_path.write_text(json.dumps({"test": {"scene_a": [{"subdir": "subdir_a"}]}}))
        scene_list.write_text("scene_a\n")

        main(
            [
                "--dl3dv_dir",
                str(dl3dv_dir),
                "--split_path",
                str(split_path),
                "--scene_list_file",
                str(scene_list),
                "--output_root",
                str(output_root),
                "--colmap_zip_root",
                str(colmap_zip_root),
            ]
        )

        self.assertTrue((output_root / "scene_a" / "transforms.json").is_file())
        self.assertTrue((output_root / "scene_a" / "colmap" / "sparse" / "0" / "points3D.bin").is_file())

    def test_staged_difix_remap_scene_validates_in_compact_difix_frame_space(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dl3dv_dir = tmp_dir / "dl3dv"
        output_root = tmp_dir / "staged"
        scene_list = tmp_dir / "scenes.txt"
        split_path = tmp_dir / "trainval_test_split.json"
        (dl3dv_dir / "subdir_a").mkdir(parents=True)
        # Tiny fake DL3DV-ALL-960P frame table: the dropped DiFix frames are
        # interleaved with the kept frames, so compact indices shift after the
        # first dropped frame (kept compact table: 00000, 00001, 00200, 00400).
        kept_frames = ("frame_00000.png", "frame_00001.png", "frame_00200.png", "frame_00400.png")
        all_frames = sorted(kept_frames + DIFIX_REMAP_DROPPED_FRAMES)
        transforms = {"frames": [{"file_path": f"images/{name}"} for name in all_frames]}
        with ZipFile(dl3dv_dir / "subdir_a" / f"{DIFIX_REMAP_SCENE_ID}.zip", "w") as zip_file:
            zip_file.writestr(f"{DIFIX_REMAP_SCENE_ID}/transforms.json", json.dumps(transforms))
            zip_file.writestr(f"{DIFIX_REMAP_SCENE_ID}/colmap/sparse/0/cameras.bin", b"cameras")
            zip_file.writestr(f"{DIFIX_REMAP_SCENE_ID}/colmap/sparse/0/images.bin", b"images")
            zip_file.writestr(f"{DIFIX_REMAP_SCENE_ID}/colmap/sparse/0/points3D.bin", b"points")
            for name in ("frame_00000.png", "frame_00001.png"):
                zip_file.writestr(f"{DIFIX_REMAP_SCENE_ID}/images_4/{name}", b"image")
        split_path.write_text(json.dumps({"test": {DIFIX_REMAP_SCENE_ID: [{"subdir": "subdir_a"}]}}))
        scene_list.write_text(f"{DIFIX_REMAP_SCENE_ID}\n")

        main(
            [
                "--dl3dv_dir",
                str(dl3dv_dir),
                "--split_path",
                str(split_path),
                "--scene_list_file",
                str(scene_list),
                "--output_root",
                str(output_root),
            ]
        )

        # Staging keeps the full DL3DV-ALL-960P frame table: the prepared root is
        # shared with the (full-frame-space) ours split.
        staged_transforms = json.loads((output_root / DIFIX_REMAP_SCENE_ID / "transforms.json").read_text())
        self.assertEqual(len(staged_transforms["frames"]), len(all_frames))

        # DiFix train ids and overrides are indexed in the compact frame space.
        difix_indices_file = tmp_dir / "difix_train_ids" / DIFIX_REMAP_SCENE_ID / "difix_train_ids.json"
        difix_indices_file.parent.mkdir(parents=True)
        difix_indices_file.write_text("[0, 1]")
        pred_dir = tmp_dir / "direct" / DIFIX_REMAP_SCENE_ID / "video_frames" / "batch_0000" / "pred"
        pred_dir.mkdir(parents=True)
        for idx in (2, 3):
            (pred_dir / f"{idx:05d}.png").write_text("pred")
        runtime_dir = tmp_dir / "runtime" / DIFIX_REMAP_SCENE_ID
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = validation_main(
                [
                    str(output_root),
                    str(pred_dir),
                    "--scene_id",
                    DIFIX_REMAP_SCENE_ID,
                    "--selected_indices_file",
                    str(difix_indices_file),
                    "--num_selected_indices",
                    "100000000",
                    "--require_3dgrut_assets",
                    "--runtime_scene_dataset_path",
                    str(runtime_dir),
                    "--print_runtime_scene_info",
                ]
            )

        self.assertEqual(status, 0, stderr.getvalue())
        runtime_transforms = json.loads((runtime_dir / "transforms.json").read_text())
        runtime_frame_paths = [frame["file_path"] for frame in runtime_transforms["frames"]]
        self.assertEqual(runtime_frame_paths, [f"images/{name}" for name in kept_frames])
        self.assertEqual(json.loads((runtime_dir / "selected_indices.json").read_text()), [0, 1])
        self.assertEqual(stdout.getvalue().splitlines()[-1], "2")

    def test_transform_only_zip_can_be_staged_with_extracted_colmap_cache(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dl3dv_dir = tmp_dir / "dl3dv"
        colmap_cache_root = tmp_dir / "colmap_cache"
        output_root = tmp_dir / "staged"
        scene_list = tmp_dir / "scenes.txt"
        split_path = tmp_dir / "trainval_test_split.json"
        (dl3dv_dir / "subdir_a").mkdir(parents=True)
        scene_cache = colmap_cache_root / "scene_a"
        (scene_cache / "colmap" / "sparse" / "0").mkdir(parents=True)
        with ZipFile(dl3dv_dir / "subdir_a" / "scene_a.zip", "w") as zip_file:
            zip_file.writestr("scene_a/nerfstudio/transforms.json", json.dumps({"frames": []}))
            zip_file.writestr("scene_a/images/000.png", b"image")
        (scene_cache / "colmap" / "sparse" / "0" / "cameras.bin").write_bytes(b"cameras")
        (scene_cache / "colmap" / "sparse" / "0" / "images.bin").write_bytes(b"images")
        (scene_cache / "colmap" / "sparse" / "0" / "points3D.bin").write_bytes(b"points")
        split_path.write_text(json.dumps({"test": {"scene_a": [{"subdir": "subdir_a"}]}}))
        scene_list.write_text("scene_a\n")

        main(
            [
                "--dl3dv_dir",
                str(dl3dv_dir),
                "--split_path",
                str(split_path),
                "--scene_list_file",
                str(scene_list),
                "--output_root",
                str(output_root),
                "--colmap_zip_root",
                str(colmap_cache_root),
            ]
        )

        self.assertTrue((output_root / "scene_a" / "transforms.json").is_file())
        self.assertTrue((output_root / "scene_a" / "colmap" / "sparse" / "0" / "points3D.bin").is_file())


if __name__ == "__main__":
    unittest.main()
