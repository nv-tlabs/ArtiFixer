# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from data_processing.run_sparse_reconstruction import Scene
from data_processing.run_sparse_reconstruction import parse_args as parse_sparse_args
from data_processing.run_sparse_reconstruction import run_scene
from data_processing.scene_utils import (
    discover_scene_zips,
    downsampled_image_path,
    load_scene_transforms_from_dir,
    load_scene_transforms_from_zip,
    load_source_scene_transforms,
    resolve_scene_path,
    safe_extractall,
    scene_zip_member,
)
from data_processing.sparse_recon.half_covisibility_sampling import write_half_covisibility_sampled_indices
from data_processing.sparse_recon.workflow import (
    copy_reconstruction_outputs,
    extract_colmap_zip,
    extract_scene_zip,
    find_experiment_scene_dir,
    image_dir_for_scene,
    reconstruction_outputs,
)


def write_zip(path: Path, members: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as zf:
        for member, contents in members.items():
            zf.writestr(member, contents)


class SceneUtilsTests(unittest.TestCase):
    def test_downsampled_image_path_rewrites_only_path_component(self) -> None:
        self.assertEqual(downsampled_image_path("images/frame_00001.png", 4), "images_4/frame_00001.png")
        self.assertEqual(downsampled_image_path("./scene/images/frame.png", 8), "scene/images_8/frame.png")
        self.assertEqual(downsampled_image_path("scene/images_4/frame.png", 4), "scene/images_4/frame.png")
        self.assertEqual(downsampled_image_path("scene/images/frame.png", 1), "scene/images/frame.png")

        with self.assertRaises(ValueError):
            downsampled_image_path("scene/not_images/frame.png", 4)

    def test_image_dir_for_scene_rejects_fractional_downsample(self) -> None:
        self.assertEqual(image_dir_for_scene(Path("/scene"), 1.0), Path("/scene/images"))
        self.assertEqual(image_dir_for_scene(Path("/scene"), 4.0), Path("/scene/images_4"))
        with self.assertRaises(AssertionError):
            image_dir_for_scene(Path("/scene"), 4.5)

    def test_safe_extractall_rejects_path_traversal(self) -> None:
        work_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        zip_path = work_dir / "bad.zip"
        write_zip(zip_path, {"../escape.txt": "bad"})

        with ZipFile(zip_path, "r") as zf:
            with self.assertRaises(ValueError):
                safe_extractall(zf, work_dir / "extract")

        self.assertFalse((work_dir / "escape.txt").exists())

    def test_resolve_scene_path_rejects_ambiguous_scene_id(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_zip(root / "1K" / "scene_a.zip", {"scene_a/transforms.json": "{}"})
        write_zip(root / "2K" / "scene_a.zip", {"scene_a/transforms.json": "{}"})

        with self.assertRaises(ValueError):
            resolve_scene_path(root, "scene_a")

        self.assertEqual(resolve_scene_path(root, "1K/scene_a.zip"), root / "1K" / "scene_a.zip")

    def test_resolve_scene_path_finds_unique_zip_basename_in_subdir(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        zip_path = root / "1K" / "scene_a.zip"
        write_zip(zip_path, {"scene_a/transforms.json": "{}"})

        self.assertEqual(resolve_scene_path(root, "scene_a.zip"), zip_path)

    def test_resolve_scene_path_rejects_ambiguous_zip_basename(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_zip(root / "1K" / "scene_a.zip", {"scene_a/transforms.json": "{}"})
        write_zip(root / "2K" / "scene_a.zip", {"scene_a/transforms.json": "{}"})

        with self.assertRaises(ValueError):
            resolve_scene_path(root, "scene_a.zip")

    def test_resolve_scene_path_finds_scene_directory_in_subdir(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = root / "1K" / "scene_a"
        scene_dir.mkdir(parents=True)

        self.assertEqual(resolve_scene_path(root, "scene_a"), scene_dir)

    def test_discover_scene_zips_recurses_under_dl3dv_root(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_zip(root / "2K" / "scene_b.zip", {"scene_b/transforms.json": "{}"})
        write_zip(root / "1K" / "scene_a.zip", {"scene_a/transforms.json": "{}"})

        self.assertEqual(
            discover_scene_zips(root),
            [root / "1K" / "scene_a.zip", root / "2K" / "scene_b.zip"],
        )

    def test_extract_colmap_zip_supports_rootless_archive(self) -> None:
        work_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        colmap_zip = work_root / "colmap" / "scene_a.zip"
        task_dir = work_root / "work"
        scene_dir = task_dir / "scene_a"
        scene_dir.mkdir(parents=True)

        write_zip(colmap_zip, {"colmap/sparse/0/images.bin": "colmap"})
        extract_colmap_zip(colmap_zip, task_dir, scene_dir, "scene_a")

        self.assertEqual((scene_dir / "colmap/sparse/0/images.bin").read_text(), "colmap")

    def test_extract_colmap_zip_supports_scene_root_archive(self) -> None:
        work_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        colmap_zip = work_root / "colmap" / "scene_a.zip"
        task_dir = work_root / "work"
        scene_dir = task_dir / "scene_a"
        scene_dir.mkdir(parents=True)

        write_zip(colmap_zip, {"scene_a/colmap/sparse/0/images.bin": "colmap"})
        extract_colmap_zip(colmap_zip, task_dir, scene_dir, "scene_a")

        self.assertEqual((scene_dir / "colmap/sparse/0/images.bin").read_text(), "colmap")

    def test_extract_scene_zip_normalizes_rootless_archive(self) -> None:
        work_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_zip = work_root / "dl3dv" / "scene_a.zip"
        task_dir = work_root / "work"

        write_zip(scene_zip, {"transforms.json": "{}", "images/00000.png": "image"})
        scene_dir = extract_scene_zip(scene_zip, task_dir, "scene_a")

        self.assertEqual(scene_dir, task_dir / "scene_a")
        self.assertEqual((scene_dir / "transforms.json").read_text(), "{}")
        self.assertEqual((scene_dir / "images/00000.png").read_text(), "image")

    def test_load_transforms_from_dir_accepts_root_transforms(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = root / "scene_a"
        scene_dir.mkdir()
        (scene_dir / "transforms.json").write_text('{"frames": []}')

        self.assertEqual(load_scene_transforms_from_dir(scene_dir), {"frames": []})

    def test_load_transforms_from_dir_prefers_nerfstudio_transforms(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = root / "scene_a"
        (scene_dir / "nerfstudio").mkdir(parents=True)
        (scene_dir / "transforms.json").write_text('{"source": "root"}')
        (scene_dir / "nerfstudio" / "transforms.json").write_text('{"source": "nerfstudio"}')

        self.assertEqual(load_scene_transforms_from_dir(scene_dir), {"source": "nerfstudio"})

    def test_load_transforms_from_zip_prefers_scene_nerfstudio_layout(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        zip_path = root / "scene_a.zip"
        write_zip(
            zip_path,
            {
                "scene_a/transforms.json": '{"source": "scene"}',
                "scene_a/nerfstudio/transforms.json": '{"source": "nerfstudio"}',
            },
        )

        with ZipFile(zip_path, "r") as zf:
            transforms, scene_root = load_scene_transforms_from_zip(zf, "scene_a")

        self.assertEqual(transforms, {"source": "nerfstudio"})
        self.assertEqual(scene_root, "scene_a")
        self.assertEqual(scene_zip_member(scene_root, "images_4", "00000.png"), "scene_a/images_4/00000.png")

    def test_source_scene_loader_accepts_scene_nerfstudio_layout(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        zip_path = root / "1K" / "scene_a.zip"
        write_zip(
            zip_path,
            {
                "scene_a/nerfstudio/transforms.json": '{"frames": []}',
                "scene_a/images_4/00000.png": "image",
            },
        )

        transforms, prefix = load_source_scene_transforms(root, "1K", "scene_a")

        self.assertEqual(transforms, {"frames": []})
        self.assertEqual(prefix, "scene_a/")


class ReconstructionWorkflowTests(unittest.TestCase):
    def test_half_covisibility_writes_json_outputs(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = root / "scene_a"
        scene_dir.mkdir()
        frames = []
        for index, translation in enumerate(((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))):
            matrix = [
                [1, 0, 0, translation[0]],
                [0, 1, 0, translation[1]],
                [0, 0, 1, translation[2]],
                [0, 0, 0, 1],
            ]
            frames.append({"file_path": f"images/{index:05d}.png", "transform_matrix": matrix})
        (scene_dir / "transforms.json").write_text(json.dumps({"frames": frames}))

        output_dir = root / "out"
        write_half_covisibility_sampled_indices(scene_dir, output_dir)

        for half in ("0", "1"):
            sampled_path = output_dir / f"half_covisibility_sampled_indices_{half}.json"
            self.assertTrue(sampled_path.is_file())
            self.assertEqual(len(json.loads(sampled_path.read_text())), 2)

    def test_reconstruction_outputs_and_complete_marker(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        outputs = reconstruction_outputs(root, "dl3dv_1K", "scene_a", "0", 6)

        self.assertFalse(outputs.complete())
        for path in outputs.paths():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name)

        self.assertTrue(outputs.complete())
        self.assertEqual(outputs.data_h5, root / "dl3dv_1K/scene_a/data_scene_a_0_6.h5")

    def test_copy_reconstruction_outputs_is_complete_and_named(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source = root / "work" / "experiment" / "scene_a"
        source.mkdir(parents=True)
        (source / "data.h5").write_text("data")
        (source / "parsed.yaml").write_text("parsed")
        (source / "ckpt_last.pt").write_text("checkpoint")

        outputs = reconstruction_outputs(root / "out", "dl3dv_1K", "scene_a", "1", 12)
        copy_reconstruction_outputs(source, outputs)

        self.assertTrue(outputs.complete())
        self.assertEqual(outputs.data_h5.read_text(), "data")
        self.assertEqual(outputs.parsed_yaml.read_text(), "parsed")
        self.assertEqual(outputs.checkpoint.read_text(), "checkpoint")

    def test_copy_reconstruction_outputs_prevalidates_all_sources(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source = root / "work" / "experiment" / "scene_a"
        source.mkdir(parents=True)
        (source / "data.h5").write_text("data")
        (source / "parsed.yaml").write_text("parsed")

        outputs = reconstruction_outputs(root / "out", "dl3dv_1K", "scene_a", "1", 12)
        with self.assertRaises(FileNotFoundError):
            copy_reconstruction_outputs(source, outputs)

        for path in outputs.paths():
            self.assertFalse(path.exists())

    def test_find_experiment_scene_dir_prefers_explicit_name(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        task_dir = root / "0@6"
        expected = task_dir / "known_experiment" / "scene_a"
        older = task_dir / "other_experiment" / "scene_a"
        expected.mkdir(parents=True)
        older.mkdir(parents=True)

        self.assertEqual(
            find_experiment_scene_dir(task_dir, "scene_a", expected_experiment_name="known_experiment"),
            expected,
        )

    def test_find_experiment_scene_dir_falls_back_under_explicit_experiment(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        task_dir = root / "0@6"
        fallback = task_dir / "known_experiment" / "rootless_input_name"
        fallback.mkdir(parents=True)

        self.assertEqual(
            find_experiment_scene_dir(task_dir, "scene_a", expected_experiment_name="known_experiment"),
            fallback,
        )

    def test_run_scene_skips_complete_reconstruction_outputs(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene = Scene(
            path=root / "dl3dv" / "1K" / "scene_a.zip",
            name="scene_a",
            dl3dv_subdir="1K",
            reconstruction_subdir="dl3dv_1K",
        )
        scene_work_dir = root / "work" / scene.reconstruction_subdir / scene.name
        scene_work_dir.mkdir(parents=True)

        outputs = reconstruction_outputs(root / "out", scene.reconstruction_subdir, scene.name, "0", 6)
        for path in outputs.paths():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name)

        args = SimpleNamespace(
            output_root=root / "out",
            work_root=root / "work",
            scene_half=["0"],
            num_selected_indices=[6],
            replace_if_exists=False,
            repo_root=root,
        )

        with patch("data_processing.run_sparse_reconstruction.write_half_covisibility_sampled_indices") as sample_mock:
            run_scene(args, scene)

        sample_mock.assert_not_called()

    def test_run_scene_writes_missing_selected_indices(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene = Scene(
            path=root / "dl3dv" / "1K" / "scene_a.zip",
            name="scene_a",
            dl3dv_subdir="1K",
            reconstruction_subdir="dl3dv_1K",
        )
        args = SimpleNamespace(
            output_root=root / "out",
            work_root=root / "work",
            scene_half=["0"],
            num_selected_indices=[6],
            replace_if_exists=False,
            repo_root=root,
        )

        with (
            patch("data_processing.run_sparse_reconstruction.locate_colmap_archive"),
            patch("data_processing.run_sparse_reconstruction.write_half_covisibility_sampled_indices") as sample_mock,
            patch("data_processing.run_sparse_reconstruction.run_one_reconstruction") as run_one_mock,
        ):
            run_scene(args, scene)

        sample_mock.assert_called_once_with(scene.path, root / "work" / scene.reconstruction_subdir / scene.name)
        run_one_mock.assert_called_once()

    def test_sparse_reconstruction_defaults_to_all_scene_zips(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dl3dv_dir = root / "dl3dv"
        write_zip(dl3dv_dir / "2K" / "scene_b.zip", {"scene_b/transforms.json": "{}"})
        write_zip(dl3dv_dir / "1K" / "scene_a.zip", {"scene_a/transforms.json": "{}"})

        with patch(
            "sys.argv",
            [
                "run_sparse_reconstruction.py",
                "--dl3dv_dir",
                str(dl3dv_dir),
                "--output_root",
                str(root / "out"),
                "--work_root",
                str(root / "work"),
                "--num_selected_indices",
                "6",
            ],
        ):
            args = parse_sparse_args()

        self.assertEqual([scene.name for scene in args.scenes], ["scene_a", "scene_b"])
        self.assertEqual([scene.dl3dv_subdir for scene in args.scenes], ["1K", "2K"])


if __name__ == "__main__":
    unittest.main()
