# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from data_processing.run_sparse_reconstruction import (
    EXTRACTION_SENTINEL,
    MissingColmapModelError,
    Scene,
    locate_colmap_archive,
    prepare_scene_input,
    run_scenes,
)
from data_processing.scene_utils import COLMAP_SPARSE_FILES

SCENE_ID = "0123456789abcdef"


def write_zip(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def sparse_members(prefix: str, names: tuple[str, ...] = COLMAP_SPARSE_FILES) -> dict[str, bytes]:
    return {f"{prefix}/{name}": b"" for name in names}


def write_scene_zip(root: Path, with_model: bool = False) -> Path:
    members = {f"{SCENE_ID}/images_4/frame_00001.png": b"", f"{SCENE_ID}/transforms.json": b"{}"}
    if with_model:
        members.update(sparse_members(f"{SCENE_ID}/sparse/0"))
    return write_zip(root / "8K" / f"{SCENE_ID}.zip", members)


def write_colmap_zip(
    root: Path, names: tuple[str, ...] = COLMAP_SPARSE_FILES, sparse_prefix: str = f"{SCENE_ID}/colmap/sparse/0"
) -> Path:
    members = {f"{SCENE_ID}/transforms.json": b"{}", **sparse_members(sparse_prefix, names)}
    return write_zip(root / "8K" / f"{SCENE_ID}.zip", members)


def scene_for(path: Path) -> Scene:
    return Scene(path=path, name=SCENE_ID, dl3dv_subdir="8K", reconstruction_subdir="dl3dv_8K")


class PrepareSceneInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.scene = scene_for(write_scene_zip(self.root / "dl3dv"))
        self.colmap_root = self.root / "cache"
        self.colmap_root.mkdir()
        self.work_dir = self.root / "work"
        self.work_dir.mkdir()

    def args(self, colmap_zip_root: Path | None) -> argparse.Namespace:
        return argparse.Namespace(colmap_zip_root=colmap_zip_root)

    def assert_nothing_extracted(self) -> None:
        self.assertEqual(list(self.work_dir.iterdir()), [])

    def test_stages_colmap_model_next_to_scene_images(self) -> None:
        write_colmap_zip(self.colmap_root)

        scene_dir = prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir)

        self.assertTrue((scene_dir / "images_4" / "frame_00001.png").is_file())
        self.assertTrue((scene_dir / "colmap" / "sparse" / "0" / "points3D.bin").is_file())
        self.assertTrue((self.work_dir / "source" / EXTRACTION_SENTINEL).is_file())

    def test_fails_without_colmap_model_or_cache_root(self) -> None:
        with self.assertRaisesRegex(MissingColmapModelError, "--colmap_zip_root"):
            prepare_scene_input(self.args(None), self.scene, self.work_dir)

        self.assert_nothing_extracted()

    def test_fails_when_cache_has_no_archive_for_the_scene(self) -> None:
        with self.assertRaisesRegex(MissingColmapModelError, f"{SCENE_ID}: no COLMAP archive under"):
            prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir)

        self.assert_nothing_extracted()

    def test_fails_when_cache_archive_has_no_model(self) -> None:
        write_colmap_zip(self.colmap_root, names=())

        with self.assertRaisesRegex(MissingColmapModelError, "lacks cameras.bin, images.bin, points3D.bin"):
            prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir)

        self.assert_nothing_extracted()

    def test_fails_when_cache_archive_is_incomplete(self) -> None:
        write_colmap_zip(self.colmap_root, names=("cameras.bin", "images.bin"))

        with self.assertRaisesRegex(MissingColmapModelError, "lacks points3D.bin"):
            prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir)

        self.assert_nothing_extracted()

    def test_fails_when_cache_archive_is_not_a_zip(self) -> None:
        (self.colmap_root / "8K").mkdir()
        (self.colmap_root / "8K" / f"{SCENE_ID}.zip").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")

        with self.assertRaisesRegex(MissingColmapModelError, "is not a valid zip archive"):
            prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir)

        self.assert_nothing_extracted()

    def test_accepts_rootless_cache_archive(self) -> None:
        write_zip(self.colmap_root / "8K" / f"{SCENE_ID}.zip", sparse_members("colmap/sparse/0"))

        scene_dir = prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir)

        self.assertTrue((scene_dir / "colmap" / "sparse" / "0" / "points3D.bin").is_file())

    def test_rejects_model_nested_below_the_accepted_positions(self) -> None:
        write_colmap_zip(self.colmap_root, sparse_prefix=f"{SCENE_ID}/extra/colmap/sparse/0")

        with self.assertRaisesRegex(MissingColmapModelError, "lacks cameras.bin, images.bin, points3D.bin"):
            prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir)

        self.assert_nothing_extracted()

    def test_does_not_pool_partial_models_across_directories(self) -> None:
        write_zip(
            self.colmap_root / "8K" / f"{SCENE_ID}.zip",
            {
                **sparse_members(f"{SCENE_ID}/colmap/sparse/0", ("cameras.bin", "images.bin")),
                **sparse_members(f"{SCENE_ID}/sparse/0", ("points3D.bin",)),
            },
        )

        with self.assertRaisesRegex(MissingColmapModelError, "lacks points3D.bin"):
            prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir)

        self.assert_nothing_extracted()

    def test_reuses_completed_extraction_without_reading_archives(self) -> None:
        write_colmap_zip(self.colmap_root)
        scene_dir = prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir)

        with mock.patch("data_processing.run_sparse_reconstruction.locate_colmap_archive") as locate:
            self.assertEqual(prepare_scene_input(self.args(self.colmap_root), self.scene, self.work_dir), scene_dir)

        locate.assert_not_called()

    def test_scene_zip_with_its_own_model_does_not_need_a_cache_archive(self) -> None:
        scene = scene_for(write_scene_zip(self.root / "dl3dv_with_model", with_model=True))

        scene_dir = prepare_scene_input(self.args(self.colmap_root), scene, self.work_dir)

        self.assertTrue((scene_dir / "sparse" / "0" / "images.bin").is_file())

    def test_extracted_scene_directory_must_carry_its_own_model(self) -> None:
        scene_dir = self.root / "extracted" / SCENE_ID
        (scene_dir / "images_4").mkdir(parents=True)

        with self.assertRaisesRegex(MissingColmapModelError, "lacks cameras.bin, images.bin, points3D.bin"):
            locate_colmap_archive(self.args(self.colmap_root), scene_for(scene_dir))


class RunScenesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "reconstructions"

    @staticmethod
    def scenes(*names: str, subdir: str = "8K") -> list[Scene]:
        return [
            Scene(path=Path(f"{name}.zip"), name=name, dl3dv_subdir=subdir, reconstruction_subdir=f"dl3dv_{subdir}")
            for name in names
        ]

    @staticmethod
    def failing(*names: str):
        def fake_run_scene(args: argparse.Namespace, scene: Scene) -> None:
            if scene.name in names:
                raise MissingColmapModelError(f"{scene.name}: no COLMAP archive")

        return fake_run_scene

    def skipped_list(self, subdir: str = "8K") -> Path:
        return self.output_root / f"dl3dv_{subdir}" / "skipped_scenes.txt"

    def run_scenes(self, scenes: list[Scene], *failing: str, explicit: bool = False) -> mock.MagicMock:
        with mock.patch(
            "data_processing.run_sparse_reconstruction.run_scene", side_effect=self.failing(*failing)
        ) as run_scene:
            run_scenes(argparse.Namespace(scenes=scenes, explicit_scenes=explicit, output_root=self.output_root))
        return run_scene

    def test_skipped_lists_track_scans_only(self) -> None:
        run_scene = self.run_scenes(self.scenes("a", "b", "c"), "b")
        self.assertEqual(run_scene.call_count, 3)
        self.assertEqual(self.skipped_list().read_text(), "b\n")

        self.run_scenes(self.scenes("d", "e", subdir="10K"), "e")
        self.assertEqual(self.skipped_list().read_text(), "b\n")
        self.assertEqual(self.skipped_list("10K").read_text(), "e\n")

        self.run_scenes(self.scenes("b"), explicit=True)
        self.assertEqual(self.skipped_list().read_text(), "b\n")
        self.assertEqual(self.skipped_list("10K").read_text(), "e\n")

        self.run_scenes(self.scenes("a", "b", "c"))
        self.assertEqual(self.skipped_list().read_text(), "")
        self.assertEqual(self.skipped_list("10K").read_text(), "e\n")

    def test_explicitly_requested_scene_without_colmap_model_fails(self) -> None:
        with self.assertRaises(MissingColmapModelError):
            self.run_scenes(self.scenes("a", "b"), "a", explicit=True)

        self.assertFalse(self.skipped_list().exists())


if __name__ == "__main__":
    unittest.main()
