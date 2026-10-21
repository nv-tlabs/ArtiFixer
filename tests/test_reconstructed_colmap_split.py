# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from data_processing.reconstructed_colmap_split import (
    path_for_split,
    read_split_scenes,
    resolve_has_gt,
    resolve_optional_path,
    resolve_required_number,
    resolve_required_path,
    split_entry,
    write_split,
)

PATH_FIELDS = (
    "transforms_path",
    "image_root",
    "render_dir",
    "opacity_dir",
    "selected_indices_path",
    "prompt_path",
    "reconstruction_checkpoint",
)


def scene_paths(scene_root: Path) -> dict[str, Path]:
    return {
        "transforms_path": scene_root / "transforms.json",
        "image_root": scene_root / "images",
        "render_dir": scene_root / "renders",
        "opacity_dir": scene_root / "opacity",
        "selected_indices_path": scene_root / "selected_indices.json",
        "prompt_path": scene_root / "caption.h5",
        "reconstruction_checkpoint": scene_root / "checkpoint.pt",
    }


class PathForSplitTests(unittest.TestCase):
    def test_relativizes_paths_inside_the_split_root(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.assertEqual(
            path_for_split(root / "split.json", root / "scene" / "transforms.json"), "scene/transforms.json"
        )

    def test_keeps_paths_outside_the_split_root_absolute(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        outside = Path(self.enterContext(tempfile.TemporaryDirectory()))
        stored = path_for_split(root / "split.json", outside / "transforms.json")
        self.assertEqual(stored, str((outside / "transforms.json").resolve()))
        self.assertTrue(Path(stored).is_absolute())


class SplitRoundTripTests(unittest.TestCase):
    def test_round_trip_stores_inside_paths_relative_and_resolves_them_back(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        split_path = root / "split.json"
        paths = scene_paths(root / "scene")
        target_indices_path = root / "scene" / "target_indices.json"

        entry = split_entry(
            split_path,
            "scene_a",
            **paths,
            camera_scale=0.5,
            target_indices_path=target_indices_path,
        )
        write_split(split_path, "scene_a", entry)

        scenes = read_split_scenes(split_path)
        self.assertEqual(sorted(scenes), ["scene_a"])
        metadata = scenes["scene_a"]
        self.assertEqual(metadata["scene_id"], "scene_a")

        for field in PATH_FIELDS:
            self.assertEqual(metadata[field], f"scene/{paths[field].name}")
            self.assertEqual(
                resolve_required_path(split_path.parent, metadata, field).resolve(), paths[field].resolve()
            )
        self.assertEqual(metadata["target_indices_path"], "scene/target_indices.json")
        resolved_target = resolve_optional_path(split_path.parent, metadata, "target_indices_path")
        self.assertEqual(resolved_target.resolve(), target_indices_path.resolve())

        camera_scale = resolve_required_number(metadata, "camera_scale")
        self.assertIsInstance(camera_scale, float)
        self.assertEqual(camera_scale, 0.5)

        self.assertNotIn("has_gt", metadata)
        self.assertIs(resolve_has_gt("scene_a", metadata), True)

    def test_round_trip_keeps_outside_paths_absolute_and_resolves_them_back(self) -> None:
        split_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        outside_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        split_path = split_root / "split.json"
        paths = scene_paths(outside_root / "scene")

        entry = split_entry(split_path, "scene_b", **paths, camera_scale=2.0)
        write_split(split_path, "scene_b", entry)

        metadata = read_split_scenes(split_path)["scene_b"]
        for field in PATH_FIELDS:
            self.assertEqual(metadata[field], str(paths[field].resolve()))
            self.assertTrue(Path(metadata[field]).is_absolute())
            self.assertEqual(resolve_required_path(split_path.parent, metadata, field), paths[field].resolve())
        self.assertEqual(resolve_required_number(metadata, "camera_scale"), 2.0)

    def test_round_trip_preserves_has_gt_false_and_omits_optional_target_indices(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        split_path = root / "split.json"

        entry = split_entry(split_path, "scene_c", **scene_paths(root / "scene"), camera_scale=1.0, has_gt=False)
        self.assertIs(entry["has_gt"], False)
        write_split(split_path, "scene_c", entry)

        metadata = read_split_scenes(split_path)["scene_c"]
        self.assertIs(resolve_has_gt("scene_c", metadata), False)
        self.assertNotIn("target_indices_path", metadata)
        self.assertIsNone(resolve_optional_path(split_path.parent, metadata, "target_indices_path"))


if __name__ == "__main__":
    unittest.main()
