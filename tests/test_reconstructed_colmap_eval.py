# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from model_eval.datasets import reconstructed_colmap_eval
from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset
from model_training.data.utils import NeighborSelectionMode
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT


def write_image(path: Path, size: tuple[int, int], mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)


class ReconstructedColmapEvalDatasetTests(unittest.TestCase):
    def test_camera_rays_use_render_resolution_when_context_images_are_smaller(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        image_root = root / "image_root"
        render_dir = root / "renders"
        opacity_dir = root / "opacity"

        write_image(image_root / "images/frame_00000.png", (32, 16))
        write_image(image_root / "images/frame_00001.png", (32, 16))
        write_image(render_dir / "00001.png", (64, 32))
        write_image(opacity_dir / "00001.png", (64, 32), mode="L")

        transform_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        (root / "transforms.json").write_text(
            json.dumps(
                {
                    "w": 64,
                    "h": 32,
                    "fl_x": 40.0,
                    "fl_y": 40.0,
                    "cx": 32.0,
                    "cy": 16.0,
                    "frames": [
                        {"file_path": "images/frame_00000.png", "transform_matrix": transform_matrix},
                        {"file_path": "images/frame_00001.png", "transform_matrix": transform_matrix},
                    ],
                }
            )
        )
        (root / "selected_indices.json").write_text("[0]")
        (root / "caption.h5").write_bytes(b"placeholder")
        (root / "split.json").write_text(
            json.dumps(
                {
                    "test": {
                        "scene": {
                            "transforms_path": "transforms.json",
                            "image_root": "image_root",
                            "render_dir": "renders",
                            "opacity_dir": "opacity",
                            "selected_indices_path": "selected_indices.json",
                            "prompt_path": "caption.h5",
                            "metric_scale": 2.5,
                            "camera_scale": 1.0,
                        }
                    }
                }
            )
        )

        captured_camera_kwargs = {}

        def fake_compute_camera_rays(*, image_shape: tuple[int, int], **kwargs):
            captured_camera_kwargs.update(kwargs)
            h, w = image_shape
            return {"camera_rays": torch.zeros(1, h, w, 6)}

        with (
            patch.object(
                reconstructed_colmap_eval,
                "load_prompt_payload_from_path",
                return_value=({"encoded_prompt": torch.zeros(1, 1, dtype=torch.bfloat16)}, ""),
            ),
            patch.object(reconstructed_colmap_eval, "compute_camera_rays", side_effect=fake_compute_camera_rays),
        ):
            dataset = ReconstructedColmapEvalDataset(
                split="test",
                split_path=root / "split.json",
                num_views=1,
                neighbor_selection_mode=NeighborSelectionMode.CONSECUTIVE,
                max_test_frames=1,
                prompt_format=WAN_T5_PROMPT_FORMAT,
            )
            item = dataset[0]

        self.assertEqual(tuple(item["rgb_neighbors"].shape[-2:]), (16, 32))
        self.assertEqual(tuple(item["rgb_rendered"].shape[-2:]), (32, 64))
        self.assertEqual(tuple(item["camera_rays"].shape[-3:-1]), tuple(item["rgb_rendered"].shape[-2:]))

    def test_first_target_origin_and_reference_rays_reach_getitem(self) -> None:
        from model_training.data.utils import _camera_rays_for_intrinsics, _frame_intrinsics

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        c2ws = torch.eye(4).repeat(5, 1, 1)
        c2ws[:, 0, 3] = torch.arange(5, dtype=torch.float32)
        transforms = {"w": 16, "h": 16, "fl_x": 16.0, "fl_y": 16.0, "cx": 8.0, "cy": 8.0}
        transforms["frames"] = [
            {"file_path": f"images/{index:05d}.png", "transform_matrix": pose.tolist()}
            for index, pose in enumerate(c2ws)
        ]
        transforms["frames"][4].update(fl_x=22.0, k1=0.18, k2=-0.03, p1=0.01, p2=-0.015)
        for index in range(5):
            write_image(root / "image_root" / "images" / f"{index:05d}.png", (16, 16))
            write_image(root / "renders" / f"{index:05d}.png", (16, 16))
            write_image(root / "opacity" / f"{index:05d}.png", (16, 16), mode="L")
        (root / "transforms.json").write_text(json.dumps(transforms))
        (root / "selected_indices.json").write_text("[4]")
        (root / "caption.h5").touch()
        (root / "caption.cosmos3.h5").touch()
        scene = dict(
            transforms_path="transforms.json",
            image_root="image_root",
            render_dir="renders",
            opacity_dir="opacity",
            selected_indices_path="selected_indices.json",
            prompt_path="caption.h5",
            camera_scale=1.0,
        )
        (root / "split.json").write_text(json.dumps({"test": {"scene": scene}}))
        with patch.object(
            reconstructed_colmap_eval,
            "load_prompt_payload_from_path",
            return_value=({"encoded_prompt": torch.zeros(1, 1)}, ""),
        ):
            items = [
                ReconstructedColmapEvalDataset(
                    split="test",
                    split_path=root / "split.json",
                    num_views=1,
                    neighbor_selection_mode=NeighborSelectionMode.EVENLY_SPACED,
                    include_all_frames=True,
                    prompt_format=prompt_format,
                    camera_pose_alignment="last",
                )[0]
                for prompt_format in (WAN_T5_PROMPT_FORMAT, COSMOS3_PROMPT_FORMAT)
            ]
        torch.testing.assert_close(items[0]["w2cs"][0], torch.eye(4))
        torch.testing.assert_close(items[0]["neighbor_w2cs"][:, 0, 3], torch.tensor([-4.0]))
        self.assertNotIn("neighbor_camera_rays", items[0])
        joint = items[1]
        for key in ("camera_rays", "w2cs", "Ks", "neighbor_w2cs", "neighbor_Ks"):
            torch.testing.assert_close(joint[key], items[0][key])
        self.assertEqual(joint["neighbor_camera_rays"].shape, (1, 16, 16, 6))
        rows = _frame_intrinsics(transforms, transforms["frames"][4]).unsqueeze(0)
        c2ws_ref = torch.linalg.inv(joint["neighbor_w2cs"])
        expected = _camera_rays_for_intrinsics(rows, c2ws_ref, 16, 16)
        torch.testing.assert_close(joint["neighbor_camera_rays"], expected)
        rows[:, 4:] = 0
        self.assertFalse(torch.allclose(expected, _camera_rays_for_intrinsics(rows, c2ws_ref, 16, 16)))

    def test_cosmos3_format_derives_the_caption_filename_from_the_split_recorded_name(self) -> None:
        """The split file records the base (Wan-named) caption file; the
        prompt format picks the actual file in the same directory."""
        from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (root / "image_root").mkdir()
        (root / "renders").mkdir()
        (root / "opacity").mkdir()
        (root / "transforms.json").write_text(json.dumps({"frames": [{"file_path": "a.png"}, {"file_path": "b.png"}]}))
        (root / "selected_indices.json").write_text("[0]")
        (root / "caption.cosmos3.h5").write_bytes(b"placeholder")
        (root / "split.json").write_text(
            json.dumps(
                {
                    "test": {
                        "scene": {
                            "transforms_path": "transforms.json",
                            "image_root": "image_root",
                            "render_dir": "renders",
                            "opacity_dir": "opacity",
                            "selected_indices_path": "selected_indices.json",
                            "prompt_path": "caption.h5",
                            "camera_scale": 1.0,
                        }
                    }
                }
            )
        )

        with patch.object(ReconstructedColmapEvalDataset, "_generate_inference_items"):
            dataset = ReconstructedColmapEvalDataset(
                split="test",
                split_path=root / "split.json",
                num_views=1,
                prompt_format=COSMOS3_PROMPT_FORMAT,
            )

        self.assertEqual(dataset.scenes_by_scene_id["scene"].prompt_path.name, "caption.cosmos3.h5")


if __name__ == "__main__":
    unittest.main()
