# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from model_eval.datasets.dl3dv_reconstruction_eval import DL3DVReconstructionEvalDataset
from model_eval.inference import get_output_dir
from model_eval.inference_args import add_dl3dv_args, validate_evalset_args
from model_eval.prepare_dl3dv_distill_dataset import scene_items_from_split
from model_training.data.utils import NeighborSelectionMode
from model_training.prompt_formats import WAN_T5_PROMPT_FORMAT


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_scene_inputs(tmp_dir: Path, scene_id: str) -> None:
    """Source zip, reconstruction checkpoint outputs, and canonical prompt for one scene."""
    transforms = {
        "frames": [
            {"file_path": f"images/frame_{index:05d}.png", "transform_matrix": [[1, 0, 0, index]]} for index in range(3)
        ]
    }
    transforms_zip = tmp_dir / "DL3DV-ALL-960P" / "2K" / f"{scene_id}.zip"
    transforms_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(transforms_zip, "w") as archive:
        archive.writestr(f"{scene_id}/transforms.json", json.dumps(transforms))
    checkpoint_dir = tmp_dir / "recon" / scene_id / "benchmark_distill" / scene_id / "ours_30000"
    _write_json(checkpoint_dir / "selected_indices.json", [0])
    (checkpoint_dir / "renders").mkdir(parents=True)
    (checkpoint_dir / "opacity").mkdir(parents=True)
    for index in range(3):
        (checkpoint_dir / "renders" / f"{index}.png").write_text("render")
        (checkpoint_dir / "opacity" / f"{index}.png").write_text("opacity")
    prompt_dir = tmp_dir / "prompts" / "2K" / scene_id
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "frames_all_stride_1.h5").write_text("prompt")


def _write_split(tmp_dir: Path, scene_ids: tuple[str, ...]) -> None:
    _write_json(
        tmp_dir / "split.json", {"test": {scene_id: [{"subdir": "2K", "scale": 1.0}] for scene_id in scene_ids}}
    )


def _make_dataset(tmp_dir: Path, **overrides) -> DL3DVReconstructionEvalDataset:
    kwargs = dict(
        split="test",
        split_path=tmp_dir / "split.json",
        dl3dv_dir=tmp_dir / "DL3DV-ALL-960P",
        recon_results_dir=tmp_dir / "recon",
        prompt_dir=tmp_dir / "prompts",
        num_views=1,
        dataset_scaling_factor=1.0,
        checkpoint="30000",
        neighbor_selection_mode=NeighborSelectionMode.COVISIBILITY,
        include_all_frames=True,
        recon_subdir="benchmark_distill",
        prompt_format=WAN_T5_PROMPT_FORMAT,
    )
    kwargs.update(overrides)
    return DL3DVReconstructionEvalDataset(**kwargs)


class DL3DVReconstructionEvalTests(unittest.TestCase):
    def test_filters_to_multiple_requested_scenes(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_ids = ("scene_a", "scene_b", "scene_c")
        for scene_id in scene_ids:
            _write_scene_inputs(tmp_dir, scene_id)
        _write_split(tmp_dir, scene_ids)

        with patch.object(DL3DVReconstructionEvalDataset, "_generate_inference_items"):
            dataset = _make_dataset(tmp_dir, filter_scene_ids=("scene_a", "scene_c"))

        self.assertEqual(dataset.scene_ids, ["scene_a", "scene_c"])

    def test_pose_alignment_survives_reconstruction_constructor(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _write_scene_inputs(tmp_dir, "scene_a")
        _write_split(tmp_dir, ("scene_a",))
        for pose_alignment in ("last", "mean"):
            with self.subTest(pose_alignment=pose_alignment):
                with patch.object(DL3DVReconstructionEvalDataset, "_generate_inference_items"):
                    dataset = _make_dataset(tmp_dir, camera_pose_alignment=pose_alignment)
                self.assertEqual(dataset.camera_pose_alignment, pose_alignment)

    def test_registers_only_the_canonical_prompt_when_variants_exist(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _write_scene_inputs(tmp_dir, "scene_a")
        _write_split(tmp_dir, ("scene_a",))
        prompt_dir = tmp_dir / "prompts" / "2K" / "scene_a"
        for variant in (
            "frames_all_stride_1_reverse.h5",
            "frames_all_stride_1_revisit.h5",
            "frames_81_stride_1.h5",
        ):
            (prompt_dir / variant).write_text("variant")

        with patch.object(DL3DVReconstructionEvalDataset, "_generate_inference_items"):
            dataset = _make_dataset(tmp_dir)

        self.assertEqual(
            dataset.scenes_by_scene_id["scene_a"].prompt_path,
            prompt_dir / "frames_all_stride_1.h5",
        )

    def test_cosmos3_format_pins_the_format_distinguished_prompt(self):
        from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT

        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _write_scene_inputs(tmp_dir, "scene_a")
        _write_split(tmp_dir, ("scene_a",))
        prompt_dir = tmp_dir / "prompts" / "2K" / "scene_a"
        (prompt_dir / "frames_all_stride_1.cosmos3.h5").write_text("cosmos3 prompt")

        with patch.object(DL3DVReconstructionEvalDataset, "_generate_inference_items"):
            dataset = _make_dataset(tmp_dir, prompt_format=COSMOS3_PROMPT_FORMAT)

        self.assertEqual(
            dataset.scenes_by_scene_id["scene_a"].prompt_path,
            prompt_dir / "frames_all_stride_1.cosmos3.h5",
        )

    def test_missing_cosmos3_prompt_raises_even_when_the_wan_file_exists(self):
        from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT

        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _write_scene_inputs(tmp_dir, "scene_a")
        _write_split(tmp_dir, ("scene_a",))

        with patch.object(DL3DVReconstructionEvalDataset, "_generate_inference_items"):
            with self.assertRaisesRegex(FileNotFoundError, "frames_all_stride_1.cosmos3.h5"):
                _make_dataset(tmp_dir, prompt_format=COSMOS3_PROMPT_FORMAT)

    def test_missing_canonical_prompt_raises_even_without_scene_filter(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _write_scene_inputs(tmp_dir, "scene_a")
        _write_split(tmp_dir, ("scene_a",))
        prompt_dir = tmp_dir / "prompts" / "2K" / "scene_a"
        (prompt_dir / "frames_all_stride_1.h5").unlink()
        (prompt_dir / "frames_all_stride_1_reverse.h5").write_text("variant")

        with patch.object(DL3DVReconstructionEvalDataset, "_generate_inference_items"):
            with self.assertRaisesRegex(FileNotFoundError, "canonical evaluation prompt"):
                _make_dataset(tmp_dir)

    def test_multi_scene_filter_rejects_ids_absent_from_split(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _write_split(tmp_dir, ("scene_a",))

        with self.assertRaisesRegex(ValueError, "scene_missing"):
            _make_dataset(tmp_dir, filter_scene_ids=("scene_a", "scene_missing"))

    def test_scene_filter_arguments_are_mutually_exclusive(self):
        parser = argparse.ArgumentParser()
        add_dl3dv_args(parser)

        with self.assertRaises(SystemExit):
            parser.parse_args(["--scene_id", "scene_a", "--scene_ids", "scene_a", "scene_b"])

    def test_scene_metadata_preserves_metric_scale_for_camera_conditioning(self):
        split_data = {
            "scene_a": [
                {"subdir": "2K", "scale": 2.5, "scene_half": "first", "num_selected_indices": 3},
                {"subdir": "2K", "scale": 2.5, "scene_half": "second", "num_selected_indices": 3},
            ]
        }

        self.assertEqual(
            scene_items_from_split(split_data, include_metric_scale=True),
            [("scene_a", "2K", 2.5)],
        )

    def test_scene_metadata_raises_on_genuinely_disagreeing_entries(self):
        # Entries disagreeing beyond reconstruction noise mean a corrupt
        # split; guessing a scale would silently mis-condition the eval.
        split_data = {
            "scene_a": [
                {"subdir": "2K", "scale": 2.5},
                {"subdir": "4K", "scale": 3.0},
            ]
        }

        with self.assertRaisesRegex(ValueError, "scene_a"):
            scene_items_from_split(split_data, include_metric_scale=True)

    def test_scene_metadata_accepts_metric_scale_reconstruction_noise(self):
        # Per-half reconstructions align scale independently; the production
        # split's worst observed spread is 9e-4, accepted silently.
        split_data = {
            "scene_a": [
                {"subdir": "2K", "scale": 2.5},
                {"subdir": "2K", "scale": 2.5023},
            ]
        }

        self.assertEqual(
            scene_items_from_split(split_data, include_metric_scale=True),
            [("scene_a", "2K", 2.5)],
        )

    def test_skips_extracted_benchmark_scene_when_public_zip_is_missing(self):
        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_id = "scene_a"
        _write_scene_inputs(tmp_dir, scene_id)
        _write_split(tmp_dir, (scene_id,))
        # Replace the public zip with the extracted benchmark layout the loader must NOT fall back to.
        (tmp_dir / "DL3DV-ALL-960P" / "2K" / f"{scene_id}.zip").unlink()
        transforms_root = tmp_dir / "DL3DV-ALL-960P" / scene_id / "nerfstudio"
        _write_json(
            transforms_root / "transforms.json",
            {
                "frames": [
                    {"file_path": f"images/frame_{index:05d}.png", "transform_matrix": [[1, 0, 0, index]]}
                    for index in range(3)
                ]
            },
        )
        (transforms_root / "images_4").mkdir(parents=True)

        dataset = _make_dataset(tmp_dir, resize_multiple=32)

        self.assertNotIn(scene_id, dataset.scenes_by_scene_id)
        # The Wan2.2 resize multiple must survive the constructor chain.
        self.assertEqual(dataset.resize_multiple, 32)

    def test_run_inference_uses_auto_num_views_for_dl3dv_reconstruction_evalsets(self):
        args = Namespace(
            evalset="artifixer3d_dl3dv_ours",
            bidirectional_chunk_size=81,
            output_fps=15,
            num_views=None,
            split_path=Path("/splits/trainval_test_split.json"),
            dl3dv_dir=Path("/data/DL3DV-ALL-960P"),
            prompt_dir=Path("/data/prompts"),
            recon_results_dir=Path("/runs/reconstruction"),
            render_trajectory="all_frames",
            render_diagnostics=False,
            context_parallel_size=1,
            save_dir=Path("/eval"),
            render_name="artifixer-14b_chunk7_sink7",
            neighbor_selection_mode="evenly_spaced",
            output_suffix="",
            scene_id=None,
            scene_ids=None,
        )

        validate_evalset_args(argparse.ArgumentParser(), args)

        self.assertIsNone(args.num_views)
        self.assertEqual(
            get_output_dir(args),
            Path(
                "/eval/artifixer-14b_chunk7_sink7/"
                "distilled_views_artifixer3d_dl3dv_ours_auto12_evenly_spaced_all_frames"
            ),
        )

    def test_output_dir_uses_the_supplied_render_name_verbatim(self):
        """--render_name is the run's whole identity: get_output_dir roots the
        outputs at exactly the supplied name and derives nothing from the
        checkpoint or pipeline knobs (which no longer reach it at all)."""
        args = Namespace(
            evalset="nerfbusters",
            num_views=6,
            render_trajectory="val_frames",
            save_dir=Path("/eval"),
            render_name="any-opaque-name",
            neighbor_selection_mode="evenly_spaced",
            output_suffix="3dplus",
        )

        self.assertEqual(
            get_output_dir(args),
            Path("/eval/any-opaque-name/nerfbusters_6_evenly_spaced_3dplus"),
        )

    def test_run_inference_rejects_multi_scene_filter_for_other_evalsets(self):
        args = Namespace(
            evalset="nerfbusters",
            model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            bidirectional_chunk_size=81,
            output_fps=15,
            num_views=6,
            scene_id=None,
            scene_ids=["scene_a", "scene_b"],
        )

        with self.assertRaises(SystemExit):
            validate_evalset_args(argparse.ArgumentParser(), args)


if __name__ == "__main__":
    unittest.main()
