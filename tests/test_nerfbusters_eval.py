# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from data_processing.nerfbusters import nerfbusters_downsample_factor, nerfbusters_gt_relpath, nerfbusters_image_folder

try:
    # Production types — importable only where model_training's heavy deps
    # (torch, threedgrut, ...) are installed.
    from model_training.data.utils import InferencePair, NeighborSelectionMode
except ImportError:
    from enum import Enum

    # Fallback copies of the types in model_training/data/utils.py for
    # environments without those deps; keep the fields in sync.
    @dataclass
    class InferencePair:
        neighbor_indices: list[int]
        test_indices: list[int]
        reversed: bool
        chunk_idx: int
        scene_id: str = ""
        is_test_frame: list[bool] = None

    class NeighborSelectionMode(Enum):
        CONSECUTIVE = "consecutive"
        EVENLY_SPACED = "evenly_spaced"
        COVISIBILITY = "covisibility"


@contextmanager
def patched_nerfbusters_eval_dataset():
    utils_module = types.ModuleType("model_training.data.utils")
    utils_module.InferencePair = InferencePair
    utils_module.NeighborSelectionMode = NeighborSelectionMode
    utils_module.compute_camera_rays = lambda *args, **kwargs: {}
    utils_module.generate_inference_pairs = lambda *args, **kwargs: []
    utils_module.load_prompt_payload_from_path = lambda *args, **kwargs: ({"encoded_prompt": None}, "")
    utils_module.load_indexed_frames = lambda *args, **kwargs: None
    utils_module.resize_to_multiple = lambda value, multiple: value

    torch_module = types.ModuleType("torch")
    torch_utils_module = types.ModuleType("torch.utils")
    torch_data_module = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    torch_data_module.Dataset = Dataset
    torch_utils_module.data = torch_data_module
    torch_module.utils = torch_utils_module
    torch_module.Generator = object
    torch_module.Tensor = object

    old_utils = sys.modules.get("model_training.data.utils")
    old_torch = sys.modules.get("torch")
    old_torch_utils = sys.modules.get("torch.utils")
    old_torch_data = sys.modules.get("torch.utils.data")
    old_dataset = sys.modules.pop("model_eval.datasets.nerfbusters_eval", None)
    sys.modules["model_training.data.utils"] = utils_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils_module
    sys.modules["torch.utils.data"] = torch_data_module
    try:
        module = importlib.import_module("model_eval.datasets.nerfbusters_eval")
        yield module.NerfbustersEvalDataset
    finally:
        sys.modules.pop("model_eval.datasets.nerfbusters_eval", None)
        if old_dataset is not None:
            sys.modules["model_eval.datasets.nerfbusters_eval"] = old_dataset
        if old_utils is None:
            sys.modules.pop("model_training.data.utils", None)
        else:
            sys.modules["model_training.data.utils"] = old_utils
        for module_name, old_module in (
            ("torch", old_torch),
            ("torch.utils", old_torch_utils),
            ("torch.utils.data", old_torch_data),
        ):
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module


class NerfbustersEvalTests(unittest.TestCase):
    def test_shared_resolution_mapping_matches_paper_eval_resolution(self):
        self.assertEqual(nerfbusters_downsample_factor("aloe"), 2)
        self.assertEqual(nerfbusters_image_folder("aloe"), "images_2")
        self.assertEqual(nerfbusters_gt_relpath("aloe", "images/frame_00000.png"), "images_2/frame_00000.png")
        self.assertEqual(nerfbusters_downsample_factor("flowers"), 1)
        self.assertEqual(nerfbusters_image_folder("flowers"), "images")
        self.assertEqual(nerfbusters_gt_relpath("flowers", "images/frame_00000.png"), "images/frame_00000.png")

    def test_dataset_uses_shared_resolution_mapping_by_default(self):
        with patched_nerfbusters_eval_dataset() as NerfbustersEvalDataset:
            dataset = object.__new__(NerfbustersEvalDataset)
            dataset.image_folder = None

            self.assertEqual(dataset._get_image_folder_for_scene("car"), "images_2")
            self.assertEqual(dataset._get_image_folder_for_scene("flowers"), "images")

            dataset.image_folder = "images_4"
            self.assertEqual(dataset._get_image_folder_for_scene("car"), "images_4")

    def test_captions_use_the_format_derived_filename(self):
        from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT

        tmp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        scene_dir = tmp_dir / "nerfbusters" / "aloe"
        scene_dir.mkdir(parents=True)
        (scene_dir / "transforms.json").write_text(json.dumps({"frames": []}))
        render_dir = tmp_dir / "recon" / "aloe" / "exp" / "aloe" / "ours_30000" / "renders"
        render_dir.mkdir(parents=True)
        captions_dir = tmp_dir / "captions"
        captions_dir.mkdir()
        (captions_dir / "aloe.h5").write_text("caption")
        (captions_dir / "aloe.cosmos3.h5").write_text("caption")

        def make_dataset(NerfbustersEvalDataset, **kwargs):
            return NerfbustersEvalDataset(
                nerfbusters_dir=tmp_dir / "nerfbusters",
                recon_results_dir=tmp_dir / "recon",
                captions_dir=captions_dir,
                num_views=2,
                recon_experiment_name="exp",
                **kwargs,
            )

        with patched_nerfbusters_eval_dataset() as NerfbustersEvalDataset:
            wan_dataset = make_dataset(NerfbustersEvalDataset, prompt_format=WAN_T5_PROMPT_FORMAT)
            cosmos3_dataset = make_dataset(NerfbustersEvalDataset, prompt_format=COSMOS3_PROMPT_FORMAT)

        # Wan reads the historical bare name; cosmos3 reads the
        # format-distinguished file beside it in the same tree.
        self.assertEqual(wan_dataset.captions_by_scene_id, {"aloe": captions_dir / "aloe.h5"})
        self.assertEqual(cosmos3_dataset.captions_by_scene_id, {"aloe": captions_dir / "aloe.cosmos3.h5"})

    def test_train_test_split_uses_current_nerfbusters_frame_names(self):
        from model_eval.datasets.nerfbusters_split import nerfbusters_train_test_split

        transforms = {
            "frames": [
                {"file_path": "images/frame_00000.png"},
                {"file_path": "images/frame_1_00001.png"},
                {"file_path": "images/frame_00002.png"},
            ]
        }

        train_ids, test_ids = nerfbusters_train_test_split(transforms)

        self.assertEqual(train_ids, {0, 2})
        self.assertEqual(test_ids, {1})

    def test_inference_pair_is_split_by_frame_number_gaps(self):
        with patched_nerfbusters_eval_dataset() as NerfbustersEvalDataset:
            dataset = object.__new__(NerfbustersEvalDataset)
        transforms = {
            "frames": [
                {"file_path": "images/frame_1_00010.png"},
                {"file_path": "images/frame_1_00011.png"},
                {"file_path": "images/frame_1_00020.png"},
            ]
        }
        pair = InferencePair(
            neighbor_indices=[0],
            test_indices=[2, 0, 1],
            reversed=False,
            chunk_idx=3,
            scene_id="art",
            is_test_frame=[True, True, True],
        )

        split_pairs = dataset._split_pair_by_frame_number_gaps(transforms, pair)

        self.assertEqual([[0, 1], [2]], [list(p.test_indices) for p in split_pairs])
        self.assertEqual([3000, 3001], [p.chunk_idx for p in split_pairs])

    def test_nerfbusters_metrics_read_exactly_the_supplied_render_name(self):
        """The metric script takes the render run's --render_name as an opaque
        directory under --eval_base_path; the only path parts it adds are the
        dataset facets (num_views, neighbor mode, output suffix — the suffix
        keeping 3dplus reruns apart from direct eval)."""
        try:
            from model_eval.compute_metrics_nerfbusters import main as compute_nerfbusters_metrics
        except ImportError as error:  # pragma: no cover - login nodes without torch
            self.skipTest(f"compute_metrics_nerfbusters needs metric dependencies: {error}")

        with self.assertRaisesRegex(
            FileNotFoundError,
            r"/missing-eval/artifixer-14b_chunk7_sink7/nerfbusters_3_evenly_spaced_3dplus$",
        ):
            compute_nerfbusters_metrics(
                "artifixer-14b_chunk7_sink7",
                3,
                eval_base_path=Path("/missing-eval"),
                nerfbusters_dir=Path("/missing-data"),
                use_masks=False,
                output_suffix="3dplus",
            )


@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.requires_host_ffmpeg
def test_real_nerfbusters_dataset_emits_joint_reference_rays_from_full_camera_records(tmp_path, reordered, overlap):
    import h5py
    import numpy as np
    import torch

    from model_eval.datasets.nerfbusters_eval import NerfbustersEvalDataset
    from model_training.data.utils import _camera_rays_for_intrinsics, _frame_intrinsics
    from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT

    scene_dir = tmp_path / "data/art"
    image_dir = scene_dir / "images"
    image_dir.mkdir(parents=True)
    checkpoint_dir = tmp_path / "recon/art/exp/art/ours_30000"
    (checkpoint_dir / "renders").mkdir(parents=True)
    (checkpoint_dir / "opacity").mkdir()
    captions_dir = tmp_path / "captions"
    captions_dir.mkdir()
    transforms = dict(w=16, h=16, fl_x=20.0, fl_y=18.0, cx=8.0, cy=8.0, frames=[])
    for index in range(5):
        name = f"frame_{index:05d}.png" if index in (0, 4) else f"frame_1_{index:05d}.png"
        if overlap and index == 4:
            name = "frame_00001.png"
        c2w = torch.eye(4)
        c2w[0, 3] = float(index)
        transforms["frames"].append(
            dict(
                file_path=f"images/{name}",
                transform_matrix=c2w.tolist(),
                fl_x=20.0 + index,
                k1=0.15 + 0.02 * index,
                k2=-0.03,
                p1=0.01,
                p2=-0.015,
            )
        )
        Image.new("RGB", (16, 16), (30 * index, 10, 20)).save(image_dir / name)
        Image.new("RGB", (16, 16)).save(checkpoint_dir / "renders" / f"{index:05d}.png")
        Image.new("L", (16, 16), 255).save(checkpoint_dir / "opacity" / f"{index:05d}.png")
    if reordered:
        transforms["frames"] = [transforms["frames"][i] for i in (4, 1, 3, 0, 2)]
    (scene_dir / "transforms.json").write_text(json.dumps(transforms))
    with h5py.File(captions_dir / "art.h5", "w") as file:
        record = file.create_dataset("frame_00000.png", data=np.zeros((4, 8), dtype=np.uint16))
        record.attrs["caption"] = "test scene"
    with h5py.File(captions_dir / "art.cosmos3.h5", "w") as file:
        file.attrs["pad_token_id"] = 0
        file.attrs["negative_input_ids"] = np.array([1, 2], dtype=np.int64)
        record = file.create_dataset("frame_00000.png", data=np.array([3, 4], dtype=np.int64))
        record.attrs["caption"] = "test scene"
    kwargs = dict(
        nerfbusters_dir=tmp_path / "data",
        recon_results_dir=tmp_path / "recon",
        captions_dir=captions_dir,
        num_views=2,
        recon_experiment_name="exp",
        include_all_frames=True,
        prompt_format=WAN_T5_PROMPT_FORMAT,
    )
    legacy = NerfbustersEvalDataset(**kwargs)[0]
    joint = NerfbustersEvalDataset(**(kwargs | dict(prompt_format=COSMOS3_PROMPT_FORMAT)))[0]
    assert joint["video_frame_indices"].tolist() == [0, 1, 2, 3, 4]
    if reordered:
        assert joint["frame_indices"].tolist() == ([3, 0, 1, 4, 2] if overlap else [3, 1, 4, 2, 0])
    if overlap:
        chunked = NerfbustersEvalDataset(**(kwargs | dict(max_test_frames=2)))
        chunk_ids = [index for item in chunked for index in item["video_frame_indices"].tolist()]
        assert sorted(chunk_ids) == [0, 1, 2, 3, 4]
        from model_eval import inference
        from model_training.utils.video_io import decode_video_to_frames, video_file_complete
        from tests.test_run_inference_pipeline_selection import _eval_args, _eval_item, _NoiseEvalPipe

        item = _eval_item(2, 16, cosmos3=False)
        for name in ("scene_id", "frame_indices", "gt_index", "video_frame_indices"):
            item[name] = legacy[name]
        item["rgb_rendered"] = legacy["rgb_gt"]
        args = _eval_args("Wan-AI/Wan2.1-T2V-14B-Diffusers", save_frame_outputs_only=False)
        args.inference_pipeline = "bidirectional"
        output = tmp_path / "inference"
        inference.process_item(_NoiseEvalPipe(), item, args, output, 0, torch.device("cpu"), 4)
        inference.finalize_video_outputs(output, 15)
        movie = output / "art/videos/batch_0000_rendered.mp4"
        assert video_file_complete(movie, 5)
        decoded = decode_video_to_frames(movie.read_bytes())
        expected = legacy["rgb_gt"].mean(dim=(2, 3)).numpy() * 255
        np.testing.assert_allclose(decoded.mean(axis=(1, 2)), expected, atol=5)
        for position, source_id in enumerate(legacy["gt_index"].tolist()):
            if source_id >= 0:
                metric = output / f"art/frames/batch_0000/rendered/{source_id:05d}.png"
                np.testing.assert_allclose(np.asarray(Image.open(metric)).mean(axis=(0, 1)), expected[position], atol=1)
        pipe = _NoiseEvalPipe()
        for chunk in chunked:
            assert inference.output_video_frames_complete(output / "art", chunk)
            inference.process_item(pipe, chunk, args, output, 0, torch.device("cpu"), 4)
        assert pipe.events == []
    assert "neighbor_camera_rays" not in legacy
    for name in ("camera_rays", "w2cs", "Ks", "neighbor_w2cs", "neighbor_Ks"):
        torch.testing.assert_close(joint[name], legacy[name])
    intrinsic_rows = torch.stack(
        [_frame_intrinsics(transforms, transforms["frames"][index]) for index in joint["neighbor_indices"].tolist()]
    )
    c2ws = torch.linalg.inv(joint["neighbor_w2cs"])
    expected = _camera_rays_for_intrinsics(intrinsic_rows, c2ws, 16, 16)
    assert joint["neighbor_camera_rays"].shape == (2, 16, 16, 6)
    torch.testing.assert_close(joint["neighbor_camera_rays"], expected)
    intrinsic_rows[:, 4:] = 0
    assert not torch.allclose(expected, _camera_rays_for_intrinsics(intrinsic_rows, c2ws, 16, 16))


@pytest.fixture
def nerfbusters_metrics_case(tmp_path, monkeypatch, recording_metric_suite):
    import torch

    from model_eval import compute_metrics_nerfbusters as module
    from model_eval.metrics_utils import load_image_as_tensor

    scene_root = tmp_path / "data" / "flowers"
    image_dir = scene_root / "images"
    image_dir.mkdir(parents=True)
    frame_names = ["frame_00000.png", *[f"frame_1_{i:05d}.png" for i in (1, 2, 3)]]
    transforms = {"frames": [{"file_path": f"images/{name}"} for name in frame_names]}
    (scene_root / "transforms.json").write_text(json.dumps(transforms))
    for name in frame_names:
        Image.new("RGB", (16, 16)).save(image_dir / name)
    eval_root = tmp_path / "eval"
    output = eval_root / "run" / "nerfbusters_1_evenly_spaced"
    frames = output / "flowers" / "frames" / "batch_0000"
    for kind in ("pred", "rendered"):
        (frames / kind).mkdir(parents=True)
        for index in (1, 2, 3):
            Image.new("RGB", (16, 16)).save(frames / kind / f"{index:05d}.png")
    mask_root = tmp_path / "masks"
    mask_dir = mask_root / "flowers"
    mask_dir.mkdir(parents=True)
    for index in range(3):
        Image.new("L", (16, 16), color=255).save(mask_dir / f"{index:05d}.png")

    monkeypatch.setattr(module, "NERFBUSTERS_SCENES", ("flowers",))
    monkeypatch.setattr(module, "MetricSuite", recording_metric_suite)
    monkeypatch.setattr(
        module, "load_image_as_tensor", lambda source, _: load_image_as_tensor(source, torch.device("cpu"))
    )
    monkeypatch.setattr(module, "load_visibility_mask", lambda *_args, **_kwargs: torch.ones(1, 1, 16, 16))
    kwargs = dict(
        render_name="run", num_views=1, eval_base_path=eval_root, nerfbusters_dir=tmp_path / "data", use_masks=False
    )
    return types.SimpleNamespace(
        run=lambda **overrides: module.main(**(kwargs | overrides)),
        frames=frames,
        output=output,
        image_dir=image_dir,
        mask_root=mask_root,
        mask_dir=mask_dir,
        suites=recording_metric_suite.instances,
    )


def test_nerfbusters_metrics_reject_partial_scene_frames(nerfbusters_metrics_case):
    case = nerfbusters_metrics_case
    for kind in ("pred", "rendered"):
        (case.frames / kind / "00002.png").unlink()
    with pytest.raises(ValueError, match="Incomplete frame coverage"):
        case.run()
    assert all(not suite.reports for suite in case.suites)


def test_nerfbusters_metrics_reject_missing_ground_truth(nerfbusters_metrics_case):
    case = nerfbusters_metrics_case
    (case.image_dir / "frame_1_00002.png").unlink()
    with pytest.raises(FileNotFoundError, match="GT"):
        case.run()
    assert all(not suite.reports for suite in case.suites)


def test_nerfbusters_metrics_reject_missing_requested_masks(nerfbusters_metrics_case):
    case = nerfbusters_metrics_case
    (case.mask_dir / "00001.png").unlink()
    with pytest.raises(FileNotFoundError, match="visibility mask"):
        case.run(use_masks=True, visibility_masks_dir=case.mask_root)
    assert all(not suite.reports for suite in case.suites)


def test_nerfbusters_metrics_report_complete_coverage(nerfbusters_metrics_case):
    case = nerfbusters_metrics_case
    case.run(use_masks=True, visibility_masks_dir=case.mask_root)
    suite = case.suites[0]
    assert suite.reports == [case.output]
    assert len(suite.ours.get_all_values("psnr")) == 3
    assert len(suite.ours_masked.get_all_values("psnr")) == 3


@pytest.mark.parametrize("protocol", [None, "legacy"])
def test_nerfbusters_metrics_default_to_standard_with_explicit_legacy_option(nerfbusters_metrics_case, protocol):
    case = nerfbusters_metrics_case
    case.run(**({"lpips_protocol": protocol} if protocol else {}))
    assert case.suites[0].options["lpips_protocol"] == (protocol or "standard")


if __name__ == "__main__":
    unittest.main()
