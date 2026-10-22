# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
from PIL import Image

# model_eval.metrics_utils needs the real packages. Skip when they are
# missing (e.g. on login nodes) rather than putting stubs in sys.modules,
# where they would leak into test modules collected later.
pytest.importorskip("torch", reason="model_eval.metrics_utils requires torch")
pytest.importorskip("torchvision", reason="model_eval.metrics_utils requires torchvision")
pytest.importorskip("torchmetrics", reason="model_eval.metrics_utils requires torchmetrics")

from model_eval.compute_metrics_dl3dv import compute_metrics
from model_eval.metrics_utils import _scene_frame_root


@pytest.mark.parametrize(
    ("evalset", "view_args", "expected_views"),
    [
        ("3dgrut_dl3dv_ours", [], 6),
        ("3dgrut_dl3dv_difix", [], None),
        ("artifixer3d_dl3dv_difix", [], None),
        ("artifixer3d_dl3dv_ours", [], None),
        ("3dgrut_dl3dv_difix", ["--num_views", "6"], 6),
    ],
)
def test_metric_cli_reads_the_inference_output_path(tmp_path, evalset, view_args, expected_views):
    from model_eval import compute_metrics_dl3dv as metrics
    from model_eval.inference import get_output_dir
    from model_eval.inference_args import parse_args

    shared = [
        "--evalset",
        evalset,
        "--render_name",
        "comparison",
        "--split_path",
        str(tmp_path / "split.json"),
        "--dl3dv_dir",
        str(tmp_path / "dl3dv"),
        *view_args,
    ]
    inference_args = parse_args(
        [
            *shared,
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--save_dir",
            str(tmp_path / "outputs"),
            "--prompt_dir",
            str(tmp_path / "prompts"),
            "--recon_results_dir",
            str(tmp_path / "recon"),
        ]
    )
    metric_args = metrics.parse_args(
        [*shared, "--eval_base_path", str(tmp_path / "outputs"), "--recon_results_dir", str(tmp_path / "recon")]
    )
    assert inference_args.num_views == metric_args.num_views == expected_views
    # Before either command runs a model, the scorer must resolve the same
    # directory as inference. An absent output deliberately stops before I/O.
    expected_path = get_output_dir(inference_args)
    with pytest.raises(FileNotFoundError) as exc:
        compute_metrics(
            render_name=metric_args.render_name,
            num_views=metric_args.num_views,
            evalset=metric_args.evalset,
            split_path=metric_args.split_path,
            dl3dv_dir=metric_args.dl3dv_dir,
            eval_base_path=metric_args.eval_base_path,
            use_masks=False,
            test_scene_ids=[],
        )
    assert str(exc.value) == f"Eval path not found: {expected_path}"


@pytest.mark.parametrize("value", ["auto", "0", "-1", "invalid"])
def test_hdf5_metric_cli_rejects_invalid_view_count(value):
    from model_eval.compute_metrics_dl3dv import parse_args

    with pytest.raises(SystemExit) as exc:
        parse_args(
            [
                "--evalset",
                "3dgrut_dl3dv_ours",
                "--render_name",
                "run",
                "--split_path",
                "/split.json",
                "--dl3dv_dir",
                "/data",
                "--eval_base_path",
                "/outputs",
                "--num_views",
                value,
            ]
        )
    assert exc.value.code == 2


def test_directory_metric_cli_uses_reconstruction_anchors_with_fewer_references(
    tmp_path, monkeypatch, recording_metric_suite
):
    from model_eval import compute_metrics_dl3dv as metrics

    case = _make_dl3dv_metrics_case(tmp_path, monkeypatch, recording_metric_suite, "artifixer3d_dl3dv_ours")
    reconstruction = tmp_path / "recon/scene/distill_ours_eval/scene/ours_30000"
    for kind in ("renders", "opacity"):
        (reconstruction / kind).mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (16, 16)).save(reconstruction / kind / f"{index:05d}.png")
    (reconstruction / "selected_indices.json").write_text("[0, 1]")
    for kind in ("pred", "rendered"):
        (case.frames / kind / "00001.png").unlink()
    # There is deliberately no source HDF5: reference count does not select an anchor set.
    case.hdf5_path.unlink()
    monkeypatch.setattr(metrics, "read_scene_list", lambda _: ["scene"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metrics",
            "--evalset",
            "artifixer3d_dl3dv_ours",
            "--render_name",
            "run",
            "--num_views",
            "1",
            "--split_path",
            str(tmp_path / "split.json"),
            "--dl3dv_dir",
            str(tmp_path / "dl3dv"),
            "--eval_base_path",
            str(tmp_path / "eval"),
            "--recon_results_dir",
            str(tmp_path / "recon"),
            "--no_masks",
        ],
    )
    metrics.main()
    assert len(case.suites[0].ours.get_all_values("psnr")) == 2
    assert case.suites[0].reports == [case.output]


def _make_dl3dv_metrics_case(tmp_path, monkeypatch, recording_metric_suite, evalset):
    import torch

    from model_eval import compute_metrics_dl3dv as module
    from model_eval.metrics_utils import load_image_as_tensor

    difix = evalset in module.DIFIX_EVALSETS
    scene_root = tmp_path / "dl3dv" / "scene"
    image_dir = scene_root / "images_4"
    image_dir.mkdir(parents=True)
    transforms = {"frames": [{"file_path": f"images/frame_{i:05d}.png"} for i in range(4)]}
    (scene_root / "transforms.json").write_text(json.dumps(transforms))
    for frame in transforms["frames"]:
        Image.new("RGB", (16, 16)).save(image_dir / Path(frame["file_path"]).name)

    hdf5_path = tmp_path / "scene.h5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.attrs["n_frames"] = 4
        handle.create_dataset("selected_indices", data=[0])
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "test": {
                    "scene": [
                        {"subdir": "1K", "hdf5_path": str(hdf5_path), "scene_half": "0", "num_selected_indices": 1}
                    ]
                },
                "trainval": {},
            }
        )
    )
    eval_root = tmp_path / "eval"
    output = eval_root / "run" / f"distilled_views_{evalset}_1_evenly_spaced"
    frames = output / "scene" / "frames" / "batch_0000"
    for kind in ("pred", "rendered"):
        (frames / kind).mkdir(parents=True)
        for index in (1, 2, 3):
            Image.new("RGB", (16, 16)).save(frames / kind / f"{index:05d}.png")
    mask_root = tmp_path / "masks"
    mask_dir = mask_root / "scene" / "visibility_median_count"
    mask_dir.mkdir(parents=True)
    for index in range(3) if difix else (1, 2, 3):
        Image.new("L", (16, 16), color=255).save(mask_dir / f"{index:05d}.png")

    loaded_masks = []

    def load_mask(path, *_args, **_kwargs):
        loaded_masks.append(path)
        return torch.ones(1, 1, 16, 16)

    monkeypatch.setattr(module, "MetricSuite", recording_metric_suite)
    monkeypatch.setattr(
        module, "load_image_as_tensor", lambda source, _: load_image_as_tensor(source, torch.device("cpu"))
    )
    monkeypatch.setattr(module, "load_visibility_mask", load_mask)
    kwargs = dict(
        render_name="run",
        num_views=1,
        evalset=evalset,
        split_path=split_path,
        dl3dv_dir=tmp_path / "dl3dv",
        eval_base_path=eval_root,
        use_masks=False,
    )
    if difix:
        kwargs["training_indices_for_scene"] = lambda _: {0}
    return SimpleNamespace(
        run=lambda **overrides: compute_metrics(**(kwargs | overrides)),
        frames=frames,
        output=output,
        hdf5_path=hdf5_path,
        mask_root=mask_root,
        mask_dir=mask_dir,
        suites=recording_metric_suite.instances,
        loaded_masks=loaded_masks,
    )


@pytest.fixture
def dl3dv_metrics_case(tmp_path, monkeypatch, recording_metric_suite):
    return _make_dl3dv_metrics_case(tmp_path, monkeypatch, recording_metric_suite, "3dgrut_dl3dv_ours")


@pytest.fixture
def difix_metrics_case(tmp_path, monkeypatch, recording_metric_suite):
    return _make_dl3dv_metrics_case(tmp_path, monkeypatch, recording_metric_suite, "artifixer3d_dl3dv_difix")


def test_dl3dv_metrics_reject_matching_but_incomplete_frame_sets(dl3dv_metrics_case):
    case = dl3dv_metrics_case
    for kind in ("pred", "rendered"):
        (case.frames / kind / "00002.png").unlink()
    with pytest.raises(ValueError, match="Incomplete frame coverage"):
        case.run()
    assert all(not suite.reports for suite in case.suites)


def test_dl3dv_metrics_reject_duplicate_frame_ids(dl3dv_metrics_case):
    case = dl3dv_metrics_case
    for kind in ("pred", "rendered"):
        Image.new("RGB", (16, 16)).save(case.frames / kind / "1.png")
    with pytest.raises(ValueError, match="Duplicate frame IDs"):
        case.run()
    assert all(not suite.reports for suite in case.suites)


def test_dl3dv_metrics_accept_complete_test_frames_with_training_frames(dl3dv_metrics_case):
    case = dl3dv_metrics_case
    for kind in ("pred", "rendered"):
        Image.new("RGB", (16, 16)).save(case.frames / kind / "00000.png")
    case.run()
    assert case.suites[0].reports == [case.output]
    assert len(case.suites[0].ours.get_all_values("psnr")) == 3


@pytest.mark.parametrize("fixture", ["dl3dv_metrics_case", "difix_metrics_case"])
@pytest.mark.parametrize("protocol", [None, "legacy"])
def test_dl3dv_metrics_default_to_standard_with_explicit_legacy_option(fixture, protocol, request):
    case = request.getfixturevalue(fixture)
    case.run(**({"lpips_protocol": protocol} if protocol else {}))
    assert case.suites[0].options["lpips_protocol"] == (protocol or "standard")


@pytest.mark.parametrize("metadata_source", ["n_frames", "depth_np_array"])
def test_dl3dv_metrics_use_frame_metadata_without_probing_video(dl3dv_metrics_case, monkeypatch, metadata_source):
    from model_eval import compute_metrics_dl3dv as module

    case = dl3dv_metrics_case
    with h5py.File(case.hdf5_path, "r+") as reconstruction:
        reconstruction.create_dataset("render_video", data=[0])
        if metadata_source == "depth_np_array":
            del reconstruction.attrs["n_frames"]
            reconstruction.create_dataset("depth_np_array", shape=(4, 1, 1), dtype="float32")

    def unexpected_probe(*args):
        raise AssertionError("Available frame metadata must not trigger video decoding")

    monkeypatch.setattr(module, "probe_video_frame_count", unexpected_probe)
    case.run()

    assert case.suites[0].reports == [case.output]
    assert len(case.suites[0].ours.get_all_values("psnr")) == 3


@pytest.mark.requires_host_ffmpeg
def test_dl3dv_metrics_accept_video_compressed_hdf5(dl3dv_metrics_case):
    import numpy as np

    from model_training.utils.video_io import encode_frames_to_video

    case = dl3dv_metrics_case
    video = encode_frames_to_video(np.zeros((4, 16, 16, 3), dtype=np.uint8))
    with h5py.File(case.hdf5_path, "r+") as reconstruction:
        del reconstruction.attrs["n_frames"]
        for name in ("render_video", "opacity_video"):
            reconstruction.create_dataset(name, data=np.frombuffer(video, dtype=np.uint8))

    case.run()

    assert case.suites[0].reports == [case.output]
    assert len(case.suites[0].ours.get_all_values("psnr")) == 3


def test_dl3dv_metrics_respect_source_video_frame_count(dl3dv_metrics_case):
    case = dl3dv_metrics_case
    with h5py.File(case.hdf5_path, "r+") as handle:
        handle.attrs["n_frames"] = 3
    for kind in ("pred", "rendered"):
        (case.frames / kind / "00003.png").unlink()
    case.run()
    assert len(case.suites[0].ours.get_all_values("psnr")) == 2
    assert case.suites[0].reports == [case.output]


def test_dl3dv_metrics_reject_missing_requested_masks(dl3dv_metrics_case):
    case = dl3dv_metrics_case
    (case.mask_dir / "00002.png").unlink()
    with pytest.raises(FileNotFoundError, match="visibility mask"):
        case.run(use_masks=True, visibility_masks_dir=case.mask_root)
    assert all(not suite.reports for suite in case.suites)


def test_difix_missing_compact_mask_does_not_use_another_frames_mask(difix_metrics_case):
    case = difix_metrics_case
    (case.mask_dir / "00001.png").unlink()
    with pytest.raises(FileNotFoundError, match="visibility mask"):
        case.run(use_masks=True, visibility_masks_dir=case.mask_root)
    assert all(not suite.reports for suite in case.suites)


@pytest.mark.parametrize("source_count", [2, 3])
def test_difix_metrics_use_remapped_frame_count_and_compact_masks(difix_metrics_case, monkeypatch, source_count):
    from model_eval import dl3dv_difix_frame_remap

    case = difix_metrics_case
    monkeypatch.setattr(dl3dv_difix_frame_remap, "load_difix_frame_remap", lambda: {"scene": {"frame_00003.png"}})
    for kind in ("pred", "rendered"):
        (case.frames / kind / "00003.png").unlink()
    kwargs = dict(use_masks=True, visibility_masks_dir=case.mask_root, frame_count_for_scene=lambda _: source_count)
    if source_count != 3:
        with pytest.raises(ValueError, match="2 renders but 3 remapped frames"):
            case.run(**kwargs)
        assert all(not suite.reports for suite in case.suites)
    else:
        case.run(**kwargs)
        assert case.loaded_masks == [case.mask_dir / "00000.png", case.mask_dir / "00001.png"]
        assert len(case.suites[0].ours.get_all_values("psnr")) == 2
        assert case.suites[0].reports == [case.output]


@pytest.mark.parametrize("source_error", [None, "opacity", "gap"])
def test_dl3dv_metrics_validate_reconstruction_source_frames(dl3dv_metrics_case, tmp_path, source_error):
    from model_eval.compute_metrics_dl3dv import load_reconstruction_frame_count

    case = dl3dv_metrics_case
    recon_root = tmp_path / "recon"
    checkpoint = recon_root / "scene" / "distill_ours_eval" / "scene" / "ours_30000"
    for kind in ("renders", "opacity"):
        (checkpoint / kind).mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (16, 16)).save(checkpoint / kind / f"{index:05d}.png")
    if source_error is not None:
        (checkpoint / "opacity" / "00001.png").unlink()
        if source_error == "gap":
            (checkpoint / "renders" / "00001.png").unlink()
    for kind in ("pred", "rendered"):
        (case.frames / kind / "00003.png").unlink()
    frame_count = lambda scene_id: load_reconstruction_frame_count(
        recon_root, scene_id, checkpoint="30000", recon_subdir="distill_ours_eval"
    )
    if source_error is not None:
        with pytest.raises(ValueError, match="frame"):
            case.run(frame_count_for_scene=frame_count)
        assert all(not suite.reports for suite in case.suites)
    else:
        case.run(frame_count_for_scene=frame_count)
        assert len(case.suites[0].ours.get_all_values("psnr")) == 2
        assert case.suites[0].reports == [case.output]


def test_scene_frame_root_uses_main_dl3dv_scene_layout(tmp_path):
    scene_id = "scene-a"
    scene_dir = tmp_path / scene_id
    scene_dir.mkdir()
    (scene_dir / "transforms.json").write_text("{}")

    assert _scene_frame_root(tmp_path, scene_id) == scene_dir


def test_scene_frame_root_does_not_probe_benchmark_nerfstudio_layout(tmp_path):
    scene_id = "scene-a"
    nerfstudio_dir = tmp_path / scene_id / "nerfstudio"
    nerfstudio_dir.mkdir(parents=True)
    (nerfstudio_dir / "transforms.json").write_text("{}")

    assert _scene_frame_root(tmp_path, scene_id) is None


def test_dl3dv_metrics_reject_missing_expected_scene_directories(tmp_path):
    eval_base_path = tmp_path / "eval"
    output_path = (
        eval_base_path / "artifixer-14b_chunk7_sink7" / "distilled_views_artifixer3d_dl3dv_ours_6_evenly_spaced"
    )
    (output_path / "scene_a").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"Missing 1 expected .* scene_b"):
        compute_metrics(
            "artifixer-14b_chunk7_sink7",
            6,
            "artifixer3d_dl3dv_ours",
            split_path=tmp_path / "unused-split.json",
            dl3dv_dir=tmp_path / "unused-dl3dv",
            eval_base_path=eval_base_path,
            use_masks=False,
            test_scene_ids=("scene_a", "scene_b"),
            training_indices_for_scene=lambda _: set(),
        )
