from __future__ import annotations

import json

import pytest

from tests.validate_pipeline_prompts import missing_prompt_files, num_frames_from_extra_args


def test_validation_dataset_raises_before_modulo_on_empty_data():
    """An empty validation dataset raises the actionable ValueError instead of
    the bare ZeroDivisionError from `logical_index % len(self.data)`.
    Behavioral rewrite of the old source-text scan over dl3dv_test.py."""
    from model_training.data.dl3dv_test import DL3DVPairedDatasetTest

    dataset = DL3DVPairedDatasetTest.__new__(DL3DVPairedDatasetTest)
    dataset.data = []
    dataset.start_index = 0

    with pytest.raises(ValueError, match="empty validation dataset"):
        dataset[0]


def test_num_frames_from_extra_args_supports_space_and_equals_forms():
    assert num_frames_from_extra_args("") == 81
    assert num_frames_from_extra_args("--num_frames 85 --dataset_scaling_factor 0.01") == 85
    assert num_frames_from_extra_args("--num_frames=97") == 97


def test_missing_prompt_files_reports_scenes_without_requested_frame_count(tmp_path):
    split_path = tmp_path / "split.json"
    prompt_dir = tmp_path / "prompts"
    scene_id = "scene-a"
    split_path.write_text(
        json.dumps(
            {
                "trainval": {scene_id: [{"subdir": "1K"}]},
                "test": {"scene-b": [{"subdir": "2K"}]},
            }
        )
    )
    (prompt_dir / "1K" / scene_id).mkdir(parents=True)
    (prompt_dir / "1K" / scene_id / "frames_81_stride_1.h5").touch()

    missing = missing_prompt_files(split_path, prompt_dir, num_frames=81)

    assert missing == ["2K/scene-b/frames_81_stride_1.h5"]


@pytest.fixture
def met3r_case(tmp_path):
    from types import SimpleNamespace

    import torch
    from PIL import Image

    from model_eval.compute_met3r import run_eval

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("scene-a\nscene-b\n")
    args = SimpleNamespace(
        evalset="dl3dv_ours",
        results_dir=results_dir,
        scene_list_file=scene_list,
        source="pred",
        img_size=16,
        batch_size=2,
        output_json=tmp_path / "metrics.json",
    )

    def add_scene(scene_id, frames=2):
        frame_dir = results_dir / scene_id / "frames" / "batch_0000" / "pred"
        frame_dir.mkdir(parents=True)
        for index in range(frames):
            Image.new("RGB", (16, 16)).save(frame_dir / f"{index:05d}.png")

    def metric(*, images):
        return (torch.full((len(images),), 0.25),)

    return SimpleNamespace(
        args=args,
        add_scene=add_scene,
        run=lambda: run_eval({"mast3r:cosine": metric}, args, torch.device("cpu")),
    )


def test_met3r_rejects_empty_metrics(met3r_case):
    import torch

    from model_eval.compute_met3r import run_eval

    case = met3r_case
    case.add_scene("scene-a")
    case.add_scene("scene-b")
    with pytest.raises(ValueError, match="At least one metric is required"):
        run_eval({}, case.args, torch.device("cpu"))
    assert not case.args.output_json.exists()


def test_met3r_rejects_missing_benchmark_scenes(met3r_case):
    case = met3r_case
    case.add_scene("scene-a")
    with pytest.raises(FileNotFoundError, match="scene-b"):
        case.run()
    assert not case.args.output_json.exists()


def test_met3r_rejects_entries_without_a_frame_pair(met3r_case):
    case = met3r_case
    case.add_scene("scene-a")
    case.add_scene("scene-b", frames=1)
    with pytest.raises(ValueError, match="at least two frames"):
        case.run()
    assert not case.args.output_json.exists()


@pytest.mark.parametrize("scores", [[], [float("nan")]])
def test_met3r_rejects_missing_or_nonfinite_pair_scores(met3r_case, scores):
    import torch

    from model_eval.compute_met3r import run_eval

    case = met3r_case
    case.add_scene("scene-a")
    case.add_scene("scene-b")
    metric = lambda **kwargs: (torch.tensor(scores),)
    with pytest.raises(ValueError, match="Invalid scores"):
        run_eval({"mast3r:cosine": metric}, case.args, torch.device("cpu"))
    assert not case.args.output_json.exists()


def test_met3r_records_evaluated_coverage(met3r_case):
    case = met3r_case
    case.add_scene("scene-a", frames=3)
    case.add_scene("scene-b")
    case.run()
    results = json.loads(case.args.output_json.read_text())
    assert results["metrics"]["mast3r:cosine"]["overall"] == 0.25
    assert results["coverage"] == {
        "expected_entries": 2,
        "evaluated_entries": 2,
        "frames_per_entry": {"scene-a": 3, "scene-b": 2},
    }
