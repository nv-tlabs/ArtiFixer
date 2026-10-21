import json
from pathlib import Path

import pytest

from data_processing.artifixer3d import artifixer3d_paths, load_prepared_scene, write_artifixer3d_plus_inference_split
from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset


def _write_prepared_split(tmp_path: Path) -> Path:
    (tmp_path / "colmap").mkdir()
    (tmp_path / "selected_indices.json").write_text("[0]")
    (tmp_path / "prompt.h5").write_bytes(b"prompt")
    (tmp_path / "transforms.json").write_text(json.dumps({"frames": [{"file_path": "frame.png"}]}))
    entry: dict[str, object] = {
        "transforms_path": "transforms.json",
        "image_root": "colmap",
        "selected_indices_path": "selected_indices.json",
        "prompt_path": "prompt.h5",
        "camera_scale": 0.025,
        "metric_scale": 2.5,
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"test": {"scene": entry}}))
    return split_path


def _write_artifixer3d_plus_split(tmp_path: Path, split_path: Path) -> Path:
    scene = load_prepared_scene(tmp_path, split_path, "scene")
    output_split = tmp_path / "split_artifixer3d_plus.json"
    paths = artifixer3d_paths(scene, tmp_path / "artifixer3d", output_split, steps=1)
    paths.distillation_selected_indices_path.parent.mkdir(parents=True)
    paths.distillation_selected_indices_path.write_text("[0]")
    render_dir = tmp_path / "render"
    for directory in (render_dir / "renders", render_dir / "opacity"):
        directory.mkdir(parents=True)
        (directory / "00000.png").write_bytes(b"frame")
    (render_dir / "selected_indices.json").write_text("[0]")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    write_artifixer3d_plus_inference_split(scene, paths, render_dir, checkpoint)
    return output_split


def test_metric_scale_is_ignored_on_input_and_never_written_to_output(tmp_path: Path):
    """The stored metric_scale field was removed: camera_scale is the value the
    pipeline consumes, and the meters factor is recoverable from it by dividing
    out the fixed conditioning constant. Legacy split files that still carry
    the key must load fine (the key is ignored), and newly written splits must
    never contain it."""
    split_path = _write_prepared_split(tmp_path)

    scene = load_prepared_scene(tmp_path, split_path, "scene")
    assert not hasattr(scene, "metric_scale")

    output_split = _write_artifixer3d_plus_split(tmp_path, split_path)

    entry = json.loads(output_split.read_text())["test"]["scene"]
    assert "metric_scale" not in entry
    loaded = ReconstructedColmapEvalDataset._load_scene_metadata(tmp_path, "scene", entry)
    assert loaded.camera_scale == 0.025


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_reconstructed_split_rejects_invalid_camera_scale(tmp_path, scale):
    split_path = _write_prepared_split(tmp_path)
    data = json.loads(split_path.read_text())
    data["test"]["scene"]["camera_scale"] = scale
    split_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="finite and positive"):
        load_prepared_scene(tmp_path, split_path, "scene")
