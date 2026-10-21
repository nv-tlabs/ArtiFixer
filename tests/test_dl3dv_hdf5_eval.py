# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt pinning in the DL3DV HDF5 evaluation dataset.

Evaluation must always condition on the scene's canonical forward-order
caption, frames_all_stride_1.h5. The reverse/revisit files next to it are
training augmentation with different captions; drawing one of those would tie
the reported numbers to a random pick whose seed shifts with sharding and
scene filtering.
"""

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from model_eval.compute_metrics_dl3dv import load_hdf5_frame_count
from model_eval.datasets.dl3dv_hdf5_eval import DL3DVHDF5EvalDataset
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT
from model_training.utils.video_io import encode_frames_to_video

pytestmark = pytest.mark.requires_host_ffmpeg

SUBDIR = "960P"
NUM_VIEWS = 2
NUM_SCENE_FRAMES = 9
DATASET_SCALING_FACTOR = 0.01

CANONICAL_PROMPT = "frames_all_stride_1.h5"
COSMOS3_CANONICAL_PROMPT = "frames_all_stride_1.cosmos3.h5"
VARIANT_PROMPTS = (
    "frames_all_stride_1_reverse.h5",
    "frames_all_stride_1_revisit.h5",
    "frames_all_stride_1_revisit_reverse.h5",
)


def _c2w_x(x: float) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _write_prompt_file(path: Path, caption: str, image_indices: list[int]) -> None:
    with h5py.File(path, "w") as f:
        if ".cosmos3." in path.name:
            f.attrs["pad_token_id"] = 0
            f.attrs["negative_input_ids"] = np.array([1, 2], dtype=np.int64)
            data = np.array([3, 4], dtype=np.int64)
        else:
            data = np.zeros((4, 8), dtype=np.uint16)
        dataset = f.create_dataset("frame_00001.png", data=data)
        dataset.attrs["image_indices"] = np.asarray(image_indices, dtype=np.int64)
        dataset.attrs["caption"] = caption


def _write_scene(
    tmp_path: Path,
    scene_id: str,
    prompt_names_and_captions: dict[str, str],
    *,
    frame_intrinsics: dict[int, dict] | None = None,
) -> dict:
    """Write one loadable eval scene and return its split entry."""
    transforms = {"w": 16, "h": 16, "fl_x": 16.0, "fl_y": 16.0, "cx": 8.0, "cy": 8.0}
    transforms["frames"] = [
        {
            "file_path": f"images/frame_{i:05d}.png",
            "transform_matrix": _c2w_x(float(i)),
            **(frame_intrinsics or {}).get(i, {}),
        }
        for i in range(NUM_SCENE_FRAMES)
    ]
    (tmp_path / "dl3dv" / SUBDIR).mkdir(parents=True, exist_ok=True)
    with ZipFile(tmp_path / "dl3dv" / SUBDIR / f"{scene_id}.zip", "w") as zf:
        zf.writestr(f"{scene_id}/transforms.json", json.dumps(transforms))
        for frame in transforms["frames"]:
            png = BytesIO()
            Image.new("RGB", (16, 16), (100, 100, 100)).save(png, format="PNG")
            zf.writestr(f"{scene_id}/{frame['file_path'].replace('images/', 'images_4/')}", png.getvalue())

    prompt_scene_dir = tmp_path / "prompts" / SUBDIR / scene_id
    prompt_scene_dir.mkdir(parents=True, exist_ok=True)
    forward_indices = list(range(NUM_SCENE_FRAMES))
    for prompt_name, caption in prompt_names_and_captions.items():
        indices = forward_indices if prompt_name == CANONICAL_PROMPT else forward_indices[::-1]
        _write_prompt_file(prompt_scene_dir / prompt_name, caption, indices)

    video = encode_frames_to_video(np.full((NUM_SCENE_FRAMES, 16, 16, 3), 100, dtype=np.uint8))
    recon_dir = tmp_path / "recons" / scene_id
    recon_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = recon_dir / f"data_{scene_id}_0_{NUM_VIEWS}.h5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("selected_indices", data=np.arange(NUM_VIEWS))
        f.create_dataset("render_video", data=np.frombuffer(video, dtype=np.uint8))
        f.create_dataset("opacity_video", data=np.frombuffer(video, dtype=np.uint8))

    return {
        "scene_half": "0",
        "num_selected_indices": NUM_VIEWS,
        "subdir": SUBDIR,
        "scale": 1.0,
        "hdf5_path": str(hdf5_path),
    }


def _write_split(tmp_path: Path, scenes: dict[str, dict]) -> Path:
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"trainval": {}, "test": {scene_id: [entry] for scene_id, entry in scenes.items()}})
    )
    return split_path


def _dataset(tmp_path: Path, split_path: Path, filter_scene_id: str | None = None, **kwargs) -> DL3DVHDF5EvalDataset:
    kwargs.setdefault("prompt_format", WAN_T5_PROMPT_FORMAT)
    return DL3DVHDF5EvalDataset(
        split="test",
        split_path=split_path,
        dl3dv_dir=tmp_path / "dl3dv",
        prompt_dir=tmp_path / "prompts",
        num_views=NUM_VIEWS,
        dataset_scaling_factor=DATASET_SCALING_FACTOR,
        filter_scene_id=filter_scene_id,
        **kwargs,
    )


def test_video_compressed_hdf5_frame_count_matches_inference(tmp_path):
    scenes = {"scene_a": _write_scene(tmp_path, "scene_a", {CANONICAL_PROMPT: "forward caption"})}
    split_path = _write_split(tmp_path, scenes)
    with h5py.File(scenes["scene_a"]["hdf5_path"], "r") as reconstruction:
        assert not reconstruction.attrs
        assert set(reconstruction) == {"selected_indices", "render_video", "opacity_video"}

    item = _dataset(tmp_path, split_path, include_all_frames=True)[0]

    assert item["frame_indices"].tolist() == list(range(NUM_SCENE_FRAMES))
    assert load_hdf5_frame_count("scene_a", NUM_VIEWS, split_path) == NUM_SCENE_FRAMES


@pytest.mark.parametrize("pose_alignment", ["mean", "last"])
def test_hdf5_camera_tensors_use_the_first_target_origin(tmp_path, pose_alignment):
    scenes = {"scene_a": _write_scene(tmp_path, "scene_a", {CANONICAL_PROMPT: "forward caption"})}
    with h5py.File(scenes["scene_a"]["hdf5_path"], "r+") as reconstruction:
        reconstruction["selected_indices"][:] = [1, 2]
    split_path = _write_split(tmp_path, scenes)
    item = _dataset(tmp_path, split_path, include_all_frames=True, camera_pose_alignment=pose_alignment)[0]
    torch.testing.assert_close(item["w2cs"][0], torch.eye(4))
    torch.testing.assert_close(item["camera_rays"][0, ..., :3], torch.zeros_like(item["camera_rays"][0, ..., :3]))
    scale = scenes["scene_a"]["scale"] * DATASET_SCALING_FACTOR
    expected_positions = -(item["neighbor_indices"].float() - item["frame_indices"][0]) * scale
    torch.testing.assert_close(item["neighbor_w2cs"][:, 0, 3], expected_positions)


@pytest.mark.parametrize("prompt_format", [WAN_T5_PROMPT_FORMAT, COSMOS3_PROMPT_FORMAT])
@pytest.mark.parametrize("pose_alignment", ["mean", "last"])
def test_short_heldout_chunk_defers_padding_to_inference(tmp_path, prompt_format, pose_alignment):
    scenes = {
        "scene_a": _write_scene(
            tmp_path,
            "scene_a",
            {CANONICAL_PROMPT: "forward caption", COSMOS3_CANONICAL_PROMPT: "forward caption"},
        )
    }
    chunk_size = 5
    dataset = _dataset(
        tmp_path,
        _write_split(tmp_path, scenes),
        max_test_frames=chunk_size,
        frames_per_block=None,
        prompt_format=prompt_format,
        camera_pose_alignment=pose_alignment,
    )
    item = dataset[len(dataset) - 1]
    expected_indices = list(range(NUM_VIEWS + chunk_size, NUM_SCENE_FRAMES))
    assert item["frame_indices"].tolist() == expected_indices
    assert item["valid_frames_mask"].tolist() == [True] * len(expected_indices)
    torch.testing.assert_close(item["w2cs"][0], torch.eye(4))
    expected_position = -(expected_indices[-1] - expected_indices[0]) * DATASET_SCALING_FACTOR
    torch.testing.assert_close(item["w2cs"][-1, 0, 3], torch.tensor(expected_position))
    assert torch.isfinite(item["camera_rays"]).all()


def test_joint_mode_reaches_real_hdf5_items_and_preserves_reference_lens_distortion(tmp_path):
    from model_training.data.utils import _camera_rays_for_intrinsics, _frame_intrinsics

    overrides = {
        0: dict(fl_x=22.0, k1=0.18, k2=-0.03, p1=0.015, p2=-0.01),
        1: dict(fl_y=20.0, k1=-0.08, k2=0.01, p1=-0.01, p2=0.02),
    }
    scenes = {
        "scene_a": _write_scene(
            tmp_path,
            "scene_a",
            {CANONICAL_PROMPT: "forward caption", COSMOS3_CANONICAL_PROMPT: "forward caption"},
            frame_intrinsics=overrides,
        )
    }
    split_path = _write_split(tmp_path, scenes)
    legacy = _dataset(tmp_path, split_path, include_all_frames=True)[0]
    dataset = _dataset(tmp_path, split_path, include_all_frames=True, prompt_format=COSMOS3_PROMPT_FORMAT)
    joint = dataset[0]
    assert "neighbor_camera_rays" not in legacy
    assert joint["neighbor_camera_rays"].shape == (NUM_VIEWS, 16, 16, 6)
    transforms = dataset.transforms_by_scene_id["scene_a"]
    rows = torch.stack(
        [_frame_intrinsics(transforms, transforms["frames"][index]) for index in joint["neighbor_indices"].tolist()]
    )
    c2ws = torch.linalg.inv(joint["neighbor_w2cs"])
    expected = _camera_rays_for_intrinsics(rows, c2ws, 16, 16)
    torch.testing.assert_close(joint["neighbor_camera_rays"], expected)
    rows[:, 4:] = 0
    assert not torch.allclose(expected, _camera_rays_for_intrinsics(rows, c2ws, 16, 16))


def test_every_item_uses_the_canonical_prompt(tmp_path):
    scenes = {
        "scene_a": _write_scene(
            tmp_path,
            "scene_a",
            {CANONICAL_PROMPT: "scene_a forward caption", **{name: f"scene_a {name}" for name in VARIANT_PROMPTS}},
        ),
        "scene_b": _write_scene(tmp_path, "scene_b", {CANONICAL_PROMPT: "scene_b forward caption"}),
    }
    dataset = _dataset(tmp_path, _write_split(tmp_path, scenes))

    assert len(dataset) > 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert item["prompt_file"] == f"{item['scene_id']}/{CANONICAL_PROMPT}"
        assert item["prompt"] == f"{item['scene_id']} forward caption"


def test_cosmos3_format_pins_the_format_distinguished_canonical_prompt(tmp_path):
    from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT

    scenes = {
        "scene_a": _write_scene(
            tmp_path,
            "scene_a",
            {CANONICAL_PROMPT: "scene_a forward caption", COSMOS3_CANONICAL_PROMPT: "scene_a cosmos3 caption"},
        ),
    }
    dataset = _dataset(tmp_path, _write_split(tmp_path, scenes), prompt_format=COSMOS3_PROMPT_FORMAT)

    assert dataset.prompts_by_scene_id["scene_a"] == [
        tmp_path / "prompts" / SUBDIR / "scene_a" / COSMOS3_CANONICAL_PROMPT
    ]


def test_missing_cosmos3_prompt_raises_even_when_the_wan_file_exists(tmp_path):
    from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT

    scenes = {"scene_a": _write_scene(tmp_path, "scene_a", {CANONICAL_PROMPT: "scene_a forward caption"})}
    with pytest.raises(FileNotFoundError, match=COSMOS3_CANONICAL_PROMPT):
        _dataset(tmp_path, _write_split(tmp_path, scenes), prompt_format=COSMOS3_PROMPT_FORMAT)


def test_missing_canonical_prompt_raises(tmp_path):
    scenes = {
        "scene_a": _write_scene(tmp_path, "scene_a", {name: f"scene_a {name}" for name in VARIANT_PROMPTS}),
    }
    with pytest.raises(FileNotFoundError, match="canonical evaluation prompt"):
        _dataset(tmp_path, _write_split(tmp_path, scenes))


def test_scene_filter_restricts_items_to_that_scene(tmp_path):
    """Filtering renumbers item indices, which fed the old variant pick's seed;
    the kept scene must still read its canonical caption."""
    scenes = {
        "scene_a": _write_scene(tmp_path, "scene_a", {CANONICAL_PROMPT: "scene_a forward caption"}),
        "scene_b": _write_scene(
            tmp_path,
            "scene_b",
            {CANONICAL_PROMPT: "scene_b forward caption", **{name: f"scene_b {name}" for name in VARIANT_PROMPTS}},
        ),
    }
    dataset = _dataset(tmp_path, _write_split(tmp_path, scenes), filter_scene_id="scene_b")

    assert len(dataset) > 0
    for idx in range(len(dataset)):
        assert dataset[idx]["prompt"] == "scene_b forward caption"


def test_missing_canonical_prompt_fails_even_for_filtered_out_scenes(tmp_path):
    """A broken benchmark scene is a data error; filtering must not mask it."""
    scenes = {
        "scene_a": _write_scene(tmp_path, "scene_a", {CANONICAL_PROMPT: "scene_a forward caption"}),
        "scene_b": _write_scene(tmp_path, "scene_b", {CANONICAL_PROMPT: "scene_b forward caption"}),
    }
    split_path = _write_split(tmp_path, scenes)
    (tmp_path / "prompts" / SUBDIR / "scene_a" / CANONICAL_PROMPT).unlink()

    with pytest.raises(FileNotFoundError, match="scene_a"):
        _dataset(tmp_path, split_path, filter_scene_id="scene_b")


def test_multi_record_canonical_prompt_raises(tmp_path):
    scenes = {"scene_a": _write_scene(tmp_path, "scene_a", {CANONICAL_PROMPT: "scene_a forward caption"})}
    split_path = _write_split(tmp_path, scenes)
    with h5py.File(tmp_path / "prompts" / SUBDIR / "scene_a" / CANONICAL_PROMPT, "a") as f:
        second = f.create_dataset("frame_00002.png", data=np.zeros((4, 8), dtype=np.uint16))
        second.attrs["caption"] = "a second caption"

    with pytest.raises(ValueError, match="exactly one"):
        _dataset(tmp_path, split_path)
