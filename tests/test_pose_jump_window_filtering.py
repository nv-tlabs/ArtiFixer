# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose-jump window filtering in the DL3DV train dataset.

At construction the dataset computes, per entry, which consecutive-frame
transitions are pose jumps: camera centers more than
POSE_JUMP_MAX_TRANSLATION_M apart in meters, using the entry's reconstruction
scale. Three train-only behaviors follow, all enabled by
filter_pose_jump_windows:
- At construction, a scene-half with no jump-free run of
  min(num_frames, scene_num_frames) frames is dropped: scenes shorter than
  the window are kept when fully jump-free, since their whole clip is the
  one (pingpong-expanded) window. Val/test datasets never drop scenes.
- At fetch time, prompt windows whose span contains a jump are removed from
  the candidates before sampling, so a window crossing a jump is never
  sampled.
- Neighbor views are drawn from training views in the target window's own
  jump-free segment; windows whose segment cannot supply the requested
  neighbor count are dropped from the candidates. A scene that cannot supply
  the count from any long-enough segment sits out the draw entirely: the
  train sampler serves the fetch from the next viable scene in the shuffled
  order.
Scenes without jumps behave as if the filter were off.
"""

import json
import logging
import math
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from model_training.data.dl3dv_base import (
    DL3DVPairedDatasetBase,
    _has_pose_jump_free_window,
    _pose_jump_transitions_for_scale,
    _scene_camera_position_deltas,
    _segment_neighbor_capacity,
    _window_span_contains_pose_jump,
    _windows_with_same_segment_neighbors,
)
from model_training.data.dl3dv_test import DL3DVPairedDatasetTest
from model_training.data.dl3dv_train import DL3DVPairedDatasetTrain
from model_training.utils.video_io import encode_frames_to_video

pytestmark = pytest.mark.requires_host_ffmpeg

BASE_LOGGER = "model_training.data.dl3dv_base"
NUM_FRAMES = 5
SUBDIR = "960P"


def _c2w_x(x: float) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _transforms_with_x_positions(positions: list[float]) -> dict:
    return {
        "frames": [
            {"file_path": f"images/frame_{i:05d}.png", "transform_matrix": _c2w_x(x)} for i, x in enumerate(positions)
        ]
    }


def _write_scene(
    tmp_path: Path,
    scene_id: str,
    *,
    x_positions: list[float],
    prompt_windows: dict[str, list[int]],
    halves: tuple[str, ...] = ("first",),
    splits: tuple[int, ...] = (2,),
    selected_indices: list[int] | None = None,
    prompt_filename: str | None = None,
    image_size: int = 16,
) -> None:
    """Write one loadable scene: source zip, prompt h5, reconstruction h5s.

    Each frame's image is a solid gray of value 50 + 10 * frame_index, so a
    test can read back which frame an item's images came from.
    """
    transforms = {
        "w": image_size,
        "h": image_size,
        "fl_x": float(image_size),
        "fl_y": float(image_size),
        "cx": image_size / 2,
        "cy": image_size / 2,
    }
    transforms.update(_transforms_with_x_positions(x_positions))
    (tmp_path / "dl3dv" / SUBDIR).mkdir(parents=True, exist_ok=True)
    with ZipFile(tmp_path / "dl3dv" / SUBDIR / f"{scene_id}.zip", "w") as zf:
        zf.writestr(f"{scene_id}/transforms.json", json.dumps(transforms))
        for i, frame in enumerate(transforms["frames"]):
            png = BytesIO()
            Image.new("RGB", (image_size, image_size), (50 + 10 * i,) * 3).save(png, format="PNG")
            zf.writestr(f"{scene_id}/{frame['file_path'].replace('images/', 'images_4/')}", png.getvalue())

    prompt_scene_dir = tmp_path / "prompts" / SUBDIR / scene_id
    prompt_scene_dir.mkdir(parents=True, exist_ok=True)
    prompt_name = prompt_filename or f"frames_{NUM_FRAMES}_stride_1.h5"
    with h5py.File(prompt_scene_dir / prompt_name, "w") as f:
        for name, image_indices in prompt_windows.items():
            dataset = f.create_dataset(name, data=np.zeros((4, 8), dtype=np.uint16))
            dataset.attrs["image_indices"] = np.asarray(image_indices, dtype=np.int64)
            dataset.attrs["caption"] = "synthetic scene"

    video = encode_frames_to_video(np.full((len(x_positions), image_size, image_size, 3), 100, dtype=np.uint8))
    recon_dir = tmp_path / "recons" / scene_id
    recon_dir.mkdir(parents=True, exist_ok=True)
    for half in halves:
        for num_selected in splits:
            with h5py.File(recon_dir / f"data_{scene_id}_{half}_{num_selected}.h5", "w") as f:
                views = np.arange(num_selected) if selected_indices is None else np.asarray(selected_indices)
                assert len(views) == num_selected
                f.create_dataset("selected_indices", data=views)
                f.create_dataset("render_video", data=np.frombuffer(video, dtype=np.uint8))
                f.create_dataset("opacity_video", data=np.frombuffer(video, dtype=np.uint8))


def _entry(tmp_path: Path, scene_id: str, half: str, scale: float, num_selected_indices: int = 2) -> dict:
    return {
        "scene_half": half,
        "num_selected_indices": num_selected_indices,
        "subdir": SUBDIR,
        "scale": scale,
        "hdf5_path": str(tmp_path / "recons" / scene_id / f"data_{scene_id}_{half}_{num_selected_indices}.h5"),
    }


def _write_split(tmp_path: Path, scenes: dict[str, list[dict]]) -> Path:
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"trainval": scenes, "test": {}}))
    return split_path


def _dataset_kwargs(tmp_path: Path, split_path: Path) -> dict:
    return dict(
        split="trainval",
        split_path=split_path,
        dl3dv_dir=tmp_path / "dl3dv",
        prompt_dir=tmp_path / "prompts",
        num_frames=NUM_FRAMES,
        frames_per_block=None,
        dataset_scaling_factor=0.01,
    )


def _train_dataset(tmp_path: Path, split_path: Path) -> DL3DVPairedDatasetTrain:
    dataset = DL3DVPairedDatasetTrain(num_fetches=1, **_dataset_kwargs(tmp_path, split_path))
    dataset.set_random_splits("cpu", seed=0)
    return dataset


def _val_dataset(tmp_path: Path, split_path: Path) -> DL3DVPairedDatasetTest:
    return DL3DVPairedDatasetTest(num_views=2, **_dataset_kwargs(tmp_path, split_path))


# ---------------------------------------------------------------------------
# Jump-transition metric
# ---------------------------------------------------------------------------


def test_jump_classification_applies_each_entrys_own_scale():
    # Deltas [0.1, 0.1, 2.0, 0.1]: only transition 2 (between frames 2 and 3)
    # can exceed the 1 m threshold, depending on the entry's scale.
    deltas = _scene_camera_position_deltas(_transforms_with_x_positions([0.0, 0.1, 0.2, 2.2, 2.3]))

    assert _pose_jump_transitions_for_scale(deltas, 1.0) == {2}
    # 2.0 x 0.4 = 0.8 m, under the 1 m threshold at this reconstruction scale.
    assert _pose_jump_transitions_for_scale(deltas, 0.4) == frozenset()
    # 2.0 x 0.6 = 1.2 m > 1 m.
    assert _pose_jump_transitions_for_scale(deltas, 0.6) == {2}
    # 0.1 x 6.0 = 0.6 m < 1 m but 2.0 x 6.0 = 12 m: still only the junction.
    assert _pose_jump_transitions_for_scale(deltas, 6.0) == {2}


def test_distance_exactly_at_threshold_is_not_a_jump():
    # Exact float arithmetic: 2.0 x 0.5 = 1.0, and the comparison is strict.
    deltas = _scene_camera_position_deltas(_transforms_with_x_positions([0.0, 2.0]))

    assert _pose_jump_transitions_for_scale(deltas, 0.5) == frozenset()
    assert _pose_jump_transitions_for_scale(deltas, 0.5001) == {0}


def test_scenes_with_fewer_than_two_frames_have_no_transitions():
    assert _scene_camera_position_deltas({"frames": []}).shape == (0, 3)
    single = _scene_camera_position_deltas(_transforms_with_x_positions([1.0]))
    assert single.shape == (0, 3)
    assert _pose_jump_transitions_for_scale(single, 1.0) == frozenset()


def test_jump_classification_ignores_rotation():
    # Only camera centers feed the check. The per-frame 1.5 rad rotation
    # steps alone are ~1.9 in Frobenius norm, so a check that included
    # rotation would also flag the 0.9-unit transition 0 as a jump.
    transforms = _transforms_with_x_positions([0.0, 0.9, 2.4])
    for i, frame in enumerate(transforms["frames"]):
        c, s = math.cos(1.5 * i), math.sin(1.5 * i)
        frame["transform_matrix"][0][:2] = [c, -s]
        frame["transform_matrix"][1][:2] = [s, c]

    assert _pose_jump_transitions_for_scale(_scene_camera_position_deltas(transforms), 1.0) == {1}


def test_has_pose_jump_free_window_segment_arithmetic():
    # 6 frames, jump between frames 2 and 3: segments of 3 and 3 frames.
    assert not _has_pose_jump_free_window({2}, scene_num_frames=6, num_frames=5)
    assert _has_pose_jump_free_window({2}, scene_num_frames=6, num_frames=3)
    # 13 frames, jump at 6: segments of 7 and 6 frames.
    assert _has_pose_jump_free_window({6}, scene_num_frames=13, num_frames=5)
    assert not _has_pose_jump_free_window({6}, scene_num_frames=13, num_frames=8)
    assert _has_pose_jump_free_window(frozenset(), scene_num_frames=5, num_frames=5)


def test_window_span_containment():
    # Transition 6 sits between frames 6 and 7: inside span [low, high] iff
    # low <= 6 <= high - 1.
    assert _window_span_contains_pose_jump([4, 5, 6, 7, 8], frozenset({6}))
    assert _window_span_contains_pose_jump([8, 7, 6, 5, 4], frozenset({6}))  # order-free
    assert _window_span_contains_pose_jump([6, 7], frozenset({6}))
    assert not _window_span_contains_pose_jump([0, 1, 2, 3, 4], frozenset({6}))
    assert not _window_span_contains_pose_jump([7, 8, 9], frozenset({6}))  # starts after the break
    assert not _window_span_contains_pose_jump([6], frozenset({6}))  # a single frame spans no transition
    # Windows with stride > 1 are judged by their span on purpose: the loaded
    # clip would bracket the break even though no two adjacent frames are in it.
    assert _window_span_contains_pose_jump([0, 12], frozenset({6}))


# ---------------------------------------------------------------------------
# Init-time entry drop (train) / retention (val)
# ---------------------------------------------------------------------------


def test_unviable_scene_dropped_from_train_retained_in_val(tmp_path, caplog):
    # droppy's jump at transition 2 leaves segments of 3+3 < 5 frames.
    _write_scene(tmp_path, "droppy", x_positions=[0.0, 0.05, 0.10, 3.10, 3.15, 3.20], prompt_windows={"w0": [0, 1]})
    _write_scene(tmp_path, "keepy", x_positions=[0.05 * i for i in range(6)], prompt_windows={"w0": [0, 1]})
    split_path = _write_split(
        tmp_path,
        {
            "droppy": [_entry(tmp_path, "droppy", "first", 1.0)],
            "keepy": [_entry(tmp_path, "keepy", "first", 1.0)],
        },
    )

    with caplog.at_level(logging.DEBUG, logger=BASE_LOGGER):
        train = _train_dataset(tmp_path, split_path)
    val = _val_dataset(tmp_path, split_path)

    assert len(train.data) == 1
    assert train.data[0][2][0].name == "data_keepy_first_2.h5"
    assert train.data[0][2][2] == frozenset()  # clean scene: computed jump set is empty
    summaries = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert summaries == ["Excluded 1 scene-halves with no pose-jump-free 5-frame window"]
    assert any("droppy" in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    # Val keeps both scenes; val never drops or filters.
    assert len(val.data) == 2


def test_mixed_scene_keeps_clean_entries(tmp_path):
    # Same scene, two halves with different recon scales: the 3.0-unit jump is
    # 3 m at scale 1.0 but 0.03 m at scale 0.01, so only one half drops.
    _write_scene(
        tmp_path,
        "mixed",
        x_positions=[0.0, 0.05, 0.10, 3.10, 3.15, 3.20],
        prompt_windows={"w0": [0, 1]},
        halves=("first", "second"),
    )
    split_path = _write_split(
        tmp_path,
        {
            "mixed": [
                _entry(tmp_path, "mixed", "first", 1.0),
                _entry(tmp_path, "mixed", "second", 0.01),
            ]
        },
    )

    train = _train_dataset(tmp_path, split_path)

    assert len(train.data) == 1
    assert train.data[0][2][0].name == "data_mixed_second_2.h5"


def test_partially_viable_scene_half_drops_whole_half(tmp_path):
    # One entry of the half is unusable (its scale turns the 3-unit jump into
    # 3 m) while its sibling is fine (0.03 m). Dropping just the bad entry
    # would leave this half with different keys than every other dict in
    # self.data, which the constructor rejects — so the whole half must go.
    _write_scene(
        tmp_path,
        "hetero",
        x_positions=[0.0, 0.05, 0.10, 3.10, 3.15, 3.20],
        prompt_windows={"w0": [0, 1]},
        splits=(2, 3),
    )
    _write_scene(
        tmp_path,
        "clean",
        x_positions=[0.05 * i for i in range(6)],
        prompt_windows={"w0": [0, 1]},
        splits=(2, 3),
    )
    split_path = _write_split(
        tmp_path,
        {
            "hetero": [
                _entry(tmp_path, "hetero", "first", 1.0),  # 3 m jump: unviable
                _entry(tmp_path, "hetero", "first", 0.01, num_selected_indices=3),  # viable sibling
            ],
            "clean": [
                _entry(tmp_path, "clean", "first", 1.0),
                _entry(tmp_path, "clean", "first", 1.0, num_selected_indices=3),
            ],
        },
    )

    train = _train_dataset(tmp_path, split_path)

    assert len(train.data) == 1  # the whole half dropped, viable sibling included
    assert train.available_splits == (2, 3)
    assert train.data[0][2][0].name == "data_clean_first_2.h5"
    # Val keeps both scenes untouched.
    val = _val_dataset(tmp_path, split_path)
    assert len(val.data) == 2


def test_filtering_requires_explicit_num_frames():
    with pytest.raises(ValueError, match="explicit num_frames"):
        DL3DVPairedDatasetBase(
            split="trainval",
            split_path=Path("/nonexistent.json"),
            dl3dv_dir=Path("/nonexistent"),
            prompt_dir=Path("/nonexistent"),
            num_frames=None,
            frames_per_block=None,
            return_unencoded_prompt=False,
            dataset_scaling_factor=0.01,
            filter_pose_jump_windows=True,
        )


# ---------------------------------------------------------------------------
# Fetch-time candidate filtering (train)
# ---------------------------------------------------------------------------


def _jumpy_positions() -> list[float]:
    # 13 frames with a 3-unit jump between frames 6 and 7. Segments of 7 and 6
    # frames both fit a 5-frame window, so the entry survives construction.
    return [0.05 * i for i in range(7)] + [0.05 * i + 3.0 for i in range(7, 13)]


@pytest.fixture()
def jumpy_dataset(tmp_path):
    # One poisoned prompt window (span [4, 8] contains the jump at 6) next to
    # one clean window: the filter must leave exactly the clean one.
    _write_scene(
        tmp_path,
        "jumpy",
        x_positions=_jumpy_positions(),
        prompt_windows={"clean0": [0, 1, 2, 3, 4], "poisoned4": [4, 5, 6, 7, 8]},
    )
    split_path = _write_split(tmp_path, {"jumpy": [_entry(tmp_path, "jumpy", "first", 1.0)]})
    dataset = _train_dataset(tmp_path, split_path)
    dataset.iter_plan = torch.tensor([[2, 2]] * 4, dtype=torch.int64)
    dataset.data_perm = torch.tensor([0], dtype=torch.int64)
    return dataset


def test_fetches_sample_only_jump_free_windows(jumpy_dataset):
    assert jumpy_dataset.data[0][2][2] == frozenset({6})  # computed from the transforms

    prompt_path = jumpy_dataset.prompts_by_scene_id["jumpy"][0]
    with h5py.File(prompt_path, "r") as f:
        candidates = jumpy_dataset._pose_jump_free_prompt_windows(f, prompt_path, frozenset({6}))

    assert [name for name, _ in candidates] == ["clean0"]
    assert (prompt_path, frozenset({6})) in jumpy_dataset._pose_jump_free_windows_cache

    # Every fetch draws from the filtered candidates, so every fetch loads
    # successfully and serves its own scene.
    for idx in range(4):
        torch.manual_seed(idx)
        item = jumpy_dataset[idx]
        assert item["rgb_gt"].shape == (NUM_FRAMES, 3, 16, 16)


def test_prompt_file_without_jump_free_window_is_reported(tmp_path):
    # Construction keeps the entry (segments of 7 and 6 frames fit a 5-frame
    # window), but every window in the prompt file crosses the jump. That
    # mismatch must raise an error instead of sampling a bad window.
    _write_scene(tmp_path, "jumpy", x_positions=_jumpy_positions(), prompt_windows={"poisoned": [6, 7]})
    split_path = _write_split(tmp_path, {"jumpy": [_entry(tmp_path, "jumpy", "first", 1.0)]})
    dataset = _train_dataset(tmp_path, split_path)
    dataset.iter_plan = torch.tensor([[2, 2]], dtype=torch.int64)
    dataset.data_perm = torch.tensor([0], dtype=torch.int64)

    with pytest.raises(ValueError, match="no pose-jump-free window"):
        dataset[0]


def test_val_dataset_never_filters_windows(tmp_path):
    # The same prompt coverage the train dataset refuses to sample loads fine
    # on val, which never drops or filters.
    _write_scene(tmp_path, "jumpy", x_positions=_jumpy_positions(), prompt_windows={"poisoned": [6, 7]})
    split_path = _write_split(tmp_path, {"jumpy": [_entry(tmp_path, "jumpy", "first", 1.0)]})
    dataset = _val_dataset(tmp_path, split_path)

    item = dataset[0]

    assert item["rgb_gt"].shape == (NUM_FRAMES, 3, 16, 16)


def test_clean_scenes_never_pay_for_the_filter(tmp_path):
    _write_scene(tmp_path, "clean", x_positions=[0.05 * i for i in range(13)], prompt_windows={"w0": [0, 1, 2, 3, 4]})
    split_path = _write_split(tmp_path, {"clean": [_entry(tmp_path, "clean", "first", 1.0)]})

    dataset = _train_dataset(tmp_path, split_path)
    dataset.iter_plan = torch.tensor([[2, 2]], dtype=torch.int64)
    dataset.data_perm = torch.tensor([0], dtype=torch.int64)
    item = dataset[0]

    assert item["rgb_gt"].shape == (NUM_FRAMES, 3, 16, 16)
    assert dataset.data[0][2][2] == frozenset()
    assert not dataset._pose_jump_free_windows_cache  # an empty jump set skips the filter entirely


# ---------------------------------------------------------------------------
# Same-segment neighbor sampling
# ---------------------------------------------------------------------------


def _frame_ids_from_images(images: torch.Tensor) -> set[int]:
    """Recover frame indices from the fixture's per-frame gray values."""
    return {int(round((float(frame.mean()) * 255 - 50) / 10)) for frame in images}


def test_windows_sharing_a_segment_all_pair_with_that_segments_pool():
    # Production prompt files have one window per start frame, so several
    # windows in one segment is the normal shape. Pairing pools by segment
    # order instead of window membership would hand a1 the other segment's
    # views.
    candidates = _windows_with_same_segment_neighbors(
        [("a0", 0), ("a1", 1), ("b7", 7)], frozenset({6}), 13, np.array([0, 1, 8]), 2, Path("prompts.h5")
    )
    assert [(name, pool.tolist()) for name, pool in candidates] == [("a0", [0, 1]), ("a1", [0, 1])]


def test_segment_neighbor_capacity_counts_only_segments_that_fit_a_window():
    # Segment b (frames 7-12) holds two views; segment a (frames 0-6) holds one.
    assert _segment_neighbor_capacity(frozenset({6}), 13, NUM_FRAMES, np.array([0, 7, 8])) == 2
    # A segment too short for a window contributes nothing even with views in it:
    # jumps at 2 and 6 leave runs of 3, 4, and 6 frames; only the last fits a
    # 5-frame window and it holds one view.
    assert _segment_neighbor_capacity(frozenset({2, 6}), 13, NUM_FRAMES, np.array([0, 1, 3, 4, 8])) == 1
    assert _segment_neighbor_capacity(frozenset({6}), 13, NUM_FRAMES, np.array([], dtype=np.int64)) == 0
    # A segment exactly num_frames long qualifies (jump at 4 leaves frames 0-4).
    assert _segment_neighbor_capacity(frozenset({4}), 13, NUM_FRAMES, np.array([0, 1, 2])) == 3
    # A view on a segment's last frame belongs to that segment.
    assert _segment_neighbor_capacity(frozenset({6}), 13, NUM_FRAMES, np.array([5, 6])) == 2


def test_views_on_a_segments_last_frame_stay_in_its_pool():
    candidates = _windows_with_same_segment_neighbors(
        [("a", 0)], frozenset({6}), 13, np.array([5, 6]), 2, Path("prompts.h5")
    )
    assert [(name, pool.tolist()) for name, pool in candidates] == [("a", [5, 6])]


def _two_segment_dataset(tmp_path, selected_indices: list[int]):
    # 13 frames, jump at transition 6; one clean window per segment.
    _write_scene(
        tmp_path,
        "jumpy",
        x_positions=_jumpy_positions(),
        prompt_windows={"a0": [0, 1, 2, 3, 4], "b7": [7, 8, 9, 10, 11]},
        splits=(len(selected_indices),),
        selected_indices=selected_indices,
    )
    split_path = _write_split(
        tmp_path, {"jumpy": [_entry(tmp_path, "jumpy", "first", 1.0, num_selected_indices=len(selected_indices))]}
    )
    return _train_dataset(tmp_path, split_path)


def test_neighbors_come_from_the_target_windows_segment(tmp_path):
    dataset = _two_segment_dataset(tmp_path, selected_indices=[0, 1, 7, 8])

    seen_segments = set()
    for seed in range(24):
        generator = torch.Generator().manual_seed(seed)
        item = dataset.get_item_inner(0, 4, 2, generator=generator)
        target_frames = _frame_ids_from_images(item["rgb_gt"])
        neighbor_frames = _frame_ids_from_images(item["rgb_neighbors"])
        if target_frames <= set(range(0, 7)):
            assert neighbor_frames <= {0, 1}
            seen_segments.add("a")
        else:
            assert target_frames <= set(range(7, 13))
            assert neighbor_frames <= {7, 8}
            seen_segments.add("b")

    # Both windows can supply two same-segment neighbors, so both stay in play.
    assert seen_segments == {"a", "b"}


def test_dropped_window_is_never_served_and_pools_stay_paired(tmp_path):
    # Segment a holds one training view, segment b holds two. With two
    # neighbors requested, window a0 must be dropped: every fetch serves the
    # b7 window with segment-b neighbors. Pins the pairing between the kept
    # window list and its pools (a mis-indexed unfiltered list would serve a0
    # or cross-segment neighbors).
    dataset = _two_segment_dataset(tmp_path, selected_indices=[0, 7, 8])

    for seed in range(12):
        item = dataset.get_item_inner(0, 3, 2, generator=torch.Generator().manual_seed(seed))
        assert _frame_ids_from_images(item["rgb_gt"]) <= set(range(7, 13))
        assert _frame_ids_from_images(item["rgb_neighbors"]) <= {7, 8}


def test_scene_sits_out_draws_it_cannot_supply(tmp_path):
    # The jumpy scene holds at most two views per segment; a three-neighbor
    # draw must be served by the next scene in the shuffled order instead.
    # Disjoint training views make the serving scene identifiable from the
    # neighbor images.
    _write_scene(
        tmp_path,
        "jumpy",
        x_positions=_jumpy_positions(),
        prompt_windows={"a0": [0, 1, 2, 3, 4], "b7": [7, 8, 9, 10, 11]},
        splits=(4,),
        selected_indices=[0, 1, 7, 8],
    )
    _write_scene(
        tmp_path,
        "clean",
        x_positions=[0.05 * i for i in range(13)],
        prompt_windows={"w2": [2, 3, 4, 5, 6]},
        splits=(4,),
        selected_indices=[3, 4, 5, 6],
    )
    split_path = _write_split(
        tmp_path,
        {
            "jumpy": [_entry(tmp_path, "jumpy", "first", 1.0, num_selected_indices=4)],
            "clean": [_entry(tmp_path, "clean", "first", 1.0, num_selected_indices=4)],
        },
    )
    dataset = _train_dataset(tmp_path, split_path)
    jumpy_idx = next(i for i in range(len(dataset.data)) if "jumpy" in str(dataset.data[i][4][0]))
    dataset.data_perm = torch.tensor([jumpy_idx, 1 - jumpy_idx], dtype=torch.int64)

    # Three neighbors: jumpy cannot supply them from one segment, clean can.
    dataset.iter_plan = torch.tensor([[4, 3]], dtype=torch.int64)
    torch.manual_seed(0)
    item = dataset[0]
    assert item["rgb_neighbors"].shape[0] == 3
    assert _frame_ids_from_images(item["rgb_neighbors"]) <= {3, 4, 5, 6}

    # Two neighbors: jumpy can supply them itself, so it is not skipped.
    dataset.iter_plan = torch.tensor([[4, 2]], dtype=torch.int64)
    torch.manual_seed(0)
    item = dataset[0]
    assert _frame_ids_from_images(item["rgb_neighbors"]) <= {0, 1} or _frame_ids_from_images(item["rgb_neighbors"]) <= {
        7,
        8,
    }


def test_prompt_file_not_covering_a_viable_segment_is_reported(tmp_path):
    # Every training view sits in the second segment but the only prompt
    # window is in the first: same-segment sampling is impossible for this
    # prompt file, which is a coverage mismatch, not a samplable state.
    _write_scene(
        tmp_path,
        "jumpy",
        x_positions=_jumpy_positions(),
        prompt_windows={"a0": [0, 1, 2, 3, 4]},
        splits=(2,),
        selected_indices=[7, 8],
    )
    split_path = _write_split(tmp_path, {"jumpy": [_entry(tmp_path, "jumpy", "first", 1.0)]})
    dataset = _train_dataset(tmp_path, split_path)

    with pytest.raises(ValueError, match="no jump-free window in a segment with 2 training views"):
        dataset.get_item_inner(0, 2, 2, generator=torch.Generator().manual_seed(0))


# ---------------------------------------------------------------------------
# Short scenes (fewer frames than the training window)
# ---------------------------------------------------------------------------


def test_short_clean_scene_trains_with_pingpong_expansion(tmp_path):
    # Three frames, no jumps: the full-clip caption window expands to
    # NUM_FRAMES target frames by bouncing over the clip.
    _write_scene(
        tmp_path,
        "short",
        x_positions=[0.05 * i for i in range(3)],
        prompt_windows={"all": [0, 1, 2]},
        prompt_filename="frames_all_stride_1.h5",
    )
    split_path = _write_split(tmp_path, {"short": [_entry(tmp_path, "short", "first", 1.0)]})
    dataset = _train_dataset(tmp_path, split_path)

    item = dataset.get_item_inner(0, 2, 2, generator=torch.Generator().manual_seed(0))

    assert item["rgb_gt"].shape[0] == NUM_FRAMES
    assert _frame_ids_from_images(item["rgb_gt"]) <= {0, 1, 2}


def test_short_scene_with_a_jump_is_dropped_from_train(tmp_path):
    # A jump inside a clip shorter than the window leaves no usable window at
    # all, so the scene-half is dropped at construction like any other
    # unviable entry.
    _write_scene(
        tmp_path,
        "shortjump",
        x_positions=[0.0, 0.05, 3.0],
        prompt_windows={"all": [0, 1, 2]},
        prompt_filename="frames_all_stride_1.h5",
    )
    _write_scene(tmp_path, "clean", x_positions=[0.05 * i for i in range(13)], prompt_windows={"w0": [0, 1, 2, 3, 4]})
    split_path = _write_split(
        tmp_path,
        {
            "shortjump": [_entry(tmp_path, "shortjump", "first", 1.0)],
            "clean": [_entry(tmp_path, "clean", "first", 1.0)],
        },
    )
    dataset = _train_dataset(tmp_path, split_path)

    assert len(dataset.data) == 1  # only the clean scene survives


def test_short_clean_scene_serves_val_with_pingpong_expansion(tmp_path):
    # Val never filters, but short scenes now enter it too and must expand to
    # the full window length.
    _write_scene(
        tmp_path,
        "short",
        x_positions=[0.05 * i for i in range(3)],
        prompt_windows={"all": [0, 1, 2]},
        prompt_filename="frames_all_stride_1.h5",
    )
    split_path = _write_split(tmp_path, {"short": [_entry(tmp_path, "short", "first", 1.0)]})
    dataset = _val_dataset(tmp_path, split_path)

    item = dataset[0]

    assert item["rgb_gt"].shape[0] == NUM_FRAMES
    assert _frame_ids_from_images(item["rgb_gt"]) <= {0, 1, 2}
