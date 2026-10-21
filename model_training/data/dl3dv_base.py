# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import math
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import h5py
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from model_training.data.reconstruction_path import parse_reconstruction_hdf5_path
from model_training.data.scene_utils import load_scene_transforms_from_zip

# The prompt-payload contract lives ONCE in data/utils.py; the train path
# re-exports it so train and eval can never diverge on how prompt files are
# decoded.
from model_training.data.utils import (
    compute_camera_conditioning,
    frame_intrinsics_for,
    load_frames_from_zip,
    load_prompt_payload,
    resize_to_multiple,
)
from model_training.model_profiles import requires_reference_camera_rays
from model_training.prompt_formats import PROMPT_FORMAT_CHOICES, WAN_T5_PROMPT_FORMAT, caption_filename_for_format
from model_training.utils.validation_artifacts import content_id, file_sha256
from model_training.utils.video_io import decode_video_to_frames

# Permanent data quarantine shared with split generation.
BAD_SCENE_IDS: frozenset[str] = frozenset(
    (Path(__file__).resolve().parents[2] / "data_processing" / "bad_scene_ids.txt").read_text().split()
)

logger = logging.getLogger(__name__)


# About 9% of DL3DV scenes have broken pose tracks: the video is continuous,
# but the recorded camera position jumps between some consecutive frames. We
# treat a transition as broken when the two camera positions (in meters, using
# the scene's reconstruction scale) are more than POSE_JUMP_MAX_TRANSLATION_M
# apart — real camera motion between adjacent video frames is never that
# large. Healthy scenes fall far below this threshold and broken ones far
# above it, so the exact value is not sensitive.
POSE_JUMP_MAX_TRANSLATION_M = 1.0


# The eval layout writes "<base>.cosmos3.h5" beside the training captions; the
# "{stem}_*.h5" glob below must not pull them into the prompt pool.
_FORMAT_TAGGED_SUFFIXES: frozenset[str] = frozenset(
    caption_filename_for_format(".h5", prompt_format) for prompt_format in PROMPT_FORMAT_CHOICES
) - {".h5"}


def _default_prompt_paths_for_scene(
    scene_prompt_dir: Path, num_frames: int | None, scene_num_frames: int
) -> list[Path]:
    """Prompt files for one scene, preferring windowed captions when they fit.

    Scenes shorter than the training window carry a single full-clip
    frames_all caption; training expands its indices to num_frames with a
    pingpong walk. Two exact-stem globs: "stride_1*" would also match
    stride_10..19.
    """
    if num_frames is None or scene_num_frames < num_frames:
        stem = "frames_all_stride_1"
    else:
        stem = f"frames_{num_frames}_stride_1"
    candidates = [*scene_prompt_dir.glob(f"{stem}.h5"), *scene_prompt_dir.glob(f"{stem}_*.h5")]
    return sorted(
        path for path in candidates if not any(path.name.endswith(suffix) for suffix in _FORMAT_TAGGED_SUFFIXES)
    )


def _pingpong_frame_indices(
    frame_indices,
    num_frames: int,
    *,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Walk num_frames steps over the prompt indices, bouncing between the ends.

    Starts at a random index in a random direction. For windows of exactly
    num_frames indices this visits frames the same way the training data
    always has; for shorter windows (scenes below the training length, whose
    captions cover the whole clip) it repeats the clip forward and backward
    instead of padding one endpoint.
    """
    original_frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    assert len(original_frame_indices) > 0, "Cannot expand an empty prompt frame sequence"
    assert num_frames > 0, f"num_frames must be positive, got {num_frames}"
    if len(original_frame_indices) == 1:
        return np.repeat(original_frame_indices, num_frames)

    start_index = int(torch.randint(0, len(original_frame_indices), size=(1,), generator=generator).item())
    direction = int(torch.randint(0, 2, size=(1,), generator=generator).item()) * 2 - 1

    period = 2 * len(original_frame_indices) - 2
    steps = start_index + direction * np.arange(num_frames)
    positions = np.mod(steps, period)
    positions = np.minimum(positions, period - positions)
    return original_frame_indices[positions]


def _scene_camera_position_deltas(transforms: dict) -> np.ndarray:
    """Camera position change between consecutive frames, [T-1, 3], in reconstruction units.

    Delta i is between ``transforms["frames"][i]`` and ``[i + 1]``. The frames
    array is sorted by file_path, i.e. video order, so these indices line up
    with prompt image_indices and selected_indices.
    """
    if len(transforms["frames"]) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    positions = np.asarray([frame["transform_matrix"] for frame in transforms["frames"]], dtype=np.float32)[:, :3, 3]
    return positions[1:] - positions[:-1]


def _pose_jump_transitions_for_scale(position_deltas: np.ndarray, scale: float) -> frozenset[int]:
    """Indices of transitions where the camera moves more than POSE_JUMP_MAX_TRANSLATION_M meters.

    Deltas are computed once per scene, but each entry applies its own
    reconstruction scale: entries of the same scene can have different scales.
    """
    distances_m = np.linalg.norm(position_deltas * scale, axis=-1)
    return frozenset(np.flatnonzero(distances_m > POSE_JUMP_MAX_TRANSLATION_M).tolist())


def _pose_jump_segments(jump_transitions: frozenset, scene_num_frames: int) -> list[tuple[int, int]]:
    """Inclusive frame ranges of the jump-free runs (transition t sits between frames t and t + 1)."""
    bounds = (-1, *sorted(jump_transitions), scene_num_frames - 1)
    return [(left + 1, right) for left, right in zip(bounds, bounds[1:])]


def _has_pose_jump_free_window(jump_transitions: frozenset, scene_num_frames: int, num_frames: int) -> bool:
    """True if at least one contiguous num_frames-long window avoids every jump.

    A window starting at frame s covers transitions s .. s + num_frames - 2
    (transition t is between frames t and t + 1), so a clean window exists
    exactly when some jump-free run of frames is at least num_frames long.
    """
    return any(hi - lo + 1 >= num_frames for lo, hi in _pose_jump_segments(jump_transitions, scene_num_frames))


def _window_span_contains_pose_jump(image_indices: np.ndarray | list[int], jump_transitions: frozenset) -> bool:
    """True if the window's frame range contains a jump transition.

    Transition t is between frames t and t + 1, so it falls inside the range
    [min, max] of the window's frames when min <= t <= max - 1. The reflection
    walk over a window with a jump-free range can never emit two frames on
    opposite sides of a jump. For stride > 1 windows this check may reject
    more than strictly needed, which is fine: any window whose range contains
    a jump loads frames from both sides of the break.
    """
    low = int(min(image_indices))
    high = int(max(image_indices))
    return any(low <= t <= high - 1 for t in jump_transitions)


def _segment_neighbor_capacity(
    jump_transitions: frozenset, scene_num_frames: int, num_frames: int, training_indices: np.ndarray
) -> int:
    """Most training views any single jump-free segment long enough for a window can supply."""
    training_indices = np.asarray(training_indices)
    return max(
        (
            int(((training_indices >= lo) & (training_indices <= hi)).sum())
            for lo, hi in _pose_jump_segments(jump_transitions, scene_num_frames)
            if hi - lo + 1 >= num_frames
        ),
        default=0,
    )


def _windows_with_same_segment_neighbors(
    windows: list[tuple[str, int]],
    jump_transitions: frozenset,
    scene_num_frames: int,
    training_indices: np.ndarray,
    num_neighbors: int,
    prompt_path: Path,
) -> list[tuple[str, np.ndarray]]:
    """Keep windows whose jump-free segment holds enough training views to supply the neighbors.

    Neighbor views must come from the same jump-free segment as the target
    window: poses on opposite sides of a jump are not in one consistent world
    frame, so a cross-jump neighbor would carry a wrong camera relative to the
    targets. Windows whose segment holds fewer than num_neighbors training
    views are dropped. The train sampler only routes a fetch here after
    checking that some long-enough segment holds num_neighbors views, so an
    empty result means this prompt file has no window in any such segment —
    the same coverage mismatch _pose_jump_free_prompt_windows raises for.
    """
    training_indices = np.asarray(training_indices)
    pool_by_segment = {
        (lo, hi): training_indices[(training_indices >= lo) & (training_indices <= hi)]
        for lo, hi in _pose_jump_segments(jump_transitions, scene_num_frames)
    }
    # A jump-free window lies inside exactly one segment, so its lowest frame identifies it.
    pools = [next(pool for (lo, hi), pool in pool_by_segment.items() if lo <= low <= hi) for _, low in windows]
    kept = [(windows[i][0], pool) for i, pool in enumerate(pools) if len(pool) >= num_neighbors]
    if not kept:
        raise ValueError(
            f"Prompt file {prompt_path} has no jump-free window in a segment with {num_neighbors} "
            "training views; its window coverage does not match the scene's jump-free segments "
            "(regenerate the scene's prompt files, or quarantine the scene in bad_scene_ids.txt)."
        )
    return kept


class DL3DVPairedDatasetBase(torch.utils.data.Dataset):

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        prompt_dir: Path,
        num_frames: int | None,
        frames_per_block: int | None,
        return_unencoded_prompt: bool,
        dataset_scaling_factor: float,  # make sure magnitude of translations does not get too large
        max_neighbors: int | None = None,
        verbose: bool = False,
        skip_vae_check: bool = False,
        filter_pose_jump_windows: bool = False,
        resize_multiple: int = 16,
        prompt_format: str = WAN_T5_PROMPT_FORMAT,
        camera_pose_alignment: str = "mean",
        capture_validation_metadata: bool = False,
    ):
        self.dataset_scaling_factor = dataset_scaling_factor
        self.dl3dv_dir = dl3dv_dir
        self.skip_vae_check = skip_vae_check
        self.return_unencoded_prompt = return_unencoded_prompt
        self.capture_validation_metadata = capture_validation_metadata
        self.dl3dv_downsample_factor = 4
        self.resize_multiple = resize_multiple
        self.prompt_format = prompt_format
        self.camera_pose_alignment = camera_pose_alignment
        # Set only by the train subclass. When on, jump transitions are found
        # from the scene transforms at construction, scene-halves with no
        # usable window are dropped, sampling only draws prompt windows that
        # avoid the jumps, and neighbor views are drawn from the target
        # window's own jump-free segment (a scene sits out draws it cannot
        # serve; see DL3DVPairedDatasetTrain.__getitem__). Val/test never
        # drop or filter.
        self.filter_pose_jump_windows = filter_pose_jump_windows
        # (prompt file, jump set) -> (name, lowest frame) of that file's
        # jump-free prompt windows. Recomputing gives the same answer, so
        # each dataloader worker keeping its own copy is fine.
        self._pose_jump_free_windows_cache: dict[tuple[Path, frozenset], list[tuple[str, int]]] = {}
        # (hdf5 path, jump set) -> most neighbors one jump-free segment can supply for that entry.
        self._segment_capacity_cache: dict[tuple[Path | str, frozenset], int] = {}
        self.frames_per_block = frames_per_block
        # go through all available hdf5 files (one per reconstruction)
        reconstructions_by_scene_id_and_half = defaultdict(dict)
        self.transforms_by_scene_id = {}
        self.prompts_by_scene_id = {}
        self.scene_id_to_subdir = {}
        self.scene_id_zip_has_nested_prefix = {}
        if num_frames is not None and num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if filter_pose_jump_windows and num_frames is None:
            raise ValueError("filter_pose_jump_windows requires an explicit num_frames to know which windows fit")
        self.num_frames = num_frames  # None during auto-discovery, resolved after scene loop

        with split_path.open() as f:
            reconstructions_by_scene_id = json.load(f)[split]

        scene_metadata = self._load_scene_metadata(reconstructions_by_scene_id, prompt_dir, num_frames, verbose)

        unviable_pose_jump_keys: set[tuple[str, str]] = set()
        for scene_id, reconstructions in reconstructions_by_scene_id.items():
            if scene_id in BAD_SCENE_IDS:
                continue
            subdir, transforms, has_nested_prefix, prompt_paths = scene_metadata[scene_id]

            scene_num_frames = len(transforms["frames"])

            if len(prompt_paths) == 0:
                if verbose:
                    print(f"WARNING: No prompt files found for scene {scene_id} in {prompt_dir / subdir / scene_id}")
                continue

            self.scene_id_to_subdir[scene_id] = subdir
            self.transforms_by_scene_id[scene_id] = transforms
            self.prompts_by_scene_id[scene_id] = prompt_paths
            self.scene_id_zip_has_nested_prefix[scene_id] = has_nested_prefix
            if num_frames is None:
                self.num_frames = max(self.num_frames or 0, scene_num_frames)

            position_deltas = _scene_camera_position_deltas(transforms) if filter_pose_jump_windows else None
            for reconstruction in reconstructions:
                num_selected_indices = int(reconstruction["num_selected_indices"])
                if max_neighbors is not None and num_selected_indices > max_neighbors:
                    continue
                hdf5_path = Path(reconstruction["hdf5_path"])
                # Explicit raise (not assert) so validation survives `python -O`.
                if not hdf5_path.exists():
                    raise FileNotFoundError(f"HDF5 file {hdf5_path} does not exist")
                scale = float(reconstruction["scale"])
                if not (math.isfinite(scale) and scale > 0):
                    raise ValueError(f"HDF5 file {hdf5_path} has scale {scale} in split JSON; expected finite > 0")
                pose_jump_transitions = (
                    _pose_jump_transitions_for_scale(position_deltas, scale)
                    if filter_pose_jump_windows
                    else frozenset()
                )
                recon_key = (scene_id, reconstruction["scene_half"])
                reconstructions_by_scene_id_and_half[recon_key][num_selected_indices] = (
                    hdf5_path,
                    scale,
                    pose_jump_transitions,
                )
                # A scene shorter than the training window is one whole-clip
                # window (pingpong-expanded), so it is usable only when it has
                # no jumps at all.
                if filter_pose_jump_windows and not _has_pose_jump_free_window(
                    pose_jump_transitions, scene_num_frames, min(num_frames, scene_num_frames)
                ):
                    unviable_pose_jump_keys.add(recon_key)

        # Drop the whole scene-half, never a single entry: every dict in
        # self.data must have the same keys (checked below), so removing one
        # entry would break that. Dropped scenes stay in val/test, which
        # never filter.
        for key in unviable_pose_jump_keys:
            del reconstructions_by_scene_id_and_half[key]
        if unviable_pose_jump_keys:
            logger.info(
                "Excluded %d scene-halves with no pose-jump-free %d-frame window",
                len(unviable_pose_jump_keys),
                num_frames,
            )
            logger.debug("Pose-jump-excluded scene-halves: %s", sorted(unviable_pose_jump_keys))

        self.available_splits: tuple[int, ...] = ()
        canonical_key: tuple[str, str] | None = None
        self.data = []
        for key, recons in sorted(reconstructions_by_scene_id_and_half.items()):
            splits = tuple(sorted(recons.keys()))
            if canonical_key is None:
                self.available_splits = splits
                canonical_key = key
            elif splits != self.available_splits:
                raise ValueError(
                    f"Heterogeneous reconstruction splits: scene-half {key} has {splits}, "
                    f"but {canonical_key} established {self.available_splits}"
                )
            self.data.append(recons)

        if verbose:
            print(f"Found {len(self.data)} data items with splits {self.available_splits}")

        if self.num_frames is None or self.num_frames <= 0:
            raise ValueError("No scenes found with frames - cannot determine num_frames")
        if num_frames is None:
            # Auto-discovered max frame count: round up to satisfy VAE and AR constraints.
            vae_acceptable = math.ceil((self.num_frames - 1) / 4) * 4 + 1
            if self.frames_per_block is not None:
                num_blocks = (vae_acceptable - 1) // 4 + 1
                ar_blocks = math.ceil(num_blocks / self.frames_per_block) * self.frames_per_block
                self.num_frames = (ar_blocks - 1) * 4 + 1
            else:
                self.num_frames = vae_acceptable
        else:
            # Explicitly provided: strict validation; the user should pass a valid value.
            if (self.num_frames - 1) % 4 != 0:
                raise ValueError(
                    f"(num_frames - 1) must be divisible by 4 (VAE temporal scale factor), "
                    f"but got num_frames={self.num_frames}. Try {(self.num_frames - 1) // 4 * 4 + 1} or "
                    f"{((self.num_frames - 1) // 4 + 1) * 4 + 1}."
                )
            if self.frames_per_block is not None:
                latent_frames = (self.num_frames - 1) // 4 + 1
                if latent_frames % self.frames_per_block != 0:
                    raise ValueError(
                        f"Latent frame count must be divisible by frames_per_block={self.frames_per_block}, "
                        f"but got {latent_frames} latent frames (from num_frames={self.num_frames})."
                    )
        latent_frames = (self.num_frames - 1) // 4 + 1
        if verbose:
            print(f"Using {self.num_frames} frames ({latent_frames} latent frames)")

    def _prompt_paths_for_scene(
        self, scene_prompt_dir: Path, num_frames: int | None, scene_num_frames: int
    ) -> list[Path]:
        """Prompt files a fetch may draw from, starting at frame stride one
        since other strides might be too choppy. Evaluation subclasses narrow
        this to one file so metrics never depend on the draw."""
        return _default_prompt_paths_for_scene(scene_prompt_dir, num_frames, scene_num_frames)

    def _load_scene_metadata(
        self,
        reconstructions_by_scene_id: dict[str, Any],
        prompt_dir: Path,
        num_frames: int | None,
        verbose: bool,
    ) -> dict[str, tuple[str, dict, bool, list[Path]]]:
        """Per-scene (subdir, transforms, has_nested_prefix, prompt_paths) for non-quarantined scenes.

        Opening every scene zip on shared storage to parse transforms.json is the slow part
        of dataset construction, so when torch.distributed is initialized only rank 0 scans
        and the result is broadcast to the other ranks.

        Runs once per rank, from __init__ on the main process. Keep it there: DataLoader
        workers cannot run collectives, so the broadcast would hang if this became lazy.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        metadata: dict[str, tuple[str, dict, bool, list[Path]]] | None = None
        failure: str | None = None
        # Rank 0 can spend more than the default process-group timeout scanning shared storage.
        metadata_group = (
            dist.new_group(backend="gloo", timeout=timedelta(minutes=60)) if dist.is_initialized() else None
        )
        try:
            if rank == 0:
                metadata = {}
                try:
                    for scene_id, reconstructions in tqdm(reconstructions_by_scene_id.items(), disable=not verbose):
                        if scene_id in BAD_SCENE_IDS:
                            continue
                        subdir = reconstructions[0]["subdir"]
                        with ZipFile(self.dl3dv_dir / subdir / f"{scene_id}.zip", "r") as zf:
                            transforms, scene_root = load_scene_transforms_from_zip(zf, scene_id)
                        prompt_paths = self._prompt_paths_for_scene(
                            prompt_dir / subdir / scene_id, num_frames, len(transforms["frames"])
                        )
                        metadata[scene_id] = (subdir, transforms, bool(scene_root), prompt_paths)
                except Exception as e:
                    if not dist.is_initialized():
                        raise
                    # Surface the error on every rank instead of hanging the others at the broadcast.
                    failure = f"Failed to load scene metadata: {e!r}"

            if dist.is_initialized():
                values = [metadata, failure]
                dist.broadcast_object_list(values, src=0, group=metadata_group)
                metadata, failure = values
        finally:
            if metadata_group is not None:
                dist.destroy_process_group(metadata_group)
        if failure is not None:
            raise RuntimeError(failure)
        return metadata

    def can_supply_same_segment_neighbors(self, data_idx: int, num_train_views: int, num_neighbors: int) -> bool:
        """True if this entry can serve num_neighbors neighbor views from one jump-free segment.

        Scenes without jumps always can. For jump scenes the answer is cached:
        it needs the entry's training views, which live in the reconstruction
        file.
        """
        hdf5_path, _, jump_transitions = self.data[data_idx][num_train_views]
        if not (self.filter_pose_jump_windows and jump_transitions):
            return True
        key = (hdf5_path, jump_transitions)
        capacity = self._segment_capacity_cache.get(key)
        if capacity is None:
            scene_id, _, _ = parse_reconstruction_hdf5_path(hdf5_path)
            with h5py.File(hdf5_path, "r") as f:
                training_indices = f["selected_indices"][:]
            capacity = _segment_neighbor_capacity(
                jump_transitions,
                len(self.transforms_by_scene_id[scene_id]["frames"]),
                self.num_frames,
                training_indices,
            )
            self._segment_capacity_cache[key] = capacity
        return capacity >= num_neighbors

    def _pose_jump_free_prompt_windows(
        self, prompt_file: h5py.File, prompt_path: Path, jump_transitions: frozenset
    ) -> list[tuple[str, int]]:
        """(name, lowest frame) of the prompt windows that avoid every jump.

        A kept entry always has at least one such window: the constructor
        already dropped entries without a long-enough jump-free run of frames,
        and prompt generation writes one window per start frame. An empty
        result means this prompt file does not cover the scene the way
        training expects, so raise rather than return it.
        """
        key = (prompt_path, jump_transitions)
        candidates = self._pose_jump_free_windows_cache.get(key)
        if candidates is None:
            candidates = []
            for name in prompt_file.keys():
                image_indices = prompt_file[name].attrs["image_indices"]
                if not _window_span_contains_pose_jump(image_indices, jump_transitions):
                    candidates.append((name, int(min(image_indices))))
            if not candidates:
                raise ValueError(
                    f"Prompt file {prompt_path} has no pose-jump-free window for jump transitions "
                    f"{sorted(jump_transitions)}; its window coverage does not match the scene's "
                    "jump-free segments (regenerate the scene's prompt files, or quarantine the "
                    "scene in bad_scene_ids.txt)."
                )
            self._pose_jump_free_windows_cache[key] = candidates
        return candidates

    def get_item_inner(
        self,
        data_idx: int,
        num_train_views: int,
        num_neighbors: int,
        override_frame_indices: list[int] | None = None,
        override_neighbor_indices: list[int] | None = None,
        generator: torch.Generator | None = None,
        temporal_slice: dict[str, Any] | None = None,
        validation_selection: dict | None = None,
    ) -> dict[str, Any]:
        hdf5_path, scale, pose_jump_transitions = self.data[data_idx][num_train_views]

        scene_id, scene_half, _ = parse_reconstruction_hdf5_path(hdf5_path)
        transforms = self.transforms_by_scene_id[scene_id]

        source_identity = None
        capture_metadata = self.capture_validation_metadata or validation_selection is not None
        if capture_metadata:
            source_identity = {
                "geometry_sha256": content_id(transforms),
                "scene_archive_sha256": file_sha256(
                    self.dl3dv_dir / self.scene_id_to_subdir[scene_id] / f"{scene_id}.zip"
                ),
                # A manifest also replays regular validation at 6/12 views. Pin every
                # available reconstruction so those variants verify their own scale/assets.
                "reconstructions": {
                    str(views): {"metric_scale": recon_scale, "sha256": file_sha256(recon_path)}
                    for views, (recon_path, recon_scale, _) in self.data[data_idx].items()
                },
            }
            if validation_selection is not None:
                for key, value in source_identity.items():
                    if value != validation_selection["source_identity"][key]:
                        raise ValueError(f"Validation manifest source_identity.{key} mismatch for {scene_id}")

        item = dict()

        load_error = (
            f"Failed to load reconstruction data from {hdf5_path} "
            f"(scene_id={scene_id}, num_train_views={num_train_views})"
        )
        try:
            with h5py.File(hdf5_path, "r") as recon_file:
                training_indices = recon_file["selected_indices"][:]
                # Compressed video bytes; decoded to frames only after the
                # target window is known.
                render_video = recon_file["render_video"][:].tobytes()
                opacity_video = recon_file["opacity_video"][:].tobytes()
        except Exception as e:
            raise RuntimeError(load_error) from e

        prompt_path = self.prompts_by_scene_id[scene_id][
            int(torch.randint(0, len(self.prompts_by_scene_id[scene_id]), size=(1,), generator=generator).item())
        ]
        if validation_selection is not None:
            prompt_path = next(
                path for path in self.prompts_by_scene_id[scene_id] if path.name == validation_selection["prompt_file"]
            )
        with h5py.File(prompt_path, "r") as f:
            if self.filter_pose_jump_windows and pose_jump_transitions:
                windows = self._pose_jump_free_prompt_windows(f, prompt_path, pose_jump_transitions)
                candidates = _windows_with_same_segment_neighbors(
                    windows,
                    pose_jump_transitions,
                    len(transforms["frames"]),
                    training_indices,
                    num_neighbors,
                    prompt_path,
                )
            else:
                candidates = [(name, None) for name in f.keys()]
            choice = int(torch.randint(0, len(candidates), size=(1,), generator=generator))
            window_name, neighbor_pool = candidates[choice]
            if validation_selection is not None:
                window_name = validation_selection["prompt_window"]
            prompt_dataset = f[window_name]
            if override_frame_indices is not None:
                frame_indices = np.asarray(override_frame_indices, dtype=np.int64)
            else:
                frame_indices = _pingpong_frame_indices(
                    prompt_dataset.attrs["image_indices"], self.num_frames, generator=generator
                )

            if temporal_slice is None:
                target_frame_indices = frame_indices
                conditioning_frame_indices = frame_indices
                camera_latent_slice = None
            else:
                vae_raw_start, vae_raw_end, crop_start, crop_end = temporal_slice["vae_raw"]
                cond_raw_start, cond_raw_end = temporal_slice["conditioning_raw"]
                latent_start, latent_end = temporal_slice["latent"]
                target_frame_indices = frame_indices[vae_raw_start:vae_raw_end]
                conditioning_frame_indices = frame_indices[cond_raw_start:cond_raw_end]
                camera_latent_slice = (latent_start, latent_end)

            item.update(
                load_prompt_payload(
                    f,
                    prompt_dataset,
                    prompt_format=self.prompt_format,
                )
            )

            if self.return_unencoded_prompt:
                item["prompt"] = prompt_dataset.attrs["caption"]
                item["prompt_file"] = f"{prompt_path.parent.name}/{prompt_path.name}"

        metric_scale = scale
        geometry_scale = metric_scale * self.dataset_scaling_factor
        try:
            rendered_frames = decode_video_to_frames(render_video, target_frame_indices)
            opacity_frames = decode_video_to_frames(opacity_video, conditioning_frame_indices)[..., :1]
        except Exception as e:
            raise RuntimeError(load_error) from e

        rgb_rendered = torch.FloatTensor(rendered_frames).permute(0, 3, 1, 2) / 255.0
        item["rgb_rendered"] = resize_to_multiple(rgb_rendered, self.resize_multiple)

        rgb_gt_file_paths = [transforms["frames"][x]["file_path"] for x in target_frame_indices]
        if override_neighbor_indices is not None:
            neighbor_indices = np.array(override_neighbor_indices)
        else:
            pool = neighbor_pool if neighbor_pool is not None else training_indices
            neighbor_indices = pool[
                torch.multinomial(torch.ones(len(pool)), num_neighbors, replacement=False, generator=generator).numpy()
            ]
        neighbor_file_paths = [transforms["frames"][x]["file_path"] for x in neighbor_indices]

        has_nested_prefix = self.scene_id_zip_has_nested_prefix[scene_id]
        with ZipFile(self.dl3dv_dir / self.scene_id_to_subdir[scene_id] / f"{scene_id}.zip", "r") as zf:
            rgb_gt = load_frames_from_zip(
                zf, scene_id, rgb_gt_file_paths, has_nested_prefix, self.dl3dv_downsample_factor
            )
            rgb_neighbors = load_frames_from_zip(
                zf, scene_id, neighbor_file_paths, has_nested_prefix, self.dl3dv_downsample_factor
            )

        item["target_h"] = rgb_gt.shape[-2]
        item["target_w"] = rgb_gt.shape[-1]
        item["rgb_gt"] = resize_to_multiple(rgb_gt, self.resize_multiple)
        item["rgb_neighbors"] = resize_to_multiple(rgb_neighbors, self.resize_multiple)
        if capture_metadata:
            sample = {
                "scene_id": scene_id,
                "half": scene_half,
                "prompt_file": prompt_path.name,
                "prompt_window": window_name,
                "prompt_sha256": file_sha256(prompt_path),
                "frame_indices": target_frame_indices.tolist(),
                "neighbor_indices": neighbor_indices.tolist(),
                "resolution": list(item["rgb_gt"].shape[-2:]),
                "source_identity": source_identity,
            }
            if validation_selection is not None:
                for key in ("prompt_sha256", "resolution"):
                    if sample[key] != validation_selection[key]:
                        raise ValueError(f"Validation manifest {key} mismatch for {scene_id}")
            item["validation_sample_json"] = json.dumps(sample)
        opacity = torch.FloatTensor(opacity_frames).permute(0, 3, 1, 2) / 255.0
        item["opacity"] = resize_to_multiple(opacity, self.resize_multiple).squeeze(1)

        item.update(
            compute_camera_conditioning(
                intrinsics=transforms,
                c2ws_world=torch.stack(
                    [torch.FloatTensor(transforms["frames"][idx]["transform_matrix"]) for idx in frame_indices]
                ),
                neighbor_c2ws_world=torch.stack(
                    [torch.FloatTensor(transforms["frames"][idx]["transform_matrix"]) for idx in neighbor_indices]
                ),
                scale=geometry_scale,
                # Camera conditioning matches the rendered (conditioning) resolution.
                image_height=item["rgb_rendered"].shape[-2],
                image_width=item["rgb_rendered"].shape[-1],
                skip_vae_check=self.skip_vae_check,
                latent_slice=camera_latent_slice,
                pose_alignment=self.camera_pose_alignment,
                include_neighbor_rays=requires_reference_camera_rays(self.prompt_format),
                target_intrinsics=frame_intrinsics_for(transforms, frame_indices),
                neighbor_intrinsics=frame_intrinsics_for(transforms, neighbor_indices),
            )
        )

        if temporal_slice is not None:
            item["_cp_data_sharded"] = torch.tensor(True)
            item["_cp_vae_crop"] = torch.tensor([crop_start, crop_end], dtype=torch.int64)
            item["_cp_vae_raw_start"] = torch.tensor(vae_raw_start, dtype=torch.int64)
            item["_cp_conditioning_raw_start"] = torch.tensor(cond_raw_start, dtype=torch.int64)
            item["_cp_global_raw_frames"] = torch.tensor(self.num_frames, dtype=torch.int64)

        return item
