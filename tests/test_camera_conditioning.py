import json
from types import SimpleNamespace
from zipfile import ZipFile

import h5py
import numpy as np
import pytest
import torch

import model_eval.validation as validation_module
import model_training.data.dl3dv_base as dataset_module
import model_training.data.utils as camera_utils
import model_training.trainers.dmd as dmd_module
from model_eval.generation import SamplingConfig
from model_eval.inference import latent_num_frames_from_rgb_num_frames
from model_eval.validation import EvaluationRunner
from model_eval.validation_config import EvaluationConfig, EvaluationContext
from model_eval.validation_scoring import ValidationRecorder
from model_eval.validation_tasks import ValidationTaskSpec, build_validation_task
from model_training.common.context_parallel import cp_rank_temporal_ranges
from model_training.data.dl3dv_base import DL3DVPairedDatasetBase
from model_training.data.dl3dv_test import DL3DVPairedDatasetTest
from model_training.data.dl3dv_train import DL3DVPairedDatasetTrain
from model_training.data.utils import compute_camera_conditioning, compute_camera_rays
from model_training.net.conditioning import validate_source_camera_conditioning_lengths
from model_training.trainers.context_parallel import ContextParallelTrainer
from model_training.trainers.dmd import DMD
from model_training.trainers.trainer import Trainer


def test_compute_camera_conditioning_skips_reference_rays_by_default(monkeypatch):
    intrinsics = {"w": 4, "h": 4, "fl_x": 4.0, "fl_y": 4.0, "cx": 2.0, "cy": 2.0}
    c2ws = torch.eye(4).expand(5, -1, -1).clone()
    neighbor_c2ws = torch.eye(4).expand(3, -1, -1).clone()
    ray_calls = []
    compute_rays = camera_utils._camera_rays_for_intrinsics

    def record_rays(intrinsics, poses, height, width):
        ray_calls.append(intrinsics.shape[0])
        return compute_rays(intrinsics, poses, height, width)

    monkeypatch.setattr(camera_utils, "_camera_rays_for_intrinsics", record_rays)
    item = compute_camera_conditioning(intrinsics, c2ws, neighbor_c2ws, scale=1.0, image_height=4, image_width=4)
    assert sorted(item) == ["Ks", "camera_rays", "neighbor_Ks", "neighbor_w2cs", "w2cs"]
    assert ray_calls == [2]


@pytest.mark.parametrize(
    ("field", "camera_rays", "w2cs", "Ks"),
    [
        ("camera_rays", torch.zeros(1, 4, 4, 6), torch.zeros(2, 4, 4), torch.zeros(2, 3, 3)),
        ("w2cs", torch.zeros(2, 4, 4, 6), torch.zeros(1, 4, 4), torch.zeros(2, 3, 3)),
        ("Ks", torch.zeros(2, 4, 4, 6), torch.zeros(2, 4, 4), torch.zeros(1, 3, 3)),
    ],
)
def test_source_camera_conditioning_rejects_truncated_streams(field, camera_rays, w2cs, Ks):
    with pytest.raises(ValueError, match=field):
        validate_source_camera_conditioning_lengths(
            camera_rays=camera_rays,
            w2cs=w2cs,
            Ks=Ks,
            latent_num_frames=2,
        )


def test_frame_intrinsics_for_preserves_order_defaults_and_empty_selections():
    transforms = dict(
        w=8,
        h=4,
        fl_x=8.0,
        fl_y=4.0,
        cx=4.0,
        cy=2.0,
        k1=0.1,
        frames=[{}, dict(fl_x=12.0, k2=0.2, p1=0.3, p2=0.4)],
    )
    expected = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.1, 0.0, 0.0, 0.0], [1.5, 1.0, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4]])
    torch.testing.assert_close(camera_utils.frame_intrinsics_for(transforms, [1, 0, 1]), expected[[1, 0, 1]])
    empty = camera_utils.frame_intrinsics_for(transforms, [])
    assert empty.shape == (0, 8) and empty.dtype == torch.float32
    top_level = {key: value for key, value in transforms.items() if key != "frames"}
    for scene in (top_level, {**top_level, "frames": []}):
        torch.testing.assert_close(camera_utils.frame_intrinsics_for(scene, [0, 1]), expected[:1].expand(2, -1))
        assert camera_utils.frame_intrinsics_for(scene, []).shape == (0, 8)


@pytest.mark.parametrize("num_frames", [2, 3, 4])
def test_short_eval_chunks_use_ceil_vae_camera_control_length(num_frames):
    intrinsics = {"w": 4, "h": 4, "fl_x": 4.0, "fl_y": 4.0, "cx": 2.0, "cy": 2.0}
    c2ws = torch.eye(4).expand(num_frames, -1, -1).clone()

    conditioning = compute_camera_conditioning(
        intrinsics,
        c2ws,
        torch.eye(4).unsqueeze(0),
        scale=1.0,
        image_height=4,
        image_width=4,
        skip_vae_check=True,
    )
    latent_num_frames = latent_num_frames_from_rgb_num_frames(num_frames, vae_temporal_scale=4)

    assert conditioning["camera_rays"].shape[0] == latent_num_frames
    validate_source_camera_conditioning_lengths(
        camera_rays=conditioning["camera_rays"],
        w2cs=conditioning["w2cs"],
        Ks=conditioning["Ks"],
        latent_num_frames=latent_num_frames,
    )


@pytest.fixture
def joint_cameras():
    poses = torch.eye(4).repeat(13, 1, 1)
    poses[:, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    poses[:, :3, 3] = torch.tensor([3.0, -2.0, 7.0]) + torch.arange(13)[:, None] * torch.tensor([0.25, 0.05, 0.1])
    neighbors = poses[[0, 8, 12]].clone()
    neighbors[1, :3, :3] = torch.eye(3)
    neighbors[2, :3, 3] += torch.tensor([0.5, -0.2, 0.8])
    target_lenses = torch.tensor([1.05, 0.9, 0.45, 0.55, 0.12, -0.02, 0.01, -0.015]).repeat(13, 1)
    neighbor_lenses = torch.stack(
        [
            target_lenses[0],
            torch.tensor([0.85, 1.1, 0.6, 0.4, -0.08, 0.015, -0.01, 0.02]),
            torch.tensor([1.2, 0.8, 0.5, 0.5, 0.15, -0.025, 0.008, -0.006]),
        ]
    )
    return dict(
        intrinsics={"w": 7, "h": 5, "fl_x": 7.0, "fl_y": 5.0, "cx": 3.5, "cy": 2.5},
        c2ws_world=poses,
        neighbor_c2ws_world=neighbors,
        target_intrinsics=target_lenses,
        neighbor_intrinsics=neighbor_lenses,
        scale=2.5,
        image_height=5,
        image_width=7,
        include_neighbor_rays=True,
    )


def test_joint_reference_rays_preserve_distortion_and_share_the_target_origin(joint_cameras):
    original_poses = joint_cameras["c2ws_world"].clone()
    original_neighbors = joint_cameras["neighbor_c2ws_world"].clone()
    item = compute_camera_conditioning(**joint_cameras)
    rays = item["neighbor_camera_rays"]
    assert rays.shape == (3, 5, 7, 6)
    torch.testing.assert_close(rays[0], item["camera_rays"][0])
    torch.testing.assert_close(rays[0, ..., :3], torch.zeros_like(rays[0, ..., :3]))

    targets = original_poses.clone()
    neighbors = original_neighbors.clone()
    targets[:, :3, 3] *= joint_cameras["scale"]
    neighbors[:, :3, 3] *= joint_cameras["scale"]
    relative_poses = torch.linalg.inv(targets[0]) @ neighbors
    directions = rays[..., 3:]
    origins = relative_poses[:, None, None, :3, 3].expand_as(directions)
    torch.testing.assert_close(rays[..., :3], torch.linalg.cross(origins, directions))
    camera_directions = torch.einsum("nhwi,nij->nhwj", directions, relative_poses[:, :3, :3])
    xy = camera_directions[..., :2] / camera_directions[..., 2:]
    x, y = xy.unbind(-1)
    fx, fy, cx, cy, k1, k2, p1, p2 = joint_cameras["neighbor_intrinsics"][:, :, None, None].unbind(1)
    radius2 = x.square() + y.square()
    radial = 1 + k1 * radius2 + k2 * radius2.square()
    projected_x = (fx * (x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x.square())) + cx) * 7
    projected_y = (fy * (y * radial + p1 * (radius2 + 2 * y.square()) + 2 * p2 * x * y) + cy) * 5
    projected = torch.stack([projected_x, projected_y], dim=-1)
    pixels = torch.as_tensor(camera_utils._image_points_for_resolution(5, 7)).reshape(1, 5, 7, 2)
    torch.testing.assert_close(projected, pixels.to(projected).expand_as(projected), atol=2e-4, rtol=1e-4)

    pinhole_lenses = joint_cameras["neighbor_intrinsics"].clone()
    pinhole_lenses[:, 4:] = 0
    pinhole = compute_camera_conditioning(**{**joint_cameras, "neighbor_intrinsics": pinhole_lenses})
    torch.testing.assert_close(item["neighbor_Ks"], pinhole["neighbor_Ks"])
    assert not torch.allclose(rays, pinhole["neighbor_camera_rays"]), "Ks cannot recover the distorted ray map"
    torch.testing.assert_close(item["camera_rays"], pinhole["camera_rays"])
    torch.testing.assert_close(original_poses, joint_cameras["c2ws_world"])
    torch.testing.assert_close(original_neighbors, joint_cameras["neighbor_c2ws_world"])


@pytest.mark.parametrize("pose_alignment", ["mean", "last"])
@pytest.mark.parametrize("latent_slice", [None, (2, 4)])
@pytest.mark.parametrize("include_neighbor_rays", [False, True])
def test_reference_permutations_leave_target_rays_invariant(
    joint_cameras, pose_alignment, latent_slice, include_neighbor_rays
):
    kwargs = dict(
        joint_cameras,
        pose_alignment=pose_alignment,
        latent_slice=latent_slice,
        include_neighbor_rays=include_neighbor_rays,
    )
    original = compute_camera_conditioning(**kwargs)
    permutation = torch.tensor([2, 0, 1])
    permuted = compute_camera_conditioning(
        **{
            **kwargs,
            "neighbor_c2ws_world": kwargs["neighbor_c2ws_world"][permutation],
            "neighbor_intrinsics": kwargs["neighbor_intrinsics"][permutation],
        }
    )
    for key in original:
        expected = original[key][permutation] if key.startswith("neighbor_") else original[key]
        torch.testing.assert_close(permuted[key], expected, atol=0, rtol=0)


@pytest.mark.parametrize("pose_alignment", ["mean", "last"])
def test_joint_camera_cp_slices_keep_every_reference_ray(joint_cameras, pose_alignment):
    full = compute_camera_conditioning(**joint_cameras, pose_alignment=pose_alignment)
    target_only = compute_camera_conditioning(
        **{**joint_cameras, "include_neighbor_rays": False}, pose_alignment=pose_alignment
    )
    for key in target_only:
        torch.testing.assert_close(full[key], target_only[key], atol=0, rtol=0)
    for start, end in ((0, 2), (2, 4)):
        local = compute_camera_conditioning(**joint_cameras, pose_alignment=pose_alignment, latent_slice=(start, end))
        for key in ("camera_rays", "w2cs", "Ks"):
            torch.testing.assert_close(local[key], full[key][start:end])
        for key in ("neighbor_camera_rays", "neighbor_w2cs", "neighbor_Ks"):
            torch.testing.assert_close(local[key], full[key], atol=0, rtol=0)
        collated = torch.utils.data.default_collate([local])
        assert collated["neighbor_camera_rays"].shape == (1, 3, 5, 7, 6)
        assert collated["camera_rays"].shape == (1, 2, 5, 7, 6)


def test_joint_camera_conditioning_accepts_an_empty_reference_set(joint_cameras):
    item = compute_camera_conditioning(
        **{**joint_cameras, "neighbor_c2ws_world": torch.empty(0, 4, 4), "neighbor_intrinsics": torch.empty(0, 8)}
    )
    assert item["neighbor_camera_rays"].shape == (0, 5, 7, 6)
    assert item["camera_rays"].shape == (4, 5, 7, 6)


@pytest.fixture
def paired_camera_scene(tmp_path, monkeypatch, joint_cameras):
    scene_id = "camera_scene"
    poses = torch.cat([joint_cameras["c2ws_world"], joint_cameras["neighbor_c2ws_world"]])
    lenses = torch.cat([joint_cameras["target_intrinsics"], joint_cameras["neighbor_intrinsics"]])
    frames = []
    for index, (pose, lens) in enumerate(zip(poses, lenses)):
        fx, fy, cx, cy, k1, k2, p1, p2 = lens.tolist()
        frames.append(
            dict(
                transform_matrix=pose.tolist(),
                file_path=f"{index:04d}.png",
                fl_x=fx * 7,
                fl_y=fy * 5,
                cx=cx * 7,
                cy=cy * 5,
                k1=k1,
                k2=k2,
                p1=p1,
                p2=p2,
            )
        )
    transforms = dict(joint_cameras["intrinsics"], frames=frames)
    reconstruction_path = tmp_path / f"data_{scene_id}_first_3.h5"
    neighbor_indices = [13, 14, 15]
    with h5py.File(reconstruction_path, "w") as handle:
        handle.create_dataset("selected_indices", data=neighbor_indices)
        handle.create_dataset("render_video", data=np.zeros(1, dtype=np.uint8))
        handle.create_dataset("opacity_video", data=np.zeros(1, dtype=np.uint8))
    prompt_path = tmp_path / "prompt.h5"
    with h5py.File(prompt_path, "w") as handle:
        record = handle.create_dataset("window", data=np.zeros((2, 3), dtype=np.uint16))
        record.attrs["image_indices"] = list(range(13))
        record.attrs["caption"] = "camera test"
    with ZipFile(tmp_path / f"{scene_id}.zip", "w"):
        pass
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "trainval": {
                    scene_id: [
                        dict(num_selected_indices=3, hdf5_path=str(reconstruction_path), scale=0.1, scene_half="first")
                    ]
                }
            }
        )
    )
    monkeypatch.setattr(
        DL3DVPairedDatasetBase,
        "_load_scene_metadata",
        lambda *args: {scene_id: (".", transforms, False, [prompt_path])},
    )
    monkeypatch.setattr(
        dataset_module,
        "decode_video_to_frames",
        lambda video, indices: np.zeros((len(indices), 5, 7, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        dataset_module,
        "load_frames_from_zip",
        lambda zf, scene, paths, nested, downsample: torch.zeros(len(paths), 3, 5, 7),
    )
    monkeypatch.setattr(
        dataset_module, "load_prompt_payload", lambda *args, **kwargs: {"encoded_prompt": torch.ones(2, 3)}
    )
    kwargs = dict(
        split="trainval",
        split_path=split_path,
        dl3dv_dir=tmp_path,
        prompt_dir=tmp_path,
        num_frames=13,
        frames_per_block=2,
        dataset_scaling_factor=25.0,
        resize_multiple=1,
    )
    return kwargs, transforms, neighbor_indices


@pytest.mark.parametrize(
    ("dataset_cls", "extra_kwargs"),
    [
        (DL3DVPairedDatasetBase, {"return_unencoded_prompt": False}),
        (DL3DVPairedDatasetTrain, {"num_fetches": 2, "context_parallel_size": 2}),
        (DL3DVPairedDatasetTest, {"num_views": 3}),
    ],
)
@pytest.mark.parametrize("prompt_format", [camera_utils.WAN_T5_PROMPT_FORMAT, camera_utils.COSMOS3_PROMPT_FORMAT])
def test_paired_datasets_preserve_full_lenses_and_unsliced_reference_rays(
    paired_camera_scene, dataset_cls, extra_kwargs, prompt_format
):
    kwargs, transforms, neighbor_indices = paired_camera_scene
    dataset = dataset_cls(**kwargs, **extra_kwargs, prompt_format=prompt_format)
    include_neighbor_rays = prompt_format == camera_utils.COSMOS3_PROMPT_FORMAT
    item_kwargs = dict(
        num_train_views=3,
        num_neighbors=3,
        override_frame_indices=list(range(13)),
        override_neighbor_indices=neighbor_indices,
    )
    full = dataset.get_item_inner(0, **item_kwargs)
    expected = compute_camera_rays(
        transforms,
        list(range(13)),
        neighbor_indices,
        scale=2.5,
        image_shape=(5, 7),
        include_neighbor_rays=include_neighbor_rays,
    )
    for key in expected:
        torch.testing.assert_close(full[key], expected[key])
    assert ("neighbor_camera_rays" in full) == include_neighbor_rays

    temporal_slice = cp_rank_temporal_ranges(1, 2, total_raw_frames=13, temporal_scale=4, halo_latent_frames=0)
    local = dataset.get_item_inner(0, **item_kwargs, temporal_slice=temporal_slice)
    for key in expected:
        torch.testing.assert_close(local[key], full[key] if key.startswith("neighbor_") else full[key][2:4])
    collated = torch.utils.data.default_collate([local])
    assert collated["camera_rays"].shape == (1, 2, 5, 7, 6)
    assert collated["rgb_neighbors"].shape == (1, 3, 3, 5, 7)
    if include_neighbor_rays:
        assert collated["neighbor_camera_rays"].shape == (1, 3, 5, 7, 6)
    else:
        assert "neighbor_camera_rays" not in collated


class _CameraTransformer:
    def __init__(self, cp_size=1):
        self.cp_rank = cp_size - 1
        self.cp_world_size = cp_size
        self.cp_mesh = None
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (torch.zeros_like(kwargs["hidden_states"]),)

    def disable_context_parallel(self):
        self.cp_mesh = None

    def enable_context_parallel(self, mesh):
        self.cp_mesh = mesh


class _CameraPipe:
    def __init__(self, cp_size=1):
        self.vae = SimpleNamespace(dtype=torch.bfloat16, config=SimpleNamespace(scale_factor_temporal=4))
        self.transformer = _CameraTransformer(cp_size)
        self.ar_frames_per_chunk = 2
        self.generation_calls = []
        self.inference_calls = []
        self.scheduler = SimpleNamespace(
            num_train_timesteps=1,
            timesteps=torch.tensor([500.0]),
            sigmas=torch.tensor([0.5]),
            add_noise=lambda latent, noise, timestep: latent,
            add_noise_multi_timestep=lambda latent, noise, timestep: latent,
            training_target=lambda latent, noise: torch.zeros_like(latent),
            training_weight=lambda timestep: torch.ones_like(timestep),
            training_weight_multi_timestep=lambda timestep: torch.ones_like(timestep),
            step=lambda noise, timestep, latent, **kwargs: latent - noise,
        )

    def encode_video_frames(self, frames):
        return frames[:, ::4].permute(0, 2, 1, 3, 4).to(dtype=self.vae.dtype)

    def encode_video_frames_chunked_for_cp(self, frames, halo, **kwargs):
        return self.encode_video_frames(frames)

    def encode_neighbors(self, frames):
        return frames.permute(0, 2, 1, 3, 4).to(dtype=self.vae.dtype)

    def prepare_latents(self, condition, opacity, first_frame, **kwargs):
        return torch.zeros_like(condition)

    def prepare_transformer_timestep(self, timestep, model_input, **kwargs):
        return timestep

    def generate_samples_from_batch(self, condition, *args, **kwargs):
        self.generation_calls.append(kwargs)
        return condition.clone().requires_grad_(True)

    def forward_inference(self, **kwargs):
        self.inference_calls.append(kwargs)
        return kwargs["rendered_rgb"]

    def forward_inference_local_cp(self, **kwargs):
        self.inference_calls.append(kwargs)
        return torch.zeros(1, 3, kwargs["camera_rays"].shape[1], 2, 3)

    denoise_to_latents = forward_inference_local_cp


def _camera_batch(with_rays):
    data = {
        "rgb_gt": torch.ones(1, 13, 3, 2, 3),
        "rgb_rendered": torch.ones(1, 13, 3, 2, 3),
        "opacity": torch.ones(1, 13, 2, 3),
        "rgb_neighbors": torch.ones(1, 3, 3, 2, 3),
        "encoded_prompt": torch.ones(1, 2, 3),
        "camera_rays": torch.arange(144, dtype=torch.float32).reshape(1, 4, 2, 3, 6) / 100,
        "w2cs": torch.eye(4).repeat(1, 4, 1, 1),
        "Ks": torch.eye(3).repeat(1, 4, 1, 1),
        "neighbor_w2cs": torch.eye(4).repeat(1, 3, 1, 1),
        "neighbor_Ks": torch.eye(3).repeat(1, 3, 1, 1),
    }
    if with_rays:
        data["neighbor_camera_rays"] = torch.arange(108, dtype=torch.float32).reshape(1, 3, 2, 3, 6) / 100
    return data


def _camera_trainer(trainer_cls, cp_size=1, ignore_neighbors=False):
    trainer = object.__new__(trainer_cls)
    trainer.pipe = _CameraPipe(cp_size)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.args = SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        num_frames=13,
        vae_chunk_halo_latent_frames=0,
        text_guidance_scale=2.0,
        num_inference_steps=4,
        max_validation_tasks=1,
    )
    trainer.train_scheduler = trainer.pipe.scheduler
    trainer.frames_per_block = 2
    trainer._zero_encode_cache = {}
    trainer._sample_input_dropout = lambda *args, **kwargs: (False, 0, False, ignore_neighbors)
    trainer._broadcast_cp_tensor = lambda value: value.zero_()
    trainer.get_step = lambda: 0
    return trainer


def _local_camera_batch(data):
    ranges = cp_rank_temporal_ranges(1, 2, total_raw_frames=13, temporal_scale=4, halo_latent_frames=0)
    raw_start, raw_end, crop_start, crop_end = ranges["vae_raw"]
    cond_start, cond_end = ranges["conditioning_raw"]
    start, end = ranges["latent"]
    local = dict(data)
    for key in ("rgb_gt", "rgb_rendered"):
        local[key] = data[key][:, raw_start:raw_end]
    local["opacity"] = data["opacity"][:, cond_start:cond_end]
    for key in ("camera_rays", "w2cs", "Ks"):
        local[key] = data[key][:, start:end]
    local.update(
        _cp_data_sharded=torch.tensor([True]),
        _cp_vae_crop=torch.tensor([[crop_start, crop_end]]),
        _cp_vae_raw_start=torch.tensor([raw_start]),
        _cp_conditioning_raw_start=torch.tensor([cond_start]),
        _cp_global_raw_frames=torch.tensor([13]),
    )
    return local


def _assert_reference_camera_forwarding(kwargs, original, dtype):
    if "neighbor_camera_rays" in original:
        assert kwargs["neighbor_camera_rays"].shape == (1, 3, 2, 3, 6)
        torch.testing.assert_close(kwargs["neighbor_camera_rays"], original["neighbor_camera_rays"].to(dtype=dtype))
    else:
        assert "neighbor_camera_rays" not in kwargs, "Do not add even a None keyword to legacy/Wan calls"


@pytest.mark.parametrize(
    ("trainer_cls", "cp_size"),
    [(Trainer, 1), (ContextParallelTrainer, 1), (ContextParallelTrainer, 2)],
)
@pytest.mark.parametrize("with_rays", [False, True])
@pytest.mark.parametrize("ignore_neighbors", [False, True])
def test_supervised_trainers_forward_whole_reference_rays(trainer_cls, cp_size, with_rays, ignore_neighbors):
    trainer = _camera_trainer(trainer_cls, cp_size, ignore_neighbors)
    data = _camera_batch(with_rays)
    local = _local_camera_batch(data) if cp_size > 1 else data
    loss, _ = trainer.training_batch(local)
    assert torch.isfinite(loss)
    call = trainer.pipe.transformer.calls[0]
    _assert_reference_camera_forwarding(call, data, trainer.pipe.vae.dtype)
    assert call["camera_rays"].shape[1] == 4 // cp_size
    assert call["neighbor_hidden_states"].shape[2] == 3
    assert call["ignore_neighbors"] is ignore_neighbors


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("with_rays", [False, True])
@pytest.mark.parametrize("ignore_neighbors", [False, True])
def test_dmd_generator_and_all_score_branches_forward_whole_reference_rays(
    cp_size, with_rays, ignore_neighbors, monkeypatch
):
    trainer = _camera_trainer(DMD, cp_size, ignore_neighbors)
    cp_group = object()
    trainer.pipe.transformer.cp_mesh = SimpleNamespace(get_group=lambda: cp_group, size=lambda: cp_size)

    def all_reduce(value, group):
        assert group is cp_group
        value.mul_(cp_size)

    def all_gather(outputs, value, group):
        assert group is cp_group
        for output in outputs:
            output.copy_(value)

    monkeypatch.setattr(dmd_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(dmd_module.dist, "all_gather", all_gather)
    trainer.real_score = _CameraTransformer(cp_size)
    trainer.fake_score = _CameraTransformer(cp_size)
    trainer.negative_prompt = torch.zeros(1, 2, 3)
    trainer.is_student_phase = lambda: True
    trainer.get_timestep = lambda batch_size: torch.full((batch_size,), 500.0)
    data = _camera_batch(with_rays)
    local = _local_camera_batch(data) if cp_size > 1 else data
    loss, _ = trainer.training_batch(local)
    assert torch.isfinite(loss)
    _assert_reference_camera_forwarding(trainer.pipe.generation_calls[0], data, trainer.pipe.vae.dtype)
    assert len(trainer.fake_score.calls) == 3
    assert len(trainer.real_score.calls) == 2
    for call in trainer.fake_score.calls + trainer.real_score.calls:
        _assert_reference_camera_forwarding(call, data, trainer.pipe.vae.dtype)
        assert call["camera_rays"].shape[1] == 4 // cp_size
        assert call["local_cp_input"] is (cp_size > 1)
        assert call["frame_offset"] == (0 if cp_size == 1 else 2)
        assert call["ignore_neighbors"] is ignore_neighbors


def test_dmd_stages_reference_rays_without_moving_the_raw_video_batch():
    trainer = _camera_trainer(DMD)
    trainer.accelerator.device = torch.device("meta")
    data = _camera_batch(with_rays=True)
    processed = trainer.process_inputs(data)
    assert all(value.device.type == "cpu" for value in data.values())
    assert processed["camera_rays"].device.type == "cpu"
    assert processed["neighbor_camera_rays"].device.type == "meta"
    assert processed["neighbor_camera_rays"].dtype == trainer.pipe.vae.dtype
    assert processed["neighbor_camera_rays"].shape == (1, 3, 2, 3, 6)


class _CameraDataset:
    num_frames = 13

    def __init__(self, data):
        self.item = {key: value[0] for key, value in data.items()}

    def __getitem__(self, index):
        return self.item


@pytest.mark.parametrize("with_rays", [False, True])
@pytest.mark.parametrize("use_neighbors", [False, True])
def test_validation_tasks_keep_reference_rays_whole_through_cp_and_device_moves(with_rays, use_neighbors):
    trainer = _camera_runner()
    data = _camera_batch(with_rays)
    task = build_validation_task(
        trainer.pipe,
        trainer.config,
        trainer.context,
        {3: _CameraDataset(data)},
        ValidationTaskSpec("camera", 3, use_neighbors=use_neighbors),
        0,
    )
    local, offset, _ = trainer._slice_validation_task_for_cp(task, cp_group_size=2, group_rank=1, global_num_frames=13)
    assert offset == 2
    assert local.camera_rays.shape[1] == 2
    assert local.neighbor_camera_rays is task.neighbor_camera_rays
    if with_rays and use_neighbors:
        torch.testing.assert_close(local.neighbor_camera_rays, data["neighbor_camera_rays"].to(trainer.pipe.vae.dtype))
    else:
        assert local.neighbor_camera_rays is None
    trainer.context.device = torch.device("meta")
    moved = trainer._move_validation_task_to_device(local)
    if with_rays and use_neighbors:
        assert moved.neighbor_camera_rays.shape == (1, 3, 2, 3, 6)
        assert moved.neighbor_camera_rays.device.type == "meta"
    else:
        assert moved.neighbor_camera_rays is None


def _camera_runner(cp_size=1):
    runner = EvaluationRunner(
        _CameraPipe(cp_size),
        EvaluationConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            validation_metrics=False,
            sampling=SamplingConfig(num_inference_steps=4, vae_chunk_halo_latent_frames=0),
        ),
        EvaluationContext(torch.device("cpu")),
    )
    runner.recorder = ValidationRecorder(runner.config, runner.context, 4)
    return runner


def _camera_validation_runner(monkeypatch, cp_size, *, kv=False):
    runner = _camera_runner(cp_size)
    runner.pipe.stages_conditioning_per_chunk = kv
    if kv:
        runner.pipe.transformer.cp_mesh = SimpleNamespace(size=lambda: cp_size)
    runner.recorder.save = lambda *args, **kwargs: None
    runner._gather_cp_validation_latents = lambda output: output
    runner.context.cp_meshes[2] = object()
    runner.pipe.cp_size_candidates = lambda frames, world_size: [cp_size]
    runner.pipe.cp_pass_latent_frames = lambda frames: frames
    runner.pipe.use_local_cp_inference = lambda size, frames: size > 1
    monkeypatch.setattr(validation_module.dist, "is_initialized", lambda: cp_size > 1)
    monkeypatch.setattr(validation_module.dist, "get_world_size", lambda: cp_size)
    monkeypatch.setattr(validation_module.dist, "get_rank", lambda: cp_size - 1)
    monkeypatch.setattr(validation_module.dist, "new_group", lambda **kwargs: object())
    monkeypatch.setattr(validation_module.dist, "destroy_process_group", lambda group: None)
    monkeypatch.setattr(validation_module, "barrier_if_distributed", lambda *args: None)
    return runner


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("with_rays", [False, True])
@pytest.mark.parametrize("use_neighbors", [False, True])
def test_validation_forwards_optional_reference_rays_to_full_and_local_cp_inference(
    monkeypatch, tmp_path, cp_size, with_rays, use_neighbors
):
    trainer = _camera_validation_runner(monkeypatch, cp_size)
    data = _camera_batch(with_rays)
    trainer.generate_tasks(
        {3: _CameraDataset(data)}, [(ValidationTaskSpec("camera", 3, use_neighbors=use_neighbors), 0)], tmp_path
    )
    call = trainer.pipe.inference_calls[0]
    expected = data if use_neighbors else {}
    _assert_reference_camera_forwarding(call, expected, trainer.pipe.vae.dtype)
    assert call["camera_rays"].shape[1] == 4 // cp_size
    assert (call["neighbors"] is not None) is use_neighbors
    if cp_size > 1:
        assert call["local_frame_offset"] == 2


@pytest.mark.parametrize("with_rays", [False, True])
@pytest.mark.parametrize("use_neighbors", [False, True])
def test_dmd_cp_validation_forwards_whole_reference_rays(monkeypatch, tmp_path, with_rays, use_neighbors):
    trainer = _camera_validation_runner(monkeypatch, 2, kv=True)
    data = _camera_batch(with_rays)
    trainer.generate_tasks(
        {3: _CameraDataset(data)}, [(ValidationTaskSpec("camera", 3, use_neighbors=use_neighbors), 0)], tmp_path
    )
    call = trainer.pipe.inference_calls[0]
    _assert_reference_camera_forwarding(call, data if use_neighbors else {}, trainer.pipe.vae.dtype)
    assert call["camera_rays"].shape[1] == 4
    assert (call["neighbors"] is not None) is use_neighbors
