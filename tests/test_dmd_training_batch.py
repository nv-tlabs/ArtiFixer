# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import types
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import wandb
from torch.distributed import DeviceMesh

from model_eval.validation_tasks import ValidationTaskSpec
from model_training.model_profiles import default_camera_pose_alignment, is_cosmos3_model
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.trainers.dmd import DMD
from model_training.utils.train_utils import get_train_dataloader
from tests._helpers import (
    _advance_dmd_ema,
    _advance_ema_stage,
    _ema_stage_trainer,
    _run_cp_workers,
    _small_dmd_ema_trainer,
)
from tests.test_pose_jump_window_filtering import _entry, _write_scene, _write_split
from tests.test_prompt_payload_loading import _write_cosmos3_prompt_file


def _processed_batch() -> dict:
    return {
        "condition": torch.zeros(1),
        "opacity": torch.zeros(1),
        "neighbors_condition": torch.zeros(1),
        "camera_rays": torch.zeros(1),
        "w2cs": torch.zeros(1),
        "neighbor_w2cs": torch.zeros(1),
        "Ks": torch.zeros(1),
        "neighbor_Ks": torch.zeros(1),
        "encoded_prompt": torch.zeros(1),
        "ignore_neighbors": True,
    }


class _CapturingPipe:
    def __init__(self, prediction: torch.Tensor):
        self.prediction = prediction
        self.calls = []
        self.grad_enabled = []

    def generate_samples_from_batch(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.prediction


def _make_dmd(processed: dict, *, is_student: bool, prediction: torch.Tensor):
    dmd = object.__new__(DMD)
    dmd.args = types.SimpleNamespace(
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        num_inference_steps=4,
    )
    dmd.pipe = _CapturingPipe(prediction)
    state = {}

    def process_inputs(self, data):
        state["raw_data"] = data
        return processed

    def is_student_phase(self):
        return is_student

    def critic_loss(self, data, critic_prediction):
        state["critic_data"] = data
        state["critic_prediction"] = critic_prediction
        return torch.tensor(2.0, requires_grad=True)

    def generator_loss(self, data, generator_prediction):
        state["generator_data"] = data
        state["generator_prediction"] = generator_prediction
        return torch.tensor(3.0, requires_grad=True)

    dmd.process_inputs = types.MethodType(process_inputs, dmd)
    dmd.is_student_phase = types.MethodType(is_student_phase, dmd)
    dmd.critic_loss = types.MethodType(critic_loss, dmd)
    dmd.generator_loss = types.MethodType(generator_loss, dmd)
    return dmd, state


def test_process_inputs_keeps_raw_video_on_cpu_until_encoding(monkeypatch):
    dmd = object.__new__(DMD)
    dmd.args = types.SimpleNamespace(dropout_rate=0.5, vae_chunk_halo_latent_frames=2)
    dmd.accelerator = types.SimpleNamespace(device=torch.device("meta"))
    dmd._zero_encode_cache = {}
    data = {
        "rgb_gt": torch.ones(1, 5, 3, 1, 1),
        "rgb_rendered": torch.ones(1, 5, 3, 1, 1),
        "opacity": torch.ones(1, 5, 1, 1),
        "rgb_neighbors": torch.ones(1, 2, 3, 1, 1),
        "encoded_prompt": torch.ones(1, 3, 4),
        "neighbor_w2cs": torch.ones(1, 2, 4, 4),
        "neighbor_Ks": torch.ones(1, 2, 3, 3),
        "camera_rays": torch.ones(1, 5, 1, 1, 6),
        "w2cs": torch.ones(1, 5, 4, 4),
        "Ks": torch.ones(1, 5, 3, 3),
    }
    encodes = []

    def encode_video(frames, halo, *, gather=True):
        assert gather
        assert frames is data["rgb_rendered"]
        assert frames.device.type == "cpu"
        encodes.append(("video", halo))
        return torch.empty(1, 3, 5, 1, 1, device="meta")

    def encode_neighbors(frames):
        assert frames is data["rgb_neighbors"]
        assert frames.device.type == "cpu"
        encodes.append(("neighbors", None))
        return torch.empty(1, 3, 2, 1, 1, device="meta")

    dmd.pipe = types.SimpleNamespace(
        transformer=types.SimpleNamespace(cp_mesh=None, cp_rank=0, cp_world_size=1),
        vae=types.SimpleNamespace(dtype=torch.bfloat16),
        encode_video_frames_chunked_for_cp=encode_video,
        encode_neighbors=encode_neighbors,
    )

    def rand(size, *, device):
        assert size == 3
        assert device == dmd.accelerator.device
        return torch.tensor([0.9, 0.9, 0.9])

    def randint(low, high, size, *, device):
        assert (low, high, size) == (0, 5, (1,))
        assert device == dmd.accelerator.device
        return torch.tensor([0])

    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", randint)

    processed = dmd.process_inputs(data)

    assert encodes == [("video", 2), ("neighbors", None)]
    assert all(value.device.type == "cpu" for value in data.values())
    assert not {"rgb_gt", "rgb_rendered", "rgb_neighbors"} & processed.keys()
    for key in ("condition", "neighbors_condition", "encoded_prompt", "neighbor_w2cs", "neighbor_Ks"):
        assert processed[key].device == dmd.accelerator.device
    for key in ("opacity", "camera_rays", "w2cs", "Ks"):
        assert processed[key].device.type == "cpu"
    assert processed["opacity"].dtype == processed["camera_rays"].dtype == dmd.pipe.vae.dtype


@pytest.fixture(params=["Wan-AI/Wan2.2-TI2V-5B-Diffusers", "nvidia/Cosmos3-Nano"], ids=["wan", "cosmos3"])
def dmd_loader_batches(tmp_path, request):
    return _loader_dataset_batches(tmp_path, request.param, halo_latent_frames=4)


def _loader_dataset_batches(tmp_path, model_id, halo_latent_frames, *, num_frames=29):
    prompt_filename = f"frames_{num_frames}_stride_1.h5"
    _write_scene(
        tmp_path,
        "scene",
        x_positions=[0.02 * i for i in range(num_frames)],
        prompt_windows={"window": list(range(num_frames))},
        prompt_filename=prompt_filename,
        image_size=32,
    )
    if is_cosmos3_model(model_id):
        _write_cosmos3_prompt_file(tmp_path / "prompts/960P/scene" / prompt_filename)
    args = SimpleNamespace(
        model_id=model_id,
        max_iterations=1,
        gradient_accumulation_steps=1,
        split_path=_write_split(tmp_path, {"scene": [_entry(tmp_path, "scene", "first", 1.0)]}),
        dl3dv_dir=tmp_path / "dl3dv",
        prompt_dir=tmp_path / "prompts",
        num_frames=num_frames,
        dataset_scaling_factor=0.01,
        context_parallel_size=2,
        vae_chunk_halo_latent_frames=halo_latent_frames,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
        seed=123,
        camera_pose_alignment=default_camera_pose_alignment(model_id),
        dropout_rate=0.5,
        num_inference_steps=2,
        text_guidance_scale=2.0,
    )
    accelerator = SimpleNamespace(num_processes=2, device=torch.device("cpu"), is_main_process=False)
    loader = get_train_dataloader(args, accelerator, frames_per_block=2)
    batches = list(loader)
    dataset = loader.dataset
    split, neighbors = dataset.iter_plan[0].tolist()
    full = torch.utils.data.default_collate(
        [dataset.get_item_inner(dataset.data_perm[0].item(), split, neighbors, generator=dataset._fetch_generator(0))]
    )
    local_latent_frames = ((num_frames - 1) // 4 + 1) // 2
    first_raw_frames = 4 * local_latent_frames - 3
    follower_raw_start = 4 * max(0, local_latent_frames - halo_latent_frames - 1)
    assert [batch["rgb_rendered"].shape[1] for batch in batches] == [first_raw_frames, num_frames - follower_raw_start]
    assert [batch["opacity"].shape[1] for batch in batches] == [first_raw_frames, 4 * local_latent_frames]
    assert [batch["camera_rays"].shape[1] for batch in batches] == [local_latent_frames] * 2
    for batch in batches:
        assert batch["_cp_data_sharded"].item()
        assert batch["_cp_global_raw_frames"].item() == num_frames
        for key in ("rgb_neighbors", "neighbor_w2cs", "neighbor_Ks", "encoded_prompt"):
            torch.testing.assert_close(batch[key], full[key])
    return args, batches, full


def _loader_conditioning_collectives(monkeypatch, expected, cp_rank, cp_mesh):
    offsets = dict.fromkeys(("condition", "opacity", "camera_rays", "w2cs", "Ks"), 0)
    padded_opacity = torch.cat([expected["opacity"][:, :1].expand(-1, 3, -1, -1), expected["opacity"]], dim=1)

    def all_gather(outputs, tensor, group):
        assert group is cp_mesh.get_group()
        if tensor.ndim == 5:
            key, dim = ("camera_rays", 1) if tensor.shape[-1] == 6 else ("condition", 2)
        else:
            key, dim = {(32, 32): "opacity", (4, 4): "w2cs", (3, 3): "Ks"}[tuple(tensor.shape[-2:])], 1
        source = padded_opacity if key == "opacity" else expected[key]
        shards = source.chunk(cp_mesh.size(), dim=dim)
        if key == "condition":
            assert tensor.shape == shards[cp_rank].shape, "DMD must encode each dataset VAE shard exactly once"
        count = tensor.shape[dim]
        if key in ("opacity", "camera_rays"):
            assert count <= (8 if key == "opacity" else 2), "Stage only a rollout chunk of pixel conditioning on-device"
        elif key != "condition":
            assert count == shards[cp_rank].shape[dim], "Gather each camera matrix once per rank"
        parts = [shard.narrow(dim, offsets[key], count) for shard in shards]
        torch.testing.assert_close(tensor, parts[cp_rank])
        for output, part in zip(outputs, parts):
            output.copy_(part)
        offsets[key] += count

    monkeypatch.setattr(dist, "all_gather", all_gather)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: cp_rank)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group: tensor.mul_(cp_mesh.size()))
    return offsets


def _loader_boundary_trainer(args, cp_rank, cp_size, expected, *, device="cpu", dtype=torch.float32):
    device = torch.device(device)
    trainer = object.__new__(DMD)
    trainer.args = SimpleNamespace(**vars(args))
    trainer.args.context_parallel_size = cp_size
    trainer.accelerator = SimpleNamespace(device=device)
    trainer._zero_encode_cache = {}
    prediction = expected["condition"].to(device=device, dtype=dtype).clone().requires_grad_(True)
    pipe = _CapturingPipe(prediction)
    group = object()
    cp_mesh = SimpleNamespace(size=lambda: cp_size, get_group=lambda: group, mesh=torch.arange(4, 4 + cp_size))
    pipe.transformer = SimpleNamespace(cp_rank=cp_rank, cp_world_size=cp_size, cp_mesh=cp_mesh if cp_size > 1 else None)
    pipe.vae = SimpleNamespace(dtype=dtype, device=device, config=SimpleNamespace(scale_factor_temporal=4))
    pipe.ar_frames_per_chunk = 2
    pipe.video_inputs = []

    def encode_video(frames):
        assert frames.device.type == "cpu"
        pipe.video_inputs.append(frames.clone())
        return (frames[:, ::4].permute(0, 2, 1, 3, 4).contiguous() + 7).to(device=device, dtype=dtype)

    pipe.encode_video_frames = encode_video
    pipe.encode_neighbors = lambda frames: (frames.permute(0, 2, 1, 3, 4) + 7).to(device=device, dtype=dtype)
    pipe._validate_spatial_alignment = lambda *args, **kwargs: None
    pipe.encode_video_frames_chunked_for_cp = types.MethodType(
        ArtifixerPipelineBase.encode_video_frames_chunked_for_cp, pipe
    )
    pipe.prepare_latents = lambda condition, *args, **kwargs: torch.zeros_like(condition)
    pipe.prepare_transformer_timestep = lambda timestep, latent: timestep
    pipe.scheduler = _GuidanceScheduler()
    pipe.scheduler.training_target = lambda prediction, source: source - prediction
    trainer.pipe = pipe
    trainer.is_student_phase = lambda: True
    trainer.get_timestep = lambda batch_size: torch.full((batch_size,), 500.0)
    trainer.negative_prompt = (
        torch.zeros_like(expected["encoded_prompt"]) if not is_cosmos3_model(args.model_id) else None
    )

    class Score(_RecordingScore):
        def __init__(self, value):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(value, device=device))

        def __call__(self, **kwargs):
            super().__call__(**kwargs)
            return (torch.ones_like(kwargs["hidden_states"]) * self.value,)

    trainer.fake_score, trainer.real_score = Score(2.0), Score(1.0)
    return trainer, cp_mesh


@pytest.mark.requires_host_ffmpeg
@pytest.mark.parametrize("input_mode", ["native_cp2", "full_cp1"])
@pytest.mark.parametrize(
    "drop_rendered,frames_to_drop,drop_text,ignore_neighbors",
    [(False, 0, False, False), (False, 8, False, False), (False, 20, True, True), (True, 0, True, True)],
    ids=["keep", "suffix-last-shard", "suffix-crosses-shards", "drop-all"],
)
def test_dmd_loader_conditioning_reaches_rollout_and_local_scores(
    dmd_loader_batches, monkeypatch, input_mode, drop_rendered, frames_to_drop, drop_text, ignore_neighbors
):
    args, batches, full = dmd_loader_batches
    rendered, opacity = full["rgb_rendered"].clone(), full["opacity"].clone()
    drop_start = 0 if drop_rendered else args.num_frames - frames_to_drop
    rendered[:, drop_start:] = 0
    opacity[:, drop_start:] = 0
    expected = dict(full, condition=rendered[:, ::4].permute(0, 2, 1, 3, 4) + 7, opacity=opacity)
    cp_size = 1 if input_mode == "full_cp1" else 2
    decisions = torch.tensor([0.0 if flag else 1.0 for flag in (drop_rendered, drop_text, ignore_neighbors)])

    for cp_rank in range(cp_size):
        trainer, mesh = _loader_boundary_trainer(args, cp_rank, cp_size, expected)
        offsets = _loader_conditioning_collectives(monkeypatch, expected, cp_rank, mesh)
        data = batches[cp_rank] if input_mode == "native_cp2" else full
        originals = {key: value.clone() for key, value in data.items() if isinstance(value, torch.Tensor)}

        def rand(size, *, device):
            assert cp_rank == 0 and size == 3
            return decisions.clone()

        def randint(low, high, size, *, device):
            assert cp_rank == 0 and (low, high, size) == (0, 29, (1,))
            return torch.tensor([frames_to_drop])

        def broadcast(tensor, src, group):
            assert src == 4 and group is mesh.get_group()
            tensor.copy_(decisions if tensor.numel() == 3 else torch.tensor([frames_to_drop]))

        monkeypatch.setattr(torch, "rand", rand)
        monkeypatch.setattr(torch, "randint", randint)
        monkeypatch.setattr(dist, "broadcast", broadcast)
        loss, metrics = trainer.training_batch(data)
        loss.backward()
        assert torch.isfinite(loss) and set(metrics) == {"critic_loss", "generator_loss"}
        assert trainer.fake_score.value.grad.abs().item() > 0
        assert trainer.real_score.value.grad is None
        rollout, kwargs = trainer.pipe.calls[0]
        for index, key in ((0, "condition"), (1, "opacity"), (3, "camera_rays"), (4, "w2cs"), (6, "Ks")):
            torch.testing.assert_close(rollout[index], expected[key])
        assert kwargs["ignore_neighbors"] is ignore_neighbors
        reference = torch.zeros_like(full["rgb_neighbors"]) if ignore_neighbors else full["rgb_neighbors"]
        torch.testing.assert_close(rollout[2], reference.permute(0, 2, 1, 3, 4) + 7)
        prompt = full["encoded_prompt"]
        if drop_text:
            prompt = full["encoded_negative_prompt"] if "encoded_negative_prompt" in full else torch.zeros_like(prompt)
        torch.testing.assert_close(rollout[8], prompt)
        if "neighbor_camera_rays" in full:
            torch.testing.assert_close(kwargs["neighbor_camera_rays"], full["neighbor_camera_rays"])
            mask = full["negative_prompt_attention_mask"] if drop_text else full["prompt_attention_mask"]
            torch.testing.assert_close(kwargs["prompt_attention_mask"], mask)
        start, end = cp_rank * (8 // cp_size), (cp_rank + 1) * (8 // cp_size)
        raw_start, raw_end = (0 if start == 0 else 1 + 4 * (start - 1)), 1 + 4 * (end - 1)
        for score in (trainer.real_score, trainer.fake_score):
            for call in score.calls:
                assert call["local_cp_input"] is (cp_size > 1) and call["frame_offset"] == start
                for key in ("camera_rays", "w2cs", "Ks"):
                    torch.testing.assert_close(call[key], full[key][:, start:end])
                torch.testing.assert_close(call["opacity"], opacity[:, raw_start:raw_end])
                if "neighbor_camera_rays" in full:
                    torch.testing.assert_close(call["neighbor_camera_rays"], full["neighbor_camera_rays"])
        expected_grad = torch.zeros_like(trainer.pipe.prediction)
        expected_grad[:, :, start:end] = -1 / trainer.pipe.prediction[:, :, start:end].numel()
        torch.testing.assert_close(trainer.pipe.prediction.grad, expected_grad)
        assert len(trainer.pipe.video_inputs) == 1
        encode_end = raw_end if cp_size > 1 else args.num_frames
        torch.testing.assert_close(trainer.pipe.video_inputs[0], rendered[:, :encode_end])
        for key, original in originals.items():
            torch.testing.assert_close(data[key], original)
            assert data[key].device.type == "cpu"
        if input_mode == "native_cp2":
            assert offsets == {"condition": 4, "opacity": 16, "camera_rays": 4, "w2cs": 4, "Ks": 4}


@pytest.mark.requires_host_ffmpeg
@pytest.mark.parametrize(
    "invalid_field", ["batch_size", "latent_crop", "conditioning_start", "opacity", "camera_rays", "w2cs", "Ks"]
)
def test_dmd_native_loader_rejects_invalid_shard_before_gather(dmd_loader_batches, monkeypatch, invalid_field):
    args, batches, full = dmd_loader_batches
    expected = dict(full, condition=full["rgb_rendered"][:, ::4].permute(0, 2, 1, 3, 4) + 7)

    def unexpected_gather(*args, **kwargs):
        pytest.fail("Invalid native CP batch reached all_gather")

    monkeypatch.setattr(dist, "all_gather", unexpected_gather)
    for cp_rank, batch in enumerate(batches):
        data = {key: value.clone() for key, value in batch.items()}
        if invalid_field == "batch_size":
            data = {
                key: value if key.startswith("_cp_") else value.repeat_interleave(2, dim=0)
                for key, value in data.items()
            }
        elif invalid_field == "latent_crop":
            data["_cp_vae_crop"][:, 1] -= 1
        elif invalid_field == "conditioning_start":
            data["_cp_conditioning_raw_start"] = batches[1 - cp_rank]["_cp_conditioning_raw_start"].clone()
        else:
            data[invalid_field] = data[invalid_field][:, :-1]
        trainer, _ = _loader_boundary_trainer(args, cp_rank, 2, expected)
        trainer._sample_input_dropout = lambda *args, **kwargs: (False, 0, False, False)

        with pytest.raises((AssertionError, ValueError)):
            trainer.process_inputs(data)


@pytest.mark.requires_host_ffmpeg
def test_dmd_native_loader_gathers_ragged_opacity_in_rank_order(tmp_path, monkeypatch):
    args, batches, full = _loader_dataset_batches(
        tmp_path, "Wan-AI/Wan2.2-TI2V-5B-Diffusers", halo_latent_frames=4, num_frames=21
    )
    full["opacity"] = torch.arange(args.num_frames, dtype=torch.float32).view(1, -1, 1, 1).expand_as(
        full["opacity"]
    ) / (args.num_frames - 1)
    expected = dict(full, condition=full["rgb_rendered"][:, ::4].permute(0, 2, 1, 3, 4) + 7)
    for cp_rank, batch in enumerate(batches):
        raw_start = batch["_cp_conditioning_raw_start"].item()
        data = dict(batch, opacity=full["opacity"][:, raw_start : raw_start + batch["opacity"].shape[1]])
        trainer, mesh = _loader_boundary_trainer(args, cp_rank, 2, expected)
        trainer._sample_input_dropout = lambda *args, **kwargs: (False, 0, False, False)
        offsets = _loader_conditioning_collectives(monkeypatch, expected, cp_rank, mesh)

        processed = trainer.process_inputs(data)

        for key in ("condition", "opacity", "camera_rays", "w2cs", "Ks"):
            torch.testing.assert_close(processed[key], expected[key])
            assert processed[key].device.type == "cpu"
        assert offsets == {"condition": 3, "opacity": 12, "camera_rays": 3, "w2cs": 3, "Ks": 3}


def _run_dmd_input_routing_worker(rank, world_size, init_file, args, batches, full, backend):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    dtype = torch.bfloat16 if backend == "nccl" else torch.float32
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend, init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        mesh = DeviceMesh(device.type, torch.arange(world_size))
        local = batches[rank]
        originals = [{key: value.clone() for key, value in data.items()} for data in (full, local)]
        expected = dict(full, condition=full["rgb_rendered"][:, ::4].permute(0, 2, 1, 3, 4) + 7)
        for drop_rendered, frames_to_drop, drop_text, ignore_neighbors in (
            (False, 0, False, False),
            (False, 8, False, False),
            (False, 20, True, True),
            (True, 0, True, True),
        ):
            trainer, _ = _loader_boundary_trainer(args, rank, world_size, expected, device=device, dtype=dtype)
            trainer.pipe.transformer.cp_mesh = mesh
            reference, _ = _loader_boundary_trainer(args, 0, 1, expected, device=device, dtype=dtype)
            flags = [0.0 if flag else 1.0 for flag in (drop_rendered, drop_text, ignore_neighbors)]

            def rand(size, *, device):
                assert size == 3
                return torch.tensor(flags, device=device)

            def randint(low, high, size, *, device):
                assert (low, high, size) == (0, 29, (1,))
                return torch.tensor([frames_to_drop], device=device)

            with patch.object(torch, "rand", rand), patch.object(torch, "randint", randint):
                full_processed = reference.process_inputs(full)
                for _ in range(2):
                    processed = trainer.process_inputs(local)
                    assert processed.keys() == full_processed.keys()
                    assert not {"rgb_gt", "rgb_rendered", "rgb_neighbors", "_cp_vae_crop"} & processed.keys()
                    for key, value in processed.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(value, full_processed[key], rtol=0, atol=0)
                            cpu_staged = key in ("opacity", "camera_rays", "w2cs", "Ks")
                            assert value.device == (torch.device("cpu") if cpu_staged else device)
                        else:
                            assert value == full_processed[key]
            assert len(trainer.pipe.video_inputs) == (1 if drop_rendered else 2)
            dropped = full["rgb_rendered"].clone()
            dropped[:, 0 if drop_rendered else 29 - frames_to_drop :] = 0
            raw_start = int(local["_cp_vae_raw_start"].item())
            raw_end = raw_start + local["rgb_rendered"].shape[1]
            torch.testing.assert_close(trainer.pipe.video_inputs[-1], dropped[:, raw_start:raw_end])
            for data, original in zip((full, local), originals):
                for key, value in data.items():
                    assert value.device.type == "cpu"
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.requires_host_ffmpeg
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("model_id", ["Wan-AI/Wan2.2-TI2V-5B-Diffusers", "nvidia/Cosmos3-Nano"], ids=["wan", "cosmos3"])
def test_dmd_native_loader_distributed_input_routing_matches_cp1(tmp_path, model_id, backend):
    args, batches, full = _loader_dataset_batches(tmp_path, model_id, halo_latent_frames=1)
    _run_cp_workers(_run_dmd_input_routing_worker, tmp_path, args, batches, full, backend=backend)


def test_training_batch_reuses_one_prediction_for_student_phase():
    processed = _processed_batch()
    prediction = torch.ones(1, requires_grad=True)
    dmd, state = _make_dmd(processed, is_student=True, prediction=prediction)

    loss, metrics = dmd.training_batch({"sample": "raw"})

    assert torch.equal(loss.detach(), torch.tensor(5.0))
    assert torch.equal(metrics["critic_loss"], torch.tensor(2.0))
    assert torch.equal(metrics["generator_loss"], torch.tensor(3.0))
    assert not metrics["critic_loss"].requires_grad
    assert not metrics["generator_loss"].requires_grad

    assert state["raw_data"] == {"sample": "raw"}
    assert state["critic_data"] is processed
    assert state["generator_data"] is processed
    assert state["critic_prediction"].requires_grad is False
    assert state["generator_prediction"] is prediction

    assert len(dmd.pipe.calls) == 1
    args, kwargs = dmd.pipe.calls[0]
    assert args == (
        processed["condition"],
        processed["opacity"],
        processed["neighbors_condition"],
        processed["camera_rays"],
        processed["w2cs"],
        processed["neighbor_w2cs"],
        processed["Ks"],
        processed["neighbor_Ks"],
        processed["encoded_prompt"],
        4,
        True,
    )
    assert kwargs == {"ignore_neighbors": True}
    assert dmd.pipe.grad_enabled == [True]


def test_training_batch_runs_critic_only_rollout_without_gradients():
    processed = _processed_batch()
    prediction = torch.ones(1, requires_grad=True)
    dmd, state = _make_dmd(processed, is_student=False, prediction=prediction)

    def generator_loss(self, data, generator_prediction):  # pragma: no cover - assertion path
        raise AssertionError("generator_loss should not run during critic-only phases")

    dmd.generator_loss = types.MethodType(generator_loss, dmd)

    loss, metrics = dmd.training_batch({})

    assert torch.equal(loss.detach(), torch.tensor(2.0))
    assert list(metrics) == ["critic_loss"]
    assert state["critic_prediction"].requires_grad is False
    assert dmd.pipe.grad_enabled == [False]


def test_student_phase_is_constant_across_gradient_accumulation_window():
    dmd = object.__new__(DMD)
    dmd.args = types.SimpleNamespace(gradient_accumulation_steps=3, dfake_gen_update_ratio=5)
    dmd.step_offset = 0
    dmd.accelerator = types.SimpleNamespace(step=0)

    phase_by_optimizer_step = []
    for optimizer_step in range(1, 7):
        microbatch_phases = []
        for microbatch_index in range(1, dmd.args.gradient_accumulation_steps + 1):
            dmd.accelerator.step = (optimizer_step - 1) * dmd.args.gradient_accumulation_steps + microbatch_index
            microbatch_phases.append(dmd.is_student_phase())
        assert len(set(microbatch_phases)) == 1
        phase_by_optimizer_step.append(microbatch_phases[0])

    assert phase_by_optimizer_step == [True, False, False, False, False, True]


def test_student_phase_respects_checkpoint_step_offset():
    dmd = object.__new__(DMD)
    dmd.args = types.SimpleNamespace(gradient_accumulation_steps=2, dfake_gen_update_ratio=5)
    dmd.step_offset = 10
    dmd.accelerator = types.SimpleNamespace(step=1)

    assert dmd.is_student_phase()


# --- generator_loss CFG text-guidance path (real generator_loss, mocked score models) ---


class _RecordingScore:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (torch.zeros_like(kwargs["hidden_states"]),)


class _GuidanceScheduler:
    num_train_timesteps = 1000
    shift = 1

    def add_noise(self, prediction, source_latent, timestep):
        return prediction + source_latent

    def step(self, noise_pred, timestep, noisy_latent, to_final=True):
        return noisy_latent - noise_pred


class _GuidancePipe:
    def __init__(self):
        self.scheduler = _GuidanceScheduler()
        self.vae = types.SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
        # generator_loss now consults the transformer's CP metadata; single-GPU stub.
        self.transformer = types.SimpleNamespace(cp_mesh=None, cp_rank=0, cp_world_size=1)

    def prepare_latents(self, condition, opacity, use_flow_matching_source, broadcast_cp_noise=True):
        return torch.zeros_like(condition)

    def prepare_transformer_timestep(self, timestep, noisy_latent):
        return timestep


def _guidance_batch(batch_size: int = 2) -> dict:
    latent_shape = (batch_size, 1, 2, 2, 2)
    return {
        "condition": torch.zeros(latent_shape),
        "opacity": torch.zeros(latent_shape),
        "neighbors_condition": torch.zeros(1),
        "camera_rays": torch.zeros(1),
        "w2cs": torch.zeros(1),
        "neighbor_w2cs": torch.zeros(1),
        "Ks": torch.zeros(1),
        "neighbor_Ks": torch.zeros(1),
        "ignore_neighbors": True,
        "encoded_prompt": torch.randn(batch_size, 3, 4),
    }


def _make_guidance_dmd(*, negative_prompt) -> DMD:
    dmd = object.__new__(DMD)
    dmd.args = types.SimpleNamespace(text_guidance_scale=3.0)
    dmd.pipe = _GuidancePipe()
    # _score_batch moves the batch via self.accelerator.device on the no-CP path.
    dmd.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    dmd.fake_score = _RecordingScore()
    dmd.real_score = _RecordingScore()
    dmd.negative_prompt = negative_prompt
    # Same 2%/98% clamp bounds the constructor derives from the scheduler.
    dmd.min_step = int(0.02 * dmd.pipe.scheduler.num_train_timesteps)
    dmd.max_step = int(0.98 * dmd.pipe.scheduler.num_train_timesteps)
    return dmd


def test_generator_loss_feeds_the_batch_negative_pair_to_both_uncond_score_forwards():
    """Cosmos3 (token-prompt) DMD with CFG active: the uncond forwards of BOTH
    score models must receive the batch's stored EMPTY-caption unconditional
    AND its attention mask — silently dropping the mask would attend all pad
    tokens of the 1024-wide payload."""
    dmd = _make_guidance_dmd(negative_prompt=None)
    data = _guidance_batch()
    data["prompt_attention_mask"] = torch.ones(2, 3, dtype=torch.long)
    data["encoded_negative_prompt"] = torch.randn(2, 3, 4)
    data["negative_prompt_attention_mask"] = torch.zeros(2, 3, dtype=torch.long)
    prediction = torch.randn(2, 1, 2, 2, 2)

    loss = DMD.generator_loss(dmd, data, prediction)

    assert torch.isfinite(loss)
    for score in (dmd.fake_score, dmd.real_score):
        cond_call, uncond_call = score.calls
        assert cond_call["encoder_hidden_states"] is data["encoded_prompt"]
        assert cond_call["encoder_attention_mask"] is data["prompt_attention_mask"]
        assert uncond_call["encoder_hidden_states"] is data["encoded_negative_prompt"]
        assert uncond_call["encoder_attention_mask"] is data["negative_prompt_attention_mask"]


def test_generator_loss_expands_wans_stored_negative_prompt_to_batch_size():
    """Wan DMD carries no per-batch negative: the loaded default negative T5
    embedding is expanded to the batch and passed without an attention mask."""
    negative_prompt = torch.randn(1, 3, 4)
    dmd = _make_guidance_dmd(negative_prompt=negative_prompt)
    data = _guidance_batch()
    prediction = torch.randn(2, 1, 2, 2, 2)

    loss = DMD.generator_loss(dmd, data, prediction)

    assert torch.isfinite(loss)
    for score in (dmd.fake_score, dmd.real_score):
        cond_call, uncond_call = score.calls
        assert cond_call["encoder_hidden_states"] is data["encoded_prompt"]
        assert "encoder_attention_mask" not in cond_call
        assert uncond_call["encoder_hidden_states"].shape == (2, 3, 4)
        assert torch.equal(uncond_call["encoder_hidden_states"], negative_prompt.expand(2, -1, -1))
        assert "encoder_attention_mask" not in uncond_call


def test_generator_loss_refuses_token_prompt_batches_without_a_stored_negative():
    """A token-prompt model (negative_prompt=None) whose batch carries no
    encoded_negative_prompt has no unconditional to guide with — fail loudly
    instead of guiding against garbage."""
    dmd = _make_guidance_dmd(negative_prompt=None)
    data = _guidance_batch()
    prediction = torch.randn(2, 1, 2, 2, 2)

    with pytest.raises(AssertionError, match="negative prompt"):
        DMD.generator_loss(dmd, data, prediction)


# --- generator_loss normalization (real score-batch slicing and CP collectives) ---

_DISTRIBUTED_COMMAND = (
    "python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest "
    "tests/test_dmd_training_batch.py -q -k normalization_matches_full_video"
)

# Different means in each temporal half expose rank-local normalization.
_NORMALIZATION_CASES = [
    pytest.param(torch.float32, [1, 1, 9, 9], 1, id="fp32"),
    pytest.param(torch.float64, [1.1, 3.7, 9.3, 13.9], 0.3, id="fp64"),
    # Shard means fall between adjacent bf16 values; a bf16 reduction would round them.
    pytest.param(torch.bfloat16, [1, 1.0078125, 2, 2.015625], 0.125, id="bf16-rounding"),
    # The sum of shard means exceeds the fp16 maximum at both CP=2 and CP=4.
    pytest.param(torch.float16, [32768, 32768, 49152, 49152], 512, id="fp16-overflow"),
]


class _TemporalScore:
    """Read a fixed noise prediction at this rank's frame window."""

    def __init__(self, noise):
        self.noise = noise

    def __call__(self, hidden_states, frame_offset, **kwargs):
        return (self.noise[:, :, frame_offset : frame_offset + hidden_states.shape[2]],)


def _normalization_inputs(dtype, frame_errors, group_index=0):
    errors = torch.tensor(frame_errors, dtype=dtype).view(1, 1, 4, 1, 1)
    # Distinct sample and group statistics expose accidental batch or world-wide normalization.
    errors = torch.cat((errors, errors / 2)).expand(2, 2, 4, 2, 2).clone() / 2**group_index
    data = _guidance_batch()
    # _score_batch slices these along the frame axis; the score stubs ignore their values.
    frames = torch.zeros(2, 4)
    data.update(condition=torch.zeros_like(errors), opacity=frames, camera_rays=frames, w2cs=frames, Ks=frames)
    return data, errors


def _normalization_dmd(errors, gap, cp_mesh=None):
    dmd = _make_guidance_dmd(negative_prompt=torch.zeros(1, 3, 4))
    dmd.pipe.vae.config = types.SimpleNamespace(scale_factor_temporal=1)
    if cp_mesh is not None:
        dmd.pipe.transformer = types.SimpleNamespace(
            cp_mesh=cp_mesh, cp_rank=cp_mesh.get_local_rank(), cp_world_size=cp_mesh.size()
        )
    # Both prompt branches return the same score, so CFG leaves the expected x0 unchanged.
    # With prediction == 0, the scheduler gives real_x0 == -errors and fake_x0 == -(errors + gap).
    dmd.real_score = _TemporalScore(errors)
    dmd.fake_score = _TemporalScore(errors + gap)
    return dmd


def _loss_and_prediction_grad(dmd, data):
    prediction = torch.zeros_like(data["condition"], requires_grad=True)
    loss = dmd.generator_loss(data, prediction)
    loss.backward()
    return loss.detach(), prediction.grad.double()


def _assert_single_rank_formula(errors, gap, loss, prediction_grad):
    real_x0, fake_x0 = -errors, -(errors + gap)
    direction = torch.nan_to_num((fake_x0 - real_x0) / real_x0.abs().mean(dim=[1, 2, 3, 4], keepdim=True))
    # The loss is evaluated in double and autograd casts the gradient back to the prediction dtype.
    expected_grad = (direction.double() / errors.numel()).to(errors.dtype).double()
    torch.testing.assert_close(prediction_grad, expected_grad, rtol=1e-12, atol=0)
    torch.testing.assert_close(loss, 0.5 * direction.double().square().mean(), rtol=1e-12, atol=0)


@pytest.mark.parametrize("dtype,frame_errors,gap", _NORMALIZATION_CASES)
def test_generator_loss_normalization_preserves_single_rank_formula(dtype, frame_errors, gap):
    data, errors = _normalization_inputs(dtype, frame_errors)
    _assert_single_rank_formula(errors, gap, *_loss_and_prediction_grad(_normalization_dmd(errors, gap), data))


@pytest.fixture(scope="module")
def dmd_distributed_env():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip(f"needs four processes: {_DISTRIBUTED_COMMAND}")
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("torch.distributed process groups are incompatible with xdist workers")
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=60))
    try:
        yield
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=[1, 2, 4], ids=["cp1", "cp2", "cp4"])
def dmd_cp_mesh(request, dmd_distributed_env):
    """Use consecutive ranks per CP group, matching training."""
    mesh = DeviceMesh("cpu", torch.arange(4).view(-1, request.param), mesh_dim_names=("dp", "cp"))
    return mesh["cp"], mesh.get_coordinate()[0]


@pytest.mark.parametrize("dtype,frame_errors,gap", _NORMALIZATION_CASES)
def test_generator_loss_normalization_matches_full_video(dmd_cp_mesh, dtype, frame_errors, gap):
    cp_mesh, group_index = dmd_cp_mesh
    data, errors = _normalization_inputs(dtype, frame_errors, group_index)
    reference_loss, reference_grad = _loss_and_prediction_grad(_normalization_dmd(errors, gap), data)
    loss, prediction_grad = _loss_and_prediction_grad(_normalization_dmd(errors, gap, cp_mesh), data)

    # Average shard losses and prediction gradients before comparing with the full video.
    dist.all_reduce(loss, group=cp_mesh.get_group())
    dist.all_reduce(prediction_grad, group=cp_mesh.get_group())

    # Check the unsplit reference independently, even with a process group initialized.
    _assert_single_rank_formula(errors, gap, reference_loss, reference_grad)
    torch.testing.assert_close(loss / cp_mesh.size(), reference_loss, rtol=1e-12, atol=0)
    torch.testing.assert_close(prediction_grad / cp_mesh.size(), reference_grad, rtol=1e-12, atol=0)


@pytest.mark.parametrize("accumulation", [1, 3])
def test_dmd_ema_preserves_generator_cadence_and_warmup_copy(tmp_path, accumulation):
    trainer = _small_dmd_ema_trainer(tmp_path)
    trainer.args.gradient_accumulation_steps = accumulation
    assert trainer.ema.active
    _advance_dmd_ema(trainer, 200)
    assert trainer.ema.optimizer_updates == 40
    assert trainer.ema.initialized_update == 40
    assert trainer.ema.ema_updates == 0
    for param in trainer.ema.parameters():
        torch.testing.assert_close(param, torch.full_like(param, 196.0))
    _advance_dmd_ema(trainer, 206)
    assert trainer.ema.optimizer_updates == 42
    assert trainer.ema.ema_updates == 2
    for param in trainer.ema.parameters():
        torch.testing.assert_close(param, torch.full_like(param, 0.99 * (0.99 * 196 + 0.01 * 201) + 0.01 * 206))
    assert {id(param) for param in trainer.ema.parameters()}.isdisjoint(
        id(param) for group in trainer.optimizers[0].param_groups for param in group["params"]
    )


def test_dmd_ema_disabled_has_no_shadow_parameters(tmp_path):
    trainer = _small_dmd_ema_trainer(tmp_path, extra_args=("--ema_weight", "0"))
    _advance_dmd_ema(trainer, 206)
    assert not trainer.ema.active
    assert list(trainer.ema.parameters()) == []
    for hook in trainer.accelerator.save_hooks:
        hook([], [], str(tmp_path))
    progress = json.loads((tmp_path / "ema.json").read_text())
    assert progress["ema_weight"] == 0
    assert progress["ema_initialized_update"] is None
    assert not (tmp_path / "pytorch_model_fsdp_ema").exists()


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_dmd_ema_validation_restores_weights_and_mode(tmp_path, cp, training, fail, active):
    trainer = _small_dmd_ema_trainer(tmp_path)
    _advance_dmd_ema(trainer, 206 if active else 196)
    model = trainer.pipe.transformer
    model.train(training)
    raw = [param.detach().clone() for param in model.parameters()]
    expected = list(trainer.ema.parameters()) if active else raw
    trainer.pipe.stages_conditioning_per_chunk = True
    trainer.pipe.transformer._cp_mesh = SimpleNamespace(size=lambda: 2) if cp else None
    calls = []

    def validate(*args):
        calls.append(args)
        assert not model.training
        for actual, value in zip(model.parameters(), expected, strict=True):
            torch.testing.assert_close(actual, value)
        if fail:
            raise RuntimeError("DMD validation failed")

    trainer.evaluation._generate_standard = lambda *args: validate("single", *args)
    trainer.evaluation._generate_kv_cp = lambda *args: validate("cp", *args)
    updates = trainer.ema.optimizer_updates
    if fail:
        with pytest.raises(RuntimeError, match="DMD validation failed"):
            with trainer.validation_context():
                trainer.evaluation.run_pass({}, [(ValidationTaskSpec("sample", 3), 0)], tmp_path)
    else:
        with trainer.validation_context():
            trainer.evaluation.run_pass({}, [(ValidationTaskSpec("sample", 3), 0)], tmp_path)
    assert len(calls) == 1
    assert calls[0][0] == ("cp" if cp else "single")
    assert trainer.ema.optimizer_updates == updates
    assert model.training == training
    assert not trainer.ema.is_cached
    for actual, value in zip(model.parameters(), raw, strict=True):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)


@pytest.mark.parametrize("stage,cutoff", [("dmd", 196), ("dmd", 200), ("dmd", 206)])
def test_shared_ema_checkpoint_schema_and_resume(tmp_path, stage, cutoff):
    import torch.distributed.checkpoint as dcp

    from model_training.utils.ema import ModelEMA
    from model_training.utils.train_utils import load_model_weights_from_dcp

    source = _ema_stage_trainer(stage, tmp_path / "source")
    _advance_ema_stage(stage, source, cutoff)
    checkpoint = tmp_path / f"checkpoint_{cutoff}"
    dcp.save({"model": source.pipe.transformer.state_dict()}, checkpoint_id=checkpoint / "pytorch_model_fsdp_0")
    for hook in source.accelerator.save_hooks:
        hook([], [], str(checkpoint))
    assert isinstance(source.ema, ModelEMA)
    progress = json.loads((checkpoint / "ema.json").read_text())
    assert set(progress) == {
        "version",
        "training_stage",
        "checkpoint_step",
        "optimizer_updates",
        "ema_weight",
        "ema_start_step",
        "initialize_after_updates",
        "ema_initialized_update",
        "ema_updates",
    }
    assert progress["training_stage"] == stage
    assert progress["checkpoint_step"] == cutoff
    assert progress["ema_updates"] == source.ema.ema_updates
    assert not (checkpoint / "long_ar_ema.json").exists()
    assert not (checkpoint / "pytorch_model_fsdp_2").exists()
    restored = _ema_stage_trainer(stage, tmp_path / "restored")
    for hook in restored.accelerator.load_hooks:
        hook([], str(checkpoint))
    load_model_weights_from_dcp(restored.pipe.transformer, checkpoint / "pytorch_model_fsdp_0")
    restored.accelerator.step = cutoff
    restored.ema.restore_after_model_load()
    _advance_ema_stage(stage, source, cutoff + 7)
    _advance_ema_stage(stage, restored, cutoff + 7)
    assert restored.ema.optimizer_updates == source.ema.optimizer_updates
    assert restored.ema.initialized_update == source.ema.initialized_update
    for actual, expected in zip(restored.ema.parameters(), source.ema.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["dmd"])
def test_shared_ema_requires_new_progress_in_training_resume(tmp_path, stage):
    trainer = _ema_stage_trainer(stage, tmp_path)
    (tmp_path / "long_ar_ema.json").write_text(json.dumps({"version": 1, "optimizer_updates": 150}))
    for hook in trainer.accelerator.load_hooks:
        hook([], str(tmp_path))
    trainer.accelerator.step = 150
    with pytest.raises(ValueError, match="ema.json"):
        trainer.ema.restore_after_model_load()


@pytest.mark.parametrize("stage", ["dmd"])
@pytest.mark.parametrize("slot", ["pytorch_model_fsdp_0", "pytorch_model_fsdp_ema"])
def test_dmd_ema_export_labels_raw_and_ema_slots(tmp_path, monkeypatch, stage, slot):
    import torch.distributed.checkpoint as dcp

    import model_eval.export_checkpoint_pt as export
    from tests._helpers import _EMATinyModel

    source = _ema_stage_trainer(stage, tmp_path)
    _advance_ema_stage(stage, source, 206)
    dcp.save({"model": source.pipe.transformer.state_dict()}, checkpoint_id=tmp_path / "pytorch_model_fsdp_0")
    for hook in source.accelerator.save_hooks:
        hook([], [], str(tmp_path))
    args = export.parse_args(
        [
            "--checkpoint_dir",
            str(tmp_path / slot),
            "--output_pt",
            str(tmp_path / "export.pt"),
            "--run_id",
            "tiny",
            "--checkpoint",
            str(source.get_step()),
            "--slot",
            slot,
        ]
    )
    monkeypatch.setattr(export, "create_export_transformer", lambda args: _EMATinyModel(2, 2))
    output, metadata_path = export.export_checkpoint(args)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["training_stage"] == stage
    assert metadata["weight_type"] == ("raw" if slot.endswith("_0") else "ema")
    expected_model = source.pipe.transformer if slot.endswith("_0") else source.ema
    state = torch.load(output, weights_only=True)
    for name, expected in zip(source.pipe.transformer.state_dict(), expected_model.parameters(), strict=True):
        torch.testing.assert_close(state[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("cutoff", [196, 200, 206])
def test_dmd_ema_accelerate_save_resume_equivalence(tmp_path, cutoff, isolated_accelerator_state):
    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin

    from tests._helpers import _EMATinyModel

    def make():
        accelerator = Accelerator(
            cpu=True, gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=3, sync_with_dataloader=False)
        )
        model, critic = _EMATinyModel(2, 2), _EMATinyModel(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=0.01)
        model, optimizer, critic, critic_optimizer = accelerator.prepare(model, optimizer, critic, critic_optimizer)
        trainer = _small_dmd_ema_trainer(tmp_path, accelerator=accelerator, model=model, optimizer=optimizer)
        trainer.fake_score = critic
        trainer.optimizers.append(critic_optimizer)
        trainer.args.gradient_accumulation_steps = 3
        assert len(accelerator._models) == 2
        return trainer

    def advance(trainer, count):
        for _ in range(count * 3):
            with trainer.accelerator.accumulate(*trainer.to_accumulate()):
                data = torch.randn(1, 2)
                loss = trainer.fake_score(data).square().mean()
                if trainer.is_student_phase():
                    loss = loss + trainer.pipe.transformer(data).square().mean()
                trainer.accelerator.backward(loss)
                for optimizer in trainer.optimizers:
                    optimizer.step()
                trainer.on_before_zero_grad()
                for optimizer in trainer.optimizers:
                    optimizer.zero_grad()

    torch.manual_seed(1234)
    uninterrupted = make()
    advance(uninterrupted, cutoff)
    checkpoint = tmp_path / f"checkpoint_{cutoff}"
    uninterrupted.accelerator.save_state(str(checkpoint))
    assert (checkpoint / "ema.json").is_file()
    assert (checkpoint / "pytorch_model_fsdp_ema" / ".metadata").is_file()
    advance(uninterrupted, 211 - cutoff)
    restored = make()
    restored.accelerator.load_state(str(checkpoint))
    restored.step_offset = cutoff - restored.accelerator.step // 3
    restored.ema.restore_after_model_load()
    advance(restored, 211 - cutoff)
    assert restored.ema.optimizer_updates == uninterrupted.ema.optimizer_updates == 43
    assert restored.ema.ema_updates == uninterrupted.ema.ema_updates == 3
    for actual_model, expected_model in (
        (restored.pipe.transformer, uninterrupted.pipe.transformer),
        (restored.fake_score, uninterrupted.fake_score),
        (restored.ema, uninterrupted.ema),
    ):
        for actual, expected in zip(actual_model.parameters(), expected_model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_optimizer, expected_optimizer in zip(restored.optimizers, uninterrupted.optimizers, strict=True):
        actual_states = actual_optimizer.state_dict()["state"]
        for key, expected_state in expected_optimizer.state_dict()["state"].items():
            for name, expected in expected_state.items():
                torch.testing.assert_close(actual_states[key][name], expected, rtol=0, atol=0)


def test_shared_ema_checkpoint_preserves_parameter_aliases_and_buffers(tmp_path):
    from model_training.utils.ema import ModelEMA
    from model_training.utils.train_utils import load_model_weights_from_dcp
    from tests._helpers import _EMATestAccelerator

    def make():
        model = torch.nn.Module()
        model.first = torch.nn.Linear(2, 2)
        model.second = model.first
        model.register_buffer("marker", torch.tensor(3.0))
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(2)
        return model

    model = make()
    accelerator = _EMATestAccelerator()
    ema = ModelEMA(
        model,
        accelerator,
        lambda: accelerator.step,
        ema_weight=0.99,
        ema_start_step=0,
        initialize_after_updates=0,
        training_stage="dmd",
        camera_pose_alignment="mean",
    )
    ema.restore_after_model_load()
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(6)
        model.marker.fill_(9)
    accelerator.step = 1
    ema.update()
    ema.save_checkpoint(tmp_path)
    restored = make()
    load_model_weights_from_dcp(restored, tmp_path / "pytorch_model_fsdp_ema")
    assert restored.first is restored.second
    assert restored.marker.item() == 9
    for actual, expected in zip(restored.parameters(), ema.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_ema_validation_restores_after_partial_copy_failure(tmp_path, monkeypatch):
    trainer = _small_dmd_ema_trainer(tmp_path)
    model = trainer.pipe.transformer
    with torch.no_grad():
        for param in model.parameters():
            param.add_(7)
    raw = [param.detach().clone() for param in model.parameters()]
    model.eval()

    @torch.no_grad()
    def fail_copy(source, target):
        next(target.parameters()).copy_(next(source.parameters()))
        raise RuntimeError("copy failed")

    monkeypatch.setattr(trainer.ema.worker, "copy_to", fail_copy)
    with pytest.raises(RuntimeError, match="copy failed"):
        with trainer.ema.apply():
            pytest.fail("validation must not start after a failed swap")
    assert not trainer.ema.is_cached
    assert not model.training
    for actual, expected in zip(model.parameters(), raw, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dmd_ema_does_not_advance_on_skipped_optimizer_update(tmp_path):
    trainer = _small_dmd_ema_trainer(tmp_path)
    _advance_dmd_ema(trainer, 201)
    previous = [param.detach().clone() for param in trainer.ema.parameters()]
    trainer.accelerator.step = 206
    trainer.accelerator.optimizer_step_was_skipped = True
    trainer.on_before_zero_grad()
    assert trainer.ema.optimizer_updates == 41
    assert trainer.ema.ema_updates == 1
    for actual, expected in zip(trainer.ema.parameters(), previous, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["dmd"])
@pytest.mark.parametrize("marker", [None, "long_ar_ema.json", "pytorch_model_fsdp_2"])
def test_ema_schema_change_recovery_preserves_the_teacher(tmp_path, marker, stage):
    trainer = _ema_stage_trainer(stage, tmp_path)
    if marker == "long_ar_ema.json":
        (tmp_path / marker).write_text("{}")
    elif marker is not None:
        (tmp_path / marker).mkdir()
    for hook in trainer.accelerator.load_hooks:
        hook([], str(tmp_path))
    with pytest.raises(ValueError) as error:
        trainer.ema.restore_after_model_load()
    message = str(error.value)
    assert "schema" in message and "ema.json" in message
    assert "warm-start" in message and "--project_dir" in message
    assert "--base_checkpoint_dir" in message and "pytorch_model_fsdp_0" in message
    assert "--base_checkpoint_dir_critic" in message and "original stage-1 teacher" in message
    assert "pytorch_model_fsdp_1" not in message
    if marker is not None:
        assert marker in message


@pytest.mark.parametrize("stage", ["dmd"])
@pytest.mark.parametrize("field", ["training_stage", "initialize_after_updates"])
def test_shared_ema_resume_rejects_config_mismatch(tmp_path, stage, field):
    source = _ema_stage_trainer(stage, tmp_path)
    _advance_ema_stage(stage, source, 206)
    source.ema.save_checkpoint(tmp_path)
    state_path = tmp_path / "ema.json"
    progress = json.loads(state_path.read_text())
    progress[field] = "long_ar" if field == "training_stage" else 1
    state_path.write_text(json.dumps(progress))
    for hook in source.accelerator.load_hooks:
        hook([], str(tmp_path))
    with pytest.raises(ValueError, match=field):
        source.ema.restore_after_model_load()


@pytest.mark.parametrize("stage", ["dmd"])
def test_raw_slot_export_error_identifies_the_slot(tmp_path, stage):
    from model_training.utils.ema import ema_export_metadata

    source = _ema_stage_trainer(stage, tmp_path)
    _advance_ema_stage(stage, source, 206)
    source.ema.save_checkpoint(tmp_path)
    slot = "pytorch_model_fsdp_0"
    with pytest.raises(ValueError) as error:
        ema_export_metadata(tmp_path / slot, source.get_step() + 1, slot)
    message = str(error.value)
    assert "EMA export" not in message
    assert "checkpoint" in message and slot in message


@pytest.mark.parametrize("consumer", ["reader", "export"])
@pytest.mark.parametrize("marker", [None, "long_ar_ema.json", "pytorch_model_fsdp_2"])
def test_ema_missing_metadata_errors_are_context_neutral(tmp_path, consumer, marker):
    from model_training.utils.ema import ema_export_metadata, read_ema_progress

    if marker == "long_ar_ema.json":
        (tmp_path / marker).write_text("{}")
    elif marker is not None:
        (tmp_path / marker).mkdir()
    with pytest.raises(ValueError) as error:
        if consumer == "reader":
            read_ema_progress(tmp_path)
        else:
            slot = "pytorch_model_fsdp_ema"
            ema_export_metadata(tmp_path / slot, 150, slot)
    message = str(error.value)
    assert "schema" in message and "ema.json" in message
    assert "resume" not in message.lower()
    assert "warm-start" not in message and "--project_dir" not in message
    if marker is not None:
        assert marker in message


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("step,active,averages", [(199, False, 0), (200, True, 0), (206, True, 2)])
def test_dmd_ema_video_labels_and_wandb_keys_for_both_validation_paths(
    tmp_path, monkeypatch, cp, step, active, averages
):
    from pathlib import Path

    import model_training.trainers.trainer_base as base
    from model_eval.validation_tasks import ValidationTask

    trainer = _small_dmd_ema_trainer(tmp_path)
    trainer.evaluation.config.validation_metrics = False
    _advance_dmd_ema(trainer, step)
    trainer.pipe.stages_conditioning_per_chunk = True
    trainer.pipe.transformer._cp_mesh = SimpleNamespace(size=lambda: 2) if cp else None
    video = torch.zeros(1, 2, 3, 1, 1)
    task = ValidationTask(
        key="dmd_sample",
        rendered_rgb=video,
        rendered_opacity=None,
        neighbors=None,
        camera_rays=None,
        w2cs=None,
        neighbor_w2cs=None,
        Ks=None,
        neighbor_Ks=None,
        prompt=None,
        prompt_attention_mask=None,
        encoded_negative_prompt=None,
        negative_prompt_attention_mask=None,
        num_inference_steps=4,
        rgb_gt=video,
    )
    routes = []
    videos = []

    def validate(route, datasets, task_plan, save_dir):
        routes.append(route)
        assert trainer.ema.is_cached == active
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.evaluation.recorder.save(video, task, save_dir)

    def save_video(result, path, fps):
        assert result.shape[0] == 2
        path.touch()

    def wandb_video(path, **kwargs):
        videos.append((Path(path), kwargs))
        return Path(path)

    trainer.evaluation._generate_standard = lambda *args: validate("single", *args)
    trainer.evaluation._generate_kv_cp = lambda *args: validate("cp", *args)

    class CaptionDataset:
        def __getitem__(self, index):
            return {"prompt_file": "prompt", "prompt": "text"}

    trainer.val_datasets = {0: CaptionDataset()}
    import model_eval.validation_scoring as scoring

    monkeypatch.setattr(scoring, "save_video", save_video)
    monkeypatch.setattr(base, "uses_wandb_logging", lambda mode: True)
    monkeypatch.setattr(wandb, "Video", wandb_video)
    updates = trainer.ema.optimizer_updates
    logs = {}
    trainer._validation_step(logs)
    key = "dmd_sample_ema" if active else "dmd_sample"
    assert routes == ["cp" if cp else "single"]
    assert logs["val/ema_active"] == int(active)
    assert logs["val/ema_updates"] == averages
    assert logs[f"val/{key}"].name == f"{key}.mp4"
    assert len(videos) == 1 and videos[0][1]["format"] == "mp4"
    assert task.key == "dmd_sample"
    assert trainer.ema.optimizer_updates == updates
    assert not trainer.ema.is_cached and trainer.pipe.transformer.training
