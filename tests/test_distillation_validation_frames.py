from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import model_eval.validation as validation_module
from model_eval.generation import SamplingConfig
from model_eval.validation import EvaluationRunner
from model_eval.validation_config import EvaluationConfig, EvaluationContext
from model_eval.validation_scoring import ValidationRecorder
from model_eval.validation_tasks import ValidationTask, ValidationTaskSpec
from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline
from model_training.trainers.dmd import DMD


class _DummyVae:
    device = torch.device("cpu")
    dtype = torch.bfloat16


class _DummyPipe:
    vae = _DummyVae()
    transformer = SimpleNamespace(cp_mesh=None, cp_rank=0, cp_world_size=1)

    def __init__(self) -> None:
        self.video_encode_calls = 0
        self.chunked_video_encode_calls = []

    def encode_video_frames(self, video_frames: torch.Tensor) -> torch.Tensor:
        self.video_encode_calls += 1
        batch_size, num_frames = video_frames.shape[:2]
        return torch.zeros(batch_size, 1, num_frames, 1, 1)

    def encode_video_frames_chunked_for_cp(
        self, video_frames: torch.Tensor, halo_latent_frames: int, *, gather: bool = True
    ) -> torch.Tensor:
        self.chunked_video_encode_calls.append((tuple(video_frames.shape), halo_latent_frames, gather))
        batch_size, num_frames = video_frames.shape[:2]
        return torch.zeros(batch_size, 1, num_frames, 1, 1)

    def encode_neighbors(self, neighbor_frames: torch.Tensor) -> torch.Tensor:
        batch_size, num_neighbors = neighbor_frames.shape[:2]
        return torch.zeros(batch_size, 1, num_neighbors, 1, 1)


def test_dmd_process_inputs_drops_raw_video_tensors_after_encoding() -> None:
    trainer = DMD.__new__(DMD)
    trainer.args = SimpleNamespace(dropout_rate=0.0, vae_chunk_halo_latent_frames=0)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.pipe = _DummyPipe()

    data = {
        "rgb_gt": torch.zeros(1, 1, 3, 4, 4),
        "rgb_rendered": torch.zeros(1, 1, 3, 4, 4),
        "opacity": torch.ones(1, 1, 4, 4),
        "rgb_neighbors": torch.zeros(1, 2, 3, 4, 4),
        "encoded_prompt": torch.zeros(1, 4, 8),
        "neighbor_w2cs": torch.zeros(1, 2, 4, 4),
        "neighbor_Ks": torch.zeros(1, 2, 3, 3),
        "camera_rays": torch.zeros(1, 1, 6),
        "w2cs": torch.zeros(1, 1, 4, 4),
        "Ks": torch.zeros(1, 1, 3, 3),
    }

    processed = trainer.process_inputs(data)

    assert "rgb_gt" not in processed
    assert "rgb_rendered" not in processed
    assert "rgb_neighbors" not in processed
    assert processed["condition"].shape == (1, 1, 1, 1, 1)
    assert processed["camera_rays"].dtype == torch.bfloat16
    assert processed["neighbors_condition"].shape == (1, 1, 2, 1, 1)


def test_dmd_process_inputs_requires_shards_only_with_context_parallel() -> None:
    trainer = DMD.__new__(DMD)
    trainer.args = SimpleNamespace(dropout_rate=0.0, vae_chunk_halo_latent_frames=3)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.pipe = _DummyPipe()

    data = {
        "rgb_gt": torch.zeros(1, 9, 3, 4, 4),
        "rgb_rendered": torch.zeros(1, 9, 3, 4, 4),
        "opacity": torch.ones(1, 9, 4, 4),
        "rgb_neighbors": torch.zeros(1, 2, 3, 4, 4),
        "encoded_prompt": torch.zeros(1, 4, 8),
        "neighbor_w2cs": torch.zeros(1, 2, 4, 4),
        "neighbor_Ks": torch.zeros(1, 2, 3, 3),
        "camera_rays": torch.zeros(1, 9, 6),
        "w2cs": torch.zeros(1, 9, 4, 4),
        "Ks": torch.zeros(1, 9, 3, 3),
    }

    trainer.process_inputs(data)

    assert trainer.pipe.video_encode_calls == 0
    assert trainer.pipe.chunked_video_encode_calls == [((1, 9, 3, 4, 4), 3, True)]

    trainer.pipe.transformer = SimpleNamespace(cp_mesh=None, cp_rank=0, cp_world_size=2)
    with pytest.raises(AssertionError, match="pre-sharded"):
        trainer.process_inputs(data)


def test_dmd_process_inputs_uses_cp_broadcast_dropout_decisions() -> None:
    class RecordingPipe(_DummyPipe):
        def __init__(self) -> None:
            super().__init__()
            self.encoded_video = None

        def encode_video_frames_chunked_for_cp(
            self, video_frames: torch.Tensor, halo_latent_frames: int, *, gather: bool = True
        ) -> torch.Tensor:
            self.encoded_video = video_frames.clone()
            return super().encode_video_frames_chunked_for_cp(video_frames, halo_latent_frames, gather=gather)

    trainer = DMD.__new__(DMD)
    trainer.args = SimpleNamespace(dropout_rate=0.5, vae_chunk_halo_latent_frames=0)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.pipe = RecordingPipe()
    trainer._is_cp_source_rank = lambda: False
    broadcasts = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([3])]

    def broadcast(tensor):
        tensor.copy_(broadcasts.pop(0).to(device=tensor.device, dtype=tensor.dtype))
        return tensor

    trainer._broadcast_cp_tensor = broadcast
    rendered = torch.arange(1 * 5 * 3 * 2 * 2, dtype=torch.float32).view(1, 5, 3, 2, 2)
    data = {
        "rgb_gt": torch.zeros(1, 5, 3, 2, 2),
        "rgb_rendered": rendered,
        "opacity": torch.ones(1, 5, 2, 2),
        "rgb_neighbors": torch.zeros(1, 2, 3, 2, 2),
        "encoded_prompt": torch.ones(1, 4, 8),
        "neighbor_w2cs": torch.zeros(1, 2, 4, 4),
        "neighbor_Ks": torch.zeros(1, 2, 3, 3),
        "camera_rays": torch.zeros(1, 5, 6),
        "w2cs": torch.zeros(1, 5, 4, 4),
        "Ks": torch.zeros(1, 5, 3, 3),
    }

    processed = trainer.process_inputs(data)

    assert broadcasts == []
    assert trainer.pipe.encoded_video is not None
    torch.testing.assert_close(trainer.pipe.encoded_video[:, :2], rendered[:, :2])
    torch.testing.assert_close(trainer.pipe.encoded_video[:, 2:], torch.zeros_like(rendered[:, 2:]))
    assert processed["ignore_neighbors"] is False


def test_kv_cache_validation_uses_cp_chunked_video_encode() -> None:
    pipe = ArtifixerKvCachePipeline.__new__(ArtifixerKvCachePipeline)
    pipe.vae = _DummyVae()
    # denoise_to_latents asserts the transformer is in eval mode.
    pipe.transformer = SimpleNamespace(training=False)
    pipe.encode_calls = []

    def encode_video_frames(_video_frames: torch.Tensor) -> torch.Tensor:
        raise AssertionError("validation must not full-encode all frames on each CP rank")

    def encode_video_frames_chunked_for_cp(
        video_frames: torch.Tensor, halo_latent_frames: int, *, gather: bool = True
    ) -> torch.Tensor:
        pipe.encode_calls.append((tuple(video_frames.shape), halo_latent_frames, gather))
        return torch.zeros(video_frames.shape[0], 1, 4, 1, 1)

    def generate_samples_from_batch(
        condition,
        rendered_opacity,
        neighbors_condition,
        camera_rays,
        w2cs,
        neighbor_w2cs,
        Ks,
        neighbor_Ks,
        encoded_prompt,
        num_inference_steps,
        use_exit_flag,
        show_progress=False,
        progress_bar_leave=True,
        **_kwargs,
    ):
        return condition

    pipe.encode_video_frames = encode_video_frames
    pipe.encode_video_frames_chunked_for_cp = encode_video_frames_chunked_for_cp
    pipe.generate_samples_from_batch = generate_samples_from_batch

    output = pipe.denoise_to_latents(
        rendered_rgb=torch.zeros(1, 17, 3, 4, 4),
        rendered_opacity=torch.ones(1, 17, 4, 4),
        neighbors=None,
        camera_rays=torch.zeros(1, 4, 6),
        w2cs=torch.zeros(1, 4, 4, 4),
        neighbor_w2cs=None,
        Ks=torch.zeros(1, 4, 3, 3),
        neighbor_Ks=None,
        prompt=torch.zeros(1, 4, 8),
        vae_chunk_halo_latent_frames=5,
    )

    assert pipe.encode_calls == [((1, 17, 3, 4, 4), 5, True)]
    assert output.shape == (1, 1, 4, 1, 1)


def test_dmd_cp_validation_keeps_task_cpu_and_decodes_latents_chunked(tmp_path: Path, monkeypatch) -> None:
    task = ValidationTask(
        key="default_2",
        rendered_rgb=torch.zeros(1, 17, 3, 4, 4),
        rendered_opacity=torch.ones(1, 17, 4, 4),
        neighbors=None,
        camera_rays=torch.zeros(1, 4, 6),
        w2cs=torch.zeros(1, 4, 4, 4),
        neighbor_w2cs=None,
        Ks=torch.zeros(1, 4, 3, 3),
        neighbor_Ks=None,
        prompt=torch.zeros(1, 4, 8),
        # Wan prompts are pre-encoded T5 embeddings: no token masks or stored negative.
        prompt_attention_mask=None,
        encoded_negative_prompt=None,
        negative_prompt_attention_mask=None,
        num_inference_steps=4,
        rgb_gt=torch.zeros(1, 17, 3, 4, 4),
    )
    copy_to_device_calls = []
    saved = []

    def build_task(*_args, copy_to_device=None, **_kwargs):
        copy_to_device_calls.append(copy_to_device)
        return task

    class Pipe:
        def __init__(self) -> None:
            self.denoise_kwargs = None
            self.decode_calls = []

        def forward_inference(self, **kwargs):
            raise AssertionError("DMD latent validation should call denoise_to_latents")

        def denoise_to_latents(self, **kwargs):
            self.denoise_kwargs = kwargs
            return torch.zeros(1, 1, 16, 1, 1)

        def decode_latents_to_video(self, latents, temporal_chunk_size=None):
            self.decode_calls.append((tuple(latents.shape), temporal_chunk_size))
            return torch.zeros(1, 17, 3, 4, 4)

    pipe = Pipe()
    pipe.stages_conditioning_per_chunk = True
    pipe.transformer = SimpleNamespace(cp_mesh=SimpleNamespace(size=lambda: 4))
    trainer = EvaluationRunner(
        pipe,
        EvaluationConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            sampling=SamplingConfig(num_inference_steps=4, vae_chunk_halo_latent_frames=7),
        ),
        EvaluationContext(torch.device("cpu")),
    )
    trainer.recorder = ValidationRecorder(trainer.config, trainer.context, 4)
    monkeypatch.setattr(validation_module, "build_validation_task", build_task)
    trainer.recorder.save = lambda output, save_task, save_dir, **kwargs: saved.append(
        (tuple(output.shape), save_task.key, save_dir)
    )

    trainer.generate_tasks({2: object()}, [(ValidationTaskSpec("default_2", 2), 0)], tmp_path)

    assert copy_to_device_calls == [False]
    assert pipe.denoise_kwargs is not None
    assert "return_latents" not in pipe.denoise_kwargs
    assert pipe.denoise_kwargs["vae_chunk_halo_latent_frames"] == 7
    assert pipe.decode_calls == [((1, 1, 16, 1, 1), 4)]
    assert saved == [((1, 17, 3, 4, 4), "default_2", tmp_path)]
