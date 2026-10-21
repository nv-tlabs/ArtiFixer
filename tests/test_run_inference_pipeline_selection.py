# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""run_inference wiring for --inference_pipeline=kv_cache (causal/AR eval of a
distilled model).

The flag is family-dispatched: Wan builds the KV-cache pipeline; Cosmos3
builds ArtifixerCosmos3KvCachePipeline whose chunked AR rollout runs against a
write-once per-layer KV cache with the sliding attention window sized by
the same --local_attn_size/--sink_size options as the Wan KV cache.
Rollout numerics are pinned against stored expected outputs in
tests/test_cosmos3_kv_cache_rollout.py and the pipeline plumbing in
tests/test_cosmos3_ar_rollout.py; here we pin the eval-script wiring — arg
parsing, pipeline construction, and process_item's per-branch
conditioning/saving behavior.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model_eval import inference as run_inference
from model_eval import inference_args
from model_training import stages
from model_training.checkpoint_metadata import checkpoint_metadata_path
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID, WAN22_TI2V_5B_MODEL_ID, flow_shift_for_model
from model_training.pipeline.cosmos3_kv_cache_pipeline import ArtifixerCosmos3KvCachePipeline
from model_training.pipeline.cosmos3_pipeline import ArtifixerCosmos3Pipeline
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline
from tests._helpers import (
    _reduced_real_transformer,
    _TinyCosmosBaseTransformer,
    _TinyCosmosVae,
    patch_kv_cache_pipeline_factories,
)

_BASE_ARGV = [
    "--checkpoint",
    "/ckpts/checkpoint_100/pytorch_model_fsdp_0",
    "--save_dir",
    "/eval",
    "--render_name",
    "checkpoint_100_pytorch_model_fsdp_0_chunk7_sink7",
    "--split_path",
    "/splits/split.json",
    "--dl3dv_dir",
    "/data/DL3DV-ALL-960P",
    "--prompt_dir",
    "/data/prompts",
]


def _parse_args(extra_argv: list[str], checkpoint_argv: list[str] | None = None):
    return inference_args.parse_args((checkpoint_argv or _BASE_ARGV[:2]) + _BASE_ARGV[2:] + extra_argv)


def test_inference_transformer_construction_defaults():
    args = _parse_args([])
    assert args.gradient_checkpointing is False
    assert args.checkpoint_every_n_blocks == 1


@pytest.mark.parametrize(
    "module_name",
    [
        "model_eval.run_inference",
        "model_eval.export_checkpoint_pt",
        "model_training.train",
        "model_training.diffusion_forcing",
        "model_training.distillation",
    ],
)
def test_cli_argument_parsing_without_runtime_imports(module_name):
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import runpy
import sys

class BlockRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'cv2', 'numpy', 'PIL', 'accelerate', 'diffusers',
            'einops', 'imageio', 'scipy', 'tqdm', 'h5py', 'transformers',
        }:
            raise RuntimeError(f'Argument parsing imported {fullname}')

sys.meta_path.insert(0, BlockRuntimeImports())
sys.argv = [sys.argv[1], '--help']
runpy.run_module(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, module_name],
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_nerfbusters_requires_experiment_before_runtime_imports():
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import runpy
import sys

class BlockRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'accelerate', 'diffusers'}:
            raise RuntimeError(f'Invalid arguments imported {fullname}')

sys.meta_path.insert(0, BlockRuntimeImports())
sys.argv = [
    'model_eval.run_inference', '--evalset', 'nerfbusters',
    '--checkpoint', '/unused.pt', '--save_dir', '/outputs', '--render_name', 'run',
    '--nerfbusters_dir', '/data', '--nerfbusters_recon_results_dir', '/recon',
    '--nerfbusters_captions_dir', '/captions',
]
runpy.run_module(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "--nerfbusters_recon_experiment_name is required" in result.stderr


def _stamped_checkpoint(checkpoint, pose_alignment="last") -> list[str]:
    """Give the parser the checkpoint's saved pose alignment."""
    path = checkpoint_metadata_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"camera_pose_alignment": pose_alignment}))
    return ["--checkpoint", str(checkpoint)]


@pytest.mark.parametrize("model_id", ["Wan-AI/Wan2.1-T2V-14B-Diffusers", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"])
@pytest.mark.parametrize("suffix", ["", ".pt"])
def test_parser_uses_mean_for_unrecorded_wan_and_rejects_last(tmp_path, model_id, suffix):
    argv = ["--model_id", model_id, "--checkpoint", str(tmp_path / f"model{suffix}")]
    args = _parse_args(argv)
    assert args.camera_pose_alignment == "mean"
    with pytest.raises(
        ValueError,
        match="has no camera_pose_alignment record; unrecorded Wan checkpoints use 'mean', but 'last' was requested",
    ):
        _parse_args(argv + ["--camera_pose_alignment", "last"])


@pytest.mark.parametrize("model_id", [COSMOS3_NANO_MODEL_ID, WAN22_TI2V_5B_MODEL_ID])
@pytest.mark.parametrize("metadata", [None, {}, {"run_id": "run"}])
@pytest.mark.parametrize("requested", [None, "mean", "last"])
def test_parser_requires_pose_alignment_metadata_for_other_models(tmp_path, model_id, metadata, requested):
    checkpoint = tmp_path / "model.pt"
    if metadata is not None:
        checkpoint.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    argv = ["--model_id", model_id, "--checkpoint", str(checkpoint)]
    if requested is not None:
        argv += ["--camera_pose_alignment", requested]
    with pytest.raises(ValueError, match=f"has no camera_pose_alignment record; required for {model_id!r}"):
        _parse_args(argv)


@pytest.mark.parametrize(
    ("model_id", "metadata", "pose_alignment"),
    [
        (WAN22_TI2V_5B_MODEL_ID, {"camera_pose_alignment": "mean"}, "mean"),
        (WAN22_TI2V_5B_MODEL_ID, {"camera_pose_alignment": "last"}, "last"),
        (COSMOS3_NANO_MODEL_ID, {"camera_pose_alignment": "mean"}, "mean"),
        (COSMOS3_NANO_MODEL_ID, {"camera_pose_alignment": "last"}, "last"),
    ],
)
def test_parser_honors_recorded_pose_alignment(tmp_path, model_id, metadata, pose_alignment):
    checkpoint = tmp_path / "model.pt"
    checkpoint.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    argv = ["--model_id", model_id, "--checkpoint", str(checkpoint)]
    args = _parse_args(argv)
    assert args.camera_pose_alignment == pose_alignment
    explicit = _parse_args(argv + ["--camera_pose_alignment", pose_alignment])
    assert explicit.camera_pose_alignment == pose_alignment


@pytest.mark.parametrize("pose", [None, "first", "invalid"])
@pytest.mark.parametrize("model_id", [WAN22_TI2V_5B_MODEL_ID, COSMOS3_NANO_MODEL_ID])
def test_parser_rejects_invalid_pose_alignment_metadata(tmp_path, pose, model_id):
    checkpoint = tmp_path / "model.pt"
    checkpoint.with_suffix(".metadata.json").write_text(json.dumps({"camera_pose_alignment": pose}))
    with pytest.raises(ValueError, match=f"records camera_pose_alignment={pose!r}; expected"):
        _parse_args(["--checkpoint", str(checkpoint), "--model_id", model_id])


@pytest.mark.parametrize("evalset", inference_args.EVALSETS)
@pytest.mark.parametrize("pose_alignment", ["mean", "last"])
@pytest.mark.parametrize(
    ("model_id", "resize_multiple"),
    [("Wan-AI/Wan2.1-T2V-14B-Diffusers", 16), (WAN22_TI2V_5B_MODEL_ID, 32), (COSMOS3_NANO_MODEL_ID, 32)],
)
def test_create_dataset_forwards_model_profile_and_pose_alignment(
    monkeypatch, tmp_path, evalset, pose_alignment, model_id, resize_multiple
):
    extra = []
    if evalset == "3dgrut_dl3dv_ours":
        factory = "model_eval.inference.DL3DVHDF5EvalDataset"
    elif evalset == "nerfbusters":
        factory = "model_eval.inference.NerfbustersEvalDataset"
        extra = [
            "--nerfbusters_dir",
            "/n",
            "--nerfbusters_recon_results_dir",
            "/r",
            "--nerfbusters_recon_experiment_name",
            "reconstruction",
            "--nerfbusters_captions_dir",
            "/p",
        ]
    elif run_inference.is_dl3dv_reconstruction_evalset(evalset):
        factory = "model_eval.datasets.dl3dv_reconstruction_eval.DL3DVReconstructionEvalDataset"
        extra = ["--recon_results_dir", "/r"]
    else:
        factory = "model_eval.datasets.reconstructed_colmap_eval.ReconstructedColmapEvalDataset"
    module_name, class_name = factory.rsplit(".", 1)
    monkeypatch.setattr(importlib.import_module(module_name), class_name, lambda **kwargs: kwargs)
    checkpoint = tmp_path / "model.pt"
    checkpoint.with_suffix(".metadata.json").write_text(json.dumps({"camera_pose_alignment": pose_alignment}))
    args = _parse_args(
        ["--model_id", model_id, "--evalset", evalset] + extra,
        checkpoint_argv=["--checkpoint", str(checkpoint)],
    )
    kwargs = run_inference.create_dataset(args, rank=0)
    assert kwargs["camera_pose_alignment"] == pose_alignment
    assert kwargs["prompt_format"] == run_inference.prompt_format_for_model(model_id)
    assert kwargs["resize_multiple"] == resize_multiple


@pytest.mark.parametrize("module_name", ["train", "diffusion_forcing", "distillation"])
def test_cosmos3_training_defaults_to_last_pose_alignment(module_name):
    parser = importlib.import_module(f"model_training.{module_name}").build_parser()
    argv = _BASE_ARGV[-6:] + ["--project_dir", "/training", "--model_id", COSMOS3_NANO_MODEL_ID]
    if module_name == "distillation":
        argv += ["--base_checkpoint_dir", "/student", "--base_checkpoint_dir_critic", "/critic"]
    if module_name == "diffusion_forcing":
        argv += ["--base_checkpoint_dir", "/student"]
    args = parser.parse_args(argv)
    assert args.camera_pose_alignment == "last"


@pytest.mark.parametrize("suffix", [".pt", ""])
def test_eval_reads_cosmos3_pose_alignment_from_pt_and_dcp(tmp_path, suffix):
    checkpoint = tmp_path / f"model{suffix}"
    path = checkpoint_metadata_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(camera_pose_alignment="last")))
    argv = ["--model_id", COSMOS3_NANO_MODEL_ID, "--checkpoint", str(checkpoint)]
    args = _parse_args(argv)
    assert args.camera_pose_alignment == "last"


@pytest.mark.parametrize("factory_name", ["get_pipe", "get_kv_cache_pipe"])
def test_shared_factories_construct_cosmos3(monkeypatch, factory_name):
    from model_training.utils import train_utils

    monkeypatch.setattr(train_utils, "configure_attention", lambda *a, **k: None)
    monkeypatch.setattr(train_utils, "_vae_for_model", lambda *a, **k: object())
    monkeypatch.setattr(train_utils, "_scheduler_for_model", lambda *a, **k: object())
    monkeypatch.setattr(train_utils, "_transformer_from_config", lambda *a, **k: object())
    monkeypatch.setattr(train_utils, "validate_model_profile", lambda *a: None)
    monkeypatch.setattr(train_utils, "ArtifixerCosmos3Pipeline", lambda **kwargs: kwargs)
    monkeypatch.setattr(train_utils, "ArtifixerCosmos3KvCachePipeline", lambda **kwargs: kwargs)
    args = SimpleNamespace(
        model_id=COSMOS3_NANO_MODEL_ID,
        attention_backend=None,
        gradient_checkpointing=False,
        frames_per_block=2,
        local_attn_size=4,
        sink_size=0,
    )
    if factory_name == "get_pipe":
        pipe = train_utils.get_pipe(args, False, frames_per_block=None, device="cpu")
    else:
        pipe = train_utils.get_kv_cache_pipe(args, device="cpu")
    assert pipe["gradient_checkpointing"] is False


@pytest.mark.parametrize("role", ["teacher", "fake_score"])
def test_distillation_factories_use_the_single_cosmos3_architecture(monkeypatch, role):
    monkeypatch.setattr(stages, "_transformer_from_config", lambda *a, **k: object())
    calls = []

    def wrap(transformer, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(stages, "ArtifixerCosmos3Transformer", wrap)
    args = SimpleNamespace(model_id=COSMOS3_NANO_MODEL_ID, gradient_checkpointing=True)
    pipe = SimpleNamespace(
        attention_backend=None,
        vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8)),
    )
    model = stages.build_score_transformer(
        args, COSMOS3_NANO_MODEL_ID, pipe, gradient_checkpointing=role == "fake_score"
    )
    assert len(calls) == 1
    assert model.gradient_checkpointing == (role != "teacher")
    assert model.vae_scale_factor_temporal == 4
    assert model.vae_scale_factor_spatial == 8


def _patch_pipeline_factories(monkeypatch) -> tuple[list, list]:
    cosmos3_constructed: list = []
    wan_constructed: list = []
    patch_kv_cache_pipeline_factories(monkeypatch, cosmos3_constructed, wan_constructed)
    # These tests check pipeline selection, not loading weights or reading metadata.
    monkeypatch.setattr(run_inference, "load_transformer_checkpoint", lambda *args, **kwargs: None)
    return cosmos3_constructed, wan_constructed


def test_parser_no_longer_rejects_cosmos3_kv_cache(tmp_path):
    args = _parse_args(
        ["--model_id", COSMOS3_NANO_MODEL_ID, "--inference_pipeline", "kv_cache"],
        checkpoint_argv=_stamped_checkpoint(tmp_path),
    )

    assert args.inference_pipeline == "kv_cache"
    # AR chunk size and few-step defaults shared with the Wan kv_cache eval.
    assert args.frames_per_block == 7
    assert args.num_inference_steps == 4


def test_all_pipelines_accept_context_parallelism_at_parse_time(tmp_path):
    """Both families and both pipelines run under CP; per-run sizing is
    enforced where the frame counts are known (the pipeline's
    validate_cp_size)."""
    checkpoints = {
        COSMOS3_NANO_MODEL_ID: _stamped_checkpoint(tmp_path / "cosmos3"),
        WAN22_TI2V_5B_MODEL_ID: _stamped_checkpoint(tmp_path / "wan22", "mean"),
    }
    for model_id, checkpoint_argv in checkpoints.items():
        for pipeline in ("kv_cache", "bidirectional"):
            _parse_args(
                ["--model_id", model_id, "--inference_pipeline", pipeline, "--context_parallel_size", "2"],
                checkpoint_argv=checkpoint_argv,
            )


@pytest.mark.parametrize("scale", ["1", "3", "5"])
def test_bidirectional_parser_accepts_text_guidance(scale):
    args = _parse_args(["--inference_pipeline", "bidirectional", "--text_guidance_scale", scale])

    assert args.text_guidance_scale == float(scale)


@pytest.mark.parametrize("scale", ["0", "-1", "nan", "inf"])
def test_parser_rejects_invalid_text_guidance(scale, capsys):
    with pytest.raises(SystemExit):
        _parse_args(["--inference_pipeline", "bidirectional", "--text_guidance_scale", scale])

    assert "must be finite and at least 1" in capsys.readouterr().err


def test_kv_cache_parser_rejects_ignored_text_guidance(capsys):
    with pytest.raises(SystemExit):
        _parse_args(["--inference_pipeline", "kv_cache", "--text_guidance_scale", "3"])

    assert "only supported with --inference_pipeline bidirectional" in capsys.readouterr().err


def test_unspecified_guidance_preserves_pipeline_defaults():
    assert _parse_args(["--inference_pipeline", "bidirectional"]).text_guidance_scale is None
    assert _parse_args(["--inference_pipeline", "kv_cache"]).text_guidance_scale is None


def test_cosmos3_kv_cache_constructs_the_ar_pipeline(monkeypatch, tmp_path):
    cosmos3_constructed, wan_constructed = _patch_pipeline_factories(monkeypatch)
    args = _parse_args(
        ["--model_id", COSMOS3_NANO_MODEL_ID, "--inference_pipeline", "kv_cache"],
        checkpoint_argv=_stamped_checkpoint(tmp_path),
    )

    run_inference.get_eval_pipe(args, torch.device("cpu"))

    assert wan_constructed == []
    (kwargs,) = cosmos3_constructed
    # --frames_per_block IS the AR rollout chunk size; the scheduler shift is
    # the model profile's (get_kv_cache_pipe with flow_shift=None), and the
    # --local_attn_size/--sink_size window options reach the Cosmos3 pipe
    # exactly as they reach the Wan pipe.
    assert kwargs["ar_frames_per_chunk"] == 7
    assert kwargs["flow_shift"] == flow_shift_for_model(COSMOS3_NANO_MODEL_ID)
    assert kwargs["local_attn_size"] == 21
    assert kwargs["sink_size"] == 7


def test_cosmos3_kv_cache_forwards_an_explicit_frames_per_block(monkeypatch, tmp_path):
    cosmos3_constructed, _ = _patch_pipeline_factories(monkeypatch)
    args = _parse_args(
        ["--model_id", COSMOS3_NANO_MODEL_ID, "--inference_pipeline", "kv_cache", "--frames_per_block", "3"],
        checkpoint_argv=_stamped_checkpoint(tmp_path),
    )

    run_inference.get_eval_pipe(args, torch.device("cpu"))

    assert cosmos3_constructed[0]["ar_frames_per_chunk"] == 3


def test_wan_kv_cache_construction_forwards_the_windowing_knobs(monkeypatch, tmp_path):
    cosmos3_constructed, wan_constructed = _patch_pipeline_factories(monkeypatch)
    args = _parse_args(
        ["--model_id", WAN22_TI2V_5B_MODEL_ID, "--inference_pipeline", "kv_cache"],
        checkpoint_argv=_stamped_checkpoint(tmp_path, "mean"),
    )

    run_inference.get_eval_pipe(args, torch.device("cpu"))

    assert cosmos3_constructed == []
    (kwargs,) = wan_constructed
    assert kwargs["ar_frames_per_chunk"] == 7
    assert kwargs["local_attn_size"] == 21
    assert kwargs["sink_size"] == 7
    assert kwargs["flow_shift"] == flow_shift_for_model(WAN22_TI2V_5B_MODEL_ID)
    assert "frames_per_block" not in kwargs


def test_latent_frame_multiple_reads_each_familys_chunk_size():
    """The chunk-boundary padding multiple is pinned against the REAL
    Cosmos3 rollout pipeline, so renaming ar_frames_per_chunk breaks here
    instead of silently breaking the eval script's padding. (Wan's
    constructor coverage lives in test_kv_cache_pipeline.py.)"""
    cosmos3_pipe = ArtifixerCosmos3KvCachePipeline(
        vae=_TinyCosmosVae(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
        ar_frames_per_chunk=7,
        local_attn_size=21,
        sink_size=7,
    )
    assert cosmos3_pipe.latent_frame_multiple() == 7

    # Both rollout families read their chunk size; the base rule is 1 —
    # bidirectional pipes need no chunk padding.
    assert ArtifixerKvCachePipeline.latent_frame_multiple(SimpleNamespace(ar_frames_per_chunk=3)) == 3
    assert ArtifixerPipelineBase.latent_frame_multiple(SimpleNamespace()) == 1


def _eval_item(latent_frames: int, hw: int, *, cosmos3: bool) -> dict:
    pixel_frames = (latent_frames - 1) * 4 + 1
    item = dict(
        scene_id="scene",
        rgb_rendered=torch.rand(pixel_frames, 3, hw, hw),
        rgb_gt=torch.rand(pixel_frames, 3, hw, hw),
        rgb_neighbors=torch.rand(2, 3, hw, hw),
        opacity=torch.rand(pixel_frames, hw, hw),
        camera_rays=torch.rand(latent_frames, hw, hw, 6),
        w2cs=torch.eye(4).expand(latent_frames, 4, 4).clone(),
        Ks=torch.eye(3).expand(latent_frames, 3, 3).clone(),
        neighbor_w2cs=torch.eye(4).expand(2, 4, 4).clone(),
        neighbor_Ks=torch.eye(3).expand(2, 3, 3).clone(),
    )
    if cosmos3:
        item.update(
            encoded_prompt=torch.tensor([11, 12, 7, 8]),
            prompt_attention_mask=torch.ones(4, dtype=torch.bool),
            encoded_negative_prompt=torch.tensor([1, 1, 1, 1]),
            negative_prompt_attention_mask=torch.ones(4, dtype=torch.bool),
        )
    else:
        item["encoded_prompt"] = torch.rand(4, 16)
    return item


def _eval_args(model_id: str, *, save_frame_outputs_only: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        inference_pipeline="kv_cache",
        replace_if_exists=False,
        save_frame_outputs_only=save_frame_outputs_only,
        max_neighbors_per_encode=None,
        num_inference_steps=4,
        text_guidance_scale=None,
        seed=42,
        render_diagnostics=False,
        output_fps=15,
    )


class _NoiseEvalPipe:
    stages_conditioning_per_chunk = True
    cp_raw_frame_padding = ArtifixerCosmos3Pipeline.cp_raw_frame_padding
    latent_frame_multiple = ArtifixerPipelineBase.latent_frame_multiple

    def __init__(self, device=torch.device("cpu")):
        self.device = device
        self.vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=1))
        self.transformer = SimpleNamespace(_cp_world_size=1, cp_mesh=None)
        self.noise = []
        self.events = []

    def validate_cp_size(self, *args):
        pass

    def denoise_to_latents(self, rendered_rgb, **kwargs):
        self.events.append("denoise")
        batch, self.pixel_frames, channels, height, width = rendered_rgb.shape
        latent_frames = (self.pixel_frames - 1) // 4 + 1
        condition = torch.zeros(batch, channels, latent_frames, height, width, device=self.device)
        opacity = torch.zeros(batch, self.pixel_frames, height, width, device=self.device)
        latents = ArtifixerPipelineBase.prepare_latents(self, condition, opacity, is_first_chunk=True)
        self.noise.append(latents.cpu().clone())
        return latents

    def decode_latents_to_video(self, latents):
        self.events.append("decode")
        return latents.repeat_interleave(4, dim=2)[:, :, : self.pixel_frames].transpose(1, 2).sigmoid()

    def forward_inference(self, **kwargs):
        return self.decode_latents_to_video(self.denoise_to_latents(**kwargs))

    def clear_inference_caches(self):
        self.events.append("clear")


def _noise_args(pipeline, *extra):
    return _parse_args(["--inference_pipeline", pipeline, "--save_frame_outputs_only", "--seed", "71", *extra])


def _noise_item(scene_id="target"):
    item = _eval_item(2, 4, cosmos3=False)
    item.update(scene_id=scene_id, frame_indices=torch.arange(20, 25), chunk_idx=2)
    return item


def _process_noise_item(pipe, item, args, output_dir, *, save_outputs=False):
    run_inference.process_item(
        pipe, item, args, output_dir, rank=0, device=pipe.device, vae_temporal_scale=4, save_outputs=save_outputs
    )


@pytest.mark.parametrize("pipeline", ["kv_cache", "bidirectional"])
def test_item_seed_is_stable_across_order_skips_and_sharding(monkeypatch, tmp_path, pipeline):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    args = _noise_args(pipeline)
    earlier, target = _noise_item("earlier"), _noise_item()
    pipe = _NoiseEvalPipe()

    def run(items, output_name, rank=0, world_size=1):
        before = len(pipe.noise)
        run_inference.process_items_with_context_parallel(
            pipe, items, args, tmp_path / output_name, rank, world_size, device=pipe.device
        )
        return pipe.noise[before:]

    with torch.random.fork_rng(devices=[]):
        (alone,) = run([target], "alone")
        after_other = run([earlier, target], "ordered")
        torch.rand(19)
        (filtered,) = run([target], "filtered")
        run([earlier], "resumed")
        (resumed,) = run([earlier, target], "resumed")
        (resharded,) = run([earlier, target], "resharded", rank=1, world_size=2)
        assert run([target], "alone") == []
    assert len(after_other) == 2
    for actual in (after_other[-1], filtered, resumed, resharded):
        assert torch.equal(actual, alone)


@pytest.mark.parametrize("pipeline", ["kv_cache", "bidirectional"])
def test_item_seed_changes_noise_when_base_seed_changes(tmp_path, pipeline):
    item = _noise_item()
    pipe = _NoiseEvalPipe()
    with torch.random.fork_rng(devices=[]):
        for seed in (71, 72):
            torch.random.default_generator.manual_seed(123)
            _process_noise_item(pipe, item, _noise_args(pipeline, "--seed", str(seed)), tmp_path)
    assert len(pipe.noise) == 2
    assert not torch.equal(*pipe.noise)


@pytest.mark.parametrize(
    ("field", "value"),
    [("scene_id", "other"), ("split", 3), ("frame_indices", [24, 23, 22, 21, 20]), ("chunk_idx", 4)],
)
def test_item_seed_uses_stable_source_identity(tmp_path, field, value):
    item = _noise_item()
    pipe = _NoiseEvalPipe()
    args = _noise_args("kv_cache", "--replace_if_exists")
    with torch.random.fork_rng(devices=[]):
        for current in (item, dict(item, **{field: value})):
            torch.random.default_generator.manual_seed(123)
            _process_noise_item(pipe, current, args, tmp_path)
    assert not torch.equal(*pipe.noise)


def test_item_seed_ignores_metric_names_and_frame_index_container(tmp_path):
    item = _noise_item()
    same_source = dict(item, frame_indices=item["frame_indices"].tolist(), gt_index=[-1, 4, 5, -1, 8])
    pipe = _NoiseEvalPipe()
    args = _noise_args("kv_cache", "--replace_if_exists")
    _process_noise_item(pipe, item, args, tmp_path)
    torch.rand(13)
    _process_noise_item(pipe, same_source, args, tmp_path)
    assert torch.equal(*pipe.noise)


@pytest.mark.parametrize("pipeline", ["kv_cache", "bidirectional"])
@pytest.mark.parametrize("save_outputs", [False, True])
def test_item_seed_restores_caller_rng(tmp_path, pipeline, save_outputs):
    item = _noise_item()
    before = torch.random.get_rng_state().clone()
    _process_noise_item(_NoiseEvalPipe(), item, _noise_args(pipeline), tmp_path, save_outputs=save_outputs)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("pipeline", ["kv_cache", "bidirectional"])
def test_item_seed_restores_caller_rng_when_inference_raises(monkeypatch, tmp_path, pipeline):
    item = _noise_item()
    pipe = _NoiseEvalPipe()

    def fail(**kwargs):
        torch.randn(7)
        raise RuntimeError("inference failed")

    method = "denoise_to_latents" if pipeline == "kv_cache" else "forward_inference"
    monkeypatch.setattr(pipe, method, fail)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="inference failed"):
        _process_noise_item(pipe, item, _noise_args(pipeline), tmp_path)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("pipeline", ["kv_cache", "bidirectional"])
def test_item_seed_matches_across_context_parallel_group_ranks(monkeypatch, tmp_path, pipeline):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        run_inference,
        "create_context_parallel_meshes",
        lambda rank, world_size, size, device_type: (None, world_size // size, rank // size, rank % size),
    )
    args = _noise_args(pipeline, "--context_parallel_size", "2")
    items = [_noise_item("earlier"), _noise_item()]
    pipes = [_NoiseEvalPipe(), _NoiseEvalPipe()]
    for rank, pipe in zip((2, 3), pipes):
        run_inference.process_items_with_context_parallel(
            pipe, items, args, tmp_path / str(rank), rank, world_size=4, device=pipe.device
        )
        assert len(pipe.noise) == 1
    assert torch.equal(pipes[0].noise[0], pipes[1].noise[0])
    if pipeline == "kv_cache":
        assert pipes[0].events == ["denoise", "decode"]
        assert pipes[1].events == ["denoise", "clear"]


@pytest.mark.parametrize("pipeline", ["kv_cache", "bidirectional"])
def test_item_seed_keeps_diagnostics_reproducible_when_other_diagnostics_are_skipped(monkeypatch, tmp_path, pipeline):
    def save_comparison(**kwargs):
        (kwargs["save_dir"] / f"{kwargs['key']}.mp4").touch()

    monkeypatch.setattr(run_inference, "save_comparison_output", save_comparison)
    monkeypatch.setattr(run_inference, "video_file_complete", lambda path, count: path.exists())
    item = _noise_item()
    pipe = _NoiseEvalPipe()
    args = _noise_args(pipeline, "--render_diagnostics")
    args.save_frame_outputs_only = False
    before = torch.random.get_rng_state().clone()
    _process_noise_item(pipe, item, args, tmp_path, save_outputs=True)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert len(pipe.noise) == 4
    original_main, original_no_text = pipe.noise[0], pipe.noise[-1]
    scene_dir = tmp_path / item["scene_id"]
    (scene_dir / "no_text_chunk0002.mp4").unlink()
    torch.rand(17)
    before = torch.random.get_rng_state().clone()
    _process_noise_item(pipe, item, args, tmp_path, save_outputs=True)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert len(pipe.noise) == 6
    assert torch.equal(pipe.noise[-2], original_main)
    assert torch.equal(pipe.noise[-1], original_no_text)
    assert (scene_dir / "no_text_chunk0002.mp4").is_file()
    other = dict(item, chunk_idx=3, frame_indices=torch.arange(25, 30))
    _process_noise_item(pipe, other, args, tmp_path, save_outputs=True)
    for name in run_inference.DIAGNOSTIC_NAMES:
        assert (scene_dir / f"{name}_chunk0002.mp4").is_file()
        assert (scene_dir / f"{name}_chunk0003.mp4").is_file()


def test_frame_only_output_does_not_generate_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_inference, "save_comparison_output", lambda **kwargs: pytest.fail("frame-only output wrote media")
    )
    pipe = _NoiseEvalPipe()
    _process_noise_item(
        pipe, _noise_item(), _noise_args("kv_cache", "--render_diagnostics"), tmp_path, save_outputs=True
    )
    assert len(pipe.noise) == 1


def test_item_seed_preserves_cuda_rng_states(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    item = _noise_item()
    device = torch.device("cuda", torch.cuda.current_device())
    pipe = _NoiseEvalPipe(device)
    args = _noise_args("kv_cache")
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all()
        _process_noise_item(pipe, item, args, tmp_path)
        assert torch.equal(torch.random.get_rng_state(), cpu_before)
        for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()):
            assert torch.equal(before, after)
        torch.randn(13, device=device)
        _process_noise_item(pipe, item, args, tmp_path)
        assert torch.equal(*pipe.noise)
        _process_noise_item(pipe, item, _noise_args("kv_cache", "--seed", "72"), tmp_path)
        assert not torch.equal(pipe.noise[0], pipe.noise[-1])


@pytest.mark.parametrize("pipeline", ["kv_cache", "bidirectional"])
@pytest.mark.parametrize("model_id", [COSMOS3_NANO_MODEL_ID, WAN22_TI2V_5B_MODEL_ID])
@pytest.mark.parametrize("with_reference_rays", [False, True])
def test_process_item_forwards_cosmos3_reference_rays_without_temporal_padding(
    monkeypatch, tmp_path, pipeline, model_id, with_reference_rays
):
    pipe = _NoiseEvalPipe()
    monkeypatch.setattr(pipe, "latent_frame_multiple", lambda: 2)
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1)

    method = "denoise_to_latents" if pipeline == "kv_cache" else "forward_inference"
    monkeypatch.setattr(pipe, method, capture)
    item = _eval_item(3, 4, cosmos3=model_id == COSMOS3_NANO_MODEL_ID)
    if with_reference_rays:
        item["neighbor_camera_rays"] = torch.randn(2, 4, 4, 6)
    args = _eval_args(model_id, save_frame_outputs_only=True)
    args.inference_pipeline = pipeline
    _process_noise_item(pipe, item, args, tmp_path, save_outputs=False)
    assert captured["camera_rays"].shape == (1, 4, 4, 4, 6)
    if model_id == COSMOS3_NANO_MODEL_ID and with_reference_rays:
        actual = captured["neighbor_camera_rays"]
        assert actual.shape == (1, 2, 4, 4, 6)
        assert torch.equal(actual, item["neighbor_camera_rays"].unsqueeze(0).to(torch.bfloat16))
    else:
        assert "neighbor_camera_rays" not in captured


@pytest.mark.parametrize("pipeline", ["kv_cache", "bidirectional"])
@pytest.mark.parametrize("neighbor_hw", [(32, 64), (64, 32)])
def test_joint_process_item_rejects_mismatched_spatial_grids_before_inference(
    monkeypatch, tmp_path, pipeline, neighbor_hw
):
    pipe = _NoiseEvalPipe()
    item = _eval_item(3, 32, cosmos3=True)
    item["scene_id"] = "mismatched_views"
    item["rgb_neighbors"] = torch.rand(2, 3, *neighbor_hw)
    item["neighbor_camera_rays"] = torch.rand(2, *neighbor_hw, 6)
    args = _eval_args(COSMOS3_NANO_MODEL_ID, save_frame_outputs_only=True)
    args.inference_pipeline = pipeline

    def fail(**kwargs):
        raise AssertionError("mismatched grids reached the pipeline")

    method = "denoise_to_latents" if pipeline == "kv_cache" else "forward_inference"
    monkeypatch.setattr(pipe, method, fail)
    with pytest.raises(ValueError, match="mismatched_views") as error:
        _process_noise_item(pipe, item, args, tmp_path, save_outputs=False)
    assert "(32, 32)" in str(error.value)
    assert str(neighbor_hw) in str(error.value)


@pytest.mark.parametrize("pipeline", ["kv_cache", "bidirectional"])
def test_wan_process_item_preserves_distinct_reference_spatial_grids(monkeypatch, tmp_path, pipeline):
    pipe = _NoiseEvalPipe()
    item = _eval_item(3, 32, cosmos3=False)
    item["rgb_neighbors"] = torch.rand(2, 3, 64, 32)
    args = _eval_args(WAN22_TI2V_5B_MODEL_ID, save_frame_outputs_only=True)
    args.inference_pipeline = pipeline
    captured = []

    def capture(**kwargs):
        captured.append(kwargs["neighbors"].shape[-2:])
        return torch.zeros(1)

    method = "denoise_to_latents" if pipeline == "kv_cache" else "forward_inference"
    monkeypatch.setattr(pipe, method, capture)
    _process_noise_item(pipe, item, args, tmp_path, save_outputs=False)
    assert captured == [(64, 32)]


def test_process_item_cosmos3_kv_cache_keeps_the_denoise_decode_split(monkeypatch, tmp_path):
    """cosmos3 + kv_cache runs the same denoise -> decode split as the Wan
    branch: the rollout receives the full padded clip and the complete
    conditioning payload, non-saving CP ranks return after denoise without
    decoding (the Cosmos3 cache clear is a no-op — the rollout cache is
    local to the denoise call), and the saving rank writes frames/comparison
    exactly like the Wan kv_cache branch."""
    pipe = ArtifixerCosmos3KvCachePipeline(
        vae=_TinyCosmosVae(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
        ar_frames_per_chunk=2,
        local_attn_size=21,
        sink_size=7,
    )
    latent_frames, hw = 3, 32  # 3 latent frames pad to 4 (two 2-frame AR chunks)
    pixel_frames = (latent_frames - 1) * 4 + 1
    padded_pixel_frames = (4 - 1) * 4 + 1
    captured: dict = {}
    calls: list[str] = []

    def fake_denoise(**kwargs):
        calls.append("denoise")
        captured.update(kwargs)
        return torch.zeros(1)

    def fake_decode(latents):
        calls.append("decode")
        return torch.rand(1, padded_pixel_frames, 3, hw, hw)

    def rejected_forward_inference(*args, **kwargs):
        raise AssertionError("the kv_cache eval path must use the denoise/decode split, not forward_inference")

    monkeypatch.setattr(pipe, "denoise_to_latents", fake_denoise)
    monkeypatch.setattr(pipe, "decode_latents_to_video", fake_decode)
    monkeypatch.setattr(pipe, "clear_inference_caches", lambda: calls.append("clear"))
    monkeypatch.setattr(pipe, "forward_inference", rejected_forward_inference)
    comparisons: list = []
    monkeypatch.setattr(run_inference, "save_comparison_output", lambda **kwargs: comparisons.append(kwargs))

    item = _eval_item(latent_frames, hw, cosmos3=True)
    args = _eval_args(COSMOS3_NANO_MODEL_ID, save_frame_outputs_only=False)

    run_inference.process_item(pipe, item, args, tmp_path, rank=0, device=torch.device("cpu"), vae_temporal_scale=4)

    assert calls == ["denoise", "decode"]
    # The AR rollout receives the FULL clip, padded to the chunk boundary.
    assert captured["rendered_rgb"].shape == (1, padded_pixel_frames, 3, hw, hw)
    assert captured["rendered_opacity"].shape == (1, padded_pixel_frames, hw, hw)
    assert captured["camera_rays"].shape == (1, 4, hw, hw, 6)
    assert captured["w2cs"].shape == (1, 4, 4, 4)
    assert captured["Ks"].shape == (1, 4, 3, 3)
    # Conditioning parity with the bidirectional cosmos3 branch: prompt payload
    # and negative prompt (ignored by the KV-cache pipe — no CFG on the distilled
    # few-step path) both arrive.
    assert torch.equal(captured["prompt"], item["encoded_prompt"].unsqueeze(0))
    assert torch.equal(captured["prompt_attention_mask"], item["prompt_attention_mask"].unsqueeze(0))
    assert torch.equal(captured["negative_prompt"], item["encoded_negative_prompt"].unsqueeze(0))
    assert captured["num_inference_steps"] == 4

    # Frame-output saving parity with the Wan kv_cache branch: the comparison
    # grid is saved (bidirectional skips it) plus metric and video PNGs for
    # every original (unpadded) frame.
    assert len(comparisons) == 1
    assert comparisons[0]["key"] == "comparison"
    expected_names = [f"{index:05d}.png" for index in range(pixel_frames)]
    for kind in ("pred", "gt", "rendered"):
        for subdir in ("frames", "video_frames"):
            saved = sorted(path.name for path in (tmp_path / "scene" / subdir / "batch_0000" / kind).glob("*.png"))
            assert saved == expected_names, (kind, subdir)

    # Non-saving CP ranks return after denoise: clear (a no-op on Cosmos3)
    # runs and the decode is skipped, matching the Wan branch.
    calls.clear()
    run_inference.process_item(
        pipe,
        _eval_item(latent_frames, hw, cosmos3=True),
        args,
        tmp_path / "cp",
        rank=1,
        device=torch.device("cpu"),
        vae_temporal_scale=4,
        save_outputs=False,
    )
    assert calls == ["denoise", "clear"]


def test_process_item_wan_kv_cache_keeps_the_denoise_decode_split(tmp_path):
    latent_frames, hw = 2, 32
    pixel_frames = (latent_frames - 1) * 4 + 1
    calls: list[str] = []
    captured: dict = {}

    class _FakeWanKvPipe:
        ar_frames_per_chunk = 1
        transformer = SimpleNamespace(_cp_world_size=1)
        # The base raw-padding rule and the rollout chunk multiple, bound
        # onto this bare stub.
        cp_raw_frame_padding = ArtifixerPipelineBase.cp_raw_frame_padding
        latent_frame_multiple = ArtifixerKvCachePipeline.latent_frame_multiple
        stages_conditioning_per_chunk = ArtifixerKvCachePipeline.stages_conditioning_per_chunk

        def denoise_to_latents(self, **kwargs):
            calls.append("denoise")
            captured.update(kwargs)
            return torch.zeros(1)

        def decode_latents_to_video(self, latents):
            calls.append("decode")
            return torch.rand(1, pixel_frames, 3, hw, hw)

        def clear_inference_caches(self):
            calls.append("clear")

    pipe = _FakeWanKvPipe()
    item = _eval_item(latent_frames, hw, cosmos3=False)
    args = _eval_args("Wan-AI/Wan2.1-T2V-14B-Diffusers", save_frame_outputs_only=True)

    run_inference.process_item(pipe, item, args, tmp_path, rank=0, device=torch.device("cpu"), vae_temporal_scale=4)

    assert calls == ["denoise", "decode"]
    expected_names = [f"{index:05d}.png" for index in range(pixel_frames)]
    for kind in ("pred", "gt", "rendered"):
        saved = sorted(path.name for path in (tmp_path / "scene" / "frames" / "batch_0000" / kind).glob("*.png"))
        assert saved == expected_names, kind

    # Non-saving CP ranks still clear the Wan caches without decoding.
    calls.clear()
    run_inference.process_item(
        pipe,
        _eval_item(latent_frames, hw, cosmos3=False),
        args,
        tmp_path / "cp",
        rank=1,
        device=torch.device("cpu"),
        vae_temporal_scale=4,
        save_outputs=False,
    )
    assert calls == ["denoise", "clear"]


@pytest.mark.parametrize("guidance_scale", [None, 1.0, 3.0])
def test_bidirectional_process_item_forwards_guidance(tmp_path, guidance_scale):
    class CapturePipe:
        ar_frames_per_chunk = 1
        stages_conditioning_per_chunk = False
        transformer = SimpleNamespace(_cp_world_size=1)
        cp_raw_frame_padding = ArtifixerPipelineBase.cp_raw_frame_padding
        latent_frame_multiple = ArtifixerKvCachePipeline.latent_frame_multiple

        def forward_inference(self, **kwargs):
            self.kwargs = kwargs
            self.noise = torch.randn(4)

    pipe = CapturePipe()
    args = _eval_args(COSMOS3_NANO_MODEL_ID, save_frame_outputs_only=True)
    args.inference_pipeline = "bidirectional"
    args.text_guidance_scale = guidance_scale
    item = _eval_item(2, 32, cosmos3=True)
    noises = []
    with torch.random.fork_rng(devices=[]):
        for _ in range(2):
            run_inference.process_item(
                pipe,
                item,
                args,
                tmp_path,
                rank=0,
                device=torch.device("cpu"),
                vae_temporal_scale=4,
                save_outputs=False,
            )
            noises.append(pipe.noise)
            torch.rand(7)

    assert torch.equal(*noises)
    if guidance_scale is None:
        assert "text_guidance_scale" not in pipe.kwargs
    else:
        assert pipe.kwargs["text_guidance_scale"] == guidance_scale
    assert "negative_prompt" in pipe.kwargs


def test_process_item_stages_kv_cache_conditioning_on_the_cpu(tmp_path):
    """A pipeline that consumes the conditioning per chunk (both KV-cache
    rollouts) receives the full clip on the CPU; the rest of the payload, and
    every input of a pipeline that takes the whole clip in one pass, is
    staged on the device. The callee side of the contract is pinned in the
    rollout tests (test_cosmos3_ar_rollout.py, test_wan_rollout_cache.py)."""
    assert ArtifixerKvCachePipeline.stages_conditioning_per_chunk
    assert ArtifixerCosmos3KvCachePipeline.stages_conditioning_per_chunk
    assert not ArtifixerPipelineBase.stages_conditioning_per_chunk

    class _CapturePipe:
        ar_frames_per_chunk = 1
        transformer = SimpleNamespace(_cp_world_size=1)
        cp_raw_frame_padding = ArtifixerPipelineBase.cp_raw_frame_padding
        latent_frame_multiple = ArtifixerKvCachePipeline.latent_frame_multiple

        def __init__(self, stages_conditioning_per_chunk: bool):
            self.stages_conditioning_per_chunk = stages_conditioning_per_chunk

        def denoise_to_latents(self, **kwargs):
            self.captured = kwargs

        forward_inference = denoise_to_latents

        def clear_inference_caches(self):
            pass

    # The meta device stands in for the accelerator: distinguishable from the CPU, no CUDA needed.
    device = torch.device("meta")
    staged = ("rendered_rgb", "rendered_opacity", "camera_rays", "w2cs", "Ks")
    for pipeline, staging in (("kv_cache", "cpu"), ("bidirectional", "meta")):
        pipe = _CapturePipe(stages_conditioning_per_chunk=(pipeline == "kv_cache"))
        args = _eval_args("Wan-AI/Wan2.1-T2V-14B-Diffusers", save_frame_outputs_only=True)
        args.inference_pipeline = pipeline
        run_inference.process_item(
            pipe,
            _eval_item(2, 32, cosmos3=False),
            args,
            tmp_path,
            rank=0,
            device=device,
            vae_temporal_scale=4,
            save_outputs=False,
        )
        assert {name: pipe.captured[name].device.type for name in staged} == dict.fromkeys(staged, staging), pipeline
        for name in ("neighbors", "neighbor_w2cs", "neighbor_Ks", "prompt"):
            assert pipe.captured[name].device.type == "meta", (pipeline, name)


# ---------------------------------------------------------------------------
# CP-eval setup: the pre-mesh checks of process_items_with_context_parallel
# ---------------------------------------------------------------------------


class _ReachedMeshSetup(Exception):
    """Raised by the stubbed mesh factory: every pre-mesh CP check passed."""


class _CpEvalPipe:
    """Minimal eval pipe for the CP-setup seam: a REAL reduced-config
    cosmos3 transformer (4 query / 2 KV heads) behind the attributes the
    pre-mesh checks read, with the base pipeline's CP methods bound below
    (a bare stub, not an ArtifixerPipelineBase subclass): frame rule plus
    the head rule."""

    # The base pipeline's CP and padding methods, verbatim.
    cp_head_divisor = ArtifixerPipelineBase.cp_head_divisor
    cp_raw_frame_padding = ArtifixerPipelineBase.cp_raw_frame_padding
    validate_cp_size = ArtifixerPipelineBase.validate_cp_size
    latent_frame_multiple = ArtifixerPipelineBase.latent_frame_multiple

    def __init__(self):
        self.vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4))
        self.transformer = _reduced_real_transformer()

    def _cp_frame_divisor(self, total_latent_frames: int) -> int:
        # The standard-pipeline rule (ArtifixerPipeline): every latent frame
        # count shards, so the frame assert never masks the head assert in
        # these tests.
        return total_latent_frames


def _cp_eval_setup(context_parallel_size: int):
    pipe = _CpEvalPipe()
    # first_item_num_frames reads only item 0's frame count: 17 pixel frames
    # = 5 latent frames, which the CP padding grows to a multiple of the CP
    # group size before the divisibility checks run.
    dataset = [dict(rgb_rendered=torch.zeros(17, 3, 8, 8))]
    args = SimpleNamespace(context_parallel_size=context_parallel_size)
    return pipe, dataset, args


def test_cp_eval_rejects_a_size_that_splits_a_kv_head():
    """The eval entry point's own head guard: context_parallel_size=4 divides
    the reduced cosmos3 transformer's 4 query heads but NOT its 2 KV heads
    (real GQA), so process_items_with_context_parallel refuses the run with
    the named head-count constraint before any process-group work — frame
    divisibility alone (8 padded latent frames) would have admitted it."""
    pipe, dataset, args = _cp_eval_setup(context_parallel_size=4)

    with pytest.raises(
        AssertionError,
        match=r"must divide the transformer's attention head divisor 2",
    ):
        run_inference.process_items_with_context_parallel(
            pipe, dataset, args, output_dir=None, rank=0, world_size=4, device=torch.device("cpu")
        )


def test_cp_eval_admits_a_size_dividing_both_head_counts(monkeypatch):
    """context_parallel_size=2 divides both of the reduced transformer's head
    counts (4 query, 2 KV), so the same entry point walks past the frame and
    head asserts and reaches mesh construction — pinned by a sentinel in
    place of the mesh factory, the first step that needs a real process
    group."""
    pipe, dataset, args = _cp_eval_setup(context_parallel_size=2)

    def _sentinel_meshes(*call_args, **call_kwargs):
        raise _ReachedMeshSetup

    monkeypatch.setattr(run_inference, "create_context_parallel_meshes", _sentinel_meshes)

    with pytest.raises(_ReachedMeshSetup):
        run_inference.process_items_with_context_parallel(
            pipe, dataset, args, output_dir=None, rank=0, world_size=2, device=torch.device("cpu")
        )


def test_cp_eval_feeds_padded_sharding_pipelines_the_same_raw_frames(monkeypatch, tmp_path):
    """Eval must not raw-pad a clip for CP when the pipeline's transformer
    pads its own frame axis (cosmos3's cp_raw_frame_padding is 0): the
    production 21-latent-frame clip reaches the VAE with the SAME raw frame
    count at CP=2 as at CP=1, so eval and validation compute the same
    numbers for the same clip. A pipeline on the base rule keeps the raw CP
    padding — its transformer needs the latent count to divide the CP
    size."""
    latent_frames, hw = 21, 32
    pixel_frames = (latent_frames - 1) * 4 + 1  # 81, the production clip

    def vae_frames(pipe, item, cp_world_size: int) -> tuple[int, int]:
        pipe.transformer._cp_world_size = cp_world_size
        captured: dict = {}

        def fake_denoise(**kwargs):
            captured.update(kwargs)
            return torch.zeros(1)

        monkeypatch.setattr(pipe, "denoise_to_latents", fake_denoise, raising=False)
        monkeypatch.setattr(pipe, "clear_inference_caches", lambda: None, raising=False)
        args = _eval_args(COSMOS3_NANO_MODEL_ID, save_frame_outputs_only=False)
        run_inference.process_item(
            pipe,
            item,
            args,
            tmp_path,
            rank=1,
            device=torch.device("cpu"),
            vae_temporal_scale=4,
            save_outputs=False,
        )
        return captured["rendered_rgb"].shape[1], captured["w2cs"].shape[1]

    padded_pipe = ArtifixerCosmos3KvCachePipeline(
        vae=_TinyCosmosVae(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
        ar_frames_per_chunk=7,  # 21 latent frames need no chunk padding
        local_attn_size=21,
        sink_size=7,
    )
    assert padded_pipe.cp_raw_frame_padding(pixel_frames, 2, 4) == 0, "cosmos3 pipes take no raw CP padding"
    cp1_frames, cp1_latents = vae_frames(padded_pipe, _eval_item(latent_frames, hw, cosmos3=True), cp_world_size=1)
    cp2_frames, cp2_latents = vae_frames(padded_pipe, _eval_item(latent_frames, hw, cosmos3=True), cp_world_size=2)
    assert cp1_frames == cp2_frames == pixel_frames, "CP must not change the raw frames fed to the VAE"
    assert cp1_latents == cp2_latents == latent_frames

    class _RawPaddingKvPipe:
        """A pipeline without the internal-padding path (the Wan shape):
        the base cp_raw_frame_padding rule, bound onto a bare stub."""

        ar_frames_per_chunk = 1
        transformer = SimpleNamespace(_cp_world_size=1)
        cp_raw_frame_padding = ArtifixerPipelineBase.cp_raw_frame_padding
        latent_frame_multiple = ArtifixerKvCachePipeline.latent_frame_multiple
        stages_conditioning_per_chunk = ArtifixerKvCachePipeline.stages_conditioning_per_chunk

        def clear_inference_caches(self):
            pass

    raw_pipe = _RawPaddingKvPipe()
    cp1_frames, _ = vae_frames(raw_pipe, _eval_item(latent_frames, hw, cosmos3=False), cp_world_size=1)
    cp2_frames, cp2_latents = vae_frames(raw_pipe, _eval_item(latent_frames, hw, cosmos3=False), cp_world_size=2)
    assert cp1_frames == pixel_frames
    assert cp2_frames == pixel_frames + 4 and cp2_latents == 22, "pipelines without the path keep the raw CP pad"


class _PaddedEvalPipe(_CpEvalPipe):
    """The cosmos3 dense CP methods on the eval stub: the transformer pads
    its own frame axis, so only the head rule applies."""

    cp_raw_frame_padding = ArtifixerCosmos3Pipeline.cp_raw_frame_padding
    validate_cp_size = ArtifixerCosmos3Pipeline.validate_cp_size


class _PaddedKvEvalPipe(_CpEvalPipe):
    """The cosmos3 KV-cache CP methods on the eval stub: 7-frame chunks
    shard frame-first; a chunk that does not divide the CP size is padded
    and splits heads."""

    ar_frames_per_chunk = 7
    cp_raw_frame_padding = ArtifixerCosmos3Pipeline.cp_raw_frame_padding
    validate_cp_size = ArtifixerCosmos3KvCachePipeline.validate_cp_size
    latent_frame_multiple = ArtifixerCosmos3KvCachePipeline.latent_frame_multiple


def test_cp_eval_frame_assert_relaxes_only_for_padded_sharding_pipelines(monkeypatch):
    """A CP size that does not divide the frame divisor is admitted when the
    pipeline's transformer pads its own frame axis, and the head rule still
    applies to padded KV-cache chunks: chunks that do not divide the CP size
    are split by heads, so a CP size that splits a KV head is refused even
    though the frame-sharded cache path has no head rule."""
    _, dataset, args = _cp_eval_setup(context_parallel_size=2)
    pipe = _PaddedEvalPipe()
    # 2 never divides 7; the cosmos3 rules must not read the frame divisor.
    pipe._cp_frame_divisor = lambda total_latent_frames: 7

    def _sentinel_meshes(*call_args, **call_kwargs):
        raise _ReachedMeshSetup

    monkeypatch.setattr(run_inference, "create_context_parallel_meshes", _sentinel_meshes)

    with pytest.raises(_ReachedMeshSetup):
        run_inference.process_items_with_context_parallel(
            pipe, dataset, args, output_dir=None, rank=0, world_size=2, device=torch.device("cpu")
        )

    # Same geometry on the KV-cache rules at CP=4: the frame-sharded rule
    # alone would have no head constraint, but the padded chunk splits
    # heads and 4 does not divide the 2 KV heads.
    _, dataset, args = _cp_eval_setup(context_parallel_size=4)
    kv_pipe = _PaddedKvEvalPipe()

    with pytest.raises(AssertionError, match=r"attention head divisor 2"):
        run_inference.process_items_with_context_parallel(
            kv_pipe, dataset, args, output_dir=None, rank=0, world_size=4, device=torch.device("cpu")
        )

    # Without the internal-padding path the frame rule still refuses outright.
    strict_pipe, dataset, args = _cp_eval_setup(context_parallel_size=2)
    strict_pipe._cp_frame_divisor = lambda total_latent_frames: 7

    with pytest.raises(AssertionError, match=r"cannot pad its own frame axis"):
        run_inference.process_items_with_context_parallel(
            strict_pipe, dataset, args, output_dir=None, rank=0, world_size=2, device=torch.device("cpu")
        )


def test_get_eval_pipe_loads_each_checkpoint_format_in_the_order_it_needs(monkeypatch, tmp_path):
    """A .pt export is built with the transformer on the meta device, which
    cannot be moved: the load materializes the parameters on the device first
    and the cast/move that places the rest follows. A DCP checkpoint loads
    into the already cast and placed model. Either way the caller gets back a
    loaded pipe on the device."""
    from model_training.utils import train_utils

    _patch_pipeline_factories(monkeypatch)
    events: list = []
    capture_pipe = train_utils.ArtifixerKvCachePipeline

    class _RecordingPipe(capture_pipe):
        def to(self, *args, **kwargs):
            events.append(("to", args))
            return self

    monkeypatch.setattr(train_utils, "ArtifixerKvCachePipeline", _RecordingPipe)
    monkeypatch.setattr(
        run_inference,
        "load_transformer_checkpoint",
        lambda transformer, args, **placement: events.append(("load", placement)),
    )
    common = _BASE_ARGV[2:] + ["--model_id", WAN22_TI2V_5B_MODEL_ID, "--inference_pipeline", "kv_cache"]
    placed = [("to", (torch.bfloat16,)), ("to", (torch.device("cpu"),))]
    load = ("load", {"device": torch.device("cpu"), "dtype": torch.bfloat16})

    pipe = run_inference.get_eval_pipe(
        inference_args.parse_args(_stamped_checkpoint(tmp_path / "model.pt", "mean") + common), torch.device("cpu")
    )
    assert isinstance(pipe, _RecordingPipe)
    assert events == [load] + placed

    events.clear()
    run_inference.get_eval_pipe(
        inference_args.parse_args(_stamped_checkpoint(tmp_path / "slot", "mean") + common),
        torch.device("cpu"),
    )
    assert events[:3] == placed + [load]


def _tiny_wan_vae():
    """A real AutoencoderKLWan at toy width: the tiling flag under test is
    diffusers' own, read by its encode/decode at call time."""
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan(
        base_dim=4,
        z_dim=2,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
    )


@pytest.mark.parametrize("model_id", [WAN22_TI2V_5B_MODEL_ID, COSMOS3_NANO_MODEL_ID])
def test_vae_tiling_flag_arms_tiling_on_the_built_vae(monkeypatch, model_id, tmp_path):
    """--vae_tiling is applied inside get_eval_pipe, on the VAE it built, so
    the pipe every caller (and every rank) receives already tiles its
    encode/decode; without the flag the VAE keeps diffusers' untiled default,
    which is what keeps the default eval numbers bit-reproducible."""
    from model_training.utils import train_utils

    _patch_pipeline_factories(monkeypatch)
    monkeypatch.setattr(train_utils.AutoencoderKLWan, "from_pretrained", lambda *args, **kwargs: _tiny_wan_vae())
    common = ["--model_id", model_id, "--inference_pipeline", "kv_cache"]
    checkpoint_argv = _stamped_checkpoint(tmp_path, "last" if model_id == COSMOS3_NANO_MODEL_ID else "mean")
    tiled_args = _parse_args(common + ["--vae_tiling"], checkpoint_argv)
    untiled_args = _parse_args(common, checkpoint_argv)

    tiled = run_inference.get_eval_pipe(tiled_args, torch.device("cpu"))
    untiled = run_inference.get_eval_pipe(untiled_args, torch.device("cpu"))

    assert tiled.vae.use_tiling is True
    assert untiled.vae.use_tiling is False
