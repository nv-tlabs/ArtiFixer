# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
import shutil
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import h5py
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch import nn

import model_eval.validation as validation_module
from model_eval.validation import EvaluationRunner
from model_eval.validation_config import EvaluationConfig, EvaluationIdentity
from model_eval.validation_reporting import EvaluationResult
from model_eval.validation_scoring import EvaluationPass, ValidationRecorder, gather_task_dict
from model_eval.validation_tasks import build_evaluation_plan, build_test_panel_plan, build_validation_task_specs
from model_training import stages
from model_training.data.dl3dv_test import DL3DVPairedDatasetTest
from model_training.pipeline.cosmos3_kv_cache_pipeline import ArtifixerCosmos3KvCachePipeline
from model_training.trainers.dmd import DMD
from model_training.trainers.trainer import Trainer
from model_training.trainers.trainer_base import TrainerBase
from model_training.utils import train_utils
from model_training.utils.validation_artifacts import (
    VALIDATION_SCHEMA_VERSION,
    content_id,
    file_sha256,
    pass_report,
    validate_manifest,
    write_json,
)
from tests._helpers import _small_dmd_ema_trainer
from tests.test_validation_tasks import _panel_runner, _runner, _scoring_task


def _manifest(samples, camera_pose_alignment="mean"):
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "noise_seed": 42,
        "num_frames": len(samples[0]["frame_indices"]),
        "num_neighbors": 3,
        "dataset_scaling_factor": 0.01,
        "camera_pose_alignment": camera_pose_alignment,
        "samples": [{**sample, "negative_conditioning": sample.get("negative_conditioning")} for sample in samples],
    }


def _source_identity():
    return {
        "geometry_sha256": "0" * 64,
        "scene_archive_sha256": "1" * 64,
        "reconstructions": {"3": {"metric_scale": 1.0, "sha256": "2" * 64}},
    }


@pytest.mark.requires_host_ffmpeg
@pytest.mark.parametrize("pose_alignment", ["mean", "last"])
def test_manifest_replays_real_dataset_frames_neighbors_and_prompt(tmp_path, pose_alignment):
    from tests.test_pose_jump_window_filtering import _dataset_kwargs, _entry, _write_scene, _write_split

    _write_scene(
        tmp_path,
        "scene",
        x_positions=[0.1 * i for i in range(9)],
        prompt_windows={"w0": [0, 1, 2, 3, 4], "w1": [4, 5, 6, 7, 8]},
        splits=(3,),
    )
    split_path = _write_split(tmp_path, {"scene": [_entry(tmp_path, "scene", "first", 1.0, 3)]})
    split = json.loads(split_path.read_text())
    split["test"] = split["trainval"]
    write_json(split_path, split)
    kwargs = _dataset_kwargs(tmp_path, split_path) | {"camera_pose_alignment": pose_alignment}
    original = DL3DVPairedDatasetTest(num_views=3, capture_validation_metadata=True, **kwargs)[0]
    torch.testing.assert_close(original["w2cs"][0], torch.eye(4))
    sample = json.loads(original["validation_sample_json"])
    manifest = _manifest([sample], pose_alignment)
    validate_manifest(manifest)
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    args = (
        importlib.import_module("model_training.train")
        .build_parser()
        .parse_args(
            [
                "--project_dir",
                str(tmp_path / "eval"),
                "--split_path",
                str(split_path),
                "--dl3dv_dir",
                str(kwargs["dl3dv_dir"]),
                "--prompt_dir",
                str(kwargs["prompt_dir"]),
                "--camera_pose_alignment",
                pose_alignment,
                "--validation_manifest",
                str(manifest_path),
            ]
        )
    )
    replay = train_utils.get_val_datasets(args, SimpleNamespace(is_main_process=False), None, None)[3]
    assert args.test_panel_scenes == 1
    replay.validation_seed = 12345
    replayed = replay[0]
    for key in ("rgb_gt", "rgb_neighbors", "rgb_rendered", "w2cs", "neighbor_w2cs", "camera_rays", "encoded_prompt"):
        assert torch.equal(original[key], replayed[key]), key
    assert json.loads(replayed["validation_sample_json"]) == sample
    replay.validation_manifest["samples"][0]["resolution"] = [999, 999]
    with pytest.raises(ValueError, match="resolution mismatch"):
        replay[0]


@pytest.fixture
def identity_panel(tmp_path):
    from tests.test_pose_jump_window_filtering import _dataset_kwargs, _entry, _write_scene, _write_split

    root = tmp_path / "source"
    root.mkdir()
    views = (3, 6, 12)
    _write_scene(
        root, "scene", x_positions=[0.1 * i for i in range(13)], prompt_windows={"w0": list(range(5))}, splits=views
    )
    split = _write_split(root, {"scene": [_entry(root, "scene", "first", 1.0, n) for n in views]})
    kwargs = _dataset_kwargs(root, split)
    original = DL3DVPairedDatasetTest(num_views=3, capture_validation_metadata=True, **kwargs)[0]
    manifest = _manifest([json.loads(original["validation_sample_json"])])
    return root, kwargs, manifest, original


@pytest.mark.requires_host_ffmpeg
@pytest.mark.parametrize(
    "change,identity_key",
    [
        ("scale3", "reconstructions"),
        ("scale6", "reconstructions"),
        ("reconstruction", "reconstructions"),
        ("geometry", "geometry_sha256"),
        ("image", "scene_archive_sha256"),
    ],
)
def test_manifest_rejects_changed_camera_scale_and_source_assets(identity_panel, change, identity_key):
    root, kwargs, manifest, _ = identity_panel
    if change.startswith("scale"):
        split = kwargs["split_path"]
        contents = json.loads(split.read_text())
        for entry in contents["trainval"]["scene"]:
            if entry["num_selected_indices"] == int(change.removeprefix("scale")):
                entry["scale"] = 2.0
        split.write_text(json.dumps(contents))
    elif change == "reconstruction":
        with h5py.File(root / "recons/scene/data_scene_first_3.h5", "a") as recon:
            recon["selected_indices"][:] = [1, 2, 3]
    else:
        archive_path = root / "dl3dv/960P/scene.zip"
        with ZipFile(archive_path) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        if change == "geometry":
            transforms = json.loads(entries["scene/transforms.json"])
            transforms["frames"][0]["transform_matrix"][0][3] = 1.0
            entries["scene/transforms.json"] = json.dumps(transforms).encode()
        else:
            buffer = BytesIO()
            Image.new("RGB", (16, 16), (210, 210, 210)).save(buffer, format="PNG")
            entries["scene/images_4/frame_00000.png"] = buffer.getvalue()
        with ZipFile(archive_path, "w") as archive:
            for name, data in entries.items():
                archive.writestr(name, data)

    replay = DL3DVPairedDatasetTest(num_views=3, validation_manifest=manifest, **kwargs)
    with pytest.raises(ValueError, match=f"source_identity.{identity_key} mismatch"):
        replay[0]
    fresh = DL3DVPairedDatasetTest(num_views=3, capture_validation_metadata=True, **kwargs)[0]
    new_manifest = _manifest([json.loads(fresh["validation_sample_json"])])
    assert content_id(new_manifest) != content_id(manifest)


@pytest.mark.requires_host_ffmpeg
@pytest.mark.parametrize("num_views", [6, 12])
def test_manifest_checks_sources_for_regular_validation_variants(identity_panel, num_views):
    _, kwargs, manifest, _ = identity_panel
    replay = DL3DVPairedDatasetTest(num_views=num_views, validation_manifest=manifest, **kwargs)[0]
    assert replay["rgb_neighbors"].shape[0] == num_views
    assert json.loads(replay["validation_sample_json"])["source_identity"] == manifest["samples"][0]["source_identity"]


@pytest.mark.requires_host_ffmpeg
def test_source_identity_allows_dataset_relocation(identity_panel, tmp_path):
    from tests.test_pose_jump_window_filtering import _dataset_kwargs

    root, _, manifest, original = identity_panel
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    split = relocated / "split.json"
    contents = json.loads(split.read_text())
    for scenes in contents.values():
        for entries in scenes.values():
            for entry in entries:
                entry["hdf5_path"] = str(relocated / Path(entry["hdf5_path"]).relative_to(root))
    split.write_text(json.dumps(contents))
    replay = DL3DVPairedDatasetTest(num_views=3, validation_manifest=manifest, **_dataset_kwargs(relocated, split))[0]
    for key in ("rgb_gt", "rgb_rendered", "w2cs", "rgb_neighbors"):
        assert torch.equal(original[key], replay[key])
    assert json.loads(replay["validation_sample_json"]) == {
        key: value for key, value in manifest["samples"][0].items() if key != "negative_conditioning"
    }


def test_source_fingerprint_cache_detects_preserved_mtime_rewrites(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"before")
    stat = path.stat()
    before = file_sha256(path)
    path.write_bytes(b"after!")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert file_sha256(path) != before


def test_legacy_manifests_and_results_require_regeneration():
    from model_training.compare_validation_results import compare_results

    with pytest.raises(ValueError, match="version 3 is required"):
        validate_manifest({"schema_version": 1})
    legacy = {"schema_version": 1, "status": "complete", "panel": {"tasks": {}}}
    with pytest.raises(ValueError, match="version-3"):
        compare_results(legacy, legacy)


@pytest.mark.parametrize("validate_only,override,expected", [(False, None, None), (True, None, 125), (False, 253, 253)])
def test_dmd_validation_length_override_preserves_training_default(monkeypatch, validate_only, override, expected):
    captured = []
    monkeypatch.setattr(
        train_utils, "DL3DVPairedDatasetTest", lambda **kwargs: captured.append(kwargs) or SimpleNamespace()
    )
    args = SimpleNamespace(
        split_path=None,
        dl3dv_dir=None,
        prompt_dir=None,
        num_frames=125,
        dataset_scaling_factor=0.01,
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        camera_pose_alignment="mean",
        validate_only=validate_only,
        validation_num_frames=override,
    )
    train_utils.get_val_datasets(args, SimpleNamespace(is_main_process=False), None, 8)
    assert captured[0]["num_frames"] == expected


@pytest.mark.parametrize("module_name", ["train", "diffusion_forcing", "distillation"])
@pytest.mark.parametrize("use_manifest", [False, True])
@pytest.mark.parametrize("pose_alignment", ["mean", "last"])
def test_validate_only_entrypoints_do_not_build_training_objects(
    monkeypatch, tmp_path, module_name, use_manifest, pose_alignment
):
    module = importlib.import_module("model_training." + module_name)
    checkpoint = tmp_path / "weights"
    train_utils.write_checkpoint_metadata(checkpoint, {"camera_pose_alignment": pose_alignment})
    cli_args = [
        "--validate_only",
        "--base_checkpoint_dir",
        str(checkpoint),
        "--project_dir",
        str(tmp_path),
        "--split_path",
        "split.json",
        "--dl3dv_dir",
        "data",
        "--prompt_dir",
        "prompts",
        "--model_id",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "--num_frames",
        "125",
        "--log_with",
        "none",
    ]
    if module_name != "train":
        cli_args += ["--frames_per_block", "8"]
    if use_manifest:
        manifest = _manifest(
            [
                {
                    "scene_id": "scene",
                    "half": "first",
                    "prompt_file": "frames_125_stride_1.h5",
                    "prompt_window": "w0",
                    "prompt_sha256": "caption",
                    "source_identity": _source_identity(),
                    "resolution": [16, 16],
                    "frame_indices": list(range(125)),
                    "neighbor_indices": [0, 1, 2],
                }
            ],
            pose_alignment,
        )
        write_json(tmp_path / "manifest.json", manifest)
        # 81 frames would violate the block layout; resolve the manifest before checking it.
        cli_args += ["--num_frames", "81", "--validation_manifest", str(tmp_path / "manifest.json")]
    args = module.build_parser().parse_args(cli_args)
    pipe = SimpleNamespace(
        transformer=nn.Linear(1, 1),
        vae=SimpleNamespace(device=torch.device("cpu"), config=SimpleNamespace(scale_factor_temporal=4)),
        validate_cp_size=lambda *a: None,
    )
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), is_fsdp2=True, _prepare_fsdp2=lambda model: (model,), print=print
    )
    loaded, validated = [], []

    def forbidden(*a, **kw):
        raise AssertionError("Evaluation must not construct training objects")

    def get_accelerator(resolved_args, run_id):
        assert resolved_args.camera_pose_alignment == pose_alignment
        return accelerator

    for target in (train_utils, stages):
        for name, replacement in {
            "get_accelerator": get_accelerator,
            "get_pipe": lambda *a, **kw: pipe,
            "get_kv_cache_pipe": lambda *a, **kw: pipe,
            "get_train_dataloader": forbidden,
            "build_score_transformer": forbidden,
            "build_ema_transformer": forbidden,
        }.items():
            if hasattr(target, name):
                monkeypatch.setattr(target, name, replacement)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(train_utils, "load_model_weights_from_dcp", lambda model, path: loaded.append(path))
    monkeypatch.setattr(train_utils, "get_val_datasets", lambda *a: {3: object()})
    for trainer_class in (TrainerBase, Trainer, DMD):
        monkeypatch.setattr(trainer_class, "__init__", forbidden)
    monkeypatch.setattr(
        EvaluationRunner,
        "run",
        lambda self, plan, output, identity: validated.append(type(self).__name__)
        or EvaluationResult({}, None, output),
    )
    module.main(args)
    assert args.num_frames == 125
    assert loaded == [checkpoint]
    assert len(validated) == 1
    assert not pipe.transformer.weight.requires_grad


def test_dmd_training_still_requires_a_critic_checkpoint():
    from model_training.distillation import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--base_checkpoint_dir",
                "weights",
                "--project_dir",
                "run",
                "--split_path",
                "split.json",
                "--dl3dv_dir",
                "data",
                "--prompt_dir",
                "prompts",
            ]
        )


def _cosmos3_decoder():
    """The real pipeline decode methods, with only the expensive VAE replaced."""

    class Vae(nn.Module):
        dtype = torch.float32
        device = torch.device("cpu")
        config = SimpleNamespace(scale_factor_temporal=4)

        def __init__(self):
            super().__init__()
            self.decode_lengths = []

        def decode(self, latents, return_dict=False):
            self.decode_lengths.append(latents.shape[2])
            return (torch.cat((latents[:, :, :1], latents[:, :, 1:].repeat_interleave(4, dim=2)), dim=2),)

    pipe = ArtifixerCosmos3KvCachePipeline.__new__(ArtifixerCosmos3KvCachePipeline)
    nn.Module.__init__(pipe)
    pipe.vae = Vae()
    pipe.latents_mean = 0.0
    pipe.latents_inv_std = 1.0
    pipe.video_processor = SimpleNamespace(postprocess_video=lambda video, output_type: video.permute(0, 2, 1, 3, 4))
    return pipe


@pytest.mark.parametrize("chunk_size", [None, 2, 4, 9, 16])
def test_cosmos3_decode_keeps_frame_order_and_bounds_chunk_memory(chunk_size):
    pipe = _cosmos3_decoder()
    latents = torch.arange(9.0).view(1, 1, 9, 1, 1)
    output = pipe.decode_latents_to_video(latents, temporal_chunk_size=chunk_size)
    expected = torch.cat((torch.tensor([0.0]), torch.arange(1.0, 9.0).repeat_interleave(4)))
    assert torch.equal(output.flatten(), expected)
    assert output.shape == (1, 33, 1, 1, 1)
    if chunk_size is not None and chunk_size < 9:
        assert len(pipe.vae.decode_lengths) > 1
        assert max(pipe.vae.decode_lengths) <= chunk_size + 1  # one latent of halo
    else:
        assert pipe.vae.decode_lengths == [9]


def _dmd_panel(monkeypatch, tmp_path, *, fail=False):
    transformer = nn.Linear(1, 1, bias=False)
    transformer.weight.data.fill_(2)
    transformer.cp_mesh = SimpleNamespace(size=lambda: 4)
    transformer._scope_accumulator = None
    trainer = _small_dmd_ema_trainer(tmp_path / "dmd", model=transformer)
    trainer.accelerator.step = 200
    trainer.args.model_id = "nvidia/Cosmos3-Nano"
    trainer.args.camera_pose_alignment = "last"
    trainer.args.test_panel_scenes = 2
    trainer.val_datasets = _panel_runner(num_scenes=2).datasets
    trainer.evaluation.config = EvaluationConfig.from_args(trainer.args)
    trainer.pipe.stages_conditioning_per_chunk = True
    for parameter in trainer.ema.parameters():
        parameter.data.fill_(4)
    called = []

    def build_task(pipe, config, context, datasets, spec, sample_index, **kwargs):
        task = _scoring_task()
        task.key = spec.key
        task.has_unmodified_rendered_baseline = spec.has_unmodified_rendered_baseline
        return task

    def denoise(**kwargs):
        assert transformer.weight.item() == 4
        assert not transformer.training
        assert not torch.is_grad_enabled()
        assert transformer._scope_accumulator is None
        assert "vae_chunk_halo_latent_frames" not in kwargs
        assert kwargs["prompt_attention_mask"] is not None
        called.append(kwargs["noise_seed"])
        if fail:
            raise RuntimeError("rollout failed")
        return torch.zeros(1, 3, 9, 16, 16)

    monkeypatch.setattr(validation_module, "build_validation_task", build_task)
    trainer.pipe.denoise_to_latents = denoise
    decoder = _cosmos3_decoder()
    trainer.pipe.vae = decoder.vae
    trainer.pipe.decode_latents_to_video = decoder.decode_latents_to_video
    return trainer, called


def test_dmd_cp_panel_uses_ema_for_every_scene_and_restores(monkeypatch, tmp_path):
    trainer, called = _dmd_panel(monkeypatch, tmp_path)
    plan = build_test_panel_plan(trainer.val_datasets, 2)
    with trainer.validation_context():
        result = trainer.evaluation.run_pass(trainer.val_datasets, plan, None, weights="ema")
    assert len(result.metrics) == len(called) == 4
    assert len(set(called)) == 4
    assert trainer.pipe.transformer.weight.item() == 2
    assert trainer.pipe.transformer.training
    assert not trainer.ema.is_cached
    assert not list(tmp_path.glob("*.mp4"))


def test_dmd_panel_restores_ema_and_mode_after_failure(monkeypatch, tmp_path):
    trainer, _ = _dmd_panel(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="rollout failed"), trainer.validation_context():
        trainer.evaluation.run_pass(
            trainer.val_datasets, build_test_panel_plan(trainer.val_datasets, 2), None, weights="ema"
        )
    assert trainer.pipe.transformer.weight.item() == 2
    assert trainer.pipe.transformer.training
    assert not trainer.ema.is_cached


def test_dmd_cp_regular_validation_uses_real_cosmos3_decoder(monkeypatch, tmp_path):
    trainer, called = _dmd_panel(monkeypatch, tmp_path)
    saved = []
    monkeypatch.setattr(
        ValidationRecorder, "save", lambda self, output, task, path, **kwargs: saved.append((output.shape, path))
    )
    plan = [(build_validation_task_specs(trainer.val_datasets)[0], 0)]
    with trainer.validation_context():
        trainer.evaluation.run_pass(trainer.val_datasets, plan, tmp_path, weights="ema")
    assert len(called) == 1
    assert saved == [(torch.Size([1, 33, 3, 16, 16]), tmp_path)]
    assert trainer.pipe.transformer.weight.item() == 2


@pytest.mark.parametrize("fail", [False, True])
def test_validation_step_preserves_main_scope_logging(monkeypatch, tmp_path, fail):
    trainer = _small_dmd_ema_trainer(tmp_path)
    trainer.val_datasets = _panel_runner(num_scenes=1).datasets
    trainer.evaluation.config.test_panel_scenes = 1
    transformer = trainer.pipe.transformer
    injection = nn.Module()
    injection.alpha = nn.Parameter(torch.tensor(0.25))
    transformer.scope = nn.ModuleList([injection])
    transformer._scope_accumulator = None

    def sample(datasets, plan, save_dir, **kwargs):
        if save_dir is None:
            assert transformer._scope_accumulator is None
        else:
            transformer._scope_accumulator.extend([(2.0, 4.0), (1.0, 4.0)])
            if fail:
                raise RuntimeError("validation failed")
        return EvaluationPass({}, {})

    monkeypatch.setattr(trainer.evaluation, "run_pass", sample)
    monkeypatch.setattr(
        trainer.evaluation, "finish", lambda *a: EvaluationResult({"val/mean/psnr": 21.0}, None, tmp_path)
    )
    logged = {}
    if fail:
        with pytest.raises(RuntimeError, match="validation failed"):
            trainer._validation_step(logged)
    else:
        trainer._validation_step(logged)
        assert logged["scope/alpha_mean"] == 0.25
        assert logged["scope_to_query/mean"] == 0.375
        assert logged["scope_to_query/max"] == 0.5
        assert logged["val/mean/psnr"] == 21.0
    assert transformer._scope_accumulator is None
    assert transformer.training


def test_missing_and_nonfinite_metrics_are_incomplete():
    report = pass_report(["a", "b"], {"a": {"psnr": 20.0, "ssim": float("nan")}}, {"a": {}})
    assert report["status"] == "incomplete"
    assert report["expected_tasks"] == 2
    assert report["complete_tasks"] == 0
    assert report["metric_counts"] == {"psnr": 1}
    json.dumps(report, allow_nan=False)


def test_strict_validation_writes_failure_artifact_without_wandb(monkeypatch, tmp_path):
    trainer = _runner()
    trainer.config.log_with = "none"
    trainer.config.validation_strict = True
    trainer.config.test_panel_scenes = 0
    trainer.config.max_validation_tasks = 1
    trainer.val_datasets = {3: object()}
    trainer.get_step = lambda: 0
    with pytest.raises(RuntimeError, match="Incomplete validation"):
        trainer.finish(
            build_evaluation_plan(trainer.val_datasets, trainer.config, 0),
            EvaluationIdentity(tmp_path),
            EvaluationPass({}, {}),
            EvaluationPass({}, {}),
            tmp_path,
        )
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["status"] == "incomplete"
    assert report["validation"]["expected_tasks"] == 1
    assert report["validation"]["scored_tasks"] == 0


def test_metric_family_failure_is_recorded_and_other_scores_survive(monkeypatch):
    trainer = _runner()
    task = _scoring_task()

    def fail(*a, **kw):
        raise RuntimeError("broken pose track")

    monkeypatch.setattr("model_training.utils.validation_metrics.view_consistency_metrics", fail)
    trainer.recorder.record(task.rgb_gt[0], task.rgb_gt[0], task.rendered_rgb[0], task)
    record = trainer.recorder.records[task.key]
    assert record["revisit_status"] == "failed"
    assert "broken pose track" in record["errors"]["view_consistency"]
    assert "num_repeat_pairs" in record["missing_metrics"]
    assert trainer.recorder.metrics[task.key]["psnr"] == 120


def test_nonfinite_pixels_fail_even_when_clamping_would_hide_them():
    trainer = _runner()
    task = _scoring_task()
    generated = torch.full_like(task.rgb_gt[0], float("inf"))
    trainer.recorder.record(generated, task.rgb_gt[0], task.rendered_rgb[0], task)
    assert "inputs" in trainer.recorder.records[task.key]["errors"]


def test_manifest_identifier_is_independent_of_json_key_order(tmp_path):
    write_json(tmp_path / "manifest.json", {"a": 1, "b": 2})
    assert content_id(json.loads((tmp_path / "manifest.json").read_text())) == content_id({"b": 2, "a": 1})


@pytest.mark.parametrize("using_ema", [False, True])
def test_complete_report_exports_replayable_manifest(tmp_path, using_ema):
    trainer = _runner()
    trainer.config.test_panel_scenes = 1
    trainer.config.max_validation_tasks = 1
    trainer.config.dataset_scaling_factor = 0.01
    trainer.config.validation_strict = True
    trainer.val_datasets = _panel_runner(num_scenes=1, dataset_len=1).datasets
    trainer.recorder.weights = "ema" if using_ema else "raw"
    task = _scoring_task()
    task.sample_metadata = {
        "scene_id": "scene",
        "half": "first",
        "prompt_file": "frames_33_stride_1.h5",
        "prompt_window": "w0",
        "prompt_sha256": "fixed-caption",
        "source_identity": _source_identity(),
        "resolution": [16, 16],
        "frame_indices": list(range(33)),
        "neighbor_indices": [0, 1, 2],
    }
    for key in ("default_3", "default_3@scene00", "no_rendered_rgb_3@scene00"):
        task.key = key
        trainer.recorder.save(task.rgb_gt, task, None)
    trainer._panel_records = {key: value for key, value in trainer.recorder.records.items() if "@" in key}
    trainer.finish(
        build_evaluation_plan(trainer.val_datasets, trainer.config, 0),
        EvaluationIdentity(tmp_path, weights="ema" if using_ema else "raw"),
        EvaluationPass(trainer.recorder.metrics, trainer.recorder.records),
        EvaluationPass({key: trainer.recorder.metrics[key] for key in trainer._panel_records}, trainer._panel_records),
        tmp_path,
    )
    report = json.loads((tmp_path / "results.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    validate_manifest(manifest)
    assert manifest == _manifest([task.sample_metadata], trainer.config.camera_pose_alignment)
    assert report["status"] == "complete"
    from model_training.utils.validation_metrics import METRIC_PROTOCOL

    assert report["metric_protocol"] == METRIC_PROTOCOL
    assert report["weights"] == ("ema" if using_ema else "raw")
    assert report["panel"]["complete_tasks"] == 2
    assert report["manifest_id"] == content_id(manifest)


def _distributed_scoring_worker(rank, store_path, output_dir):
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=2)
    try:
        trainer = _runner()
        trainer.recorder.records = {}
        if rank == 1:
            task = _scoring_task()
            trainer.recorder.record(task.rgb_gt[0], task.rgb_gt[0], task.rendered_rgb[0], task)
        metrics = gather_task_dict(trainer.recorder.metrics)
        records = gather_task_dict(trainer.recorder.records)
        assert metrics["default_3"]["psnr"] == 120
        trainer.config.project_dir = Path(output_dir)
        trainer.config.test_panel_scenes = 0
        trainer.config.validation_strict = True
        trainer.config.max_validation_tasks = 2  # second task intentionally absent
        trainer.val_datasets = {3: object()}
        trainer.get_step = lambda: 0
        with pytest.raises(RuntimeError, match="Incomplete validation"):
            trainer.finish(
                build_evaluation_plan(trainer.val_datasets, trainer.config, 0),
                EvaluationIdentity(Path(output_dir)),
                EvaluationPass(metrics, records),
                EvaluationPass({}, {}),
                Path(output_dir),
            )
    finally:
        dist.destroy_process_group()


def test_task_gather_and_strict_failure_agree_on_two_cpu_ranks(tmp_path):
    mp.spawn(_distributed_scoring_worker, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2, join=True)
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["validation"]["complete_tasks"] == 1
    assert report["validation"]["expected_tasks"] == 2


def test_teacher_comparison_rejects_different_inputs_and_hardware_speedup():
    import copy

    from model_training.compare_validation_results import compare_results

    reference = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "complete",
        "checkpoint": "teacher",
        "manifest_id": "same-inputs",
        "device_name": "GPU-A",
        "world_size": 8,
        "sampling": {"validation_chunk_frames": 32},
        "panel": {
            "tasks": {"default_3@scene00": {"cp_size": 4, "metrics": {"psnr": 30.0, "generation_seconds": 20.0}}}
        },
    }
    candidate = copy.deepcopy(reference)
    candidate["checkpoint"] = "student"
    candidate["panel"]["tasks"]["default_3@scene00"]["metrics"] = {"psnr": 29.0, "generation_seconds": 5.0}
    compared = compare_results(reference, candidate)
    assert compared["candidate_minus_reference"]["default_3"]["psnr"] == -1
    assert compared["generation_speedup"]["default_3"] == 4
    candidate["device_name"] = "GPU-B"
    assert compare_results(reference, candidate)["generation_speedup"] == {}
    from model_training.utils.validation_metrics import METRIC_PROTOCOL

    candidate["metric_protocol"] = copy.deepcopy(METRIC_PROTOCOL)
    # Legacy v3 reports without explicit protocol used the same floor/window.
    assert compare_results(reference, candidate)["candidate_minus_reference"]["default_3"]["psnr"] == -1
    candidate["metric_protocol"]["psnr"]["mse_floor"] = 1e-8
    with pytest.raises(ValueError, match="metric protocols"):
        compare_results(reference, candidate)
    candidate["metric_protocol"] = copy.deepcopy(METRIC_PROTOCOL)
    candidate["manifest_id"] = "different-inputs"
    with pytest.raises(ValueError, match="different or missing input manifests"):
        compare_results(reference, candidate)
