# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from accelerate import Accelerator
from torch import nn

from model_eval.validation import EvaluationRunner
from model_eval.validation_config import EvaluationIdentity
from model_eval.validation_reporting import EvaluationResult
from model_eval.validation_scoring import EvaluationPass
from model_eval.validation_tasks import ValidationTaskSpec, build_evaluation_plan, build_validation_task
from model_training.checkpoint_metadata import write_checkpoint_metadata
from model_training.compare_validation_results import compare_results
from model_training.data.dl3dv_test import DL3DVPairedDatasetTest
from model_training.pipeline.wan_pipeline import ArtifixerPipeline
from model_training.train import build_parser
from model_training.utils import train_utils, validation_metrics
from tests.test_validation_automation import identity_panel
from tests.test_validation_tasks import _runner


@pytest.mark.parametrize("family", ["wan", "cosmos3"])
@pytest.mark.parametrize("slot", ["pytorch_model_fsdp_0", "pytorch_model_fsdp_ema"])
@pytest.mark.parametrize("requested_pose", [None, "mean", "last"])
@pytest.mark.parametrize("recorded_pose", ["mean", "last", "missing", "invalid"])
def test_validate_only_checks_exact_dcp_slot_before_dataset(
    monkeypatch, tmp_path, family, slot, requested_pose, recorded_pose
):
    checkpoint = tmp_path / "checkpoint" / slot
    source = nn.Linear(2, 2)
    dcp.save({"model": source.state_dict()}, checkpoint_id=checkpoint)
    # A valid parent record must not hide an invalid or absent model-slot record.
    write_checkpoint_metadata(checkpoint.parent, {"camera_pose_alignment": "last"})
    expected_pose = "mean" if recorded_pose == "missing" else recorded_pose
    error = None
    if recorded_pose == "invalid":
        error = "records camera_pose_alignment='invalid'; expected"
    elif recorded_pose == "missing":
        if family == "cosmos3":
            error = "has no camera_pose_alignment record; required for 'nvidia/Cosmos3-Nano'"
        elif requested_pose == "last":
            error = (
                "has no camera_pose_alignment record; unrecorded Wan checkpoints use 'mean', but 'last' was requested"
            )
    elif requested_pose is not None and requested_pose != recorded_pose:
        error = f"uses camera_pose_alignment='{recorded_pose}', but '{requested_pose}' was requested"
    if recorded_pose != "missing":
        write_checkpoint_metadata(checkpoint, {"camera_pose_alignment": recorded_pose})
    argv = [
        "--validate_only",
        "--base_checkpoint_dir",
        str(checkpoint),
        "--project_dir",
        str(tmp_path / "eval"),
        "--split_path",
        "unused",
        "--dl3dv_dir",
        "unused",
        "--prompt_dir",
        "unused",
        "--log_with",
        "none",
        "--model_id",
        "nvidia/Cosmos3-Nano" if family == "cosmos3" else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    ]
    if requested_pose is not None:
        argv += ["--camera_pose_alignment", requested_pose]
    args = build_parser().parse_args(argv)
    pipe = SimpleNamespace(transformer=nn.Linear(2, 2), vae=SimpleNamespace(device=torch.device("cpu")))
    accelerator = Accelerator(cpu=True)
    reached = []

    def datasets(*unused):
        assert error is None, "Invalid camera metadata reached dataset construction"
        assert args.camera_pose_alignment == expected_pose
        for actual, expected in zip(pipe.transformer.parameters(), source.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reached.append("loaded")
        return {3: object()}

    monkeypatch.setattr(train_utils, "get_val_datasets", datasets)
    monkeypatch.setattr(
        EvaluationRunner,
        "run",
        lambda self, plan, output, identity: reached.append("validated") or EvaluationResult({}, None, output),
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            train_utils.get_run_id_and_should_resume(args)
        assert not reached
    else:
        assert train_utils.get_run_id_and_should_resume(args) == (None, False)
        train_utils.run_validation_only(args, accelerator, pipe, frames_per_block=None)
        assert reached == ["loaded", "validated"]


@pytest.mark.requires_host_ffmpeg
def test_disabled_metrics_avoid_fingerprints_but_replay_checks_sources(identity_panel, monkeypatch):
    _, kwargs, manifest, _ = identity_panel
    split = json.loads(kwargs["split_path"].read_text())
    split["test"] = split["trainval"]
    kwargs["split_path"].write_text(json.dumps(split))
    args = SimpleNamespace(
        **{key: kwargs[key] for key in ("split_path", "dl3dv_dir", "prompt_dir", "dataset_scaling_factor")},
        validation_metrics=False,
        test_panel_scenes=0,
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        camera_pose_alignment="mean",
    )
    datasets = train_utils.get_val_datasets(args, SimpleNamespace(is_main_process=False), 5, None)
    calls = []
    import model_training.data.dl3dv_base as base

    fingerprint = base.file_sha256
    monkeypatch.setattr(base, "file_sha256", lambda path: calls.append(path) or fingerprint(path))
    sample = datasets[3][0]
    assert "prompt" in sample and "prompt_file" in sample
    assert "validation_sample_json" not in sample
    assert calls == []
    replay = DL3DVPairedDatasetTest(
        num_views=3, validation_manifest=manifest, capture_validation_metadata=False, **kwargs
    )
    replay[0]
    assert len(calls) == 5


def _wan_pipe(negative_path):
    pipe = ArtifixerPipeline.__new__(ArtifixerPipeline)
    nn.Module.__init__(pipe)
    pipe.transformer = nn.Linear(1, 1)
    pipe.vae = SimpleNamespace(
        device=torch.device("cpu"), dtype=torch.bfloat16, config=SimpleNamespace(scale_factor_temporal=4)
    )
    pipe.default_negative_prompt = nn.Buffer(torch.load(negative_path, weights_only=True), persistent=False)
    return pipe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for split pipeline placement")
def test_wan_validation_moves_default_negative_prompt_to_vae_device(tmp_path):
    negative = torch.randn(1, 4, 8)
    path = tmp_path / "negative.pt"
    torch.save(negative, path)
    pipe = _wan_pipe(path)
    pipe.vae.device = torch.device("cuda", torch.cuda.current_device())
    # Training/evaluation prepares only the transformer, leaving pipeline buffers on CPU.
    pipe.transformer = nn.Linear(8, 8).to(pipe.vae.device)
    prompt = torch.randn(2, 4, 8)
    conditional, unconditional = pipe._resolve_text_conditioning(prompt, None, None, None, 2, pipe.vae.device)
    assert conditional["encoder_hidden_states"].device == pipe.vae.device
    assert unconditional["encoder_hidden_states"].device == pipe.vae.device
    torch.testing.assert_close(unconditional["encoder_hidden_states"].cpu(), negative.expand(2, -1, -1))
    assert torch.isfinite(pipe.transformer(unconditional["encoder_hidden_states"])).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for Wan default CFG")
def test_wan_model_only_gpu_placement_runs_default_cfg():
    from tests.test_wan_pipeline_cp_inference import _full_clip_inputs, _pipe

    pipe = _pipe()
    device = torch.device("cuda", torch.cuda.current_device())
    pipe.transformer.to(device=device, dtype=torch.float32).eval()
    pipe.vae.device = device
    pipe.latents_mean = pipe.latents_mean.to(device)
    pipe.latents_inv_std = pipe.latents_inv_std.to(device)
    pipe.default_negative_prompt = nn.Buffer(torch.randn(1, 5, 32), persistent=False)
    inputs = _full_clip_inputs()
    inputs.pop("negative_prompt")
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with torch.no_grad():
        output = pipe.forward_inference(**inputs)
    assert output.shape == inputs["rendered_rgb"].shape
    assert torch.isfinite(output).all()


def _capture(trainer, dataset, directory):
    trainer.val_datasets = {3: dataset}
    trainer.config.test_panel_scenes = 1
    trainer.config.max_validation_tasks = 1
    trainer.config.dataset_scaling_factor = 0.01
    trainer.config.validation_strict = True
    for key in ("default_3", "default_3@scene00", "no_rendered_rgb_3@scene00"):
        task = build_validation_task(
            trainer.pipe,
            trainer.config,
            trainer.context,
            {3: dataset},
            ValidationTaskSpec(key, 3, zero_rendered=key.startswith("no_rendered")),
            0,
        )
        trainer.recorder.save(task.rgb_gt, task, None)
    trainer._panel_records = {key: value for key, value in trainer.recorder.records.items() if "@" in key}
    trainer.finish(
        build_evaluation_plan(trainer.val_datasets, trainer.config, 0),
        EvaluationIdentity(directory),
        EvaluationPass(trainer.recorder.metrics, trainer.recorder.records),
        EvaluationPass({key: trainer.recorder.metrics[key] for key in trainer._panel_records}, trainer._panel_records),
        directory,
    )
    return json.loads((directory / "manifest.json").read_text()), json.loads((directory / "results.json").read_text())


@pytest.mark.requires_host_ffmpeg
def test_negative_prompt_changes_replay_and_comparison_identity(identity_panel, tmp_path):
    _, kwargs, _, _ = identity_panel
    dataset = DL3DVPairedDatasetTest(num_views=3, capture_validation_metadata=True, **kwargs)
    negative = tmp_path / "negative.pt"
    torch.save(torch.zeros(1, 4, 8, dtype=torch.bfloat16), negative)
    trainer = _runner()
    trainer.pipe = _wan_pipe(negative)
    manifest, report = _capture(trainer, dataset, tmp_path / "before")
    replay = DL3DVPairedDatasetTest(num_views=3, validation_manifest=manifest, **kwargs)
    task = build_validation_task(
        trainer.pipe, trainer.config, trainer.context, {3: replay}, ValidationTaskSpec("default_3", 3), 0
    )
    assert task.negative_conditioning == manifest["samples"][0]["negative_conditioning"]
    torch.save(torch.ones(1, 4, 8, dtype=torch.bfloat16), negative)
    changed = _runner()
    changed.pipe = _wan_pipe(negative)
    with pytest.raises(ValueError, match="negative_conditioning mismatch"):
        build_validation_task(
            changed.pipe, changed.config, changed.context, {3: replay}, ValidationTaskSpec("default_3", 3), 0
        )
    _, changed_report = _capture(changed, dataset, tmp_path / "after")
    assert changed_report["manifest_id"] != report["manifest_id"]
    with pytest.raises(ValueError, match="different or missing input manifests"):
        compare_results(report, changed_report)
    assert compare_results(report, report)["manifest_id"] == report["manifest_id"]


def test_revisit_selection_and_transfers_are_chunked_with_unchunked_parity(monkeypatch):
    centers = torch.cat((torch.arange(40), torch.arange(39, -1, -1))).float()
    poses = torch.eye(4).repeat(80, 1, 1)
    poses[:, 0, 3] = -centers
    pairs = validation_metrics.camera_repeat_pairs(poses)
    assert len(pairs) > 4 * validation_metrics.FRAME_CHUNK
    video = torch.rand(80, 3, 16, 16)
    left, right = zip(*pairs)
    expected_psnr = validation_metrics.psnr(video[list(left)], video[list(right)]).mean().item()
    expected_ssim = validation_metrics.ssim(video[list(left)], video[list(right)]).mean().item()
    batches = []
    original_to, original_getitem = torch.Tensor.to, torch.Tensor.__getitem__

    def transfer(tensor, *args, **kwargs):
        if tensor.ndim == 4:
            batches.append(len(tensor))
            assert len(tensor) <= validation_metrics.FRAME_CHUNK
        return original_to(tensor, *args, **kwargs)

    def select(tensor, index):
        if tensor is video and isinstance(index, torch.Tensor):
            assert index.numel() <= validation_metrics.FRAME_CHUNK
        return original_getitem(tensor, index)

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    monkeypatch.setattr(torch.Tensor, "__getitem__", select)
    actual = validation_metrics.view_consistency_metrics({"generated": video}, poses, device="cpu")
    assert len(batches) == 2 * ((len(pairs) + validation_metrics.FRAME_CHUNK - 1) // validation_metrics.FRAME_CHUNK)
    assert actual["revisit_psnr_generated"] == pytest.approx(expected_psnr, abs=1e-6)
    assert actual["revisit_ssim_generated"] == pytest.approx(expected_ssim, abs=1e-6)
