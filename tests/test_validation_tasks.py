from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from model_eval.generation import SamplingConfig
from model_eval.validation import EvaluationRunner
from model_eval.validation_config import EvaluationConfig, EvaluationContext, EvaluationIdentity
from model_eval.validation_scoring import ValidationRecorder
from model_eval.validation_tasks import (
    ValidationTask,
    ValidationTaskSpec,
    build_evaluation_plan,
    build_test_panel_plan,
    build_validation_task,
    build_validation_task_specs,
    validation_noise_seed,
)


class _OneItemDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return {
            "rgb_rendered": torch.ones(2, 3, 4, 4),
            "opacity": torch.ones(2, 4, 4),
            "encoded_prompt": torch.tensor([11, 12, 13], dtype=torch.long),
            "prompt_attention_mask": torch.tensor([True, True, True]),
            "encoded_negative_prompt": torch.tensor([0, 0, 0], dtype=torch.long),
            "negative_prompt_attention_mask": torch.tensor([False, False, False]),
            "rgb_neighbors": torch.ones(2, 3, 4, 4),
            "camera_rays": torch.ones(2, 4, 4, 6),
            "w2cs": torch.eye(4).expand(2, -1, -1).clone(),
            "neighbor_w2cs": torch.eye(4).expand(2, -1, -1).clone(),
            "Ks": torch.eye(3).expand(2, -1, -1).clone(),
            "neighbor_Ks": torch.eye(3).expand(2, -1, -1).clone(),
            "rgb_gt": torch.zeros(2, 3, 4, 4),
        }


def _runner():
    config = EvaluationConfig(
        model_id="nvidia/Cosmos3-Nano", camera_pose_alignment="last", sampling=SamplingConfig(num_inference_steps=4)
    )
    context = EvaluationContext(device=torch.device("cpu"))
    pipe = SimpleNamespace(
        vae=SimpleNamespace(
            dtype=torch.bfloat16, device=torch.device("cpu"), config=SimpleNamespace(scale_factor_temporal=4)
        ),
        transformer=nn.Linear(1, 1),
        spatial_alignment=lambda: (4, 4),
    )
    runner = EvaluationRunner(pipe, config, context)
    runner.recorder = ValidationRecorder(config, context, 4)
    return runner


def test_text_only_validation_keeps_camera_rays():
    runner = _runner()
    specs = build_validation_task_specs({2: _OneItemDataset()})
    text_only = next(spec for spec in specs if spec.key == "text_only_2")

    task = build_validation_task(runner.pipe, runner.config, runner.context, {2: _OneItemDataset()}, text_only, 0)

    assert task.neighbors is None
    assert torch.count_nonzero(task.rendered_rgb) == 0
    assert torch.count_nonzero(task.rendered_opacity) == 0
    assert torch.count_nonzero(task.camera_rays) == task.camera_rays.numel()
    assert torch.equal(task.prompt, torch.tensor([[11, 12, 13]], dtype=torch.long))
    assert torch.equal(task.prompt_attention_mask, torch.tensor([[True, True, True]]))


def test_validation_save_emits_only_composite_artifact(tmp_path):
    runner = _runner()
    output = torch.zeros(1, 1, 3, 4, 4)
    task = ValidationTask(
        key="default_1",
        rendered_rgb=torch.ones(1, 1, 3, 4, 4),
        rendered_opacity=torch.ones(1, 1, 4, 4),
        neighbors=None,
        camera_rays=torch.zeros(1, 1, 4, 4, 6),
        w2cs=torch.eye(4).view(1, 1, 4, 4),
        neighbor_w2cs=None,
        Ks=torch.eye(3).view(1, 1, 3, 3),
        neighbor_Ks=None,
        prompt=torch.tensor([[11, 12, 13]], dtype=torch.long),
        prompt_attention_mask=torch.tensor([[True, True, True]]),
        encoded_negative_prompt=torch.tensor([[0, 0, 0]], dtype=torch.long),
        negative_prompt_attention_mask=torch.tensor([[False, False, False]]),
        num_inference_steps=4,
        rgb_gt=torch.zeros(1, 1, 3, 4, 4),
    )

    runner.recorder.save(output, task, tmp_path)

    assert (tmp_path / "default_1.png").is_file()


def _scoring_task(num_raw_frames: int = 33, temporal_scale: int = 4) -> ValidationTask:
    """A task with one pose per LATENT frame (as the dataset produces) on an out-and-back track."""
    num_latent_frames = (num_raw_frames - 1) // temporal_scale + 1
    # Out and back, so frames far apart in time share a camera centre.
    half = (num_latent_frames + 1) // 2
    offsets = list(range(half)) + list(range(num_latent_frames - half - 1, -1, -1))
    w2cs = torch.eye(4).view(1, 4, 4).repeat(num_latent_frames, 1, 1)
    for index, offset in enumerate(offsets[:num_latent_frames]):
        w2cs[index, :3, 3] = torch.tensor([float(offset), 0.0, 0.0])

    return ValidationTask(
        key="default_3",
        rendered_rgb=torch.rand(1, num_raw_frames, 3, 16, 16),
        rendered_opacity=torch.ones(1, num_raw_frames, 16, 16),
        neighbors=None,
        camera_rays=torch.zeros(1, num_latent_frames, 16, 16, 6),
        w2cs=w2cs.unsqueeze(0),
        neighbor_w2cs=None,
        Ks=torch.eye(3).view(1, 1, 3, 3).expand(1, num_latent_frames, -1, -1).clone(),
        neighbor_Ks=None,
        prompt=torch.tensor([[11, 12, 13]], dtype=torch.long),
        prompt_attention_mask=torch.tensor([[True, True, True]]),
        encoded_negative_prompt=torch.tensor([[0, 0, 0]], dtype=torch.long),
        negative_prompt_attention_mask=torch.tensor([[False, False, False]]),
        num_inference_steps=4,
        rgb_gt=torch.rand(1, num_raw_frames, 3, 16, 16),
    )


def _score(runner, task: ValidationTask, generated: torch.Tensor) -> None:
    runner.recorder.record(generated, task.rgb_gt.squeeze(0), task.rendered_rgb.squeeze(0), task)


def test_scoring_covers_the_view_consistency_family():
    """Guards the raw/latent frame mapping, which fails silently rather than loudly."""
    runner = _runner()
    task = _scoring_task()

    _score(runner, task, task.rgb_gt.squeeze(0))

    recorded = runner.recorder.metrics["default_3"]
    assert recorded["num_repeat_pairs"] > 0
    assert {"revisit_psnr_generated", "revisit_psnr_gt", "revisit_psnr_rendered"} <= set(recorded)


def test_scoring_records_the_expected_metric_families():
    runner = _runner()
    task = _scoring_task()

    _score(runner, task, task.rgb_gt.squeeze(0))

    recorded = runner.recorder.metrics["default_3"]
    assert {"psnr", "ssim", "rendered_psnr", "rendered_ssim", "psnr_gain", "ssim_gain"} <= set(recorded)
    # Generated is the ground truth here, so it must beat the random render.
    assert recorded["psnr_gain"] > 0


def test_one_failing_metric_family_does_not_discard_the_others(monkeypatch):
    runner = _runner()
    task = _scoring_task()

    def boom(*_args, **_kwargs):
        raise RuntimeError("pose track unusable")

    monkeypatch.setattr(
        "model_training.utils.validation_metrics.view_consistency_metrics",
        boom,
    )

    _score(runner, task, task.rgb_gt.squeeze(0))

    recorded = runner.recorder.metrics["default_3"]
    assert "revisit_psnr_generated" not in recorded
    # Everything that did not depend on the failed family survives.
    assert {"psnr", "ssim", "rendered_psnr", "psnr_gain"} <= set(recorded)


def test_ablation_tasks_get_no_rendered_side_metrics():
    runner = _runner()
    task = _scoring_task()
    task.key = "text_only_3"
    task.rendered_rgb = torch.zeros_like(task.rendered_rgb)
    task.has_unmodified_rendered_baseline = False

    _score(runner, task, task.rgb_gt.squeeze(0))

    recorded = runner.recorder.metrics["text_only_3"]
    assert "psnr" in recorded
    assert not [name for name in recorded if name.startswith("rendered_") or name.endswith("_gain")]
    assert "revisit_psnr_rendered" not in recorded


def test_scoring_honours_a_non_default_vae_temporal_scale():
    """Each raw frame is painted with the pose of the latent that covers it, so the correct raw/latent
    mapping scores near-infinite revisit PSNR and a wrong one (e.g. a hardcoded 4) scores low."""
    temporal_scale, num_raw_frames = 8, 65
    runner = _runner()
    runner.pipe.vae.config = SimpleNamespace(scale_factor_temporal=temporal_scale)
    runner.recorder.temporal_scale = temporal_scale
    task = _scoring_task(num_raw_frames=num_raw_frames, temporal_scale=temporal_scale)

    def covering_latent(raw_index: int) -> int:
        return 0 if raw_index == 0 else (raw_index - 1) // temporal_scale + 1

    video = torch.zeros(num_raw_frames, 3, 16, 16)
    for raw_index in range(num_raw_frames):
        # x-translation of the covering latent's pose, normalised into [0, 1].
        video[raw_index] = task.w2cs[0, covering_latent(raw_index), 0, 3] / 10.0
    task.rgb_gt = video.unsqueeze(0)

    _score(runner, task, video)

    recorded = runner.recorder.metrics["default_3"]
    assert recorded["num_repeat_pairs"] > 0
    assert recorded["revisit_psnr_gt"] > 60.0


def test_spec_flags_decide_ablation_and_rendered_scoring():
    """The spec carries the semantics; nothing parses task keys."""
    specs = {spec.key: spec for spec in build_validation_task_specs({3: None, 6: None})}

    assert {key for key, spec in specs.items() if not spec.is_ablation} == {"default_3", "default_6"}
    assert specs["last_drop_0.5_3"].is_ablation
    # The last_drop tasks zero the render's tail, so scoring the render against gt would be scoring black frames.
    assert not specs["last_drop_0.5_3"].has_unmodified_rendered_baseline
    assert not specs["no_rendered_rgb_3"].has_unmodified_rendered_baseline
    assert specs["no_text_3"].has_unmodified_rendered_baseline
    assert ValidationTaskSpec(key="default_3", num_neighbors=3).has_unmodified_rendered_baseline


def _panel_runner(num_scenes: int = 8, dataset_len: int = 28):
    runner = _runner()
    runner.config.test_panel_scenes = num_scenes

    class _Dataset:
        def __len__(self):
            return dataset_len

    runner.datasets = {3: _Dataset(), 6: _Dataset(), 12: _Dataset()}
    return runner


def test_panel_plan_sweeps_fixed_scenes_with_both_configurations():
    plan = build_test_panel_plan(_panel_runner().datasets, 8)

    configs = {spec.key.split("@")[0] for spec, _ in plan}
    # Both are required: with the rendered conditioning present it carries the geometry,
    # so a camera-conditioning difference is invisible.
    assert configs == {"default_3", "no_rendered_rgb_3"}
    assert len(plan) == 8 * 2
    assert sorted({index for _, index in plan}) == list(range(8))


def test_panel_plan_gives_every_scene_its_own_metric_key():
    plan = build_test_panel_plan(_panel_runner().datasets, 4)
    keys = [spec.key for spec, _ in plan]
    # Without the scene suffix all four scenes would collide on one task key and only
    # the last would survive the gather.
    assert len(set(keys)) == len(keys)
    assert "default_3@scene00" in keys and "no_rendered_rgb_3@scene03" in keys


def test_panel_plan_is_capped_by_the_available_scenes():
    runner = _panel_runner(dataset_len=3)
    plan = build_test_panel_plan(runner.datasets, 8, runner.config.max_validation_tasks)
    assert sorted({index for _, index in plan}) == [0, 1, 2]


def test_panel_plan_honours_the_validation_task_cap():
    runner = _panel_runner()
    runner.config.max_validation_tasks = 3
    plan = build_test_panel_plan(runner.datasets, 8, runner.config.max_validation_tasks)
    assert len(plan) == 3


def test_validation_noise_seed_is_stable_task_specific_and_independent_of_global_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        first = validation_noise_seed("default_3")
        torch.manual_seed(99)
        assert validation_noise_seed("default_3") == first
    assert first != validation_noise_seed("no_rendered_rgb_3")
    assert validation_noise_seed("default_3@scene00") != validation_noise_seed("default_3@scene01")


def test_standalone_runner_restores_model_mode_on_generation_failure(monkeypatch, tmp_path):
    import pytest

    runner = _panel_runner(num_scenes=1)
    plan = build_evaluation_plan(runner.datasets, runner.config, 0)

    def fail(*args, **kwargs):
        assert not runner.pipe.transformer.training
        assert not torch.is_grad_enabled()
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(runner, "run_pass", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        runner.run(plan, tmp_path, EvaluationIdentity(tmp_path))
    assert runner.pipe.transformer.training
