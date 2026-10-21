# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two evaluation owners agree on compatible inputs and sampling options."""

from types import SimpleNamespace

import pytest
import torch

from model_eval.generation import SamplingConfig
from model_eval.inference import process_item
from model_eval.validation import EvaluationRunner
from model_eval.validation_config import EvaluationConfig, EvaluationContext
from model_eval.validation_scoring import ValidationRecorder
from model_eval.validation_tasks import ValidationTaskSpec
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID, WAN22_TI2V_5B_MODEL_ID
from tests.test_run_inference_pipeline_selection import _eval_args, _eval_item


@pytest.mark.parametrize("cosmos3", [False, True])
@pytest.mark.parametrize("kv", [False, True])
def test_offline_and_validation_preserve_the_same_conditioning_and_sampling(tmp_path, cosmos3, kv):
    item = _eval_item(2, 4, cosmos3=cosmos3)
    if cosmos3:
        item["neighbor_camera_rays"] = torch.rand(2, 4, 4, 6)
    calls = []

    class Transformer:
        cp_mesh = SimpleNamespace(size=lambda: 2) if kv else None
        _cp_world_size = 1

        def disable_context_parallel(self):
            self.cp_mesh = None

    class Pipe:
        stages_conditioning_per_chunk = kv
        transformer = Transformer()
        vae = SimpleNamespace(dtype=torch.bfloat16, config=SimpleNamespace(scale_factor_temporal=4))

        def latent_frame_multiple(self):
            return 1

        def cp_raw_frame_padding(self, *args):
            return 0

        def cp_size_candidates(self, *args):
            return [1]

        def cp_pass_latent_frames(self, frames):
            return frames

        def forward_inference(self, **kwargs):
            calls.append(kwargs)
            return kwargs["rendered_rgb"]

        denoise_to_latents = forward_inference

        def decode_latents_to_video(self, latents, **kwargs):
            return latents

        def clear_inference_caches(self):
            pass

    model_id = COSMOS3_NANO_MODEL_ID if cosmos3 else WAN22_TI2V_5B_MODEL_ID
    args = _eval_args(model_id, save_frame_outputs_only=True)
    args.inference_pipeline = "kv_cache" if kv else "bidirectional"
    args.max_neighbors_per_encode = 1
    pipe = Pipe()
    process_item(pipe, item, args, tmp_path, 0, torch.device("cpu"), 4, save_outputs=False)
    offline = calls.pop()

    class Dataset:
        num_frames = len(item["rgb_rendered"])

        def __getitem__(self, index):
            return item

    runner = EvaluationRunner(
        pipe,
        EvaluationConfig(model_id=model_id, validation_metrics=False, sampling=SamplingConfig(num_inference_steps=4)),
        EvaluationContext(torch.device("cpu")),
    )
    runner.recorder = ValidationRecorder(runner.config, runner.context, 4)
    runner.recorder.save = lambda *args, **kwargs: None
    runner.generate_tasks({2: Dataset()}, [(ValidationTaskSpec("default_2", 2), 0)], None)
    validation = calls.pop()
    # Owners deliberately retain distinct seed policies and progress presentation.
    for key in ("noise_seed", "progress_bar_leave"):
        validation.pop(key, None)
    assert validation.keys() == offline.keys()
    for key, value in offline.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(validation[key], value, rtol=0, atol=0)
            assert validation[key].device == value.device
        else:
            assert validation[key] == value, key
