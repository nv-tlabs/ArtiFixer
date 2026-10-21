# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from model_eval.checkpoint_loading import (
    _torch_load_state_dict,
    add_checkpoint_args,
    eval_pipe_build_context,
    load_model_weights_from_pt,
    load_transformer_checkpoint,
)
from model_training.checkpoint_metadata import checkpoint_metadata_path, resolve_camera_pose_alignment

COSMOS3 = "nvidia/Cosmos3-Nano"
WAN = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


class CheckpointLoadingTests(unittest.TestCase):
    def parse_checkpoint_args(self, argv):
        parser = argparse.ArgumentParser()
        add_checkpoint_args(parser)
        return parser.parse_args(argv)

    def test_requires_checkpoint_source(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                self.parse_checkpoint_args([])

    def test_load_transformer_checkpoint_dispatches_by_checkpoint_extension(self):
        """device/dtype only reach the .pt loader (they materialize its meta
        parameters); the DCP loader fills the already cast and placed model."""
        calls = []
        model = object()
        fake_train_utils = SimpleNamespace(
            load_model_weights_from_dcp=lambda loaded_model, path: calls.append(("dcp", loaded_model, path))
        )

        with (
            patch.dict("sys.modules", {"model_training.utils.train_utils": fake_train_utils}),
            patch(
                "model_eval.checkpoint_loading.load_model_weights_from_pt",
                lambda loaded_model, path, **placement: calls.append(("pt", loaded_model, path, placement)),
            ),
        ):
            load_transformer_checkpoint(
                model,
                Namespace(checkpoint=Path("/runs/checkpoint_400/pytorch_model_fsdp_2")),
                device="cuda:0",
                dtype="bf16",
            )
            load_transformer_checkpoint(
                model, Namespace(checkpoint=Path("/exports/model.pt")), device="cuda:0", dtype="bf16"
            )

        self.assertEqual(
            calls,
            [
                ("dcp", model, Path("/runs/checkpoint_400/pytorch_model_fsdp_2")),
                ("pt", model, Path("/exports/model.pt"), {"device": "cuda:0", "dtype": "bf16"}),
            ],
        )

    def test_build_context_puts_parameters_on_meta_only_for_pt_checkpoints(self):
        import torch
        from torch import nn

        with eval_pipe_build_context(Namespace(checkpoint=Path("/exports/model.pt"))) as on_meta:
            module = nn.Linear(4, 3)
            module.register_buffer("table", torch.arange(3.0))
        self.assertTrue(on_meta)
        self.assertTrue(module.weight.is_meta)
        self.assertFalse(module.table.is_meta)  # init-time tables are computed for real

        with eval_pipe_build_context(
            Namespace(checkpoint=Path("/runs/checkpoint_400/pytorch_model_fsdp_2"))
        ) as on_meta:
            module = nn.Linear(4, 3)
        self.assertFalse(on_meta)
        self.assertFalse(module.weight.is_meta)

    def test_build_context_lets_from_pretrained_materialize_the_vae(self):
        """get_eval_pipe builds the VAE inside the context as well. It leaves
        the block with real weights only because diffusers' from_pretrained
        builds under its own init_empty_weights and materializes what it built
        (the low_cpu_mem_usage default); an eagerly built module would stay on
        meta and fail at .to(). This pins that default."""
        import torch
        from diffusers import AutoencoderKLWan

        tiny = AutoencoderKLWan(
            base_dim=4,
            z_dim=2,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            latents_mean=[0.0, 0.0],
            latents_std=[1.0, 1.0],
        )
        with TemporaryDirectory() as tmpdir:
            tiny.save_pretrained(tmpdir)
            with eval_pipe_build_context(Namespace(checkpoint=Path("/exports/model.pt"))) as on_meta:
                vae = AutoencoderKLWan.from_pretrained(tmpdir, torch_dtype=torch.bfloat16)

        self.assertTrue(on_meta)
        self.assertFalse(any(parameter.is_meta for parameter in vae.parameters()))
        vae.to("cpu")  # what _vae_for_model does next; raises on a meta-built module

    def _resolve(
        self,
        metadata: dict | None,
        requested: str | None,
        suffix: str = ".pt",
        model_id: str = WAN,
    ) -> str:
        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / f"model{suffix}"
            path = checkpoint_metadata_path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            if metadata is not None:
                path.write_text(json.dumps(metadata))
            return resolve_camera_pose_alignment(checkpoint, model_id, camera_pose_alignment=requested)

    def test_pt_eval_adopts_the_alignment_recorded_in_the_export_metadata(self):
        """Evaluation uses the recorded pose alignment, since the wrong one silently
        degrades every metric. An unrecorded Wan export uses "mean"."""
        self.assertEqual(self._resolve({"camera_pose_alignment": "last"}, requested=None), "last")
        self.assertEqual(self._resolve({"camera_pose_alignment": "last"}, requested="last"), "last")
        self.assertEqual(self._resolve({}, requested=None), "mean")

    def test_pt_eval_rejects_a_flag_that_contradicts_the_export_metadata(self):
        for model_id in (COSMOS3, WAN):
            for recorded, requested in (("last", "mean"), ("mean", "last")):
                with self.subTest(model_id=model_id, recorded=recorded, requested=requested):
                    metadata = {"camera_pose_alignment": recorded}
                    message = f"uses camera_pose_alignment={recorded!r}, but {requested!r} was requested"
                    with self.assertRaisesRegex(ValueError, message):
                        self._resolve(metadata, requested, model_id=model_id)

    def test_released_wan_checkpoints_use_mean_without_an_alignment_record(self):
        for model_id in (WAN, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "/models/Wan2.1-T2V-14B-Diffusers"):
            for suffix in (".pt", ""):
                for metadata in (None, {}, {"model_id": model_id}):
                    with self.subTest(model_id=model_id, suffix=suffix, metadata=metadata):
                        for requested in (None, "mean"):
                            self.assertEqual(self._resolve(metadata, requested, suffix, model_id), "mean")
                        message = (
                            "has no camera_pose_alignment record; unrecorded Wan checkpoints use 'mean', "
                            "but 'last' was requested"
                        )
                        with self.assertRaisesRegex(ValueError, message):
                            self._resolve(metadata, "last", suffix, model_id)

    def test_other_models_require_recorded_pose_alignment(self):
        for model_id in (COSMOS3, "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "/models/custom-model"):
            for suffix in (".pt", ""):
                for metadata in (None, {}, {"model_id": model_id}):
                    for requested in (None, "mean", "last"):
                        with self.subTest(model_id=model_id, suffix=suffix, metadata=metadata, requested=requested):
                            with self.assertRaisesRegex(ValueError, "has no camera_pose_alignment record"):
                                self._resolve(metadata, requested, suffix, model_id)

    def test_a_stamped_checkpoint_preserves_both_pose_alignments(self):
        for model_id in (COSMOS3, WAN):
            for pose in ("mean", "last"):
                for suffix in (".pt", ""):
                    with self.subTest(model_id=model_id, pose=pose, suffix=suffix):
                        metadata = {"camera_pose_alignment": pose}
                        self.assertEqual(
                            self._resolve(metadata, requested=None, suffix=suffix, model_id=model_id), pose
                        )
                        self.assertEqual(
                            self._resolve(metadata, requested=pose, suffix=suffix, model_id=model_id), pose
                        )

    def test_an_unknown_recorded_rule_is_refused_for_either_family(self):
        """Reject unsupported names rather than giving them a different meaning."""
        for model_id in (COSMOS3, WAN):
            for pose in ("first", "invalid", None):
                for suffix in (".pt", ""):
                    with self.subTest(model_id=model_id, pose=pose, suffix=suffix):
                        with self.assertRaisesRegex(ValueError, f"records camera_pose_alignment={pose!r}; expected"):
                            self._resolve(
                                {"camera_pose_alignment": pose}, requested=None, suffix=suffix, model_id=model_id
                            )

    def test_checkpoint_metadata_must_be_a_json_object(self):
        for model_id in (COSMOS3, WAN):
            for metadata in (["camera_pose_alignment"], "camera_pose_alignment", 1):
                with self.subTest(model_id=model_id, metadata=metadata):
                    with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
                        self._resolve(metadata, requested=None, model_id=model_id)

    def _meta_linear(self, in_features: int = 4, out_features: int = 3):
        from torch import nn

        with eval_pipe_build_context(Namespace(checkpoint=Path("/exports/model.pt"))):
            return nn.Linear(in_features, out_features)

    def test_pt_load_rejects_a_key_mismatch_before_touching_any_weight(self):
        import torch

        model = self._meta_linear()  # keys: weight, bias
        loaded = {"weight": torch.ones(3, 4), "extra": torch.ones(1)}

        with patch("model_eval.checkpoint_loading._torch_load_state_dict", return_value=loaded):
            with self.assertRaisesRegex(RuntimeError, r"missing.*bias.*unexpected.*extra"):
                load_model_weights_from_pt(
                    model, Path("/exports/model.pt"), device=torch.device("cpu"), dtype=torch.bfloat16
                )

        self.assertTrue(model.weight.is_meta)  # the matching key was not materialized either

    def test_pt_load_rejects_a_shape_mismatch_before_touching_any_weight(self):
        import torch
        from torch import nn

        model = self._meta_linear(in_features=5)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            torch.save(nn.Linear(4, 3).state_dict(), path)
            with self.assertRaisesRegex(RuntimeError, r"shape mismatch.*weight.*\(3, 4\).*\(3, 5\)"):
                load_model_weights_from_pt(model, path, device=torch.device("cpu"), dtype=torch.bfloat16)

        self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))  # the bias matched

    def test_pt_load_materializes_meta_parameters_on_the_requested_device_and_dtype(self):
        """The eval path: parameters built on the meta device are created from
        the checkpoint tensors, cast per tensor, under inference mode
        (run_inference.main); the cast (on the GPU path also the device
        transfer) gives them their own memory instead of the file mapping."""
        import torch
        from torch import nn

        reference = nn.Linear(4, 3)
        model = self._meta_linear()
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            torch.save(reference.state_dict(), path)
            loaded = _torch_load_state_dict(path)
            with patch("model_eval.checkpoint_loading._torch_load_state_dict", return_value=loaded):
                with torch.inference_mode():
                    load_model_weights_from_pt(model, path, device=torch.device("cpu"), dtype=torch.bfloat16)

        self.assertFalse(any(parameter.is_meta for parameter in model.parameters()))
        self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.bfloat16})
        self.assertTrue(torch.equal(model.weight, reference.weight.to(torch.bfloat16)))
        self.assertTrue(torch.equal(model.bias, reference.bias.to(torch.bfloat16)))
        for name, parameter in model.named_parameters():
            self.assertNotEqual(parameter.data_ptr(), loaded[name].data_ptr(), name)

    def test_pt_load_replaces_the_weights_of_an_already_materialized_model(self):
        """Direct callers only; the eval path always arrives on meta."""
        import torch
        from torch import nn

        model = nn.Linear(4, 3)
        reference = nn.Linear(4, 3)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            torch.save(reference.state_dict(), path)
            load_model_weights_from_pt(model, path)

        self.assertTrue(torch.equal(model.weight, reference.weight))
        self.assertTrue(torch.equal(model.bias, reference.bias))

    def test_dcp_eval_reads_pose_alignment_from_the_slot_local_training_record(self):
        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "pytorch_model_fsdp_0"
            checkpoint.mkdir()
            checkpoint_metadata_path(checkpoint).write_text(json.dumps(dict(camera_pose_alignment="last")))
            self.assertEqual(resolve_camera_pose_alignment(checkpoint, COSMOS3), "last")
            with self.assertRaisesRegex(ValueError, "uses camera_pose_alignment='last', but 'mean' was requested"):
                resolve_camera_pose_alignment(checkpoint, COSMOS3, camera_pose_alignment="mean")

    def test_torch_load_uses_docker_supported_kwargs(self):
        calls = []
        expected_state_dict = {"blocks.0.weight": object()}

        def torch_load(path, **kwargs):
            calls.append((path, kwargs))
            return expected_state_dict

        with patch.dict("sys.modules", {"torch": SimpleNamespace(load=torch_load)}):
            state_dict = _torch_load_state_dict(Path("/exports/model.pt"))

        self.assertIs(state_dict, expected_state_dict)
        self.assertEqual(
            calls,
            [
                (
                    Path("/exports/model.pt"),
                    {"map_location": "cpu", "weights_only": True, "mmap": True},
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
