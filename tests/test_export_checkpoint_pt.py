# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import subprocess
import sys
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from accelerate.utils import DistributedType

from model_eval.export_checkpoint_pt import (
    DEFAULT_MODEL_ID,
    EXPORT_TRANSFORMER_DEFAULTS,
    _export_transformer_args,
    build_metadata,
    create_export_transformer,
    export_checkpoint,
    parse_args,
)
from model_training.checkpoint_metadata import (
    checkpoint_metadata_path,
    read_checkpoint_metadata,
    resolve_camera_pose_alignment,
)
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID

EXPORT_MODELS = (DEFAULT_MODEL_ID, COSMOS3_NANO_MODEL_ID)
_RECORDED_MEAN_CONFLICT = "uses camera_pose_alignment='mean', but 'last' was requested"


@pytest.mark.parametrize("stage", ["dmd"])
@pytest.mark.parametrize(
    "model_id,pose", [(DEFAULT_MODEL_ID, "last"), (COSMOS3_NANO_MODEL_ID, "mean")], ids=["wan", "cosmos3"]
)
def test_ema_checkpoint_pose_alignment_round_trip(tmp_path, isolated_accelerator_state, stage, model_id, pose):
    from model_training.utils import train_utils
    from tests._helpers import _advance_ema_stage, _EMATinyModel, _small_dmd_ema_trainer

    record = {"camera_pose_alignment": pose}
    accelerator = Accelerator(cpu=True)
    train_utils._register_checkpoint_metadata_hooks(Namespace(**record), accelerator)
    model = _EMATinyModel(2, 2)
    model, optimizer = accelerator.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
    accelerator.prepare(_EMATinyModel(2, 2))
    trainer = _small_dmd_ema_trainer(
        tmp_path,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        extra_args=("--model_id", model_id, "--camera_pose_alignment", pose),
    )
    _advance_ema_stage(stage, trainer, 206)
    checkpoint = tmp_path / f"checkpoint_{trainer.get_step()}"
    accelerator.save_state(str(checkpoint))
    ema_slot = checkpoint / "pytorch_model_fsdp_ema"
    raw_slots = [checkpoint / f"pytorch_model_fsdp_{i}" for i in range(2)]
    for path in (checkpoint, *raw_slots, ema_slot):
        assert read_checkpoint_metadata(path) == record
        assert resolve_camera_pose_alignment(path, model_id) == pose

    argv = [
        "--checkpoint_dir",
        str(ema_slot),
        "--output_pt",
        str(tmp_path / "ema.pt"),
        "--model_id",
        model_id,
        "--run_id",
        "camera-round-trip",
        "--checkpoint",
        str(trainer.get_step()),
        "--slot",
        ema_slot.name,
    ]
    args = parse_args(argv)
    consumer = _EMATinyModel(2, 2)
    with patch("model_eval.export_checkpoint_pt.create_export_transformer", return_value=consumer) as build:
        output, metadata_path = export_checkpoint(args)
    build.assert_called_once_with(args)
    saved = torch.load(output, weights_only=True)
    for (name, raw), expected in zip(model.named_parameters(), trainer.ema.parameters(), strict=True):
        torch.testing.assert_close(saved[name], expected, rtol=0, atol=0)
        assert not torch.equal(saved[name], raw)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["weight_type"] == "ema"
    assert metadata["training_stage"] == stage
    assert resolve_camera_pose_alignment(output, model_id) == pose

    opposite = "mean" if pose == "last" else "last"
    with pytest.raises(ValueError, match=f"uses camera_pose_alignment={pose!r}, but {opposite!r} was requested"):
        parse_args([*argv, "--camera_pose_alignment", opposite])


class ExportCheckpointPtTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def _stamp(checkpoint, **metadata):
        path = checkpoint_metadata_path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata))
        return path

    def _export_args(self, *extra, model_id=COSMOS3_NANO_MODEL_ID):
        source = self.root / "source"
        if model_id == COSMOS3_NANO_MODEL_ID:
            self._stamp(source, camera_pose_alignment="last")
        return parse_args(
            [
                "--checkpoint_dir",
                str(source),
                "--output_pt",
                str(self.root / "exports/model.pt"),
                "--run_id",
                "run-id",
                "--checkpoint",
                "1",
                "--slot",
                "pytorch_model_fsdp_0",
                "--model_id",
                model_id,
                *extra,
            ]
        )

    def _metadata(self, args):
        return build_metadata(
            args, branch="branch", commit="commit", export_date=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

    def test_checkpoint_metadata_does_not_import_the_evaluation_layer(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import model_training.checkpoint_metadata; "
                "assert 'torch' not in sys.modules; "
                "assert not any(name == 'model_eval' or name.startswith('model_eval.') for name in sys.modules)",
            ],
            check=True,
        )

    def test_metadata_paths(self):
        self.assertEqual(checkpoint_metadata_path(Path("/exports/model.pt")), Path("/exports/model.metadata.json"))
        self.assertEqual(checkpoint_metadata_path(Path("/runs/slot")), Path("/runs/slot/artifixer.metadata.json"))

    def test_build_metadata_records_release_source_and_pose_alignment(self):
        args = self._export_args("--source_path", "/canonical/checkpoint_1/pytorch_model_fsdp_0")
        metadata = self._metadata(args)
        self.assertEqual(metadata["run_id"], "run-id")
        self.assertEqual(metadata["checkpoint"], 1)
        self.assertEqual(metadata["slot"], "pytorch_model_fsdp_0")
        self.assertEqual(metadata["source_path"], "/canonical/checkpoint_1/pytorch_model_fsdp_0")
        self.assertEqual(metadata["camera_pose_alignment"], "last")
        self.assertEqual(metadata["branch"], "branch")
        self.assertEqual(metadata["commit"], "commit")
        self.assertEqual(metadata["export_date"], "2026-01-01T00:00:00+00:00")

    def test_unrecorded_wan_checkpoints_use_mean_alignment(self):
        args = self._export_args(model_id=DEFAULT_MODEL_ID)
        self.assertEqual(args.camera_pose_alignment, "mean")
        with self.assertRaisesRegex(
            ValueError,
            "has no camera_pose_alignment record; unrecorded Wan checkpoints use 'mean', but 'last' was requested",
        ):
            self._export_args("--camera_pose_alignment", "last", model_id=DEFAULT_MODEL_ID)

    def test_cosmos3_requires_recorded_pose_alignment(self):
        self._stamp(self.root / "incomplete")
        with self.assertRaisesRegex(ValueError, "has no camera_pose_alignment record; required for"):
            self._export_args("--checkpoint_dir", str(self.root / "incomplete"))

    def test_pose_alignment_is_read_from_pt_and_dcp_records(self):
        for checkpoint in (self.root / "model.pt", self.root / "slot"):
            self._stamp(checkpoint, camera_pose_alignment="mean")
            self.assertEqual(resolve_camera_pose_alignment(checkpoint, COSMOS3_NANO_MODEL_ID), "mean")
            with self.assertRaisesRegex(ValueError, _RECORDED_MEAN_CONFLICT):
                resolve_camera_pose_alignment(checkpoint, COSMOS3_NANO_MODEL_ID, camera_pose_alignment="last")

    def test_invalid_pose_alignment_is_rejected(self):
        self._stamp(self.root / "invalid", camera_pose_alignment="invalid")
        for model_id in EXPORT_MODELS:
            with self.subTest(model_id=model_id):
                with self.assertRaisesRegex(ValueError, "records camera_pose_alignment='invalid'; expected"):
                    resolve_camera_pose_alignment(self.root / "invalid", model_id)

    def test_export_transformer_args_add_hidden_defaults(self):
        args = self._export_args()
        self.assertFalse(hasattr(args, "frames_per_block"))
        transformer_args = _export_transformer_args(args)
        for name, value in EXPORT_TRANSFORMER_DEFAULTS.items():
            self.assertEqual(getattr(transformer_args, name), value)

    def test_create_export_transformer_exports_both_families(self):
        from model_training.utils import train_utils

        for model_id in EXPORT_MODELS:
            transformer = object()
            with patch.object(
                train_utils, "get_kv_cache_pipe", return_value=SimpleNamespace(transformer=transformer)
            ) as build:
                exported = create_export_transformer(self._export_args(model_id=model_id))
            self.assertIs(exported, transformer)
            self.assertEqual(build.call_args.args[0].model_id, model_id)
            self.assertEqual(build.call_args.args[0].frames_per_block, 7)
            self.assertEqual(build.call_args.kwargs, {"device": "cpu"})

    def test_export_checkpoint_round_trips_weights_and_pose_alignment(self):
        from model_eval.run_inference import parse_args as parse_eval_args
        from model_training.utils import train_utils

        for model_id in EXPORT_MODELS:
            pose = "mean" if model_id == COSMOS3_NANO_MODEL_ID else "last"
            source = self.root / model_id.split("/")[-1] / "slot"
            output = source.parent / "model.pt"
            self._stamp(source, camera_pose_alignment=pose)
            args = self._export_args(
                "--checkpoint_dir",
                str(source),
                "--output_pt",
                str(output),
                "--branch",
                "branch",
                "--commit",
                "commit",
                model_id=model_id,
            )
            transformer = torch.nn.Linear(2, 2)
            torch.nn.init.constant_(transformer.weight, 3.0)
            torch.nn.init.constant_(transformer.bias, -2.0)
            with (
                patch("model_eval.export_checkpoint_pt.create_export_transformer", return_value=transformer) as build,
                patch.object(train_utils, "load_model_weights_from_dcp") as load,
            ):
                checkpoint, metadata_path = export_checkpoint(args)
            build.assert_called_once_with(args)
            load.assert_called_once_with(transformer, source)
            self.assertEqual((checkpoint, metadata_path), (output, checkpoint_metadata_path(output)))
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved.keys(), transformer.state_dict().keys())
            for name, value in transformer.state_dict().items():
                torch.testing.assert_close(saved[name], value, rtol=0, atol=0)
            metadata = read_checkpoint_metadata(output)
            self.assertEqual(metadata["camera_pose_alignment"], pose)
            eval_args = parse_eval_args(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--model_id",
                    model_id,
                    "--save_dir",
                    str(self.root / "eval"),
                    "--render_name",
                    "round-trip",
                    "--split_path",
                    str(self.root / "split.json"),
                    "--dl3dv_dir",
                    str(self.root / "dataset"),
                    "--prompt_dir",
                    str(self.root / "prompts"),
                ]
            )
            self.assertEqual(eval_args.camera_pose_alignment, pose)

    def test_failed_weight_export_preserves_previous_weights_and_metadata(self):
        from model_training.utils import train_utils

        args = self._export_args("--overwrite")
        args.output_pt.parent.mkdir(parents=True)
        torch.save({"weight": torch.tensor([1.0])}, args.output_pt)
        metadata_path = self._stamp(args.output_pt, camera_pose_alignment="mean")
        previous = (args.output_pt.read_bytes(), metadata_path.read_bytes())

        def fail_save(state, path):
            path.write_bytes(b"partial")
            raise OSError("export interrupted")

        with (
            patch("model_eval.export_checkpoint_pt.create_export_transformer", return_value=torch.nn.Linear(2, 2)),
            patch.object(train_utils, "load_model_weights_from_dcp"),
            patch.object(torch, "save", fail_save),
        ):
            with self.assertRaisesRegex(OSError, "export interrupted"):
                export_checkpoint(args)
        self.assertEqual((args.output_pt.read_bytes(), metadata_path.read_bytes()), previous)
        self.assertEqual(resolve_camera_pose_alignment(args.output_pt, COSMOS3_NANO_MODEL_ID), "mean")

    def test_interrupted_export_publication_is_refused_and_can_be_replaced(self):
        from model_training.utils import train_utils, video_io

        args = self._export_args("--overwrite")
        args.output_pt.parent.mkdir(parents=True)
        torch.save({"old": torch.tensor([1.0])}, args.output_pt)
        self._stamp(args.output_pt, camera_pose_alignment="mean")
        replace = video_io.os.replace

        def fail_weights(source, destination):
            if destination == args.output_pt:
                raise OSError("publication interrupted")
            return replace(source, destination)

        with (
            patch("model_eval.export_checkpoint_pt.create_export_transformer", return_value=torch.nn.Linear(2, 2)),
            patch.object(train_utils, "load_model_weights_from_dcp"),
        ):
            with patch.object(video_io.os, "replace", fail_weights):
                with self.assertRaisesRegex(OSError, "publication interrupted"):
                    export_checkpoint(args)
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint export"):
                resolve_camera_pose_alignment(args.output_pt, COSMOS3_NANO_MODEL_ID)
            export_checkpoint(args)
        self.assertEqual(
            resolve_camera_pose_alignment(args.output_pt, COSMOS3_NANO_MODEL_ID), args.camera_pose_alignment
        )

    def test_dcp_load_requires_exact_model_keys_before_loading_weights(self):
        from model_training.utils import train_utils

        model = torch.nn.Linear(2, 2)
        original = {name: value.clone() for name, value in model.state_dict().items()}
        for case, state in (
            ("missing", {"weight": original["weight"]}),
            ("unexpected", original | {"removed_reference.weight": torch.ones(2, 2)}),
        ):
            checkpoint = self.root / case
            train_utils.dcp.save({"model": state}, checkpoint_id=str(checkpoint))
            with patch.object(train_utils.dcp, "load") as load:
                with self.assertRaisesRegex(RuntimeError, f"{case} key"):
                    train_utils.load_model_weights_from_dcp(model, checkpoint)
                load.assert_not_called()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        checkpoint = self.root / "matching"
        train_utils.dcp.save({"model": original, "optimizer": {"step": torch.tensor(1)}}, checkpoint_id=str(checkpoint))
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        train_utils.load_model_weights_from_dcp(model, checkpoint)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_accelerate_save_records_pose_alignment_in_root_and_all_model_slots(self):
        from model_training.utils import train_utils

        record = {"camera_pose_alignment": "last"}
        hooks = []
        accelerator = SimpleNamespace(is_main_process=True, register_save_state_pre_hook=hooks.append)
        train_utils._register_checkpoint_metadata_hooks(Namespace(**record), accelerator)
        (save_hook,) = hooks
        checkpoint = self.root / "checkpoint_12"
        save_hook([object(), object(), object()], [], str(checkpoint))
        for path in (checkpoint, *(checkpoint / f"pytorch_model_fsdp_{i}" for i in range(3))):
            self.assertEqual(read_checkpoint_metadata(path), record)
        accelerator.is_main_process = False
        save_hook([object()], [], str(self.root / "not-main"))
        self.assertFalse((self.root / "not-main").exists())

    def test_explicit_and_auto_resume_validate_pose_alignment_and_model_keys(self):
        from model_training.utils import train_utils

        record = {"camera_pose_alignment": "last"}
        for resume in ("auto", "explicit"):
            project = self.root / resume
            checkpoint = project / "checkpoints/checkpoint_12"
            self._stamp(checkpoint, **record)
            (checkpoint / "checkpoint_complete").touch()
            models = [torch.nn.Linear(2, 2) for _ in range(3)]
            slots = [checkpoint / f"pytorch_model_fsdp_{i}" for i in range(3)]
            for slot, model in zip(slots, models, strict=True):
                self._stamp(slot, **record)
                train_utils.dcp.save({"model": model.state_dict()}, checkpoint_id=str(slot))
            args = Namespace(
                model_id=COSMOS3_NANO_MODEL_ID,
                resume_from_checkpoint="auto" if resume == "auto" else str(checkpoint),
                **record,
            )
            accelerator = SimpleNamespace(is_main_process=True, project_dir=project, _models=models, load_state=Mock())
            with patch.object(train_utils.dist, "is_initialized", return_value=False):
                self.assertEqual(train_utils.load_checkpoint_state(args, accelerator), 12)
                accelerator.load_state.assert_called_once_with(str(checkpoint))
                for damaged_path in (checkpoint, slots[1], slots[2]):
                    accelerator.load_state.reset_mock()
                    self._stamp(damaged_path, camera_pose_alignment="mean")
                    with self.assertRaisesRegex(ValueError, _RECORDED_MEAN_CONFLICT):
                        train_utils.load_checkpoint_state(args, accelerator)
                    accelerator.load_state.assert_not_called()
                    self._stamp(damaged_path, **record)
                accelerator.load_state.reset_mock()
                models[1] = torch.nn.Linear(2, 2, bias=False)
                with self.assertRaisesRegex(RuntimeError, "unexpected key"):
                    train_utils.load_checkpoint_state(args, accelerator)
                accelerator.load_state.assert_not_called()

    def test_non_fsdp_explicit_and_auto_resume_round_trip_state(self):
        from model_training.utils import train_utils

        record = {"camera_pose_alignment": "last"}
        formats = ((True, "model.safetensors"), (False, "pytorch_model.bin"))
        for model_id, (safe_serialization, filename) in product(EXPORT_MODELS, formats):
            with (
                self.subTest(model_id=model_id, filename=filename),
                patch.dict("os.environ", {"ACCELERATE_USE_CPU": "true"}),
                patch.object(AcceleratorState, "_shared_state", {}),
                patch.object(PartialState, "_shared_state", {}),
                patch.object(GradientState, "_shared_state", {}),
            ):
                project = self.root / f"{model_id.split('/')[-1]}-{safe_serialization}"
                args = Namespace(
                    model_id=model_id,
                    project_dir=project,
                    gradient_accumulation_steps=1,
                    shard_size=None,
                    max_checkpoints=2,
                    log_with="none",
                    process_group_timeout_minutes=60,
                    seed=0,
                    **record,
                )
                accelerator = train_utils.get_accelerator(args, run_id=None)
                self.assertEqual(accelerator.distributed_type, DistributedType.NO)
                model = torch.nn.Linear(2, 2)
                optimizer = torch.optim.AdamW(model.parameters())
                model, optimizer = accelerator.prepare(model, optimizer)
                accelerator.backward(model(torch.ones(1, 2)).sum())
                optimizer.step()
                expected = {name: value.clone() for name, value in model.state_dict().items()}
                expected_optimizer = copy.deepcopy(optimizer.state_dict())
                accelerator.step = 12
                accelerator.project_configuration.iteration = 12
                checkpoint = Path(accelerator.save_state(safe_serialization=safe_serialization))
                train_utils.mark_checkpoint_complete(project, 12, accelerator)
                self.assertTrue((checkpoint / filename).is_file())
                slot = checkpoint / "pytorch_model_fsdp_0"
                self.assertTrue(checkpoint_metadata_path(slot).is_file())
                self.assertFalse((slot / ".metadata").exists())

                for resume in ("auto", "explicit"):
                    with self.subTest(resume=resume):
                        args.resume_from_checkpoint = "auto" if resume == "auto" else str(checkpoint)
                        self._stamp(checkpoint, **(record | {"camera_pose_alignment": "mean"}))
                        with patch.object(accelerator, "load_state") as load:
                            with self.assertRaisesRegex(ValueError, _RECORDED_MEAN_CONFLICT):
                                train_utils.load_checkpoint_state(args, accelerator)
                            load.assert_not_called()
                        self._stamp(checkpoint, **record)
                        with torch.no_grad():
                            for parameter in model.parameters():
                                parameter.zero_()
                        optimizer.state.clear()
                        accelerator.step = 0
                        self.assertEqual(train_utils.load_checkpoint_state(args, accelerator), 12)
                        self.assertEqual(accelerator.step, 12)
                        torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
                        torch.testing.assert_close(optimizer.state_dict(), expected_optimizer, rtol=0, atol=0)


@pytest.mark.parametrize(
    "updates,slot",
    [(n, "pytorch_model_fsdp_0") for n in (0, 99, 100, 102)] + [(n, "pytorch_model_fsdp_ema") for n in (100, 102)],
)
def test_retired_teacher_forcing_checkpoints_remain_exportable(tmp_path, monkeypatch, updates, slot):
    import torch.distributed.checkpoint as dcp

    from model_training.utils.ema import ModelEMA
    from tests._helpers import _EMATestAccelerator, _EMATinyModel

    # Reproduce the tag's checkpoint schema without importing its retired trainer.
    model = _EMATinyModel(2, 2)
    ema = ModelEMA(
        model,
        _EMATestAccelerator(),
        lambda: updates,
        ema_weight=0.99,
        ema_start_step=100,
        initialize_after_updates=100,
        training_stage="long_ar",
        camera_pose_alignment="mean",
    )
    for update in range(1, updates + 1):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float(update))
        ema.update()
    dcp.save({"model": model.state_dict()}, checkpoint_id=tmp_path / "pytorch_model_fsdp_0")
    ema.save_checkpoint(tmp_path)
    args = parse_args(
        [
            "--checkpoint_dir",
            str(tmp_path / slot),
            "--output_pt",
            str(tmp_path / "export.pt"),
            "--run_id",
            "legacy-tf",
            "--checkpoint",
            str(updates),
            "--slot",
            slot,
        ]
    )
    monkeypatch.setattr("model_eval.export_checkpoint_pt.create_export_transformer", lambda args: _EMATinyModel(2, 2))
    output, record_path = export_checkpoint(args)
    record = json.loads(record_path.read_text())
    expected = ema if slot.endswith("_ema") else model
    assert record["training_stage"] == "long_ar"
    assert record["weight_type"] == ("ema" if slot.endswith("_ema") else "raw")
    state = torch.load(output, weights_only=True)
    for name, parameter in zip(model.state_dict(), expected.parameters(), strict=True):
        torch.testing.assert_close(state[name], parameter, rtol=0, atol=0)
