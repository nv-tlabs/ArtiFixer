import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import wandb

import model_eval.export_checkpoint_pt as export_module
import model_eval.validation as validation_module
import model_training.distillation as distillation_entrypoint
import model_training.net.wan_transformer as transformer_module
import model_training.pipeline.pipeline_base as pipeline_module
import model_training.train as train_entrypoint
import model_training.trainers.trainer_base as trainer_base_module
import model_training.utils.train_utils as train_utils_module
import model_training.utils.video_io as video_io_module
from model_eval.generation import SamplingConfig
from model_eval.validation import EvaluationRunner
from model_eval.validation_config import EvaluationConfig, EvaluationContext
from model_eval.validation_scoring import ValidationRecorder
from model_eval.validation_tasks import ValidationTask, ValidationTaskSpec
from model_training.arguments import get_common_opts
from model_training.common.context_parallel import (
    _merge_intervals,
    all_gather_cat_with_grad,
    frame_block_causal_mask,
    latent_range_to_raw_frame_slice,
    validate_latent_frame_layout,
)
from model_training.data.dl3dv_train import DL3DVPairedDatasetTrain
from model_training.data.utils import compute_camera_conditioning
from model_training.net import attention_config
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.pipeline.wan_pipeline import ArtifixerPipeline
from model_training.trainers.context_parallel import ContextParallelTrainer
from model_training.trainers.dmd import DMD
from model_training.trainers.trainer_base import TrainerBase
from model_training.utils.ema import DTensorFastEmaModelUpdater
from tests._helpers import (
    REPO_ROOT,
    _advance_dmd_ema,
    _advance_ema_stage,
    _ema_stage_trainer,
    _EMATinyModel,
    _FakeCPProperties,
    _small_dmd_ema_trainer,
)
from tests._teacher_forcing_reference import teacher_forcing_block_mask


class _FakeCPTransformer(_FakeCPProperties, SimpleNamespace):
    pass


class _StubTrainer(TrainerBase):
    """Concrete TrainerBase with no-op abstract methods, for base-class validation tests."""

    def training_batch(self, data):
        raise NotImplementedError

    def to_accumulate(self):
        return []

    def on_before_optimizer_step(self):
        pass

    def on_before_zero_grad(self):
        pass


class _CPValFakeMesh:
    """DeviceMesh stand-in for the CP-validation environment."""

    def __init__(self, device_type, ranks):
        self.ranks = ranks

    def size(self):
        return len(self.ranks)

    def get_group(self):
        return "cp"


class _CPValAttn:
    heads = 40
    processor = object()


class _CPValBlock:
    attn1 = _CPValAttn()


class _CPValTransformer(_FakeCPProperties):
    frames_per_block = None
    blocks = [_CPValBlock()]
    _cp_mesh = None

    def enable_context_parallel(self, mesh):
        self._cp_mesh = mesh

    def disable_context_parallel(self):
        self._cp_mesh = None


def _validation_runner(pipe, *, halo=0):
    runner = EvaluationRunner(
        pipe,
        EvaluationConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            validation_metrics=False,
            vae_chunk_halo_latent_frames=halo,
            sampling=SamplingConfig(num_inference_steps=2),
        ),
        EvaluationContext(torch.device("cpu")),
    )
    runner.recorder = ValidationRecorder(runner.config, runner.context, 4)
    return runner


def _patch_cp_validation_dist(monkeypatch, all_gather, world_size=2):
    """Fake an initialized 2-rank process group for EvaluationRunner CP paths."""
    monkeypatch.setattr(validation_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(validation_module.dist, "get_world_size", lambda group=None: world_size)
    monkeypatch.setattr(validation_module.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(validation_module.dist, "new_group", lambda backend=None, timeout=None: object())
    monkeypatch.setattr(validation_module.dist, "destroy_process_group", lambda group: None)
    monkeypatch.setattr(validation_module.dist, "all_gather", all_gather)
    monkeypatch.setattr(validation_module, "DeviceMesh", _CPValFakeMesh)
    monkeypatch.setattr(validation_module, "barrier_if_distributed", lambda group=None: None)


def test_distillation_parser_accepts_context_parallel_smoke_args():
    args = distillation_entrypoint.build_parser().parse_args(
        [
            "--project_dir",
            "/tmp/s3",
            "--base_checkpoint_dir",
            "/tmp/s2/checkpoints/checkpoint_2/pytorch_model_fsdp_0",
            "--base_checkpoint_dir_critic",
            "/tmp/s2/checkpoints/checkpoint_2/pytorch_model_fsdp_0",
            "--split_path",
            "/tmp/split.json",
            "--dl3dv_dir",
            "/tmp/dl3dv",
            "--prompt_dir",
            "/tmp/prompts",
            "--context_parallel_size",
            "2",
            "--shard_size",
            "2",
            "--frames_per_block",
            "4",
            "--local_attn_size",
            "8",
            "--sink_size",
            "4",
            "--max_validation_tasks",
            "1",
        ]
    )
    assert args.context_parallel_size == 2
    assert args.shard_size == 2
    assert args.max_validation_tasks == 1


def test_distillation_defaults_fit_the_training_attention_window():
    args = distillation_entrypoint.build_parser().parse_args(
        [
            "--project_dir",
            "/tmp/s3",
            "--base_checkpoint_dir",
            "/tmp/s2/checkpoints/checkpoint_2/pytorch_model_fsdp_0",
            "--base_checkpoint_dir_critic",
            "/tmp/s1/checkpoints/checkpoint_2/pytorch_model_fsdp_0",
            "--split_path",
            "/tmp/split.json",
            "--dl3dv_dir",
            "/tmp/dl3dv",
            "--prompt_dir",
            "/tmp/prompts",
        ]
    )

    shared_defaults = {action.dest: action.default for action in get_common_opts()._actions}
    assert args.model_id == shared_defaults["model_id"]
    assert args.num_frames == shared_defaults["num_frames"]
    assert args.frames_per_block == 7
    assert args.local_attn_size == 21
    assert args.sink_size == 7
    assert args.context_parallel_size == 1
    pipeline_module.assert_training_rollout_fits_window(
        (args.num_frames - 1) // 4 + 1,
        args.local_attn_size // args.frames_per_block,
        args.frames_per_block,
        training=True,
    )


def test_stage1_entry_point_keeps_shared_training_defaults():
    """model_training.train does not override the shared training defaults;
    long-clip CP runs opt in via flags."""
    args = train_entrypoint.build_parser().parse_args(
        [
            "--project_dir",
            "/tmp/s1",
            "--split_path",
            "/tmp/split.json",
            "--dl3dv_dir",
            "/tmp/dl3dv",
            "--prompt_dir",
            "/tmp/prompts",
        ]
    )
    shared_defaults = {action.dest: action.default for action in get_common_opts()._actions}

    assert args.model_id == shared_defaults["model_id"]
    assert args.num_frames == shared_defaults["num_frames"]
    assert args.context_parallel_size == 1
    assert not hasattr(args, "prompt_num_frames")


def test_common_training_options_live_in_shared_parser():
    parser = get_common_opts()
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert "--max_validation_tasks" in option_strings
    assert "--context_parallel_size" in option_strings
    assert "--vae_chunk_halo_latent_frames" in option_strings


def test_validate_latent_frame_layout_checks_cp_block_shards():
    latent_frames = validate_latent_frame_layout(
        num_frames=1533,
        temporal_scale=4,
        frames_per_block=8,
        context_parallel_size=8,
        require_cp_rank_frames_per_block=True,
    )

    assert latent_frames == 384
    with pytest.raises(AssertionError, match="CP shard has"):
        validate_latent_frame_layout(
            num_frames=1533,
            temporal_scale=4,
            frames_per_block=8,
            context_parallel_size=32,
            require_cp_rank_frames_per_block=True,
        )


def test_validate_latent_frame_layout_rejects_unaligned_raw_frames():
    with pytest.raises(AssertionError, match="num_frames - 1"):
        validate_latent_frame_layout(
            num_frames=1532,
            temporal_scale=4,
            frames_per_block=8,
            context_parallel_size=1,
        )


def test_merge_intervals_clamps_before_filtering_empty_ranges():
    merged = _merge_intervals([(10, 20), (-10, -1), (-2, 2), (2, 5)], total_length=5)

    assert merged == [(0, 5)]


def test_pipeline_script_passes_stage3_context_parallel_args():
    text = (REPO_ROOT / "tests/test_pipeline.sh").read_text()
    assert "--context_parallel_size $CONTEXT_PARALLEL_SIZE --shard_size $SHARD_SIZE" in text
    assert "--local_attn_size $LOCAL_ATTN_SIZE --sink_size $SINK_SIZE" in text


def test_pipeline_script_uses_diffusion_forcing_without_context_parallel():
    source = (REPO_ROOT / "tests" / "test_pipeline.sh").read_text()
    stage2 = source.split("# STAGE 2:", 1)[1].split("# STAGE 3:", 1)[0]
    assert "$LAUNCH model_training.diffusion_forcing" in stage2
    assert "--context_parallel_size 1 --shard_size $SHARD_SIZE" in stage2


def test_pipeline_script_defaults_to_single_gpu_s1_smoke_with_explicit_cp_opt_in():
    text = (REPO_ROOT / "tests/test_pipeline.sh").read_text()
    assert ': "${NUM_GPUS:=1}"' in text
    assert "NUM_FRAMES=81" in text
    assert "NUM_FRAMES=125" in text
    assert 'CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-8}"' in text
    assert 'SHARD_SIZE="$NUM_GPUS"' in text
    assert "S1_CONTEXT_PARALLEL_SIZE=1" in text
    assert 'S1_SHARD_SIZE="$NUM_GPUS"' in text
    assert "NUM_GPUS % S1_CONTEXT_PARALLEL_SIZE" in text


def test_auto_resume_skips_malformed_checkpoint_directories(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "checkpoint_tmp").mkdir(parents=True)
    valid = checkpoints / "checkpoint_7"
    valid.mkdir()
    (valid / "checkpoint_complete").touch()
    loaded = []
    accelerator = SimpleNamespace(
        is_main_process=True,
        project_dir=tmp_path,
        load_state=lambda path: loaded.append(Path(path)),
    )
    accelerator._models = []
    args = SimpleNamespace(
        resume_from_checkpoint="auto",
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        camera_pose_alignment="mean",
    )

    step = train_utils_module.load_checkpoint_state(args, accelerator)

    assert step == 7
    assert loaded == [valid]


def test_training_dataset_reuses_cp_source_fetch_for_group_members():
    dataset = DL3DVPairedDatasetTrain.__new__(DL3DVPairedDatasetTrain)
    dataset.context_parallel_size = 4
    dataset.num_frames = 61
    dataset.vae_chunk_halo_latent_frames = 1
    dataset.random_seed = 123
    dataset.iter_plan = torch.tensor([[2, 1], [3, 2], [6, 5], [12, 8], [2, 2]], dtype=torch.long)
    dataset.data_perm = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    dataset.data = [object()] * 20
    # __getitem__ consults the pose-jump filter before fetching; every stub
    # scene is viable here.
    dataset.can_supply_same_segment_neighbors = lambda *args: True

    def fake_get_item_inner(data_idx, chosen_split, num_neighbors, *, generator=None, temporal_slice=None):
        return {
            "data_idx": int(data_idx),
            "chosen_split": int(chosen_split),
            "num_neighbors": int(num_neighbors),
            "rng_value": int(torch.randint(0, 1000, (1,), generator=generator).item()),
            "temporal_slice": temporal_slice,
        }

    dataset.get_item_inner = fake_get_item_inner

    rank0 = dataset[0]
    rank1 = dataset[1]
    rank3 = dataset[3]
    next_group = dataset[4]

    assert (rank0["data_idx"], rank0["chosen_split"], rank0["num_neighbors"]) == (10, 2, 1)
    assert (rank1["data_idx"], rank1["chosen_split"], rank1["num_neighbors"]) == (10, 2, 1)
    assert (rank3["data_idx"], rank3["chosen_split"], rank3["num_neighbors"]) == (10, 2, 1)
    assert (next_group["data_idx"], next_group["chosen_split"], next_group["num_neighbors"]) == (11, 3, 2)
    assert rank0["rng_value"] == rank1["rng_value"] == rank3["rng_value"]
    assert rank0["temporal_slice"]["latent"] == (0, 4)
    assert rank0["temporal_slice"]["conditioning_raw"] == (0, 13)
    assert rank0["temporal_slice"]["vae_raw"] == (0, 13, 0, 4)
    assert rank1["temporal_slice"]["latent"] == (4, 8)
    assert rank1["temporal_slice"]["conditioning_raw"] == (13, 29)
    assert rank1["temporal_slice"]["vae_raw"] == (8, 29, 2, 6)


def test_training_dataset_cp_disabled_preserves_global_fetch_call():
    dataset = DL3DVPairedDatasetTrain.__new__(DL3DVPairedDatasetTrain)
    dataset.context_parallel_size = 1
    dataset.random_seed = 123
    dataset.iter_plan = torch.tensor([[2, 1]], dtype=torch.long)
    dataset.data_perm = torch.tensor([10], dtype=torch.long)
    dataset.data = [object()] * 20
    # __getitem__ consults the pose-jump filter before fetching; every stub
    # scene is viable here.
    dataset.can_supply_same_segment_neighbors = lambda *args: True

    def fake_get_item_inner(data_idx, chosen_split, num_neighbors, *, generator=None, temporal_slice=None):
        return generator, temporal_slice

    dataset.get_item_inner = fake_get_item_inner

    generator, temporal_slice = dataset[0]
    assert generator.initial_seed() == 123
    assert temporal_slice is None


def test_training_dataset_cp_disabled_routes_each_plan_row_exactly_once():
    # cp=1 is the production stage-1 path: every fetch index must consume its
    # own iter_plan row exactly once under Accelerate's round-robin sharding
    # and walk data_perm cyclically for scene selection.
    num_processes = 4
    num_fetches = 16
    num_scenes = 3
    dataset = DL3DVPairedDatasetTrain.__new__(DL3DVPairedDatasetTrain)
    dataset.context_parallel_size = 1
    dataset.random_seed = 123
    dataset.num_fetches = num_fetches
    dataset.num_group_fetches = num_fetches
    dataset.iter_plan = torch.stack(
        [torch.arange(num_fetches, dtype=torch.long), torch.arange(num_fetches, dtype=torch.long) + 100],
        dim=1,
    )
    dataset.data_perm = torch.tensor([7, 5, 9], dtype=torch.long)
    dataset.data = [object()] * num_scenes
    # __getitem__ consults the pose-jump filter before fetching; every stub
    # scene is viable here.
    dataset.can_supply_same_segment_neighbors = lambda *args: True

    calls = []

    def fake_get_item_inner(data_idx, chosen_split, num_neighbors, *, generator=None, temporal_slice=None):
        assert generator.initial_seed() == 123 + chosen_split
        assert temporal_slice is None
        calls.append((int(data_idx), int(chosen_split), int(num_neighbors)))
        return {}

    dataset.get_item_inner = fake_get_item_inner

    # Accelerate's BatchSamplerShard hands rank r the indices fetch * W + r.
    for fetch_index in range(num_fetches // num_processes):
        for rank in range(num_processes):
            dataset[fetch_index * num_processes + rank]

    assert sorted(chosen_split for _, chosen_split, _ in calls) == list(range(num_fetches))
    assert [num_neighbors for _, _, num_neighbors in calls] == [idx + 100 for idx in range(num_fetches)]
    assert [data_idx for data_idx, _, _ in calls[: 2 * num_scenes]] == [7, 5, 9, 7, 5, 9]


def test_training_dataset_samples_one_plan_row_per_cp_group_fetch():
    dataset = DL3DVPairedDatasetTrain.__new__(DL3DVPairedDatasetTrain)
    dataset.num_fetches = 12
    dataset.context_parallel_size = 4
    dataset.num_group_fetches = 3
    dataset.available_splits = [2, 4]
    dataset._intervals = torch.tensor([(1, 2), (3, 4)], dtype=torch.long)
    dataset.data = [object()] * 5

    dataset.set_random_splits(torch.device("cpu"), seed=123)

    assert dataset.iter_plan.shape == (3, 2)
    assert dataset.data_perm.shape == (5,)


def test_get_train_dataloader_keeps_cp_group_indices_adjacent(monkeypatch):
    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured["dataset_kwargs"] = kwargs

        def set_random_splits(self, device, seed):
            captured["set_random_splits"] = (device, seed)

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["dataloader_kwargs"] = kwargs

    monkeypatch.setattr(train_utils_module, "DL3DVPairedDatasetTrain", FakeDataset)
    monkeypatch.setattr(torch.utils.data, "DataLoader", FakeDataLoader)
    args = SimpleNamespace(
        max_iterations=2,
        gradient_accumulation_steps=3,
        split_path=Path("/tmp/split.json"),
        dl3dv_dir=Path("/tmp/dl3dv"),
        prompt_dir=Path("/tmp/prompts"),
        num_frames=1533,
        dataset_scaling_factor=1.0,
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        context_parallel_size=8,
        vae_chunk_halo_latent_frames=28,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
        seed=123,
        camera_pose_alignment="mean",
    )
    accelerator = SimpleNamespace(num_processes=16, device=torch.device("cpu"), is_main_process=True)

    train_utils_module.get_train_dataloader(args, accelerator, frames_per_block=8)

    assert captured["dataset_kwargs"]["num_fetches"] == 96
    assert captured["dataset_kwargs"]["context_parallel_size"] == 8
    assert captured["dataset_kwargs"]["vae_chunk_halo_latent_frames"] == 28
    assert captured["set_random_splits"] == (torch.device("cpu"), 123)
    assert captured["dataloader_kwargs"]["shuffle"] is False


def test_accelerate_batch_sampler_preserves_cp_group_fetch_adjacency():
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler, SequentialSampler

    num_processes = 8
    context_parallel_size = 4
    sampler = BatchSampler(SequentialSampler(range(64)), batch_size=1, drop_last=False)
    per_rank_indices = []
    for process_index in range(num_processes):
        # Production accelerator.prepare defaults: even_batches=True. The
        # dataset length (num_fetches) is always a multiple of num_processes,
        # so no wrap-around batches are inserted; assert that below.
        shard = BatchSamplerShard(
            sampler,
            num_processes=num_processes,
            process_index=process_index,
            split_batches=False,
            even_batches=True,
        )
        per_rank_indices.append([batch[0] for batch in shard])

    consumed = [index for rank_indices in per_rank_indices for index in rank_indices]
    assert sorted(consumed) == list(range(64))

    for fetch_index in range(len(per_rank_indices[0])):
        for group_start in range(0, num_processes, context_parallel_size):
            logical_fetches = {
                per_rank_indices[rank][fetch_index] // context_parallel_size
                for rank in range(group_start, group_start + context_parallel_size)
            }
            assert len(logical_fetches) == 1


def test_get_train_dataloader_requires_world_size_divisible_by_cp(monkeypatch):
    class FakeDataset:
        def __init__(self, **kwargs):
            pass

        def set_random_splits(self, device, seed):
            pass

    monkeypatch.setattr(train_utils_module, "DL3DVPairedDatasetTrain", FakeDataset)
    args = SimpleNamespace(
        max_iterations=1,
        gradient_accumulation_steps=1,
        split_path=Path("/tmp/split.json"),
        dl3dv_dir=Path("/tmp/dl3dv"),
        prompt_dir=Path("/tmp/prompts"),
        num_frames=1533,
        dataset_scaling_factor=1.0,
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        context_parallel_size=8,
        vae_chunk_halo_latent_frames=28,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
        seed=123,
    )
    accelerator = SimpleNamespace(num_processes=10, device=torch.device("cpu"), is_main_process=True)

    with pytest.raises(AssertionError, match="num_processes"):
        train_utils_module.get_train_dataloader(args, accelerator, frames_per_block=8)


def test_camera_conditioning_latent_slice_matches_full_camera_slice():
    intrinsics = {"fl_x": 1.0, "fl_y": 1.0, "cx": 0.5, "cy": 0.5, "w": 1.0, "h": 1.0}
    c2ws_world = torch.eye(4).unsqueeze(0).repeat(9, 1, 1)
    c2ws_world[:, 0, 3] = torch.arange(9, dtype=torch.float32)
    neighbor_c2ws_world = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)

    full = compute_camera_conditioning(
        intrinsics=intrinsics,
        c2ws_world=c2ws_world,
        neighbor_c2ws_world=neighbor_c2ws_world,
        scale=1.0,
        image_height=1,
        image_width=1,
    )
    sliced = compute_camera_conditioning(
        intrinsics=intrinsics,
        c2ws_world=c2ws_world,
        neighbor_c2ws_world=neighbor_c2ws_world,
        scale=1.0,
        image_height=1,
        image_width=1,
        latent_slice=(1, 3),
    )

    torch.testing.assert_close(sliced["w2cs"], full["w2cs"][1:3])
    torch.testing.assert_close(sliced["Ks"], full["Ks"][1:3])
    torch.testing.assert_close(sliced["camera_rays"], full["camera_rays"][1:3])


def test_context_parallel_mixin_normalises_collated_cp_data_metadata():
    trainer = ContextParallelTrainer.__new__(ContextParallelTrainer)
    data = {
        "_cp_data_sharded": torch.tensor([True]),
        "_cp_vae_crop": torch.tensor([[2, 10]]),
        "_cp_vae_raw_start": torch.tensor([24]),
        "_cp_conditioning_raw_start": torch.tensor([29]),
        "_cp_global_raw_frames": torch.tensor([61]),
    }

    assert trainer._is_data_sharded_for_cp(data)
    normalised = trainer._normalise_cp_sharded_batch_metadata(data)

    assert normalised["_cp_data_sharded"] is True
    assert normalised["_cp_vae_crop"] == (2, 10)
    assert normalised["_cp_vae_raw_start"] == 24
    assert normalised["_cp_conditioning_raw_start"] == 29
    assert normalised["_cp_global_raw_frames"] == 61


def test_process_inputs_accepts_data_sharded_cp_batch_without_resync():
    class RecordingPipe:
        def __init__(self):
            self.vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scale_factor_temporal=1))
            self.encoded_shapes = []

        def encode_video_frames(self, frames):
            self.encoded_shapes.append(tuple(frames.shape))
            return frames.permute(0, 2, 1, 3, 4).contiguous()

        def encode_neighbors(self, frames):
            return torch.zeros(frames.shape[0], 1, frames.shape[1], 1, 1)

    trainer = ContextParallelTrainer.__new__(ContextParallelTrainer)
    trainer.args = SimpleNamespace(dropout_rate=0.0, vae_chunk_halo_latent_frames=1)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.pipe = RecordingPipe()
    trainer._uses_context_parallel = lambda: True
    trainer._is_cp_source_rank = lambda: True

    def broadcast(tensor):
        if tensor.numel() == 3:
            tensor.fill_(1.0)
        else:
            tensor.zero_()
        return tensor

    trainer._broadcast_cp_tensor = broadcast

    data = {
        "rgb_gt": torch.zeros(1, 5, 3, 1, 1),
        "rgb_rendered": torch.ones(1, 5, 3, 1, 1),
        "opacity": torch.ones(1, 3, 1, 1),
        "encoded_prompt": torch.ones(1, 4, 8),
        "rgb_neighbors": torch.zeros(1, 2, 3, 1, 1),
        "neighbor_w2cs": torch.zeros(1, 2, 4, 4),
        "neighbor_Ks": torch.zeros(1, 2, 3, 3),
        "camera_rays": torch.zeros(1, 3, 1, 1, 6),
        "w2cs": torch.zeros(1, 3, 4, 4),
        "Ks": torch.zeros(1, 3, 3, 3),
        "_cp_data_sharded": torch.tensor([True]),
        "_cp_vae_crop": torch.tensor([[1, 4]]),
        "_cp_vae_raw_start": torch.tensor([8]),
        "_cp_conditioning_raw_start": torch.tensor([13]),
        "_cp_global_raw_frames": torch.tensor([61]),
    }

    processed = trainer.process_inputs(data)

    assert processed["_cp_vae_crop"] == (1, 4)
    assert trainer.pipe.encoded_shapes == [(1, 5, 3, 1, 1)]
    assert processed["condition"].shape == (1, 3, 3, 1, 1)
    torch.testing.assert_close(processed["opacity"], data["opacity"])

    # The no-resync guarantee is structural: batches missing the dataset's
    # pre-sharded metadata are rejected outright under active CP.
    unsharded = {key: value for key, value in data.items() if key != "_cp_data_sharded"}
    with pytest.raises(AssertionError, match="pre-sharded"):
        trainer.process_inputs(unsharded)


def test_sparse_video_decode_preserves_requested_frame_order(monkeypatch):
    yielded = []

    def fake_decoder():
        for index in range(5):
            yielded.append(index)
            yield index

    to_rgb = lambda index: torch.full((1, 1, 3), index, dtype=torch.uint8).numpy()
    monkeypatch.setattr(video_io_module, "_start_decoder", lambda path: (fake_decoder(), to_rgb))

    decoded = video_io_module.decode_video_to_frames(b"unused", [3, 1, 3])

    assert decoded[:, 0, 0, 0].tolist() == [3, 1, 3]
    # The decode stream is not pulled past the highest requested frame.
    assert yielded == [0, 1, 2, 3]


def test_context_parallel_mixin_reads_cp_rank_from_wrapped_transformer():
    trainer = ContextParallelTrainer.__new__(ContextParallelTrainer)
    trainer.pipe = SimpleNamespace(transformer=SimpleNamespace(module=_FakeCPTransformer(_cp_rank=1, _cp_world_size=4)))

    assert trainer._cp_rank_size() == (1, 4)


def test_context_parallel_mixin_broadcast_uses_cp_mesh_source(monkeypatch):
    class FakeMesh:
        mesh = torch.tensor([4, 5])

        def size(self):
            return 2

        def get_group(self):
            return "cp_group"

    trainer = ContextParallelTrainer.__new__(ContextParallelTrainer)
    trainer.pipe = SimpleNamespace(
        transformer=SimpleNamespace(module=_FakeCPTransformer(_cp_rank=0, _cp_world_size=2, _cp_mesh=FakeMesh()))
    )
    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 4)
    monkeypatch.setattr(dist, "broadcast", lambda tensor, src, group: calls.append((src, group)))

    tensor = torch.ones(1)
    assert trainer._broadcast_cp_tensor(tensor) is tensor

    assert calls == [(4, "cp_group")]


def test_context_parallel_mixin_rejects_mesh_rank_mismatch(monkeypatch):
    class FakeMesh:
        mesh = torch.tensor([4, 5])

        def size(self):
            return 2

        def get_group(self):
            return "cp_group"

    trainer = ContextParallelTrainer.__new__(ContextParallelTrainer)
    trainer.pipe = SimpleNamespace(
        transformer=SimpleNamespace(module=_FakeCPTransformer(_cp_rank=1, _cp_world_size=2, _cp_mesh=FakeMesh()))
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 6)
    monkeypatch.setattr(dist, "broadcast", lambda tensor, src, group: None)

    with pytest.raises(AssertionError, match="CP rank"):
        trainer._broadcast_cp_tensor(torch.ones(1))


def test_context_parallel_process_inputs_uses_broadcast_dropout_decisions():
    class RecordingPipe:
        def __init__(self):
            self.vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scale_factor_temporal=1))
            self.encoded_rendered = None

        def encode_video_frames(self, frames):
            self.encoded_rendered = frames.clone()
            return frames[:, :, :1]

        def encode_neighbors(self, frames):
            return torch.zeros(frames.shape[0], 1, frames.shape[1], 1, 1)

    trainer = ContextParallelTrainer.__new__(ContextParallelTrainer)
    trainer.args = SimpleNamespace(dropout_rate=0.5, vae_chunk_halo_latent_frames=0)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.pipe = RecordingPipe()
    trainer._uses_context_parallel = lambda: True
    trainer._is_cp_source_rank = lambda: False
    broadcasts = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2])]

    def broadcast(tensor):
        tensor.copy_(broadcasts.pop(0).to(device=tensor.device, dtype=tensor.dtype))
        return tensor

    trainer._broadcast_cp_tensor = broadcast

    rendered = torch.arange(1 * 5 * 3 * 1 * 1, dtype=torch.float32).view(1, 5, 3, 1, 1)
    data = {
        "rgb_gt": torch.zeros(1, 5, 3, 1, 1),
        "rgb_rendered": rendered,
        "opacity": torch.ones(1, 5, 1, 1),
        "encoded_prompt": torch.ones(1, 4, 8),
        "rgb_neighbors": torch.zeros(1, 2, 3, 1, 1),
        "neighbor_w2cs": torch.zeros(1, 2, 4, 4),
        "neighbor_Ks": torch.zeros(1, 2, 3, 3),
        "camera_rays": torch.zeros(1, 5, 1, 1, 6),
        "w2cs": torch.zeros(1, 5, 4, 4),
        "Ks": torch.zeros(1, 5, 3, 3),
        "_cp_global_raw_frames": 5,
        "_cp_vae_raw_start": 1,
        "_cp_conditioning_raw_start": 1,
        "_cp_data_sharded": True,
        "_cp_vae_crop": (0, 1),
    }

    processed = trainer.process_inputs(data)

    assert broadcasts == []
    assert trainer.pipe.encoded_rendered is not None
    torch.testing.assert_close(trainer.pipe.encoded_rendered[:, :2], rendered[:, :2])
    torch.testing.assert_close(trainer.pipe.encoded_rendered[:, 2:], torch.zeros_like(rendered[:, 2:]))
    torch.testing.assert_close(processed["opacity"][:, :2], torch.ones(1, 2, 1, 1))
    torch.testing.assert_close(processed["opacity"][:, 2:], torch.zeros(1, 3, 1, 1))
    torch.testing.assert_close(processed["encoded_prompt"], data["encoded_prompt"])
    assert not bool(processed["ignore_neighbors"])


class _DropoutRecordingPipe:
    def __init__(self, cp_rank):
        mesh = SimpleNamespace(mesh=torch.tensor([4, 5]), size=lambda: 2, get_group=lambda: "dropout_cp_group")
        self.transformer = _FakeCPTransformer(
            _cp_rank=cp_rank or 0,
            _cp_world_size=1 if cp_rank is None else 2,
            _cp_mesh=mesh if cp_rank is not None else None,
        )
        self.vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scale_factor_temporal=1))
        self.ar_frames_per_chunk = 2
        self.video_inputs, self.neighbor_inputs, self.chunked_calls = [], [], []

    def encode_video_frames(self, frames):
        self.video_inputs.append(frames)
        return frames.permute(0, 2, 1, 3, 4) + 7.0

    def encode_video_frames_chunked_for_cp(self, frames, halo, *, gather=True):
        self.chunked_calls.append((tuple(frames.shape), halo, gather))
        return self.encode_video_frames(frames)

    def encode_neighbors(self, frames):
        self.neighbor_inputs.append(frames)
        return frames.permute(0, 2, 1, 3, 4) + 7.0


@pytest.mark.parametrize(
    ("trainer_cls", "cp_rank"),
    [
        pytest.param(_StubTrainer, None, id="base"),
        pytest.param(ContextParallelTrainer, 0, id="supervised-source"),
        pytest.param(ContextParallelTrainer, 1, id="supervised-follower"),
        pytest.param(DMD, 0, id="dmd-source"),
        pytest.param(DMD, 1, id="dmd-follower"),
    ],
)
@pytest.mark.parametrize(
    ("dropout_probs", "frames_to_drop"),
    [
        pytest.param((0.1, 0.1, 0.1), None, id="drop-all"),
        pytest.param((0.5, 0.5, 0.5), 0, id="keep-at-threshold"),
        pytest.param((0.9, 0.1, 0.9), 3, id="suffix-and-text"),
        pytest.param((0.9, 0.9, 0.1), 7, id="suffix-and-neighbors"),
    ],
)
@pytest.mark.parametrize("token_prompt", [False, True], ids=["wan", "cosmos"])
def test_training_process_inputs_preserves_dropout_policy(
    monkeypatch, trainer_cls, cp_rank, dropout_probs, frames_to_drop, token_prompt
):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(num_frames=8, dropout_rate=0.5, vae_chunk_halo_latent_frames=1)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.pipe = _DropoutRecordingPipe(cp_rank)
    trainer._zero_encode_cache = {}
    sharded_raw = trainer_cls in (ContextParallelTrainer, DMD)
    raw_start = 2 if sharded_raw and cp_rank == 1 else 0
    raw_frames = (4 if cp_rank == 0 else 6) if sharded_raw else 8
    conditioning_start = 4 * cp_rank if sharded_raw else 0
    conditioning_frames = 4 if sharded_raw else 8
    data = {
        "rgb_gt": torch.zeros(1, raw_frames, 3, 1, 1),
        "rgb_rendered": torch.arange(1, 1 + raw_frames * 3, dtype=torch.float32).view(1, raw_frames, 3, 1, 1),
        "opacity": torch.ones(1, conditioning_frames, 1, 1),
        "rgb_neighbors": torch.ones(1, 2, 3, 1, 1),
        "encoded_prompt": torch.ones(1, 3, 4),
        "neighbor_w2cs": torch.zeros(1, 2, 4, 4),
        "neighbor_Ks": torch.zeros(1, 2, 3, 3),
        "camera_rays": torch.zeros(1, conditioning_frames, 1, 1, 6),
        "w2cs": torch.zeros(1, conditioning_frames, 4, 4),
        "Ks": torch.zeros(1, conditioning_frames, 3, 3),
    }
    if token_prompt:
        data.update(
            encoded_prompt=torch.tensor([[5, 6, 7]]),
            prompt_attention_mask=torch.tensor([[1, 1, 1]]),
            encoded_negative_prompt=torch.tensor([[2, 3, 0]]),
            negative_prompt_attention_mask=torch.tensor([[1, 1, 0]]),
        )
    if sharded_raw:
        data.update(
            _cp_data_sharded=True,
            _cp_global_raw_frames=8,
            _cp_vae_raw_start=raw_start,
            _cp_conditioning_raw_start=conditioning_start,
            _cp_vae_crop=(0, 4) if cp_rank == 0 else (2, 6),
        )
    events = []

    def rand(size, *, device=None):
        assert cp_rank != 1, "CP followers must not consume dropout RNG"
        events.append(("rand", size, device))
        return torch.tensor(dropout_probs)

    def randint(low, high, size, *, device=None):
        assert cp_rank != 1, "CP followers must not consume suffix RNG"
        assert frames_to_drop is not None, "fully dropped rendering must not consume suffix RNG"
        events.append(("randint", low, high, size, device))
        return torch.tensor([frames_to_drop])

    def broadcast(tensor, src, group):
        events.append(("broadcast", tuple(tensor.shape), tensor.dtype, tensor.device, src, group))
        values = dropout_probs if tensor.numel() == 3 else [frames_to_drop]
        tensor.copy_(torch.tensor(values, dtype=tensor.dtype))

    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", randint)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 4 + (cp_rank or 0))
    monkeypatch.setattr(dist, "broadcast", broadcast)

    def all_gather(outputs, value, group):
        assert group == "dropout_cp_group"
        for output in outputs:
            output.copy_(value)

    monkeypatch.setattr(dist, "all_gather", all_gather)

    drop_rendered, drop_text, ignore_neighbors = (prob < trainer.args.dropout_rate for prob in dropout_probs)
    drop_start = 0 if drop_rendered else 8 - frames_to_drop
    rendered_before = data["rgb_rendered"].clone()
    opacity_before = data["opacity"].clone()
    expected_rendered = rendered_before.clone()
    expected_rendered[:, max(0, drop_start - raw_start) :] = 0
    expected_opacity = opacity_before.clone()
    expected_opacity[:, max(0, drop_start - conditioning_start) :] = 0
    expected_condition = expected_rendered.permute(0, 2, 1, 3, 4) + 7.0
    if sharded_raw:
        crop_start, crop_end = data["_cp_vae_crop"]
        expected_condition = expected_condition[:, :, crop_start:crop_end]
    if trainer_cls is DMD:
        expected_condition = torch.cat([expected_condition] * 2, dim=2)
        expected_opacity = torch.cat([expected_opacity] * 2, dim=1)

    first = trainer.process_inputs(data)
    processed = trainer.process_inputs(data)

    expected_events = []
    device = None if cp_rank is None else trainer.accelerator.device
    if cp_rank != 1:
        expected_events.append(("rand", 3, device))
    if cp_rank is not None:
        expected_events.append(("broadcast", (3,), torch.float32, device, 4, "dropout_cp_group"))
    if not drop_rendered:
        if cp_rank != 1:
            expected_events.append(("randint", 0, 8, (1,), device))
        if cp_rank is not None:
            expected_events.append(("broadcast", (1,), torch.long, device, 4, "dropout_cp_group"))
    assert events == expected_events * 2
    torch.testing.assert_close(data["rgb_rendered"], rendered_before)
    torch.testing.assert_close(data["opacity"], opacity_before)
    torch.testing.assert_close(trainer.pipe.video_inputs[-1], expected_rendered)
    torch.testing.assert_close(processed["condition"], expected_condition)
    torch.testing.assert_close(processed["opacity"], expected_opacity)
    expected_prompt = data["encoded_prompt"]
    if drop_text:
        expected_prompt = data["encoded_negative_prompt"] if token_prompt else torch.zeros_like(expected_prompt)
    torch.testing.assert_close(processed["encoded_prompt"], expected_prompt)
    if token_prompt:
        mask_key = "negative_prompt_attention_mask" if drop_text else "prompt_attention_mask"
        torch.testing.assert_close(processed["prompt_attention_mask"], data[mask_key])
        torch.testing.assert_close(processed["encoded_negative_prompt"], data["encoded_negative_prompt"])
        torch.testing.assert_close(processed["negative_prompt_attention_mask"], data["negative_prompt_attention_mask"])
    else:
        assert "prompt_attention_mask" not in processed
        assert "encoded_negative_prompt" not in processed
    assert bool(processed["ignore_neighbors"]) == ignore_neighbors
    expected_neighbors = torch.zeros_like(data["rgb_neighbors"]) if ignore_neighbors else data["rgb_neighbors"]
    torch.testing.assert_close(processed["neighbors_condition"], expected_neighbors.permute(0, 2, 1, 3, 4) + 7.0)
    assert len(trainer.pipe.neighbor_inputs) == (1 if ignore_neighbors else 2)
    if ignore_neighbors:
        assert first["neighbors_condition"] is processed["neighbors_condition"]
    assert len(trainer.pipe.video_inputs) == (1 if drop_rendered else 2)
    if drop_rendered and trainer_cls is not DMD:
        assert first["condition"] is processed["condition"]
    assert trainer.pipe.chunked_calls == []
    if trainer_cls is DMD:
        assert not {"rgb_gt", "rgb_rendered", "rgb_neighbors", "_cp_vae_crop"} & processed.keys()
    else:
        assert processed["rgb_gt"] is data["rgb_gt"]
    if sharded_raw and trainer_cls is not DMD:
        assert processed["_cp_vae_crop"] == data["_cp_vae_crop"]


def test_short_scene_frames_all_prompts_are_eligible_for_long_training_rollout(tmp_path):
    scene_id = "short_scene_with_all_prompts"
    subdir = "1K"
    dl3dv_dir = tmp_path / "dl3dv"
    prompt_dir = tmp_path / "prompts"
    reconstruction_path = tmp_path / f"data_{scene_id}_left_2.h5"
    split_path = tmp_path / "split.json"

    (dl3dv_dir / subdir).mkdir(parents=True)
    (prompt_dir / subdir / scene_id).mkdir(parents=True)
    reconstruction_path.touch()
    (prompt_dir / subdir / scene_id / "frames_all_stride_1.h5").touch()

    # Identity poses: no pose jumps, so the train-only pose-jump filter keeps
    # the short scene.
    transforms = {
        "frames": [
            {"file_path": f"images/{index:04d}.png", "transform_matrix": torch.eye(4).tolist()} for index in range(81)
        ]
    }
    with ZipFile(dl3dv_dir / subdir / f"{scene_id}.zip", "w") as zf:
        zf.writestr("transforms.json", json.dumps(transforms))

    split_path.write_text(
        json.dumps(
            {
                "trainval": {
                    scene_id: [
                        {
                            "subdir": subdir,
                            "hdf5_path": str(reconstruction_path),
                            "num_selected_indices": 2,
                            "scene_half": "left",
                            "scale": 1.0,
                        }
                    ]
                }
            }
        )
    )

    dataset = DL3DVPairedDatasetTrain(
        split="trainval",
        split_path=split_path,
        dl3dv_dir=dl3dv_dir,
        prompt_dir=prompt_dir,
        num_frames=1533,
        frames_per_block=8,
        num_fetches=1,
        dataset_scaling_factor=1.0,
        max_neighbors=2,
    )

    assert dataset.num_frames == 1533
    assert dataset.available_splits == (2,)
    assert len(dataset.data) == 1


def test_dataset_fails_when_split_scene_raw_zip_is_missing(tmp_path):
    valid_scene_id = "scene_with_zip"
    missing_scene_id = "scene_without_zip"
    subdir = "1K"
    dl3dv_dir = tmp_path / "dl3dv"
    prompt_dir = tmp_path / "prompts"
    split_path = tmp_path / "split.json"
    valid_reconstruction_path = tmp_path / f"data_{valid_scene_id}_left_2.h5"

    (dl3dv_dir / subdir).mkdir(parents=True)
    for scene_id in (valid_scene_id, missing_scene_id):
        scene_prompt_dir = prompt_dir / subdir / scene_id
        scene_prompt_dir.mkdir(parents=True)
        (scene_prompt_dir / "frames_all_stride_1.h5").touch()
    valid_reconstruction_path.touch()

    transforms = {"frames": [{"file_path": f"images/{index:04d}.png"} for index in range(81)]}
    with ZipFile(dl3dv_dir / subdir / f"{valid_scene_id}.zip", "w") as zf:
        zf.writestr("transforms.json", json.dumps(transforms))

    def reconstruction(scene_id: str, hdf5_path: Path) -> dict:
        return {
            "subdir": subdir,
            "hdf5_path": str(hdf5_path),
            "num_selected_indices": 2,
            "scene_half": "left",
            "scale": 1.0,
        }

    split_path.write_text(
        json.dumps(
            {
                "trainval": {
                    missing_scene_id: [
                        reconstruction(missing_scene_id, tmp_path / "missing_scene_hdf5_is_not_read.h5")
                    ],
                    valid_scene_id: [reconstruction(valid_scene_id, valid_reconstruction_path)],
                }
            }
        )
    )

    missing_zip = dl3dv_dir / subdir / f"{missing_scene_id}.zip"
    with pytest.raises(FileNotFoundError, match=f"No such file or directory: '{missing_zip}'"):
        DL3DVPairedDatasetTrain(
            split="trainval",
            split_path=split_path,
            dl3dv_dir=dl3dv_dir,
            prompt_dir=prompt_dir,
            num_frames=1533,
            frames_per_block=8,
            num_fetches=1,
            dataset_scaling_factor=1.0,
            max_neighbors=2,
        )


@pytest.fixture
def flex_block_size(monkeypatch):
    monkeypatch.setattr(attention_config, "_FLEX_KERNEL_OPTIONS", {})
    attention_config.configure_attention("cpu")
    return attention_config.flex_block_size()


def test_frame_block_causal_mask_uses_global_length_for_ulysses_cp(flex_block_size):
    block_mask = frame_block_causal_mask(
        device=torch.device("cpu"),
        num_frames=16,
        frame_seqlen=510,
        frames_per_block=2,
        block_size=flex_block_size,
    )

    assert block_mask.seq_lengths == (16 * 510, 16 * 510)

    def allowed(q_idx: int, kv_idx: int) -> bool:
        result = block_mask.mask_mod(
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(q_idx),
            torch.tensor(kv_idx),
        )
        return bool(result.item())

    second_block_query = 2 * 510
    assert allowed(second_block_query, 3 * 510 - 1)
    assert not allowed(second_block_query, 4 * 510)


@pytest.mark.parametrize("attention_heads,expected_heads", [(None, 1), (3, 3)])
def test_frame_block_causal_mask_can_materialize_per_head_shape(flex_block_size, attention_heads, expected_heads):
    block_mask = frame_block_causal_mask(
        device=torch.device("cpu"),
        num_frames=16,
        frame_seqlen=510,
        frames_per_block=2,
        block_size=flex_block_size,
        attention_heads=attention_heads,
    )

    q_blocks = (16 * 510 + flex_block_size - 1) // flex_block_size
    assert block_mask.full_kv_num_blocks.shape == (1, expected_heads, q_blocks)
    assert block_mask.full_kv_indices.shape == (1, expected_heads, q_blocks, q_blocks)
    assert block_mask.kv_num_blocks.shape == (1, expected_heads, q_blocks)
    assert block_mask.kv_indices.shape == (1, expected_heads, q_blocks, q_blocks)
    batch, head, query = torch.tensor(0), torch.tensor(expected_heads - 1), torch.tensor(2 * 510)
    assert bool(block_mask.mask_mod(batch, head, query, torch.tensor(3 * 510 - 1)))
    assert not bool(block_mask.mask_mod(batch, head, query, torch.tensor(4 * 510)))


def test_frame_block_causal_mask_scales_without_dense_predicate_construction(flex_block_size):
    block_mask = frame_block_causal_mask(
        device=torch.device("cpu"),
        num_frames=384,
        frame_seqlen=510,
        frames_per_block=8,
        block_size=flex_block_size,
    )

    assert block_mask.seq_lengths == (195840, 195840)


def test_frame_block_causal_mask_matches_int_reference_across_geometries():
    """block_end is precomputed as a q_idx-indexed tensor rather than
    recomputed from a captured block_token_count inside attention_mask,
    because validation and training can build this mask with different frame
    counts in the same process and a captured int would get promoted to a
    SymInt on the second distinct value. Checked across two geometries
    against a plain-int reference of the same block-causal rule."""
    for num_frames, frames_per_block, frame_seqlen in ((4, 2, 2), (6, 3, 3)):
        block_mask = frame_block_causal_mask(
            device=torch.device("cpu"),
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            frames_per_block=frames_per_block,
            block_size=4,
        )

        sequence_length = num_frames * frame_seqlen
        block_token_count = frames_per_block * frame_seqlen
        assert block_mask.seq_lengths == (sequence_length, sequence_length)

        def expected(q_idx: int, kv_idx: int) -> bool:
            block_end = (q_idx // block_token_count + 1) * block_token_count
            return kv_idx < block_end

        zero = torch.tensor(0)
        for q_idx in range(sequence_length):
            for kv_idx in range(sequence_length):
                got = bool(block_mask.mask_mod(zero, zero, torch.tensor(q_idx), torch.tensor(kv_idx)))
                assert got == expected(q_idx, kv_idx), (num_frames, frames_per_block, frame_seqlen, q_idx, kv_idx)


@pytest.mark.parametrize("attention_heads", [None, 1])
def test_teacher_forcing_block_mask_can_materialize_singleton_head_shape(flex_block_size, attention_heads):
    block_mask = teacher_forcing_block_mask(
        device=torch.device("cpu"),
        num_frames=4,
        frame_seqlen=128,
        frames_per_block=1,
        block_size=flex_block_size,
        attention_heads=attention_heads,
    )

    assert block_mask.full_kv_num_blocks.shape == (1, 1, 8)
    assert block_mask.full_kv_indices.shape == (1, 1, 8, 8)
    assert block_mask.kv_num_blocks.shape == (1, 1, 8)
    assert block_mask.kv_indices.shape == (1, 1, 8, 8)
    assert block_mask.full_kv_num_blocks[0, 0, 5].item() == 2
    assert block_mask.full_kv_indices[0, 0, 5, :2].tolist() == [0, 5]


def test_teacher_forcing_block_mask_requires_complete_frame_blocks():
    with pytest.raises(AssertionError, match="num_frames 5"):
        teacher_forcing_block_mask(
            device=torch.device("cpu"),
            num_frames=5,
            frame_seqlen=3,
            frames_per_block=2,
            block_size=4,
        )


def test_teacher_forcing_block_mask_prefix_rules():
    """The attention rules with a text prefix (cosmos3's [text || gen] key
    layout): every gen query sees all prefix columns, clean queries in block i
    see clean blocks <= i, noisy queries in block j see clean blocks strictly
    below j plus their own noisy block — and noisy block j must NOT see clean
    block j, or training would learn a function no write-once KV cache can
    reproduce at rollout. Checked across several prefix_len values AND two
    geometries: attention_mask captures prefix_len and every
    sequence_length/block_token_count-derived value as q_idx-indexed tensors,
    not Python ints, because validation and training can build this mask
    with different geometries in the same process and any captured int would
    get promoted to a SymInt the moment it saw a second distinct value."""
    for num_frames, frames_per_block, frame_seqlen in ((4, 2, 2), (6, 3, 3)):
        q_len = 2 * num_frames * frame_seqlen
        tokens_per_block = frames_per_block * frame_seqlen
        clean_tokens = num_frames * frame_seqlen

        for prefix_len in (0, 3, 5):
            block_mask = teacher_forcing_block_mask(
                device=torch.device("cpu"),
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                frames_per_block=frames_per_block,
                block_size=4,
                prefix_len=prefix_len,
            )

            kv_len = prefix_len + q_len
            assert block_mask.seq_lengths == (q_len, kv_len)

            def expected(q_idx: int, kv_idx: int) -> bool:
                if kv_idx < prefix_len:
                    return True
                gen_kv = kv_idx - prefix_len
                q_clean = q_idx < clean_tokens
                q_block = (q_idx if q_clean else q_idx - clean_tokens) // tokens_per_block
                kv_clean = gen_kv < clean_tokens
                kv_block = (gen_kv if kv_clean else gen_kv - clean_tokens) // tokens_per_block
                if q_clean:
                    return kv_clean and kv_block <= q_block
                if kv_clean:
                    return kv_block < q_block
                return kv_block == q_block

            zero = torch.tensor(0)
            for q_idx in range(q_len):
                for kv_idx in range(kv_len):
                    got = bool(block_mask.mask_mod(zero, zero, torch.tensor(q_idx), torch.tensor(kv_idx)))
                    assert got == expected(q_idx, kv_idx), (
                        num_frames,
                        frames_per_block,
                        frame_seqlen,
                        prefix_len,
                        q_idx,
                        kv_idx,
                    )


def test_wan_block_frame_modulation_matches_eager_when_timestep_is_frame_constant():
    scale_shift_table = torch.randn(1, 6, 4)
    frame_values = torch.randn(2, 3, 6, 4, dtype=torch.bfloat16)
    frame_seqlen = 5
    temb = frame_values.repeat_interleave(frame_seqlen, dim=1)
    eager = scale_shift_table[:, 4] + temb[:, :, 4].float()

    frame_modulation = transformer_module._wan_block_frame_modulation_value(
        scale_shift_table,
        temb,
        4,
        batch_size=2,
        hidden_seq_len=temb.shape[1],
        frame_seqlen=frame_seqlen,
    )
    actual = frame_modulation.expand(-1, -1, frame_seqlen, -1).flatten(1, 2)

    torch.testing.assert_close(actual, eager)


def test_wan_output_frame_modulation_matches_eager_when_timestep_is_frame_constant():
    scale_shift_table = torch.randn(1, 2, 4)
    frame_values = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    frame_seqlen = 5
    temb = frame_values.repeat_interleave(frame_seqlen, dim=1)
    eager = (scale_shift_table.unsqueeze(0) + temb.unsqueeze(2).float()).chunk(2, dim=2)[1].squeeze(2)

    frame_modulation = transformer_module._wan_output_frame_modulation_value(
        scale_shift_table,
        temb,
        1,
        batch_size=2,
        hidden_seq_len=temb.shape[1],
        frame_seqlen=frame_seqlen,
    )
    actual = frame_modulation.expand(-1, -1, frame_seqlen, -1).flatten(1, 2)

    torch.testing.assert_close(actual, eager)


def test_modulated_layer_norm_fallback_matches_eager_frame_broadcast():
    norm = torch.nn.LayerNorm(4, elementwise_affine=False, eps=1e-6)
    hidden_states = torch.randn(2, 15, 4, dtype=torch.bfloat16)
    scale = torch.randn(2, 3, 1, 4)
    shift = torch.randn(2, 3, 1, 4)
    eager = (
        (norm(hidden_states.float()).unflatten(1, (3, 5)) * (1 + scale) + shift).flatten(1, 2).type_as(hidden_states)
    )

    actual = transformer_module._modulated_layer_norm(
        norm,
        hidden_states,
        scale,
        shift,
        frame_seqlen=5,
    )

    torch.testing.assert_close(actual, eager)


def test_modulated_layer_norm_cuda_training_preserves_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this gradient-preservation test")

    norm = torch.nn.LayerNorm(8, elementwise_affine=False, eps=1e-6).cuda()
    hidden_states = torch.randn(1, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scale = torch.randn(1, 2, 1, 8, device="cuda", requires_grad=True)
    shift = torch.randn(1, 2, 1, 8, device="cuda", requires_grad=True)

    with torch.enable_grad():
        output = transformer_module._modulated_layer_norm(
            norm,
            hidden_states,
            scale,
            shift,
            frame_seqlen=2,
        )
        output.float().sum().backward()

    assert hidden_states.grad is not None
    assert scale.grad is not None
    assert shift.grad is not None


def test_wan_block_expanded_timestep_modulation_avoids_eager_chunk(monkeypatch):
    class IdentityModule(torch.nn.Module):
        def forward(self, value):
            return value

    class FakeAttention(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return torch.zeros_like(hidden_states)

    class FakeFFN(torch.nn.Module):
        def forward(self, hidden_states):
            return torch.zeros_like(hidden_states)

    block = transformer_module.ArtifixerTransformerBlock.__new__(transformer_module.ArtifixerTransformerBlock)
    torch.nn.Module.__init__(block)
    block.scale_shift_table = torch.nn.Parameter(torch.randn(1, 6, 4))
    block.norm1 = IdentityModule()
    block.norm2 = IdentityModule()
    block.norm3 = IdentityModule()
    block.attn1 = FakeAttention()
    block.attn2 = FakeAttention()
    block.ffn = FakeFFN()
    block.opacity_embedding = None
    block.camera_embedding = None

    original_chunk = torch.Tensor.chunk

    def fail_large_modulation_chunk(tensor, chunks, dim=0):
        if tensor.shape == (1, 3, 6, 4):
            raise AssertionError("block modulation must not materialize and chunk the full six-way tensor")
        return original_chunk(tensor, chunks, dim=dim)

    monkeypatch.setattr(torch.Tensor, "chunk", fail_large_modulation_chunk)

    hidden_states = torch.randn(1, 3, 4)
    temb = torch.randn(1, 3, 6, 4, dtype=torch.bfloat16)
    output = block(
        hidden_states,
        encoder_hidden_states=torch.empty(1, 0, 4),
        neighbor_hidden_states=None,
        ignore_neighbors=True,
        temb=temb,
        rotary_emb=(torch.empty(1, 3, 2), torch.empty(1, 3, 2)),
        opacity_extra_patches=torch.empty(1, 3, 0),
        camera_extra_patches=torch.empty(1, 3, 0),
        kv_cache=None,
        layer_index=0,
        kv_cache_num_history_chunks=0,
        kv_cache_stage=None,
        context_kv_cache=None,
        frame_seqlen=3,
        prope_target=None,
        prope_reference=None,
        block_mask=None,
    )

    torch.testing.assert_close(output, hidden_states)


def test_latent_range_to_raw_frame_slice_matches_longlive_left_halo_mapping():
    assert latent_range_to_raw_frame_slice(0, 4, total_raw_frames=61, temporal_scale=4, halo_latent_frames=1) == (
        0,
        13,
        0,
        4,
    )
    assert latent_range_to_raw_frame_slice(4, 8, total_raw_frames=61, temporal_scale=4, halo_latent_frames=1) == (
        8,
        29,
        2,
        6,
    )
    assert latent_range_to_raw_frame_slice(12, 16, total_raw_frames=61, temporal_scale=4, halo_latent_frames=2) == (
        36,
        61,
        3,
        7,
    )


def test_latent_range_to_raw_frame_slice_rejects_out_of_range_latents():
    with pytest.raises(AssertionError, match="exceeds 16 latent frames"):
        latent_range_to_raw_frame_slice(
            15,
            17,
            total_raw_frames=61,
            temporal_scale=4,
            halo_latent_frames=1,
        )


def test_encode_video_frames_chunked_for_cp_can_keep_local_latents(monkeypatch):
    class ConcretePipeline(ArtifixerPipelineBase):
        def forward_inference(self, *args, **kwargs):
            raise NotImplementedError

    class FakeMesh:
        def size(self):
            return 2

        def get_group(self):
            return object()

    pipe = ConcretePipeline.__new__(ConcretePipeline)
    pipe.transformer = _FakeCPTransformer(_cp_mesh=FakeMesh())
    pipe.vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4))
    pipe._validate_spatial_alignment = lambda *args, **kwargs: None

    def encode_video_frames(frames):
        latent_frames = (frames.shape[1] - 1) // 4 + 1
        return torch.arange(latent_frames, dtype=torch.float32).view(1, 1, latent_frames, 1, 1)

    pipe.encode_video_frames = encode_video_frames
    monkeypatch.setattr(pipeline_module.dist, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(
        pipeline_module.dist,
        "all_gather",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local CP VAE must not all-gather")),
    )

    local_latents = pipe.encode_video_frames_chunked_for_cp(
        torch.zeros(1, 61, 3, 8, 8),
        halo_latent_frames=1,
        gather=False,
    )

    assert local_latents.shape == (1, 1, 8, 1, 1)
    torch.testing.assert_close(local_latents.flatten(), torch.arange(2, 10, dtype=torch.float32))


@pytest.mark.requires_host_ffmpeg
def test_plain_validation_does_not_pass_cp_local_inference_flags():
    class Transformer(_FakeCPProperties):
        frames_per_block = None

        def disable_context_parallel(self):
            return None

    class Pipe:
        vae = SimpleNamespace(dtype=torch.bfloat16, config=SimpleNamespace(scale_factor_temporal=4))
        transformer = Transformer()
        # The pipeline owns the block-causal contract; None = not block-causal.
        causal_block_frames = None

        def __init__(self):
            self.forward_kwargs = None
            self.local_cp_kwargs = None

        # The base pipeline's CP-selection methods, bound onto this bare stub
        # (not an ArtifixerPipelineBase subclass).
        cp_head_divisor = ArtifixerPipelineBase.cp_head_divisor
        cp_size_candidates = ArtifixerPipelineBase.cp_size_candidates
        cp_pass_latent_frames = ArtifixerPipelineBase.cp_pass_latent_frames
        use_local_cp_inference = ArtifixerPipelineBase.use_local_cp_inference

        def _cp_frame_divisor(self, total_latent_frames):
            return 1

        def spatial_alignment(self):
            return (16, 16)

        def forward_inference(self, **kwargs):
            self.forward_kwargs = kwargs
            return torch.zeros_like(kwargs["rendered_rgb"])

        def supports_local_cp_inference(self):
            return True

        def forward_inference_local_cp(self, **kwargs):
            self.local_cp_kwargs = kwargs
            raise AssertionError("plain validation should not use CP-local inference")

    class Dataset:
        num_frames = 5

        def __getitem__(self, index):
            latent_frames = 2
            return {
                "rgb_rendered": torch.zeros(self.num_frames, 3, 16, 16),
                "opacity": torch.zeros(self.num_frames, 16, 16),
                "encoded_prompt": torch.zeros(4, 8, dtype=torch.bfloat16),
                "rgb_neighbors": torch.zeros(3, 3, 16, 16),
                "camera_rays": torch.zeros(latent_frames, 16, 16, 6),
                "w2cs": torch.eye(4).unsqueeze(0).expand(latent_frames, -1, -1),
                "neighbor_w2cs": torch.eye(4).unsqueeze(0).expand(3, -1, -1),
                "Ks": torch.eye(3).unsqueeze(0).expand(latent_frames, -1, -1),
                "neighbor_Ks": torch.eye(3).unsqueeze(0).expand(3, -1, -1),
                "rgb_gt": torch.zeros(self.num_frames, 3, 16, 16),
            }

    trainer = _validation_runner(Pipe())

    trainer.generate_tasks({3: Dataset()}, [(ValidationTaskSpec("default_3", 3), 0)], Path(tempfile.mkdtemp()))

    kwargs = trainer.pipe.forward_kwargs
    assert trainer.pipe.local_cp_kwargs is None
    assert "local_cp_input" not in kwargs
    assert "local_vae_crop" not in kwargs
    assert "local_frame_offset" not in kwargs
    assert "return_latents" not in kwargs


def test_validation_task_cp_slice_uses_local_rgb_and_conditioning_windows():
    trainer = _validation_runner(
        SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4))), halo=1
    )
    task = ValidationTask(
        key="task",
        rendered_rgb=torch.arange(61).view(1, 61, 1, 1, 1).expand(-1, -1, 3, -1, -1).float(),
        rendered_opacity=torch.arange(61).view(1, 61, 1, 1).float(),
        neighbors=None,
        camera_rays=torch.arange(16).view(1, 16, 1, 1, 1).expand(-1, -1, -1, -1, 6).float(),
        w2cs=torch.arange(16).view(1, 16, 1, 1).expand(-1, -1, 4, 4).float(),
        neighbor_w2cs=None,
        Ks=torch.arange(16).view(1, 16, 1, 1).expand(-1, -1, 3, 3).float(),
        neighbor_Ks=None,
        prompt=torch.zeros(1, 4, 8),
        prompt_attention_mask=None,
        encoded_negative_prompt=None,
        negative_prompt_attention_mask=None,
        num_inference_steps=2,
        rgb_gt=torch.arange(61).view(1, 61, 1, 1, 1).expand(-1, -1, 3, -1, -1).float(),
    )

    local, frame_offset, vae_crop = trainer._slice_validation_task_for_cp(
        task,
        cp_group_size=2,
        group_rank=1,
        global_num_frames=61,
    )

    assert frame_offset == 8
    assert vae_crop == (2, 10)
    assert local.rendered_rgb[:, 0, 0, 0, 0].item() == 24
    assert local.rendered_rgb.shape[1] == 37
    assert local.rendered_opacity[:, 0, 0, 0].item() == 29
    assert local.rendered_opacity.shape[1] == 32
    assert local.camera_rays[:, 0, 0, 0, 0].item() == 8
    assert local.rgb_gt[:, 0, 0, 0, 0].item() == 32
    assert local.rgb_gt.shape[1] == 29


def test_cp_validation_latent_gather_reassembles_global_sequence(monkeypatch):
    class Mesh:
        def size(self):
            return 3

        def get_group(self):
            return object()

    class Transformer(_FakeCPProperties):
        _cp_mesh = Mesh()

    class Pipe:
        transformer = Transformer()

    def all_gather(outputs, local, group=None):
        for rank, output in enumerate(outputs):
            output.copy_(local + rank)

    trainer = _validation_runner(Pipe())
    local_latents = torch.zeros(1, 1, 2, 1, 1)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

    gathered = trainer._gather_cp_validation_latents(local_latents)

    assert gathered is not None
    assert gathered.shape == (1, 1, 6, 1, 1)
    torch.testing.assert_close(gathered.flatten(), torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]))


def test_cp_validation_saves_global_key_after_latent_gather(monkeypatch, tmp_path):
    class Pipe:
        vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scale_factor_temporal=1))
        # The pipeline owns the block-causal contract; None = not block-causal.
        causal_block_frames = None

        def __init__(self):
            self.transformer = _CPValTransformer()
            self.forward_kwargs = None
            self.decode_args = None

        # The base pipeline's CP-selection methods, bound onto this bare stub
        # (not an ArtifixerPipelineBase subclass).
        cp_head_divisor = ArtifixerPipelineBase.cp_head_divisor
        cp_size_candidates = ArtifixerPipelineBase.cp_size_candidates
        cp_pass_latent_frames = ArtifixerPipelineBase.cp_pass_latent_frames
        use_local_cp_inference = ArtifixerPipelineBase.use_local_cp_inference

        def _cp_frame_divisor(self, total_latent_frames):
            return 2

        def spatial_alignment(self):
            return (1, 1)

        def forward_inference(self, **kwargs):
            raise AssertionError("CP validation should use forward_inference_local_cp")

        def supports_local_cp_inference(self):
            return True

        def forward_inference_local_cp(self, **kwargs):
            self.forward_kwargs = kwargs
            assert kwargs["local_frame_offset"] == 0
            assert kwargs["local_vae_crop"] == (0, 1)
            assert "local_cp_input" not in kwargs
            assert "return_latents" not in kwargs
            return torch.zeros(1, 3, 1, 1, 1)

        def decode_latents_to_video(self, latents, temporal_chunk_size=None):
            self.decode_args = (latents.clone(), temporal_chunk_size)
            return latents.permute(0, 2, 1, 3, 4)

    class Dataset:
        num_frames = 2

        def __getitem__(self, index):
            return {
                "rgb_rendered": torch.zeros(self.num_frames, 3, 1, 1),
                "opacity": torch.zeros(self.num_frames, 1, 1),
                "encoded_prompt": torch.zeros(4, 8, dtype=torch.bfloat16),
                "rgb_neighbors": torch.zeros(3, 3, 1, 1),
                "camera_rays": torch.zeros(self.num_frames, 1, 1, 6),
                "w2cs": torch.eye(4).unsqueeze(0).expand(self.num_frames, -1, -1),
                "neighbor_w2cs": torch.eye(4).unsqueeze(0).expand(3, -1, -1),
                "Ks": torch.eye(3).unsqueeze(0).expand(self.num_frames, -1, -1),
                "neighbor_Ks": torch.eye(3).unsqueeze(0).expand(3, -1, -1),
                "rgb_gt": torch.zeros(self.num_frames, 3, 1, 1),
            }

    def all_gather(outputs, local, group=None):
        outputs[0].copy_(local)
        outputs[1].copy_(local + 1)

    _patch_cp_validation_dist(monkeypatch, all_gather)

    trainer = _validation_runner(Pipe())
    trainer.saved = []
    trainer.recorder.save = lambda output, task, save_dir, **kwargs: trainer.saved.append((task.key, output.clone()))

    trainer.generate_tasks({3: Dataset()}, [(ValidationTaskSpec("default_3", 3), 0)], tmp_path)

    assert trainer.saved[0][0] == "default_3"
    assert "_cp" not in trainer.saved[0][0]
    latents, temporal_chunk_size = trainer.pipe.decode_args
    torch.testing.assert_close(latents[:, :, 0], torch.zeros(1, 3, 1, 1))
    torch.testing.assert_close(latents[:, :, 1], torch.ones(1, 3, 1, 1))
    assert temporal_chunk_size == 1


def test_cp_validation_output_matches_plain_validation_with_deterministic_pipe(monkeypatch, tmp_path):
    class Pipe:
        vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scale_factor_temporal=1))
        # The pipeline owns the block-causal contract; None = not block-causal.
        causal_block_frames = None

        def __init__(self):
            self.transformer = _CPValTransformer()
            self.local_calls = []

        # The base pipeline's CP-selection methods, bound onto this bare stub
        # (not an ArtifixerPipelineBase subclass).
        cp_head_divisor = ArtifixerPipelineBase.cp_head_divisor
        cp_size_candidates = ArtifixerPipelineBase.cp_size_candidates
        cp_pass_latent_frames = ArtifixerPipelineBase.cp_pass_latent_frames
        use_local_cp_inference = ArtifixerPipelineBase.use_local_cp_inference

        def _cp_frame_divisor(self, total_latent_frames):
            return total_latent_frames

        def spatial_alignment(self):
            return (1, 1)

        def _deterministic_video(self, rendered_rgb, rendered_opacity):
            return rendered_rgb + rendered_opacity.unsqueeze(2)

        def forward_inference(self, **kwargs):
            return self._deterministic_video(kwargs["rendered_rgb"], kwargs["rendered_opacity"])

        def supports_local_cp_inference(self):
            return True

        def forward_inference_local_cp(self, **kwargs):
            local_video = self._deterministic_video(kwargs["rendered_rgb"], kwargs["rendered_opacity"])
            self.local_calls.append(
                {
                    "offset": kwargs["local_frame_offset"],
                    "crop": kwargs["local_vae_crop"],
                    "frames": local_video.clone(),
                }
            )
            return local_video.permute(0, 2, 1, 3, 4).contiguous()

        def decode_latents_to_video(self, latents, temporal_chunk_size=None):
            return latents.permute(0, 2, 1, 3, 4).contiguous()

    class Dataset:
        num_frames = 4

        def __getitem__(self, index):
            rgb_rendered = torch.arange(self.num_frames * 3, dtype=torch.float32).view(self.num_frames, 3, 1, 1)
            opacity = torch.arange(self.num_frames, dtype=torch.float32).view(self.num_frames, 1, 1) / 10
            return {
                "rgb_rendered": rgb_rendered,
                "opacity": opacity,
                "encoded_prompt": torch.zeros(4, 8, dtype=torch.bfloat16),
                "rgb_neighbors": torch.zeros(3, 3, 1, 1),
                "camera_rays": torch.zeros(self.num_frames, 1, 1, 6),
                "w2cs": torch.eye(4).unsqueeze(0).expand(self.num_frames, -1, -1),
                "neighbor_w2cs": torch.eye(4).unsqueeze(0).expand(3, -1, -1),
                "Ks": torch.eye(3).unsqueeze(0).expand(self.num_frames, -1, -1),
                "neighbor_Ks": torch.eye(3).unsqueeze(0).expand(3, -1, -1),
                "rgb_gt": torch.zeros(self.num_frames, 3, 1, 1),
            }

    def build_trainer(pipe):
        trainer = _validation_runner(pipe)
        trainer.saved = []
        trainer.recorder.save = lambda output, task, save_dir, **kwargs: trainer.saved.append(
            (task.key, output.clone())
        )
        return trainer

    dataset = Dataset()
    plain_pipe = Pipe()
    plain_trainer = build_trainer(plain_pipe)
    monkeypatch.setattr(validation_module.dist, "is_initialized", lambda: False)

    plain_trainer.generate_tasks({3: dataset}, [(ValidationTaskSpec("default_3", 3), 0)], tmp_path / "plain")
    plain_key, plain_output = plain_trainer.saved[0]
    expected_latents = plain_output.permute(0, 2, 1, 3, 4).contiguous()

    def all_gather(outputs, local, group=None):
        torch.testing.assert_close(local, expected_latents[:, :, :2])
        outputs[0].copy_(local)
        outputs[1].copy_(expected_latents[:, :, 2:])

    cp_pipe = Pipe()
    cp_trainer = build_trainer(cp_pipe)
    _patch_cp_validation_dist(monkeypatch, all_gather)

    cp_trainer.generate_tasks({3: dataset}, [(ValidationTaskSpec("default_3", 3), 0)], tmp_path / "cp")

    assert cp_trainer.saved[0][0] == plain_key
    torch.testing.assert_close(cp_trainer.saved[0][1], plain_output)
    assert cp_pipe.local_calls[0]["offset"] == 0
    assert cp_pipe.local_calls[0]["crop"] == (0, 2)
    torch.testing.assert_close(cp_pipe.local_calls[0]["frames"], plain_output[:, :2])


def test_validation_cp_capability_hooks_are_pipeline_owned():
    """The evaluation runner reads CP sizing from pipeline methods —
    cp_size_candidates for runnable sizes, cp_pass_latent_frames for the
    per-pass frame cost — and routes each task to local or full inputs via
    use_local_cp_inference; eval reads raw padding (cp_raw_frame_padding)
    and size checks (validate_cp_size) from the same methods. The shared
    local-CP implementation lives on the pipeline base with the predicate
    defaulting to False."""
    assert not hasattr(ArtifixerPipelineBase, "supports_context_parallel_validation")
    assert not hasattr(TrainerBase, "_supports_context_parallel_validation")
    assert hasattr(ArtifixerPipelineBase, "forward_inference_local_cp")
    for method_name in (
        "cp_size_candidates",
        "cp_pass_latent_frames",
        "cp_raw_frame_padding",
        "validate_cp_size",
        "use_local_cp_inference",
    ):
        assert hasattr(ArtifixerPipelineBase, method_name)
    assert ArtifixerPipelineBase.supports_local_cp_inference(SimpleNamespace()) is False
    assert ArtifixerPipeline.supports_local_cp_inference(SimpleNamespace()) is True
    # The routing rule keeps whole latent frames per rank: a clip the CP
    # size does not divide runs full-input even on a slicing-capable config.
    local_capable = SimpleNamespace(supports_local_cp_inference=lambda: True)
    assert ArtifixerPipelineBase.use_local_cp_inference(local_capable, 2, num_latent_frames=4) is True
    assert ArtifixerPipelineBase.use_local_cp_inference(local_capable, 2, num_latent_frames=21) is False


def test_standard_pipeline_explicitly_opts_into_generic_cp_validation():
    pipe = object.__new__(ArtifixerPipeline)

    assert ArtifixerPipeline._cp_frame_divisor is not ArtifixerPipelineBase._cp_frame_divisor
    assert pipe._cp_frame_divisor(total_latent_frames=384) == 384


def test_cp_validation_builds_full_task_on_cpu_then_moves_sliced_task(monkeypatch, tmp_path):
    class Pipe:
        vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scale_factor_temporal=1))
        transformer = _CPValTransformer()
        # The pipeline owns the block-causal contract; None = not block-causal.
        causal_block_frames = None

        # The base pipeline's CP-selection methods, bound onto this bare stub
        # (not an ArtifixerPipelineBase subclass).
        cp_head_divisor = ArtifixerPipelineBase.cp_head_divisor
        cp_size_candidates = ArtifixerPipelineBase.cp_size_candidates
        cp_pass_latent_frames = ArtifixerPipelineBase.cp_pass_latent_frames
        use_local_cp_inference = ArtifixerPipelineBase.use_local_cp_inference

        def _cp_frame_divisor(self, total_latent_frames):
            return 2

        def spatial_alignment(self):
            return (1, 1)

        def forward_inference(self, **kwargs):
            raise AssertionError("CP validation should use forward_inference_local_cp")

        def supports_local_cp_inference(self):
            return True

        def forward_inference_local_cp(self, **kwargs):
            assert kwargs["local_frame_offset"] == 0
            assert kwargs["local_vae_crop"] == (0, 1)
            assert "local_cp_input" not in kwargs
            return torch.zeros(1, 3, 1, 1, 1)

        def decode_latents_to_video(self, latents, temporal_chunk_size=None):
            return latents.permute(0, 2, 1, 3, 4)

    class Dataset:
        num_frames = 2

        def __getitem__(self, index):
            return {
                "rgb_rendered": torch.zeros(self.num_frames, 3, 1, 1),
                "opacity": torch.zeros(self.num_frames, 1, 1),
                "encoded_prompt": torch.zeros(4, 8, dtype=torch.bfloat16),
                "rgb_neighbors": torch.zeros(3, 3, 1, 1),
                "camera_rays": torch.zeros(self.num_frames, 1, 1, 6),
                "w2cs": torch.eye(4).unsqueeze(0).expand(self.num_frames, -1, -1),
                "neighbor_w2cs": torch.eye(4).unsqueeze(0).expand(3, -1, -1),
                "Ks": torch.eye(3).unsqueeze(0).expand(self.num_frames, -1, -1),
                "neighbor_Ks": torch.eye(3).unsqueeze(0).expand(3, -1, -1),
                "rgb_gt": torch.zeros(self.num_frames, 3, 1, 1),
            }

    copy_flags = []
    moved_shapes = []
    build_task = validation_module.build_validation_task

    def record_build(*args, **kwargs):
        copy_flags.append(kwargs.get("copy_to_device"))
        return build_task(*args, **kwargs)

    monkeypatch.setattr(validation_module, "build_validation_task", record_build)

    def all_gather(outputs, local, group=None):
        outputs[0].copy_(local)
        outputs[1].copy_(local)

    _patch_cp_validation_dist(monkeypatch, all_gather)

    trainer = _validation_runner(Pipe())
    move_task = trainer._move_validation_task_to_device

    def record_move(task):
        moved_shapes.append(task.rendered_rgb.shape[1])
        return move_task(task)

    trainer._move_validation_task_to_device = record_move
    trainer.recorder.save = lambda *args, **kwargs: None

    trainer.generate_tasks({3: Dataset()}, [(ValidationTaskSpec("default_3", 3), 0)], tmp_path)

    assert copy_flags == [False]
    assert moved_shapes == [1]


def test_chunked_latent_decode_uses_left_overlap_and_restores_global_length():
    class ConcretePipeline(ArtifixerPipelineBase):
        def forward_inference(self, *args, **kwargs):
            raise NotImplementedError

    class Processor:
        def postprocess_video(self, video, output_type):
            assert output_type == "pt"
            return video.permute(0, 2, 1, 3, 4)

    pipe = ConcretePipeline.__new__(ConcretePipeline)
    pipe.video_processor = Processor()
    calls = []

    def latents_to_rgb(latents):
        calls.append(latents.flatten().tolist())
        raw_frames = 4 * (latents.shape[2] - 1) + 1
        return torch.zeros(1, 1, raw_frames, 1, 1)

    pipe.latents_to_rgb = latents_to_rgb

    decoded = pipe.decode_latents_to_video(
        torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1),
        temporal_chunk_size=3,
    )

    assert calls == [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
    assert decoded.shape == (1, 29, 1, 1, 1)


def test_all_gather_cat_with_grad_single_rank_keeps_autograd():
    tensor = torch.randn(2, 3, requires_grad=True)

    gathered = all_gather_cat_with_grad(tensor, dim=1, group=None)
    loss = gathered.square().sum()
    loss.backward()

    assert torch.equal(gathered, tensor)
    torch.testing.assert_close(tensor.grad, 2 * tensor)


def _run_all_gather_cat_with_grad_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        local = torch.full((1, 2), float(rank + 1), requires_grad=True)
        gathered = all_gather_cat_with_grad(local, dim=1, group=None)
        weights = torch.arange(1, 1 + gathered.numel(), dtype=gathered.dtype).view_as(gathered)
        (gathered * weights).sum().backward()

        expected = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        expected_grad = torch.tensor([[2.0, 4.0]]) if rank == 0 else torch.tensor([[6.0, 8.0]])
        torch.testing.assert_close(gathered, expected)
        torch.testing.assert_close(local.grad, expected_grad)
    finally:
        dist.destroy_process_group()


def test_all_gather_cat_with_grad_two_rank_backward_accumulates_remote_gradients():
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("spawns its own torch.distributed process group; incompatible with xdist workers")

    ctx = mp.get_context("spawn")
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "pg_init")
        processes = [
            ctx.Process(target=_run_all_gather_cat_with_grad_worker, args=(rank, world_size, init_file))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(60)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
        for process in processes:
            assert process.exitcode == 0


def _run_attention_gather_worker(rank, world_size, init_file, paired, backend):
    from datetime import timedelta

    from torch.nn.attention import SDPBackend, sdpa_kernel

    from model_training.utils.precision import disable_tf32_matmul

    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
        disable_tf32_matmul()
    dist.init_process_group(
        backend, init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=180)
    )
    try:

        def tensors(index):
            values = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
            return (values + index * 6) / 17, (values * 3 - index * 6) / 23

        def loss(keys, values, index):
            query = (torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3) + index).sin()
            output = torch.nn.functional.scaled_dot_product_attention(
                query[None, None], keys[None, None], values[None, None]
            )
            return output.square().sum() * (index + 1)

        full_keys = torch.cat([tensors(index)[0] for index in range(world_size)]).requires_grad_(True)
        full_values = torch.cat([tensors(index)[1] for index in range(world_size)]).requires_grad_(True)
        with sdpa_kernel([SDPBackend.MATH]):
            sum(loss(full_keys, full_values, index) for index in range(world_size)).backward()
        keys, values = [tensor.requires_grad_(True) for tensor in tensors(rank)]
        if paired:
            gathered_keys, gathered_values = all_gather_cat_with_grad(
                torch.stack([keys, values]), dim=1, group=None
            ).unbind(0)
        else:
            gathered_keys = all_gather_cat_with_grad(keys, dim=0, group=None)
            gathered_values = full_values.detach()
        incoming_gradients = []
        gathered_keys.register_hook(lambda gradient: incoming_gradients.append(gradient.is_contiguous()))
        with sdpa_kernel([SDPBackend.MATH]):
            loss(gathered_keys, gathered_values, rank).backward()
        assert incoming_gradients == [False]
        torch.testing.assert_close(keys.grad, full_keys.grad.chunk(world_size)[rank], rtol=1e-5, atol=1e-6)
        if paired:
            torch.testing.assert_close(values.grad, full_values.grad.chunk(world_size)[rank], rtol=1e-5, atol=1e-6)
        else:
            assert values.grad is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("paired", [False, True], ids=["single_key", "paired_kv"])
def test_all_gather_cat_with_grad_sdpa_backward_matches_global_oracle(tmp_path, paired, backend):
    from tests._helpers import _run_cp_workers

    _run_cp_workers(_run_attention_gather_worker, tmp_path, paired, backend=backend)


def _run_cp_local_latent_noise_worker(rank: int, world_size: int, init_file: str) -> None:
    use_cuda = torch.cuda.device_count() >= world_size
    if use_cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:

        class ConcretePipeline(ArtifixerPipelineBase):
            def forward_inference(self, *args, **kwargs):
                raise NotImplementedError

        class Mesh:
            mesh = torch.arange(world_size)

            def size(self):
                return world_size

            def get_group(self):
                return dist.group.WORLD

        pipe = ConcretePipeline.__new__(ConcretePipeline)
        torch.nn.Module.__init__(pipe)
        pipe.transformer = _FakeCPTransformer(_cp_mesh=Mesh())
        pipe.vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=1, scale_factor_spatial=1))

        device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
        local_shape = (1, 1, 3, 1, 1)
        condition = torch.zeros(local_shape, device=device)
        opacity = torch.zeros(1, local_shape[2], 1, 1, device=device)
        torch.manual_seed(123)
        local_latents = pipe.prepare_latents(
            condition,
            opacity,
            is_first_chunk=False,
            broadcast_cp_noise=False,
        )

        gathered = [torch.empty_like(local_latents) for _ in range(world_size)]
        dist.all_gather(gathered, local_latents)
        if rank == 0:
            torch.manual_seed(123)
            expected = torch.randn(1, 1, local_shape[2] * world_size, 1, 1, device=device)
            torch.testing.assert_close(torch.cat(gathered, dim=2), expected)
    finally:
        dist.destroy_process_group()


def test_prepare_latents_local_cp_noise_matches_cp1_global_sequence():
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("spawns its own torch.distributed process group; incompatible with xdist workers")

    ctx = mp.get_context("spawn")
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "pg_init")
        processes = [
            ctx.Process(target=_run_cp_local_latent_noise_worker, args=(rank, world_size, init_file))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(180)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
        for process in processes:
            assert process.exitcode == 0


def _run_shared_ema_fsdp2_worker(rank, world_size, init_file, stage):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    from tests._helpers import _small_dmd_ema_trainer

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(123)
        model = _EMATinyModel(8, 8).cuda()
        fully_shard(
            model,
            mesh=init_device_mesh("cuda", (world_size,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        )
        trainer = _small_dmd_ema_trainer(Path(init_file).parent / f"rank-{rank}", model=model)
        trainer.accelerator.device = torch.device("cuda", rank)
        trainer.accelerator.process_index = rank
        trainer.accelerator.is_main_process = rank == 0
        expected_ema = None
        final_step = 206
        for update in range(1, final_step + 1):
            trainer.accelerator.step = update
            model_updated = trainer.is_student_phase()
            if model_updated:
                loss = model(torch.full((1, 8), 0.1, device="cuda")).float().square().mean()
                loss.backward()
                trainer.optimizers[0].step()
            trainer.on_before_zero_grad()
            trainer.optimizers[0].zero_grad()
            if model_updated:
                current = [param.to_local().detach().clone() for param in model.parameters()]
                if expected_ema is None or update < 200:
                    expected_ema = current
                else:
                    expected_ema = [0.99 * old + 0.01 * new for old, new in zip(expected_ema, current, strict=True)]
        for index, (raw, ema) in enumerate(zip(model.parameters(), trainer.ema.parameters(), strict=True)):
            assert isinstance(ema, DTensor)
            assert not ema.requires_grad
            assert ema.to_local().shape == raw.to_local().shape
            assert ema.to_local().numel() * world_size == ema.numel()
            torch.testing.assert_close(ema.to_local(), expected_ema[index])
        raw = [param.to_local().detach().clone() for param in model.parameters()]
        ema_full = [param.full_tensor() for param in trainer.ema.parameters()]

        def validate(*args):
            actual = model(torch.ones(1, 8, device="cuda"))
            expected = torch.nn.functional.linear(
                torch.ones(1, 8, device="cuda", dtype=torch.bfloat16),
                ema_full[0].to(torch.bfloat16),
                ema_full[1].to(torch.bfloat16),
            )
            torch.testing.assert_close(actual, expected)
            raise RuntimeError("FSDP validation failed")

        trainer.evaluation.generate_tasks = validate
        with pytest.raises(RuntimeError, match="FSDP validation failed"):
            with trainer.validation_context():
                trainer.evaluation.run_pass({}, [(ValidationTaskSpec("sample", 3), 0)], Path(init_file).parent)
        for param, expected in zip(model.parameters(), raw, strict=True):
            torch.testing.assert_close(param.to_local(), expected, rtol=0, atol=0)
        assert not trainer.ema.is_cached
        checkpoint = Path(init_file).parent / f"checkpoint_{final_step}"
        dcp.save({"model": model.state_dict()}, checkpoint_id=checkpoint / "pytorch_model_fsdp_0")
        for hook in trainer.accelerator.save_hooks:
            hook([], [], str(checkpoint))
        for slot, expected_model in (("pytorch_model_fsdp_0", model), ("pytorch_model_fsdp_ema", trainer.ema)):
            consumer = _EMATinyModel(8, 8)
            train_utils_module.load_model_weights_from_dcp(consumer, checkpoint / slot)
            for actual, expected in zip(consumer.parameters(), expected_model.parameters(), strict=True):
                torch.testing.assert_close(actual, expected.full_tensor().cpu(), rtol=0, atol=0)
        for hook in trainer.accelerator.load_hooks:
            hook([], str(checkpoint))
        expected_ema = [p.to_local().detach().clone() for p in trainer.ema.parameters()]
        trainer.ema.restore_after_model_load()
        for param, expected in zip(trainer.ema.parameters(), expected_ema, strict=True):
            torch.testing.assert_close(param.to_local(), expected, rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices for FSDP2/NCCL")
@pytest.mark.parametrize("stage", ["dmd"])
def test_shared_ema_fsdp2_shards_validation_and_checkpoint(tmp_path, stage):
    mp.spawn(_run_shared_ema_fsdp2_worker, args=(2, str(tmp_path / "init"), stage), nprocs=2, join=True)


def _saved_ema_checkpoint(tmp_path, updates=206):
    trainer = _small_dmd_ema_trainer(tmp_path, initialize=False)
    trainer.ema.restore_after_model_load()
    _advance_dmd_ema(trainer, updates)
    for hook in trainer.accelerator.save_hooks:
        hook([], [], str(tmp_path))
    return trainer


def _trainer_for_resume(tmp_path, step=206):
    trainer = _small_dmd_ema_trainer(tmp_path, initialize=False)
    trainer.accelerator.step = step
    for hook in trainer.accelerator.load_hooks:
        hook([], str(tmp_path))
    return trainer


def _ema_export_args(tmp_path):
    return export_module.parse_args(
        [
            "--checkpoint_dir",
            str(tmp_path / "pytorch_model_fsdp_ema"),
            "--output_pt",
            str(tmp_path / "export.pt"),
            "--run_id",
            "tiny",
            "--checkpoint",
            "206",
            "--slot",
            "pytorch_model_fsdp_ema",
        ]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("checkpoint_step", 103),
        ("ema_weight", 0.5),
        ("ema_start_step", 99),
        ("training_stage", "long_ar"),
        ("initialize_after_updates", 1),
        ("optimizer_updates", 207),
        ("optimizer_updates", -1),
        ("ema_initialized_update", 99),
        ("ema_initialized_update", 103),
        ("ema_initialized_update", None),
        ("ema_updates", 3),
    ],
)
def test_dmd_ema_resume_rejects_bad_progress(tmp_path, field, value):
    _saved_ema_checkpoint(tmp_path)
    state_path = tmp_path / "ema.json"
    progress = json.loads(state_path.read_text())
    progress[field] = value
    state_path.write_text(json.dumps(progress))
    trainer = _trainer_for_resume(tmp_path)
    with pytest.raises(ValueError) as error:
        trainer.ema.restore_after_model_load()
    message = str(error.value)
    assert field in message
    assert "expected" in message and "got" in message
    assert not trainer.ema.active


@pytest.mark.parametrize("entry", ["missing", "empty_directory", "metadata_directory"])
def test_dmd_ema_resume_requires_complete_weights(tmp_path, entry):
    source = tmp_path / "source"
    _saved_ema_checkpoint(source)
    (tmp_path / "ema.json").write_text((source / "ema.json").read_text())
    if entry != "missing":
        weights = tmp_path / "pytorch_model_fsdp_ema"
        weights.mkdir()
        if entry == "metadata_directory":
            (weights / ".metadata").mkdir()
    trainer = _trainer_for_resume(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"pytorch_model_fsdp_ema/\.metadata"):
        trainer.ema.restore_after_model_load()
    assert not trainer.ema.active


@pytest.mark.parametrize("entry", ["empty_directory", "file"])
def test_dmd_ema_resume_rejects_orphans_even_without_dcp_metadata(tmp_path, entry):
    weights = tmp_path / "pytorch_model_fsdp_ema"
    if entry == "empty_directory":
        weights.mkdir()
    else:
        weights.touch()
    trainer = _trainer_for_resume(tmp_path)
    with pytest.raises(ValueError, match="ema.json"):
        trainer.ema.restore_after_model_load()
    assert not trainer.ema.active
    assert not any("using the restored model" in message for message in trainer.accelerator.messages)


def test_dmd_ema_checkpoint_refuses_validation_weights(tmp_path):
    trainer = _saved_ema_checkpoint(tmp_path)
    with trainer.ema.apply():
        with pytest.raises(RuntimeError, match="EMA validation weights"):
            trainer.ema.save_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [("slot", "pytorch_model_fsdp_0"), ("checkpoint", 101), ("version", 2), ("ema_initialized_update", None)],
)
def test_dmd_ema_export_rejects_mislabeling_before_writing(tmp_path, monkeypatch, field, value):
    _saved_ema_checkpoint(tmp_path)
    args = _ema_export_args(tmp_path)
    if field in ("slot", "checkpoint"):
        setattr(args, field, value)
    else:
        state_path = tmp_path / "ema.json"
        progress = json.loads(state_path.read_text())
        progress[field] = value
        state_path.write_text(json.dumps(progress))
    builds = []
    monkeypatch.setattr(export_module, "create_export_transformer", lambda args: builds.append(args))
    with pytest.raises(ValueError) as error:
        export_module.export_checkpoint(args)
    message = str(error.value)
    assert field in message
    assert "expected" in message and "got" in message
    assert not builds
    assert not args.output_pt.exists()


def test_dmd_ema_export_reads_progress_once(tmp_path, monkeypatch):
    _saved_ema_checkpoint(tmp_path)
    state_path = tmp_path / "ema.json"
    original_read = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        if path == state_path:
            reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(export_module, "create_export_transformer", lambda args: _EMATinyModel(2, 2))
    export_module.export_checkpoint(_ema_export_args(tmp_path))
    assert reads == [state_path]


@pytest.mark.parametrize(
    "stage,updates,active,ema_updates",
    [
        ("dmd", 196, False, 0),
        ("dmd", 200, True, 0),
        ("dmd", 206, True, 2),
    ],
)
def test_ema_validation_files_and_logs_identify_weights(tmp_path, monkeypatch, stage, updates, active, ema_updates):
    trainer = _ema_stage_trainer(stage, tmp_path)
    trainer.evaluation.config.validation_metrics = False
    _advance_ema_stage(stage, trainer, updates)
    video = torch.zeros(1, 1, 3, 1, 1)
    task = ValidationTask(
        key="sample",
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
        num_inference_steps=1,
        rgb_gt=video,
    )

    def validate(datasets, task_plan, save_dir):
        if not task_plan:
            return
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.evaluation.recorder.save(video, task, save_dir)

    trainer.evaluation.generate_tasks = validate

    class CaptionDataset:
        def __getitem__(self, index):
            return {"prompt_file": "prompt", "prompt": "text"}

    trainer.val_datasets = {0: CaptionDataset()}
    monkeypatch.setattr(trainer_base_module, "uses_wandb_logging", lambda mode: True)
    monkeypatch.setattr(wandb, "Image", lambda path, **kwargs: Path(path))
    logs = {}
    trainer._validation_step(logs)
    key = "sample_ema" if active else "sample"
    assert logs["val/ema_active"] == int(active)
    assert logs["val/ema_updates"] == ema_updates
    assert logs[f"val/{key}"].name == f"{key}.png"
    assert task.key == "sample"
    assert not trainer.ema.is_cached


@pytest.mark.parametrize("method", ["copy_to", "update_average"])
@pytest.mark.parametrize("source_larger", [False, True])
def test_dmd_ema_updater_rejects_parameter_count_mismatch(method, source_larger):
    small = torch.nn.Linear(2, 2)
    large = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    source, target = (large, small) if source_larger else (small, large)
    with pytest.raises(ValueError, match="zip"):
        getattr(DTensorFastEmaModelUpdater(), method)(source, target)


@pytest.mark.parametrize(
    "slot,sidecar", [("pytorch_model_fsdp_0", False), ("pytorch_model_fsdp_1", False), ("pytorch_model_fsdp_1", True)]
)
def test_raw_export_does_not_relabel_unrelated_slots(tmp_path, monkeypatch, slot, sidecar):
    raw = _EMATinyModel(2, 2)
    path = tmp_path / slot
    dcp.save({"model": raw.state_dict()}, checkpoint_id=path)
    if sidecar:
        (tmp_path / "ema.json").write_text("{}")
    args = _ema_export_args(tmp_path)
    args.checkpoint_dir, args.slot = path, slot
    monkeypatch.setattr(export_module, "create_export_transformer", lambda args: _EMATinyModel(2, 2))
    _, record_path = export_module.export_checkpoint(args)
    record = json.loads(record_path.read_text())
    assert "weight_type" not in record
    assert "training_stage" not in record


def test_context_parallel_batch_broadcast_receives_source_shape(monkeypatch):
    source = torch.arange(6, dtype=torch.float32).view(2, 3)
    calls = []

    def fake_broadcast(tensor: torch.Tensor, src: int, group) -> None:
        calls.append((tuple(tensor.shape), tensor.dtype, src, group))
        if tensor.dtype == torch.long and tensor.ndim == 0:
            tensor.fill_(source.ndim)
        elif tensor.dtype == torch.long:
            tensor.copy_(torch.tensor(source.shape, dtype=torch.long))
        else:
            tensor.copy_(source)

    monkeypatch.setattr(dist, "broadcast", fake_broadcast)

    trainer = ContextParallelTrainer.__new__(ContextParallelTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.pipe = SimpleNamespace(transformer=_FakeCPTransformer(_cp_rank=1, _cp_world_size=2))

    received = trainer._broadcast_cp_data_tensor(torch.zeros(1, 1), group="cp", src=0)

    torch.testing.assert_close(received, source)
    assert calls == [((), torch.long, 0, "cp"), ((2,), torch.long, 0, "cp"), ((2, 3), torch.float32, 0, "cp")]
