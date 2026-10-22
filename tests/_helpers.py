# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers: training/EMA fixtures and
the canonical tiny Cosmos3 test doubles."""

import ast
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import Cosmos3OmniTransformer
from diffusers.models.embeddings import TimestepEmbedding
from torch import nn

from model_training.net.cosmos3_transformer import ArtifixerCosmos3Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeCPProperties:
    @property
    def cp_mesh(self):
        return getattr(self, "_cp_mesh", None)

    @property
    def cp_rank(self) -> int:
        if hasattr(self, "_cp_rank"):
            return int(self._cp_rank)
        cp_mesh = self.cp_mesh
        get_local_rank = getattr(cp_mesh, "get_local_rank", None)
        return int(get_local_rank()) if get_local_rank is not None else 0

    @property
    def cp_world_size(self) -> int:
        if hasattr(self, "_cp_world_size"):
            return int(self._cp_world_size)
        cp_mesh = self.cp_mesh
        return int(cp_mesh.size()) if cp_mesh is not None else 1


class _EMATestAccelerator:
    device = torch.device("cpu")
    process_index = 0
    is_main_process = True
    step = 0
    sync_gradients = True
    optimizer_step_was_skipped = False

    def __init__(self):
        self.save_hooks = []
        self.load_hooks = []
        self.messages = []

    def register_save_state_pre_hook(self, hook):
        self.save_hooks.append(hook)

    def register_load_state_pre_hook(self, hook):
        self.load_hooks.append(hook)

    def print(self, *args):
        self.messages.append(" ".join(map(str, args)))


class _EMATinyModel(_FakeCPProperties, nn.Linear):
    pass


def _small_dmd_ema_trainer(tmp_path, *, accelerator=None, model=None, optimizer=None, initialize=True, extra_args=()):
    from model_training.distillation import build_parser
    from model_training.trainers.dmd import DMD

    tmp_path.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args(
        [
            "--project_dir",
            str(tmp_path),
            "--base_checkpoint_dir",
            str(tmp_path / "base"),
            "--base_checkpoint_dir_critic",
            str(tmp_path / "critic"),
            "--split_path",
            "/unused/split.json",
            "--dl3dv_dir",
            "/unused/data",
            "--prompt_dir",
            "/unused/prompts",
            "--log_with",
            "none",
            *extra_args,
        ]
    )
    args.default_negative_prompt_path = tmp_path / "negative.pt"
    torch.save(torch.zeros(1, 2, 2), args.default_negative_prompt_path)
    model = _EMATinyModel(2, 2) if model is None else model
    accelerator = _EMATestAccelerator() if accelerator is None else accelerator
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) if optimizer is None else optimizer
    pipe = SimpleNamespace(
        transformer=model,
        vae=SimpleNamespace(
            device=torch.device("cpu"), dtype=torch.float32, config=SimpleNamespace(scale_factor_temporal=4)
        ),
        scheduler=SimpleNamespace(num_train_timesteps=1000),
    )
    trainer = DMD(args, accelerator, [optimizer], [], None, {}, pipe, model, model)
    if initialize:
        trainer.ema.restore_after_model_load()
    return trainer


def _advance_dmd_ema(trainer, through_step):
    accumulation = trainer.args.gradient_accumulation_steps
    for step in range(trainer.get_step() + 1, through_step + 1):
        for microstep in range(1, accumulation + 1):
            trainer.accelerator.step = (step - trainer.step_offset - 1) * accumulation + microstep
            trainer.accelerator.sync_gradients = microstep == accumulation
            if trainer.accelerator.sync_gradients and trainer.is_student_phase():
                with torch.no_grad():
                    for param in trainer.pipe.transformer.parameters():
                        param.fill_(float(step))
            trainer.on_before_zero_grad()


def _ema_stage_trainer(stage, tmp_path):
    assert stage == "dmd"
    return _small_dmd_ema_trainer(tmp_path)


def _advance_ema_stage(stage, trainer, through_step):
    assert stage == "dmd"
    _advance_dmd_ema(trainer, through_step)


# A real scene from data_processing/dl3dv_difix_frame_remap.json: these frames
# exist in DL3DV-ALL-960P but are absent from the compact DiFix frame space.
DIFIX_REMAP_SCENE_ID = "280abf7bd93b81b077af1db638229dbb09869052fa0b7b57c81c94d2db893829"
DIFIX_REMAP_DROPPED_FRAMES = ("frame_00026.png", "frame_00151.png", "frame_00162.png", "frame_00337.png")


class _DMDSetupTrainer:
    def __init__(self, args, accelerator, optimizers, schedulers, dataloader, val_datasets, pipe, *scores):
        from model_training.utils.ema import ModelEMA

        self.ema = ModelEMA(
            pipe.transformer,
            accelerator,
            lambda: 0,
            ema_weight=args.ema_weight,
            ema_start_step=200,
            initialize_after_updates=0,
            training_stage="dmd",
            camera_pose_alignment="mean",
        )

    def train(self):
        assert self.ema.active == (self.ema.config["ema_weight"] > 0)
        for param in self.ema.parameters():
            torch.testing.assert_close(param, torch.full_like(param, 3.0))


def _run_main_with_event_log(monkeypatch, tmp_path, *, ema_weight: float) -> list[tuple[str, str]]:
    import itertools

    from model_training import distillation, stages
    from model_training.checkpoint_metadata import write_checkpoint_metadata

    teacher_checkpoint = tmp_path / "teacher"
    write_checkpoint_metadata(teacher_checkpoint, {"camera_pose_alignment": "mean"})

    events: list[tuple[str, str]] = []
    critic_counter = itertools.count()

    class _EventModule(torch.nn.Linear):
        def to(self, *args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            if isinstance(device, torch.Tensor):
                device = device.device
            if isinstance(device, int) or (
                isinstance(device, (str, torch.device)) and torch.device(device).type not in ("cpu", "meta")
            ):
                events.append(("move", self._role))
            return self

        def cuda(self, device=None):
            events.append(("move", self._role))
            return self

    def _built_module(role: str) -> torch.nn.Module:
        module = _EventModule(1, 1)
        module._role = role
        events.append(("build", role))
        return module

    def _record_shard(objects) -> None:
        for obj in objects:
            if isinstance(obj, torch.nn.Module) and hasattr(obj, "_role"):
                events.append(("shard", obj._role))
                obj.to(_FakeAccelerator.device)

    class _FakeAccelerator:
        # main() reads accelerator.device.type when creating the CP mesh, so
        # this must be a real torch.device rather than a plain string.
        device = torch.device("cuda:0")

        def register_save_state_pre_hook(self, hook):
            pass

        def register_load_state_pre_hook(self, hook):
            pass

        def prepare(self, *objects):
            _record_shard(objects)
            return objects

        def prepare_data_loader(self, dataloader, device_placement=None):
            return dataloader

        def _prepare_fsdp2(self, *objects):
            _record_shard(objects)
            return objects[0] if len(objects) == 1 else objects

    def _fake_fsdp2_prepare_model(accelerator, model):
        _record_shard([model])
        return model

    def _fake_get_kv_cache_pipe(args, device, *, truncate_rollout_gradients=False):
        return SimpleNamespace(
            vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)),
            transformer=_built_module("student"),
            ar_window_chunks=max(1, args.local_attn_size // args.frames_per_block),
            ar_frames_per_chunk=args.frames_per_block,
        )

    monkeypatch.setattr(stages, "get_run_id_and_should_resume", lambda args: (None, False))
    monkeypatch.setattr(stages, "get_accelerator", lambda args, run_id: _FakeAccelerator())
    monkeypatch.setattr(stages, "get_kv_cache_pipe", _fake_get_kv_cache_pipe)
    monkeypatch.setattr(
        stages,
        "build_score_transformer",
        lambda args, model_id, pipe, *, gradient_checkpointing: _built_module(f"score_{next(critic_counter)}"),
    )
    monkeypatch.setattr(stages, "get_train_dataloader", lambda args, accelerator, fpb: "dataloader")
    monkeypatch.setattr(stages, "barrier_if_distributed", lambda: None)

    def load_weights(model, path):
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(3.0 if model._role == "student" else 5.0)

    monkeypatch.setattr(stages, "load_model_weights_from_dcp", load_weights)
    monkeypatch.setattr(stages, "fsdp2_prepare_model", _fake_fsdp2_prepare_model)
    monkeypatch.setattr(stages, "maybe_write_run_id", lambda *call_args: None)
    monkeypatch.setattr(stages, "get_val_datasets", lambda *call_args: {})
    monkeypatch.setattr(stages, "DMD", _DMDSetupTrainer)

    args = SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        model_id_critic=None,
        num_frames=81,  # 21 latent frames, divisible by frames_per_block=7
        frames_per_block=7,
        # CP disabled: main() validates these up front, but with size 1 the
        # CP mesh is None and the guard stays a pure build/shard-order check.
        context_parallel_size=1,
        local_attn_size=21,
        sink_size=7,
        gradient_checkpointing=False,
        learning_rate=1e-4,
        learning_rate_critic=1e-4,
        weight_decay=0.0,
        ema_weight=ema_weight,
        truncate_rollout_gradients=False,
        project_dir=None,
        base_checkpoint_dir=None,
        base_checkpoint_dir_critic=teacher_checkpoint,
        camera_pose_alignment="mean",
        log_with="none",
    )

    distillation.main(args)
    return events


def write_fake_3dgrut_config(grut_root: Path) -> None:
    """Lay out the files _validated_grut_root requires inside a fake 3DGRUT checkout."""
    (grut_root / "configs").mkdir(parents=True)
    (grut_root / "train.py").write_text("")
    (grut_root / "thirdparty" / "tiny-cuda-nn" / "include" / "tiny-cuda-nn").mkdir(parents=True)
    (grut_root / "thirdparty" / "tiny-cuda-nn" / "include" / "tiny-cuda-nn" / "common.h").write_text("")
    (grut_root / "configs" / "base_gs.yaml").write_text(
        "\n".join(
            [
                "# SPDX-FileCopyrightText: Copyright",
                "defaults:",
                "  - override hydra/hydra_logging: disabled",
                "path: ???",
            ]
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# Canonical tiny Cosmos3 test doubles.
#
# ONE `_TinyCosmosBaseTransformer` serves every test that needs a reduced
# stand-in for the diffusers `Cosmos3OmniTransformer`: it mirrors the fork's
# packed-sequence forward contract (the exact kwargs
# `ArtifixerCosmos3Transformer.forward` passes) while recording the calls so
# behavioral tests can assert on the packing. The kwarg contract itself is
# additionally pinned against the REAL v0.39.0 class by
# tests/test_cosmos3_real_model_contract.py, so a fork rename breaks CI, not
# production. The dimensions are the production latent geometry (48 channels,
# patch 2 -> 192-dim tokens) with everything else reduced.
# ---------------------------------------------------------------------------

LATENT_CHANNELS = 48
PATCH_SIZE = 2
PATCH_DIM = LATENT_CHANNELS * PATCH_SIZE * PATCH_SIZE


class _TinyAttention(nn.Module):
    def __init__(self, hidden_size: int = PATCH_DIM, heads: int = 4):
        super().__init__()
        self.head_dim = hidden_size // heads
        self.num_attention_heads = heads
        self.num_key_value_heads = heads
        self.heads = heads
        self.add_q_proj = nn.Linear(hidden_size, hidden_size)
        self.add_k_proj = nn.Linear(hidden_size, hidden_size)
        self.add_v_proj = nn.Linear(hidden_size, hidden_size)
        self.to_add_out = nn.Linear(hidden_size, hidden_size)
        self.norm_added_q = nn.Identity()
        self.norm_added_k = nn.Identity()
        self.processor = SimpleNamespace(_attention_backend=None, _parallel_config=None)
        with torch.no_grad():
            for projection in (self.add_q_proj, self.add_k_proj, self.add_v_proj, self.to_add_out):
                torch.nn.init.eye_(projection.weight)
                projection.bias.zero_()

    def forward(self, und_seq, gen_seq, rotary_emb):
        return und_seq, gen_seq

    def set_processor(self, processor):
        self.processor = processor


class _TinyCosmosLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TinyAttention()
        self.input_layernorm_moe_gen = nn.LayerNorm(PATCH_DIM)
        self.post_attention_layernorm_moe_gen = nn.LayerNorm(PATCH_DIM)
        self.mlp_moe_gen = nn.Linear(PATCH_DIM, PATCH_DIM)
        self.inputs = []
        self.attn_outputs = []

    def forward(self, und_seq, gen_seq, rotary_emb):
        self.inputs.append(gen_seq.detach().clone())
        und_seq, gen_seq = self.self_attn(und_seq, gen_seq, rotary_emb)
        self.attn_outputs.append(gen_seq.detach().clone())
        return und_seq, gen_seq


class _FloatTimeProjection(nn.Module):
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.ones(timesteps.shape[0], 4, device=timesteps.device, dtype=torch.float32)


class _TinyDomainAwareLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


class _TinyCosmosBaseTransformer(nn.Module):
    layer_cls = _TinyCosmosLayer

    def __init__(self, num_layers: int = 2, *, config_action_dim: int = 9, domain_aware_action_proj: bool = False):
        super().__init__()
        self.config = SimpleNamespace(
            latent_patch_size=PATCH_SIZE,
            latent_channel=LATENT_CHANNELS,
            enable_fps_modulation=False,
            unified_3d_mrope_temporal_modality_margin=15000,
            unified_3d_mrope_reset_spatial_ids=True,
            base_fps=24.0,
            action_gen=True,
            action_dim=config_action_dim,
            # Same value as the reduced production config double; the wrapper
            # reads it for the camera-gain timestep features.
            timestep_scale=0.001,
        )
        self.proj_in = nn.Linear(PATCH_DIM, PATCH_DIM)
        self.proj_out = nn.Linear(PATCH_DIM, PATCH_DIM)
        # Real diffusers TimestepEmbedding: the wrapper adopts its
        # linear_1/act/linear_2 children into the fp32 island.
        self.time_embedder = TimestepEmbedding(in_channels=4, time_embed_dim=PATCH_DIM)
        self.time_proj = _FloatTimeProjection()
        self.norm_moe_gen = nn.LayerNorm(PATCH_DIM)
        if domain_aware_action_proj:
            self.action_proj_in = _TinyDomainAwareLinear(config_action_dim, PATCH_DIM)
        else:
            self.action_proj_in = nn.Linear(config_action_dim, PATCH_DIM)
        self.action_modality_embed = nn.Parameter(torch.zeros(PATCH_DIM))
        self.layers = nn.ModuleList([self.layer_cls() for _ in range(num_layers)])
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None
        self.backend = None
        self.calls = []
        self.last_sequence = None
        with torch.no_grad():
            for projection in (self.proj_in, self.proj_out):
                torch.nn.init.eye_(projection.weight)
                projection.bias.zero_()

    def enable_gradient_checkpointing(self, gradient_checkpointing_func=None):
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = gradient_checkpointing_func

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def set_attention_backend(self, backend):
        self.backend = backend
        for layer in self.layers:
            layer.self_attn.processor._attention_backend = backend

    def _patchify(self, latents):
        bsz, channels, frames, height, width = latents.shape
        patch_h = height // PATCH_SIZE
        patch_w = width // PATCH_SIZE
        tokens = latents.reshape(bsz, channels, frames, patch_h, PATCH_SIZE, patch_w, PATCH_SIZE)
        tokens = tokens.permute(0, 2, 3, 5, 1, 4, 6).reshape(bsz, frames * patch_h * patch_w, PATCH_DIM)
        return tokens, (frames, patch_h, patch_w)

    def _unpatchify(self, tokens, shape):
        frames, patch_h, patch_w = shape
        bsz = tokens.shape[0]
        latents = tokens.reshape(bsz, frames, patch_h, patch_w, LATENT_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        return latents.permute(0, 4, 1, 2, 5, 3, 6).reshape(
            bsz,
            LATENT_CHANNELS,
            frames,
            patch_h * PATCH_SIZE,
            patch_w * PATCH_SIZE,
        )

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        latents = kwargs["vision_tokens"][0]
        segments = [self._patchify(segment) for segment in kwargs["vision_tokens"]]
        vision_tokens = self.proj_in(torch.cat([tokens for tokens, _shape in segments], dim=1))
        sequence = torch.zeros(
            int(kwargs["sequence_length"]),
            PATCH_DIM,
            device=latents.device,
            dtype=vision_tokens.dtype,
        )
        sequence[kwargs["text_indexes"]] = 0.25
        sequence[kwargs["vision_sequence_indexes"]] = vision_tokens.squeeze(0)
        if "action_tokens" in kwargs:
            action_tokens = self.action_proj_in(kwargs["action_tokens"][0]) + self.action_modality_embed
            sequence[kwargs["action_sequence_indexes"]] = action_tokens
        und_seq = sequence[: kwargs["und_len"]]
        gen_seq = sequence[kwargs["und_len"] :]
        rotary_emb = (
            torch.empty(0, device=latents.device),
            torch.empty(0, device=latents.device),
            torch.empty(0, device=latents.device),
            torch.empty(0, device=latents.device),
        )
        for layer in self.layers:
            und_seq, gen_seq = layer(und_seq, gen_seq, rotary_emb)
        sequence = torch.cat([und_seq, gen_seq], dim=0).unsqueeze(0)
        self.last_sequence = sequence.detach().clone()
        predictions = torch.zeros_like(sequence)
        loss_indexes = kwargs["vision_mse_loss_indexes"]
        predictions[:, loss_indexes] = self.proj_out(sequence.index_select(1, loss_indexes))
        vision_tokens = predictions.index_select(1, kwargs["vision_sequence_indexes"])
        segment_tokens = vision_tokens.split([tokens.shape[1] for tokens, _shape in segments], dim=1)
        return (
            [
                self._unpatchify(tokens, shape)
                for tokens, (_inputs, shape) in zip(segment_tokens, segments, strict=True)
            ],
            None,
            None,
        )


class _TinyCosmosVae(nn.Module):
    device = torch.device("cpu")
    dtype = torch.bfloat16
    config = SimpleNamespace(
        scale_factor_spatial=16,
        scale_factor_temporal=4,
        latents_mean=[0.0] * LATENT_CHANNELS,
        latents_std=[1.0] * LATENT_CHANNELS,
        z_dim=LATENT_CHANNELS,
    )


# ---------------------------------------------------------------------------
# Processor-carrying doubles: same canonical fake, but the attention modules
# run the REAL ArtifixerCosmos3AttnProcessor (SDPA over the packed sequence)
# so cross-token information flow is exercised where it matters (prefix
# detachment, CP gathers, block-causal masking).
# ---------------------------------------------------------------------------


class _ProcessorAttention(nn.Module):
    def __init__(self, hidden_size: int = 16, heads: int = 4, kv_heads: int | None = None):
        # kv_heads < heads gives the double cosmos3's REAL GQA shape (the K/V
        # projections emit num_key_value_heads * head_dim features, exactly
        # like the production attention module); the default keeps every
        # existing caller byte-identical.
        super().__init__()
        kv_heads = heads if kv_heads is None else kv_heads
        assert heads % kv_heads == 0, "GQA needs the query head count to be a multiple of the KV head count"
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // heads
        self.num_attention_heads = heads
        self.num_key_value_heads = kv_heads
        kv_dim = kv_heads * self.head_dim
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, kv_dim)
        self.to_v = nn.Linear(hidden_size, kv_dim)
        self.to_out = nn.Linear(hidden_size, hidden_size)
        self.add_q_proj = nn.Linear(hidden_size, hidden_size)
        self.add_k_proj = nn.Linear(hidden_size, kv_dim)
        self.add_v_proj = nn.Linear(hidden_size, kv_dim)
        self.to_add_out = nn.Linear(hidden_size, hidden_size)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.norm_added_q = nn.Identity()
        self.norm_added_k = nn.Identity()
        with torch.no_grad():
            for projection in (
                self.to_q,
                self.to_k,
                self.to_v,
                self.to_out,
                self.add_q_proj,
                self.add_k_proj,
                self.add_v_proj,
                self.to_add_out,
            ):
                torch.nn.init.eye_(projection.weight)
                projection.bias.zero_()


class _TinyProcessorAttention(_ProcessorAttention):
    def __init__(self):
        super().__init__(hidden_size=PATCH_DIM, heads=4)
        self.processor = SimpleNamespace(_attention_backend=None, _parallel_config=None)

    def set_processor(self, processor):
        self.processor = processor

    def forward(self, und_seq, gen_seq, rotary_emb):
        head_dim = self.hidden_size // self.num_attention_heads
        identity_rotary = (
            torch.ones(und_seq.shape[0], head_dim, device=gen_seq.device, dtype=gen_seq.dtype),
            torch.zeros(und_seq.shape[0], head_dim, device=gen_seq.device, dtype=gen_seq.dtype),
            torch.ones(gen_seq.shape[0], head_dim, device=gen_seq.device, dtype=gen_seq.dtype),
            torch.zeros(gen_seq.shape[0], head_dim, device=gen_seq.device, dtype=gen_seq.dtype),
        )
        return self.processor(self, und_seq, gen_seq, identity_rotary)


class _TinyProcessorCosmosLayer(_TinyCosmosLayer):
    def __init__(self):
        super().__init__()
        self.self_attn = _TinyProcessorAttention()


class _TinyProcessorCosmosBaseTransformer(_TinyCosmosBaseTransformer):
    layer_cls = _TinyProcessorCosmosLayer


class _FakeCpMesh:
    """The single CP-mesh double (get_local_rank/size/get_group).

    Keyword-only on purpose: positional (rank, size) vs (size, rank) swaps
    across the family test files type-check and run, so they are forbidden
    at the constructor."""

    def __init__(self, *, rank: int = 0, size: int):
        self.rank = int(rank)
        self.world_size = int(size)

    def get_local_rank(self) -> int:
        return self.rank

    def size(self) -> int:
        return self.world_size

    def get_group(self):
        return None


class _GroupedFakeCpMesh(_FakeCpMesh):
    """_FakeCpMesh that hands out a stable sentinel process group: the
    grad-carrying CP collectives key their exchanges on a non-None group,
    and the base double returns None — which would silently exercise
    nothing. (test_cp_attention.py / test_kv_cache_pipeline.py carry local
    copies predating this shared one.)"""

    _group = object()

    def get_group(self):
        return self._group


def _identity_rotary(und_tokens: int, gen_tokens: int, head_dim: int):
    return (
        torch.ones(und_tokens, head_dim),
        torch.zeros(und_tokens, head_dim),
        torch.ones(gen_tokens, head_dim),
        torch.zeros(gen_tokens, head_dim),
    )


# ---------------------------------------------------------------------------
# Cosmos3-shaped CP collective fakes. The cosmos3 attention processor is
# UNBATCHED [S, H, D] with two streams and real GQA, so Wan's batched
# [B, S, H, D] collective doubles cannot be reused: these helpers rebuild the
# exchange in cosmos3's own layout (scatter_dim=1 heads / gather_dim=0 rows).
# Fakes must replace the collective symbols in the module that resolves
# them (cosmos3_transformer_module.all_to_all_with_grad for the Ulysses
# exchange; common/context_parallel's all_gather_cat / all_gather_cat_with_grad
# for the cache-read gathers, which gather_kv_cat looks up in its own module):
# the real collectives early-return the tensor unchanged while
# torch.distributed is uninitialized, so a dist_nn_func-level fake alone
# would silently test nothing.
# ---------------------------------------------------------------------------


def cosmos3_projected_gen_qkv(attention, gen_seq: torch.Tensor):
    """The cosmos3 processor's gen-stream Q/K/V exactly as they stand right
    before the CP collectives: projection + Q/K norm in the unbatched
    [S, H, D] layout. The tiny doubles run the identity rotary (cos=1,
    sin=0), so RoPE is a no-op and these ARE the post-RoPE tensors CP peers
    hold — single-process fakes reconstruct remote ranks' contributions from
    them."""
    q = attention.norm_added_q(
        attention.add_q_proj(gen_seq).view(-1, attention.num_attention_heads, attention.head_dim)
    )
    k = attention.norm_added_k(
        attention.add_k_proj(gen_seq).view(-1, attention.num_key_value_heads, attention.head_dim)
    )
    v = attention.add_v_proj(gen_seq).view(-1, attention.num_key_value_heads, attention.head_dim)
    return q, k, v


def cosmos3_rankwise_all_to_all(
    actual: torch.Tensor,
    peer_tensors: list[torch.Tensor],
    *,
    rank: int,
    world_size: int,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """Exact single-process stand-in for ONE rank's all_to_all_with_grad in
    cosmos3's unbatched [S, H, D] layout: rank ``rank`` receives scatter-chunk
    ``rank`` from every peer, concatenated in peer order along the gather dim
    — the real collective's contiguous rank-order chunk routing
    (common/context_parallel.py). ``actual`` fills the caller's own slot so
    the calling rank's OWN computation stays under test; ``peer_tensors[p]``
    is what peer ``p`` holds pre-exchange, reconstructed by the caller."""
    pieces = []
    for peer in range(world_size):
        source = actual if peer == rank else peer_tensors[peer]
        pieces.append(source.chunk(world_size, dim=scatter_dim)[rank])
    return torch.cat(pieces, dim=gather_dim)


# ---------------------------------------------------------------------------
# REAL reduced-config Cosmos3OmniTransformer doubles:
# the contract tests (tests/test_cosmos3_real_model_contract.py)
# instantiate the installed diffusers
# class from this reduced config so the ArtiFixer<->diffusers contract is
# asserted on every CI run.
# ---------------------------------------------------------------------------

# Reduced Cosmos3OmniTransformer config: production values wherever the
# wrapper or the data contract depends on them, everything else shrunk.
# Passed as direct keywords so a renamed config key in the installed
# diffusers fails the contract tests instead of production-at-load.
REDUCED_CONFIG = dict(
    # Layer geometry: 2 layers instead of 36; hidden 128 = 4 heads x head_dim
    # 32 instead of 4096 = 32 x 128. GQA stays real (2 kv heads, ratio 2).
    num_hidden_layers=2,
    hidden_size=128,
    head_dim=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=256,
    # mrope_section must sum to head_dim // 2 (production [24, 20, 20] = 64);
    # rope_theta is the production value.
    rope_scaling={"mrope_section": [6, 5, 5]},
    rope_theta=5000000.0,
    # Latent geometry is the production Wan-VAE contract the wrapper packs
    # against: 48 channels, patch 2, 192-dim patch tokens.
    latent_channel=48,
    latent_patch_size=2,
    patch_latent_dim=192,
    # Action head: the pinned base model ships action modules (frozen and
    # undriven by the wrapper), so the reduced config mirrors them — the DCP
    # cache-check and pin tests require the action keys to exist.
    action_gen=True,
    action_dim=16,
    num_embodiment_domains=4,
    # Production temporal/mRoPE policy (base_fps 24, fps modulation on,
    # margin 15000) so the packed positions match the training regime.
    base_fps=24,
    enable_fps_modulation=True,
    unified_3d_mrope_reset_spatial_ids=True,
    unified_3d_mrope_temporal_modality_margin=15000,
    timestep_scale=0.001,
    vocab_size=128,
)
# Internal consistency of the reduced config (the same identities the
# production config satisfies); a careless edit fails here, not downstream.
assert REDUCED_CONFIG["hidden_size"] == REDUCED_CONFIG["num_attention_heads"] * REDUCED_CONFIG["head_dim"]
assert REDUCED_CONFIG["patch_latent_dim"] == REDUCED_CONFIG["latent_channel"] * REDUCED_CONFIG["latent_patch_size"] ** 2
assert sum(REDUCED_CONFIG["rope_scaling"]["mrope_section"]) == REDUCED_CONFIG["head_dim"] // 2

# Production VAE factors: pin the real adapter input widths — opacity
# (16 * 2)**2 * 4 = 4096 and camera rays 6 * (16 * 2)**2 = 6144.
VAE_SCALE_FACTOR_SPATIAL = 16
VAE_SCALE_FACTOR_TEMPORAL = 4
LATENT_HEIGHT = LATENT_WIDTH = 4
PIXEL_HEIGHT = LATENT_HEIGHT * VAE_SCALE_FACTOR_SPATIAL
PIXEL_WIDTH = LATENT_WIDTH * VAE_SCALE_FACTOR_SPATIAL


def _reduced_real_transformer(
    gradient_checkpointing: bool = False,
    causal_block_frames: int | None = None,
):
    torch.manual_seed(0)
    base = Cosmos3OmniTransformer(**REDUCED_CONFIG)
    return ArtifixerCosmos3Transformer(
        base,
        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
        vae_scale_factor_temporal=VAE_SCALE_FACTOR_TEMPORAL,
        gradient_checkpointing=gradient_checkpointing,
        causal_block_frames=causal_block_frames,
    )


def _activate_conditioning_adapters(transformer: ArtifixerCosmos3Transformer) -> None:
    """Use nonzero camera and opacity weights to expose incorrect frame slices."""
    generator = torch.Generator().manual_seed(23)
    for adapter in transformer.opacity_embeddings:
        adapter.weight.data.normal_(0.0, 0.02, generator=generator)
        adapter.bias.data.normal_(0.0, 0.02, generator=generator)
    # Changing alpha has no effect until the projection weights are nonzero.
    for injection in transformer.scope:
        injection.query_projection.weight.data.normal_(0.0, 0.02, generator=generator)
        injection.key_projection.weight.data.normal_(0.0, 0.02, generator=generator)
        injection.alpha.data.fill_(0.25)


GROWING_PREFIX_FRAMES = 3  # three 1-frame chunks: prefix {0} -> prefix {0, 1}


def _growing_prefix_kwargs(
    total_frames: int,
    *,
    reference_frames: int = 1,
    latent_hw: tuple[int, int] = (LATENT_HEIGHT, LATENT_WIDTH),
) -> dict:
    """Batched inputs whose shorter passes are leading slices of the target frames."""
    torch.manual_seed(2)
    pixel_frames = (total_frames - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1
    pixel_height, pixel_width = (side * VAE_SCALE_FACTOR_SPATIAL for side in latent_hw)
    return dict(
        hidden_states=torch.randn(1, 48, total_frames, *latent_hw),
        timestep=torch.tensor([999.0]),
        encoder_hidden_states=torch.tensor([[5, 6, 7, 8]]),
        encoder_attention_mask=torch.ones(1, 4, dtype=torch.bool),
        neighbor_hidden_states=torch.randn(1, 48, reference_frames, *latent_hw),
        opacity=torch.rand(1, pixel_frames, pixel_height, pixel_width),
        camera_rays=torch.randn(1, total_frames, pixel_height, pixel_width, 6),
        neighbor_camera_rays=torch.randn(1, reference_frames, pixel_height, pixel_width, 6),
        w2cs=torch.eye(4).view(1, 1, 4, 4).expand(1, total_frames, -1, -1).clone(),
        neighbor_w2cs=torch.eye(4).view(1, 1, 4, 4).expand(1, reference_frames, -1, -1).clone(),
        Ks=torch.eye(3).view(1, 1, 3, 3).expand(1, total_frames, -1, -1).clone(),
        neighbor_Ks=torch.eye(3).view(1, 1, 3, 3).expand(1, reference_frames, -1, -1).clone(),
    )


def _joint_reference_kwargs(
    total_frames: int = GROWING_PREFIX_FRAMES,
    *,
    reference_frames: int = 3,
    latent_hw: tuple[int, int] = (LATENT_HEIGHT, LATENT_WIDTH),
    reference_translation: float = 0.0,
    reference_distortion_delta: float = 0.0,
    reference_focal_delta: float = 0.0,
    reference_order: tuple[int, ...] | None = None,
) -> dict:
    """Inputs with distorted lenses and independently permutable reference images.

    Build rays from full lens parameters in first-target coordinates, using the
    same spatial grid for references and targets.
    """
    from model_training.data.utils import compute_camera_conditioning

    kwargs = _growing_prefix_kwargs(total_frames, reference_frames=reference_frames, latent_hw=latent_hw)

    def world_poses(count: int, *, reference: bool) -> torch.Tensor:
        index = torch.arange(count, dtype=torch.float32)
        angle = (0.35 + 0.13 * index) if reference else (0.1 + 0.08 * index)
        poses = torch.eye(4).expand(count, -1, -1).clone()
        poses[:, 0, 0] = poses[:, 2, 2] = angle.cos()
        poses[:, 0, 2] = angle.sin()
        poses[:, 2, 0] = -angle.sin()
        poses[:, 0, 3] = (0.4 + 0.3 * index + reference_translation) if reference else (1.5 + 0.2 * index)
        poses[:, 1, 3] = (-0.2 + 0.12 * index) if reference else (-0.4 + 0.1 * index)
        poses[:, 2, 3] = (1.2 - 0.05 * index) if reference else (0.7 + 0.06 * index)
        return poses

    target_world = world_poses(total_frames, reference=False)
    reference_world = world_poses(reference_frames, reference=True)
    target_intrinsics = torch.tensor([0.85, 1.1, 0.49, 0.51, 0.04, -0.01, 0.003, -0.002]).repeat(total_frames, 1)
    reference_intrinsics = torch.tensor([0.72, 0.95, 0.53, 0.47, -0.07, 0.015, -0.005, 0.004]).repeat(
        reference_frames, 1
    )
    target_intrinsics[:, 0] += 0.03 * torch.arange(total_frames)
    reference_intrinsics[:, 0] += 0.05 * torch.arange(reference_frames) + reference_focal_delta
    reference_intrinsics[:, 4] += reference_distortion_delta
    if reference_order is not None:
        assert sorted(reference_order) == list(range(reference_frames)), "reference_order must be a permutation"
        order = torch.tensor(reference_order, dtype=torch.long)
        reference_world = reference_world.index_select(0, order)
        reference_intrinsics = reference_intrinsics.index_select(0, order)
        kwargs["neighbor_hidden_states"] = kwargs["neighbor_hidden_states"].index_select(2, order)

    def pixel_frames(tensor):
        return torch.cat([tensor[:1], tensor[1:].repeat_interleave(VAE_SCALE_FACTOR_TEMPORAL, dim=0)], dim=0)

    camera_inputs = dict(
        intrinsics={},  # Both full per-frame intrinsics tensors are supplied below.
        c2ws_world=pixel_frames(target_world),
        neighbor_c2ws_world=reference_world,
        scale=1.0,
        target_intrinsics=pixel_frames(target_intrinsics),
        neighbor_intrinsics=reference_intrinsics,
        include_neighbor_rays=True,
    )
    conditioning = compute_camera_conditioning(
        **camera_inputs,
        image_height=latent_hw[0] * VAE_SCALE_FACTOR_SPATIAL,
        image_width=latent_hw[1] * VAE_SCALE_FACTOR_SPATIAL,
    )
    kwargs.update({name: value.unsqueeze(0) for name, value in conditioning.items()})
    return kwargs


def _select_cosmos3_references(kwargs: dict, indexes: list[int]) -> dict:
    """Move or remove whole image/camera/ray tuples without changing the first-target origin."""
    selected = dict(kwargs)
    for name, dim in (
        ("neighbor_hidden_states", 2),
        ("neighbor_camera_rays", 1),
        ("neighbor_w2cs", 1),
        ("neighbor_Ks", 1),
    ):
        tensor = kwargs[name]
        selected[name] = tensor.index_select(dim, torch.tensor(indexes, dtype=torch.long, device=tensor.device))
    return selected


def _slice_cosmos3_target_kwargs(kwargs: dict, start: int, end: int) -> dict:
    """Absolute target-frame/pixel-window slices; references remain replicated."""
    sliced = dict(kwargs)
    sliced["hidden_states"] = kwargs["hidden_states"][:, :, start:end]
    pixel_start = 0 if start == 0 else (start - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1
    sliced["opacity"] = kwargs["opacity"][:, pixel_start : (end - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1]
    for name in ("camera_rays", "w2cs", "Ks"):
        if kwargs[name] is not None:
            sliced[name] = kwargs[name][:, start:end]
    timestep = kwargs.get("timestep")
    if isinstance(timestep, torch.Tensor) and timestep.ndim == 2:
        height, width = kwargs["hidden_states"].shape[-2:]
        spatial_tokens = (height // PATCH_SIZE) * (width // PATCH_SIZE)
        sliced["timestep"] = timestep[:, start * spatial_tokens : end * spatial_tokens]
    return sliced


def _cosmos3_noisy_sequence_indexes(base_kwargs: dict) -> torch.Tensor:
    """Resolve the native per-segment noisy-frame scatter into packed sequence rows."""
    indexes = []
    offset = 0
    for (frames, height, width), noisy in zip(
        base_kwargs["vision_token_shapes"], base_kwargs["vision_noisy_frame_indexes"], strict=True
    ):
        spatial = height * width
        indexes.append((offset + noisy[:, None] * spatial + torch.arange(spatial, device=noisy.device)).flatten())
        offset += frames * spatial
    return base_kwargs["vision_sequence_indexes"][torch.cat(indexes)]


@contextmanager
def _capture_cosmos3_forward(transformer: ArtifixerCosmos3Transformer):
    """Observe real public module calls without replacing packing or attention math."""
    layers = transformer.base_transformer.layers
    trace = SimpleNamespace(
        base_kwargs=[],
        gen_inputs=[[] for _ in layers],
        gen_outputs=[[] for _ in layers],
        und_outputs=[[] for _ in layers],
        attention_before_opacity=[[] for _ in layers],
        attention_after_opacity=[[] for _ in layers],
        attention_rotary=[[] for _ in layers],
        opacity_tokens=[[] for _ in layers],
        scope_queries=[[] for _ in layers],
        scope_keys=[[] for _ in layers],
        scope_rays=[[] for _ in layers],
        scope_outputs=[[] for _ in layers],
        prediction_inputs=[],
    )

    def snapshot(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(snapshot(item) for item in value)
        return value

    def capture_base(_module, _args, kwargs):
        trace.base_kwargs.append(snapshot(kwargs))

    def capture_input(records, position, keyword):
        def hook(_module, args, kwargs):
            records.append(snapshot(args[position] if len(args) > position else kwargs[keyword]))

        return hook

    def capture_output(records, index=None):
        def hook(_module, _args, output):
            records.append(snapshot(output if index is None else output[index]))

        return hook

    handles = [transformer.base_transformer.register_forward_pre_hook(capture_base, with_kwargs=True)]
    handles.append(
        transformer.base_transformer.proj_out.register_forward_pre_hook(
            capture_input(trace.prediction_inputs, 0, "input"), with_kwargs=True
        )
    )
    for index, layer in enumerate(layers):
        handles.extend(
            [
                layer.register_forward_pre_hook(capture_input(trace.gen_inputs[index], 1, "gen_seq"), with_kwargs=True),
                layer.register_forward_hook(capture_output(trace.gen_outputs[index], 1)),
                layer.register_forward_hook(capture_output(trace.und_outputs[index], 0)),
                layer.self_attn.register_forward_pre_hook(
                    capture_input(trace.attention_before_opacity[index], 1, "gen_seq"), with_kwargs=True, prepend=True
                ),
                layer.self_attn.register_forward_pre_hook(
                    capture_input(trace.attention_after_opacity[index], 1, "gen_seq"), with_kwargs=True
                ),
                layer.self_attn.register_forward_pre_hook(
                    capture_input(trace.attention_rotary[index], 2, "rotary_emb"), with_kwargs=True
                ),
                transformer.opacity_embeddings[index].register_forward_pre_hook(
                    capture_input(trace.opacity_tokens[index], 0, "input"), with_kwargs=True
                ),
                transformer.scope[index].register_forward_pre_hook(
                    capture_input(trace.scope_queries[index], 0, "query"), with_kwargs=True
                ),
                transformer.scope[index].register_forward_pre_hook(
                    capture_input(trace.scope_keys[index], 1, "key"), with_kwargs=True
                ),
                transformer.scope[index].register_forward_pre_hook(
                    capture_input(trace.scope_rays[index], 2, "rays"), with_kwargs=True
                ),
                transformer.scope[index].register_forward_hook(capture_output(trace.scope_outputs[index])),
            ]
        )
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def _assert_cosmos3_cp_gradient_mean(actual, expected, *, world_size: int, name: str) -> None:
    """The real-collective FSDP-mean oracle from test_cp_attention.py, also on Gloo."""
    import torch.distributed as dist

    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    present = torch.tensor(int(actual is not None), dtype=torch.long, device=device)
    dist.all_reduce(present, op=dist.ReduceOp.SUM)
    assert int(present) == (0 if expected is None else world_size), name
    if expected is None:
        return
    averaged = actual.detach().clone()
    dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
    averaged /= world_size
    assert torch.isfinite(averaged).all(), name
    scale = float(expected.abs().max().clamp_min(1e-3))
    deviation = float((averaged - expected).abs().max())
    assert (
        deviation <= 2e-3 * scale
    ), f"{name}: world-mean CP gradient differs from CP1 by {deviation:.3e} at gradient scale {scale:.3e}"


def _run_cp_workers(worker, tmp_path: Path, *worker_args, backend: str = "gloo") -> None:
    """Bounded distributed spawn/cleanup, following the existing test_causal_training.py harness."""
    import os
    import time

    import pytest
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("spawns a rank-symmetric process group; run this test without xdist")
    if not dist.is_available():
        pytest.skip("CP parity requires torch.distributed")
    if backend == "nccl":
        if not dist.is_nccl_available() or torch.cuda.device_count() < 2:
            pytest.skip("NCCL CP parity requires at least two CUDA devices")
    elif backend == "gloo":
        if not dist.is_gloo_available():
            pytest.skip("CPU CP parity requires Gloo")
    else:
        raise ValueError(f"Unsupported CP test backend: {backend}")
    if dist.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        pytest.skip("this test creates its own two-rank process group")
    context = mp.get_context("spawn")
    world_size = 2
    init_file = str(tmp_path / "cp_init")
    processes = [
        context.Process(target=worker, args=(rank, world_size, init_file, *worker_args, backend))
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + (300 if backend == "nccl" else 180)
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
    for rank, process in enumerate(processes):
        assert (
            process.exitcode == 0
        ), f"{backend}/{worker_args}: CP rank {rank} failed or timed out (exit {process.exitcode})"


class _ChunkFaithfulWanVae(torch.nn.Module):
    """Deterministic Wan-VAE double whose chunked encodes are exact.

    Encode follows the 1+4N temporal contract with no receptive field beyond
    each latent frame's own pixel window: latent frame 0 comes from pixel
    frame 0 alone, and every later latent frame is the mean of its
    ``scale_factor_temporal`` pixel frames (then spatial average pooling and
    channel tiling to ``z_dim``).  Encoding a left-complete window and
    cropping therefore equals the corresponding crop of the full-clip encode
    bitwise, which is what the pipeline's CP VAE-window slicing relies on.
    Decode inverts the layout deterministically, scaled small enough that
    ``postprocess_video`` never clamps and hides a latent mismatch.
    """

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, z_dim: int = 16, scale_factor_spatial: int = 8, scale_factor_temporal: int = 4):
        super().__init__()
        self.config = types.SimpleNamespace(
            scale_factor_temporal=scale_factor_temporal,
            scale_factor_spatial=scale_factor_spatial,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
            z_dim=z_dim,
        )

    def encode(self, rgb: torch.Tensor) -> types.SimpleNamespace:
        temporal = self.config.scale_factor_temporal
        latent_frames = (rgb.shape[2] - 1) // temporal + 1
        first = rgb[:, :, :1]
        rest = rgb[:, :, 1:]
        if latent_frames > 1:
            grouped = rest.unflatten(2, (latent_frames - 1, temporal)).mean(dim=3)
            frames = torch.cat([first, grouped], dim=2)
        else:
            frames = first
        spatial = self.config.scale_factor_spatial
        pooled = torch.nn.functional.avg_pool3d(frames, (1, spatial, spatial))
        repeats = (self.config.z_dim + pooled.shape[1] - 1) // pooled.shape[1]
        latents = pooled.repeat_interleave(repeats, dim=1)[:, : self.config.z_dim]
        # retrieve_latents(..., sample_mode="argmax") reads plain `latents`
        # from outputs without a latent_dist.
        return types.SimpleNamespace(latents=latents)

    def decode(self, latents: torch.Tensor, return_dict: bool = True) -> tuple[torch.Tensor]:
        assert not return_dict, "the pipeline always decodes with return_dict=False"
        temporal = self.config.scale_factor_temporal
        spatial = self.config.scale_factor_spatial
        rgb = latents[:, :3]
        video = torch.cat([rgb[:, :, :1], rgb[:, :, 1:].repeat_interleave(temporal, dim=2)], dim=2)
        video = video.repeat_interleave(spatial, dim=3).repeat_interleave(spatial, dim=4)
        return (0.05 * video,)


def argparse_default(script_path: str, arg_name: str):
    """Return the literal ``default=`` of ``add_argument(arg_name, ...)`` in a script.

    AST-based so the assertion is tied to the named argument rather than to a
    substring that could match any argument in the file.
    """
    tree = ast.parse((REPO_ROOT / script_path).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != arg_name:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
    raise AssertionError(f"Could not find default for {arg_name} in {script_path}")


class StopAtPromptLoad(Exception):
    """Raised by the recording payload loader once the call is recorded."""


def recording_prompt_loader(calls: list):
    """A load_prompt_payload_from_path stand-in that appends
    (prompt_path, prompt_format) to ``calls`` and raises StopAtPromptLoad, so
    dataset tests can assert the format reaches the loader without building a
    real payload."""

    def record(prompt_path, *, prompt_format):
        calls.append((prompt_path, prompt_format))
        raise StopAtPromptLoad

    return record


def make_capture_pipe(constructed: list):
    """A stub pipeline class whose constructor kwargs land in ``constructed``,
    with chainable ``.to`` and the ``spatial_alignment`` validate_model_profile
    consumes."""

    class _CapturePipe:
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            self.transformer = kwargs.get("transformer")
            self.vae = kwargs.get("vae")

        def to(self, *args, **kwargs):
            return self

        def spatial_alignment(self):
            # Wan2.2 TI2V profile alignment.
            return (32, 32)

    return _CapturePipe


def patch_kv_cache_pipeline_factories(monkeypatch, cosmos3_constructed: list, wan_constructed: list) -> None:
    """Stub get_kv_cache_pipe's hub-backed model factories and capture each
    family's pipeline constructor kwargs (pass the same list twice to record
    both families interleaved in call order)."""
    from model_training.utils import train_utils

    monkeypatch.setattr(train_utils, "Cosmos3OmniTransformer", SimpleNamespace(from_config=lambda *a, **k: None))
    monkeypatch.setattr(train_utils.WanTransformer3DModel, "from_config", classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr(train_utils.AutoencoderKLWan, "from_pretrained", lambda *a, **k: _TinyCosmosVae())
    monkeypatch.setattr(train_utils, "ArtifixerCosmos3KvCachePipeline", make_capture_pipe(cosmos3_constructed))
    monkeypatch.setattr(train_utils, "ArtifixerKvCachePipeline", make_capture_pipe(wan_constructed))
