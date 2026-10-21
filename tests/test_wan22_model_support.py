from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model_eval.generation import SamplingConfig
from model_eval.validation import EvaluationRunner
from model_eval.validation_config import EvaluationConfig, EvaluationContext
from model_eval.validation_scoring import ValidationRecorder
from model_eval.validation_tasks import build_evaluation_plan
from model_training import stages
from model_training.constants import MAX_SEQUENCE_LENGTH
from model_training.data.utils import resize_to_multiple
from model_training.model_profiles import (
    flow_shift_for_model,
    is_wan22_ti2v_5b,
    spatial_resize_multiple,
    uses_expanded_timesteps,
)
from model_training.net import attention_config
from model_training.net.wan_transformer import (
    ArtifixerTransformerBlock,
    _apply_frame_gate,
    _modulated_layer_norm,
    _wan_block_frame_modulation_value,
    _wan_output_frame_modulation_value,
)
from model_training.pipeline.pipeline_base import ArtifixerPipelineBase
from model_training.pipeline.wan_kv_cache_pipeline import ArtifixerKvCachePipeline
from model_training.timestep import expand_transformer_timestep, spatial_alignment


class _IdentityAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.inner_dim = dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm_k = nn.RMSNorm(dim, eps=1e-6)

    def forward(self, hidden_states, *args, **kwargs):
        return torch.zeros_like(hidden_states)


class _BaseBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn1 = _IdentityAttention(dim)
        self.attn1.processor = type("Processor", (), {"_attention_backend": None})()
        self.attn2 = _IdentityAttention(dim)
        self.attn2.processor = type("Processor", (), {"_attention_backend": None})()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.Identity()
        self.ffn = nn.Identity()
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, dim))


def test_known_wan22_ti2v_5b_model_id_uses_expanded_timesteps():
    assert uses_expanded_timesteps("Wan-AI/Wan2.2-TI2V-5B-Diffusers")


def test_known_wan21_model_id_keeps_scalar_timesteps():
    assert not uses_expanded_timesteps("Wan-AI/Wan2.1-T2V-14B-Diffusers")
    assert not uses_expanded_timesteps("Wan-AI/Wan2.2-TI2V-5B-Diffusers-extra")


def test_prepare_transformer_timestep_expands_scalar_to_post_patch_tokens():
    hidden_states = torch.zeros(2, 48, 5, 4, 6)
    timestep = torch.tensor(123.0)

    expanded = expand_transformer_timestep(timestep, hidden_states, patch_size=(1, 2, 2))

    assert expanded.shape == (2, 5 * 2 * 3)
    assert torch.equal(expanded, torch.full_like(expanded, 123.0))


def test_prepare_transformer_timestep_expands_batch_timesteps():
    hidden_states = torch.zeros(2, 48, 5, 4, 6)
    timestep = torch.tensor([100.0, 200.0])

    expanded = expand_transformer_timestep(timestep, hidden_states, patch_size=(1, 2, 2))

    assert expanded.shape == (2, 5 * 2 * 3)
    assert torch.equal(expanded[0], torch.full_like(expanded[0], 100.0))
    assert torch.equal(expanded[1], torch.full_like(expanded[1], 200.0))


def test_prepare_transformer_timestep_expands_causal_block_timesteps():
    hidden_states = torch.zeros(2, 48, 6, 4, 6)
    timestep = torch.tensor([10.0, 20.0, 30.0, 40.0])

    expanded = expand_transformer_timestep(timestep, hidden_states, patch_size=(1, 2, 2), frames_per_block=3)

    tokens_per_block = 3 * 2 * 3
    assert expanded.shape == (2, 2 * tokens_per_block)
    assert torch.equal(expanded[0, :tokens_per_block], torch.full((tokens_per_block,), 10.0))
    assert torch.equal(expanded[0, tokens_per_block:], torch.full((tokens_per_block,), 20.0))
    assert torch.equal(expanded[1, :tokens_per_block], torch.full((tokens_per_block,), 30.0))
    assert torch.equal(expanded[1, tokens_per_block:], torch.full((tokens_per_block,), 40.0))


def test_prepare_transformer_timestep_validates_block_timestep_invariants():
    hidden_states = torch.zeros(1, 48, 6, 4, 6)
    timestep = torch.tensor([10.0, 20.0])

    with pytest.raises(ValueError, match="frames_per_block=3"):
        expand_transformer_timestep(timestep, hidden_states, patch_size=(2, 2, 2), frames_per_block=3)

    with pytest.raises(ValueError, match="latent_frames=6"):
        expand_transformer_timestep(
            timestep,
            hidden_states,
            patch_size=(1, 2, 2),
            frames_per_block=4,
        )


def test_spatial_alignment_uses_vae_scale_and_transformer_patch_size():
    assert spatial_alignment(16, (1, 2, 2)) == (32, 32)

    height_multiple, width_multiple = spatial_alignment(16, (1, 2, 2))
    assert 704 % height_multiple == 0
    assert 1280 % width_multiple == 0

    class _AlignmentPipe(ArtifixerPipelineBase):
        def patch_size(self):
            return (1, 2, 2)

        def forward_inference(self, *args, **kwargs):
            raise NotImplementedError

    pipe = object.__new__(_AlignmentPipe)
    pipe.vae = type("Vae", (), {"config": type("Config", (), {"scale_factor_spatial": 16})()})()
    with pytest.raises(ValueError, match="divisible by"):
        pipe._validate_spatial_alignment(720, 1280, label="video_frames")


def test_spatial_resize_multiple_tracks_wan22_patch_alignment():
    wan21_model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    wan22_model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert spatial_resize_multiple(wan21_model_id) == 16
    assert spatial_resize_multiple(wan22_model_id) == 32
    assert flow_shift_for_model(wan22_model_id) == 5.0


def test_artifixer_transformer_block_accepts_wan22_expanded_timestep_modulation():
    dim = 8
    batch_size = 2
    frames = 3
    frame_seqlen = 4
    seq_len = frames * frame_seqlen
    block = ArtifixerTransformerBlock(
        _BaseBlock(dim),
        opacity_embedding_dim=0,
        camera_embedding_dim=0,
        use_kv_cache_attention=False,
    )

    hidden_states = torch.randn(batch_size, seq_len, dim)
    # Expanded timesteps are constant within a latent frame; the frame-shaped
    # helpers rely on that when they subsample one timestep row per frame.
    temb = torch.randn(batch_size, frames, 6, dim).repeat_interleave(frame_seqlen, dim=1)
    output = block(
        hidden_states=hidden_states,
        encoder_hidden_states=torch.zeros(batch_size, 1, dim),
        neighbor_hidden_states=None,
        ignore_neighbors=False,
        temb=temb,
        rotary_emb=None,
        opacity_extra_patches=torch.empty(batch_size, seq_len, 0),
        camera_extra_patches=torch.empty(batch_size, seq_len, 0),
        kv_cache=None,
        layer_index=0,
        kv_cache_num_history_chunks=0,
        kv_cache_stage=None,
        context_kv_cache=None,
        frame_seqlen=frame_seqlen,
        prope_target=None,
        prope_reference=None,
        block_mask=None,
    )

    # With zero attention, identity ffn, and a zero scale_shift_table, the
    # block reduces to the ffn modulation term; computing it directly with the
    # dense token-expanded formula pins the chunk orientation and the
    # shift/scale/gate roles, not just the shape.
    _, _, _, c_shift, c_scale, c_gate = temb.float().chunk(6, dim=2)
    norm3 = torch.nn.LayerNorm(dim, elementwise_affine=False)
    expected = (
        hidden_states.float()
        + (norm3(hidden_states.float()) * (1 + c_scale.squeeze(2)) + c_shift.squeeze(2)) * c_gate.squeeze(2)
    ).type_as(hidden_states)
    assert output.shape == hidden_states.shape
    assert torch.allclose(output, expected, atol=1e-5)


def test_wan_frame_modulation_helpers_match_dense_fp32_reference():
    torch.manual_seed(0)
    batch_size = 2
    frames = 3
    frame_seqlen = 4
    dim = 16
    seq_len = frames * frame_seqlen
    kwargs = dict(batch_size=batch_size, hidden_seq_len=seq_len, frame_seqlen=frame_seqlen)

    table = torch.randn(1, 6, dim)
    frame_temb = torch.randn(batch_size, frames, 6, dim)
    temb = frame_temb.repeat_interleave(frame_seqlen, dim=1)
    hidden_states = torch.randn(batch_size, seq_len, dim)
    update = torch.randn(batch_size, seq_len, dim)

    # Dense reference: token-expanded scale_shift_table math in fp32.
    shift_ref, scale_ref, gate_ref, _, _, _ = (table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
    shift_ref, scale_ref, gate_ref = shift_ref.squeeze(2), scale_ref.squeeze(2), gate_ref.squeeze(2)
    norm = nn.LayerNorm(dim, elementwise_affine=False)
    norm_ref = norm(hidden_states.float()) * (1 + scale_ref) + shift_ref
    residual_ref = hidden_states.float() + update.float() * gate_ref

    shift = _wan_block_frame_modulation_value(table, temb, 0, **kwargs)
    scale = _wan_block_frame_modulation_value(table, temb, 1, **kwargs)
    gate = _wan_block_frame_modulation_value(table, temb, 2, **kwargs)
    # Token-expanded and per-frame temb layouts must produce the same values.
    assert torch.equal(gate, _wan_block_frame_modulation_value(table, frame_temb, 2, **kwargs))

    norm_out = _modulated_layer_norm(norm, hidden_states, scale, shift, frame_seqlen=frame_seqlen)
    residual_out = _apply_frame_gate(hidden_states, update, gate, frame_seqlen=frame_seqlen)
    assert torch.allclose(norm_out, norm_ref, atol=1e-6, rtol=0)
    assert torch.allclose(residual_out, residual_ref, atol=1e-6, rtol=0)

    # Output-projection modulation against the dense formula, applied the way
    # the transformer applies it.
    out_table = torch.randn(1, 2, dim)
    out_temb = torch.randn(batch_size, frames, dim).repeat_interleave(frame_seqlen, dim=1)
    out_shift_ref, out_scale_ref = (out_table.unsqueeze(0) + out_temb.unsqueeze(2)).chunk(2, dim=2)
    out_ref = norm(hidden_states.float()) * (1 + out_scale_ref.squeeze(2)) + out_shift_ref.squeeze(2)
    out_shift = _wan_output_frame_modulation_value(out_table, out_temb, 0, **kwargs)
    out_scale = _wan_output_frame_modulation_value(out_table, out_temb, 1, **kwargs)
    normed = norm(hidden_states.float()).unflatten(1, (frames, frame_seqlen))
    out_value = (normed * (1 + out_scale) + out_shift).flatten(1, 2)
    assert torch.allclose(out_value, out_ref, atol=1e-6, rtol=0)

    # fp32 accumulation pin: the gate stays fp32 and the residual accumulates
    # in fp32 with a single rounding at the end. On bf16 tensors the helper
    # must match the fp32 dense reference exactly, while bf16 accumulation
    # (gate downcast, bf16 product and add) visibly diverges on this data.
    hidden_bf16 = hidden_states.to(torch.bfloat16)
    update_bf16 = update.to(torch.bfloat16)
    residual_ref_bf16 = (hidden_bf16.float() + update_bf16.float() * gate_ref).to(torch.bfloat16)
    residual_out_bf16 = _apply_frame_gate(hidden_bf16, update_bf16, gate, frame_seqlen=frame_seqlen)
    bf16_accumulated = (
        hidden_bf16.unflatten(1, (frames, frame_seqlen))
        + update_bf16.unflatten(1, (frames, frame_seqlen)) * gate.to(torch.bfloat16)
    ).flatten(1, 2)
    assert torch.equal(residual_out_bf16, residual_ref_bf16)
    assert not torch.equal(bf16_accumulated, residual_ref_bf16)


class _PipeWithWan22Alignment:
    def spatial_alignment(self):
        return (32, 32)

    def dummy_validation_prompt(self, device):
        return torch.zeros(1, 4, 8, device=device, dtype=torch.bfloat16), None, None, None


def test_base_dummy_validation_prompt_reads_the_transformers_text_width():
    """The Wan-family dummy prompt is a zero embedding sized from the
    transformer's text embedder, in bfloat16 like dataset-encoded prompts,
    with no masks and no negative prompt."""
    pipe = SimpleNamespace(
        transformer=SimpleNamespace(
            condition_embedder=SimpleNamespace(text_embedder=SimpleNamespace(linear_1=SimpleNamespace(in_features=8)))
        )
    )

    prompt, mask, negative_prompt, negative_mask = ArtifixerPipelineBase.dummy_validation_prompt(
        pipe, torch.device("cpu")
    )

    assert prompt.shape == (1, MAX_SEQUENCE_LENGTH, 8) and prompt.dtype == torch.bfloat16
    assert mask is None and negative_prompt is None and negative_mask is None


def test_minimal_validation_task_uses_active_pipeline_spatial_alignment():
    trainer = EvaluationRunner(
        _PipeWithWan22Alignment(),
        EvaluationConfig(model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        EvaluationContext(torch.device("cpu")),
    )

    task = trainer._build_minimal_validation_task(
        "__dummy__",
        num_frames=5,
        num_latent_frames=2,
        num_inference_steps=1,
        device=torch.device("cpu"),
        vae_dtype=torch.bfloat16,
    )

    assert task.rendered_rgb.shape[-2:] == (32, 32)
    assert task.rendered_opacity.shape[-2:] == (32, 32)
    assert task.camera_rays.shape[-3:-1] == (32, 32)


class _TinyVae:
    device = torch.device("cpu")
    config = type("Config", (), {"scale_factor_temporal": 4})()


class _TinyKvTransformer:
    patch_size = (1, 1, 1)
    training = False
    blocks = [object(), object()]
    cp_mesh = None

    def __call__(self, *, hidden_states, timestep, **kwargs):
        expected_shape = (
            hidden_states.shape[0],
            hidden_states.shape[2] * hidden_states.shape[3] * hidden_states.shape[4],
        )
        assert timestep.shape == expected_shape
        return (torch.zeros_like(hidden_states),)


class _RecordingScheduler:
    num_train_timesteps = 1000
    timesteps = torch.linspace(999, 0, 1000)

    def __init__(self):
        self.step_timestep_shapes = []

    def step(self, model_output, timestep, sample, to_final=False):
        self.step_timestep_shapes.append(tuple(timestep.shape) if isinstance(timestep, torch.Tensor) else None)
        return sample

    def add_noise(self, original_samples, noise, timestep):
        return original_samples


def test_kv_cache_pipeline_expands_only_transformer_timestep():
    pipe = object.__new__(ArtifixerKvCachePipeline)
    pipe.vae = _TinyVae()
    pipe.transformer = _TinyKvTransformer()
    pipe.scheduler = _RecordingScheduler()
    pipe.expand_timesteps = True
    pipe.ar_frames_per_chunk = 2
    pipe.ar_window_chunks = None
    pipe.ar_sink_chunks = 0
    pipe.truncate_rollout_gradients = False
    pipe.prepare_latents = lambda condition, opacity, first_frame, **kwargs: torch.zeros_like(condition)
    condition = torch.zeros(1, 2, 2, 2, 2)
    rendered_opacity = torch.zeros(1, 5, 2, 2)
    camera_rays = torch.zeros(1, 2, 2, 2, 3)
    w2cs = torch.eye(4).view(1, 1, 4, 4).expand(1, 2, 4, 4)
    Ks = torch.eye(3).view(1, 1, 3, 3).expand(1, 2, 3, 3)

    output = pipe.generate_samples_from_batch(
        condition=condition,
        rendered_opacity=rendered_opacity,
        neighbors_condition=None,
        camera_rays=camera_rays,
        w2cs=w2cs,
        neighbor_w2cs=None,
        Ks=Ks,
        neighbor_Ks=None,
        encoded_prompt=torch.zeros(1, 1, 2),
        num_inference_steps=2,
        use_exit_flag=False,
    )

    assert output.shape == condition.shape
    assert pipe.scheduler.step_timestep_shapes == [(), ()]


def test_kv_cache_cp_rejects_chunks_that_do_not_split_across_ranks():
    # Window/sink alignment to whole chunks is constructor-enforced (see
    # test_kv_cache_pipeline_rejects_rolling_window_smaller_than_rollout_chunk);
    # the rollout itself must still reject a chunk size the CP mesh cannot
    # split evenly.
    pipe = object.__new__(ArtifixerKvCachePipeline)
    pipe.vae = _TinyVae()
    pipe.transformer = SimpleNamespace(
        patch_size=(1, 1, 1),
        cp_mesh=SimpleNamespace(size=lambda: 2),
        cp_rank=0,
        training=False,
    )
    pipe.ar_frames_per_chunk = 3
    pipe.ar_window_chunks = None
    pipe.ar_sink_chunks = 0

    with pytest.raises(AssertionError, match="divisible by CP size"):
        pipe.generate_samples_from_batch(
            condition=torch.zeros(1, 1, 6, 1, 1),
            rendered_opacity=torch.zeros(1, 21, 1, 1),
            neighbors_condition=None,
            camera_rays=torch.zeros(1, 6, 1, 1, 3),
            w2cs=torch.eye(4).view(1, 1, 4, 4).expand(1, 6, 4, 4),
            neighbor_w2cs=None,
            Ks=torch.eye(3).view(1, 1, 3, 3).expand(1, 6, 3, 3),
            neighbor_Ks=None,
            encoded_prompt=torch.zeros(1, 1, 2),
            num_inference_steps=1,
            use_exit_flag=False,
        )


def test_kv_cache_pipeline_advertises_chunk_aligned_cp_divisor():
    # The rollout shards every pass across the CP mesh, so valid CP sizes are
    # exactly the divisors of the chunk size — the pipeline owns that answer.
    pipe = object.__new__(ArtifixerKvCachePipeline)
    pipe.ar_frames_per_chunk = 8

    assert pipe._cp_frame_divisor(total_latent_frames=384) == 8
    # Blockwise cache state cannot be served by generic pre-sliced inference.
    assert not pipe.supports_local_cp_inference()


class _LatentDistribution:
    def __init__(self, sample):
        self._sample = sample

    def mode(self):
        return self._sample


class _RecordingVae:
    device = torch.device("cpu")
    dtype = torch.bfloat16
    config = type(
        "Config",
        (),
        {
            "scale_factor_spatial": 16,
            "scale_factor_temporal": 4,
            "z_dim": 2,
            "latents_mean": [0.0, 0.0],
            "latents_std": [1.0, 1.0],
        },
    )()

    def __init__(self):
        self.encoded = []

    def eval(self):
        return self

    def requires_grad_(self, requires_grad):
        return self

    def encode(self, rgb):
        assert rgb.device.type == self.device.type
        assert rgb.dtype == self.dtype
        self.encoded.append((rgb.device, rgb.dtype, tuple(rgb.shape)))
        # Input-derived latents (each view's spatial mean) so tests can pin
        # WHICH views land WHERE, not just output shapes.
        sample = rgb.mean(dim=(-2, -1), keepdim=True)[:, : self.config.z_dim]
        return type("EncodeOutput", (), {"latent_dist": _LatentDistribution(sample)})()


class _CpuRecordingVae(_RecordingVae):
    device = torch.device("cpu")


class _ConcretePipe(ArtifixerPipelineBase):
    def patch_size(self):
        return (1, 1, 1)

    def forward_inference(self, *args, **kwargs):
        raise NotImplementedError


class _TinyDistillationPipe:
    attention_backend = None
    vae = type(
        "Vae",
        (),
        {
            "device": torch.device("cpu"),
            "config": type(
                "Config",
                (),
                {
                    "scale_factor_spatial": 16,
                    "scale_factor_temporal": 4,
                },
            )(),
        },
    )()


class _RecordingBaseTransformer:
    """Stands in for WanTransformer3DModel.from_config's product: exposes
    set_attention_backend (like the real ModelMixin) and records what reached
    it, keeping the builder tests hermetic on GPU hosts where the resolved
    default backend is applied."""

    def __init__(self):
        self.attention_backend = None

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = backend


class _RecordingArtifixerTransformer(nn.Module):
    def __init__(
        self,
        transformer,
        *,
        frames_per_block,
        use_kv_cache_attention,
        vae_scale_factor_spatial,
        vae_scale_factor_temporal,
        gradient_checkpointing,
        checkpoint_every_n_blocks,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.transformer = transformer
        self.frames_per_block = frames_per_block
        self.use_kv_cache_attention = use_kv_cache_attention
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_every_n_blocks = checkpoint_every_n_blocks


def test_pipeline_non_expanded_scalar_timestep_expands_to_batch_vector():
    pipe = object.__new__(_ConcretePipe)
    pipe.expand_timesteps = False
    hidden_states = torch.zeros(3, 2, 4, 2, 2)

    timestep = pipe.prepare_transformer_timestep(torch.tensor(100.0), hidden_states)

    assert timestep.shape == (3,)
    assert torch.equal(timestep, torch.full((3,), 100.0))


def test_base_pipeline_rejects_nonpositive_frames_per_block():
    """The diffusion-forcing block size is validated ONCE, in the base
    pipeline every family constructs through — neither family carries a
    private copy of the check."""
    from tests._helpers import _TinyCosmosVae

    for bad in (0, -3):
        with pytest.raises(ValueError, match="must be a positive latent-frame count"):
            _ConcretePipe(
                _TinyCosmosVae(), tokenizer=None, text_encoder=None, expand_timesteps=False, frames_per_block=bad
            )


def test_pipeline_prepared_timestep_stays_fp32_for_bf16_latents():
    """prepare_transformer_timestep returns fp32 for every family — half
    precision quantizes the [0, 1000] timestep grid before the sinusoid
    projection, and the time embedders cast only after the sinusoid.
    prepare_transformer_timestep owns that dtype instead of the callers."""
    pipe = object.__new__(_ConcretePipe)
    pipe.expand_timesteps = False
    hidden_states = torch.zeros(3, 2, 4, 2, 2, dtype=torch.bfloat16)

    timestep = pipe.prepare_transformer_timestep(torch.tensor(100.0), hidden_states)

    assert timestep.dtype == torch.float32
    assert timestep.shape == (3,)


@pytest.mark.parametrize("attention_backend", [None, "native"])
def test_pipeline_base_constructor_keeps_runtime_settings(attention_backend):
    pipe = _ConcretePipe(
        _CpuRecordingVae(),
        tokenizer=None,
        text_encoder=None,
        expand_timesteps=True,
        attention_backend=attention_backend,
    )

    assert pipe.expand_timesteps is True
    assert pipe.attention_backend == attention_backend


def test_prepare_latents_pools_non_first_opacity_with_batch_dimension(monkeypatch):
    pipe = _ConcretePipe(_CpuRecordingVae(), tokenizer=None, text_encoder=None, expand_timesteps=False)
    pipe.transformer = SimpleNamespace(cp_mesh=None)
    condition = torch.ones(2, 3, 2, 2, 2)
    opacity = torch.ones(2, 8, 32, 32)

    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))

    latents = pipe.prepare_latents(condition, opacity, is_first_chunk=False)

    assert latents.shape == condition.shape
    torch.testing.assert_close(latents, condition)


def test_encode_neighbors_cap_splits_vae_calls_and_reassembles_all_views():
    # The validation memory cap: max_neighbors_per_encode=N encodes the
    # reference views N at a time (bounding VAE peak memory) and must
    # reassemble the same views in order; None keeps one batched call.
    pipe = _ConcretePipe(_CpuRecordingVae(), tokenizer=None, text_encoder=None, expand_timesteps=False)
    # Distinct per-view pixel values: the recording VAE echoes each view's
    # spatial mean into its latent, so a reordered or duplicated chunk
    # changes content. preprocess maps [0, 1] pixels to [-1, 1] latents.
    view_values = torch.tensor([0.25, 0.5, 0.75], dtype=torch.bfloat16)
    neighbors = view_values.view(1, 3, 1, 1, 1).expand(1, 3, 3, 16, 16).contiguous()

    unchunked = pipe.encode_neighbors(neighbors)
    assert len(pipe.vae.encoded) == 1

    pipe.vae.encoded.clear()
    chunked = pipe.encode_neighbors(neighbors, max_neighbors_per_encode=1)
    assert len(pipe.vae.encoded) == 3
    assert all(shape[2] == 1 for _, _, shape in pipe.vae.encoded)
    assert chunked.shape == unchunked.shape
    assert torch.equal(chunked, unchunked)
    for view_index, view_value in enumerate(view_values):
        # View i's content must land at views-dim index i.
        assert torch.all(chunked[0, :, view_index] == view_value * 2 - 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for VAE-device movement regression")
def test_encode_neighbors_moves_unchunked_cpu_neighbors_to_vae_device():
    vae = _RecordingVae()
    vae.device = torch.device("cuda:0")
    pipe = _ConcretePipe(vae, tokenizer=None, text_encoder=None, expand_timesteps=False)
    neighbors = torch.zeros(1, 2, 3, 16, 16, dtype=torch.float32)

    output = pipe.encode_neighbors(neighbors)

    assert output.device == vae.device
    assert output.shape == (1, 2, 2, 1, 1)
    assert pipe.vae.encoded[0][0].type == pipe.vae.device.type
    assert pipe.vae.encoded[0][1:] == (pipe.vae.dtype, (2, 3, 1, 16, 16))


def _distillation_args() -> SimpleNamespace:
    return SimpleNamespace(
        attention_backend=None,
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        frames_per_block=7,
        local_attn_size=21,
        sink_size=7,
        gradient_checkpointing=True,
        checkpoint_every_n_blocks=3,
    )


@pytest.mark.parametrize("requested,resolved", [(None, None), (None, "_flash_3"), ("native", "native")])
def test_distillation_score_transformer_uses_full_attention(monkeypatch, requested, resolved):
    monkeypatch.setattr(
        stages.WanTransformer3DModel, "from_config", lambda *args, **kwargs: _RecordingBaseTransformer()
    )
    monkeypatch.setattr(stages, "ArtifixerTransformer", _RecordingArtifixerTransformer)
    monkeypatch.setattr(attention_config, "_FLEX_KERNEL_OPTIONS", {"BACKEND": "FLASH"})
    args = _distillation_args()
    # The operator's backend choice must propagate to the critic transformer,
    # not just the student pipeline.
    args.attention_backend = requested
    pipe = _TinyDistillationPipe()
    pipe.attention_backend = resolved

    score = stages.build_score_transformer(
        args,
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        pipe,
        gradient_checkpointing=True,
    )

    assert score.transformer.attention_backend == resolved
    assert attention_config.flex_block_size() == (256, 128)
    assert score.frames_per_block is None
    assert score.use_kv_cache_attention is False
    assert score.gradient_checkpointing is True
    assert score.checkpoint_every_n_blocks == 3


@pytest.mark.parametrize("resolved", [None, "_flash_3", "native"])
def test_distillation_ema_preserves_the_existing_kv_cache_generator(monkeypatch, resolved):
    from model_training.utils.ema import ModelEMA
    from tests._helpers import _EMATestAccelerator

    monkeypatch.setattr(attention_config, "_FLEX_KERNEL_OPTIONS", {"BACKEND": "FLASH"})
    base = _RecordingBaseTransformer()
    base.set_attention_backend(resolved)
    generator = _RecordingArtifixerTransformer(
        base,
        frames_per_block=None,
        use_kv_cache_attention=True,
        vae_scale_factor_spatial=16,
        vae_scale_factor_temporal=4,
        gradient_checkpointing=True,
        checkpoint_every_n_blocks=3,
    )
    ema = ModelEMA(
        generator,
        _EMATestAccelerator(),
        lambda: 0,
        ema_weight=0.99,
        ema_start_step=200,
        initialize_after_updates=0,
        training_stage="dmd",
        camera_pose_alignment="mean",
    )
    ema.restore_after_model_load()
    with torch.no_grad():
        generator.weight.fill_(7)
    with ema.apply():
        torch.testing.assert_close(generator.weight, torch.ones_like(generator.weight))
        assert ema.model is generator

    # Rolling window / sink sizes now live on the KV-cache pipeline
    # (ar_window_chunks / ar_sink_chunks), not the wrapped transformer; the
    # EMA matches the generator by using KV-cache attention with no block mask.
    assert generator.transformer.attention_backend == resolved
    assert attention_config.flex_block_size() == (256, 128)
    assert generator.frames_per_block is None
    assert generator.use_kv_cache_attention is True
    assert generator.gradient_checkpointing is True
    assert generator.checkpoint_every_n_blocks == 3
    torch.testing.assert_close(generator.weight, torch.full_like(generator.weight, 7))


class _ValidationVae:
    dtype = torch.bfloat16
    config = type("Config", (), {"scale_factor_temporal": 4})()


class _ValidationTransformer:
    frames_per_block = None
    cp_mesh = None
    cp_rank = 0
    cp_world_size = 1

    def disable_context_parallel(self):
        return None


class _RecordingValidationPipe:
    vae = _ValidationVae()
    transformer = _ValidationTransformer()
    # Pipeline-owned block-causal contract: evaluation runner CP sizing
    # reads this off the pipe, not the transformer.
    causal_block_frames = None

    def __init__(self):
        self.max_neighbors_per_encode_values = []
        self.forwarded_kwargs = []

    # The base pipeline's CP-selection methods, bound onto this bare stub
    # (not an ArtifixerPipelineBase subclass).
    cp_head_divisor = ArtifixerPipelineBase.cp_head_divisor
    cp_size_candidates = ArtifixerPipelineBase.cp_size_candidates
    cp_pass_latent_frames = ArtifixerPipelineBase.cp_pass_latent_frames

    def _cp_frame_divisor(self, total_latent_frames):
        return 1

    def spatial_alignment(self):
        return (16, 16)

    def forward_inference(self, **kwargs):
        self.max_neighbors_per_encode_values.append(kwargs["max_neighbors_per_encode"])
        self.forwarded_kwargs.append(set(kwargs))
        return torch.zeros_like(kwargs["rendered_rgb"])


class _TinyValidationDataset:
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


def test_validation_forwards_neighbors_in_single_neighbor_vae_chunks(tmp_path):
    pipe = _RecordingValidationPipe()
    runner = EvaluationRunner(
        pipe,
        EvaluationConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            validation_metrics=False,
            sampling=SamplingConfig(num_inference_steps=1),
        ),
        EvaluationContext(torch.device("cpu")),
    )
    runner.recorder = ValidationRecorder(runner.config, runner.context, 4)
    runner.recorder.save = lambda *args, **kwargs: None
    datasets = {3: _TinyValidationDataset()}
    runner.generate_tasks(datasets, build_evaluation_plan(datasets, runner.config, 0).regular, tmp_path)

    assert pipe.max_neighbors_per_encode_values
    assert all(value == 1 for value in pipe.max_neighbors_per_encode_values)
    # The prompt masks ride the shared kwargs dict, so both the full-input and
    # local-CP branches receive them (cosmos3 needs them to trim pad tokens).
    assert all({"prompt_attention_mask", "negative_prompt_attention_mask"} <= keys for keys in pipe.forwarded_kwargs)


def test_resize_to_multiple_rounds_each_dimension_to_the_multiple():
    frames = torch.zeros(2, 3, 750, 1000)

    resized = resize_to_multiple(frames, 32)

    assert resized.shape[-2:] == (736, 992)
    aligned = torch.zeros(2, 3, 736, 992)
    assert resize_to_multiple(aligned, 32).shape[-2:] == (736, 992)


def test_spatial_alignment_orientation_with_asymmetric_patches():
    # Returns (height multiple, width multiple) from (t, h, w) patches — the
    # order consumers unpack. A transposed return is invisible to every
    # symmetric-patch test.
    assert spatial_alignment(16, (1, 2, 4)) == (32, 64)


def test_model_profile_detection_handles_local_paths_and_slashes():
    assert is_wan22_ti2v_5b("/ckpts/Wan2.2-TI2V-5B-Diffusers/")
    assert uses_expanded_timesteps("/ckpts/Wan2.2-TI2V-5B-Diffusers/")
    assert spatial_resize_multiple("/ckpts/Wan2.2-TI2V-5B-Diffusers/") == 32


def test_expand_transformer_timestep_passes_through_valid_expanded_input():
    # (batch, channels, frames=4, h=4, w=8) with (1, 2, 2) patches: 4 * 2 * 4 tokens.
    hidden_states = torch.zeros(2, 4, 4, 4, 8)
    expanded = torch.rand(2, 4 * 2 * 4)

    assert torch.equal(expand_transformer_timestep(expanded, hidden_states, (1, 2, 2)), expanded)

    with pytest.raises(ValueError, match="Expanded timestep shape"):
        expand_transformer_timestep(torch.rand(2, 3), hidden_states, (1, 2, 2))


def test_expand_transformer_timestep_handles_temporal_patching():
    # patch_t=2 halves the post-patch frames; per-block timesteps must spread
    # over post-patch tokens, not raw frames.
    hidden_states = torch.zeros(1, 4, 4, 4, 4)
    block_timesteps = torch.tensor([100.0, 200.0])

    expanded = expand_transformer_timestep(block_timesteps, hidden_states, (2, 2, 2), frames_per_block=2)

    tokens_per_block = (2 // 2) * (4 // 2) * (4 // 2)
    assert expanded.shape == (1, 2 * tokens_per_block)
    assert torch.equal(expanded[0, :tokens_per_block], torch.full((tokens_per_block,), 100.0))
    assert torch.equal(expanded[0, tokens_per_block:], torch.full((tokens_per_block,), 200.0))
