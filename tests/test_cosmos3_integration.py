from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_processing.captioning.prompt_hdf5 import (
    COSMOS3_VIDEO_SYSTEM_PROMPT,
    cosmos3_training_frame_size,
    write_cosmos3_prompt_dataset,
)
from model_training.cosmos3_text import COSMOS3_DEFAULT_MAX_TEXT_TOKENS, pad_cosmos3_input_ids
from model_training.data.dl3dv_base import load_prompt_payload
from model_training.model_profiles import (
    COSMOS3_FLOW_SHIFT,
    COSMOS3_NANO_MODEL_ID,
    flow_shift_for_model,
    is_cosmos3_model,
    is_cosmos3_nano,
    spatial_resize_multiple,
)
from model_training.pipeline.cosmos3_pipeline import ArtifixerCosmos3Pipeline
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, prompt_payload_for_text_dropout
from tests._helpers import _TinyCosmosBaseTransformer, _TinyCosmosVae


def test_cosmos3_scheduler_uses_flow_prediction_without_scheduler_subfolder(monkeypatch):
    from model_training.utils import train_utils

    def fail_from_pretrained(*args, **kwargs):
        raise AssertionError("Cosmos3 scheduler should not require a scheduler subfolder")

    monkeypatch.setattr(train_utils.UniPCMultistepScheduler, "from_pretrained", fail_from_pretrained)

    scheduler = train_utils._scheduler_for_model(COSMOS3_NANO_MODEL_ID)

    assert scheduler.config.flow_shift == flow_shift_for_model(COSMOS3_NANO_MODEL_ID)
    assert scheduler.config.prediction_type == "flow_prediction"
    assert scheduler.config.use_flow_sigmas is True


def test_wan_scheduler_loads_from_model_repo(monkeypatch):
    from model_training.utils import train_utils

    calls = []

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return "scheduler"

    monkeypatch.setattr(train_utils.UniPCMultistepScheduler, "from_pretrained", fake_from_pretrained)

    scheduler = train_utils._scheduler_for_model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    assert scheduler == "scheduler"
    # Loaders always pass revision= from the pin
    # registry; unpinned Wan families resolve to None (hub default).
    assert calls == [
        (
            ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers",),
            {"subfolder": "scheduler", "torch_dtype": torch.bfloat16, "revision": None},
        )
    ]


def test_dcp_cache_conversion_uses_non_distributed_save(monkeypatch, tmp_path):
    from model_training.utils import train_utils

    save_calls = []

    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class FakeTransformerCls:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs["output_loading_info"]
            return FakeTransformer(), {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    class FakeWriter:
        def __init__(self, path):
            self.path = path

    def fake_save(state_dict, *, storage_writer, no_dist, **kwargs):
        save_calls.append((state_dict, storage_writer.path, no_dist, kwargs))

    monkeypatch.setattr(train_utils, "_transformer_cls_for_model", lambda model_id: FakeTransformerCls)
    monkeypatch.setattr(train_utils.dcp, "FileSystemWriter", FakeWriter)
    monkeypatch.setattr(train_utils.dcp, "save", fake_save)
    monkeypatch.setattr(train_utils.torch.cuda, "empty_cache", lambda: None)

    cache_path = train_utils._ensure_dcp_cache(COSMOS3_NANO_MODEL_ID, tmp_path)

    assert save_calls
    assert save_calls[0][2] is True
    assert save_calls[0][3] == {}
    assert save_calls[0][1] == str(cache_path)
    assert (cache_path / ".dcp_complete").is_file()


def test_replicated_dcp_load_uses_non_distributed_load(monkeypatch, tmp_path):
    from model_training.utils import train_utils

    load_calls = []

    class FakeReader:
        def __init__(self, path):
            self.path = path

    def fake_load(state_dict, *, storage_reader, no_dist, **kwargs):
        load_calls.append((state_dict, storage_reader.path, no_dist, kwargs))

    monkeypatch.setattr(train_utils.dcp, "FileSystemReader", FakeReader)
    monkeypatch.setattr(train_utils.dcp, "load", fake_load)

    state_dict = {"weight": torch.zeros(1)}
    train_utils._load_replicated_state_dict_from_dcp(state_dict, tmp_path)

    assert load_calls == [
        (
            {"model": state_dict},
            str(tmp_path),
            True,
            {},
        )
    ]


class _FakeCosmosTokenizer:
    eos_token_id = 7
    pad_token_id = 0

    def __init__(self):
        self.messages = []

    def convert_tokens_to_ids(self, token):
        assert token == "<|vision_start|>"
        return 8

    def apply_chat_template(self, conversations, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["add_vision_id"] is False
        assert kwargs["return_dict"] is True
        self.messages.append(conversations)
        text = conversations[-1]["content"]
        # The flat unconditional wraps the EMPTY caption in the
        # inverse-metadata ("The video is not ...") sentences.
        if text.startswith("The video is not "):
            return SimpleNamespace(input_ids=[21, 22])
        return SimpleNamespace(input_ids=[11, 12, 13])


def test_cosmos3_training_frame_size_matches_dataset_resize_rule():
    assert cosmos3_training_frame_size(720, 1280) == (704, 1280)
    assert cosmos3_training_frame_size(721, 1279) == (736, 1280)


def test_known_cosmos3_nano_profile():
    assert is_cosmos3_model(COSMOS3_NANO_MODEL_ID)
    assert is_cosmos3_nano(COSMOS3_NANO_MODEL_ID)
    assert spatial_resize_multiple(COSMOS3_NANO_MODEL_ID) == 32
    # The 480p-tier flow shift, not 10.
    assert flow_shift_for_model(COSMOS3_NANO_MODEL_ID) == COSMOS3_FLOW_SHIFT


def test_cosmos3_pipeline_constructs_pipeline_base_with_static_timesteps():
    transformer = _TinyCosmosBaseTransformer()

    pipe = ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=transformer,
        gradient_checkpointing=False,
        attention_backend="_native_flash",
    )

    assert pipe.expand_timesteps is False
    assert pipe.attention_backend == "_native_flash"
    assert transformer.backend == "_native_flash"
    for layer in pipe.transformer.base_transformer.layers:
        assert layer.self_attn.processor._attention_backend == "_native_flash"
    assert pipe._cp_frame_divisor(total_latent_frames=21) == 21
    assert pipe.supports_local_cp_inference()


def test_cosmos3_dummy_validation_prompt_is_one_token_per_branch():
    """Validation's dummy forwards feed cosmos3 token ids, not embeddings:
    one pad-free token for the prompt and one for the negative prompt."""
    pipe = ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
    )

    prompt, mask, negative_prompt, negative_mask = pipe.dummy_validation_prompt(torch.device("cpu"))

    assert prompt.shape == (1, 1) and prompt.dtype == torch.long
    assert mask.shape == (1, 1) and mask.dtype == torch.bool
    assert negative_prompt.shape == (1, 1) and negative_prompt.dtype == torch.long
    assert negative_mask.shape == (1, 1) and negative_mask.dtype == torch.bool


@pytest.mark.parametrize("requested,resolved", [(None, None), (None, "_flash_3"), ("native", "native")])
@pytest.mark.parametrize("ema", [False, True])
def test_distillation_cosmos3_auxiliary_transformers_reuse_attention_backend(monkeypatch, requested, resolved, ema):
    """Critics and EMA reuse the student's backend without changing global attention settings."""
    from model_training import stages
    from model_training.net import attention_config

    monkeypatch.setattr(stages, "_transformer_from_config", lambda cls, model_id: _TinyCosmosBaseTransformer())
    monkeypatch.setattr(attention_config, "_FLEX_KERNEL_OPTIONS", {"BACKEND": "FLASH"})
    pipe = SimpleNamespace(
        attention_backend=resolved,
        vae=SimpleNamespace(
            config=SimpleNamespace(scale_factor_spatial=16, scale_factor_temporal=4),
            device=torch.device("cpu"),
        ),
    )
    args = SimpleNamespace(
        model_id=COSMOS3_NANO_MODEL_ID,
        gradient_checkpointing=False,
        attention_backend=requested,
    )

    transformer = stages.build_score_transformer(args, COSMOS3_NANO_MODEL_ID, pipe, gradient_checkpointing=False)
    if ema:
        from model_training.utils.ema import ModelEMA
        from tests._helpers import _EMATestAccelerator

        state = ModelEMA(
            transformer,
            _EMATestAccelerator(),
            lambda: 0,
            ema_weight=0.99,
            ema_start_step=200,
            initialize_after_updates=0,
            training_stage="dmd",
            camera_pose_alignment="last",
        )
        state.restore_after_model_load()
        with state.apply():
            assert state.model is transformer
            assert transformer.base_transformer.backend == resolved
        assert all(not param.requires_grad for param in state.parameters())

    assert attention_config.flex_block_size() == (256, 128)
    assert transformer.base_transformer.backend == resolved
    for layer in transformer.base_transformer.layers:
        assert layer.self_attn.processor._attention_backend == resolved


def test_cosmos3_block_causal_pipeline_sizes_validation_cp_like_bidirectional():
    """The block-causal mask is built at global geometry and row-sliced per
    CP shard, so a diffusion-forcing pipe sizes validation CP exactly like
    the bidirectional path: any equal frame split works."""
    pipe = ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
        frames_per_block=7,
    )

    assert pipe._cp_frame_divisor(total_latent_frames=21) == 21
    # Candidate CP sizes are the head-gcd divisors (the tiny attention is
    # 4/4 heads): the wrapper pads its own frame axis, so neither the frame
    # count nor the block grid constrains the size, and one pass shards the
    # whole clip.
    assert pipe.cp_size_candidates(21, 8) == [1, 2, 4]
    assert pipe.cp_pass_latent_frames(21) == 21
    # The wrapper's block-causal mask indexes the full clip, so DF configs
    # cannot take pre-sliced shards.
    assert not pipe.supports_local_cp_inference()


def test_cosmos3_pipeline_accepts_diffusion_forcing_at_construction():
    """Supervised diffusion forcing (paper Sec 4.2 Initialization): the old
    ctor guard against unimplemented DF is deliberately lifted — cosmos3 +
    frames_per_block now
    constructs and the pipeline owns the block-causal contract. Nonsensical
    granularities still fail loudly."""
    pipe = ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
        frames_per_block=7,
    )

    assert pipe.causal_block_frames == 7
    assert pipe.expand_timesteps is False

    for bad_frames_per_block in (0, -3):
        with pytest.raises(ValueError, match="must be a positive latent-frame count"):
            ArtifixerCosmos3Pipeline(
                vae=_TinyCosmosVae(),
                scheduler=SimpleNamespace(),
                transformer=_TinyCosmosBaseTransformer(),
                gradient_checkpointing=False,
                frames_per_block=bad_frames_per_block,
            )


def test_cosmos3_pipeline_owns_block_causal_contract_with_no_transformer_decoy():
    """The pipeline is the single owner of the diffusion-forcing contract. The
    wrapper's hardcoded frames_per_block=None decoy is gone; trainers read the
    pipeline attribute, and no attribute literally named frames_per_block
    reappears on the transformer (decoy guard)."""
    pipe = ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
    )

    assert pipe.causal_block_frames is None
    assert not hasattr(pipe.transformer, "frames_per_block")

    df_pipe = ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
        frames_per_block=7,
    )

    assert df_pipe.causal_block_frames == 7
    # The wrapper carries the SAME long-named contract (it arms the mask) but
    # never an attribute literally named frames_per_block (decoy guard).
    assert df_pipe.transformer.causal_block_frames == 7
    assert not hasattr(df_pipe.transformer, "frames_per_block")


def test_get_pipe_forwards_frames_per_block_into_the_cosmos3_block_causal_contract(monkeypatch):
    """End-to-end DF wiring: get_pipe constructs the cosmos3 pipe with the
    pipeline-owned block-causal contract set from frames_per_block."""
    from model_training.utils import train_utils

    class _FakeTransformerCls:
        @staticmethod
        def from_config(*args, **kwargs):
            return _TinyCosmosBaseTransformer()

    monkeypatch.setattr(train_utils, "_transformer_cls_for_model", lambda model_id: _FakeTransformerCls)
    monkeypatch.setattr(train_utils.AutoencoderKLWan, "from_pretrained", lambda *args, **kwargs: _TinyCosmosVae())
    args = SimpleNamespace(
        model_id=COSMOS3_NANO_MODEL_ID,
        gradient_checkpointing=False,
        attention_backend=None,
    )

    df_pipe = train_utils.get_pipe(args, load_pretrained_transformer_weights=False, frames_per_block=7, device="cpu")
    assert df_pipe.causal_block_frames == 7

    pipe = train_utils.get_pipe(args, load_pretrained_transformer_weights=False, frames_per_block=None, device="cpu")
    assert pipe.causal_block_frames is None


def test_get_pipe_initializes_scope_identically_across_device_specific_rng_streams(monkeypatch):
    """Workers must start with identical SCoPE weights but keep separate random draws for training."""
    from model_training.utils import train_utils

    class _FakeTransformerCls:
        @staticmethod
        def from_config(*args, **kwargs):
            return _TinyCosmosBaseTransformer()

    monkeypatch.setattr(train_utils, "_transformer_cls_for_model", lambda model_id: _FakeTransformerCls)
    monkeypatch.setattr(train_utils.AutoencoderKLWan, "from_pretrained", lambda *args, **kwargs: _TinyCosmosVae())
    args = SimpleNamespace(
        model_id=COSMOS3_NANO_MODEL_ID,
        gradient_checkpointing=False,
        attention_backend=None,
        seed=42,
    )

    scope_states = []
    post_init_draws = []
    with torch.random.fork_rng(devices=[]):
        for process_index in (0, 1):
            # Simulate the different seed used by each worker.
            rank_generator = torch.Generator(device="cpu").manual_seed(args.seed + process_index)
            torch.set_rng_state(rank_generator.get_state())
            pipe = train_utils.get_pipe(
                args,
                load_pretrained_transformer_weights=False,
                frames_per_block=None,
                device="cpu",
            )
            scope_states.append(
                {name: value.detach().clone() for name, value in pipe.transformer.scope.state_dict().items()}
            )
            post_init_draws.append(torch.rand(4))

    assert scope_states[0].keys() == scope_states[1].keys()
    for name in scope_states[0]:
        assert torch.equal(scope_states[0][name], scope_states[1][name]), name
    # Identical starting weights must not make later random draws identical across workers.
    assert not torch.equal(post_init_draws[0], post_init_draws[1])


def test_pinned_init_rng_restores_exact_ambient_state():
    from model_training.net.rng import pinned_init_rng

    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(1234)
        torch.set_rng_state(generator.get_state())
        state_before = torch.get_rng_state().clone()
        with pinned_init_rng(42):
            torch.rand(16)
        assert torch.equal(torch.get_rng_state(), state_before)


def test_cosmos3_pipeline_prepares_fp32_transformer_timesteps():
    """Cosmos3 timesteps enter the transformer as fp32 —
    prepare_transformer_timestep owns the dtype policy, so bf16 latents must
    NOT quantize t (999 would round to 1000 on the bf16 grid)."""
    pipe = ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
    )
    latents = torch.zeros(2, 48, 1, 2, 2, dtype=torch.bfloat16)

    scalar = pipe.prepare_transformer_timestep(torch.tensor(999.0), latents)
    integer = pipe.prepare_transformer_timestep(torch.tensor(999, dtype=torch.long), latents)

    assert scalar.dtype == torch.float32
    assert scalar.tolist() == [999.0, 999.0]
    assert integer.dtype == torch.float32
    assert integer.tolist() == [999.0, 999.0]


def _df_pipe(frames_per_block: int) -> ArtifixerCosmos3Pipeline:
    return ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
        frames_per_block=frames_per_block,
    )


def test_cosmos3_per_block_timesteps_expand_sample_major_to_per_token_rows():
    """Diffusion-forcing expansion contract (paper Sec 4.2 Initialization):
    the trainer's flat (B * num_blocks,) SAMPLE-MAJOR timesteps (index =
    b * num_blocks + block, the add_noise_multi_timestep ordering) expand
    per-block -> per-frame -> per-token into exact
    [B, num_noisy_vision_tokens] fp32 — the shape the wrapper's strict
    _vision_timesteps contract already accepts. Spatial token count is exact
    division by the patch size (cosmos3 grids are always patch-divisible):
    latent 6x4 with patch 2 -> 3x2 = 6 tokens per frame."""
    pipe = _df_pipe(frames_per_block=3)
    # B=2, T=6 latent frames (2 blocks of 3), 6x4 spatial (patch-divisible).
    latents = torch.zeros(2, 48, 6, 6, 4, dtype=torch.bfloat16)
    per_block = torch.tensor([10.0, 20.0, 30.0, 40.0])  # [b0blk0, b0blk1, b1blk0, b1blk1]

    expanded = pipe.prepare_transformer_timestep(per_block, latents, frames_per_block=3)

    spatial_tokens_per_frame = 3 * 2
    tokens_per_block = 3 * spatial_tokens_per_frame
    assert expanded.shape == (2, 6 * spatial_tokens_per_frame)
    assert expanded.dtype == torch.float32
    assert expanded[0].tolist() == [10.0] * tokens_per_block + [20.0] * tokens_per_block
    assert expanded[1].tolist() == [30.0] * tokens_per_block + [40.0] * tokens_per_block


def test_cosmos3_per_block_expansion_passes_scalar_and_per_sample_timesteps_through():
    """Validation semantics: full-sequence sampling uses a scalar per-step
    timestep — scalar and per-sample [B] timesteps keep today's passthrough
    behavior (fp32, no per-token expansion) even with the DF contract set."""
    pipe = _df_pipe(frames_per_block=3)
    latents = torch.zeros(2, 48, 6, 4, 4, dtype=torch.bfloat16)

    scalar = pipe.prepare_transformer_timestep(torch.tensor(999.0), latents)
    per_sample = pipe.prepare_transformer_timestep(torch.tensor([5.0, 7.0]), latents)

    assert scalar.dtype == per_sample.dtype == torch.float32
    assert scalar.tolist() == [999.0, 999.0]
    assert per_sample.tolist() == [5.0, 7.0]


def test_cosmos3_per_block_expansion_keeps_the_audited_bug_class_guarded():
    """Regression guards around the expansion:
    - a 1-D tensor that is neither per-sample nor per-block passes through
      UNCHANGED, so the wrapper's strict per-block rejection still fires;
    - a pipe without the block-causal contract never expands (byte-identical
      non-DF behavior);
    - half-precision per-block timesteps are rejected instead of being
      laundered through the fp32 cast (the fp32 timestep island);
    - a frames_per_block kwarg that disagrees with the pipeline-owned
      contract fails loudly (the pipeline is the single owner)."""
    pipe = _df_pipe(frames_per_block=3)
    latents = torch.zeros(2, 48, 6, 4, 4, dtype=torch.bfloat16)

    wrong_size = pipe.prepare_transformer_timestep(torch.tensor([1.0, 2.0, 3.0]), latents)
    assert wrong_size.shape == (3,)
    assert wrong_size.tolist() == [1.0, 2.0, 3.0]

    non_df_pipe = ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=_TinyCosmosBaseTransformer(),
        gradient_checkpointing=False,
    )
    per_block_sized = torch.tensor([1.0, 2.0, 3.0, 4.0])
    passthrough = non_df_pipe.prepare_transformer_timestep(per_block_sized, latents)
    assert passthrough.shape == (4,)
    assert passthrough.tolist() == [1.0, 2.0, 3.0, 4.0]

    with pytest.raises(AssertionError, match="half-precision timesteps quantize"):
        pipe.prepare_transformer_timestep(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16), latents)

    with pytest.raises(AssertionError, match="single owner"):
        pipe.prepare_transformer_timestep(per_block_sized, latents, frames_per_block=7)


class _FakeHdf5Dataset:
    def __init__(self):
        self.attrs = {}

    def __getitem__(self, item):
        if item == slice(None):
            return self.data
        raise KeyError(item)


class _FakeHdf5File:
    def __init__(self):
        self.attrs = {}
        self.datasets = {}

    def create_dataset(self, name, data):
        dataset = _FakeHdf5Dataset()
        dataset.data = data
        self.datasets[name] = dataset
        return dataset


def test_cosmos3_prompt_writer_stores_negative_prompt_once():
    tokenizer = _FakeCosmosTokenizer()
    hf = _FakeHdf5File()

    write_cosmos3_prompt_dataset(
        hf,
        "frames_0",
        "a bright training scene",
        np.asarray([0, 1], dtype=np.int64),
        tokenizer,
        num_frames=2,
        height=704,
        width=1280,
        fps=24,
    )

    dataset = hf.datasets["frames_0"]
    # Fake tokenizer: the inverse-metadata negative -> [21, 22]; manual
    # [eos=7, <|vision_start|>=8] append.
    assert hf.attrs["negative_input_ids"].tolist() == [21, 22, 7, 8]
    assert int(hf.attrs["pad_token_id"]) == 0
    assert "negative_input_ids" not in dataset.attrs
    assert "pad_token_id" not in dataset.attrs
    assert float(dataset.attrs["fps"]) == 24.0
    # The raw caption stays retokenizable; only data attrs are written.
    assert dataset.attrs["caption"] == "a bright training scene"
    assert "caption_format" not in dataset.attrs
    assert "prompt_format" not in dataset.attrs


def test_cosmos3_prompt_writer_conversations_and_data_attrs():
    """The writer carries the system-prompt conversation to the tokenizer,
    wraps the empty-caption uncond in the NON-empty inverse-metadata
    negative, and stores the clip's TRUE fps — data attrs only, no format
    stamps."""
    tokenizer = _FakeCosmosTokenizer()
    hf = _FakeHdf5File()

    write_cosmos3_prompt_dataset(
        hf,
        "frames_0",
        "a bright training scene",
        np.asarray([0, 1], dtype=np.int64),
        tokenizer,
        num_frames=2,
        height=704,
        width=1280,
        fps=24,
    )

    dataset = hf.datasets["frames_0"]
    assert dataset.attrs["caption"] == "a bright training scene"
    assert sorted(dataset.attrs) == ["caption", "fps", "image_indices"]
    assert float(dataset.attrs["fps"]) == 24.0
    # Both conversations carry the system prompt, and the uncond user message
    # is the inverse-metadata text of the EMPTY caption — NOT the empty
    # string.
    cond_conversation, uncond_conversation = tokenizer.messages
    assert cond_conversation[0] == {"role": "system", "content": COSMOS3_VIDEO_SYSTEM_PROMPT}
    assert uncond_conversation[0] == {"role": "system", "content": COSMOS3_VIDEO_SYSTEM_PROMPT}
    assert cond_conversation[1]["content"].startswith("a bright training scene.")
    assert uncond_conversation[1]["content"].startswith("The video is not ")
    # Fake tokenizer: the inverse-metadata negative -> [21, 22]; manual
    # [eos=7, <|vision_start|>=8] append.
    assert hf.attrs["negative_input_ids"].tolist() == [21, 22, 7, 8]


def test_run_captioning_forwards_reverse_frames_and_prompt_format(monkeypatch, tmp_path):
    from data_processing import run_captioning

    calls = []

    def fake_generate_caption_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(run_captioning, "generate_caption_hdf5", fake_generate_caption_hdf5)
    # Frame-count resolution (short-scene frames_all collapse) reads the scene
    # zip; this test only pins flag forwarding, so skip the filesystem lookup.
    monkeypatch.setattr(run_captioning, "scene_limited_num_frames", lambda num_frames, scene_path: num_frames)
    # The outputs don't exist, so the lazy first-missing-output model load
    # would fire; this test only pins flag forwarding, so stub it out.
    monkeypatch.setattr(run_captioning, "load_models", lambda args: {})

    dl3dv_dir = tmp_path / "dl3dv"
    scene_path = dl3dv_dir / "subdir" / "scene.zip"
    args = SimpleNamespace(
        output_dir=tmp_path / "captions",
        dl3dv_dir=dl3dv_dir,
        hf_cache_dir=None,
        hf_hub_offline=False,
        max_frame_stride=1,
        include_reverse=True,
        include_revisit=False,
        num_frames=81,
        captioning_model_id="captioner",
        captioning_attn_implementation=None,
        prompt_format=COSMOS3_PROMPT_FORMAT,
        output_role="training",
        dataset_fps=29.97,
    )

    run_captioning.generate_for_scene(args, scene_path, models={})

    assert [kwargs["reverse_frames"] for _, kwargs in calls] == [False, True]
    assert all(kwargs["prompt_format"] == COSMOS3_PROMPT_FORMAT for _, kwargs in calls)
    assert all(kwargs["dataset_fps"] == 29.97 for _, kwargs in calls)
    assert all("reverse" not in kwargs for _, kwargs in calls)


class _FakePromptDataset:
    def __init__(self):
        self.attrs = {
            "caption": "caption",
            "image_indices": np.asarray([0, 1], dtype=np.int64),
        }
        self._data = np.asarray([11, 12, 7, 8], dtype=np.int32)

    def __getitem__(self, item):
        if item == slice(None):
            return self._data
        raise KeyError(item)


class _FakePromptFile:
    attrs = {
        "negative_input_ids": np.asarray([21, 7, 8], dtype=np.int32),
        "pad_token_id": 0,
    }


def test_pad_cosmos3_prompt_ids_preserves_masked_sequence_length():
    prompt_file = _FakePromptFile()
    dataset = _FakePromptDataset()
    encoded_prompt, prompt_attention_mask = pad_cosmos3_input_ids(
        dataset[:],
        pad_token_id=prompt_file.attrs["pad_token_id"],
    )
    negative_prompt, negative_prompt_attention_mask = pad_cosmos3_input_ids(
        prompt_file.attrs["negative_input_ids"],
        pad_token_id=prompt_file.attrs["pad_token_id"],
    )

    assert encoded_prompt.dtype == torch.long
    # Padded to the shared training cap: real ids up front, pad + False after.
    assert encoded_prompt.shape == prompt_attention_mask.shape == (COSMOS3_DEFAULT_MAX_TEXT_TOKENS,)
    assert encoded_prompt[:4].tolist() == [11, 12, 7, 8]
    assert not encoded_prompt[4:].any()
    assert prompt_attention_mask[:4].all() and not prompt_attention_mask[4:].any()
    assert negative_prompt[:3].tolist() == [21, 7, 8]
    assert not negative_prompt[3:].any()
    assert negative_prompt_attention_mask[:3].all() and not negative_prompt_attention_mask[3:].any()


def test_cosmos3_text_dropout_switches_to_unconditional_token_ids():
    data = {
        "encoded_prompt": torch.tensor([[11, 12, 7, 8, 0]]),
        "prompt_attention_mask": torch.tensor([[True, True, True, True, False]]),
        "encoded_negative_prompt": torch.tensor([[21, 7, 8, 0, 0]]),
        "negative_prompt_attention_mask": torch.tensor([[True, True, True, False, False]]),
    }

    processed = prompt_payload_for_text_dropout(data, drop_text=True)

    assert processed["encoded_prompt"].tolist() == [[21, 7, 8, 0, 0]]
    assert processed["prompt_attention_mask"].tolist() == [[True, True, True, False, False]]


def test_cosmos3_train_dropout_and_validation_uncond_share_the_empty_caption():
    """The 10% text dropout keeps the RATE but targets
    the EMPTY-caption unconditional (reference cosmos3 CFG uncond) stored as
    negative_input_ids — and the validation/CFG unconditional branch reads the
    SAME stored tensor, so train and eval unconditioning match by construction."""
    tokenizer = _FakeCosmosTokenizer()
    hf = _FakeHdf5File()
    write_cosmos3_prompt_dataset(
        hf,
        "frames_0",
        "a bright training scene",
        np.asarray([0, 1], dtype=np.int64),
        tokenizer,
        num_frames=2,
        height=704,
        width=1280,
        fps=24,
    )

    payload = load_prompt_payload(hf, hf.datasets["frames_0"], prompt_format=COSMOS3_PROMPT_FORMAT)
    data = {
        key: payload[key].unsqueeze(0)
        for key in (
            "encoded_prompt",
            "prompt_attention_mask",
            "encoded_negative_prompt",
            "negative_prompt_attention_mask",
        )
    }

    dropped = prompt_payload_for_text_dropout(data, drop_text=True)
    kept = prompt_payload_for_text_dropout(data, drop_text=False)

    # Fake tokenizer: chat-template("") -> [21, 22], then EOS 7 + <|vision_start|> 8.
    assert dropped["encoded_prompt"][0, :4].tolist() == [21, 22, 7, 8]
    assert torch.equal(dropped["encoded_prompt"], data["encoded_negative_prompt"])
    assert torch.equal(dropped["prompt_attention_mask"], data["negative_prompt_attention_mask"])
    assert torch.equal(kept["encoded_prompt"], data["encoded_prompt"])
