from types import SimpleNamespace

import pytest
import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from einops import rearrange
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_mask

import model_training.net.cosmos3_transformer as cosmos3_transformer_module
from model_training.net.block_mask import create_prefix_block_causal_block_mask
from model_training.net.cosmos3_transformer import (
    ArtifixerCosmos3AttnProcessor,
    ArtifixerCosmos3Transformer,
    conditioning_adapter_weight_norms,
    scope_to_query_ratios,
)
from model_training.net.scope import PLUCKER_DIM

# Canonical tiny Cosmos3 doubles shared with the pipeline/pins/flow-shift
# tests.
from tests._helpers import (
    LATENT_CHANNELS,
    PATCH_DIM,
    PATCH_SIZE,
    _FakeCpMesh,
    _GroupedFakeCpMesh,
    _identity_rotary,
    _ProcessorAttention,
    _TinyCosmosBaseTransformer,
    _TinyProcessorCosmosBaseTransformer,
    cosmos3_projected_gen_qkv,
    cosmos3_rankwise_all_to_all,
)
from tests._teacher_forcing_reference import cosmos3_teacher_forcing_reference

VAE_SCALE_FACTOR_SPATIAL = 2
VAE_SCALE_FACTOR_TEMPORAL = 4
CAMERA_PATCH_SIZE = VAE_SCALE_FACTOR_SPATIAL * PATCH_SIZE
# Full-pixel opacity footprint per token, f_o(PixelUnshuffle(O)):
# 1 channel x (vae_spatial * latent_patch)^2 pixels x vae_temporal frames.
OPACITY_TOKEN_DIM = CAMERA_PATCH_SIZE * CAMERA_PATCH_SIZE * VAE_SCALE_FACTOR_TEMPORAL


def _prime_scope_projections(transformer, *, seed: int = 23, std: float = 0.02) -> None:
    """Give the projections nonzero weights so camera changes can affect the output.

    Changing alpha alone has no effect while the projections are zero.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for injection in transformer.scope:
            injection.query_projection.weight.normal_(0.0, std, generator=generator)
            injection.key_projection.weight.normal_(0.0, std, generator=generator)


def _tiny_transformer(
    gradient_checkpointing: bool = False,
    base_transformer: nn.Module | None = None,
    causal_block_frames: int | None = None,
):
    return ArtifixerCosmos3Transformer(
        _TinyCosmosBaseTransformer() if base_transformer is None else base_transformer,
        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
        vae_scale_factor_temporal=VAE_SCALE_FACTOR_TEMPORAL,
        gradient_checkpointing=gradient_checkpointing,
        causal_block_frames=causal_block_frames,
    )


def _camera_tensors(batch: int = 1, target_frames: int = 1, reference_frames: int = 1):
    w2cs = torch.eye(4).view(1, 1, 4, 4).expand(batch, target_frames, -1, -1).clone()
    neighbor_w2cs = torch.eye(4).view(1, 1, 4, 4).expand(batch, reference_frames, -1, -1).clone()
    Ks = torch.eye(3).view(1, 1, 3, 3).expand(batch, target_frames, -1, -1).clone()
    neighbor_Ks = torch.eye(3).view(1, 1, 3, 3).expand(batch, reference_frames, -1, -1).clone()
    return w2cs, neighbor_w2cs, Ks, neighbor_Ks


def _forward_kwargs(hidden_states, neighbor_hidden_states):
    batch_size = hidden_states.shape[0]
    latent_frames = hidden_states.shape[2]
    w2cs, neighbor_w2cs, Ks, neighbor_Ks = _camera_tensors(
        batch=batch_size,
        target_frames=latent_frames,
    )
    return dict(
        hidden_states=hidden_states,
        timestep=torch.full((batch_size,), 0.5),
        encoder_hidden_states=torch.tensor([[11, 12, 7, 8]]).expand(batch_size, -1).clone(),
        encoder_attention_mask=torch.tensor([[True, True, True, True]]).expand(batch_size, -1).clone(),
        neighbor_hidden_states=neighbor_hidden_states,
        # Full pixel resolution (1+4N frames, spatial x VAE factor): the strict
        # opacity contract rejects anything else.
        opacity=torch.ones(
            batch_size,
            (latent_frames - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1,
            hidden_states.shape[-2] * VAE_SCALE_FACTOR_SPATIAL,
            hidden_states.shape[-1] * VAE_SCALE_FACTOR_SPATIAL,
        ),
        camera_rays=torch.zeros(
            batch_size,
            latent_frames,
            hidden_states.shape[-2] * VAE_SCALE_FACTOR_SPATIAL,
            hidden_states.shape[-1] * VAE_SCALE_FACTOR_SPATIAL,
            6,
        ),
        neighbor_camera_rays=(
            torch.zeros(
                batch_size,
                neighbor_hidden_states.shape[2],
                neighbor_hidden_states.shape[-2] * VAE_SCALE_FACTOR_SPATIAL,
                neighbor_hidden_states.shape[-1] * VAE_SCALE_FACTOR_SPATIAL,
                6,
            )
            if neighbor_hidden_states is not None
            else None
        ),
        w2cs=w2cs,
        neighbor_w2cs=neighbor_w2cs,
        Ks=Ks,
        neighbor_Ks=neighbor_Ks,
    )


# --- fp32 timestep path ---

# transformer_cosmos3.py config default: t in [0, 1000] is scaled by 0.001
# inside the base transformer before the sinusoid.
COSMOS3_TIMESTEP_SCALE = 0.001


def test_vision_timesteps_are_always_fp32():
    """Vision timesteps enter the base transformer in fp32 — never the
    (bf16) compute dtype, which quantizes [0, 1000] before the sinusoid.
    (Signature carries batch_size for the strict shape contract.)"""
    transformer = _tiny_transformer()

    per_sample = transformer._vision_timesteps(torch.tensor([0.25, 0.75]), 1, 2, 3, torch.device("cpu"))
    per_token = transformer._vision_timesteps(
        torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        0,
        1,
        3,
        torch.device("cpu"),
    )

    assert per_sample.dtype == per_token.dtype == torch.float32
    assert per_sample.tolist() == [0.75, 0.75, 0.75]
    assert per_token.tolist() == pytest.approx([0.1, 0.2, 0.3])

    # Scalars and 0-dim tensors are rejected, not broadcast: permissive
    # fallbacks are how per-block timesteps were once silently misread.
    for bad in (999.0, torch.tensor(0.5)):
        with pytest.raises(AssertionError, match="ndim >= 1"):
            transformer._vision_timesteps(bad, 0, 1, 3, torch.device("cpu"))


def test_vision_timesteps_reject_half_precision_timesteps():
    """Mirror the reference's "Timestep must be long/float32" contract —
    an already-quantized half-precision t must fail loudly, not embed silently."""
    transformer = _tiny_transformer()

    for dtype in (torch.bfloat16, torch.float16):
        with pytest.raises(AssertionError, match="half-precision timesteps quantize"):
            transformer._vision_timesteps(torch.tensor(999.0, dtype=dtype), 0, 1, 3, torch.device("cpu"))


# --- diffusion-forcing hard guard ---


def test_vision_timesteps_reject_per_block_1d_timesteps():
    """The old permissive 1-D fallback consumed diffusion-forcing per-block
    [B * num_blocks] timesteps as per-sample (all vision tokens of sample i got
    block i's timestep). Only exact per-sample [B] is accepted now."""
    transformer = _tiny_transformer()
    per_block = torch.rand(2 * 3)  # batch 2, 3 blocks

    with pytest.raises(ValueError, match=r"per-sample \[B\]=2.*Per-block diffusion-forcing timesteps"):
        transformer._vision_timesteps(per_block, 0, 2, 5, torch.device("cpu"))


def test_vision_timesteps_reject_mismatched_2d_timesteps():
    """Mismatched 2-D timesteps previously fell back to flatten()[0]
    silently; only exact [B, num_noisy_vision_tokens] is accepted now (the
    scatter-src row count — equal to all vision tokens outside SF-DMD
    conditioning-mode forwards)."""
    transformer = _tiny_transformer()

    with pytest.raises(ValueError, match=r"exactly \[B, num_noisy_vision_tokens\]"):
        transformer._vision_timesteps(torch.rand(1, 4), 0, 1, 3, torch.device("cpu"))
    with pytest.raises(ValueError, match=r"exactly \[B, num_noisy_vision_tokens\]"):
        transformer._vision_timesteps(torch.rand(2, 3), 0, 1, 3, torch.device("cpu"))


def test_vision_timesteps_reject_higher_rank_timesteps():
    transformer = _tiny_transformer()

    with pytest.raises(ValueError, match=r"scalar, \[B\], or \[B, num_noisy_vision_tokens\]"):
        transformer._vision_timesteps(torch.rand(1, 3, 1), 0, 1, 3, torch.device("cpu"))


def test_forward_rejects_per_block_diffusion_forcing_timesteps():
    """End-to-end: a [B * num_blocks] timestep tensor (what the trainer
    samples when frames_per_block is set) must raise, not train silently wrong."""
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["timestep"] = torch.rand(3)  # batch 1, but 3 per-block entries

    with pytest.raises(ValueError, match="Per-block diffusion-forcing timesteps are not supported"):
        transformer(**kwargs)


def test_cosmos3_transformer_has_no_frames_per_block_decoy():
    """The wrapper used to hardcode frames_per_block=None
    purely so trainers could read it, disagreeing with the Trainer's
    args.frames_per_block. The pipeline is the single owner now; the wrapper's
    diffusion-forcing ctor attribute deliberately carries the LONG name."""
    assert not hasattr(_tiny_transformer(), "frames_per_block")
    df_transformer = _tiny_transformer(causal_block_frames=7)
    assert not hasattr(df_transformer, "frames_per_block")
    assert df_transformer.causal_block_frames == 7
    with pytest.raises(ValueError, match="must be a positive latent-frame count"):
        _tiny_transformer(causal_block_frames=0)


def test_forward_rejects_unknown_kwargs_instead_of_swallowing_them():
    """forward's **kwargs swallow-all is gone, so a
    typo'd or stale conditioning kwarg fails loudly instead of silently
    dropping a load-bearing input."""
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["camera_raise"] = kwargs["camera_rays"]  # typo'd kwarg

    with pytest.raises(TypeError, match="camera_raise"):
        transformer(**kwargs)


# --- block-causal mask for supervised diffusion forcing ---


def _gen_block_mask(und_len: int, num_frames: int, block_frames: int, spatial_tokens_per_frame: int) -> BlockMask:
    """The wrapper's mask geometry through the shared block_mask helper:
    text prefix always visible, vision tokens block-causal."""
    return create_prefix_block_causal_block_mask(
        prefix_len=und_len,
        total_tokens=num_frames * spatial_tokens_per_frame,
        frames_per_block=block_frames,
        frame_seqlen=spatial_tokens_per_frame,
        device="cpu",
        block_size=128,
    )


def _dense_mask(block_mask: BlockMask) -> torch.Tensor:
    """Materialize a BlockMask's mask_mod into the boolean [Q, K] keep-mask."""
    q_len, kv_len = block_mask.shape[-2:]
    return create_mask(block_mask.mask_mod, None, None, q_len, kv_len, "cpu")[0, 0]


def test_block_causal_block_mask_is_causal_across_blocks_and_bidirectional_within():
    """Mask unit test at 1-frame blocks (block == frame): the gen side is
    block-causal — gen key k visible to gen row q iff block(k) <= block(q) —
    so earlier blocks are masked from every later block's keys (a write-once
    seeded KV cache computes frame 0's K/V before frame 1 exists, and this
    mask trains exactly that function). Final-block rows attend everywhere;
    every gen row keeps all text columns (text rows themselves never see
    gen keys)."""
    und_len = 3

    mask = _dense_mask(_gen_block_mask(und_len, num_frames=3, block_frames=1, spatial_tokens_per_frame=2))

    # Gen layout: 6 vision rows (frames 0,0,1,1,2,2); block_frames=1 =>
    # block == frame.
    block_of_gen_token = torch.tensor([0, 0, 1, 1, 2, 2])
    num_gen = block_of_gen_token.numel()
    assert mask.shape == (num_gen, und_len + num_gen)
    assert mask.dtype == torch.bool
    assert bool(mask[:, :und_len].all()), "every gen row keeps the text columns"

    expected_gen_block = block_of_gen_token.unsqueeze(-1) >= block_of_gen_token.unsqueeze(0)
    assert torch.equal(mask[:, und_len:], expected_gen_block)
    assert bool(mask[block_of_gen_token == 2].all()), "final-block rows attend everywhere"
    # Earliest block: keeps only its own keys — masked from BOTH later blocks.
    frame0_gen_columns = mask[block_of_gen_token == 0][:, und_len:]
    assert bool(frame0_gen_columns[:, block_of_gen_token == 0].all())
    assert not bool(
        frame0_gen_columns[:, block_of_gen_token >= 1].any()
    ), "frame-0 rows saw later-block keys — a seeded write-once KV cache can never reproduce that"
    # Middle block: frames <= 1 only.
    frame1_gen_columns = mask[block_of_gen_token == 1][:, und_len:]
    assert bool(frame1_gen_columns[:, block_of_gen_token <= 1].all())
    assert not bool(frame1_gen_columns[:, block_of_gen_token == 2].any())


def test_block_causal_block_mask_multi_frame_blocks_are_bidirectional_within():
    """Multi-frame blocks: bidirectional within a block, causal across
    blocks, text columns always visible."""
    und_len = 3

    mask = _dense_mask(_gen_block_mask(und_len, num_frames=4, block_frames=2, spatial_tokens_per_frame=2))

    # Gen layout: 8 vision rows (frames 0,0,1,1,2,2,3,3); block = frame // 2.
    block_of_gen_token = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    num_gen = block_of_gen_token.numel()
    assert mask.shape == (num_gen, und_len + num_gen)
    assert bool(mask[:, :und_len].all()), "every gen row keeps the text columns"
    expected_gen_block = block_of_gen_token.unsqueeze(-1) >= block_of_gen_token.unsqueeze(0)
    assert torch.equal(mask[:, und_len:], expected_gen_block)
    # Block 0 rows never see block-1 keys; block-1 rows attend everything.
    block0_columns = mask[block_of_gen_token == 0][:, und_len:]
    assert not bool(block0_columns[:, block_of_gen_token == 1].any())
    assert bool(block0_columns[:, block_of_gen_token == 0].all())
    assert bool(mask[block_of_gen_token == 1].all())


def test_block_causal_block_mask_prefix_rule_matches_int_reference_across_prefix_lengths():
    """prefix_len is captured as a 0-dim device tensor inside
    build_prefix_block_causal_mask_mod (not a Python int), because cosmos3
    rebuilds this mask with a new prefix_len every training step (the
    per-item caption token count) and a captured int gets promoted to a
    SymInt on the second distinct value, which the FLASH backend cannot
    compile. Check the tensor-captured rule still matches a plain-int
    reference for several distinct prefix_len values."""
    num_frames, block_frames, spatial_tokens_per_frame = 4, 2, 2
    block_token_count = block_frames * spatial_tokens_per_frame
    num_gen = num_frames * spatial_tokens_per_frame

    def expected(q_idx: int, kv_idx: int, prefix_len: int) -> bool:
        if kv_idx < prefix_len:
            return True
        gen_kv_idx = kv_idx - prefix_len
        block_end = (q_idx // block_token_count + 1) * block_token_count
        return gen_kv_idx < block_end

    for und_len in (0, 3, 5):
        mask = _dense_mask(_gen_block_mask(und_len, num_frames, block_frames, spatial_tokens_per_frame))
        assert mask.shape == (num_gen, und_len + num_gen)
        for q_idx in range(num_gen):
            for kv_idx in range(und_len + num_gen):
                assert bool(mask[q_idx, kv_idx]) == expected(q_idx, kv_idx, und_len), (und_len, q_idx, kv_idx)


def test_block_causal_block_mask_query_axis_spans_all_gen_rows():
    """The builder has no per-rank query window: under CP the Ulysses
    exchange hands every rank the full global gen sequence, so the one
    global mask covers every gen row over the full [text ⊕ gen] key axis
    on every rank."""
    mask = create_prefix_block_causal_block_mask(
        prefix_len=3,
        total_tokens=6,
        frames_per_block=1,
        frame_seqlen=2,
        device="cpu",
        block_size=128,
    )
    assert mask.shape[-2:] == (6, 9)


def test_block_causal_forward_validates_the_block_grid():
    """block(t) is only defined on a uniform block grid, and the block size
    must be a positive latent-frame count — drifted callers fail loudly."""
    with pytest.raises(ValueError, match="positive latent-frame count"):
        _tiny_transformer(causal_block_frames=0)

    transformer = _tiny_transformer(causal_block_frames=3)
    kwargs = _forward_kwargs(torch.randn(1, LATENT_CHANNELS, 4, 4, 4), neighbor_hidden_states=None)
    with pytest.raises(AssertionError, match="uniform block grid"):
        transformer(**kwargs)


def test_block_causal_block_mask_is_cached_across_identical_forwards(monkeypatch):
    """The BlockMask is a pure function of the packed geometry, and each
    build transiently materializes a full [Q, K] boolean mask, so repeated
    supervised forwards over the same geometry must reuse the cached
    BlockMask; a new frame count rebuilds."""
    calls: list[dict] = []
    original_builder = cosmos3_transformer_module.create_prefix_block_causal_block_mask

    def counting_builder(**kwargs):
        calls.append(kwargs)
        return original_builder(**kwargs)

    monkeypatch.setattr(cosmos3_transformer_module, "create_prefix_block_causal_block_mask", counting_builder)

    transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer(), causal_block_frames=2)
    hidden_states = torch.randn(1, LATENT_CHANNELS, 4, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)

    with torch.no_grad():
        first = transformer(**kwargs)[0]
        second = transformer(**kwargs)[0]
    assert len(calls) == 1, "identical supervised forwards must reuse the cached BlockMask"
    torch.testing.assert_close(first, second)

    shorter_kwargs = _forward_kwargs(torch.randn(1, LATENT_CHANNELS, 2, 4, 4), neighbor_hidden_states=None)
    with torch.no_grad():
        transformer(**shorter_kwargs)
    assert len(calls) == 2, "a new frame count must rebuild the BlockMask"
    assert calls[1]["total_tokens"] == 2 * 4


def test_block_causal_block_mask_cache_holds_both_lengths_of_a_cfg_pair(monkeypatch):
    """CFG alternates the prompt length (conditional vs unconditional) on
    every pair of forwards; the cache must keep both branches resident so a
    50-step CFG validation builds each BlockMask once, not once per
    forward."""
    calls: list[dict] = []
    original_builder = cosmos3_transformer_module.create_prefix_block_causal_block_mask

    def counting_builder(**kwargs):
        calls.append(kwargs)
        return original_builder(**kwargs)

    monkeypatch.setattr(cosmos3_transformer_module, "create_prefix_block_causal_block_mask", counting_builder)
    transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer())

    shared = dict(
        global_latent_frames=4,
        block_frames=2,
        spatial_tokens_per_frame=4,
        device="cpu",
    )
    for _ in range(3):  # three CFG pairs
        cond = transformer._armed_block_causal_block_mask(und_len=5, **shared)
        uncond = transformer._armed_block_causal_block_mask(und_len=3, **shared)
    assert len(calls) == 2, "alternating CFG prompt lengths must both stay cached"
    assert cond.shape[-1] != uncond.shape[-1]


def test_attn_processor_gen_mask_blocks_earlier_rows_from_later_keys():
    """Processor semantics: with a block-causal BlockMask the text branch and
    the final block's rows are unchanged, earlier blocks' rows drop the later
    keys, and the masked output equals a manual SDPA reference with the same
    mask."""
    torch.manual_seed(0)
    attention = _ProcessorAttention()
    processor = ArtifixerCosmos3AttnProcessor()
    und_seq = torch.randn(3, attention.hidden_size)
    gen_seq = torch.randn(6, attention.hidden_size)  # 3 frames x 2 tokens
    rotary = _identity_rotary(und_seq.shape[0], gen_seq.shape[0], attention.head_dim)
    block_mask = _gen_block_mask(und_len=3, num_frames=3, block_frames=1, spatial_tokens_per_frame=2)

    und_out, gen_out = processor(attention, und_seq, gen_seq, rotary)
    processor._gen_block_mask = block_mask
    und_out_masked, gen_out_masked = processor(attention, und_seq, gen_seq, rotary)

    torch.testing.assert_close(und_out_masked, und_out, rtol=0, atol=0)
    torch.testing.assert_close(gen_out_masked[4:], gen_out[4:], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(gen_out_masked[:4], gen_out[:4]), "earlier blocks' rows must stop seeing later keys"

    # Ground truth: identity projections + identity rotary, so the masked gen
    # branch is plain SDPA over [text ⊕ gen] keys with the boolean mask.
    heads, head_dim = attention.num_attention_heads, attention.head_dim
    query = gen_seq.view(1, -1, heads, head_dim).transpose(1, 2)
    keys = torch.cat([und_seq, gen_seq], dim=0).view(1, -1, heads, head_dim).transpose(1, 2)
    reference = torch.nn.functional.scaled_dot_product_attention(query, keys, keys, attn_mask=_dense_mask(block_mask))
    reference = reference.transpose(1, 2).reshape(6, attention.hidden_size)
    torch.testing.assert_close(gen_out_masked, reference, atol=1e-6, rtol=1e-6)


def _recording_dispatch(calls):
    def fake_dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=False,
        backend=None,
        parallel_config=None,
        **_kwargs,
    ):
        calls.append(
            {
                "backend": backend,
                "attn_mask_shape": None if attn_mask is None else tuple(attn_mask.shape),
                "is_causal": is_causal,
            }
        )
        output = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
        return output.transpose(1, 2)

    return fake_dispatch_attention_fn


@pytest.mark.parametrize("backend", [None, "_flash_3"])
def test_attn_processor_masked_gen_call_bypasses_dispatch(monkeypatch, backend):
    """The masked gen branch runs flex_attention (SDPA on CPU) directly and
    never routes through dispatch_attention_fn — the block-causal semantics
    live in the BlockMask, independent of the configured diffusers backend —
    while the unmasked text branch keeps the configured backend."""
    torch.manual_seed(0)
    calls = []
    monkeypatch.setattr(cosmos3_transformer_module, "dispatch_attention_fn", _recording_dispatch(calls))
    attention = _ProcessorAttention()
    processor = ArtifixerCosmos3AttnProcessor()
    processor._attention_backend = backend
    processor._gen_block_mask = _gen_block_mask(und_len=3, num_frames=3, block_frames=1, spatial_tokens_per_frame=2)

    processor(
        attention,
        torch.randn(3, attention.hidden_size),
        torch.randn(6, attention.hidden_size),
        _identity_rotary(3, 6, attention.head_dim),
    )

    assert [call["is_causal"] for call in calls] == [True], "only the text branch may reach dispatch"
    assert [call["attn_mask_shape"] for call in calls] == [None]
    assert [call["backend"] for call in calls] == [backend]


# --- supervised diffusion forcing: block-causal mask arming ---


def test_block_causal_supervised_forward_arms_mask_on_every_layer(monkeypatch):
    """Supervised diffusion forcing (paper Sec 4.2 Initialization): with the
    ctor contract set, every full-sequence forward arms the full-sequence
    block-causal BlockMask on every layer's processor, and the
    post-attention disarm hook clears the state after the pass. Without the
    ctor contract the same forward stays maskless (byte-identical non-DF
    behavior)."""
    base = _TinyProcessorCosmosBaseTransformer()
    transformer = _tiny_transformer(base_transformer=base, causal_block_frames=1)
    hidden_states = torch.randn(1, LATENT_CHANNELS, 3, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)

    seen: list[BlockMask | None] = []
    original_call = ArtifixerCosmos3AttnProcessor.__call__

    def recording_call(self, *args, **kw):
        seen.append(self._gen_block_mask)
        return original_call(self, *args, **kw)

    monkeypatch.setattr(ArtifixerCosmos3AttnProcessor, "__call__", recording_call)

    output = transformer(**kwargs)[0]

    assert output.shape == hidden_states.shape
    assert len(seen) == len(base.layers)
    expected_mask = _dense_mask(_gen_block_mask(und_len=4, num_frames=3, block_frames=1, spatial_tokens_per_frame=4))
    for block_mask in seen:
        assert block_mask is not None and torch.equal(_dense_mask(block_mask), expected_mask)
    # Post-attention disarm clears the supervised mask after the pass.
    for layer in base.layers:
        assert layer.self_attn.processor._gen_block_mask is None

    # Regression: without the ctor contract the same forward stays maskless.
    seen.clear()
    maskless_base = _TinyProcessorCosmosBaseTransformer()
    maskless_transformer = _tiny_transformer(base_transformer=maskless_base)
    maskless_transformer(**kwargs)
    assert seen == [None] * len(maskless_base.layers)


def test_block_causal_wrapper_rejects_rollout_contracts():
    """Mutual exclusion: the supervised block-causal contract is
    full-sequence — KV-cache rollout passes must fail loudly on a
    block-causal wrapper (the KV-cache pipe never constructs one; a drifted
    caller must not silently mix rollout and supervised semantics)."""
    hidden_states = torch.randn(1, LATENT_CHANNELS, 3, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)

    transformer = _tiny_transformer(causal_block_frames=1)
    with pytest.raises(AssertionError, match="rollout-only contract"):
        transformer(**kwargs, kv_cache=transformer.new_rollout_kv_cache(truncate_rollout_gradients=False))


def _transformer_with_real_time_modules():
    base = _TinyCosmosBaseTransformer()
    # The real Cosmos3 sinusoid/embedder pair (transformer_cosmos3.py:373-374),
    # reduced only in the embedding width.
    base.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    base.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=PATCH_DIM)
    return _tiny_transformer(base_transformer=base)


def _embed_timestep(transformer, t: float) -> torch.Tensor:
    """Replay the base transformer's exact timestep path (transformer_cosmos3.py:625-626):
    vision_timesteps * timestep_scale -> time_proj sinusoid -> time_embedder MLP."""
    base = transformer.base_transformer
    vision_timesteps = transformer._vision_timesteps(torch.tensor([float(t)]), 0, 1, 2, torch.device("cpu"))
    return base.time_embedder(base.time_proj(vision_timesteps * COSMOS3_TIMESTEP_SCALE))


def test_embedded_timestep_discriminates_adjacent_t_under_bf16_autocast():
    """Discrimination test: embedded t=999 vs t=997 differ under ambient
    autocast(bf16). bf16 rounds 999 -> 1000 and 997 -> 996 (spacing 2 in
    [512, 1024)), so t=999 vs t=1000 additionally pins that no quantization
    happens anywhere on the path, and equality with the no-autocast fp32
    reference pins the whole sinusoid + MLP as an fp32 island."""
    torch.manual_seed(0)
    transformer = _transformer_with_real_time_modules()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        emb_999 = _embed_timestep(transformer, 999.0)
        emb_997 = _embed_timestep(transformer, 997.0)
        emb_1000 = _embed_timestep(transformer, 1000.0)
    emb_999_fp32_reference = _embed_timestep(transformer, 999.0)

    assert emb_999.dtype == torch.float32
    assert not torch.equal(emb_999, emb_997)
    assert not torch.equal(emb_999, emb_1000)
    assert not torch.equal(emb_997, emb_1000)
    torch.testing.assert_close(emb_999, emb_999_fp32_reference, rtol=0, atol=0)


def test_forward_passes_unquantized_fp32_vision_timesteps_under_bf16_autocast():
    """End-to-end through forward: t=999 must reach the base transformer
    as fp32 999.0 under ambient bf16 autocast (a bf16 cast would deliver 1000)."""
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["timestep"] = torch.tensor([999.0])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        transformer(**kwargs)

    vision_timesteps = transformer.base_transformer.calls[-1]["vision_timesteps"]
    assert vision_timesteps.dtype == torch.float32
    assert vision_timesteps.unique().tolist() == [999.0]


def test_fp32_time_embedder_keeps_state_dict_keys_and_stays_trainable():
    """The fp32 island adopts TimestepEmbedding's children, so checkpoint keys are
    byte-identical to the unwrapped base transformer and the freeze policy's
    time_embedder.requires_grad_(True) still reaches the same parameters."""
    transformer = _tiny_transformer()

    keys = transformer.state_dict().keys()
    assert "base_transformer.time_embedder.linear_1.weight" in keys
    assert "base_transformer.time_embedder.linear_2.weight" in keys
    assert not any(".time_embedder.time_embedder." in key for key in keys)
    assert all(p.requires_grad for p in transformer.base_transformer.time_embedder.parameters())


def test_fp32_time_embedder_island_handles_bf16_weights_and_flows_gradients():
    """Under FSDP2 mixed precision the embedder weights can be bf16 at forward
    time; the island must still compute fp32 (upcasting weights per call) and
    route gradients back to the storage-dtype parameters."""
    island = cosmos3_transformer_module._Fp32TimestepEmbedding(
        TimestepEmbedding(in_channels=4, time_embed_dim=8).to(torch.bfloat16)
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = island(torch.full((2, 4), 0.5))

    assert output.dtype == torch.float32
    output.square().sum().backward()
    assert island.linear_1.weight.grad is not None
    assert island.linear_1.weight.grad.dtype == torch.bfloat16
    assert torch.count_nonzero(island.linear_1.weight.grad) > 0


def test_cosmos3_gradient_checkpointing_initializes_base_callback():
    transformer = _tiny_transformer(gradient_checkpointing=True)

    assert transformer.base_transformer.gradient_checkpointing
    assert callable(transformer.base_transformer._gradient_checkpointing_func)


def test_cosmos3_checkpointing_preserves_conditioning_context_for_recompute():
    transformer = _tiny_transformer(gradient_checkpointing=True)
    conditioning_context = object()
    seen_contexts = []

    def module(tensor):
        seen_contexts.append(transformer._conditioning_context)
        return tensor.sin()

    x = torch.ones((), requires_grad=True)
    transformer._conditioning_context = conditioning_context
    y = transformer.base_transformer._gradient_checkpointing_func(module, x)
    transformer._conditioning_context = None

    y.backward()

    assert len(seen_contexts) >= 2
    assert all(context is conditioning_context for context in seen_contexts)


def test_cosmos3_references_use_the_base_transformer():
    transformer = _tiny_transformer()
    assert set(transformer._modules) == {"base_transformer", "opacity_embeddings", "scope"}


def test_cosmos3_transformer_uses_per_layer_target_conditioning_embeddings():
    transformer = _tiny_transformer()

    assert not hasattr(transformer, "camera_adapter")
    assert not hasattr(transformer, "opacity_adapter")
    assert len(transformer.opacity_embeddings) == len(transformer.base_transformer.layers)
    assert len(transformer.scope) == len(transformer.base_transformer.layers)
    for opacity_embedding in transformer.opacity_embeddings:
        # Full-pixel PixelUnshuffle footprint, not the
        # maxpooled latent_patch_size**2 = 4 dims/token input.
        assert opacity_embedding.in_features == OPACITY_TOKEN_DIM
        assert opacity_embedding.out_features == PATCH_DIM
        assert torch.count_nonzero(opacity_embedding.weight) == 0
        assert torch.count_nonzero(opacity_embedding.bias) == 0
    assert not hasattr(transformer, "camera_gates")


def test_cosmos3_camera_tokens_pixel_unshuffle_full_pixel_footprint():
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 2, 4, 4)
    camera_rays = torch.arange(1 * 2 * 8 * 8 * 6, dtype=torch.float32).reshape(1, 2, 8, 8, 6)

    camera_grid = transformer._camera_grid(camera_rays, hidden_states)
    expected_grid = rearrange(
        camera_rays,
        "b t (h rh) (w rw) c -> b (c rh rw) t h w",
        rh=CAMERA_PATCH_SIZE,
        rw=CAMERA_PATCH_SIZE,
    )

    assert camera_grid.shape == (1, 6 * CAMERA_PATCH_SIZE**2, 2, 2, 2)
    torch.testing.assert_close(camera_grid, expected_grid)
    _, camera_tokens = transformer._target_condition_tokens(
        torch.ones(1, 5, 8, 8),
        camera_rays,
        hidden_states,
        0,
        2,
    )
    torch.testing.assert_close(camera_tokens, expected_grid.flatten(2).transpose(1, 2))


def test_cosmos3_camera_tokens_require_exact_pixel_resolution():
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 2, 4, 4)
    wrong_resolution = torch.zeros(1, 2, 7, 8, 6)

    with pytest.raises(AssertionError, match="camera_rays must have shape"):
        transformer._camera_grid(wrong_resolution, hidden_states)


def test_cosmos3_opacity_tokens_pixel_unshuffle_full_pixel_footprint():
    """Opacity conditioning consumes the full-pixel f_o(PixelUnshuffle(O))
    footprint: frame 0 replicated to fill the VAE 1+4N window, then channels
    packed as (h8 w8 t4). The expected grid is built literally below, so this
    pins the packed layout that both families' wrappers get from the shared
    pack_opacity_tokens, independent of the helper's internals."""
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 2, 4, 4)
    camera_rays = torch.zeros(1, 2, 8, 8, 6)
    opacity = torch.arange(1 * 5 * 8 * 8, dtype=torch.float32).reshape(1, 5, 8, 8)

    opacity_grid = transformer._opacity_tokens(opacity, hidden_states)

    padded = torch.cat([opacity[:, :1].repeat_interleave(VAE_SCALE_FACTOR_TEMPORAL - 1, dim=1), opacity], dim=1)
    expected_grid = rearrange(
        padded,
        "b (t t4) (h h8) (w w8) -> b (h8 w8 t4) t h w",
        t4=VAE_SCALE_FACTOR_TEMPORAL,
        h8=CAMERA_PATCH_SIZE,
        w8=CAMERA_PATCH_SIZE,
    )
    assert opacity_grid.shape == (1, OPACITY_TOKEN_DIM, 2, 2, 2)
    torch.testing.assert_close(opacity_grid, expected_grid)

    opacity_tokens, _ = transformer._target_condition_tokens(opacity, camera_rays, hidden_states, 0, 2)
    torch.testing.assert_close(opacity_tokens, expected_grid.flatten(2).transpose(1, 2))
    # Every pixel opacity value reaches the adapter exactly once (lossless packing).
    assert opacity_tokens.shape == (1, 2 * 2 * 2, OPACITY_TOKEN_DIM)
    torch.testing.assert_close(opacity_tokens.flatten().sort().values, padded.flatten().sort().values)


def test_cosmos3_opacity_token_dim_is_4096_for_production_vae_factors():
    """At the production Cosmos3 VAE factors (sf_s=16, sf_t=4, patch 2) the
    opacity Linear in_features is 4096 — the value the shared
    opacity_embedding_input_dim yields for these dimensions
    ((16 * 2)**2 * 4) — and the adapter starts zero-initialized."""
    transformer = ArtifixerCosmos3Transformer(
        _TinyCosmosBaseTransformer(),
        vae_scale_factor_spatial=16,
        vae_scale_factor_temporal=4,
        gradient_checkpointing=False,
    )

    for opacity_embedding in transformer.opacity_embeddings:
        assert opacity_embedding.in_features == 4096
        assert torch.count_nonzero(opacity_embedding.weight) == 0
        assert torch.count_nonzero(opacity_embedding.bias) == 0


def test_cosmos3_opacity_tokens_context_parallel_slice_matches_full_grid():
    """CP ranks slice the packed opacity grid on latent frames; each latent frame's
    channels carry exactly its own pixel frames, so the slice must equal the
    corresponding rows of the single-rank grid."""
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 4, 4, 4)
    camera_rays = torch.zeros(1, 4, 8, 8, 6)
    opacity = torch.rand(1, 13, 8, 8)

    full_tokens, _ = transformer._target_condition_tokens(opacity, camera_rays, hidden_states, 0, 4)

    transformer._cp_mesh = _FakeCpMesh(rank=1, size=2)
    sliced_tokens, _ = transformer._target_condition_tokens(opacity, camera_rays, hidden_states, 2, 4)

    tokens_per_frame = 2 * 2
    torch.testing.assert_close(sliced_tokens, full_tokens[:, 2 * tokens_per_frame :])


def test_cosmos3_opacity_tokens_reject_wrong_spatial_resolution():
    """The full-pixel footprint requires exact pixel spatial dims; the
    silent trilinear spatial resize is gone with the maxpool path."""
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 2, 4, 4)
    opacity = torch.ones(1, 5, 4, 4)

    with pytest.raises(ValueError, match="full pixel resolution"):
        transformer._opacity_tokens(opacity, hidden_states)


@pytest.mark.parametrize(
    "opacity_frames",
    [
        2,  # latent temporal resolution (previously silently trilinear-resized)
        8,  # padded pixel count without the leading 1+4N frame
        6,  # arbitrary wrong length
    ],
)
def test_cosmos3_opacity_tokens_reject_unexpected_temporal_length(opacity_frames):
    """An unexpected opacity temporal length raises loudly (mirroring the
    Wan camera_rays resolution check in ArtifixerTransformer.forward) instead
    of silently trilinear-resizing to the latent grid."""
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 2, 4, 4)
    opacity = torch.ones(1, opacity_frames, 8, 8)

    with pytest.raises(ValueError, match="pixel temporal resolution"):
        transformer._opacity_tokens(opacity, hidden_states)


def test_cosmos3_forward_rejects_latent_resolution_opacity():
    """The silent trilinear fallback is gone end-to-end, not just in the helper."""
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 2, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["opacity"] = torch.ones(1, 2, 4, 4)

    with pytest.raises(ValueError, match="got 2 opacity frames for 2 latent frames"):
        transformer(**kwargs)


def test_cosmos3_mrope_vision_spacing_is_the_pinned_rope_fps():
    """MRoPE temporal spacing comes from the profile pin
    COSMOS3_ROPE_FPS (24) — there is no per-sample fps input; the vision
    step is base_fps/rope_fps = 1.0."""
    base = _TinyCosmosBaseTransformer()
    base.config.enable_fps_modulation = True
    transformer = _tiny_transformer(base_transformer=base)
    hidden_states = torch.zeros(2, LATENT_CHANNELS, 2, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)

    transformer(**kwargs)

    for call in base.calls[-2:]:
        vision_t = call["position_ids"][0, call["vision_sequence_indexes"]].reshape(2, -1)[:, 0]
        torch.testing.assert_close(
            vision_t[1] - vision_t[0],
            torch.tensor(1.0),
            atol=1e-3,
            rtol=0,
        )


def test_singleton_mrope_segment_still_scales_start_frame_offset_by_fps():
    ids, _ = cosmos3_transformer_module.get_3d_mrope_ids_vae_tokens(
        grid_t=1,
        grid_h=1,
        grid_w=1,
        temporal_offset=7.0,
        reset_spatial_indices=True,
        fps=60.0,
        base_fps=24.0,
        temporal_compression_factor=4,
        start_frame_offset=5,
    )

    torch.testing.assert_close(ids[0, 0], torch.tensor(9.0))


def test_cosmos3_target_conditioning_embeddings_update_attention_inputs_when_enabled():
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["camera_rays"] = torch.ones_like(kwargs["camera_rays"])
    kwargs["opacity"] = torch.ones_like(kwargs["opacity"])
    for opacity_embedding in transformer.opacity_embeddings:
        with torch.no_grad():
            opacity_embedding.weight.fill_(0.01)

    output = transformer(**kwargs)[0]

    assert not torch.allclose(output, hidden_states)


def test_cosmos3_zero_init_target_conditioning_is_noop_for_changed_controls():
    transformer = _tiny_transformer()
    hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)

    baseline = transformer(**kwargs)[0]
    changed_kwargs = {
        **kwargs,
        "camera_rays": torch.randn_like(kwargs["camera_rays"]),
        "opacity": torch.rand_like(kwargs["opacity"]),
    }
    changed = transformer(**changed_kwargs)[0]

    torch.testing.assert_close(changed, baseline)


def test_conditioning_adapter_weight_norms_report_each_layer():
    transformer = _tiny_transformer()

    logs = conditioning_adapter_weight_norms(transformer)

    layers = len(transformer.base_transformer.layers)
    assert set(logs) == {f"opacity_embeddings/weight_norm_{i:02d}" for i in range(layers)}
    assert logs["opacity_embeddings/weight_norm_00"] == 0  # zero-init until trained
    with torch.no_grad():
        transformer.opacity_embeddings[0].weight.fill_(3.0)
    assert conditioning_adapter_weight_norms(transformer)["opacity_embeddings/weight_norm_00"] > 0
    assert conditioning_adapter_weight_norms(nn.Linear(2, 2)) == {}


def test_cosmos3_target_conditioning_gradients_reach_every_layer_adapter():
    transformer = _tiny_transformer(gradient_checkpointing=True)
    hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4, requires_grad=True)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["camera_rays"] = torch.ones_like(kwargs["camera_rays"])
    kwargs["opacity"] = torch.ones_like(kwargs["opacity"])

    output = transformer(**kwargs)[0]
    output.square().sum().backward()

    # Zero starting weights must still be able to learn from the first step.
    for opacity_embedding in transformer.opacity_embeddings:
        assert torch.count_nonzero(opacity_embedding.weight.grad) > 0


def _plucker_rays(template: torch.Tensor, moment, direction) -> torch.Tensor:
    """Fill every pixel with the same ray, in (position x direction, direction) order."""
    rays = torch.empty_like(template)
    rays[..., :3] = torch.tensor(moment)
    rays[..., 3:] = torch.tensor(direction)
    return rays


def test_construction_has_no_camera_adapter_parameters():
    """The old camera adapters must be absent, not merely disabled, after replacement by SCoPE."""
    transformer = _tiny_transformer()

    assert len(transformer.scope) == len(transformer.base_transformer.layers)
    assert len(transformer.opacity_embeddings) == len(transformer.base_transformer.layers)
    for attribute in ("camera_embeddings", "camera_strength_temb_proj", "camera_strength_tables"):
        assert not hasattr(transformer, attribute), f"{attribute} survived the deletion"

    names = {name for name, _ in transformer.named_parameters()}
    assert not any("camera_embeddings" in name or "camera_strength" in name for name in names)

    norms = conditioning_adapter_weight_norms(transformer)
    assert norms and not any(key.startswith("camera_") for key in norms)

    trainable = {name for name, parameter in transformer.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("scope.") for name in trainable)


def test_camera_reaches_a_scope_model_only_through_scope():
    # Use the test model that runs the attention processor, where SCoPE is applied.
    transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer())
    hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    template = kwargs["camera_rays"]
    kwargs["camera_rays"] = _plucker_rays(template, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    turned = dict(kwargs, camera_rays=_plucker_rays(template, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    baseline_at_rest = transformer(**kwargs)[0]
    # Zero projection weights mean changing the rays has no effect yet.
    torch.testing.assert_close(transformer(**turned)[0], baseline_at_rest)

    _prime_scope_projections(transformer)
    with torch.no_grad():
        for injection in transformer.scope:
            injection.alpha.fill_(0.5)

    assert not torch.allclose(transformer(**turned)[0], transformer(**kwargs)[0])


def test_scope_alpha_receives_gradient_through_the_wrapper():
    transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer(), gradient_checkpointing=True)
    hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4, requires_grad=True)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["camera_rays"] = _plucker_rays(kwargs["camera_rays"], (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    kwargs["opacity"] = torch.ones_like(kwargs["opacity"])
    # Alpha can learn only after the projection weights produce a nonzero change.
    _prime_scope_projections(transformer)

    transformer(**kwargs)[0].square().sum().backward()

    for injection in transformer.scope:
        assert injection.alpha.grad is not None and injection.alpha.grad.abs().sum() > 0


def test_scope_strength_is_captured_as_delta_over_query():
    """Measure the camera's actual change to the query, rather than relying on weight sizes."""
    transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer())
    _prime_scope_projections(transformer)
    with torch.no_grad():
        for injection in transformer.scope:
            injection.alpha.fill_(0.5)

    samples: list[tuple[float, float]] = []
    transformer._scope_accumulator = samples
    hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["camera_rays"] = _plucker_rays(kwargs["camera_rays"], (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    transformer(**kwargs)
    transformer._scope_accumulator = None

    assert len(samples) == len(transformer.base_transformer.layers)
    assert all(delta > 0 and query > 0 for delta, query in samples)
    logs = scope_to_query_ratios(samples)
    assert logs["scope_to_query/mean"] > 0
    assert logs["scope_to_query/max"] >= logs["scope_to_query/mean"]
    assert scope_to_query_ratios([]) == {}


def test_scope_injects_on_the_kv_cache_branch():
    """Camera inputs must affect the output when caching is enabled."""
    transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer())
    injection = transformer.scope[0]
    _prime_scope_projections(transformer)
    with torch.no_grad():
        injection.alpha.fill_(0.5)
    attention = transformer.base_transformer.layers[0].self_attn
    processor = attention.processor

    und = torch.randn(4, PATCH_DIM)
    gen = torch.randn(4, PATCH_DIM)
    rotary = _identity_rotary(4, 4, PATCH_DIM // 4)

    def run(scope, rays):
        # Call the processor directly so the wrapper does not overwrite these settings.
        # This first chunk has no earlier video data and saves the text keys and values.
        processor._scope = scope
        processor._scope_rays = rays
        processor._kv_cache_context = (
            transformer.new_rollout_kv_cache(truncate_rollout_gradients=False),
            0,
            0,
            None,
            gen.shape[0],
        )
        try:
            return processor(attention, und.clone(), gen.clone(), rotary)
        finally:
            processor._scope = None
            processor._scope_rays = None
            processor._kv_cache_context = None

    rays = torch.randn(4, PLUCKER_DIM)
    torch.manual_seed(0)
    without_scope = run(None, None)
    torch.manual_seed(0)
    with_scope = run(injection, rays)

    assert not torch.allclose(with_scope[1], without_scope[1]), "SCoPE did not reach the cache branch"


def test_cosmos3_wrapper_never_drives_base_action_modules():
    """The pinned base model ships per-frame action-token modules; the wrapper
    must leave them completely undriven — no action kwargs ever reach the base
    transformer (the modules stay frozen and inert, not half-driven) — and the
    base action parameters must stay frozen."""
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 1, 4, 4)

    transformer(**_forward_kwargs(hidden_states, neighbor_hidden_states=None))

    call = transformer.base_transformer.calls[-1]
    assert not any(key.startswith("action_") for key in call)
    for name, parameter in transformer.base_transformer.named_parameters():
        if "action" in name:
            assert not parameter.requires_grad, f"base action parameter {name} must stay frozen"


def test_cosmos3_target_conditioning_context_does_not_leak_between_forwards():
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    with torch.no_grad():
        # Nonzero weights expose opacity changes; zero bias leaves zero input unchanged.
        for opacity_embedding in transformer.opacity_embeddings:
            opacity_embedding.weight.fill_(0.01)
            opacity_embedding.bias.zero_()

    first_kwargs = {**kwargs, "opacity": torch.ones_like(kwargs["opacity"])}
    second_kwargs = {**kwargs, "opacity": torch.zeros_like(kwargs["opacity"])}
    first = transformer(**first_kwargs)[0]
    second = transformer(**second_kwargs)[0]

    assert not torch.allclose(first, second)
    torch.testing.assert_close(second, hidden_states)


def test_cosmos3_transformer_requires_generator_attention_projections():
    class BadAttention(nn.Module):
        num_attention_heads = 4
        num_key_value_heads = 4

        def __init__(self):
            super().__init__()
            self.processor = SimpleNamespace(_attention_backend=None, _parallel_config=None)
            self.to_q = nn.Linear(PATCH_DIM, PATCH_DIM)

        def set_processor(self, processor):
            self.processor = processor

        def forward(self, und_seq, gen_seq, rotary_emb):
            return und_seq, gen_seq

    class BadLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = BadAttention()

    base = _TinyCosmosBaseTransformer(num_layers=0)
    base.layers = nn.ModuleList([BadLayer()])

    with pytest.raises(AttributeError, match="add_q_proj"):
        ArtifixerCosmos3Transformer(
            base,
            vae_scale_factor_spatial=16,
            vae_scale_factor_temporal=4,
            gradient_checkpointing=False,
        )


def test_cosmos3_ulysses_exchange_matches_single_rank_attention(monkeypatch):
    """The non-cache CP contract (Wan parity): the processor exchanges the
    post-RoPE gen Q/K/V with all_to_all_with_grad (scatter heads, gather
    rows), head-slices the replicated text K/V with the SAME contiguous
    chunk(cp, dim=1)[cp_rank] routing, and reverses the exchange on the
    attention output. Composed per rank against an exact rank-aware
    collective fake (cosmos3's unbatched [S, H, D] layout, real GQA head
    counts), each rank must reproduce its rows of the single-rank processor
    output and the full replicated text output. The fake routes contiguous
    rank-order chunks exactly like the real collective, so a strided or
    offset und head slice mismatches the exchanged gen heads and fails here.
    Gathered gen keys stay in GLOBAL order on this path (rank slices are
    contiguous in rank order) — rank-interleaved keys are the KV-cache
    path's layout, not this one."""
    torch.manual_seed(0)
    world_size = 2
    # 4 query heads over 2 KV heads: cosmos3's real GQA ratio.
    attention = _ProcessorAttention(kv_heads=2)
    reference_processor = ArtifixerCosmos3AttnProcessor()
    und_seq = torch.randn(3, attention.hidden_size)
    global_gen_seq = torch.randn(8, attention.hidden_size)
    tokens_per_rank = global_gen_seq.shape[0] // world_size

    ref_und, ref_gen = reference_processor(
        attention,
        und_seq,
        global_gen_seq,
        _identity_rotary(und_seq.shape[0], global_gen_seq.shape[0], attention.head_dim),
    )
    # to_add_out is identity in the double, so the reference gen output IS
    # the full-head attention output — what every CP peer feeds the reverse
    # exchange.
    reference_attention_out = ref_gen.view(-1, attention.num_attention_heads, attention.head_dim)
    peer_qkv = [
        cosmos3_projected_gen_qkv(attention, global_gen_seq[peer * tokens_per_rank : (peer + 1) * tokens_per_rank])
        for peer in range(world_size)
    ]

    def run_rank(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        mesh = _GroupedFakeCpMesh(rank=rank, size=world_size)
        calls = {"forward": 0, "reverse": 0}

        def fake_all_to_all(tensor, scatter_dim, gather_dim, group=None):
            assert group is mesh._group, "the Ulysses exchange must run on the mesh group"
            if scatter_dim == 1:
                # Forward exchanges, in the processor's q/k/v order: peer p's
                # pre-exchange tensor is its shard's projection.
                assert gather_dim == 0
                kind = calls["forward"] % 3
                calls["forward"] += 1
                peers = [peer_qkv[peer][kind] for peer in range(world_size)]
            else:
                # Reverse exchange on the attention output.
                assert (scatter_dim, gather_dim) == (0, 1)
                calls["reverse"] += 1
                peers = list(reference_attention_out.chunk(world_size, dim=1))
            return cosmos3_rankwise_all_to_all(
                tensor, peers, rank=rank, world_size=world_size, scatter_dim=scatter_dim, gather_dim=gather_dim
            )

        monkeypatch.setattr(cosmos3_transformer_module, "all_to_all_with_grad", fake_all_to_all)
        cp_processor = ArtifixerCosmos3AttnProcessor()
        cp_processor._cp_mesh = mesh
        local_gen_seq = global_gen_seq[rank * tokens_per_rank : (rank + 1) * tokens_per_rank]
        cp_und, cp_gen = cp_processor(
            attention,
            und_seq,
            local_gen_seq,
            _identity_rotary(und_seq.shape[0], tokens_per_rank, attention.head_dim),
        )
        assert calls == {"forward": 3, "reverse": 1}, "the Ulysses branch exchanges q/k/v once and reverses once"
        return cp_und, cp_gen

    for rank in range(world_size):
        cp_und, cp_gen = run_rank(rank)
        # The text stream is replicated and its causal attention keeps ALL
        # heads on every rank — only the gen streams are head-sharded.
        torch.testing.assert_close(cp_und, ref_und, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            cp_gen, ref_gen[rank * tokens_per_rank : (rank + 1) * tokens_per_rank], atol=1e-6, rtol=1e-6
        )


def test_cosmos3_ulysses_grad_enabled_cp_forward_carries_gen_kv_grads(monkeypatch):
    """The capability that replaced the deleted grad+CP rejection pin (the
    Wan MR made the identical swap): the Ulysses exchange is autograd-aware,
    so a grad-enabled non-cache CP forward is valid — the CP training path —
    and gradient flows from the gen output back through the exchanged Q/K/V
    into the gen projections (mirror of Wan's grad-enabled CP forward pin in
    tests/test_kv_cache_pipeline.py)."""
    torch.manual_seed(0)
    attention = _ProcessorAttention(kv_heads=2)
    cp_processor = ArtifixerCosmos3AttnProcessor()
    mesh = _GroupedFakeCpMesh(rank=0, size=2)
    cp_processor._cp_mesh = mesh

    def fake_all_to_all(tensor, scatter_dim, gather_dim, group=None):
        # Every-rank-holds-this-shard stand-in: pure chunk/cat keeps the
        # exchange on the autograd graph, exactly like the real collective.
        assert group is mesh._group
        return torch.cat(tensor.chunk(2, dim=scatter_dim), dim=gather_dim)

    monkeypatch.setattr(cosmos3_transformer_module, "all_to_all_with_grad", fake_all_to_all)
    und_seq = torch.randn(3, attention.hidden_size)
    gen_seq = torch.randn(4, attention.hidden_size)

    _, gen_out = cp_processor(
        attention, und_seq, gen_seq, _identity_rotary(und_seq.shape[0], gen_seq.shape[0], attention.head_dim)
    )
    gen_out.square().sum().backward()

    for projection in (attention.add_q_proj, attention.add_k_proj, attention.add_v_proj):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) > 0


def test_cosmos3_ulysses_cp_rejects_head_counts_not_divisible_by_cp_size(monkeypatch):
    """The Ulysses exchange hands each rank a contiguous 1/CP chunk of BOTH
    head axes, so the non-cache branch must reject a CP size that divides
    either head count unevenly — num_attention_heads AND (cosmos3 has real
    GQA) num_key_value_heads — before any collective runs. These asserts are
    deliberately scoped to the non-cache branch: the KV-cache path shards
    frames, not heads, and production CP=7 rollouts staying legal there is
    pinned in tests/test_cosmos3_kv_cache_rollout.py."""

    def collective_must_not_run(*_args, **_kwargs):
        pytest.fail("the head-divisibility asserts must fire before any collective")

    monkeypatch.setattr(cosmos3_transformer_module, "all_to_all_with_grad", collective_must_not_run)
    und_seq = torch.randn(3, 16)

    # 4 query heads never split across 3 ranks (6 rows DO split evenly:
    # only the head count is indivisible here).
    q_processor = ArtifixerCosmos3AttnProcessor()
    q_processor._cp_mesh = _GroupedFakeCpMesh(rank=0, size=3)
    attention = _ProcessorAttention()
    gen_seq = torch.randn(6, attention.hidden_size)
    with pytest.raises(AssertionError, match="divisible"):
        q_processor(attention, und_seq, gen_seq, _identity_rotary(3, 6, attention.head_dim))

    # GQA: 4 query heads divide CP=4 but the 2 KV heads do not — the
    # KV-head assert must fire on its own.
    kv_processor = ArtifixerCosmos3AttnProcessor()
    kv_processor._cp_mesh = _GroupedFakeCpMesh(rank=0, size=4)
    gqa_attention = _ProcessorAttention(kv_heads=2)
    gen_seq = torch.randn(4, gqa_attention.hidden_size)
    with pytest.raises(AssertionError, match="divisible"):
        kv_processor(gqa_attention, und_seq, gen_seq, _identity_rotary(3, 4, gqa_attention.head_dim))


def test_cosmos3_cp_armed_block_causal_mask_is_the_global_mask(monkeypatch):
    """Post-Ulysses the gen queries span the GLOBAL sequence, so the wrapper
    arms ONE global block-causal BlockMask under CP — geometry
    [S_gen_global, S_und + S_gen_global], bit-identical to the CP=1 mask —
    instead of the old per-rank query window. Recorded at the processor seam
    so the pin survives helper-signature refactors: if a refactor drops the
    rank-window plumbing entirely, block-causal CP coverage must not
    silently disappear with it."""
    seen: list[BlockMask | None] = []
    original_call = ArtifixerCosmos3AttnProcessor.__call__

    def recording_call(self, attn, und_seq, gen_seq, rotary_emb):
        seen.append(self._gen_block_mask)
        return original_call(self, attn, und_seq, gen_seq, rotary_emb)

    monkeypatch.setattr(ArtifixerCosmos3AttnProcessor, "__call__", recording_call)
    # Shape-preserving Ulysses stand-in: values are irrelevant to this pin,
    # but the production masked branch (and its global-geometry assert) must
    # actually run on the CP forward.
    monkeypatch.setattr(
        cosmos3_transformer_module,
        "all_to_all_with_grad",
        lambda tensor, scatter_dim, gather_dim, group=None: torch.cat(tensor.chunk(2, dim=scatter_dim), dim=gather_dim),
    )
    hidden_states = torch.randn(1, LATENT_CHANNELS, 4, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)

    single_rank = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer(), causal_block_frames=2)
    single_rank(**kwargs)
    single_rank_masks = list(seen)
    seen.clear()

    cp_transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer(), causal_block_frames=2)
    cp_transformer.enable_context_parallel(_GroupedFakeCpMesh(rank=1, size=2))
    monkeypatch.setattr(cp_transformer, "_gather_cp_output", lambda output: output)
    cp_transformer(**kwargs)

    und_len = 4  # the trimmed prompt in _forward_kwargs
    global_gen_tokens = 4 * 4  # 4 latent frames x 4 spatial tokens (4x4 latents, patch 2)
    expected = _dense_mask(_gen_block_mask(und_len, num_frames=4, block_frames=2, spatial_tokens_per_frame=4))
    assert len(single_rank_masks) == 2 and len(seen) == 2  # one armed mask per layer
    for block_mask in (*single_rank_masks, *seen):
        assert block_mask is not None
        assert block_mask.shape[-2:] == (global_gen_tokens, und_len + global_gen_tokens)
        assert torch.equal(_dense_mask(block_mask), expected)


def test_cosmos3_cp_slices_per_token_timesteps_to_the_rank_window(monkeypatch):
    """Diffusion-forcing CP training feeds per-token 2-D timesteps
    [B, num_noisy_vision_tokens] built over the GLOBAL clip; under CP the
    wrapper must hand each rank exactly its token window — the rows of the
    full-clip tensor at [cp_start * spatial, cp_end * spatial) — before
    _vision_timesteps' strict shape check (which sees the LOCAL noisy token
    count). At CP=1 the full tensor passes through bit-identically."""
    hidden_states = torch.randn(1, LATENT_CHANNELS, 4, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    spatial_tokens_per_frame = 4  # 4x4 latents, patch 2
    kwargs["timestep"] = torch.rand(1, 4 * spatial_tokens_per_frame)

    reference = _tiny_transformer()
    reference(**kwargs)
    reference_timesteps = reference.base_transformer.calls[-1]["vision_timesteps"]
    torch.testing.assert_close(reference_timesteps, kwargs["timestep"][0], rtol=0, atol=0)

    cp_transformer = _tiny_transformer()
    cp_transformer.enable_context_parallel(_FakeCpMesh(rank=1, size=2))
    monkeypatch.setattr(cp_transformer, "_gather_cp_output", lambda output: output)
    with torch.no_grad():
        cp_transformer(**kwargs)
    cp_timesteps = cp_transformer.base_transformer.calls[-1]["vision_timesteps"]

    assert cp_timesteps.shape == (2 * spatial_tokens_per_frame,)
    torch.testing.assert_close(cp_timesteps, kwargs["timestep"][0, 2 * spatial_tokens_per_frame :], rtol=0, atol=0)


def test_cosmos3_references_require_camera_rays():
    transformer = _tiny_transformer()
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 1, 2, 2)
    kwargs = _forward_kwargs(hidden_states, torch.zeros_like(hidden_states))
    with pytest.raises(AssertionError, match="neighbor_camera_rays"):
        transformer(**{**kwargs, "neighbor_camera_rays": None})


def test_cosmos3_ignored_references_are_not_processed(monkeypatch):
    transformer = _tiny_transformer()

    def fail(*args):
        raise AssertionError("dropped references must not enter the transformer")

    monkeypatch.setattr(transformer, "_prepend_references", fail)
    hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, torch.randn_like(hidden_states))
    output = transformer(**kwargs, ignore_neighbors=True)[0]
    torch.testing.assert_close(output, hidden_states)


def test_cosmos3_references_affect_targets_but_not_text_at_initialization():
    torch.manual_seed(0)
    transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer())
    hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4)
    kwargs = _forward_kwargs(hidden_states, torch.randn_like(hidden_states))
    with_references = transformer(**kwargs)[0]
    call = transformer.base_transformer.calls[-1]
    text_with_references = transformer.base_transformer.last_sequence[:, call["text_indexes"]].clone()
    without_references = transformer(**{**kwargs, "neighbor_hidden_states": None})[0]
    call = transformer.base_transformer.calls[-1]
    text_without_references = transformer.base_transformer.last_sequence[:, call["text_indexes"]]
    assert not torch.allclose(with_references, without_references)
    torch.testing.assert_close(text_with_references, text_without_references, rtol=0, atol=0)


def test_cosmos3_conditioning_context_clears_after_checkpointed_forward_and_backward_succeeds():
    transformer = ArtifixerCosmos3Transformer(
        _TinyCosmosBaseTransformer(),
        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
        vae_scale_factor_temporal=4,
        gradient_checkpointing=True,
    )
    hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4, requires_grad=True)
    neighbor_hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4)

    output = transformer(**_forward_kwargs(hidden_states, neighbor_hidden_states))[0]

    assert transformer._conditioning_context is None
    output.sum().backward()
    assert transformer._conditioning_context is None


def test_cosmos3_camera_ray_tokens_are_built_once_per_sample(monkeypatch):
    calls = 0
    original = cosmos3_transformer_module.token_plucker_rays

    def counted(tokens):
        nonlocal calls
        calls += 1
        return original(tokens)

    monkeypatch.setattr(cosmos3_transformer_module, "token_plucker_rays", counted)
    transformer = _tiny_transformer()
    hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4)
    neighbor_hidden_states = torch.randn(1, LATENT_CHANNELS, 1, 4, 4)
    transformer(**_forward_kwargs(hidden_states, neighbor_hidden_states))
    assert calls == 2


def test_cosmos3_reference_patchify_matches_base_proj_in_layout():
    """proj_in is shared with the base transformer's own patchify, whose
    per-token feature order is [patch_y, patch_x, channel]
    (transformer_cosmos3.py einsum cthpwq->thwpqc); reference tokens must use
    the same layout or the pretrained projection sees a scrambled basis."""
    transformer = _tiny_transformer()
    # Distinct height/width so an axis swap cannot cancel out.
    latents = torch.randn(1, LATENT_CHANNELS, 2, 4, 6)

    tokens, token_shape = transformer._patchify_latents(latents)

    patch_h, patch_w = 4 // PATCH_SIZE, 6 // PATCH_SIZE
    reference = latents.reshape(1, LATENT_CHANNELS, 2, patch_h, PATCH_SIZE, patch_w, PATCH_SIZE)
    reference = torch.einsum("bcthpwq->bthwpqc", reference).reshape(1, -1, PATCH_SIZE * PATCH_SIZE * LATENT_CHANNELS)
    torch.testing.assert_close(tokens, reference)
    assert token_shape == (2, patch_h, patch_w)


def test_cosmos3_context_parallel_slices_target_frames_and_gathers(monkeypatch):
    transformer = _tiny_transformer()
    transformer.enable_context_parallel(_FakeCpMesh(rank=1, size=2))
    hidden_states = torch.randn(1, LATENT_CHANNELS, 4, 4, 4)
    gathered_rank0 = torch.full_like(hidden_states[:, :, :2], -5.0)

    def fake_all_gather(outputs, local_output, group=None):
        outputs[0].copy_(gathered_rank0.to(device=local_output.device, dtype=local_output.dtype))
        outputs[1].copy_(local_output)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    with torch.no_grad():
        output = transformer(**_forward_kwargs(hidden_states, neighbor_hidden_states=None))[0]
    call = transformer.base_transformer.calls[-1]
    text_length = int(call["und_len"])
    vision_position_ids = call["position_ids"][:, call["vision_sequence_indexes"]]
    temporal_offset = text_length + transformer.config.unified_3d_mrope_temporal_modality_margin

    assert call["vision_tokens"][0].shape[2] == 2
    assert output.shape == hidden_states.shape
    assert torch.allclose(output[:, :, :2], gathered_rank0)
    assert torch.allclose(output[:, :, 2:], hidden_states[:, :, 2:])
    assert torch.equal(
        vision_position_ids[0, :4],
        torch.full_like(vision_position_ids[0, :4], temporal_offset + 2),
    )


def test_cosmos3_context_parallel_slices_rays_before_pixel_unshuffle(monkeypatch):
    transformer = _tiny_transformer()
    transformer.enable_context_parallel(_FakeCpMesh(rank=1, size=2))
    hidden_states = torch.zeros(1, LATENT_CHANNELS, 4, 4, 4)
    camera_rays = torch.zeros(1, 4, 8, 8, 6)
    original_camera_grid = transformer._camera_grid

    def checked_camera_grid(local_camera_rays, local_target):
        assert local_camera_rays.shape[1] == 2
        assert local_target.shape[2] == 2
        return original_camera_grid(local_camera_rays, local_target)

    monkeypatch.setattr(transformer, "_camera_grid", checked_camera_grid)

    _, camera_tokens = transformer._target_condition_tokens(
        torch.ones(1, 13, 8, 8),
        camera_rays,
        hidden_states,
        2,
        4,
    )

    assert camera_tokens.shape[1] == 2 * 2 * 2


def test_cosmos3_context_parallel_pads_uneven_frame_shards(monkeypatch):
    """The old even-shard requirement is gone from the full-input path: a
    frame count that does not divide the CP size pads up to the next multiple
    (the 21-frame production geometry was un-CP-able otherwise — its frame
    divisors {3, 7, 21} share no divisor with the 32/8 attention heads). The
    strict slicing helper itself still raises, so any direct caller that
    forgets to pad fails loudly instead of dropping frames."""
    from model_training.net.context_parallel import cp_frame_slice, cp_padded_frame_count

    with pytest.raises(ValueError, match="Cannot evenly split"):
        cp_frame_slice(3, rank=0, world_size=2)
    assert cp_padded_frame_count(3, 2) == 4
    assert cp_padded_frame_count(21, 8) == 24
    assert cp_padded_frame_count(4, 2) == 4, "already-even counts must not pad"

    transformer = _tiny_transformer()
    transformer.enable_context_parallel(_FakeCpMesh(rank=0, size=2))
    monkeypatch.setattr(transformer, "_gather_cp_output", lambda output: output)
    hidden_states = torch.randn(1, LATENT_CHANNELS, 3, 4, 4)

    with torch.no_grad():
        output = transformer(**_forward_kwargs(hidden_states, neighbor_hidden_states=None))[0]
    # Rank 0's local shard covers frames [0, 2) of the 4-frame padded clip
    # (the strip after the real gather is pinned by the padded CP tests).
    assert output.shape == (1, LATENT_CHANNELS, 2, 4, 4)


def test_cosmos3_context_parallel_keeps_reference_rays_whole(monkeypatch):
    transformer = _tiny_transformer()
    transformer.enable_context_parallel(_FakeCpMesh(rank=1, size=2))
    hidden_states = torch.randn(1, LATENT_CHANNELS, 4, 4, 4)
    neighbors = torch.randn(1, LATENT_CHANNELS, 3, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbors)
    seen = []
    original = transformer._camera_grid

    def capture(rays, latents):
        seen.append((rays.shape[1], latents.shape[2]))
        return original(rays, latents)

    monkeypatch.setattr(transformer, "_camera_grid", capture)
    monkeypatch.setattr(transformer, "_gather_cp_output", lambda output: output)
    with torch.no_grad():
        transformer(**kwargs)
    assert seen == [(2, 2), (3, 3)]


@pytest.mark.parametrize("mode", ["bidirectional", "diffusion_forcing"])
def test_cosmos3_context_parallel_matches_single_rank_transformer(monkeypatch, mode):
    """End-to-end CP equality: the COMPOSED 2-rank CP forward (frame-sliced
    hidden states, sliced ray/opacity conditioning, per-layer Ulysses
    all-to-all exchange, one GLOBAL block mask) must equal the single-rank
    forward — for the bidirectional path AND the supervised block-causal
    masked path (the AR rollout's CP contract — rank-local staging with
    grad-gated gathered reads — is pinned in
    tests/test_cosmos3_kv_cache_rollout.py). The piecewise CP tests above
    pin each mechanism alone; only this test pins their composition, so an
    off-by-one in _cp_frame_slice, a mis-sliced mask, or a mis-routed
    exchange chunk fails here instead of silently corrupting multi-GPU
    outputs. Comparison is at tolerance, NOT bitwise: each CP rank batches
    the attention kernel over a 1/CP head chunk, and that
    head-count-dependent reduction order can drift in the last float bits.
    (Gathered gen keys keep GLOBAL order on this path — rank slices are
    contiguous in rank order, no interleaving.)"""
    torch.manual_seed(0)
    causal_block_frames = 2 if mode == "diffusion_forcing" else None
    reference = _tiny_transformer(
        base_transformer=_TinyProcessorCosmosBaseTransformer(num_layers=2),
        causal_block_frames=causal_block_frames,
    )
    # Enable opacity inputs so an incorrect frame slice changes the output.
    with torch.no_grad():
        for module in reference.opacity_embeddings:
            module.weight.normal_(std=0.02)
            module.bias.normal_(std=0.02)
    rank0 = _tiny_transformer(
        base_transformer=_TinyProcessorCosmosBaseTransformer(num_layers=2),
        causal_block_frames=causal_block_frames,
    )
    rank1 = _tiny_transformer(
        base_transformer=_TinyProcessorCosmosBaseTransformer(num_layers=2),
        causal_block_frames=causal_block_frames,
    )
    rank0.load_state_dict(reference.state_dict())
    rank1.load_state_dict(reference.state_dict())
    rank0.enable_context_parallel(_GroupedFakeCpMesh(rank=0, size=2))
    rank1.enable_context_parallel(_GroupedFakeCpMesh(rank=1, size=2))
    monkeypatch.setattr(rank0, "_gather_cp_output", lambda output: output)
    monkeypatch.setattr(rank1, "_gather_cp_output", lambda output: output)

    hidden_states = torch.randn(1, LATENT_CHANNELS, 4, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    # Randomize the conditioning controls (the shared defaults are constant
    # across frames): frame mis-slices must change values, not just shapes.
    kwargs["camera_rays"] = torch.randn_like(kwargs["camera_rays"])
    kwargs["opacity"] = torch.rand_like(kwargs["opacity"])

    # Record video features after opacity is added, then use them to rebuild
    # each worker's query, key, and value arrays below.
    conditioned_gen_inputs: list[list[torch.Tensor]] = [[] for _ in reference.base_transformer.layers]

    def make_recorder(layer_index: int):
        def recorder(_module, inputs, hook_kwargs):
            gen_seq = inputs[1] if len(inputs) >= 2 else hook_kwargs["gen_seq"]
            conditioned_gen_inputs[layer_index].append(gen_seq.detach().clone())

        return recorder

    for layer_index, layer in enumerate(reference.base_transformer.layers):
        layer.self_attn.register_forward_pre_hook(make_recorder(layer_index), with_kwargs=True)

    expected = reference(**kwargs)[0]
    vision_tokens_per_rank = (
        hidden_states.shape[2] // 2 * (hidden_states.shape[3] // PATCH_SIZE) * (hidden_states.shape[4] // PATCH_SIZE)
    )
    heads = reference.base_transformer.layers[0].self_attn.num_attention_heads
    head_dim = reference.base_transformer.layers[0].self_attn.head_dim
    # to_add_out is identity in the tiny double, so each reference layer's
    # recorded gen output IS its full-head attention output — what every CP
    # peer feeds the reverse exchange.
    reference_attention_outs = [
        layer.attn_outputs[-1].view(-1, heads, head_dim) for layer in reference.base_transformer.layers
    ]

    def all_to_all_from_reference(rank: int):
        calls = {"forward": 0, "reverse": 0}

        def fake_all_to_all(tensor, scatter_dim, gather_dim, group=None):
            # One reverse exchange closes each layer's attention, so the
            # reverse count IS the current layer index.
            layer_index = calls["reverse"]
            attention = reference.base_transformer.layers[layer_index].self_attn
            reference_gen_seq = conditioned_gen_inputs[layer_index][-1]
            if scatter_dim == 1:
                # Forward exchanges in the processor's q/k/v order: peer p's
                # pre-exchange tensor is the projection of the reference's
                # conditioned rows for shard p (identity norms and identity
                # rotary in the double). The caller's own slot stays the
                # actual tensor, so this rank's compute is what's verified.
                assert gather_dim == 0
                kind = calls["forward"] % 3
                calls["forward"] += 1
                peers = []
                for peer in range(2):
                    rows = reference_gen_seq[peer * vision_tokens_per_rank : (peer + 1) * vision_tokens_per_rank]
                    peers.append(cosmos3_projected_gen_qkv(attention, rows)[kind])
            else:
                assert (scatter_dim, gather_dim) == (0, 1)
                calls["reverse"] += 1
                peers = list(reference_attention_outs[layer_index].chunk(2, dim=1))
            return cosmos3_rankwise_all_to_all(
                tensor, peers, rank=rank, world_size=2, scatter_dim=scatter_dim, gather_dim=gather_dim
            )

        return fake_all_to_all

    # Grad-enabled on purpose: the Ulysses exchange is autograd-aware, so CP
    # forwards no longer hide behind no_grad — this is the CP training path.
    monkeypatch.setattr(cosmos3_transformer_module, "all_to_all_with_grad", all_to_all_from_reference(rank=0))
    rank0_output = rank0(**kwargs)[0]
    monkeypatch.setattr(cosmos3_transformer_module, "all_to_all_with_grad", all_to_all_from_reference(rank=1))
    rank1_output = rank1(**kwargs)[0]

    cp_output = torch.cat([rank0_output, rank1_output], dim=2)
    torch.testing.assert_close(cp_output, expected, atol=1e-5, rtol=1e-5)


def test_cosmos3_local_shard_cp_forward_matches_single_rank(monkeypatch):
    """The trainer's CP contract, composed: each rank receives PRE-SLICED
    inputs with frame_offset naming the shard's absolute start
    (local_cp_input=True) and returns its LOCAL frames. mRoPE positions,
    the opacity pad (rank 0 only), and the camera windows must land on the
    same absolute frames as the full-input convention, so the concatenated
    local outputs must equal the single-rank forward. (The no-gather half
    of the contract is invisible to this single-process fake — gathers
    no-op without an initialized process group — and is pinned by the
    torchrun test's local-shape assert instead.) Block-causal mode is
    fenced off this contract (test_local_cp_input_contract_fences), hence
    no diffusion_forcing parametrization here."""
    torch.manual_seed(0)
    reference = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer(num_layers=2))
    # Enable opacity inputs so an incorrect frame slice changes the output.
    with torch.no_grad():
        for module in reference.opacity_embeddings:
            module.weight.normal_(std=0.02)
            module.bias.normal_(std=0.02)

    hidden_states = torch.randn(1, LATENT_CHANNELS, 4, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["camera_rays"] = torch.randn_like(kwargs["camera_rays"])
    kwargs["opacity"] = torch.rand_like(kwargs["opacity"])

    conditioned_gen_inputs: list[list[torch.Tensor]] = [[] for _ in reference.base_transformer.layers]

    def make_recorder(layer_index: int):
        def recorder(_module, inputs, hook_kwargs):
            gen_seq = inputs[1] if len(inputs) >= 2 else hook_kwargs["gen_seq"]
            conditioned_gen_inputs[layer_index].append(gen_seq.detach().clone())

        return recorder

    for layer_index, layer in enumerate(reference.base_transformer.layers):
        layer.self_attn.register_forward_pre_hook(make_recorder(layer_index), with_kwargs=True)

    expected = reference(**kwargs)[0]

    local_frames = hidden_states.shape[2] // 2
    vision_tokens_per_rank = (
        local_frames * (hidden_states.shape[3] // PATCH_SIZE) * (hidden_states.shape[4] // PATCH_SIZE)
    )
    heads = reference.base_transformer.layers[0].self_attn.num_attention_heads
    head_dim = reference.base_transformer.layers[0].self_attn.head_dim
    reference_attention_outs = [
        layer.attn_outputs[-1].view(-1, heads, head_dim) for layer in reference.base_transformer.layers
    ]

    def all_to_all_from_reference(rank: int):
        calls = {"forward": 0, "reverse": 0}

        def fake_all_to_all(tensor, scatter_dim, gather_dim, group=None):
            layer_index = calls["reverse"]
            attention = reference.base_transformer.layers[layer_index].self_attn
            reference_gen_seq = conditioned_gen_inputs[layer_index][-1]
            if scatter_dim == 1:
                assert gather_dim == 0
                kind = calls["forward"] % 3
                calls["forward"] += 1
                peers = []
                for peer in range(2):
                    rows = reference_gen_seq[peer * vision_tokens_per_rank : (peer + 1) * vision_tokens_per_rank]
                    peers.append(cosmos3_projected_gen_qkv(attention, rows)[kind])
            else:
                assert (scatter_dim, gather_dim) == (0, 1)
                calls["reverse"] += 1
                peers = list(reference_attention_outs[layer_index].chunk(2, dim=1))
            return cosmos3_rankwise_all_to_all(
                tensor, peers, rank=rank, world_size=2, scatter_dim=scatter_dim, gather_dim=gather_dim
            )

        return fake_all_to_all

    outputs = []
    for rank in range(2):
        transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer(num_layers=2))
        transformer.load_state_dict(reference.state_dict())
        transformer.enable_context_parallel(_GroupedFakeCpMesh(rank=rank, size=2))
        start, end = rank * local_frames, (rank + 1) * local_frames
        # The causal VAE folds pixel frames [4k-3, 4k] into latent frame
        # k >= 1; latent frame 0 is pixel frame 0 alone.
        pixel_start = 0 if start == 0 else start * VAE_SCALE_FACTOR_TEMPORAL - (VAE_SCALE_FACTOR_TEMPORAL - 1)
        pixel_end = end * VAE_SCALE_FACTOR_TEMPORAL - (VAE_SCALE_FACTOR_TEMPORAL - 1)
        local_kwargs = dict(
            kwargs,
            hidden_states=kwargs["hidden_states"][:, :, start:end],
            opacity=kwargs["opacity"][:, pixel_start:pixel_end],
            camera_rays=kwargs["camera_rays"][:, start:end],
            w2cs=kwargs["w2cs"][:, start:end],
            Ks=kwargs["Ks"][:, start:end],
            frame_offset=start,
            local_cp_input=True,
        )
        monkeypatch.setattr(cosmos3_transformer_module, "all_to_all_with_grad", all_to_all_from_reference(rank))
        outputs.append(transformer(**local_kwargs)[0])

    torch.testing.assert_close(torch.cat(outputs, dim=2), expected, atol=1e-5, rtol=1e-5)


def test_local_cp_input_contract_fences():
    """local_cp_input covers plain supervised forwards only, and under an
    active mesh the caller's frame_offset must equal the mesh-derived shard
    start — each violation fails loudly instead of silently mis-positioning
    frames."""
    hidden_states = torch.randn(1, LATENT_CHANNELS, 2, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)

    transformer = _tiny_transformer()
    transformer.enable_context_parallel(_GroupedFakeCpMesh(rank=1, size=2))
    with pytest.raises(AssertionError, match=r"frame_offset 0 does not match CP rank 1"):
        transformer(**kwargs, frame_offset=0, local_cp_input=True)

    with pytest.raises(AssertionError, match=r"does not support KV-cache"):
        _tiny_transformer()(**kwargs, local_cp_input=True, kv_cache=object())

    with pytest.raises(AssertionError, match=r"does not support block-causal"):
        _tiny_transformer(causal_block_frames=1)(**kwargs, local_cp_input=True)


# --- CP frame padding: full-input clips that do not divide the CP size pad
# --- up to the next multiple, so the 21-frame production geometry gets
# --- head-divisor CP sizes instead of being stuck at CP=1 ---

# 3 latent frames on a 2-rank mesh: the smallest geometry that forces a pad
# frame (3 -> 4), with rank 1 holding one real and one pad frame.
PAD_REAL_FRAMES = 3
PAD_WORLD_SIZE = 2
PAD_TOTAL_FRAMES = 4
# 4x4 latents, patch 2 -> 2x2 patch tokens per latent frame.
PAD_SPATIAL_TOKENS = 4
PAD_TOKENS = (PAD_TOTAL_FRAMES - PAD_REAL_FRAMES) * PAD_SPATIAL_TOKENS
PAD_FRAMES_PER_RANK = PAD_TOTAL_FRAMES // PAD_WORLD_SIZE
PAD_ROWS_PER_RANK = PAD_FRAMES_PER_RANK * PAD_SPATIAL_TOKENS


def test_prefix_block_causal_mask_pad_query_rows_see_only_text():
    """CP pad tokens extend the block-causal mask's QUERY axis only: real
    query rows keep exactly the unpadded mask (bit-equal), pad query rows
    attend to just the text prefix — never a real gen key — and the key axis
    stays real because the processor drops pad keys after the exchange."""
    padded = create_prefix_block_causal_block_mask(
        prefix_len=3,
        total_tokens=6,
        frames_per_block=1,
        frame_seqlen=2,
        device="cpu",
        block_size=128,
        pad_query_tokens=2,
    )
    assert padded.shape[-2:] == (8, 9), "pad tokens extend the query axis; the key axis stays real"

    dense = _dense_mask(padded)
    unpadded = _dense_mask(_gen_block_mask(3, num_frames=3, block_frames=1, spatial_tokens_per_frame=2))
    assert torch.equal(dense[:6], unpadded), "real query rows must keep the unpadded mask bit-identical"
    assert bool(dense[6:, :3].all()), "pad queries keep the text columns (finite rows, never NaN)"
    assert not bool(dense[6:, 3:].any()), "pad queries must not see any real gen key"


def test_cosmos3_padded_cp_mrope_positions_are_absolute_across_pad_shards(monkeypatch):
    """mRoPE positions are computed from each shard's ABSOLUTE start inside
    the padded clip, so a real token's position is identical to its CP=1
    position no matter which rank it lands on: rank 1's first frame is
    global frame 2 and must carry the full-sequence ids of frame 2. Pad
    frames get positions past the real clip, which is irrelevant — their
    keys are dropped and their outputs stripped."""
    hidden_states = torch.randn(1, LATENT_CHANNELS, PAD_REAL_FRAMES, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)

    reference = _tiny_transformer()
    reference(**kwargs)
    reference_call = reference.base_transformer.calls[-1]
    reference_vision_ids = reference_call["position_ids"][:, reference_call["vision_sequence_indexes"]]

    cp_transformer = _tiny_transformer()
    cp_transformer.enable_context_parallel(_FakeCpMesh(rank=1, size=PAD_WORLD_SIZE))
    monkeypatch.setattr(cp_transformer, "_gather_cp_output", lambda output: output)
    with torch.no_grad():
        output = cp_transformer(**kwargs)[0]
    call = cp_transformer.base_transformer.calls[-1]
    shard_vision_ids = call["position_ids"][:, call["vision_sequence_indexes"]]

    # Rank 1 of the padded clip holds frames [2, 4): one real, one pad.
    assert shard_vision_ids.shape[1] == PAD_ROWS_PER_RANK
    torch.testing.assert_close(
        shard_vision_ids[:, :PAD_SPATIAL_TOKENS],
        reference_vision_ids[:, 2 * PAD_SPATIAL_TOKENS : 3 * PAD_SPATIAL_TOKENS],
        rtol=0,
        atol=0,
    )
    # The un-gathered local output keeps the shard's (padded) frame count;
    # the strip below the gather is exercised in the equivalence test.
    assert output.shape[2] == PAD_FRAMES_PER_RANK


def test_cosmos3_padded_cp_forward_strips_pads_and_arms_pad_tokens(monkeypatch):
    """Wrapper contract for padded CP forwards: the conditioning pre-hook
    arms the processor with the pad-token count during every layer call, the
    disarm hook clears it, and the gathered output is stripped back to the
    caller's real frame count."""
    seen_pad_tokens: list[int] = []
    original_call = ArtifixerCosmos3AttnProcessor.__call__

    def recording_call(self, attn, und_seq, gen_seq, rotary_emb):
        seen_pad_tokens.append(self._cp_pad_tokens)
        return original_call(self, attn, und_seq, gen_seq, rotary_emb)

    monkeypatch.setattr(ArtifixerCosmos3AttnProcessor, "__call__", recording_call)
    monkeypatch.setattr(
        cosmos3_transformer_module,
        "all_to_all_with_grad",
        lambda tensor, scatter_dim, gather_dim, group=None: torch.cat(
            tensor.chunk(PAD_WORLD_SIZE, dim=scatter_dim), dim=gather_dim
        ),
    )
    transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer())
    transformer.enable_context_parallel(_GroupedFakeCpMesh(rank=0, size=PAD_WORLD_SIZE))

    gathered_frames: list[int] = []

    def fake_gather(output):
        gathered_frames.append(output.shape[2])
        return torch.cat([output] * PAD_WORLD_SIZE, dim=2)

    monkeypatch.setattr(transformer, "_gather_cp_output", fake_gather)
    hidden_states = torch.randn(1, LATENT_CHANNELS, PAD_REAL_FRAMES, 4, 4)
    with torch.no_grad():
        output = transformer(**_forward_kwargs(hidden_states, neighbor_hidden_states=None))[0]

    assert output.shape == hidden_states.shape, "pad frames must be stripped after the gather"
    assert gathered_frames == [PAD_FRAMES_PER_RANK], "each rank gathers its padded frame shard"
    assert seen_pad_tokens == [PAD_TOKENS] * len(transformer.base_transformer.layers)
    for layer in transformer.base_transformer.layers:
        assert layer.self_attn.processor._cp_pad_tokens == 0, "the disarm hook must clear the pad count"


@pytest.mark.parametrize("mode", ["bidirectional", "diffusion_forcing"])
def test_cosmos3_padded_full_input_cp_forward_matches_single_rank(monkeypatch, mode):
    """THE padded-CP equivalence pin: a 3-frame clip on a 2-rank mesh pads to
    4 frames, and — composed per rank against exact rank-aware collective
    fakes — the gathered-and-stripped output must equal the CP=1 forward on
    the unpadded clip, in both the maskless and the block-causal
    (diffusion-forcing) modes. Every PAD slot in the fakes is fed ZEROS on
    purpose, while the ranks' own pad rows carry the wrapper's real pad
    values: the contract says pad rows cannot influence any real row (pad
    keys are dropped post-exchange, pad outputs stripped post-gather), so if
    either leaks, this equality fails."""
    torch.manual_seed(0)
    causal_block_frames = 1 if mode == "diffusion_forcing" else None
    reference = _tiny_transformer(
        base_transformer=_TinyProcessorCosmosBaseTransformer(num_layers=2),
        causal_block_frames=causal_block_frames,
    )
    # Enable opacity inputs so an incorrect frame slice changes the output.
    with torch.no_grad():
        for module in reference.opacity_embeddings:
            module.weight.normal_(std=0.02)
            module.bias.normal_(std=0.02)

    hidden_states = torch.randn(1, LATENT_CHANNELS, PAD_REAL_FRAMES, 4, 4)
    kwargs = _forward_kwargs(hidden_states, neighbor_hidden_states=None)
    kwargs["camera_rays"] = torch.randn_like(kwargs["camera_rays"])
    kwargs["opacity"] = torch.rand_like(kwargs["opacity"])

    conditioned_gen_inputs: list[list[torch.Tensor]] = [[] for _ in reference.base_transformer.layers]

    def make_recorder(layer_index: int):
        def recorder(_module, inputs, hook_kwargs):
            gen_seq = inputs[1] if len(inputs) >= 2 else hook_kwargs["gen_seq"]
            conditioned_gen_inputs[layer_index].append(gen_seq.detach().clone())

        return recorder

    for layer_index, layer in enumerate(reference.base_transformer.layers):
        layer.self_attn.register_forward_pre_hook(make_recorder(layer_index), with_kwargs=True)

    expected = reference(**kwargs)[0]

    heads = reference.base_transformer.layers[0].self_attn.num_attention_heads
    head_dim = reference.base_transformer.layers[0].self_attn.head_dim
    hidden_size = reference.base_transformer.layers[0].self_attn.hidden_size
    # Reference tensors extended with ZERO pad rows: what the fakes hand out
    # for every peer slot (real rows are the reference's true values; pad
    # values must be irrelevant by contract).
    padded_gen_inputs = [
        torch.cat([rows[-1], rows[-1].new_zeros(PAD_TOKENS, hidden_size)]) for rows in conditioned_gen_inputs
    ]
    reference_attention_outs = [
        torch.cat(
            [
                layer.attn_outputs[-1].view(-1, heads, head_dim),
                layer.attn_outputs[-1].new_zeros(PAD_TOKENS, heads, head_dim),
            ]
        )
        for layer in reference.base_transformer.layers
    ]
    # Expected padded output for the gather fakes' peer slots: real frames
    # from the reference, the pad frame zeros.
    expected_padded = torch.cat([expected, expected.new_zeros(1, LATENT_CHANNELS, 1, 4, 4)], dim=2)

    def all_to_all_from_reference(rank: int):
        calls = {"forward": 0, "reverse": 0}

        def fake_all_to_all(tensor, scatter_dim, gather_dim, group=None):
            layer_index = calls["reverse"]
            attention = reference.base_transformer.layers[layer_index].self_attn
            if scatter_dim == 1:
                assert gather_dim == 0
                kind = calls["forward"] % 3
                calls["forward"] += 1
                peers = []
                for peer in range(PAD_WORLD_SIZE):
                    rows = padded_gen_inputs[layer_index][peer * PAD_ROWS_PER_RANK : (peer + 1) * PAD_ROWS_PER_RANK]
                    peers.append(cosmos3_projected_gen_qkv(attention, rows)[kind])
            else:
                assert (scatter_dim, gather_dim) == (0, 1)
                calls["reverse"] += 1
                peers = list(reference_attention_outs[layer_index].chunk(PAD_WORLD_SIZE, dim=1))
            return cosmos3_rankwise_all_to_all(
                tensor, peers, rank=rank, world_size=PAD_WORLD_SIZE, scatter_dim=scatter_dim, gather_dim=gather_dim
            )

        return fake_all_to_all

    def gather_from_reference(rank: int):
        def fake_gather(output):
            assert output.shape[2] == PAD_FRAMES_PER_RANK
            shards = [
                (
                    output
                    if peer == rank
                    else expected_padded[:, :, peer * PAD_FRAMES_PER_RANK : (peer + 1) * PAD_FRAMES_PER_RANK]
                )
                for peer in range(PAD_WORLD_SIZE)
            ]
            return torch.cat(shards, dim=2)

        return fake_gather

    for rank in range(PAD_WORLD_SIZE):
        transformer = _tiny_transformer(
            base_transformer=_TinyProcessorCosmosBaseTransformer(num_layers=2),
            causal_block_frames=causal_block_frames,
        )
        transformer.load_state_dict(reference.state_dict())
        transformer.enable_context_parallel(_GroupedFakeCpMesh(rank=rank, size=PAD_WORLD_SIZE))
        monkeypatch.setattr(cosmos3_transformer_module, "all_to_all_with_grad", all_to_all_from_reference(rank))
        monkeypatch.setattr(transformer, "_gather_cp_output", gather_from_reference(rank))
        with torch.no_grad():
            output = transformer(**kwargs)[0]
        assert output.shape == expected.shape, "the wrapper must strip the pad frame after the gather"
        torch.testing.assert_close(
            output,
            expected,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda base, r=rank: f"rank {r}: padded CP forward diverged from CP=1\n{base}",
        )


# --- clean_x teacher forcing: rollout equivalence and the weave contract ---

TF_FRAMES = 4
TF_BLOCK_FRAMES = 2
TF_NUM_BLOCKS = TF_FRAMES // TF_BLOCK_FRAMES
# 4x4 latents, patch 2 -> 2x2 patch tokens per latent frame.
TF_SPATIAL_TOKENS = (4 // PATCH_SIZE) ** 2


def _teacher_forcing_timesteps(block_values: list[float]) -> torch.Tensor:
    """Per-token 2-D timesteps [1, num_noisy_vision_tokens] at per-block values."""
    per_block = torch.tensor(block_values, dtype=torch.float32)
    return per_block.repeat_interleave(TF_BLOCK_FRAMES * TF_SPATIAL_TOKENS).unsqueeze(0)


def _chunk_kwargs(kwargs: dict, start: int, end: int) -> dict:
    """Chunk-only conditioning kwargs at absolute frames [start, end): the
    KV-cache pass contract (real pixel windows, no first-frame pad off frame 0),
    exactly as tests/test_cosmos3_kv_cache_rollout.py slices them."""
    sliced = dict(kwargs)
    del sliced["hidden_states"]  # each pass supplies its own
    del sliced["timestep"]
    pixel_start = 0 if start == 0 else (start - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1
    sliced["opacity"] = kwargs["opacity"][:, pixel_start : (end - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1]
    sliced["camera_rays"] = kwargs["camera_rays"][:, start:end]
    sliced["w2cs"] = kwargs["w2cs"][:, start:end]
    sliced["Ks"] = kwargs["Ks"][:, start:end]
    return sliced


def test_teacher_forcing_forward_matches_kv_cache_rollout():
    """THE teacher-forcing equivalence pin: one clean_x forward over T frames
    in causal_block_frames-frame blocks computes, for every block j, exactly
    what the KV-cache rollout computes for noisy block j after seeding clean
    blocks < j — so the TF output's block-j slice must match the cache
    denoise output (shared weights, CPU fp32; tolerance only for the
    masked-full vs subset-keys reduction order)."""
    torch.manual_seed(0)
    tf_transformer = _tiny_transformer(
        base_transformer=_TinyProcessorCosmosBaseTransformer(), causal_block_frames=TF_BLOCK_FRAMES
    )
    rollout_transformer = _tiny_transformer(base_transformer=_TinyProcessorCosmosBaseTransformer())
    rollout_transformer.load_state_dict(tf_transformer.state_dict())

    noisy = torch.randn(1, LATENT_CHANNELS, TF_FRAMES, 4, 4)
    clean = torch.randn(1, LATENT_CHANNELS, TF_FRAMES, 4, 4)
    kwargs = _forward_kwargs(noisy, neighbor_hidden_states=None)
    block_timesteps = [800.0, 400.0]

    with torch.no_grad():
        tf_out = cosmos3_teacher_forcing_reference(
            tf_transformer,
            **{**kwargs, "timestep": _teacher_forcing_timesteps(block_timesteps)},
            clean_x=clean,
        )[0]
    assert tf_out.shape == noisy.shape

    cache = rollout_transformer.new_rollout_kv_cache(truncate_rollout_gradients=False)
    with torch.no_grad():
        for block_index in range(TF_NUM_BLOCKS):
            start = block_index * TF_BLOCK_FRAMES
            end = start + TF_BLOCK_FRAMES
            if block_index > 0:
                seed_start = start - TF_BLOCK_FRAMES
                rollout_transformer(
                    **_chunk_kwargs(kwargs, seed_start, start),
                    hidden_states=clean[:, :, seed_start:start],
                    timestep=None,
                    kv_cache=cache,
                    kv_cache_seed=True,
                    kv_cache_start_frame=seed_start,
                )
            denoise_out = rollout_transformer(
                **_chunk_kwargs(kwargs, start, end),
                hidden_states=noisy[:, :, start:end],
                timestep=torch.tensor([block_timesteps[block_index]]),
                kv_cache=cache,
                kv_cache_start_frame=start,
            )[0]
            assert denoise_out.shape == (1, LATENT_CHANNELS, TF_BLOCK_FRAMES, 4, 4)
            torch.testing.assert_close(
                tf_out[:, :, start:end],
                denoise_out,
                atol=1e-5,
                rtol=1e-5,
                msg=lambda base, i=block_index: f"block {i}: TF slice diverged from the cached rollout\n{base}",
            )


def test_teacher_forcing_weave_contract():
    """Put clean frames before noisy frames, reusing positions and camera inputs for both.

    Only the noisy half receives timesteps and predictions. This reference
    calculation stays in tests; production forwards reject clean_x.
    """
    torch.manual_seed(1)
    base = _TinyCosmosBaseTransformer()
    transformer = _tiny_transformer(base_transformer=base, causal_block_frames=TF_BLOCK_FRAMES)
    noisy = torch.randn(1, LATENT_CHANNELS, TF_FRAMES, 4, 4)
    clean = torch.randn(1, LATENT_CHANNELS, TF_FRAMES, 4, 4)
    kwargs = _forward_kwargs(noisy, neighbor_hidden_states=None)
    per_token_timesteps = _teacher_forcing_timesteps([800.0, 400.0])

    # Both halves must reuse the positions from a normal call.
    transformer(**kwargs)
    reference_call = base.calls[-1]
    reference_vision_ids = reference_call["position_ids"][:, reference_call["vision_sequence_indexes"]]

    seen_context: dict[str, torch.Tensor] = {}

    def capture_context(_module, _inputs):
        context = transformer._conditioning_context
        seen_context["opacity_tokens"] = context.opacity_tokens.detach().clone()
        seen_context["scope_rays"] = context.scope_rays.detach().clone()

    base.layers[0].self_attn.register_forward_pre_hook(capture_context)

    output = cosmos3_teacher_forcing_reference(
        transformer, **{**kwargs, "timestep": per_token_timesteps}, clean_x=clean
    )[0]
    call = base.calls[-1]

    noisy_tokens = TF_FRAMES * TF_SPATIAL_TOKENS
    assert output.shape == noisy.shape
    assert torch.equal(call["vision_noisy_frame_indexes"][0], torch.arange(TF_FRAMES, 2 * TF_FRAMES))
    vision_ids = call["position_ids"][:, call["vision_sequence_indexes"]]
    assert vision_ids.shape[1] == 2 * noisy_tokens
    assert torch.equal(vision_ids[:, :noisy_tokens], vision_ids[:, noisy_tokens:])
    assert torch.equal(vision_ids[:, :noisy_tokens], reference_vision_ids)
    assert call["vision_timesteps"].shape == (noisy_tokens,)
    torch.testing.assert_close(call["vision_timesteps"], per_token_timesteps[0], rtol=0, atol=0)
    for name in ("opacity_tokens", "scope_rays"):
        tokens = seen_context[name]
        assert tokens.shape[0] == 2 * noisy_tokens, f"{name} must double for the [clean ⊕ noisy] weave"
        assert torch.equal(tokens[:noisy_tokens], tokens[noisy_tokens:])
