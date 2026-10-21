# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward, gradient, and distributed checks using a reduced real Cosmos3 model."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

# The public diffusers export is the exact surface train_utils imports from.
from diffusers import Cosmos3OmniTransformer
from torch.distributed import DeviceMesh
from torch.nn.attention import SDPBackend, sdpa_kernel

import model_training.net.cosmos3_transformer as cosmos3_transformer_module
from model_training.net.cosmos3_transformer import ArtifixerCosmos3Transformer
from model_training.pipeline.cosmos3_pipeline import ArtifixerCosmos3Pipeline

# The REAL reduced-config doubles live in tests/_helpers.py (their canonical
# home); this module remains the contract layer that
# drives them against the installed diffusers class.
from tests._helpers import (
    GROWING_PREFIX_FRAMES,
    LATENT_HEIGHT,
    LATENT_WIDTH,
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    REDUCED_CONFIG,
    VAE_SCALE_FACTOR_SPATIAL,
    VAE_SCALE_FACTOR_TEMPORAL,
    _activate_conditioning_adapters,
    _assert_cosmos3_cp_gradient_mean,
    _capture_cosmos3_forward,
    _cosmos3_noisy_sequence_indexes,
    _growing_prefix_kwargs,
    _joint_reference_kwargs,
    _reduced_real_transformer,
    _run_cp_workers,
    _select_cosmos3_references,
    _slice_cosmos3_target_kwargs,
    _TinyCosmosVae,
)
from tests._teacher_forcing_reference import cosmos3_teacher_forcing_reference

pytestmark = pytest.mark.slow

LATENT_FRAMES = 2
PIXEL_FRAMES = (LATENT_FRAMES - 1) * VAE_SCALE_FACTOR_TEMPORAL + 1


def _contract_forward_kwargs() -> dict:
    torch.manual_seed(1)
    kwargs = dict(
        hidden_states=torch.randn(1, 48, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
        # fp32 t per prepare_transformer_timestep's fp32 contract.
        timestep=torch.tensor([999.0]),
        encoder_hidden_states=torch.tensor([[5, 6, 7, 8]]),
        encoder_attention_mask=torch.ones(1, 4, dtype=torch.bool),
        neighbor_hidden_states=torch.randn(1, 48, 1, LATENT_HEIGHT, LATENT_WIDTH),
        opacity=torch.rand(1, PIXEL_FRAMES, PIXEL_HEIGHT, PIXEL_WIDTH),
        camera_rays=torch.randn(1, LATENT_FRAMES, PIXEL_HEIGHT, PIXEL_WIDTH, 6),
        neighbor_camera_rays=torch.randn(1, 1, PIXEL_HEIGHT, PIXEL_WIDTH, 6),
        w2cs=torch.eye(4).view(1, 1, 4, 4).expand(1, LATENT_FRAMES, -1, -1).clone(),
        neighbor_w2cs=torch.eye(4).view(1, 1, 4, 4).clone(),
        Ks=torch.eye(3).view(1, 1, 3, 3).expand(1, LATENT_FRAMES, -1, -1).clone(),
        neighbor_Ks=torch.eye(3).view(1, 1, 3, 3).clone(),
    )
    return kwargs


def test_real_cosmos3_forward_contract():
    """One forward through the REAL v0.39.0 class: output keeps the latent
    shape, dtype, and finiteness."""
    transformer = _reduced_real_transformer()
    kwargs = _contract_forward_kwargs()

    with torch.no_grad():
        output = transformer(**kwargs)

    assert isinstance(output, tuple) and len(output) == 1
    assert output[0].shape == kwargs["hidden_states"].shape
    assert output[0].dtype == kwargs["hidden_states"].dtype
    assert bool(torch.isfinite(output[0]).all())


def test_real_cosmos3_forward_contract_under_ambient_bf16_autocast():
    """End-to-end replay of the production regime: bf16 compute params (what
    the FSDP2 mixed-precision policy hands the forward) and bf16 latents under
    ambient autocast (accelerate wraps the prepared forward) through the REAL
    class. The output must stay finite bf16 (the fp32 timestep island upcasts
    its bf16 weights per call)."""
    transformer = _reduced_real_transformer().to(torch.bfloat16)
    kwargs = _contract_forward_kwargs()
    kwargs["hidden_states"] = kwargs["hidden_states"].to(torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        output = transformer(**kwargs)

    assert output[0].shape == kwargs["hidden_states"].shape
    assert bool(torch.isfinite(output[0].float()).all())


# --- stage-2 SF-DMD conditioning-mode forwards on the REAL class ---

# Reduced latent geometry: 4x4 latents with latent_patch_size 2 -> 2x2 = 4
# vision tokens per latent frame.
SPATIAL_TOKENS_PER_FRAME = (LATENT_HEIGHT // REDUCED_CONFIG["latent_patch_size"]) * (
    LATENT_WIDTH // REDUCED_CONFIG["latent_patch_size"]
)


def _gen_seq_capture_pre_hook(records: list):
    """Capture the generation-stream hidden states entering a real decoder layer
    (the base transformer calls decoder_layer(und_seq, gen_seq, rotary_emb))."""

    def hook(_module, inputs):
        records.append(inputs[1].detach().clone())

    return hook


def test_real_cosmos3_seed_passes_inject_no_time_embedding():
    """KV-cache seed passes are pure conditioning mode on the REAL v0.39.0
    class: the wrapper hands the base transformer an EMPTY noisy set and an
    EMPTY vision_timesteps scatter-src (diffusers gates the timestep
    injection on vision_noisy_frame_indexes, transformer_cosmos3.py
    _apply_timestep_embeds_to_noisy_tokens), the returned prediction is the
    base's zero-fill (nothing is denoised), and the gen tokens entering
    layer 0 carry NO timestep embedding — denoise passes over the same
    latents move with t, the seed pass matches neither t."""
    transformer = _reduced_real_transformer()
    kwargs = _contract_forward_kwargs()
    clean_chunk = kwargs.pop("hidden_states")
    kwargs.pop("timestep")

    base_kwargs_seen: list[dict] = []

    def capture_base_kwargs(_module, _args, base_kwargs):
        base_kwargs_seen.append(base_kwargs)
        return None

    records: list[torch.Tensor] = []
    base = transformer.base_transformer
    capture_handle = base.register_forward_pre_hook(capture_base_kwargs, with_kwargs=True)
    layer_handle = base.layers[0].register_forward_pre_hook(_gen_seq_capture_pre_hook(records))
    try:
        with torch.no_grad():
            seed_output = transformer(
                **kwargs,
                hidden_states=clean_chunk,
                timestep=None,
                kv_cache=transformer.new_rollout_kv_cache(truncate_rollout_gradients=False),
                kv_cache_seed=True,
            )[0]
            # Denoise passes over the SAME latents at two different t (fresh
            # caches: chunk-0 denoise passes write no gen history).
            transformer(
                **kwargs,
                hidden_states=clean_chunk,
                timestep=torch.tensor([999.0]),
                kv_cache=transformer.new_rollout_kv_cache(truncate_rollout_gradients=False),
            )
            transformer(
                **kwargs,
                hidden_states=clean_chunk,
                timestep=torch.tensor([1.0]),
                kv_cache=transformer.new_rollout_kv_cache(truncate_rollout_gradients=False),
            )
    finally:
        capture_handle.remove()
        layer_handle.remove()

    # Wrapper -> base seam: zero noisy tokens, empty timestep scatter-src.
    seed_base_kwargs = base_kwargs_seen[0]
    assert seed_base_kwargs["vision_timesteps"].numel() == 0
    assert all(indexes.numel() == 0 for indexes in seed_base_kwargs["vision_noisy_frame_indexes"])
    # Nothing is denoised: the base zero-fills every (conditioned) frame.
    assert seed_output.shape == clean_chunk.shape
    assert torch.count_nonzero(seed_output) == 0
    # No timestep embedding on seed rows: t moves every denoise row but the
    # seed pass matches neither denoise variant.
    gen_seed, gen_hi_t, gen_lo_t = records
    assert not torch.equal(gen_hi_t, gen_lo_t)
    assert not torch.equal(gen_seed, gen_hi_t)
    assert not torch.equal(gen_seed, gen_lo_t)


# --- supervised diffusion forcing: per-block sigma routing on the REAL class ---


def _reduced_real_df_pipe(frames_per_block: int) -> ArtifixerCosmos3Pipeline:
    torch.manual_seed(0)
    return ArtifixerCosmos3Pipeline(
        vae=_TinyCosmosVae(),
        scheduler=SimpleNamespace(),
        transformer=Cosmos3OmniTransformer(**REDUCED_CONFIG),
        gradient_checkpointing=False,
        frames_per_block=frames_per_block,
    )


def test_real_cosmos3_per_block_timesteps_route_block_sigmas_to_block_tokens():
    """Diffusion-forcing semantics end-to-end on the REAL v0.39.0 class (paper
    Sec 4.2 Initialization): the trainer-shaped flat (B * num_blocks,)
    timestep goes through the pipeline's prepare_transformer_timestep
    expansion and the wrapper forward, and every vision token of block k
    receives block k's sigma:

    - the base transformer's vision_timesteps kwarg (the scatter src — one
      embedding row per noisy token) carries block-0's t on frame-0 rows and
      block-1's t on frame-1 rows, fp32;
    - at the embedding level, changing ONLY block k's t moves ONLY block k's
      vision tokens entering layer 0 (the additive time embed lands per
      token), pinning the sample-major routing against off-by-one/-block."""
    pipe = _reduced_real_df_pipe(frames_per_block=1)  # LATENT_FRAMES=2 -> 2 blocks
    kwargs = _contract_forward_kwargs()
    hidden_states = kwargs.pop("hidden_states")
    kwargs.pop("timestep")

    seen_vision_timesteps: list[torch.Tensor] = []
    base = pipe.transformer.base_transformer

    def capture_vision_timesteps(_module, _args, base_kwargs):
        seen_vision_timesteps.append(base_kwargs["vision_timesteps"].detach().clone())
        return None

    gen_records: list[torch.Tensor] = []
    capture_handle = base.register_forward_pre_hook(capture_vision_timesteps, with_kwargs=True)
    layer_handle = base.layers[0].register_forward_pre_hook(_gen_seq_capture_pre_hook(gen_records))

    def run(per_block_t: list[float]) -> torch.Tensor:
        transformer_timestep = pipe.prepare_transformer_timestep(
            torch.tensor(per_block_t), hidden_states, frames_per_block=1
        )
        with torch.no_grad():
            return pipe.transformer(hidden_states=hidden_states, timestep=transformer_timestep, **kwargs)[0]

    try:
        output = run([999.0, 1.0])
        run([999.0, 999.0])
        run([1.0, 1.0])
    finally:
        capture_handle.remove()
        layer_handle.remove()

    assert output.shape == hidden_states.shape
    assert bool(torch.isfinite(output).all())

    mixed_timesteps = seen_vision_timesteps[0]
    assert mixed_timesteps.dtype == torch.float32
    assert mixed_timesteps.shape == (LATENT_FRAMES * SPATIAL_TOKENS_PER_FRAME,)
    assert mixed_timesteps[:SPATIAL_TOKENS_PER_FRAME].tolist() == [999.0] * SPATIAL_TOKENS_PER_FRAME
    assert mixed_timesteps[SPATIAL_TOKENS_PER_FRAME:].tolist() == [1.0] * SPATIAL_TOKENS_PER_FRAME

    reference_rows = kwargs["neighbor_hidden_states"].shape[2] * SPATIAL_TOKENS_PER_FRAME
    frame0_rows = slice(reference_rows, reference_rows + SPATIAL_TOKENS_PER_FRAME)
    frame1_rows = slice(reference_rows + SPATIAL_TOKENS_PER_FRAME, reference_rows + 2 * SPATIAL_TOKENS_PER_FRAME)
    gen_mixed, gen_block0_t, gen_block1_t = gen_records
    assert torch.equal(gen_mixed[:reference_rows], gen_block0_t[:reference_rows])
    assert torch.equal(gen_mixed[:reference_rows], gen_block1_t[:reference_rows])
    # [999, 1] vs [999, 999]: block 0 untouched, block 1 moved.
    assert torch.equal(gen_mixed[frame0_rows], gen_block0_t[frame0_rows])
    assert not torch.equal(gen_mixed[frame1_rows], gen_block0_t[frame1_rows])
    # [999, 1] vs [1, 1]: block 1 untouched, block 0 moved.
    assert torch.equal(gen_mixed[frame1_rows], gen_block1_t[frame1_rows])
    assert not torch.equal(gen_mixed[frame0_rows], gen_block1_t[frame0_rows])


# --- supervised diffusion forcing: block-causal semantics on the REAL class ---


def test_real_cosmos3_block_causal_forward_is_causal_across_blocks_only_with_the_contract():
    """Causality semantics on the real class (paper Sec 4.2
    Initialization): with causal_block_frames set, perturbing a
    later block's latents leaves every EARLIER block's predictions
    bit-identical (block-causal information flow — training-time semantics
    match AR inference); without the contract the same perturbation
    contaminates block 0 (full bidirectional attention), proving the
    invariance is enforced by the supervised mask, not vacuous."""
    kwargs = _growing_prefix_kwargs(GROWING_PREFIX_FRAMES)  # 3 frames, 1-frame blocks
    perturbed_hidden = kwargs["hidden_states"].clone()
    perturbed_hidden[:, :, GROWING_PREFIX_FRAMES - 1 :] += 1.0

    transformer = _reduced_real_transformer(causal_block_frames=1)
    with torch.no_grad():
        baseline = transformer(**kwargs)[0]
        perturbed = transformer(**{**kwargs, "hidden_states": perturbed_hidden})[0]

    assert baseline.shape == kwargs["hidden_states"].shape
    earlier_blocks = slice(0, GROWING_PREFIX_FRAMES - 1)
    assert torch.equal(
        baseline[:, :, earlier_blocks], perturbed[:, :, earlier_blocks]
    ), "perturbing the last block leaked into earlier blocks' predictions despite the block-causal mask"
    assert not torch.equal(baseline[:, :, -1:], perturbed[:, :, -1:])

    bidirectional = _reduced_real_transformer()
    with torch.no_grad():
        bidirectional_baseline = bidirectional(**kwargs)[0]
        bidirectional_perturbed = bidirectional(**{**kwargs, "hidden_states": perturbed_hidden})[0]
    assert not torch.equal(
        bidirectional_baseline[:, :, earlier_blocks], bidirectional_perturbed[:, :, earlier_blocks]
    ), "without the contract the perturbation must reach earlier blocks — the equality above would prove nothing"


def test_real_cosmos3_block_causal_supervised_gradients_flow_to_all_blocks(monkeypatch):
    """The anti-truncation check: supervised DF gradients flow ACROSS
    blocks — unlike the cached AR rollout, where truncated BPTT (detached
    cache stores) gives a later chunk's loss exactly zero gradient on
    earlier chunks — a last-block loss must send nonzero gradient to every
    earlier block's input frames through attention K/V, and a full loss must
    reach every block. The processor spy pins that the mask was armed on
    every attention call."""
    transformer = _reduced_real_transformer(causal_block_frames=1)
    kwargs = _growing_prefix_kwargs(GROWING_PREFIX_FRAMES)

    armed_states: list[bool] = []
    processor_cls = cosmos3_transformer_module.ArtifixerCosmos3AttnProcessor
    original_call = processor_cls.__call__

    def recording_call(self, *args, **kw):
        armed_states.append(self._gen_block_mask is not None)
        return original_call(self, *args, **kw)

    monkeypatch.setattr(processor_cls, "__call__", recording_call)

    def run(loss_frames: slice) -> torch.Tensor:
        for parameter in transformer.parameters():
            parameter.grad = None
        hidden_states = kwargs["hidden_states"].detach().clone().requires_grad_(True)
        output = transformer(**{**kwargs, "hidden_states": hidden_states})[0]
        output[:, :, loss_frames].square().sum().backward()
        assert hidden_states.grad is not None
        return hidden_states.grad.detach().clone()

    last_block = GROWING_PREFIX_FRAMES - 1
    last_block_grad = run(slice(last_block, GROWING_PREFIX_FRAMES))
    for frame in range(GROWING_PREFIX_FRAMES):
        assert torch.count_nonzero(last_block_grad[:, :, frame]) > 0, (
            f"a last-block loss must reach block {frame}'s input through attention — "
            "zero gradient means rollout-style truncation leaked onto the supervised path"
        )

    full_grad = run(slice(0, GROWING_PREFIX_FRAMES))
    for frame in range(GROWING_PREFIX_FRAMES):
        assert torch.count_nonzero(full_grad[:, :, frame]) > 0, f"block {frame} received no gradient"
    for adapter in transformer.opacity_embeddings:
        assert adapter.weight.grad is not None
        assert torch.count_nonzero(adapter.weight.grad) > 0

    assert armed_states, "the processor spy recorded nothing"
    assert all(armed_states), "the supervised mask must be armed on every attention call"


def test_real_cosmos3_block_causal_gradients_identical_under_checkpointing():
    """Gradient-checkpointing recompute re-arms the supervised mask through
    the same conditioning-context pre-hook (the rollout twin is the
    snapshotted-history checkpointing test in
    tests/test_cosmos3_kv_cache_rollout.py): parameter gradients must be
    bit-identical with checkpointing on and off, and every processor must be
    disarmed after forward AND after backward (the recompute re-arms and
    re-clears)."""
    kwargs = _growing_prefix_kwargs(GROWING_PREFIX_FRAMES)

    def assert_disarmed(transformer: ArtifixerCosmos3Transformer) -> None:
        for layer in transformer.base_transformer.layers:
            assert layer.self_attn.processor._gen_block_mask is None

    def run(gradient_checkpointing: bool) -> dict[str, torch.Tensor | None]:
        transformer = _reduced_real_transformer(
            gradient_checkpointing=gradient_checkpointing,
            causal_block_frames=1,
        )
        output = transformer(**kwargs)[0]
        assert_disarmed(transformer)
        output.square().sum().backward()
        assert_disarmed(transformer)
        return {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad
        }

    plain_grads = run(gradient_checkpointing=False)
    checkpointed_grads = run(gradient_checkpointing=True)
    assert plain_grads.keys() == checkpointed_grads.keys()
    graded = 0
    for name, plain_grad in plain_grads.items():
        checkpointed_grad = checkpointed_grads[name]
        if plain_grad is None:
            assert checkpointed_grad is None, name
            continue
        graded += 1
        assert checkpointed_grad is not None and torch.equal(
            plain_grad, checkpointed_grad
        ), f"{name}: gradient checkpointing changed the supervised block-causal parameter gradient"
    assert graded > 0, "gradient comparison compared nothing"


def test_real_cosmos3_diffusion_forcing_real_trainer_training_batch_end_to_end(monkeypatch):
    """Integration pin: drive the REAL shared ``Trainer.training_batch`` — the exact production
    call chain (process_inputs -> z_mix prepare_latents ->
    add_noise_multi_timestep -> ``prepare_transformer_timestep(timestep,
    model_input, frames_per_block=...)`` -> wrapper forward under the armed
    block-causal mask -> bell-weighted global-mean loss -> backward) — with
    the REAL reduced Cosmos3OmniTransformer, the real FlowMatchScheduler
    (cosmos3 profile shift), and frames_per_block set. The component tests
    pin each hop in isolation; this pins that the hops AGREE end to end:
    the flat per-block timestep the trainer samples reaches the wrapper as
    the exact per-token [B, N] fp32 rows, the supervised mask is armed on
    every attention call, the weighted loss is a
    finite scalar, and backward reaches every trainable adapter. Only the
    frozen-VAE pixel encodes are stubbed (shape-preserving, deterministic);
    every trainer/scheduler/pipeline/wrapper line is the shipped code."""
    from model_training.model_profiles import flow_shift_for_model
    from model_training.schedulers.flow_match import FlowMatchScheduler
    from model_training.trainers.trainer import Trainer

    torch.manual_seed(4)
    pipe = _reduced_real_df_pipe(frames_per_block=1)  # 2 latent frames -> 2 blocks

    def fake_encode_video_frames(video_frames):
        # [B, 1+4N, 3, H, W] -> [B, 48, N+1, H/16, W/16] without the frozen VAE.
        latents = video_frames[
            :, ::VAE_SCALE_FACTOR_TEMPORAL, :, ::VAE_SCALE_FACTOR_SPATIAL, ::VAE_SCALE_FACTOR_SPATIAL
        ]
        return latents.permute(0, 2, 1, 3, 4).repeat(1, 16, 1, 1, 1)

    def fake_encode_neighbors(neighbor_frames, max_neighbors_per_encode=None):
        del max_neighbors_per_encode
        latents = neighbor_frames[:, :, :, ::VAE_SCALE_FACTOR_SPATIAL, ::VAE_SCALE_FACTOR_SPATIAL]
        return latents.permute(0, 2, 1, 3, 4).repeat(1, 16, 1, 1, 1)

    monkeypatch.setattr(pipe, "encode_video_frames", fake_encode_video_frames)
    monkeypatch.setattr(pipe, "encode_neighbors", fake_encode_neighbors)

    trainer = object.__new__(Trainer)
    trainer.args = SimpleNamespace(model_id="nvidia/Cosmos3-Nano", dropout_rate=0.0)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    trainer.pipe = pipe
    trainer.frames_per_block = 1
    trainer._zero_encode_cache = {}
    # The exact ctor recipe of Trainer.__init__ (shift comes from the cosmos3
    # profile — the single shift source).
    trainer.train_scheduler = FlowMatchScheduler(
        shift=flow_shift_for_model(trainer.args.model_id),
        sigma_min=0.0,
        extra_one_step=True,
        device=pipe.vae.device,
    )

    recorded: dict = {}
    original_prepare = ArtifixerCosmos3Pipeline.prepare_transformer_timestep

    def spying_prepare(self, timestep, hidden_states, frames_per_block=None):
        result = original_prepare(self, timestep, hidden_states, frames_per_block)
        recorded["timestep"] = timestep.detach().clone()
        recorded["frames_per_block"] = frames_per_block
        recorded["result"] = result.detach().clone()
        return result

    monkeypatch.setattr(ArtifixerCosmos3Pipeline, "prepare_transformer_timestep", spying_prepare)

    armed_states: list[bool] = []
    processor_cls = cosmos3_transformer_module.ArtifixerCosmos3AttnProcessor
    original_call = processor_cls.__call__

    def recording_call(self, *args, **kw):
        armed_states.append(self._gen_block_mask is not None)
        return original_call(self, *args, **kw)

    monkeypatch.setattr(processor_cls, "__call__", recording_call)

    data = dict(
        rgb_gt=torch.rand(1, PIXEL_FRAMES, 3, PIXEL_HEIGHT, PIXEL_WIDTH),
        rgb_rendered=torch.rand(1, PIXEL_FRAMES, 3, PIXEL_HEIGHT, PIXEL_WIDTH),
        opacity=torch.rand(1, PIXEL_FRAMES, PIXEL_HEIGHT, PIXEL_WIDTH),
        rgb_neighbors=torch.rand(1, 1, 3, PIXEL_HEIGHT, PIXEL_WIDTH),
        encoded_prompt=torch.tensor([[5, 6, 7, 8]]),
        prompt_attention_mask=torch.ones(1, 4, dtype=torch.bool),
        camera_rays=torch.randn(1, LATENT_FRAMES, PIXEL_HEIGHT, PIXEL_WIDTH, 6),
        neighbor_camera_rays=torch.randn(1, 1, PIXEL_HEIGHT, PIXEL_WIDTH, 6),
        w2cs=torch.eye(4).view(1, 1, 4, 4).expand(1, LATENT_FRAMES, -1, -1).clone(),
        neighbor_w2cs=torch.eye(4).view(1, 1, 4, 4).clone(),
        Ks=torch.eye(3).view(1, 1, 3, 3).expand(1, LATENT_FRAMES, -1, -1).clone(),
        neighbor_Ks=torch.eye(3).view(1, 1, 3, 3).clone(),
    )

    loss, metrics = trainer.training_batch(data)

    # The trainer sampled PER-BLOCK timesteps (B * num_blocks = 2, not per-
    # sample [B] = 1) and passed its frames_per_block kwarg to the pipeline
    # override, which expanded them to the wrapper's exact per-token
    # [B, num_noisy_vision_tokens] fp32 contract in block -> frame -> spatial
    # order.
    per_block = recorded["timestep"]
    assert per_block.shape == (LATENT_FRAMES,)
    assert recorded["frames_per_block"] == 1
    assert recorded["result"].shape == (1, LATENT_FRAMES * SPATIAL_TOKENS_PER_FRAME)
    assert recorded["result"].dtype == torch.float32
    assert torch.equal(
        recorded["result"][0],
        per_block.to(torch.float32).repeat_interleave(SPATIAL_TOKENS_PER_FRAME),
    )

    # The supervised block-causal mask was armed on every attention call of
    # the training forward.
    assert armed_states, "the processor spy recorded nothing"
    assert all(armed_states)

    # Weighted DF loss: one finite scalar with a live graph; the logged metric
    # is the unweighted elementwise-MSE mean.
    assert loss.ndim == 0 and loss.requires_grad
    assert bool(torch.isfinite(loss))
    assert bool(torch.isfinite(metrics["mse_loss"]))

    loss.backward()
    for adapter in pipe.transformer.opacity_embeddings:
        assert adapter.weight.grad is not None
        assert bool(torch.isfinite(adapter.weight.grad).all())
        assert torch.count_nonzero(adapter.weight.grad) > 0
    # The base model's action modules are undriven and frozen: no gradient may
    # ever reach them through the training forward.
    action_parameters = list(pipe.transformer.base_transformer.action_proj_in.parameters())
    assert action_parameters
    assert all(
        not parameter.requires_grad and parameter.grad is None for parameter in action_parameters
    ), "the undriven base action projection must stay frozen and gradient-free"


def test_real_cosmos3_gradient_checkpointed_backward_reaches_adapters():
    """The wrapper hands its context-snapshotting checkpoint function to the
    real ``ModelMixin.enable_gradient_checkpointing`` and the real layers call
    it during forward; backward must recompute through the conditioning hooks
    and deliver finite grads to the trainable ArtiFixer additions while the
    frozen reasoner stays grad-free (freeze_cosmos3_reasoner_parameters)."""
    transformer = _reduced_real_transformer(gradient_checkpointing=True)
    assert transformer.base_transformer.gradient_checkpointing

    output = transformer(**_contract_forward_kwargs())[0]
    output.square().sum().backward()

    for adapter in transformer.opacity_embeddings:
        assert adapter.weight.grad is not None
        assert bool(torch.isfinite(adapter.weight.grad).all())
        assert torch.count_nonzero(adapter.weight.grad) > 0
    for layer in transformer.base_transformer.layers:
        for projection in (layer.self_attn.add_k_proj, layer.self_attn.add_v_proj):
            assert projection.weight.grad is not None
            assert torch.isfinite(projection.weight.grad).all()
    # Trainable generation-branch module of the real base transformer.
    assert transformer.base_transformer.proj_out.weight.grad is not None
    # Frozen reasoner: no grads accumulate on the understanding stack.
    assert not transformer.base_transformer.embed_tokens.weight.requires_grad
    assert transformer.base_transformer.embed_tokens.weight.grad is None


def _vision_positions(base_kwargs: dict) -> torch.Tensor:
    return base_kwargs["position_ids"][:, base_kwargs["vision_sequence_indexes"]]


def _expected_scope_rays(pixel_rays: torch.Tensor) -> torch.Tensor:
    """Independent pixel-window mean, converting moment-first maps to SCoPE's order."""
    batch, frames, height, width, _ = pixel_rays.shape
    patch = VAE_SCALE_FACTOR_SPATIAL * REDUCED_CONFIG["latent_patch_size"]
    means = pixel_rays.reshape(batch, frames, height // patch, patch, width // patch, patch, 6)
    means = means.mean(dim=(3, 5)).reshape(-1, 6)
    moment, direction = means[:, :3], means[:, 3:]
    return torch.cat([direction, moment], dim=-1) / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)


@pytest.mark.parametrize("latent_hw", [(LATENT_HEIGHT, LATENT_WIDTH), (LATENT_HEIGHT, LATENT_WIDTH + 2)])
def test_joint_fixture_has_distorted_moment_first_rays_in_the_first_target_frame(latent_hw):
    kwargs = _joint_reference_kwargs(latent_hw=latent_hw)
    assert kwargs["hidden_states"].shape[-2:] == kwargs["neighbor_hidden_states"].shape[-2:] == latent_hw
    pixel_hw = tuple(side * VAE_SCALE_FACTOR_SPATIAL for side in latent_hw)
    torch.testing.assert_close(kwargs["w2cs"][0, 0], torch.eye(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        kwargs["camera_rays"][:, :1, ..., :3],
        torch.zeros_like(kwargs["camera_rays"][:, :1, ..., :3]),
        atol=1e-6,
        rtol=0,
    )
    for ray_name, pose_name in (("camera_rays", "w2cs"), ("neighbor_camera_rays", "neighbor_w2cs")):
        rays = kwargs[ray_name][0]
        assert rays.shape == (kwargs[pose_name].shape[1], *pixel_hw, 6)
        origins = torch.linalg.inv(kwargs[pose_name][0])[:, :3, 3]
        moments = torch.linalg.cross(origins[:, None, None].expand_as(rays[..., 3:]), rays[..., 3:])
        torch.testing.assert_close(rays[..., :3], moments)
    assert (
        torch.count_nonzero(kwargs["neighbor_camera_rays"][0, 0, ..., :3]) > 0
    ), "the first reference must not become the camera anchor"
    changed_lens = _joint_reference_kwargs(latent_hw=latent_hw, reference_distortion_delta=0.2)
    for name in ("neighbor_Ks", "neighbor_w2cs", "camera_rays", "w2cs"):
        torch.testing.assert_close(kwargs[name], changed_lens[name], rtol=0, atol=0)
    assert not torch.equal(kwargs["neighbor_camera_rays"], changed_lens["neighbor_camera_rays"])


@pytest.mark.parametrize("frame_offset", [0, 5])
@pytest.mark.parametrize("latent_hw", [(LATENT_HEIGHT, LATENT_WIDTH), (LATENT_HEIGHT, LATENT_WIDTH + 2)])
def test_real_joint_positions_and_scope_keep_target_coordinates_when_references_change(frame_offset, latent_hw):
    """References repeat frame zero and the shared spatial grid, never consume target time IDs."""
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    full = _joint_reference_kwargs(frame_offset + 3, latent_hw=latent_hw)
    assert full["hidden_states"].shape[-2:] == full["neighbor_hidden_states"].shape[-2:] == latent_hw
    kwargs = _slice_cosmos3_target_kwargs(full, frame_offset, frame_offset + 3)
    kwargs["frame_offset"] = frame_offset
    with torch.no_grad(), _capture_cosmos3_forward(transformer) as baseline_trace:
        baseline = transformer(**_select_cosmos3_references(kwargs, []))[0]
    baseline_call = baseline_trace.base_kwargs[0]
    target_positions = _vision_positions(baseline_call)
    text_positions = baseline_call["position_ids"][:, baseline_call["text_indexes"]]
    zero_anchor = float(text_positions[0].max()) + 1 + REDUCED_CONFIG["unified_3d_mrope_temporal_modality_margin"]
    patch_h, patch_w = (side // REDUCED_CONFIG["latent_patch_size"] for side in latent_hw)
    per_reference = patch_h * patch_w
    expected_reference_positions = torch.stack(
        [
            torch.full((per_reference,), zero_anchor),
            torch.arange(patch_h).repeat_interleave(patch_w),
            torch.arange(patch_w).repeat(patch_h),
        ]
    ).to(target_positions)
    for selected in ([0], [0, 1, 2], [2, 0, 1]):
        inputs = _select_cosmos3_references(kwargs, selected)
        with torch.no_grad(), _capture_cosmos3_forward(transformer) as trace:
            output = transformer(**inputs)[0]
        call = trace.base_kwargs[0]
        packed_frames = [frame for segment in call["vision_tokens"] for frame in segment.unbind(dim=2)]
        expected_frames = [
            *inputs["neighbor_hidden_states"].unbind(dim=2),
            *inputs["hidden_states"].unbind(dim=2),
        ]
        assert len(packed_frames) == len(expected_frames)
        for packed, expected in zip(packed_frames, expected_frames, strict=True):
            torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        positions = _vision_positions(call)
        reference_tokens = len(selected) * per_reference
        assert output.shape == baseline.shape == kwargs["hidden_states"].shape
        assert positions.shape[1] == reference_tokens + target_positions.shape[1]
        torch.testing.assert_close(positions[:, reference_tokens:], target_positions, rtol=0, atol=0)
        torch.testing.assert_close(
            positions[:, :reference_tokens], expected_reference_positions.repeat(1, len(selected)), rtol=0, atol=0
        )
        torch.testing.assert_close(call["position_ids"][:, call["text_indexes"]], text_positions, rtol=0, atol=0)
        expected_rays = torch.cat(
            [_expected_scope_rays(inputs["neighbor_camera_rays"]), _expected_scope_rays(inputs["camera_rays"])], dim=0
        )
        for rays_seen in trace.scope_rays:
            assert len(rays_seen) == 1
            torch.testing.assert_close(rays_seen[0], expected_rays, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "target_hw,reference_hw",
    [((4, 4), (4, 6)), ((4, 6), (6, 4))],
    ids=["mismatched_width", "transposed_rectangle_same_token_count"],
)
def test_real_joint_rejects_mismatched_reference_and_target_latent_grids(target_hw, reference_hw):
    """Joint mode retains PRoPE's H/W-match constraint, not merely equal token counts."""
    transformer = _reduced_real_transformer()
    kwargs = _joint_reference_kwargs(total_frames=1, latent_hw=target_hw)
    reference_inputs = _joint_reference_kwargs(total_frames=1, latent_hw=reference_hw)
    for name in ("w2cs", "neighbor_w2cs"):
        torch.testing.assert_close(kwargs[name], reference_inputs[name], rtol=0, atol=0)
    for name in ("neighbor_hidden_states", "neighbor_camera_rays", "neighbor_w2cs", "neighbor_Ks"):
        kwargs[name] = reference_inputs[name]
    with pytest.raises(AssertionError, match="matching latent patch grids"):
        transformer(**kwargs)


def test_real_joint_rope_precedes_live_scope_for_every_visual_token():
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    kwargs = _joint_reference_kwargs()
    with torch.no_grad(), _capture_cosmos3_forward(transformer) as trace:
        transformer(**kwargs)
    reference_tokens = 3 * SPATIAL_TOKENS_PER_FRAME
    with torch.no_grad():
        for layer_index, layer in enumerate(transformer.base_transformer.layers):
            attention = layer.self_attn
            features = trace.attention_after_opacity[layer_index][0]
            cos_gen, sin_gen = trace.attention_rotary[layer_index][0][2:]
            assert features.shape[0] == reference_tokens + GROWING_PREFIX_FRAMES * SPATIAL_TOKENS_PER_FRAME
            for projection, norm, heads, observed, scoped in (
                (
                    attention.add_q_proj,
                    attention.norm_added_q,
                    attention.num_attention_heads,
                    trace.scope_queries[layer_index][0],
                    trace.scope_outputs[layer_index][0][0],
                ),
                (
                    attention.add_k_proj,
                    attention.norm_added_k,
                    attention.num_key_value_heads,
                    trace.scope_keys[layer_index][0],
                    trace.scope_outputs[layer_index][0][1],
                ),
            ):
                unrotated = norm(projection(features).view(-1, heads, attention.head_dim))
                first, second = unrotated.chunk(2, dim=-1)
                expected = unrotated * cos_gen[:, None] + torch.cat([-second, first], dim=-1) * sin_gen[:, None]
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                for rows in (slice(0, reference_tokens), slice(reference_tokens, None)):
                    assert not torch.equal(expected[rows], unrotated[rows]), "RoPE must be active on both streams"
                    assert (
                        (scoped[rows] - expected[rows]).abs().sum(dim=(-2, -1)) > 0
                    ).all(), "every reference AND target query/key must receive its live SCoPE delta AFTER RoPE"


def test_real_joint_reference_image_camera_permutation_is_equivariant_with_live_scope():
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    kwargs = _joint_reference_kwargs()
    permutation = [2, 0, 1]
    permuted_inputs = _joint_reference_kwargs(reference_order=tuple(permutation))
    for name, expected in _select_cosmos3_references(kwargs, permutation).items():
        torch.testing.assert_close(permuted_inputs[name], expected, rtol=0, atol=0, msg=name)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        with _capture_cosmos3_forward(transformer) as original:
            baseline = transformer(**kwargs)[0]
        with _capture_cosmos3_forward(transformer) as reordered:
            actual = transformer(**permuted_inputs)[0]
        for name in ("neighbor_hidden_states", "neighbor_camera_rays"):
            mismatched = transformer(**{**kwargs, name: permuted_inputs[name]})[0]
            assert not torch.allclose(mismatched, baseline, rtol=1e-5, atol=1e-6), name
    torch.testing.assert_close(actual, baseline, rtol=1e-5, atol=1e-6)
    reference_tokens = 3 * SPATIAL_TOKENS_PER_FRAME
    for before, after in zip(original.gen_outputs, reordered.gen_outputs, strict=True):
        expected_refs = before[0][:reference_tokens].reshape(3, SPATIAL_TOKENS_PER_FRAME, -1)[permutation].flatten(0, 1)
        torch.testing.assert_close(after[0][:reference_tokens], expected_refs, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(after[0][reference_tokens:], before[0][reference_tokens:], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("camera_change", ["lens", "focal_length", "pose"])
def test_real_joint_reference_camera_sensitivity_requires_regenerated_raymaps(camera_change):
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    kwargs = _joint_reference_kwargs()
    alternative = _joint_reference_kwargs(
        reference_distortion_delta=0.2 if camera_change == "lens" else 0.0,
        reference_focal_delta=0.3 if camera_change == "focal_length" else 0.0,
        reference_translation=0.7 if camera_change == "pose" else 0.0,
    )
    for name in kwargs.keys() - {"neighbor_camera_rays", "neighbor_Ks", "neighbor_w2cs"}:
        torch.testing.assert_close(alternative[name], kwargs[name], rtol=0, atol=0, msg=name)
    assert not torch.equal(alternative["neighbor_camera_rays"], kwargs["neighbor_camera_rays"])
    if camera_change == "lens":
        torch.testing.assert_close(alternative["neighbor_Ks"], kwargs["neighbor_Ks"], rtol=0, atol=0)
    else:
        matrix = "neighbor_Ks" if camera_change == "focal_length" else "neighbor_w2cs"
        assert not torch.equal(alternative[matrix], kwargs[matrix])
    ray_only = {**kwargs, "neighbor_camera_rays": alternative["neighbor_camera_rays"]}
    matrices_only = {**kwargs, **{name: alternative[name] for name in ("neighbor_w2cs", "neighbor_Ks")}}
    with torch.no_grad():
        baseline = transformer(**kwargs)[0]
        moved_camera = transformer(**alternative)[0]
        moved_rays = transformer(**ray_only)[0]
        stale_rays = transformer(**matrices_only)[0]
    assert not torch.allclose(baseline, moved_camera, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(moved_rays, moved_camera, rtol=0, atol=0)
    torch.testing.assert_close(stale_rays, baseline, rtol=0, atol=0)
    with torch.no_grad():
        for scope in transformer.scope:
            scope.alpha.zero_()
        baseline_without_scope = transformer(**kwargs)[0]
        changed_without_scope = transformer(**alternative)[0]
    torch.testing.assert_close(baseline_without_scope, changed_without_scope, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad_payload", ["omitted", "missing", "latent_resolution", "reference_count", "unbatched", "batch", "channels"]
)
def test_real_joint_active_references_require_full_pixel_neighbor_rays(bad_payload):
    transformer = _reduced_real_transformer()
    kwargs = _joint_reference_kwargs(total_frames=1, reference_frames=2)
    rays = kwargs["neighbor_camera_rays"]
    if bad_payload == "omitted":
        del kwargs["neighbor_camera_rays"]
    else:
        kwargs["neighbor_camera_rays"] = {
            "missing": None,
            "latent_resolution": rays[:, :, ::VAE_SCALE_FACTOR_SPATIAL, ::VAE_SCALE_FACTOR_SPATIAL],
            "reference_count": rays[:, :1],
            "unbatched": rays[0],
            "batch": rays.repeat(2, 1, 1, 1, 1),
            "channels": rays[..., :5],
        }[bad_payload]
    with pytest.raises((ValueError, AssertionError), match="neighbor_camera_rays|camera_rays"):
        transformer(**kwargs)


@pytest.mark.parametrize("causal_block_frames", [None, 1])
@pytest.mark.parametrize("perturbation", ["noise", "timestep", "opacity", "camera_rays", "future_target"])
def test_real_joint_reference_features_never_depend_on_target_stream(causal_block_frames, perturbation):
    transformer = _reduced_real_transformer(causal_block_frames=causal_block_frames)
    _activate_conditioning_adapters(transformer)
    kwargs = _joint_reference_kwargs()
    changed = dict(kwargs)
    if perturbation == "timestep":
        changed["timestep"] = torch.tensor([1.0])
    elif perturbation == "opacity":
        changed["opacity"] = 1 - kwargs["opacity"]
    elif perturbation == "camera_rays":
        changed["camera_rays"] = kwargs["camera_rays"].flip(1)
    else:
        changed["hidden_states"] = kwargs["hidden_states"].clone()
        frames = slice(-1, None) if perturbation == "future_target" else slice(None)
        changed["hidden_states"][:, :, frames] += torch.linspace(-1, 1, 48).view(1, 48, 1, 1, 1)
        if perturbation == "future_target":
            changed["opacity"] = kwargs["opacity"].clone()
            changed["opacity"][:, -VAE_SCALE_FACTOR_TEMPORAL:] *= 0.1
            changed["camera_rays"] = kwargs["camera_rays"].clone()
            changed["camera_rays"][:, -1] = kwargs["camera_rays"][:, 0]
            changed["timestep"] = torch.full((1, GROWING_PREFIX_FRAMES * SPATIAL_TOKENS_PER_FRAME), 999.0)
            changed["timestep"][:, -SPATIAL_TOKENS_PER_FRAME:] = 1.0
    with torch.no_grad():
        with _capture_cosmos3_forward(transformer) as original:
            baseline = transformer(**kwargs)[0]
        with _capture_cosmos3_forward(transformer) as perturbed:
            actual = transformer(**changed)[0]
    reference_tokens = 3 * SPATIAL_TOKENS_PER_FRAME
    for before, after in zip(original.gen_inputs, perturbed.gen_inputs, strict=True):
        torch.testing.assert_close(before[0][:reference_tokens], after[0][:reference_tokens], rtol=0, atol=0)
    for before, after in zip(original.gen_outputs, perturbed.gen_outputs, strict=True):
        torch.testing.assert_close(before[0][:reference_tokens], after[0][:reference_tokens], rtol=0, atol=0)
    for before, after in zip(original.und_outputs, perturbed.und_outputs, strict=True):
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    assert not torch.equal(baseline, actual), "the target perturbation must be observable"
    if perturbation == "future_target":
        if causal_block_frames is not None:
            torch.testing.assert_close(baseline[:, :, :-1], actual[:, :, :-1], rtol=0, atol=0)
        else:
            assert not torch.equal(baseline[:, :, :-1], actual[:, :, :-1])


@pytest.mark.parametrize("source", ["text", "another_reference"])
def test_real_joint_reference_queries_can_see_text_and_other_references(source):
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    kwargs = _joint_reference_kwargs()
    changed = dict(kwargs)
    if source == "text":
        changed["encoder_hidden_states"] = kwargs["encoder_hidden_states"].clone()
        changed["encoder_hidden_states"][0, 0] = 9
    else:
        changed["neighbor_hidden_states"] = kwargs["neighbor_hidden_states"].clone()
        changed["neighbor_hidden_states"][:, :, 0] += torch.linspace(-1, 1, 48).view(1, 48, 1, 1)
    with torch.no_grad():
        with _capture_cosmos3_forward(transformer) as original:
            baseline = transformer(**kwargs)[0]
        with _capture_cosmos3_forward(transformer) as perturbed:
            actual = transformer(**changed)[0]
    last_reference = slice(2 * SPATIAL_TOKENS_PER_FRAME, 3 * SPATIAL_TOKENS_PER_FRAME)
    for before, after in zip(original.gen_outputs, perturbed.gen_outputs, strict=True):
        assert not torch.equal(before[0][last_reference], after[0][last_reference])
    if source == "another_reference":
        for before, after in zip(original.und_outputs, perturbed.und_outputs, strict=True):
            torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    assert not torch.equal(baseline, actual)


@pytest.mark.parametrize("loss_frame", [0, GROWING_PREFIX_FRAMES - 1])
@pytest.mark.parametrize("gradient_checkpointing", [False, True])
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        pytest.param(
            "cuda", torch.bfloat16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
        ),
    ],
)
def test_real_joint_block_causal_gradients_reach_all_refs_and_only_allowed_target_frames(
    loss_frame, gradient_checkpointing, device, dtype
):
    transformer = _reduced_real_transformer(causal_block_frames=1, gradient_checkpointing=gradient_checkpointing)
    _activate_conditioning_adapters(transformer)
    transformer.to(device)
    kwargs = {
        name: value.to(device=device, dtype=dtype if value.is_floating_point() and name != "timestep" else value.dtype)
        for name, value in _joint_reference_kwargs().items()
    }
    kwargs["hidden_states"].requires_grad_(True)
    kwargs["neighbor_hidden_states"].requires_grad_(True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        prediction = transformer(**kwargs)[0]
    prediction[:, :, loss_frame].float().square().sum().backward()
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in transformer.parameters() if parameter.grad is not None
    )
    target_grad = kwargs["hidden_states"].grad
    reference_grad = kwargs["neighbor_hidden_states"].grad
    assert target_grad is not None and torch.isfinite(target_grad).all()
    assert reference_grad is not None and torch.isfinite(reference_grad).all()
    for frame in range(GROWING_PREFIX_FRAMES):
        if frame <= loss_frame:
            assert torch.count_nonzero(target_grad[:, :, frame]) > 0, f"allowed history frame {frame} was detached"
        else:
            assert torch.count_nonzero(target_grad[:, :, frame]) == 0, f"future target frame {frame} leaked"
    for reference in range(kwargs["neighbor_hidden_states"].shape[2]):
        assert torch.count_nonzero(reference_grad[:, :, reference]) > 0, f"reference {reference} was detached"


@pytest.mark.parametrize("teacher_forcing", [False, True])
def test_real_joint_camera_conditioning_does_not_require_prope_matrices(teacher_forcing):
    transformer = _reduced_real_transformer(causal_block_frames=1 if teacher_forcing else None)
    _activate_conditioning_adapters(transformer)
    kwargs = _joint_reference_kwargs(4)
    if teacher_forcing:
        kwargs["clean_x"] = kwargs["hidden_states"].clone()
    forward = (
        (lambda **inputs: cosmos3_teacher_forcing_reference(transformer, **inputs)) if teacher_forcing else transformer
    )
    with torch.no_grad():
        expected = forward(**kwargs)[0]
        kwargs.update(w2cs=None, Ks=None, neighbor_w2cs=None, neighbor_Ks=None)
        actual = forward(**kwargs)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("pass_kind", ["denoise", "teacher_forcing", "seed"])
def test_real_joint_refs_receive_no_target_timestep_opacity_or_prediction_rows(pass_kind):
    transformer = _reduced_real_transformer(causal_block_frames=1 if pass_kind == "teacher_forcing" else None)
    _activate_conditioning_adapters(transformer)  # Includes nonzero opacity BIAS.
    kwargs = _joint_reference_kwargs(total_frames=2)
    target_shape = kwargs["hidden_states"].shape
    target_tokens = 2 * SPATIAL_TOKENS_PER_FRAME
    kwargs["timestep"] = torch.tensor([[900.0, 20.0]]).repeat_interleave(SPATIAL_TOKENS_PER_FRAME, dim=1)
    expected_timesteps = kwargs["timestep"][0]
    if pass_kind == "teacher_forcing":
        kwargs["clean_x"] = kwargs["hidden_states"].flip(2)
    elif pass_kind == "seed":
        kwargs.update(
            timestep=None,
            kv_cache=transformer.new_rollout_kv_cache(truncate_rollout_gradients=False),
            kv_cache_seed=True,
        )
    with torch.no_grad(), _capture_cosmos3_forward(transformer) as trace:
        output = (
            cosmos3_teacher_forcing_reference(transformer, **kwargs)
            if pass_kind == "teacher_forcing"
            else transformer(**kwargs)
        )[0]
    call = trace.base_kwargs[0]
    reference_tokens = 3 * SPATIAL_TOKENS_PER_FRAME
    target_gen_tokens = target_tokens * (2 if pass_kind == "teacher_forcing" else 1)
    noisy_tokens = 0 if pass_kind == "seed" else target_tokens
    assert output.shape == target_shape
    assert call["sequence_length"] == call["und_len"] + reference_tokens + target_gen_tokens
    assert call["vision_timesteps"].shape == (noisy_tokens,)
    expected_loss_rows = (
        call["vision_sequence_indexes"][-noisy_tokens:] if noisy_tokens else torch.empty(0, dtype=torch.long)
    )
    torch.testing.assert_close(call["vision_mse_loss_indexes"], expected_loss_rows, rtol=0, atol=0)
    torch.testing.assert_close(_cosmos3_noisy_sequence_indexes(call), expected_loss_rows, rtol=0, atol=0)
    if noisy_tokens:
        torch.testing.assert_close(call["vision_timesteps"], expected_timesteps, rtol=0, atol=0)
    else:
        assert torch.count_nonzero(output) == 0
    for prediction_inputs in trace.prediction_inputs:
        assert prediction_inputs.shape[-2] == noisy_tokens
    for layer_index in range(len(transformer.base_transformer.layers)):
        before = trace.attention_before_opacity[layer_index][0]
        after = trace.attention_after_opacity[layer_index][0]
        assert before.shape[0] == reference_tokens + target_gen_tokens
        assert trace.opacity_tokens[layer_index][0].shape[0] == target_gen_tokens
        torch.testing.assert_close(before[:reference_tokens], after[:reference_tokens], rtol=0, atol=0)
        assert not torch.equal(before[reference_tokens:], after[reference_tokens:])


def test_real_joint_ignore_neighbors_empty_and_absent_references_are_equivalent():
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    kwargs = _joint_reference_kwargs()
    absent = {
        **kwargs,
        "neighbor_hidden_states": None,
        "neighbor_camera_rays": None,
        "neighbor_w2cs": None,
        "neighbor_Ks": None,
    }
    ignored = {
        **kwargs,
        "ignore_neighbors": True,
        "neighbor_camera_rays": None,
        "neighbor_w2cs": None,
        "neighbor_Ks": None,
    }
    with torch.no_grad():
        baseline = transformer(**absent)[0]
        empty = transformer(**_select_cosmos3_references(kwargs, []))[0]
        dropped = transformer(**ignored)[0]
        actual = transformer(**kwargs)[0]
        dict_output = transformer(**kwargs, return_dict=True)
    torch.testing.assert_close(empty, baseline, rtol=0, atol=0)
    torch.testing.assert_close(dropped, baseline, rtol=0, atol=0)
    assert not torch.equal(actual, baseline), "active references must affect the prediction"
    torch.testing.assert_close(dict_output.sample, actual, rtol=0, atol=0)


def test_real_joint_batched_forward_keeps_per_sample_reference_rays_and_prompt():
    transformer = _reduced_real_transformer()
    _activate_conditioning_adapters(transformer)
    first = _joint_reference_kwargs(total_frames=2)
    second = _joint_reference_kwargs(total_frames=2, reference_translation=0.7)
    second["hidden_states"] = second["hidden_states"] * -0.3 + 0.9
    second["neighbor_hidden_states"] = second["neighbor_hidden_states"].flip(2) * 0.6
    second["opacity"] = 1 - second["opacity"]
    second["timestep"] = torch.tensor([35.0])
    second["encoder_hidden_states"] = torch.tensor([[9, 10, 11, 0]])
    second["encoder_attention_mask"] = torch.tensor([[True, True, True, False]])
    batch = {name: torch.cat([value, second[name]], dim=0) for name, value in first.items()}
    with torch.no_grad():
        expected = torch.cat([transformer(**first)[0], transformer(**second)[0]], dim=0)
        with _capture_cosmos3_forward(transformer) as trace:
            actual = transformer(**batch)[0]
    assert actual.shape == batch["hidden_states"].shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [call["und_len"] for call in trace.base_kwargs] == [4, 3]
    for layer_rays in trace.scope_rays:
        for sample_index, inputs in enumerate((first, second)):
            expected_rays = torch.cat(
                [_expected_scope_rays(inputs["neighbor_camera_rays"]), _expected_scope_rays(inputs["camera_rays"])],
                dim=0,
            )
            torch.testing.assert_close(layer_rays[sample_index], expected_rays, rtol=1e-5, atol=1e-6)


def test_real_references_share_generator_parameters():
    transformer = _reduced_real_transformer()
    assert set(transformer._modules) == {"base_transformer", "opacity_embeddings", "scope"}
    assert not hasattr(transformer, "reference_attention_mode")
    assert not any("reference_attentions" in name for name in transformer.state_dict())


@pytest.mark.parametrize("causal_block_frames", [None, 1])
def test_real_joint_backward_and_gradient_checkpointing_are_identical(causal_block_frames):
    def run(checkpointing):
        transformer = _reduced_real_transformer(
            gradient_checkpointing=checkpointing,
            causal_block_frames=causal_block_frames,
        )
        _activate_conditioning_adapters(transformer)
        kwargs = _joint_reference_kwargs()
        kwargs["hidden_states"].requires_grad_(True)
        kwargs["neighbor_hidden_states"].requires_grad_(True)
        output = transformer(**kwargs)[0]
        output.square().mean().backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad
        }
        for name in ("hidden_states", "neighbor_hidden_states"):
            grad = kwargs[name].grad
            assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0, name
            gradients[name] = grad.detach().clone()
        for layer_index in range(len(transformer.base_transformer.layers)):
            for name in (
                f"opacity_embeddings.{layer_index}.weight",
                f"scope.{layer_index}.query_projection.weight",
                f"scope.{layer_index}.key_projection.weight",
                f"base_transformer.layers.{layer_index}.self_attn.add_k_proj.weight",
            ):
                grad = gradients[name]
                assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0, name
        assert transformer.base_transformer.embed_tokens.weight.grad is None
        return output.detach(), gradients

    plain_output, plain_grads = run(False)
    checkpointed_output, checkpointed_grads = run(True)
    torch.testing.assert_close(plain_output, checkpointed_output, rtol=0, atol=0)
    assert plain_grads.keys() == checkpointed_grads.keys()
    for name, grad in plain_grads.items():
        if grad is None:
            assert checkpointed_grads[name] is None, name
        else:
            assert torch.isfinite(grad).all(), name
            torch.testing.assert_close(grad, checkpointed_grads[name], rtol=0, atol=0, msg=name)


def _run_real_joint_cp_parity_worker(rank: int, world_size: int, init_file: str, mode: str, backend: str) -> None:
    from model_training.utils.precision import disable_tf32_matmul

    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
        disable_tf32_matmul()
    dist.init_process_group(
        backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    try:
        total_frames = 3 if mode in ("padded", "block_causal_padded") else 4
        causal_block_frames = 1 if mode in ("block_causal", "block_causal_padded") else None
        inputs = _joint_reference_kwargs(total_frames=total_frames)
        inputs["timestep"] = torch.linspace(900, 20, total_frames).repeat_interleave(SPATIAL_TOKENS_PER_FRAME)[None]
        inputs.update(w2cs=None, Ks=None, neighbor_w2cs=None, neighbor_Ks=None)
        inputs = {name: value.to(device) if torch.is_tensor(value) else value for name, value in inputs.items()}
        reference = _reduced_real_transformer(
            causal_block_frames=causal_block_frames,
            gradient_checkpointing=mode == "checkpointed",
        )
        _activate_conditioning_adapters(reference)
        reference.to(device)
        reference_inputs = dict(inputs)
        for name in ("hidden_states", "neighbor_hidden_states"):
            reference_inputs[name] = inputs[name].detach().clone().requires_grad_(True)
        with sdpa_kernel([SDPBackend.MATH]):
            with _capture_cosmos3_forward(reference) as reference_trace:
                reference_output = reference(**reference_inputs)[0]
            reference_loss = reference_output.square().mean() if mode == "local" else reference_output.square().sum()
            reference_loss.backward()
        reference_grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
            if parameter.requires_grad
        }

        cp_transformer = _reduced_real_transformer(
            causal_block_frames=causal_block_frames,
            gradient_checkpointing=mode == "checkpointed",
        )
        cp_transformer.to(device)
        cp_transformer.load_state_dict(reference.state_dict(), strict=True)
        # Preserve the requested transport even when CUDA is visible: some NGC
        # Torch builds replace a CPU mesh's Gloo group with a mixed backend.
        mesh = DeviceMesh.from_group(dist.group.WORLD, device_type=device.type)
        assert dist.get_backend(mesh.get_group()) == backend
        cp_transformer.enable_context_parallel(mesh)
        local_frames = (total_frames + world_size - 1) // world_size
        start, end = rank * local_frames, min((rank + 1) * local_frames, total_frames)
        cp_inputs = dict(inputs)
        if mode == "local":
            cp_inputs = _slice_cosmos3_target_kwargs(inputs, start, end)
            cp_inputs.update(frame_offset=start, local_cp_input=True)
        for name in ("hidden_states", "neighbor_hidden_states"):
            cp_inputs[name] = cp_inputs[name].detach().clone().requires_grad_(True)
        with sdpa_kernel([SDPBackend.MATH]):
            with _capture_cosmos3_forward(cp_transformer) as cp_trace:
                cp_output = cp_transformer(**cp_inputs)[0]
            expected_output = reference_output[:, :, start:end] if mode == "local" else reference_output
            assert cp_output.shape == expected_output.shape
            torch.testing.assert_close(cp_output, expected_output, rtol=1e-4, atol=1e-5)
            cp_loss = cp_output.square().mean() if mode == "local" else cp_output.square().sum()
            cp_loss.backward()

        reference_tokens = 3 * SPATIAL_TOKENS_PER_FRAME
        expected_positions = _vision_positions(reference_trace.base_kwargs[0])
        cp_positions = _vision_positions(cp_trace.base_kwargs[0])
        assert cp_positions.shape[1] == reference_tokens + local_frames * SPATIAL_TOKENS_PER_FRAME
        torch.testing.assert_close(
            cp_positions[:, :reference_tokens], expected_positions[:, :reference_tokens], rtol=0, atol=0
        )
        valid_local_tokens = (end - start) * SPATIAL_TOKENS_PER_FRAME
        expected_start = reference_tokens + start * SPATIAL_TOKENS_PER_FRAME
        expected_end = reference_tokens + end * SPATIAL_TOKENS_PER_FRAME
        torch.testing.assert_close(
            cp_positions[:, reference_tokens : reference_tokens + valid_local_tokens],
            expected_positions[:, expected_start:expected_end],
            rtol=0,
            atol=0,
        )
        for layer_index in range(len(reference.base_transformer.layers)):
            torch.testing.assert_close(
                cp_trace.gen_outputs[layer_index][0][:reference_tokens],
                reference_trace.gen_outputs[layer_index][0][:reference_tokens],
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                cp_trace.scope_rays[layer_index][0][:reference_tokens],
                reference_trace.scope_rays[layer_index][0][:reference_tokens],
                rtol=0,
                atol=0,
            )

        for name, parameter in cp_transformer.named_parameters():
            if parameter.requires_grad:
                _assert_cosmos3_cp_gradient_mean(
                    parameter.grad, reference_grads[name], world_size=world_size, name=f"{mode}/{name}"
                )
        for name in ("hidden_states", "neighbor_hidden_states"):
            expected_grad = reference_inputs[name].grad
            assert expected_grad is not None and torch.count_nonzero(expected_grad) > 0, name
            actual_grad = cp_inputs[name].grad
            if name == "hidden_states" and mode == "local" and actual_grad is not None:
                global_grad = torch.zeros_like(reference_inputs[name])
                global_grad[:, :, start:end] = actual_grad
                actual_grad = global_grad
            _assert_cosmos3_cp_gradient_mean(actual_grad, expected_grad, world_size=world_size, name=f"{mode}/{name}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("mode", ["full", "block_causal", "padded", "block_causal_padded", "local", "checkpointed"])
def test_real_joint_cp1_cp2_forward_backward_and_positions_match(tmp_path, mode, backend):
    """Real collectives, real GQA, nonzero SCoPE, and replicated clean references."""
    _run_cp_workers(_run_real_joint_cp_parity_worker, tmp_path, mode, backend=backend)
