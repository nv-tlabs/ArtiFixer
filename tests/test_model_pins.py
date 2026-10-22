# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-repo pin registry is
the single source of truth for pretrained-artifact provenance. These tests pin
the registry values, the fingerprint/cache-key derivation, and that the
loaders actually consume the registry (pass ``revision=``) instead of
resolving the hub default."""

from pathlib import Path
from types import SimpleNamespace

import diffusers
import pytest
import torch

from model_training.model_profiles import COSMOS3_NANO_MODEL_ID
from model_training.pins import (
    COSMOS3_NANO_HF_REVISION,
    COSMOS3_NANO_PINS,
    DIFFUSERS_COMMIT,
    DIFFUSERS_VERSION,
    MODEL_PIN_REGISTRY,
    SAFETENSORS_VERSION,
    compute_config_fingerprint,
    dcp_cache_dir_name,
    hf_revision_for_model,
    pins_for_model,
    tokenizer_revision_for_model,
)

# Canonical tiny Cosmos3 doubles shared across the cosmos3 tests.
from tests._helpers import _TinyCosmosBaseTransformer, _TinyCosmosVae

REPO_ROOT = Path(__file__).resolve().parents[1]

WAN_14B_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


def _assert_full_git_sha(value: str) -> None:
    assert len(value) == 40
    int(value, 16)  # raises if not hex


class TestRegistryRecord:
    def test_cosmos3_nano_record_carries_the_audited_provenance(self):
        """The registry values are the audited production pins:
        hub revision 23314034 (the action-capable Cosmos3-Nano snapshot both
        arms train from) and diffusers v0.39.0 = a3608b51."""
        assert COSMOS3_NANO_PINS.model_id == "nvidia/Cosmos3-Nano"
        assert COSMOS3_NANO_PINS.hf_revision == "23314034b82f46b45339035dba67c3ee9bbcb8ba"
        assert COSMOS3_NANO_PINS.diffusers_commit == "a3608b512ed7248499a44c61d954965ed9bdae4d"
        assert COSMOS3_NANO_HF_REVISION == COSMOS3_NANO_PINS.hf_revision

    def test_registry_revisions_are_full_git_shas(self):
        """Branch names ('main') would reintroduce the mutable-pointer failure
        class (a hub branch can move under a launched run); only full commit shas
        are immutable pins."""
        for pins in MODEL_PIN_REGISTRY:
            _assert_full_git_sha(pins.hf_revision)
            _assert_full_git_sha(pins.diffusers_commit)

    def test_record_fingerprint_matches_the_derivation(self):
        for pins in MODEL_PIN_REGISTRY:
            assert pins.config_fingerprint == compute_config_fingerprint(
                pins.model_id, pins.hf_revision, pins.diffusers_commit
            )


class TestRegistryLookup:
    def test_hub_id_and_local_path_forms_resolve_to_the_same_record(self):
        """model_profiles family predicates drive the lookup, so a local
        snapshot directory ending in Cosmos3-Nano gets the same pins as the
        hub id (launchers pass local paths under HF_HUB_OFFLINE)."""
        assert pins_for_model(COSMOS3_NANO_MODEL_ID) is COSMOS3_NANO_PINS
        assert pins_for_model("/data/snapshots/nvidia/Cosmos3-Nano") is COSMOS3_NANO_PINS

    def test_unpinned_families_resolve_to_none(self):
        assert pins_for_model(WAN_14B_MODEL_ID) is None
        # Cosmos3-Super has no audited revision; it must NOT inherit the Nano
        # record (a wrong pin is worse than no pin).
        # Unpinned families (e.g. the public Wan releases) have no record.
        assert pins_for_model("Wan-AI/Wan2.1-T2V-14B-Diffusers") is None

    def test_revision_accessors_return_pin_or_none(self):
        assert hf_revision_for_model(COSMOS3_NANO_MODEL_ID) == COSMOS3_NANO_PINS.hf_revision
        # The Cosmos3 text tokenizer ships in the same hub snapshot
        # (subfolder text_tokenizer), so tokenizer loads pin the same revision.
        assert tokenizer_revision_for_model(COSMOS3_NANO_MODEL_ID) == COSMOS3_NANO_PINS.hf_revision
        assert hf_revision_for_model(WAN_14B_MODEL_ID) is None
        assert tokenizer_revision_for_model(WAN_14B_MODEL_ID) is None


class TestFingerprintDerivation:
    def test_fingerprint_is_deterministic_hex(self):
        first = compute_config_fingerprint("m", "r", "d")
        assert first == compute_config_fingerprint("m", "r", "d")
        assert len(first) == 16
        int(first, 16)

    @pytest.mark.parametrize(
        "other",
        [
            ("m2", "r", "d"),  # model id bump
            ("m", "r2", "d"),  # hub revision bump
            ("m", "d", "r"),  # field transposition must not collide
            ("m", None, "d"),  # unpinned marker differs from any real revision
            ("m", "r", "d2"),  # diffusers (state-dict layout owner) bump
        ],
    )
    def test_fingerprint_is_sensitive_to_every_component(self, other):
        assert compute_config_fingerprint(*other) != compute_config_fingerprint("m", "r", "d")


class TestDcpCacheKeyDerivation:
    def test_cache_name_is_flattened_model_id_plus_pin_fingerprint(self):
        name = dcp_cache_dir_name(COSMOS3_NANO_MODEL_ID)
        assert name == f"nvidia--Cosmos3-Nano--{COSMOS3_NANO_PINS.config_fingerprint}"
        assert "/" not in name

    def test_cosmos3_nano_cache_key_literal(self):
        """Literal pin of the derived cache key: any change to the registry
        values or the fingerprint formula renames the production DCP cache
        (a deliberate, reviewable event — it forces a ~32 GB reconversion on
        the cluster)."""
        assert COSMOS3_NANO_PINS.config_fingerprint == "29fedffc7f06b617"
        assert dcp_cache_dir_name(COSMOS3_NANO_MODEL_ID) == "nvidia--Cosmos3-Nano--29fedffc7f06b617"

    def test_revision_bump_renames_the_cache(self):
        """The core cache-naming property: a different pinned revision may not
        resolve to the old conversion."""
        bumped = compute_config_fingerprint(COSMOS3_NANO_PINS.model_id, "0" * 40, COSMOS3_NANO_PINS.diffusers_commit)
        assert bumped != COSMOS3_NANO_PINS.config_fingerprint

    def test_unpinned_models_still_get_a_fingerprinted_cache(self):
        """Wan caches are keyed against the diffusers commit too: the model
        implementation defines the cached state-dict layout even when the hub
        revision floats."""
        name = dcp_cache_dir_name(WAN_14B_MODEL_ID)
        expected_fingerprint = compute_config_fingerprint(WAN_14B_MODEL_ID, None, DIFFUSERS_COMMIT)
        assert name == f"Wan-AI--Wan2.1-T2V-14B-Diffusers--{expected_fingerprint}"
        assert expected_fingerprint != COSMOS3_NANO_PINS.config_fingerprint


class TestDependencyPinCoherence:
    def test_dockerfiles_install_the_registry_diffusers_version(self):
        """The container images must install exactly the registry's diffusers
        version — pin drift across files becomes a test failure instead of a
        silent mismatch. The images install the PyPI release; the release
        tag's commit (DIFFUSERS_COMMIT) authenticates the source checkout the
        launch gates inject, and the requirements comment must carry that
        commit beside the version pin so the two stay tied. All Dockerfiles
        share docker/requirements.txt, so the pin lives there once and every
        Dockerfile must install that file."""
        requirements = (REPO_ROOT / "docker" / "requirements.txt").read_text()
        assert (
            f"diffusers=={DIFFUSERS_VERSION}" in requirements
        ), f"docker/requirements.txt does not pin diffusers=={DIFFUSERS_VERSION}"
        assert DIFFUSERS_COMMIT in requirements, (
            "docker/requirements.txt must name the release commit "
            f"{DIFFUSERS_COMMIT} beside the version pin (the launch gates pin "
            "the injected source checkout by that commit)"
        )

    def test_dockerfiles_install_the_registry_safetensors_version(self):
        """Same single-source rule for safetensors: the preflight asserts the
        in-container install against pins.SAFETENSORS_VERSION, so the image
        build must install exactly that version."""
        requirements = (REPO_ROOT / "docker" / "requirements.txt").read_text()
        assert (
            f"safetensors=={SAFETENSORS_VERSION}" in requirements
        ), f"docker/requirements.txt does not pin safetensors=={SAFETENSORS_VERSION}"
        dockerfiles = sorted(REPO_ROOT.glob("Dockerfile*"))
        assert dockerfiles, "expected Dockerfiles at the repo root"
        for dockerfile in dockerfiles:
            assert (
                "docker/requirements.txt" in dockerfile.read_text()
            ), f"{dockerfile.name} does not install docker/requirements.txt"

    def test_installed_diffusers_matches_registry_version(self):
        assert diffusers.__version__ == DIFFUSERS_VERSION


class _TinyPretrainedTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def _fake_pretrained_transformer_cls(calls: list, missing_keys: tuple[str, ...] = ()):
    """A ``from_pretrained`` double with diffusers' return contract: the bare model, or
    ``(model, loading_info)`` when ``output_loading_info`` is requested."""

    class _FakeTransformerCls:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            model = _TinyPretrainedTransformer()
            if kwargs.get("output_loading_info"):
                return model, {
                    "missing_keys": list(missing_keys),
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
            return model

    return _FakeTransformerCls


class TestLoaderRegistryConsumption:
    """Loaders must consume the registry — pass its revision= and derive the
    DCP cache directory from its fingerprint."""

    def test_scheduler_loader_reads_the_registry_revision(self, monkeypatch):
        from model_training.utils import train_utils

        calls = []
        monkeypatch.setattr(
            train_utils.UniPCMultistepScheduler,
            "from_pretrained",
            lambda *args, **kwargs: calls.append((args, kwargs)) or "scheduler",
        )
        # Sentinel registry: proves the revision is plumbed from the pins
        # accessor, not hardcoded (Wan's real pin is None today).
        monkeypatch.setattr(train_utils, "hf_revision_for_model", lambda model_id: f"pin::{model_id}")

        train_utils._scheduler_for_model(WAN_14B_MODEL_ID)

        assert calls[0][1]["revision"] == f"pin::{WAN_14B_MODEL_ID}"

    def test_dcp_cache_conversion_pins_revision_and_fingerprints_directory(self, monkeypatch, tmp_path):
        """Both halves at once: the HF->DCP conversion downloads
        the pinned revision, and the cache lands in the fingerprinted
        directory so a pin bump cannot resolve to this conversion."""
        from model_training.utils import train_utils

        pretrained_calls = []
        _FakeTransformerCls = _fake_pretrained_transformer_cls(pretrained_calls)

        monkeypatch.setattr(train_utils, "_transformer_cls_for_model", lambda model_id: _FakeTransformerCls)
        monkeypatch.setattr(train_utils.torch.cuda, "empty_cache", lambda: None)

        cache_path = train_utils._ensure_dcp_cache(COSMOS3_NANO_MODEL_ID, tmp_path)

        assert pretrained_calls[0][0] == (COSMOS3_NANO_MODEL_ID,)
        assert pretrained_calls[0][1]["revision"] == COSMOS3_NANO_PINS.hf_revision
        assert pretrained_calls[0][1]["output_loading_info"] is True
        assert cache_path == tmp_path / dcp_cache_dir_name(COSMOS3_NANO_MODEL_ID) / "transformer"
        assert cache_path.parent.name.endswith(COSMOS3_NANO_PINS.config_fingerprint)
        assert (cache_path / ".dcp_complete").is_file()

    def test_dcp_cache_conversion_refuses_a_checkpoint_missing_declared_keys(self, monkeypatch, tmp_path):
        """HF random-fills a key the checkpoint lacks on rank 0 and the cache would freeze
        that draw for the cluster, so a pin bump to such a checkpoint must fail loudly
        rather than leave a marker behind."""
        from model_training.utils import train_utils

        _FakeTransformerCls = _fake_pretrained_transformer_cls([], missing_keys=("blocks.0.attn2.add_k_proj.weight",))
        monkeypatch.setattr(train_utils, "_transformer_cls_for_model", lambda model_id: _FakeTransformerCls)
        monkeypatch.setattr(train_utils.torch.cuda, "empty_cache", lambda: None)

        with pytest.raises(RuntimeError, match="does not ship 1 transformer key.*add_k_proj"):
            train_utils._ensure_dcp_cache(COSMOS3_NANO_MODEL_ID, tmp_path)

        assert not list(tmp_path.rglob(".dcp_complete"))

    def test_get_pipe_resolves_config_and_vae_at_the_pinned_revision(self, monkeypatch):
        from model_training.utils import train_utils

        from_config_calls = []
        vae_calls = []

        class _FakeTransformerCls:
            @staticmethod
            def from_config(*args, **kwargs):
                from_config_calls.append((args, kwargs))
                return _TinyCosmosBaseTransformer()

        monkeypatch.setattr(train_utils, "_transformer_cls_for_model", lambda model_id: _FakeTransformerCls)
        monkeypatch.setattr(
            train_utils.AutoencoderKLWan,
            "from_pretrained",
            lambda *args, **kwargs: vae_calls.append((args, kwargs)) or _TinyCosmosVae(),
        )
        args = SimpleNamespace(
            model_id=COSMOS3_NANO_MODEL_ID,
            gradient_checkpointing=False,
            attention_backend=None,
        )

        train_utils.get_pipe(args, load_pretrained_transformer_weights=False, frames_per_block=None, device="cpu")

        assert from_config_calls[0][1]["revision"] == COSMOS3_NANO_PINS.hf_revision
        assert vae_calls[0][1]["revision"] == COSMOS3_NANO_PINS.hf_revision

    def test_get_kv_cache_pipe_resolves_config_and_vae_at_the_pinned_revision(self, monkeypatch):
        from model_training.utils import train_utils

        from_config_calls = []
        vae_calls = []

        class _FakeTransformerCls:
            @staticmethod
            def from_config(*args, **kwargs):
                from_config_calls.append((args, kwargs))
                return _TinyCosmosBaseTransformer()

        monkeypatch.setattr(train_utils, "Cosmos3OmniTransformer", _FakeTransformerCls)
        monkeypatch.setattr(
            train_utils.AutoencoderKLWan,
            "from_pretrained",
            lambda *args, **kwargs: vae_calls.append((args, kwargs)) or _TinyCosmosVae(),
        )
        args = SimpleNamespace(
            model_id=COSMOS3_NANO_MODEL_ID,
            gradient_checkpointing=False,
            attention_backend=None,
            frames_per_block=7,
            local_attn_size=21,
            sink_size=7,
        )

        train_utils.get_kv_cache_pipe(args, "cpu")

        assert from_config_calls[0][1]["revision"] == COSMOS3_NANO_PINS.hf_revision
        assert vae_calls[0][1]["revision"] == COSMOS3_NANO_PINS.hf_revision

    def test_distillation_cosmos3_builder_resolves_the_pinned_revision(self, monkeypatch):
        from model_training import stages

        from_config_calls = []

        class _FakeCosmosCls:
            @staticmethod
            def from_config(*args, **kwargs):
                from_config_calls.append((args, kwargs))
                return "base-transformer"

        monkeypatch.setattr(stages, "Cosmos3OmniTransformer", _FakeCosmosCls)
        monkeypatch.setattr(
            stages,
            "ArtifixerCosmos3Transformer",
            lambda base, **kwargs: SimpleNamespace(base=base, **kwargs),
        )
        pipe = SimpleNamespace(
            attention_backend=None,
            vae=SimpleNamespace(config=SimpleNamespace(scale_factor_spatial=16, scale_factor_temporal=4), device="cpu"),
        )
        args = SimpleNamespace(gradient_checkpointing=False, attention_backend=None)

        stages.build_score_transformer(args, COSMOS3_NANO_MODEL_ID, pipe, gradient_checkpointing=False)

        assert from_config_calls[0][0] == (COSMOS3_NANO_MODEL_ID,)
        assert from_config_calls[0][1]["revision"] == COSMOS3_NANO_PINS.hf_revision

    def test_distillation_wan_builder_reads_the_registry_revision(self, monkeypatch):
        """The critic may run a different --model_id_critic; the builder must
        resolve pins per call-site model id, via the registry accessor."""
        from model_training import stages
        from model_training.utils import train_utils

        from_config_calls = []

        class _FakeWanCls:
            @staticmethod
            def from_config(*args, **kwargs):
                from_config_calls.append((args, kwargs))
                return "base-transformer"

        monkeypatch.setattr(stages, "WanTransformer3DModel", _FakeWanCls)
        monkeypatch.setattr(stages, "ArtifixerTransformer", lambda base, **kwargs: SimpleNamespace(base=base, **kwargs))
        # The builders share
        # train_utils._transformer_from_config, so the registry accessor is
        # looked up in train_utils, not per trainer module.
        monkeypatch.setattr(train_utils, "hf_revision_for_model", lambda model_id: f"pin::{model_id}")
        pipe = SimpleNamespace(
            attention_backend=None,
            vae=SimpleNamespace(config=SimpleNamespace(scale_factor_spatial=8, scale_factor_temporal=4), device="cpu"),
        )
        args = SimpleNamespace(
            gradient_checkpointing=False,
            local_attn_size=21,
            sink_size=7,
            checkpoint_every_n_blocks=1,
            attention_backend=None,
        )

        stages.build_score_transformer(args, WAN_14B_MODEL_ID, pipe, gradient_checkpointing=False)

        assert from_config_calls[0][1]["revision"] == f"pin::{WAN_14B_MODEL_ID}"

    def test_cosmos3_tokenizer_loads_at_the_pinned_tokenizer_revision(self, monkeypatch, tmp_path):
        import huggingface_hub
        import transformers

        import data_processing.captioning.prompt_hdf5 as prompt_hdf5
        from data_processing.captioning.prompt_hdf5 import get_cosmos3_tokenizer

        # The loader resolves the pinned snapshot locally first (offline caches
        # have no `.no_exist` records, so repo-id probes of optional tokenizer
        # files would hard-fail under HF_HUB_OFFLINE=1) and then loads from the
        # filesystem. Both halves of that contract are pinned here: the
        # snapshot resolution must carry the registry revision and
        # local_files_only, and the tokenizer must load from the resolved
        # text_tokenizer directory.
        snapshot_calls = []

        tokenizer_dir = tmp_path / "text_tokenizer"
        tokenizer_dir.mkdir()
        for name in ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
            (tokenizer_dir / name).touch()

        def fake_snapshot_download(model_id, **kwargs):
            snapshot_calls.append((model_id, kwargs))
            return str(tmp_path)

        monkeypatch.setattr(prompt_hdf5, "snapshot_download", fake_snapshot_download, raising=False)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

        tokenizer_calls = []
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: tokenizer_calls.append((args, kwargs)) or "tokenizer",
        )

        tokenizer = get_cosmos3_tokenizer(COSMOS3_NANO_MODEL_ID)

        assert tokenizer == "tokenizer"
        assert snapshot_calls == [
            (
                COSMOS3_NANO_MODEL_ID,
                {
                    "revision": COSMOS3_NANO_PINS.hf_revision,
                    "allow_patterns": ["text_tokenizer/*"],
                    "local_files_only": True,
                },
            )
        ]
        assert tokenizer_calls == [((tmp_path / "text_tokenizer",), {})]


def _write_dcp_cache(cache_path: Path, state_dict: dict) -> None:
    """Lay out a complete-looking DCP weight cache (payload + .dcp_complete
    marker) the way _ensure_dcp_cache's rank-0 conversion does."""
    import torch.distributed.checkpoint as dcp

    cache_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"model": state_dict}, storage_writer=dcp.FileSystemWriter(str(cache_path)), no_dist=True)
    (cache_path / ".dcp_complete").touch()
