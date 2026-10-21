# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-repo pin registry for pretrained-model provenance.

The single in-repo source of truth for the HF weights/tokenizer revision and
the diffusers commit: loaders pass ``revision=`` from here, the DCP cache
directory is named after a fingerprint of
``(model_id, hf_revision, diffusers_commit)`` (so a revision bump cannot
silently reuse a stale cache), and tests pin the Dockerfile/install pins
against it. Upgrading a model family =
bump one record (plus regenerating the fixtures its tests point at).

Unpinned families (the public Wan releases) intentionally have no record:
their loaders receive ``revision=None`` (hub default) and their DCP cache is
still fingerprinted against the diffusers commit, whose model code defines the
state-dict layout being cached.
"""

import hashlib
import json
from dataclasses import dataclass

from model_training.model_profiles import COSMOS3_NANO_MODEL_ID, is_cosmos3_nano

# The diffusers dependency pin (v0.39.0). Must match the Dockerfile installs;
# tests/test_model_pins.py asserts they cannot drift apart.
DIFFUSERS_COMMIT = "a3608b512ed7248499a44c61d954965ed9bdae4d"
DIFFUSERS_VERSION = "0.39.0"

# The safetensors dependency pin. diffusers 0.39.0 requires >=0.8.0 and the
# deployed images bake exactly 0.8.0; the launch preflight asserts the
# in-container install against this record.
SAFETENSORS_VERSION = "0.8.0"


def compute_config_fingerprint(model_id: str, hf_revision: str | None, diffusers_commit: str) -> str:
    """Deterministic fingerprint of a pinned pretrained-artifact identity.

    Hashes the (model_id, hf_revision, diffusers_commit) triple: the hub
    revision pins the weights/config bytes and the diffusers commit pins the
    model implementation that materializes them (state-dict layout, key
    names). Any component changing must invalidate derived artifacts such as
    the DCP weight cache. ``hf_revision=None`` marks an unpinned family and
    still yields a stable fingerprint.
    """
    payload = json.dumps(
        {"model_id": model_id, "hf_revision": hf_revision, "diffusers_commit": diffusers_commit},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ModelPins:
    """Provenance record for one pretrained model family."""

    model_id: str
    # Git revision of the HF hub weights repo (commit sha, not a branch name).
    # The text tokenizer ships in the same repo, so tokenizer loads use it too.
    hf_revision: str
    # Commit of the diffusers dependency whose model code loads the weights.
    diffusers_commit: str
    # compute_config_fingerprint(model_id, hf_revision, diffusers_commit);
    # names the DCP weight cache directory.
    config_fingerprint: str


def _make_pins(model_id: str, hf_revision: str) -> ModelPins:
    return ModelPins(
        model_id=model_id,
        hf_revision=hf_revision,
        diffusers_commit=DIFFUSERS_COMMIT,
        config_fingerprint=compute_config_fingerprint(model_id, hf_revision, DIFFUSERS_COMMIT),
    )


# The action-capable Cosmos3-Nano snapshot both ablation arms train from.
COSMOS3_NANO_HF_REVISION = "23314034b82f46b45339035dba67c3ee9bbcb8ba"

COSMOS3_NANO_PINS = _make_pins(
    model_id=COSMOS3_NANO_MODEL_ID,
    hf_revision=COSMOS3_NANO_HF_REVISION,
)

# (family predicate, record). Predicates come from model_profiles so path-form
# model ids (e.g. a local snapshot directory ending in Cosmos3-Nano) resolve to
# the same record the hub id does. Only families with a vetted revision get a
# record; returning the Nano record for an unvetted model would be a lie.
_MODEL_PIN_MATCHERS = ((is_cosmos3_nano, COSMOS3_NANO_PINS),)

MODEL_PIN_REGISTRY: tuple[ModelPins, ...] = tuple(pins for _, pins in _MODEL_PIN_MATCHERS)


def pins_for_model(model_id: str) -> ModelPins | None:
    for matches_family, pins in _MODEL_PIN_MATCHERS:
        if matches_family(model_id):
            return pins
    return None


def hf_revision_for_model(model_id: str) -> str | None:
    """Hub revision loaders must pass as ``revision=``; None = unpinned."""
    pins = pins_for_model(model_id)
    return pins.hf_revision if pins is not None else None


def tokenizer_revision_for_model(model_id: str) -> str | None:
    """Hub revision for tokenizer loads (prompt generation); None = unpinned."""
    pins = pins_for_model(model_id)
    return pins.hf_revision if pins is not None else None


def dcp_cache_dir_name(model_id: str) -> str:
    """Leaf directory name for the DCP-converted transformer weight cache.

    Keyed by the pin fingerprint, not just the flattened model id, so bumping
    the pinned hub revision or the diffusers commit stops resolving to the old
    conversion instead of silently training from stale weights. The flattened
    model id stays in the name for operators browsing the cache directory.
    """
    pins = pins_for_model(model_id)
    fingerprint = (
        pins.config_fingerprint if pins is not None else compute_config_fingerprint(model_id, None, DIFFUSERS_COMMIT)
    )
    return f"{model_id.replace('/', '--')}--{fingerprint}"
