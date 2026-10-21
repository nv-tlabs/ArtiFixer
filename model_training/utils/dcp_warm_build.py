# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-node DCP weight-cache warm build with provenance stamping.

The launch gates require a DCP transformer cache carrying both the
``.dcp_complete`` marker and a ``codex_source_provenance.txt`` with
``model_revision=`` / ``code_sha=`` / ``diffusers_commit=`` lines; the
trainer's first-start conversion writes only the marker. This module is the
sole producer of the provenance file. Run it single-node before any
multi-node launch: a first-start conversion on rank 0 is slow enough to trip
the NCCL watchdog on the waiting ranks.

Usage (inside the training container, one process, no torch.distributed):

    python3 -m model_training.utils.dcp_warm_build \
        --model_id nvidia/Cosmos3-Nano \
        --dcp_cache_dir /path/to/dcp-cache \
        --code_sha "$(git rev-parse HEAD)"

Behavior:
  * fresh cache: converts the pinned HF weights via the trainer's own
    ``_ensure_dcp_cache`` (same fingerprinted directory), then writes the
    provenance file atomically.
  * complete cache WITH matching provenance: verifies and exits 0
    (idempotent re-run).
  * complete cache with MISSING or MISMATCHED provenance: refuses loudly —
    the bytes cannot be attested for this launch; delete the cache directory
    and re-run to rebuild it at the pinned identities.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from model_training import pins
from model_training.utils import train_utils

PROVENANCE_FILENAME = "codex_source_provenance.txt"
# Keys the launch gates grep for as full lines (rq_gate_marker_line).
GATED_PROVENANCE_KEYS = ("model_revision", "code_sha", "diffusers_commit")


def _provenance_entries(model_id: str, code_sha: str) -> dict[str, str]:
    revision = pins.hf_revision_for_model(model_id)
    if revision is None:
        raise SystemExit(
            f"{model_id} has no pin-registry record (model_training.pins); a provenance file "
            "without a pinned model_revision cannot attest a launch. Add the family to the "
            "pin registry before warm-building its DCP cache."
        )
    return {
        "model_id": model_id,
        "model_revision": revision,
        "code_sha": code_sha,
        "diffusers_commit": pins.DIFFUSERS_COMMIT,
        "diffusers_version": pins.DIFFUSERS_VERSION,
    }


def _parse_provenance(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            parsed[key] = value
    return parsed


def _write_provenance_atomically(path: Path, entries: dict[str, str]) -> None:
    # created_utc is stamped here, at write time, so the attestation records
    # when the build completed rather than when the conversion started.
    stamped = {**entries, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text("".join(f"{key}={value}\n" for key, value in stamped.items()))
    tmp_path.replace(path)


def _verify_existing_provenance(path: Path, expected: dict[str, str]) -> None:
    recorded = _parse_provenance(path)
    mismatches = {key: (recorded.get(key), value) for key, value in expected.items() if recorded.get(key) != value}
    if mismatches:
        details = ", ".join(f"{key}: recorded={found!r} expected={want!r}" for key, (found, want) in mismatches.items())
        raise SystemExit(
            f"Existing DCP cache provenance at {path} does not attest this launch ({details}). "
            "The cached conversion was produced under different pins/code; delete the cache "
            "directory and re-run this tool to rebuild it."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--dcp_cache_dir", required=True, type=Path)
    parser.add_argument(
        "--code_sha",
        required=True,
        help="Repo commit the launch will run (the preflight script gates the checkout against it first)",
    )
    args = parser.parse_args(argv)

    import diffusers

    if diffusers.__version__ != pins.DIFFUSERS_VERSION:
        raise SystemExit(
            f"diffusers {diffusers.__version__} in this environment != pinned {pins.DIFFUSERS_VERSION} "
            "(model_training.pins); the conversion would not match the launch container."
        )

    cache_path = args.dcp_cache_dir / pins.dcp_cache_dir_name(args.model_id) / "transformer"
    marker_path = cache_path / ".dcp_complete"
    provenance = cache_path / PROVENANCE_FILENAME
    entries = _provenance_entries(args.model_id, args.code_sha)
    expected_gated = {key: entries[key] for key in GATED_PROVENANCE_KEYS}

    if marker_path.exists():
        if not provenance.exists():
            raise SystemExit(
                f"DCP cache at {cache_path} is complete but carries no {PROVENANCE_FILENAME} — it was "
                "built by a trainer first-start conversion (or predates the provenance contract), so its "
                "source cannot be attested. Delete the cache directory and re-run this tool."
            )
        _verify_existing_provenance(provenance, expected_gated)
        print("DCP_WARM_BUILD=verified")
    else:
        built_path = train_utils._ensure_dcp_cache(args.model_id, args.dcp_cache_dir)
        assert built_path == cache_path, f"cache path drifted: {built_path} != {cache_path}"
        _write_provenance_atomically(provenance, entries)
        print("DCP_WARM_BUILD=built")

    print(f"DCP_CACHE_PATH={cache_path}")
    print(f"DCP_PROVENANCE_PATH={provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
