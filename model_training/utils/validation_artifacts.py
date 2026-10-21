# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable panel inputs and machine-readable evaluation completeness."""

import hashlib
import json
import math
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import torch

# Fixed sampling and noise policy shared by dataset capture and generation.
VALIDATION_SEED = 42

VALIDATION_SCHEMA_VERSION = 3


def tensor_identity(value: torch.Tensor | None) -> dict | None:
    if value is None:
        return None
    value = value.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def content_id(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read_file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=256)
def _file_sha256(path: Path, size: int, mtime_ns: int, ctime_ns: int, device: int, inode: int) -> str:
    return _read_file_sha256(path)


def file_sha256(path: Path) -> str:
    stat = path.stat()
    # Two rapid writes can share a filesystem timestamp, even when Python exposes
    # nanoseconds. Hash recent files directly until that ambiguity has passed.
    if time.time_ns() - max(stat.st_mtime_ns, stat.st_ctime_ns) < 1_000_000_000:
        return _read_file_sha256(path)
    return _file_sha256(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino)


@lru_cache(maxsize=1)
def code_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != VALIDATION_SCHEMA_VERSION:
        raise ValueError("Validation manifest version 3 is required; regenerate older manifests to pin conditioning")
    if manifest.get("noise_seed") != 42:
        raise ValueError("Unsupported validation manifest noise seed")
    if manifest.get("num_neighbors") != 3 or not manifest.get("samples"):
        raise ValueError("A panel manifest must contain samples with three neighbors")
    frames = manifest.get("num_frames", 0)
    if frames <= 0 or (frames - 1) % 4:
        raise ValueError("Manifest num_frames must be a positive 1+4*n")
    for sample in manifest["samples"]:
        for key in ("scene_id", "half", "prompt_file", "prompt_window", "resolution", "prompt_sha256"):
            if key not in sample:
                raise ValueError(f"Manifest sample is missing {key}")
        if Path(sample["prompt_file"]).name != sample["prompt_file"]:
            raise ValueError("Manifest prompt_file must be a filename")
        if len(sample["frame_indices"]) != frames or len(sample["neighbor_indices"]) != 3:
            raise ValueError("Manifest frame/neighbor counts do not match its configuration")
        if "negative_conditioning" not in sample or not isinstance(sample["negative_conditioning"], (dict, type(None))):
            raise ValueError("Manifest sample is missing effective negative conditioning identity")
        identity = sample.get("source_identity", {})
        if not all(identity.get(key) for key in ("geometry_sha256", "scene_archive_sha256", "reconstructions")):
            raise ValueError("Manifest sample is missing source identity fingerprints")
        if "3" not in identity["reconstructions"]:
            raise ValueError("Manifest source identity is missing the three-view reconstruction")
        for source in identity["reconstructions"].values():
            scale = source.get("metric_scale")
            if (
                not source.get("sha256")
                or not isinstance(scale, (int, float))
                or not math.isfinite(scale)
                or scale <= 0
            ):
                raise ValueError("Manifest reconstruction must have a fingerprint and a positive finite metric_scale")


def pass_report(expected_keys, metrics: dict, records: dict) -> dict:
    """Keep partial scores visible without confusing missing data with success."""
    tasks = {}
    counts = {}
    for key in expected_keys:
        record = records.get(key, {})
        values = metrics.get(key, {})
        errors = dict(record.get("errors", {}))
        if key not in records:
            errors["task"] = "Task did not produce a scoring record"
        nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
        if nonfinite:
            errors["nonfinite"] = ", ".join(nonfinite)
        values = {name: value for name, value in values.items() if math.isfinite(value)}
        for name in values:
            counts[name] = counts.get(name, 0) + 1
        missing = record.get("missing_metrics", [])
        complete = not errors and not missing and bool(values)
        tasks[key] = {**record, "metrics": values, "errors": errors, "complete": complete}
    completed = sum(task["complete"] for task in tasks.values())
    return {
        "status": "complete" if tasks and completed == len(tasks) else "incomplete",
        "expected_tasks": len(tasks),
        "scored_tasks": sum(bool(task["metrics"]) for task in tasks.values()),
        "complete_tasks": completed,
        "metric_counts": counts,
        "tasks": tasks,
    }
