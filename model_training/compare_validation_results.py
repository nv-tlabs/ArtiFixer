# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare checkpoint/teacher evaluation artifacts that replayed the same panel."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from model_training.utils.validation_artifacts import VALIDATION_SCHEMA_VERSION, write_json
from model_training.utils.validation_metrics import METRIC_PROTOCOL


def compare_results(reference: dict, candidate: dict) -> dict:
    for result in (reference, candidate):
        if (
            result.get("schema_version") != VALIDATION_SCHEMA_VERSION
            or result.get("status") != "complete"
            or not result.get("panel")
        ):
            raise ValueError("Both results must contain complete version-3 panel evaluations; regenerate older results")
    if not reference.get("manifest_id") or reference["manifest_id"] != candidate.get("manifest_id"):
        raise ValueError("Results use different or missing input manifests")
    if reference.get("metric_protocol", METRIC_PROTOCOL) != candidate.get("metric_protocol", METRIC_PROTOCOL):
        raise ValueError("Results use different metric protocols")
    if reference["sampling"]["validation_chunk_frames"] != candidate["sampling"]["validation_chunk_frames"]:
        raise ValueError("Results use different chunk-metric grids")
    before, after = reference["panel"]["tasks"], candidate["panel"]["tasks"]
    if before.keys() != after.keys():
        raise ValueError("Panel task coverage differs")
    same_hardware = all(reference.get(key) == candidate.get(key) for key in ("device_name", "world_size"))
    deltas, speedups = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for key in before:
        config = key.split("@", 1)[0]
        a, b = before[key]["metrics"], after[key]["metrics"]
        for name in a.keys() & b.keys():
            if name in {"num_repeat_pairs", "scoring_seconds", "generation_seconds", "generation_fps"}:
                continue
            if name == "sampler_boundary_delta_l1" and reference["sampling"].get("frames_per_block") != candidate[
                "sampling"
            ].get("frames_per_block"):
                continue
            deltas[config][name].append(b[name] - a[name])
        if (
            same_hardware
            and before[key]["cp_size"] == after[key]["cp_size"]
            and a.get("generation_seconds", 0) > 0
            and b.get("generation_seconds", 0) > 0
        ):
            speedups[config].append(a["generation_seconds"] / b["generation_seconds"])
    return {
        "manifest_id": reference["manifest_id"],
        "reference_checkpoint": reference["checkpoint"],
        "candidate_checkpoint": candidate["checkpoint"],
        "candidate_minus_reference": {
            config: {metric: sum(values) / len(values) for metric, values in metrics.items()}
            for config, metrics in deltas.items()
        },
        "generation_speedup": {config: sum(values) / len(values) for config, values in speedups.items()},
        "timing_note": "Generation includes encoding, denoising, decoding and CPU output transfer; excludes scoring. "
        "Speedups require matching GPU model, world size and per-task CP size; they are not warmed benchmarks.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(
        args.output, compare_results(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    )


if __name__ == "__main__":
    main()
