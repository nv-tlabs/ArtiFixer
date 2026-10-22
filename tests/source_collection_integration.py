# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the complete source collector for a VCS replacement and a pinned FA3 source.

Requires Docker and a local python:3.12-slim image. Network retrieval alone is
replaced by a deterministic fixture; inventory, selection and packaging are real.
Usage: python tests/source_collection_integration.py NEW_OUTPUT_DIRECTORY
"""

import argparse
import hashlib
import json
import subprocess
import tarfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-image", default="python:3.12-slim")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    steps = []

    def run(name, command, **kwargs):
        result = subprocess.run(command, capture_output=True, text=True, **kwargs)
        with (output / (name + ".stdout")).open("x") as handle:
            handle.write(result.stdout)
        with (output / (name + ".stderr")).open("x") as handle:
            handle.write(result.stderr)
        steps.append({"name": name, "command": command, "exit_status": result.returncode})
        (output / "commands.json").write_text(json.dumps(steps, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {output}")
        return result.stdout.strip()

    prefix = "artifixer-source-fixture:" + uuid.uuid4().hex
    original = json.loads(run("inspect-python", ["docker", "image", "inspect", args.base_image]))[0]["Id"]
    run("bind-base", ["docker", "tag", original, prefix + "-python"])
    image_ids = {}
    for name, revision in (("base", "a" * 40), ("final", "b" * 40)):
        tag = prefix + "-" + name
        run(
            "build-" + name,
            [
                "docker",
                "build",
                "--network",
                "none",
                "--build-arg",
                "BASE_IMAGE=" + prefix + "-python",
                "--build-arg",
                "REVISION=" + revision,
                "-t",
                tag,
                str(ROOT / "tests/fixtures/source-inventory"),
            ],
        )
        image_ids[name] = json.loads(run("inspect-" + name, ["docker", "image", "inspect", tag]))[0]["Id"]
    collection = output / "collection"
    run(
        "collect",
        [
            "bash",
            str(ROOT / "docker/collect_container_sources.sh"),
            image_ids["base"],
            image_ids["final"],
            str(collection),
        ],
    )
    assert (collection / "pip.added").read_text() == "flash-attn-3==3.0.0\nsource-revision-fixture==1.0\n"
    assert not (collection / "apt.added").read_text()
    for name, revision in (("base", "a" * 40), ("final", "b" * 40)):
        inventory = json.loads((collection / (name + ".json")).read_text())
        assert inventory["pip"]["source-revision-fixture"] == {"version": "1.0", "source_revision": revision}
        if name == "base":
            assert "flash-attn-3" not in inventory["pip"]
        else:
            assert inventory["pip"]["flash-attn-3"] == {"version": "3.0.0", "source_revision": ""}
    archive_path = collection / ("sources/pip/source-revision-fixture-1.0-" + "b" * 40 + ".tar.gz")
    with tarfile.open(archive_path) as archive:
        assert archive.extractfile("source/REVISION").read() == ("b" * 40 + "\n").encode()
    fa3_revision = "77aacb68d194ba9af1010eda5eac3e7c0df8e6f6"
    fa3_path = collection / f"sources/pip/flash-attn-3-3.0.0-{fa3_revision}.tar.gz"
    with tarfile.open(fa3_path) as archive:
        assert archive.extractfile("source/REVISION").read() == (fa3_revision + "\n").encode()
    assert not (collection / "NEEDS-ATTENTION.txt").read_text()
    run("source-checksums", ["sha256sum", "-c", "SHA256SUMS"], cwd=collection)
    run("archive-checksum", ["sha256sum", "-c", "artifixer-container-sources.tar.sha256"], cwd=collection)
    with tarfile.open(collection / "artifixer-container-sources.tar") as archive:
        assert {"base.json", "final.json", "MANIFEST.txt", "SHA256SUMS"} <= set(archive.getnames())
    summary = {
        "status": "pass",
        "images": image_ids,
        "version": "1.0",
        "base_revision": "a" * 40,
        "collected_revision": "b" * 40,
        "source_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "fa3_pinned_revision": fa3_revision,
        "fa3_source_sha256": hashlib.sha256(fa3_path.read_bytes()).hexdigest(),
        "scope": "Real complete shell caller: Docker inventory, VCS-aware delta, FA3 override without VCS metadata, fetch selection and source archive/checksums; deterministic network transport double",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
