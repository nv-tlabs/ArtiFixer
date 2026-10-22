# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise a real containerd cache containing two qualified fixture platforms.

Use an isolated containerd Docker daemon via DOCKER_HOST. Supply the real amd64
and arm64 flat fixture images produced by release_image_integration.py. The ARM
checks need native ARM execution or a working binfmt emulator. No GPU claim is
made. Docker/verifier commands are real; the Docker proxy only records calls
and rejects unexpected pulls. Images and evidence are retained.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCI = "application/vnd.oci.image."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--amd64-image", required=True)
    parser.add_argument("--arm64-image", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    steps = []

    def run(name, command, **kwargs):
        result = subprocess.run(command, capture_output=True, text=True, **kwargs)
        (output / (name + ".stdout")).write_text(result.stdout)
        (output / (name + ".stderr")).write_text(result.stderr)
        steps.append({"name": name, "command": command, "exit_status": result.returncode})
        (output / "commands.json").write_text(json.dumps(steps, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {output}")
        return result.stdout.strip()

    identity = json.loads(
        run(
            "daemon",
            [
                "docker",
                "info",
                "--format",
                '{"id":{{json .ID}},"driver_status":{{json .DriverStatus}},"version":{{json .ServerVersion}}}',
            ],
        )
    )
    assert ["driver-type", "io.containerd.snapshotter.v1"] in identity["driver_status"]
    manifests = []
    for architecture, image in (("amd64", args.amd64_image), ("arm64", args.arm64_image)):
        info = json.loads(
            run("inspect-" + architecture, ["docker", "image", "inspect", "--platform", "linux/" + architecture, image])
        )[0]
        assert info["Architecture"] == architecture
        manifests.append(info["Descriptor"] | {"platform": {"os": "linux", "architecture": architecture}})
        run(
            "save-" + architecture,
            [
                "docker",
                "image",
                "save",
                "--platform",
                "linux/" + architecture,
                "-o",
                str(output / (architecture + ".tar")),
                image,
            ],
        )
    tag = "artifixer-platform-fixture:" + uuid.uuid4().hex
    index = json.dumps({"schemaVersion": 2, "mediaType": OCI + "index.v1+json", "manifests": manifests}).encode()
    index_id = "sha256:" + hashlib.sha256(index).hexdigest()
    descriptor = {
        "mediaType": OCI + "index.v1+json",
        "digest": index_id,
        "size": len(index),
        "annotations": {
            "io.containerd.image.name": "docker.io/library/" + tag,
            "org.opencontainers.image.ref.name": tag.split(":")[1],
        },
    }
    with tarfile.open(output / "multi-platform.tar", "w") as dst:
        seen = set()
        for architecture in ("amd64", "arm64"):
            with tarfile.open(output / (architecture + ".tar"), "r|") as src:
                for member in src:
                    if member.isfile() and member.name.startswith("blobs/sha256/") and member.name not in seen:
                        dst.addfile(member, src.extractfile(member))
                        seen.add(member.name)
        entries = {
            "blobs/sha256/" + index_id[7:]: index,
            "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
            "index.json": json.dumps(
                {"schemaVersion": 2, "mediaType": OCI + "index.v1+json", "manifests": [descriptor]}
            ).encode(),
        }
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            dst.addfile(member, io.BytesIO(data))
    run("load-index", ["docker", "load", "-i", str(output / "multi-platform.tar")])
    root = json.loads(run("inspect-index", ["docker", "image", "inspect", tag]))[0]
    assert root["Id"] == index_id and root["Architecture"] == "amd64"
    selected = {}
    for architecture in ("amd64", "arm64"):
        info = json.loads(
            run(
                "select-" + architecture,
                ["docker", "image", "inspect", "--platform", "linux/" + architecture, index_id],
            )
        )[0]
        assert info["Architecture"] == architecture and info["Id"] != index_id
        selected[architecture] = info["Id"]
        receipt = output / ("index-" + architecture + ".json")
        run(
            "verify-index-" + architecture,
            [
                sys.executable,
                str(ROOT / "docker/release_image.py"),
                "verify",
                index_id,
                "--platform",
                "linux/" + architecture,
                "--base-image",
                index_id,
                "--report",
                str(receipt),
            ],
        )
        proof = json.loads(receipt.read_text())
        assert proof["status"] == "pass" and proof["image_id"] == index_id
        assert proof["platform_image_id"] == selected[architecture]
        assert proof["saved_payload"]["descriptor_chain"][0] == index_id
        assert selected[architecture] in proof["saved_payload"]["descriptor_chain"]
        assert proof["saved_payload"]["additional_layers"]

    # Minimal application stand-in: preserve the actual wrapper and verifier,
    # and build from the already validated Python/OpenCV-double fixture images.
    checkout = output / "checkout"
    for name in (
        "docker/build_release_image.sh",
        "docker/flatten_release_image.sh",
        "docker/release_image.py",
        "docker/image_inventory.py",
        "tests/assert_no_ffmpeg_in_image.py",
    ):
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in (
        "thirdparty/3DGRUT-ArtiFixer/setup.py",
        "thirdparty/3DGRUT-ArtiFixer/thirdparty/tiny-cuda-nn/include/tiny-cuda-nn/common.h",
    ):
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    for variant in ("cuda12", "cuda13-aarch64"):
        (checkout / ("Dockerfile." + variant)).write_text("FROM " + tag + "\n")
    tools = output / "bin"
    tools.mkdir()
    proxy = tools / "docker"
    proxy.write_text(
        "#!"
        + sys.executable
        + "\nimport os,sys,json\nwith open(os.environ['ARTIFIXER_INTEGRATION_COMMAND_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nif sys.argv[1:2]==['pull']: raise SystemExit('Unexpected pull from a populated fixture cache')\nos.execv("
        + repr(shutil.which("docker"))
        + ", ['docker', *sys.argv[1:]])\n"
    )
    proxy.chmod(0o755)
    cases = []
    for architecture, variant in (("amd64", "cuda12"), ("arm64", "cuda13-aarch64")):
        for check_only in (False, True):
            name = architecture + ("-check" if check_only else "-build")
            source_tag = tag + "-" + name
            if check_only:
                run("tag-" + name, ["docker", "tag", index_id, source_tag])
            receipt_dir = output / (name + "-receipts")
            command = [
                "bash",
                str(checkout / "docker/build_release_image.sh"),
                variant,
                "--tag",
                source_tag,
                "--flatten",
                "--report-dir",
                str(receipt_dir),
            ]
            if check_only:
                command.append("--check-only")
            log = output / (name + "-docker.jsonl")
            env = os.environ | {
                "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                "ARTIFIXER_INTEGRATION_COMMAND_LOG": str(log),
                "ARTIFIXER_GIT_REVISION": "integration-fixture",
            }
            run(name, command, env=env)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            assert not any(call[0] == "pull" for call in calls)
            if check_only:
                assert not any(call[0] == "build" and "--label" in call for call in calls)
            assert (receipt_dir / "base-image.id").read_text().strip() == selected[architecture]
            assert (receipt_dir / "base-image.root-id").read_text().strip() == index_id
            for filename in ("build-image.json", "release-image.json"):
                proof = json.loads((receipt_dir / filename).read_text())
                assert proof["status"] == "pass" and proof["platform"].startswith("linux/" + architecture)
                assert proof["inventories"]["base"]["image_id"] == index_id
                assert proof["inventories"]["base"]["platform_image_id"] == selected[architecture]
            cases.append(
                {
                    "architecture": architecture,
                    "check_only": check_only,
                    "base_platform_id": selected[architecture],
                    "base_root_id": index_id,
                    "status": "pass",
                }
            )
    summary = {
        "status": "pass",
        "daemon": identity,
        "root_index_id": index_id,
        "selected_platform_ids": selected,
        "cases": cases,
        "scope": "Actual containerd index, amd64/arm64 cache, wrapper/build/flatten/verification and inventories; no pulls in populated-cache cases. Python/OpenCV fixture images, not NGC builds or native GPU qualification.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
