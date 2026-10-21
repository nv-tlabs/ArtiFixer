# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay real Docker release boundaries; optional Enroot byte/mode comparison.

Run explicitly (not part of ordinary pytest):
python tests/release_image_integration.py NEW_OUTPUT_DIR --enroot
Requires a local python:3.12-slim image, Docker, and for --enroot Enroot/SquashFS.
Fixture images and all command/output evidence are retained for inspection.
The OpenCV/imageio modules are deliberate doubles: no GPU/library claim is made.
"""

import argparse
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/release-image"


def tree_inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        entry = {"mode": stat.S_IMODE(info.st_mode), "type": stat.S_IFMT(info.st_mode)}
        if path.is_symlink():
            entry["target"] = os.readlink(path)
        elif path.is_file():
            with path.open("rb") as handle:
                entry["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        result[str(path.relative_to(root))] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-image", default="python:3.12-slim")
    parser.add_argument("--platform", help="Explicit target platform, including non-native CPU emulation")
    parser.add_argument("--enroot", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prefix = "artifixer-release-fixture:" + uuid.uuid4().hex
    steps = []

    def run(name, command, expected=0, **kwargs):
        result = subprocess.run(command, capture_output=True, text=True, **kwargs)
        (output / f"{name}.stdout").write_text(result.stdout)
        (output / f"{name}.stderr").write_text(result.stderr)
        steps.append({"name": name, "command": command, "exit_status": result.returncode, "expected": expected})
        (output / "commands.json").write_text(json.dumps(steps, indent=2) + "\n")
        if result.returncode != expected:
            raise RuntimeError(f"{name}: expected exit {expected}, got {result.returncode}; see {output}")
        return result.stdout.strip()

    base = json.loads(run("base", ["docker", "image", "inspect", args.base_image]))[0]
    base_id = base["Id"]
    if args.platform:
        base = json.loads(run("base-platform", ["docker", "image", "inspect", "--platform", args.platform, base_id]))[0]
    target_platform = "/".join(base[key] for key in ("Os", "Architecture", "Variant") if base.get(key))
    daemon = json.loads(
        run(
            "daemon",
            [
                "docker",
                "info",
                "--format",
                '{"id":{{json .ID}},"driver":{{json .Driver}},"driver_status":{{json .DriverStatus}},"version":{{json .ServerVersion}}}',
            ],
        )
    )
    # BuildKit FROM parses sha256:ID as a repository/tag. Bind a private tag.
    base_tag = prefix + "-base"
    run("base-tag", ["docker", "tag", base_id, base_tag])
    source = prefix + "-source"
    flat = prefix + "-flat"
    run(
        "build",
        [
            "docker",
            "build",
            "--platform",
            target_platform,
            "--network",
            "none",
            "--build-arg",
            "BASE_IMAGE=" + base_tag,
            "-t",
            source,
            str(FIXTURES),
        ],
    )
    source_id = json.loads(run("source", ["docker", "image", "inspect", source]))[0]["Id"]

    def check(name, action, image, *extra, expected=0):
        selection = ["--platform", target_platform]
        command = [
            sys.executable,
            str(ROOT / "docker/release_image.py"),
            action,
            image,
            *extra,
            *selection,
            "--base-image",
            base_id,
            "--report",
            str(output / f"{name}.json"),
        ]
        return json.loads(run(name, command, expected))

    check("build-filesystem", "verify", source_id, "--build-image")
    rejected = check("deleted-layer", "verify", source_id, expected=1)
    assert rejected["gate"]["status"] == "pass" and rejected["saved_payload"] is None
    flattened = check("flatten", "flatten", source_id, flat)
    assert flattened["source_image_id"] == source_id
    assert flattened["saved_payload"]["status"] == "pass"
    assert (
        run("entrypoint", ["docker", "run", "--platform", target_platform, "--rm", "--network", "none", flat])
        == "configuration intact"
    )
    check("existing-destination", "flatten", source_id, flat, expected=1)
    missing = check("missing-image", "verify", prefix + "-missing", expected=1)
    assert missing["operation"] == "docker image inspect" and "No such image" in missing["diagnostic"]

    # Construct a one-layer image with codec bytes underneath Docker's /dev mount.
    # Building RUN touch /dev/... would only affect the temporary runtime mount.
    container = run(
        "masked-create", ["docker", "create", "--platform", target_platform, "--entrypoint", "/bin/true", flat]
    )
    masked = prefix + "-masked"
    with (
        (output / "masked-export.stderr").open("wb") as export_errors,
        (output / "masked-import.stdout").open("wb") as import_output,
        (output / "masked-import.stderr").open("wb") as import_errors,
    ):
        exporter = subprocess.Popen(["docker", "export", container], stdout=subprocess.PIPE, stderr=export_errors)
        importer = subprocess.Popen(
            [
                "docker",
                "import",
                "--platform",
                target_platform,
                "--change",
                "ENV PATH=/usr/local/bin:/usr/bin:/bin",
                "-",
                masked,
            ],
            stdin=subprocess.PIPE,
            stdout=import_output,
            stderr=import_errors,
        )
        try:
            with (
                tarfile.open(fileobj=exporter.stdout, mode="r|") as src,
                tarfile.open(fileobj=importer.stdin, mode="w|", format=tarfile.GNU_FORMAT) as dst,
            ):
                for member in src:
                    dst.addfile(member, src.extractfile(member) if member.isfile() else None)
                payload = b"\x7fELFmasked-codec-fixture"
                member = tarfile.TarInfo("dev/libavcodec.so.fixture")
                member.mode, member.size = 0o644, len(payload)
                dst.addfile(member, io.BytesIO(payload))
            exporter.stdout.close()
            importer.stdin.close()
            assert exporter.wait() == importer.wait() == 0
        finally:
            for process in (exporter, importer):
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            run("masked-remove-container", ["docker", "rm", container])
    masked_report = check("masked-rejection", "verify", masked, expected=1)
    assert masked_report["gate"]["status"] == "pass"
    assert any("dev/libavcodec.so.fixture" in path for path in masked_report["saved_payload"]["offenders"])

    enroot = {"status": "not requested"}
    if args.enroot:
        run("enroot-standard", ["enroot", "import", "--output", str(output / "standard.sqsh"), "dockerd://" + flat])
        run(
            "enroot-streamed",
            [sys.executable, str(FIXTURES / "export_streamed.py"), flat, str(output / "streamed.sqsh")],
        )
        for name in ("standard", "streamed"):
            run(
                "extract-" + name,
                ["unsquashfs", "-no-progress", "-d", str(output / name), str(output / (name + ".sqsh"))],
            )
        standard, streamed = (tree_inventory(output / name) for name in ("standard", "streamed"))
        differences = sorted(
            name for name in standard.keys() | streamed.keys() if standard.get(name) != streamed.get(name)
        )
        (output / "enroot-differences.json").write_text(json.dumps(differences, indent=2) + "\n")
        assert not differences, differences[:20]
        enroot = {
            "status": "pass",
            "entries": len(standard),
            "comparison": "paths, types, modes, symlink targets and file SHA256",
        }
    summary = {
        "status": "pass",
        "daemon": daemon,
        "platform": target_platform,
        "images": {"source": source_id, "flat": flattened["image_id"], "masked_tag": masked},
        "enroot": enroot,
        "commands": "commands.json",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
