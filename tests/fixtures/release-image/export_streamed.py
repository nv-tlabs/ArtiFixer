# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream a Docker rootfs into SquashFS with the installed Enroot configurator.

Avoids a second full extracted rootfs on a space-constrained workstation.
The original image is never started or modified. Output is only a QA derivative.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile
import tempfile
import uuid

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("output", type=pathlib.Path)
a = p.parse_args()
assert not a.output.exists(), a.output
a.output.parent.mkdir(parents=True, exist_ok=True)
meta = json.loads(subprocess.check_output(["docker", "image", "inspect", a.image]))[0]
assert not meta["Config"].get("Volumes")
name = "artifixer-sqsh-" + uuid.uuid4().hex
with tempfile.TemporaryDirectory(prefix="artifixer-enroot-config-") as temp:
    root = pathlib.Path(temp)
    (root / "config.json").write_text(json.dumps({k.lower(): v for k, v in meta.items()}))
    env = os.environ | {
        "ENROOT_LIBRARY_PATH": "/usr/lib/enroot",
        "ENROOT_CACHE_PATH": temp,
        "ENROOT_CONFIG_PATH": temp,
        "ENROOT_TRANSFER_RETRIES": "0",
        "ENROOT_CONNECT_TIMEOUT": "1",
        "ENROOT_TRANSFER_TIMEOUT": "1",
    }
    subprocess.run(
        [
            "bash",
            "-ec",
            'shopt -s lastpipe; source /usr/lib/enroot/docker.sh; docker::configure "$1" "$2" "$3"',
            "configure",
            str(root / "rootfs"),
            str(root / "config.json"),
            meta["Architecture"],
        ],
        env=env,
        check=True,
    )
    replacements = {str(f.relative_to(root / "rootfs")): f for f in (root / "rootfs/etc").iterdir()}
    subprocess.run(
        ["docker", "create", "--name", name, "--entrypoint", "/bin/true", meta["Id"]],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    export = None
    squash = None
    try:
        export = subprocess.Popen(["docker", "export", name], stdout=subprocess.PIPE)
        squash = subprocess.Popen(
            [
                "sqfstar",
                "-comp",
                "zstd",
                "-processors",
                "8",
                "-mem",
                "2G",
                "-no-progress",
                "-all-root",
                "-exit-on-error",
                str(a.output),
            ],
            stdin=subprocess.PIPE,
        )
        count = 0
        try:
            with (
                tarfile.open(fileobj=export.stdout, mode="r|") as src,
                tarfile.open(fileobj=squash.stdin, mode="w|", format=tarfile.GNU_FORMAT) as dst,
            ):
                for member in src:
                    path = member.name.removeprefix("./").rstrip("/")
                    if path == ".dockerenv" or path.startswith("dev/") or path in replacements:
                        continue
                    if not path or path.startswith("/") or ".." in pathlib.PurePosixPath(path).parts:
                        raise ValueError(member.name)
                    # Same accessibility adjustment as Enroot's common::fixperms.
                    member.mode |= 0o400
                    if member.isdir() or member.mode & 0o111:
                        member.mode |= 0o100
                    if member.isdir() and len(pathlib.PurePosixPath(path).parts) <= 5:
                        member.mode |= 0o200
                    dst.addfile(member, src.extractfile(member) if member.isfile() else None)
                    count += 1
                for path, f in replacements.items():
                    dst.add(f, arcname=path, recursive=False)
        finally:
            export.stdout.close()
            squash.stdin.close()
        es = export.wait()
        ss = squash.wait()
        assert (es, ss) == (0, 0), (es, ss)
    finally:
        for process in (export, squash):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, check=True)
    digest = hashlib.sha256()
    with a.output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    receipt = {
        "image_id": meta["Id"],
        "platform": meta["Os"] + "/" + meta["Architecture"],
        "source_sha": meta["Config"].get("Labels", {}).get("org.opencontainers.image.revision"),
        "path": str(a.output.resolve()),
        "bytes": a.output.stat().st_size,
        "sha256": digest.hexdigest(),
        "converter_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "enroot_configurator_sha256": hashlib.sha256(
            pathlib.Path("/usr/lib/enroot/docker.sh").read_bytes()
        ).hexdigest(),
        "entries_copied": count,
        "conversion": "Docker export streamed through tarfile and sqfstar; standard Enroot configuration and accessibility adjustments; omit dev contents and .dockerenv",
    }
    a.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
