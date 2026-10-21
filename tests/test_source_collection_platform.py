# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


def test_source_collector_keeps_selected_platform_for_inventory_and_fetch(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    shim = tools / "docker"
    shim.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["DOCKER_CALLS"], "a") as log:
    log.write(json.dumps(args) + "\n")
if args[:2] == ["image", "inspect"]:
    selected = "--platform" in args
    image = args[-1]
    role = "base" if "base" in image else "final"
    architecture = "arm64" if selected or role == "final" else "amd64"
    print(json.dumps([{"Id": "sha256:" + role + ("-arm64" if selected else "-root"),
                       "Os": "linux", "Architecture": architecture, "Config": {}}]))
elif args[0] == "run":
    assert args[args.index("--platform") + 1] == "linux/arm64"
    if "-v" in args:
        output = pathlib.Path(args[args.index("-v") + 1].removesuffix(":/out"))
        (output / "sources").mkdir()
        (output / "NEEDS-ATTENTION.txt").touch()
    else:
        print(json.dumps({"pip": {"pkg": {"version": "1"}}, "apt": {"base": {"version": "1"}}}))
else:
    raise AssertionError(args)
"""
    )
    shim.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    output = tmp_path / "collection"
    script = Path(__file__).resolve().parents[1] / "docker/collect_container_sources.sh"
    result = subprocess.run(
        ["bash", str(script), "base:multi", "final:arm", str(output), "linux/arm64"],
        env=os.environ | {"PATH": str(tools) + os.pathsep + os.environ["PATH"], "DOCKER_CALLS": str(calls)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    runs = [command for command in commands if command[0] == "run"]
    assert len(runs) == 3
    assert [next(arg for arg in run if arg.startswith("sha256:")) for run in runs] == [
        "sha256:base-root",
        "sha256:final-root",
        "sha256:final-root",
    ]
    assert "platform: linux/arm64" in (output / "MANIFEST.txt").read_text()
    assert (output / "artifixer-container-sources.tar").is_file()
