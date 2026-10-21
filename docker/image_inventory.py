# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit resolved pip/dpkg metadata inside an image, without network access."""

import importlib.metadata as md
import json
import re
import subprocess


def inventory():
    pip = {}
    for candidate in md.distributions():
        name = re.sub(r"[-_.]+", "-", candidate.metadata["Name"]).lower()
        if name in pip:
            continue
        dist = md.distribution(name)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        # URLs can contain credentials. Only retain the immutable VCS identity.
        pip[name] = {
            "version": dist.version,
            "source_revision": direct.get("vcs_info", {}).get("commit_id", ""),
        }
    output = subprocess.run(
        ["dpkg-query", "-W", "-f", "${binary:Package}\t${Version}\t${Architecture}\t${db:Status-Status}\n"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    apt = {}
    for line in output.splitlines():
        name, version, architecture, status = line.split("\t")
        if status == "installed":
            apt[name] = {"version": version, "architecture": architecture}
    if not pip or not apt:
        raise ValueError("Incomplete image inventory: expected installed pip and dpkg packages")
    return {"schema_version": 1, "pip": pip, "apt": apt}


if __name__ == "__main__":
    print(json.dumps(inventory(), sort_keys=True, indent=2))
