# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic transport double inside the source-collection fixture only."""

import importlib.metadata
import io
import json
import tarfile
import urllib.request
from pathlib import Path


def retrieve(url, destination, *args, **kwargs):
    direct = json.loads(importlib.metadata.distribution("source-revision-fixture").read_text("direct_url.json"))
    revision = direct["vcs_info"]["commit_id"]
    fa3_revision = "77aacb68d194ba9af1010eda5eac3e7c0df8e6f6"
    expected = {
        "https://codeload.github.com/fixture-owner/review-package/tar.gz/" + revision: revision,
        "https://codeload.github.com/Dao-AILab/flash-attention/tar.gz/" + fa3_revision: fa3_revision,
    }
    if url not in expected:
        raise AssertionError("Unexpected source URL in isolated collection fixture")
    data = (expected[url] + "\n").encode()
    with tarfile.open(destination, "w:gz") as archive:
        member = tarfile.TarInfo("source/REVISION")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    return destination, None


def refuse_network(*args, **kwargs):
    raise AssertionError("Unexpected network request in isolated collection fixture")


if Path("/opt/artifixer-source-collection-fixture").is_file():
    urllib.request.urlretrieve = retrieve
    urllib.request.urlopen = refuse_network
