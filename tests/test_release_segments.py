# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded distribution layers preserve the sanitized filesystem contract."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


@pytest.fixture
def packaging(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "docker"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("test_package_image", directory / "package_release_image.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_layout(module, root, defect=None):
    root.mkdir()
    blobs = root / "blobs/sha256"
    blobs.mkdir(parents=True)
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w", format=tarfile.PAX_FORMAT) as archive:
        entries = [
            (".", tarfile.DIRTYPE, b""),
            ("work", tarfile.DIRTYPE, b""),
            ("work/empty", tarfile.DIRTYPE, b""),
            ("work/a", tarfile.REGTYPE, b"a" * 48_000),
            ("work/b", tarfile.REGTYPE, b"b" * 48_000),
            ("work/again", tarfile.LNKTYPE, b""),
            ("work/current", tarfile.SYMTYPE, b""),
            ("work/" + "x" * 180, tarfile.REGTYPE, b"long path"),
        ]
        if defect == "duplicate":
            entries.append(("work/a", tarfile.REGTYPE, b"replacement"))
        if defect == "whiteout":
            entries.append(("work/.wh.a", tarfile.REGTYPE, b""))
        if defect == "oversized":
            entries.append(("work/huge", tarfile.REGTYPE, b"x" * 70_000))
        for name, kind, payload in entries:
            member = tarfile.TarInfo(name)
            member.type, member.mode, member.uid, member.gid = kind, 0o750, 1234, 1235
            member.mtime = 1_700_000_000
            member.pax_headers = {"mtime": "1700000000.123456789"}
            if name == "work/a":
                member.pax_headers["SCHILY.xattr.user.release"] = "retained"
            if member.issym():
                member.linkname = "a"
            if member.islnk():
                member.linkname = "work/a"
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload) if member.isfile() else None)
    raw = data.getvalue()
    identifier = "sha256:" + module.hashlib.sha256(raw).hexdigest()
    (blobs / identifier[7:]).write_bytes(raw)
    config = {
        "os": "linux",
        "architecture": "amd64",
        "created": "2026-01-01T00:00:00Z",
        "config": {"Env": ["FIXTURE=literal"], "WorkingDir": "/work", "Cmd": ["/bin/true"]},
        "rootfs": {"type": "layers", "diff_ids": [identifier]},
        "history": [{"created_by": "fixture"}],
    }
    descriptor = module.write_json_blob(root, config, module.OCI + "config.v1+json")
    manifest = {
        "schemaVersion": 2,
        "mediaType": module.OCI + "manifest.v1+json",
        "config": descriptor,
        "layers": [{"mediaType": module.OCI + "layer.v1.tar", "digest": identifier, "size": len(raw)}],
    }
    desc = module.write_json_blob(root, manifest, manifest["mediaType"])
    (root / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [desc]}))
    (root / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    return config


def test_segments_preserve_bytes_links_xattrs_and_directory_metadata(packaging, tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    config = source_layout(packaging, source)
    result = packaging.package(source, target, 64 * 1024)
    assert len(result["layers"]) >= 3
    assert max(layer["size"] for layer in result["layers"]) < 64 * 1024
    assert result["filesystem_contents_and_metadata_unchanged"]
    assert result["saved_payload"]["filesystem_layout"]["status"] == "pass"
    manifest, _, actual = packaging.inspect_layout(target)
    assert actual["config"] == config["config"]
    assert len(actual["history"]) == len(actual["rootfs"]["diff_ids"]) == len(manifest["layers"])
    with packaging.decoded_blob(target, result["layers"][-1], result["layers"][-1]["diff_id"]) as stream:
        final = packaging.tar_records(stream)
    with packaging.decoded_blob(target, result["layers"][0], result["layers"][0]["diff_id"]) as stream:
        first = packaging.tar_records(stream)
    assert first["work/again"]["kind"] == "hardlink"
    assert first["work/again"]["linkname"] == "work/a" and first["work/a"]["kind"] == "file"
    assert final["work"]["mode"] == 0o750 and final["work"]["uid"] == 1234
    assert final["work"]["mtime"] == "1700000000.123456789"


@pytest.mark.parametrize("defect", ["duplicate", "whiteout", "oversized"])
def test_unsafe_or_unrepresentable_sources_do_not_produce_a_release(packaging, tmp_path, defect):
    source, target = tmp_path / "source", tmp_path / "target"
    source_layout(packaging, source, defect)
    with pytest.raises(ValueError):
        packaging.package(source, target, 64 * 1024)
    assert not (target / "index.json").exists()


@pytest.mark.parametrize(
    "entries",
    [
        [("file", "file", ""), ("file", "file", "")],
        [("alias", "other", ""), ("alias/hidden", "file", "")],
        [("alias", "hardlink", "later"), ("later", "file", "")],
        [("work/a", "file", ""), ("work/b", "hardlink", "work/a")],
    ],
)
def test_layer_layout_rejects_overwrites_alias_ancestors_and_forward_hardlinks(packaging, entries):
    layers = [{"_entries": [entry]} for entry in entries]
    assert packaging.release_image.filesystem_layout(layers)["status"] == "fail"


def test_layer_layout_allows_disjoint_data_and_later_directory_metadata(packaging):
    layers = [
        {"_entries": [("work/a", "file", ""), ("work/b", "hardlink", "work/a")]},
        {"_entries": [("work", "directory", "")]},
    ]
    assert packaging.release_image.filesystem_layout(layers)["status"] == "pass"
