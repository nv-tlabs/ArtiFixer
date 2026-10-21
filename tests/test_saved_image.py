# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved-image identities and authenticated descriptor/layer boundaries."""

import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

OCI = "application/vnd.oci.image."


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def fixture_archive(
    identity="manifest",
    encoding="gzip",
    defect=None,
    masked_codec=False,
    legacy=False,
    other_platform=False,
    other_codec=False,
    payload_path=None,
    payload_type=tarfile.REGTYPE,
):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        data = b"\x7fELFfixture" if masked_codec else b"fixture\n"
        member = tarfile.TarInfo(payload_path or ("dev/libavcodec.so" if masked_codec else "{fixture}"))
        member.type = payload_type
        if member.isfile():
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        else:
            if member.issym() or member.islnk():
                member.linkname = "git-storage"
            archive.addfile(member)
    layer = payload.getvalue()
    if encoding == "gzip":
        packed = gzip.compress(layer, mtime=0)
    elif encoding == "zstd":
        packed = pytest.importorskip("zstandard").ZstdCompressor().compress(layer)
    else:
        packed = layer
    if defect == "compressed-payload":
        packed = (b"\x1f\x8b" if encoding == "gzip" else b"\x28\xb5\x2f\xfd") + b"invalid fixture" * 8
    blobs = {}

    def add(data, media):
        identifier = digest(data)
        blobs["blobs/sha256/" + identifier[7:]] = data
        return {"mediaType": media, "digest": identifier, "size": len(data)}

    def document(data, media):
        return add(json.dumps(data, sort_keys=True).encode(), media)

    layer_descriptor = add(packed, OCI + "layer.v1.tar" + ("+" + encoding if encoding != "none" else ""))
    expected = [digest(layer)]
    config = {"os": "linux", "architecture": "amd64", "config": {}, "rootfs": {"type": "layers", "diff_ids": expected}}
    if defect == "diff-id":
        config["rootfs"] = {"type": "layers", "diff_ids": ["sha256:" + "a" * 64]}
    if defect == "platform":
        config["architecture"] = "arm64"
    if defect == "runtime-config":
        config["config"] = {"Env": ["UNEXPECTED=value"]}
    config_descriptor = document(config, OCI + "config.v1+json")
    config_path = "blobs/sha256/" + config_descriptor["digest"][7:]
    if defect == "config-size":
        config_descriptor["size"] += 1
    if defect == "config-type":
        config_descriptor["mediaType"] = "application/unknown"
    if defect == "layer-size":
        layer_descriptor["size"] += 1
    manifest = {
        "schemaVersion": 2,
        "mediaType": OCI + "manifest.v1+json",
        "config": config_descriptor,
        "layers": [layer_descriptor],
    }
    manifest_descriptor = document(manifest, OCI + "manifest.v1+json")
    child = manifest_descriptor | {"platform": {"os": "linux", "architecture": "amd64"}}
    children = [child, child] if defect == "ambiguous" else [child]
    # A platform-filtered export need not include uncached other architectures.
    if other_platform:
        other_payload = io.BytesIO()
        with tarfile.open(fileobj=other_payload, mode="w") as archive:
            data = b"\x7fELFother" if other_codec else b"other platform\n"
            member = tarfile.TarInfo("sys/libavcodec.so" if other_codec else "other-platform")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        other_layer = other_payload.getvalue()
        other_blob = gzip.compress(other_layer, mtime=0) if encoding == "gzip" else other_layer
        other_descriptor = add(other_blob, OCI + "layer.v1.tar" + ("+gzip" if encoding == "gzip" else ""))
        other_config = document(
            {
                "os": "linux",
                "architecture": "arm64",
                "config": {},
                "rootfs": {
                    "type": "layers",
                    "diff_ids": ["sha256:" + "b" * 64 if defect == "other-diff-id" else digest(other_layer)],
                },
            },
            OCI + "config.v1+json",
        )
        other_manifest = document(
            {
                "schemaVersion": 2,
                "mediaType": OCI + "manifest.v1+json",
                "config": other_config,
                "layers": [other_descriptor],
            },
            OCI + "manifest.v1+json",
        )
        children.append(other_manifest | {"platform": {"os": "linux", "architecture": "arm64"}})
    else:
        children.append(
            {
                "mediaType": OCI + "manifest.v1+json",
                "digest": "sha256:" + "f" * 64,
                "size": 1,
                "platform": {"os": "linux", "architecture": "arm64"},
            }
        )
    index_descriptor = document(
        {"schemaVersion": 2, "mediaType": OCI + "index.v1+json", "manifests": children}, OCI + "index.v1+json"
    )
    if identity == "nested-index":
        index_descriptor = document(
            {"schemaVersion": 2, "mediaType": OCI + "index.v1+json", "manifests": [index_descriptor]},
            OCI + "index.v1+json",
        )
    layer_path = "blobs/sha256/" + layer_descriptor["digest"][7:]
    root = {
        "config": config_descriptor,
        "manifest": manifest_descriptor,
        "index": index_descriptor,
        "nested-index": index_descriptor,
    }[identity]
    if legacy:
        blobs = {config_descriptor["digest"][7:] + ".json": blobs[config_path], "layer0/layer.tar": layer}
        config_path, layer_path = list(blobs)
    if defect == "config-hash":
        blobs[config_path] = blobs[config_path].replace(b"amd64", b"arm64")
    if defect == "missing-config":
        del blobs[config_path]
    if defect == "layer-hash":
        blobs[layer_path] += b"unexpected trailing bytes"
    if defect == "extra-layer":
        add(layer + bytes(512), OCI + "layer.v1.tar")
    compat = [{"Config": config_path, "Layers": [layer_path], "RepoTags": ["fixture:flat"]}]
    if defect == "compat-config":
        compat[0]["Config"] = "unrelated.json"
    if defect == "compat-layers":
        compat[0]["Layers"] = []
    blobs["manifest.json"] = json.dumps(compat).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, data in reversed(list(blobs.items())):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    metadata = {
        "Id": root["digest"],
        "Os": "linux",
        "Architecture": "amd64",
        "Config": {},
        "RootFS": {"Layers": expected},
    }
    return output.getvalue(), metadata, config_descriptor["digest"]


@pytest.fixture
def scan(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "docker/release_image.py"
    spec = importlib.util.spec_from_file_location("release_saved_test", path)
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)

    def run(**kwargs):
        data, metadata, config_digest = fixture_archive(**kwargs)
        monkeypatch.setattr(
            release.subprocess,
            "Popen",
            lambda *args, **kwargs: SimpleNamespace(stdout=io.BytesIO(data), poll=lambda: 0, wait=lambda: 0),
        )
        return release.scan_saved_image(metadata), metadata, config_digest

    return run


@pytest.mark.parametrize("identity", ["config", "manifest", "index", "nested-index"])
@pytest.mark.parametrize("encoding", ["none", "gzip", "zstd"])
def test_configuration_manifest_and_index_ids_are_bound_to_layer_bytes(scan, identity, encoding):
    result, metadata, config_digest = scan(identity=identity, encoding=encoding)
    assert result["status"] == "pass"
    assert result["configuration_digest"] == config_digest
    assert result["descriptor_chain"][0] == metadata["Id"]
    assert result["descriptor_chain"][-1] == config_digest
    assert [layer["diff_id"] for layer in result["layers"]] == metadata["RootFS"]["Layers"]


def test_legacy_classic_archive(scan):
    result, _, config_digest = scan(identity="config", encoding="none", legacy=True)
    assert result["status"] == "pass" and result["configuration_digest"] == config_digest


@pytest.mark.parametrize("encoding", ["none", "gzip", "zstd"])
@pytest.mark.parametrize(
    "path,kind",
    [
        (".git", tarfile.REGTYPE),
        ("opt/pytorch/pytorch/.git", tarfile.REGTYPE),
        ("workspace/project/.git", tarfile.DIRTYPE),
        ("workspace/project/.git", tarfile.SYMTYPE),
        ("workspace/project/.git", tarfile.LNKTYPE),
        ("opt/pytorch/pytorch/.git/config", tarfile.REGTYPE),
        ("opt/pytorch/fuser/.CI_CONFIG_DO_NOT_PUSH_THIS_FILE_TO_PUBLIC.yaml", tarfile.REGTYPE),
    ],
)
def test_repository_metadata_is_rejected_from_distributed_layers(scan, encoding, path, kind):
    result, _, _ = scan(encoding=encoding, payload_path=path, payload_type=kind)
    assert result["status"] == "fail"
    assert any(path in offender for offender in result["offenders"])


@pytest.mark.parametrize("path", ["workspace/project/.gitignore", "workspace/project/.gitmodules"])
def test_git_source_configuration_is_allowed(scan, path):
    result, _, _ = scan(payload_path=path)
    assert result["status"] == "pass"


@pytest.mark.parametrize(
    "defect",
    [
        "config-hash",
        "missing-config",
        "config-size",
        "config-type",
        "layer-size",
        "layer-hash",
        "extra-layer",
        "diff-id",
        "platform",
        "runtime-config",
        "ambiguous",
        "compat-config",
        "compat-layers",
    ],
)
def test_broken_descriptor_chains_and_payloads_are_rejected(scan, defect):
    with pytest.raises((ValueError, OSError, EOFError, tarfile.TarError)):
        scan(identity="index", encoding="none", defect=defect)


@pytest.mark.parametrize("encoding", ["none", "gzip", "zstd"])
def test_manifest_identity_does_not_hide_runtime_masked_codecs(scan, encoding):
    result, _, _ = scan(identity="index", encoding=encoding, masked_codec=True)
    assert result["status"] == "fail"
    assert any("dev/libavcodec.so" in name for name in result["offenders"])


@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
def test_malformed_compression_has_a_controlled_failure(scan, encoding):
    with pytest.raises(ValueError, match="Malformed compressed"):
        scan(identity="manifest", encoding=encoding, defect="compressed-payload")


@pytest.mark.parametrize("other_codec", [False, True])
def test_index_with_default_only_docker_manifest_checks_both_platform_payloads(scan, other_codec):
    result, _, _ = scan(identity="index", other_platform=True, other_codec=other_codec)
    assert result["status"] == ("fail" if other_codec else "pass")
    assert len(result["additional_layers"]) == 1
    if other_codec:
        assert any("sys/libavcodec.so" in item for item in result["offenders"])


def test_other_platform_diff_id_must_match_its_authenticated_config(scan):
    with pytest.raises(ValueError, match="payload hashes"):
        scan(identity="index", other_platform=True, defect="other-diff-id")
