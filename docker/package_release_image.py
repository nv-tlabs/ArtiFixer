# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split a sanitized filesystem into bounded OCI layers without changing files."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import tarfile
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import release_image

OCI = "application/vnd.oci.image."
MAX_REGISTRY_BLOB_BYTES = 10_000_000_000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def blob_path(layout, descriptor):
    identifier = descriptor["digest"]
    if not identifier.startswith("sha256:") or len(identifier) != 71:
        raise ValueError("Only SHA-256 OCI blobs are supported")
    if any(char not in "0123456789abcdef" for char in identifier[7:]):
        raise ValueError("Invalid OCI blob digest")
    path = Path(layout) / "blobs/sha256" / identifier[7:]
    if path.stat().st_size != descriptor["size"]:
        raise ValueError("OCI blob size mismatch")
    return path


def json_blob(layout, descriptor):
    path = blob_path(layout, descriptor)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Oversized OCI JSON blob")
    raw = path.read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != descriptor["digest"]:
        raise ValueError("OCI JSON digest mismatch")
    return raw, json.loads(raw)


def write_json_blob(layout, value, media_type):
    raw = json.dumps(value, separators=(",", ":")).encode()
    identifier = hashlib.sha256(raw).hexdigest()
    with (layout / "blobs/sha256" / identifier).open("xb") as handle:
        handle.write(raw)
    return {"mediaType": media_type, "digest": "sha256:" + identifier, "size": len(raw)}


def entry_record(member, content_digest=None):
    path = release_image.member_path(member.name)
    if Path(path).name.startswith(".wh."):
        raise ValueError(f"Whiteouts are not permitted in a flattened filesystem: {path}")
    kinds = [
        (member.isdir(), "directory"),
        (member.islnk(), "hardlink"),
        (member.issym(), "symlink"),
        (member.isfile(), "file"),
        (member.ischr(), "character"),
        (member.isblk(), "block"),
        (member.isfifo(), "fifo"),
    ]
    kind = next((name for matches, name in kinds if matches), None)
    if kind is None or member.sparse is not None:
        raise ValueError(f"Unsupported filesystem entry: {path}")
    if kind != "file" and member.size:
        raise ValueError(f"Non-file entry has a payload: {path}")
    # atime/ctime are not stable under reading or container creation. All file
    # bytes, mtime, ownership, permissions, links, devices and xattrs are bound.
    header_fields = {
        "path",
        "linkpath",
        "mtime",
        "atime",
        "ctime",
        "uid",
        "gid",
        "uname",
        "gname",
        "size",
        "hdrcharset",
    }
    return {
        "path": path,
        "kind": kind,
        "mode": member.mode,
        "uid": member.uid,
        "gid": member.gid,
        "mtime": str(Decimal(str(member.pax_headers.get("mtime", member.mtime))).normalize()),
        "size": member.size,
        "linkname": member.linkname,
        "device": [member.devmajor, member.devminor],
        "xattrs": {key: value for key, value in member.pax_headers.items() if key not in header_fields},
        "content_sha256": content_digest,
    }


def fingerprint(records):
    digest = hashlib.sha256()
    for path in sorted(records):
        digest.update(json.dumps(records[path], sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def tar_records(stream):
    records = {}
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            value = entry_record(member)
            if value["path"] in records:
                raise ValueError("Repeated filesystem path")
            if member.isfile():
                payload = release_image.HashReader(archive.extractfile(member))
                while payload.read(1024 * 1024):
                    pass
                value["content_sha256"] = "sha256:" + payload.digest.hexdigest()
            records[value["path"]] = value
    return records


@contextmanager
def decoded_blob(layout, descriptor, expected_diff_id):
    path = blob_path(layout, descriptor)
    with path.open("rb") as handle:
        packed = release_image.HashReader(handle)
        media = descriptor["mediaType"]
        if media == OCI + "layer.v1.tar+gzip":
            decoded = gzip.GzipFile(fileobj=packed, mode="rb")
        elif media == OCI + "layer.v1.tar":
            decoded = packed
        else:
            raise ValueError("Source layers must be uncompressed or gzip OCI layers")
        raw = release_image.HashReader(decoded)
        try:
            yield raw
            while raw.read(1024 * 1024):
                pass
            while packed.read(1024 * 1024):
                pass
            if "sha256:" + raw.digest.hexdigest() != expected_diff_id:
                raise ValueError("Uncompressed filesystem layer digest mismatch")
            if "sha256:" + packed.digest.hexdigest() != descriptor["digest"]:
                raise ValueError("Compressed filesystem layer digest mismatch")
        finally:
            if decoded is not packed:
                decoded.close()


class HashWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, block):
        result = self.stream.write(block)
        if result != len(block):
            raise OSError("Short filesystem layer write")
        self.digest.update(block)
        return result

    def flush(self):
        self.stream.flush()


class Segments:
    def __init__(self, destination, maximum):
        self.destination, self.maximum = destination, maximum
        self.layers = []
        self.archive = None

    def start(self):
        self.partial = self.destination / f"segment-{len(self.layers):04d}.partial"
        self.output = self.partial.open("xb")
        if shutil.which("pigz"):
            self.process = subprocess.Popen(
                ["pigz", "-n", "-1", "-p", "4", "-c"], stdin=subprocess.PIPE, stdout=self.output
            )
            self.compressor = self.process.stdin
        else:
            self.process = None
            self.compressor = gzip.GzipFile(filename="", fileobj=self.output, mode="wb", compresslevel=1, mtime=0)
        self.writer = HashWriter(self.compressor)
        self.archive = tarfile.open(fileobj=self.writer, mode="w|", format=tarfile.PAX_FORMAT, copybufsize=1024**2)

    def reserve(self, size, name):
        if size + tarfile.RECORDSIZE > self.maximum:
            raise ValueError(f"Filesystem entry/hardlink group exceeds the segment limit: {name}")
        if self.archive and self.archive.offset + size + tarfile.RECORDSIZE > self.maximum:
            self.finish()
        if self.archive is None:
            self.start()

    def add(self, member, payload=None):
        size = len(member.tobuf(tarfile.PAX_FORMAT)) + (member.size + 511) // 512 * 512
        self.reserve(size, member.name)
        self.archive.addfile(member, payload)

    def finish(self):
        if self.archive is None:
            return
        self.archive.close()
        self.compressor.close()
        if self.process and self.process.wait() != 0:
            raise RuntimeError("Layer compression failed")
        self.output.close()
        if self.partial.stat().st_size >= MAX_REGISTRY_BLOB_BYTES:
            raise ValueError("Compressed layer exceeds GHCR's layer-size limit")
        identifier = sha256_file(self.partial)
        target = self.destination / "blobs/sha256" / identifier[7:]
        if target.exists():
            raise ValueError("Unexpected duplicate output layer")
        self.partial.rename(target)
        self.layers.append(
            {
                "mediaType": OCI + "layer.v1.tar+gzip",
                "digest": identifier,
                "size": target.stat().st_size,
                "diff_id": "sha256:" + self.writer.digest.hexdigest(),
            }
        )
        self.archive = None

    def abort(self):
        if self.archive is None:
            return
        try:
            self.archive.close()
        except OSError:
            pass
        try:
            self.compressor.close()
        except OSError:
            pass
        if self.process:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.output.close()
        self.archive = None


def inspect_layout(layout):
    index = json.loads((Path(layout) / "index.json").read_text())
    # regctl import can retain both an untagged descriptor and a named alias.
    # Ignore annotation differences, but never mask conflicting blob metadata.
    descriptors = {(item["digest"], item["size"], item["mediaType"]): item for item in index.get("manifests", [])}
    if len(descriptors) != 1:
        raise ValueError("Package one platform/image at a time")
    _, manifest = json_blob(layout, next(iter(descriptors.values())))
    if manifest["mediaType"] != OCI + "manifest.v1+json":
        raise ValueError("Source must contain an OCI image manifest")
    raw_config, config = json_blob(layout, manifest["config"])
    return manifest, raw_config, config


def package(source, destination, maximum=4_000_000_000, tag="release"):
    source, destination = Path(source), Path(destination)
    if not 64 * 1024 <= maximum <= 9_000_000_000:
        raise ValueError("Segment size must be between 64 KiB and 9 GB")
    manifest, raw_config, config = inspect_layout(source)
    diff_ids = config["rootfs"]["diff_ids"]
    if (
        not diff_ids
        or diff_ids[0] == release_image.EMPTY_LAYER
        or any(x != release_image.EMPTY_LAYER for x in diff_ids[1:])
    ):
        raise ValueError("Source must be one flattened filesystem payload plus canonical empty layers")
    if len(manifest["layers"]) != len(diff_ids):
        raise ValueError("Source manifest/config layer counts disagree")
    release_image.assert_exportable({"Os": config["os"], "Config": config.get("config", {})})
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "blobs/sha256").mkdir(parents=True)
    output = Segments(destination, maximum)
    try:
        return package_contents(source, destination, maximum, tag, manifest, raw_config, config, diff_ids, output)
    finally:
        output.abort()


def hardlink_groups(source, manifest, diff_ids):
    kinds, links = {}, {}
    for descriptor, diff_id in zip(manifest["layers"], diff_ids, strict=True):
        with decoded_blob(source, descriptor, diff_id) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    value = entry_record(member)
                    path = value["path"]
                    if path in kinds:
                        raise ValueError(f"Repeated source filesystem path: {path}")
                    kinds[path] = value["kind"]
                    if member.islnk():
                        links[path] = copy.deepcopy(member)
    groups = {}
    for path, member in links.items():
        target = release_image.member_path(member.linkname)
        visited = {path}
        while target in links:
            if target in visited:
                raise ValueError("Cyclic filesystem hardlinks")
            visited.add(target)
            target = release_image.member_path(links[target].linkname)
        if kinds.get(target) != "file":
            raise ValueError(f"Hardlink has no regular-file target: {path}")
        groups.setdefault(target, []).append(member)
    # Docker's classic image loader requires the target in the same layer.
    # Order aliases so chained links always follow their target.
    for target, pending in groups.items():
        known, ordered = {target}, []
        while pending:
            eligible = [member for member in pending if release_image.member_path(member.linkname) in known]
            if not eligible:
                raise ValueError("Unresolvable hardlink group")
            for member in eligible:
                known.add(release_image.member_path(member.name))
                ordered.append(member)
                pending.remove(member)
        groups[target] = ordered
    return groups


def package_contents(source, destination, maximum, tag, manifest, raw_config, config, diff_ids, output):
    groups = hardlink_groups(source, manifest, diff_ids)
    records, directories = {}, []
    for descriptor, diff_id in zip(manifest["layers"], diff_ids, strict=True):
        with decoded_blob(source, descriptor, diff_id) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for original in archive:
                    member = copy.deepcopy(original)
                    value = entry_record(member)
                    if value["path"] in records:
                        raise ValueError(f"Repeated source filesystem path: {value['path']}")
                    if member.isdir():
                        directories.append(member)
                    elif member.islnk():
                        pass  # Emitted together with its regular-file target.
                    elif member.isfile():
                        aliases = groups.get(value["path"], [])
                        size = len(member.tobuf(tarfile.PAX_FORMAT)) + (member.size + 511) // 512 * 512
                        size += sum(len(alias.tobuf(tarfile.PAX_FORMAT)) for alias in aliases)
                        output.reserve(size, member.name)
                        payload = release_image.HashReader(archive.extractfile(original))
                        output.add(member, payload)
                        for alias in aliases:
                            output.add(alias)
                        value["content_sha256"] = "sha256:" + payload.digest.hexdigest()
                    else:
                        output.add(member)
                    records[value["path"]] = value
    # Restore all directory metadata after child creation in earlier layers.
    output.finish()
    if sum(len(member.tobuf(tarfile.PAX_FORMAT)) for member in directories) + tarfile.RECORDSIZE > maximum:
        raise ValueError("Directory metadata must fit in one final segment")
    for member in directories:
        output.add(member)
    output.finish()
    new_config = copy.deepcopy(config)
    new_config["rootfs"]["diff_ids"] = [layer["diff_id"] for layer in output.layers]
    new_config["history"] = [
        dict(
            created_by=f"ArtiFixer sanitized filesystem segment {i + 1}",
            **({"created": config["created"]} if "created" in config else {}),
        )
        for i in range(len(output.layers))
    ]
    if release_image.config_identity(config.get("config", {})) != release_image.config_identity(
        new_config.get("config", {})
    ):
        raise ValueError("Runtime configuration changed during packaging")
    cfg = write_json_blob(destination, new_config, OCI + "config.v1+json")
    image = {
        "schemaVersion": 2,
        "mediaType": OCI + "manifest.v1+json",
        "config": cfg,
        "layers": [{key: layer[key] for key in ["mediaType", "digest", "size"]} for layer in output.layers],
    }
    desc = write_json_blob(destination, image, image["mediaType"])
    index = {
        "schemaVersion": 2,
        "manifests": [
            desc
            | {
                "platform": {"os": config["os"], "architecture": config["architecture"]},
                "annotations": {"org.opencontainers.image.ref.name": tag},
            }
        ],
    }
    (destination / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}\n')
    (destination / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    assembled = {}
    for descriptor in output.layers:
        with decoded_blob(destination, descriptor, descriptor["diff_id"]) as stream:
            segment = tar_records(stream)
        if set(assembled) & set(segment):
            raise ValueError("Output segments replace existing paths")
        assembled.update(segment)
    if fingerprint(records) != fingerprint(assembled):
        raise ValueError("Output filesystem content or metadata differs from the source")
    metadata = {
        "Id": desc["digest"],
        "Descriptor": desc,
        "Os": config["os"],
        "Architecture": config["architecture"],
        "Config": new_config.get("config", {}),
        "RootFS": {"Layers": new_config["rootfs"]["diff_ids"]},
    }
    process = subprocess.Popen(
        ["tar", "-C", str(destination), "-cf", "-", "oci-layout", "index.json", "blobs"], stdout=subprocess.PIPE
    )
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            gate = release_image.scan_saved_archive(archive, metadata, runpy.run_path(str(release_image.GATE)))
        while process.stdout.read(1024 * 1024):
            pass
        if process.wait() != 0 or gate["status"] != "pass":
            raise ValueError("Packaged image failed saved-payload verification")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    return {
        "status": "pass",
        "source_config_digest": "sha256:" + hashlib.sha256(raw_config).hexdigest(),
        "config_digest": cfg["digest"],
        "manifest_digest": desc["digest"],
        "manifest": desc,
        "platform": config["os"] + "/" + config["architecture"],
        "tag": tag,
        "layers": output.layers,
        "filesystem_paths": len(records),
        "filesystem_manifest_sha256": fingerprint(records),
        "runtime_configuration_unchanged": True,
        "filesystem_contents_and_metadata_unchanged": True,
        "saved_payload": gate,
        "maximum_layer_bytes": maximum,
        "publication_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local OCI layout of a flattened image")
    parser.add_argument("destination", type=Path, help="New local OCI layout directory")
    parser.add_argument("--max-layer-bytes", type=int, default=4_000_000_000)
    parser.add_argument("--tag", default="release")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Preserve existing reports; choose a new path")
    result = package(args.source, args.destination, args.max_layer_bytes, args.tag)
    with args.report.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ["status", "config_digest", "manifest_digest", "filesystem_paths"]}))


if __name__ == "__main__":
    main()
