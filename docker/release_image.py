# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify or flatten a local image; never push it to a registry."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
import zipfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "tests/assert_no_ffmpeg_in_image.py"
CONFIG_FIELDS = ("Env", "Labels", "Entrypoint", "Cmd", "WorkingDir", "User", "ExposedPorts", "StopSignal", "Shell")
# Docker may append this exact empty tar when applying metadata to an imported
# rootfs. Its content hash proves that it cannot conceal an older file.
EMPTY_LAYER = "sha256:" + hashlib.sha256(bytes(1024)).hexdigest()
SENSITIVE_VALUES = set()
CONTEXT = {}


def sanitize(value) -> str:
    text = value.decode(errors="replace") if isinstance(value, bytes) else str(value or "")
    for secret in sorted(SENSITIVE_VALUES, key=len, reverse=True):
        for spelling in (
            secret,
            json.dumps(secret)[1:-1],
            quote(secret) if not any(c in secret for c in "\r\n\0") else secret,
        ):
            if len(spelling) < 4:
                text = re.sub(r"(?<!\w)" + re.escape(spelling) + r"(?!\w)", "[redacted]", text)
            else:
                text = text.replace(spelling, "[redacted]")
    text = re.sub(r"(?im)^.*\b(?:ENV|LABEL)\s+.*$", "[redacted image metadata instruction]", text)
    text = re.sub(r"(?i)(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", text)
    text = re.sub(r"(?i)(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", text)
    text = re.sub(r"(?i)(bearer\s+|(?:token|password|secret|authorization)[=:]\s*)\S+", r"\1[redacted]", text)
    return text[-4000:]


class OperationError(RuntimeError):
    def __init__(self, operation, returncode, diagnostic):
        super().__init__(f"{operation} failed (exit {returncode})")
        self.details = {
            "operation": operation,
            "exit_status": returncode,
            "diagnostic": sanitize(diagnostic),
            **CONTEXT,
        }


def docker(*args: str, **kwargs) -> subprocess.CompletedProcess:
    if not any(key in kwargs for key in ("capture_output", "stdout", "stderr")):
        kwargs["capture_output"] = True
    try:
        return subprocess.run(["docker", *args], check=True, **kwargs)
    except subprocess.CalledProcessError as exc:
        operation = "docker " + " ".join(args[:2] if args[0] in ("image", "container") else args[:1])
        raise OperationError(operation, exc.returncode, exc.stderr or exc.stdout) from None
    except OSError as exc:
        raise OperationError("docker " + args[0], None, exc.strerror) from None


def inspect_image(image: str, *, expected_platform: str | None = None) -> dict:
    CONTEXT["image_reference"] = image
    CONTEXT.pop("image_id", None)
    root = json.loads(docker("image", "inspect", image, capture_output=True, text=True).stdout)[0]
    metadata = root
    if expected_platform or root.get("Descriptor"):
        metadata = json.loads(
            docker(
                "image",
                "inspect",
                "--platform",
                expected_platform or platform(root),
                root["Id"],
                capture_output=True,
                text=True,
            ).stdout
        )[0]
    # A containerd child manifest returned by platform inspection need not be
    # directly runnable/taggable. Keep the immutable addressable root as Id and
    # retain the selected manifest separately; all runtime operations specify
    # its platform. Never resolve the mutable input tag a second time.
    metadata = metadata | {
        "PlatformId": metadata["Id"],
        "PlatformDescriptor": metadata.get("Descriptor"),
        "Id": root["Id"],
        "Descriptor": root.get("Descriptor"),
    }
    CONTEXT["image_id"] = metadata["Id"]
    config = metadata["Config"]
    SENSITIVE_VALUES.update(entry.partition("=")[2] for entry in config.get("Env") or [] if entry.partition("=")[2])
    SENSITIVE_VALUES.update(value for value in (config.get("Labels") or {}).values() if value)
    return metadata


def lock_path(destination: str) -> Path:
    # One lock namespace per daemon, shared by all cooperating release writers.
    daemon = docker("info", "--format", "{{.ID}}", capture_output=True, text=True).stdout.strip()
    if not daemon:
        raise ValueError("Docker daemon did not supply its identity")
    tag = destination
    if ":" not in tag.rsplit("/", 1)[-1]:
        tag += ":latest"
    if "/" not in tag:
        tag = "docker.io/library/" + tag
    elif "." not in tag.split("/", 1)[0] and ":" not in tag.split("/", 1)[0] and not tag.startswith("localhost/"):
        tag = "docker.io/" + tag
    elif tag.startswith("index.docker.io/"):
        tag = tag.replace("index.docker.io/", "docker.io/", 1)
    if tag.startswith("docker.io/") and tag.count("/") == 1:
        tag = tag.replace("docker.io/", "docker.io/library/", 1)
    directory = Path(os.environ.get("ARTIFIXER_RELEASE_LOCK_DIR", f"/tmp/artifixer-release-locks-{os.getuid()}"))
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory / (hashlib.sha256(f"{daemon}\n{tag}".encode()).hexdigest() + ".lock")


@contextmanager
def destination_lock(destination):
    # Do not unlink lock files: waiters may still hold the inode.
    with lock_path(destination).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


class HashReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.source.read(size)
        self.digest.update(data)
        return data


def scan_layer(stream, predicates):
    """Inspect tar members without extracting or applying runtime mount masks."""
    offenders = []
    count = 0
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            count += 1
            name = member.name.rstrip("/").rsplit("/", 1)[-1]
            if ".git" in PurePosixPath(member.name).parts or name == ".CI_CONFIG_DO_NOT_PUSH_THIS_FILE_TO_PUBLIC.yaml":
                offenders.append(member.name)
            if (
                member.isfile()
                and (
                    "/.cache/pip/" in "/" + member.name
                    or member.name.lstrip("./").startswith("var/cache/apt/archives/")
                )
                and name != "lock"
            ):
                offenders.append(member.name)
            if predicates["matches"](name):
                if member.issym() or member.islnk():
                    offenders.append(member.name)
                elif member.isfile():
                    head = archive.extractfile(member).read(8)
                    if (
                        head.startswith(b"\x7fELF")
                        or head == b"!<arch>\n"
                        or (name.lower() in predicates["EXACT_BINARY_NAMES"] and member.mode & 0o111)
                    ):
                        offenders.append(member.name)
            elif member.isfile() and name.endswith(".whl"):
                with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as wheel_file:
                    shutil.copyfileobj(archive.extractfile(member), wheel_file)
                    wheel_file.seek(0)
                    with zipfile.ZipFile(wheel_file) as wheel:
                        offenders.extend(
                            f"{member.name}!{item}"
                            for item in wheel.namelist()
                            if predicates["matches"](item.rsplit("/", 1)[-1])
                        )
    return {"members": count, "offenders": offenders}


CONFIG_TYPES = {"application/vnd.oci.image.config.v1+json", "application/vnd.docker.container.image.v1+json"}
MANIFEST_TYPES = {"application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"}
INDEX_TYPES = {"application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json"}
LAYER_TYPES = {
    "application/vnd.oci.image.layer.v1.tar": "none",
    "application/vnd.oci.image.layer.v1.tar+gzip": "gzip",
    "application/vnd.oci.image.layer.v1.tar+zstd": "zstd",
    "application/vnd.docker.image.rootfs.diff.tar": "none",
    "application/vnd.docker.image.rootfs.diff.tar.gzip": "gzip",
}


def read_saved_layer(stream, predicates):
    """Hash distributed bytes and inspect/hash the decompressed filesystem tar."""
    prefix = stream.peek(4)[:4]
    encoding = "gzip" if prefix[:2] == b"\x1f\x8b" else "zstd" if prefix == b"\x28\xb5\x2f\xfd" else "none"
    packed = HashReader(stream)
    decoding_errors = (gzip.BadGzipFile, EOFError)
    with ExitStack() as stack:
        decoded = packed
        if encoding == "gzip":
            decoded = stack.enter_context(gzip.GzipFile(fileobj=packed))
        elif encoding == "zstd":
            try:
                import zstandard
            except ImportError:
                raise ValueError("Install zstandard to verify zstd-compressed saved layers") from None
            decoded = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(packed, closefd=False))
            decoding_errors += (zstandard.ZstdError,)
        unpacked = HashReader(decoded)
        try:
            scanned = scan_layer(unpacked, predicates)
            while unpacked.read(1024 * 1024):
                pass
        except decoding_errors:
            raise ValueError("Malformed compressed saved layer") from None
    while packed.read(1024 * 1024):
        pass
    return scanned | {
        "diff_id": "sha256:" + unpacked.digest.hexdigest(),
        "blob_digest": "sha256:" + packed.digest.hexdigest(),
        "encoding": encoding,
    }


def scan_saved_archive(archive, metadata, predicates):
    # Metadata can precede or follow layers in a streaming export. Retain only
    # bounded JSON records; scan/hash layer blobs as they pass, without extraction.
    documents, layers, paths = {}, {}, set()
    compat = None
    metadata_bytes = 0
    for member in archive:
        if not member.isfile():
            continue
        name = member.name
        if name in paths:
            raise ValueError("Duplicate saved image archive path")
        paths.add(name)
        stream = archive.extractfile(member)
        blob = re.fullmatch(r"blobs/sha256/([a-f0-9]{64})|([a-f0-9]{64})\.json", name)
        if name != "manifest.json" and not blob and not name.endswith("/layer.tar"):
            continue
        prefix = stream.peek(512)[:512]
        is_layer = name.endswith("/layer.tar") or prefix[:2] == b"\x1f\x8b" or prefix[:4] == b"\x28\xb5\x2f\xfd"
        if not is_layer and blob:
            try:
                tarfile.TarInfo.frombuf(prefix, "utf-8", "surrogateescape")
                is_layer = True
            except tarfile.EOFHeaderError:
                is_layer = True  # canonical empty tar
            except tarfile.HeaderError:
                pass
        if is_layer:
            item = read_saved_layer(stream, predicates) | {"size": member.size, "path": name}
            if blob and item["blob_digest"] != "sha256:" + (blob[1] or blob[2]):
                raise ValueError("Saved layer blob digest mismatch")
            layers[name] = item
        else:
            if member.size > 16 * 1024 * 1024 or metadata_bytes + member.size > 64 * 1024 * 1024:
                raise ValueError("Saved image metadata exceeds verification limits")
            raw = stream.read()
            metadata_bytes += len(raw)
            identifier = "sha256:" + hashlib.sha256(raw).hexdigest()
            if blob and identifier != "sha256:" + (blob[1] or blob[2]):
                raise ValueError("Saved image metadata digest mismatch")
            value = json.loads(raw)
            if name == "manifest.json":
                compat = value
            else:
                if not isinstance(value, dict):
                    raise ValueError("Saved image metadata must be a JSON object")
                documents[identifier] = {"document": value, "path": name, "size": len(raw)}

    def descriptor(record, types, *, required=True):
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("mediaType"), str)
            or record["mediaType"] not in types
        ):
            raise ValueError("Unsupported saved image descriptor type")
        identifier = record.get("digest", "")
        if not isinstance(identifier, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", identifier):
            raise ValueError("Invalid saved image descriptor digest")
        if type(record.get("size")) is not int or record["size"] < 0:
            raise ValueError("Invalid saved image descriptor size")
        item = documents.get(identifier)
        if item is None and not required:
            return None
        if item is None or record["size"] != item["size"]:
            raise ValueError("Saved image descriptor blob missing or size mismatch")
        if (
            record["mediaType"] in MANIFEST_TYPES | INDEX_TYPES
            and item["document"].get("mediaType") != record["mediaType"]
        ):
            raise ValueError("Saved image descriptor and document media types disagree")
        return identifier

    def matches_platform(record):
        if not isinstance(record, dict):
            raise ValueError("Malformed platform in saved image metadata")
        return all(
            record.get(key, "") == metadata.get(field, "")
            for key, field in (("os", "Os"), ("architecture", "Architecture"))
        ) and (not record.get("variant") or not metadata.get("Variant") or record["variant"] == metadata["Variant"])

    def resolve(identifier, chain=()):
        if identifier in chain or len(chain) >= 16 or identifier not in documents:
            raise ValueError("Saved image descriptor chain is missing, cyclic or too deep")
        chain = (*chain, identifier)
        value = documents[identifier]["document"]
        if "rootfs" in value:
            return [(identifier, None, chain)] if value.get("os") == "linux" else []
        if value.get("schemaVersion") != 2:
            raise ValueError("Unsupported saved image manifest schema")
        if value.get("mediaType") in MANIFEST_TYPES:
            config_id = descriptor(value.get("config"), CONFIG_TYPES)
            return (
                [(config_id, value, (*chain, config_id))]
                if documents[config_id]["document"].get("os") == "linux"
                else []
            )
        if value.get("mediaType") in INDEX_TYPES:
            candidates = []
            children = value.get("manifests")
            if not isinstance(children, list) or any(not isinstance(child, dict) for child in children):
                raise ValueError("Malformed saved image index")
            for child in children:
                required = not child.get("platform") or matches_platform(child["platform"])
                child_id = descriptor(child, MANIFEST_TYPES | INDEX_TYPES, required=required)
                if child_id is None:
                    continue  # other architectures may be referenced but not cached
                descendants = resolve(child_id, chain)
                for config_id, _, _ in descendants:
                    config = documents[config_id]["document"]
                    if child.get("platform") and not platform_matches(
                        "/".join(
                            child["platform"][key]
                            for key in ("os", "architecture", "variant")
                            if child["platform"].get(key)
                        ),
                        "/".join(config[key] for key in ("os", "architecture", "variant") if config.get(key)),
                    ):
                        raise ValueError("Saved index platform disagrees with its configuration")
                candidates.extend(descendants)
            return candidates
        raise ValueError("Unsupported saved image manifest media type")

    if metadata.get("Descriptor"):
        if descriptor(metadata["Descriptor"], CONFIG_TYPES | MANIFEST_TYPES | INDEX_TYPES) != metadata["Id"]:
            raise ValueError("Inspected image ID and descriptor disagree")
    images = resolve(metadata["Id"])
    candidates = [
        item
        for item in images
        if matches_platform(documents[item[0]]["document"]) and metadata.get("PlatformId", metadata["Id"]) in item[2]
    ]
    if len(candidates) != 1:
        raise ValueError("Saved image must resolve to exactly one configuration for the inspected platform")
    config_id, manifest, chain = candidates[0]
    if metadata.get("PlatformDescriptor"):
        if (
            descriptor(metadata["PlatformDescriptor"], CONFIG_TYPES | MANIFEST_TYPES | INDEX_TYPES)
            != metadata["PlatformId"]
        ):
            raise ValueError("Selected platform ID and descriptor disagree")
    if metadata.get("PlatformId", metadata["Id"]) not in chain:
        raise ValueError("Saved descriptor chain does not contain the inspected platform image")
    config = documents[config_id]["document"]
    if not isinstance(config.get("rootfs"), dict) or config["rootfs"].get("diff_ids") != metadata["RootFS"]["Layers"]:
        raise ValueError("Saved image configuration does not match inspected layer identities")
    if config_identity(config.get("config", {})) != config_identity(metadata["Config"]):
        raise ValueError("Saved image configuration does not match inspected runtime settings")
    if compat is not None and (not isinstance(compat, list) or any(not isinstance(item, dict) for item in compat)):
        raise ValueError("Malformed saved Docker manifest")
    bound_paths = set()
    selected_paths = None
    config_paths = {documents[item[0]]["path"] for item in images}
    # containerd's compatibility manifest may describe only its default
    # platform, even though the OCI index exports several cached platforms.
    if compat is not None and not {item.get("Config") for item in compat} <= config_paths:
        raise ValueError("Saved Docker manifest does not identify the authenticated configurations")
    for image_config_id, image_manifest, _ in images:
        image_config = documents[image_config_id]["document"]
        image_config_path = documents[image_config_id]["path"]
        matching = [item for item in compat or [] if item.get("Config") == image_config_path]
        if len(matching) > 1 or (not image_manifest and len(matching) != 1):
            raise ValueError("Saved Docker manifest has ambiguous configuration entries")
        image_paths = matching[0].get("Layers") if matching else None
        if image_manifest:
            descriptors = image_manifest.get("layers")
            if not isinstance(descriptors, list) or any(not isinstance(item, dict) for item in descriptors):
                raise ValueError("Malformed saved layer descriptors")
            expected_paths = []
            for item in descriptors:
                if not isinstance(item.get("digest"), str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", item["digest"]):
                    raise ValueError("Invalid saved layer descriptor digest")
                path = "blobs/sha256/" + item["digest"][7:]
                layer = layers.get(path)
                if (
                    layer is None
                    or item["digest"] != layer["blob_digest"]
                    or type(item.get("size")) is not int
                    or item["size"] != layer["size"]
                    or not isinstance(item.get("mediaType"), str)
                    or item["mediaType"] not in LAYER_TYPES
                    or LAYER_TYPES[item["mediaType"]] != layer["encoding"]
                ):
                    raise ValueError("Saved layer descriptor does not match distributed bytes")
                expected_paths.append(path)
            if image_paths is not None and image_paths != expected_paths:
                raise ValueError("Saved Docker and OCI layer manifests disagree")
            image_paths = expected_paths
        if not isinstance(image_paths, list) or any(
            not isinstance(path, str) or path not in layers for path in image_paths
        ):
            raise ValueError("Saved image has no complete layer manifest")
        if not isinstance(image_config.get("rootfs"), dict) or [
            layers[path]["diff_id"] for path in image_paths
        ] != image_config["rootfs"].get("diff_ids"):
            raise ValueError("Saved layer payload hashes do not match their configuration")
        bound_paths.update(image_paths)
        if image_config_id == config_id:
            selected_paths = image_paths
    if bound_paths != set(layers):
        raise ValueError("Saved archive contains unauthenticated filesystem layers")
    offenders = [f"{path}:{name}" for path in sorted(layers) for name in layers[path]["offenders"]]
    return {
        "status": "fail" if offenders else "pass",
        "scope": "all saved filesystem layer bytes, including /dev, /proc and /sys; known codec names, wheels and repository metadata",
        "configuration_digest": config_id,
        "descriptor_chain": list(chain),
        "layers": [layers[path] for path in selected_paths],
        "additional_layers": [layers[path] for path in sorted(bound_paths - set(selected_paths))],
        "offenders": offenders,
    }


def scan_saved_image(metadata):
    """Stream the full root: filtering save by platform discards its index proof."""
    predicates = runpy.run_path(str(GATE))
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ["docker", "image", "save", metadata["Id"]],
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                result = scan_saved_archive(archive, metadata, predicates)
            while process.stdout.read(1024 * 1024):
                pass
            status = process.wait()
            if status:
                errors.seek(0)
                raise OperationError("docker image save", status, errors.read())
            return result
        except tarfile.TarError:
            if process.poll() is None:
                process.terminate()
            status = process.wait()
            errors.seek(0)
            diagnostic = errors.read()
            if diagnostic:
                raise OperationError("docker image save", status, diagnostic) from None
            raise ValueError("Saved image archive is incomplete or malformed") from None
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()


def platform(image: dict) -> str:
    return "/".join(str(image[key]) for key in ("Os", "Architecture", "Variant") if image.get(key))


def platform_matches(expected: str, actual: str) -> bool:
    def normalize(value):
        parts = value.split("/")
        if len(parts) == 2 and parts[1] == "arm64":
            parts.append("v8")
        return parts

    wanted, found = normalize(expected), normalize(actual)
    return found[: len(wanted)] == wanted


def assert_exportable(image: dict) -> None:
    if image.get("Os") != "linux":
        raise ValueError("Only Linux images are supported")
    # Volumes are mounted over image content by docker run and omitted by export.
    # Refuse rather than attest bytes hidden from the runtime scanner.
    for field in ("Volumes", "OnBuild", "Healthcheck"):
        if image["Config"].get(field):
            raise ValueError(f"Image declares {field}; this release workflow cannot verify/preserve it safely")


def quote(value: str) -> str:
    if any(char in value for char in "\r\n\0"):
        raise ValueError("Multiline/NUL image configuration is unsupported; refusing a lossy conversion")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'


def config_instructions(config: dict) -> list[str]:
    """Metadata-only Dockerfile instructions; no source commands are executed."""
    lines = []
    for entry in config.get("Env") or []:
        name, separator, value = entry.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("Invalid image environment variable name")
        lines.append(f"ENV {name}={quote(value)}")
    for name, value in sorted((config.get("Labels") or {}).items()):
        lines.append(f"LABEL {quote(name)}={quote(value)}")
    for field, instruction in (("WorkingDir", "WORKDIR"), ("User", "USER"), ("StopSignal", "STOPSIGNAL")):
        if config.get(field):
            lines.append(f"{instruction} {quote(config[field])}")
    for port in sorted(config.get("ExposedPorts") or {}):
        if not re.fullmatch(r"[0-9]+/(tcp|udp|sctp)", port):
            raise ValueError(f"Unsupported exposed port: {port}")
        lines.append(f"EXPOSE {port}")
    for field in ("Shell", "Entrypoint", "Cmd"):
        if config.get(field) is not None:
            lines.append(f"{field.upper()} {json.dumps(config[field])}")
    return lines


def config_identity(config: dict) -> dict:
    return {key: config.get(key) or None for key in CONFIG_FIELDS}


def verify(image: str, *, require_flat: bool = True, expected_platform: str | None = None) -> dict:
    metadata = inspect_image(image, expected_platform=expected_platform)
    assert_exportable(metadata)
    layers = metadata.get("RootFS", {}).get("Layers", [])
    failures = []
    if require_flat and (not layers or layers[0] == EMPTY_LAYER or any(layer != EMPTY_LAYER for layer in layers[1:])):
        failures.append(
            "Release image must have one filesystem payload plus only canonical empty layers; flatten first"
        )
    if expected_platform is not None and not platform_matches(expected_platform, platform(metadata)):
        failures.append(f"Expected {expected_platform}, found {platform(metadata)}")
    saved = scan_saved_image(metadata) if require_flat and not failures else None
    if saved and saved["status"] != "pass":
        failures.append("Saved image payload contains known codec files or package caches")
    gate_source = GATE.read_bytes()
    name = f"artifixer-verify-{uuid.uuid4().hex}"
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--name",
                name,
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--user",
                "0",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--platform",
                platform(metadata),
                "--entrypoint",
                "python",
                "-i",
                metadata["Id"],
                "-",
                "--json",
            ],
            input=gate_source,
            capture_output=True,
            timeout=600,
        )
        try:
            gate = json.loads(result.stdout)
            if not isinstance(gate, dict):
                raise ValueError("Gate result is not an object")
        except (ValueError, UnicodeDecodeError):
            gate = {"status": "unknown", "failures": ["Gate produced no valid JSON result"]}
        if result.returncode != 0 or gate.get("status") != "pass":
            failures.append("FFmpeg filesystem/runtime check did not pass")
        # Keep bounded diagnostic output. Image configuration/environment is never printed.
        diagnostic = sanitize(result.stderr)
    finally:
        subprocess.run(["docker", "container", "rm", "-f", name], capture_output=True, check=False)
    return {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail" if failures else "pass",
        "scope": "flattened release image" if require_flat else "build image final filesystem only",
        "image_id": metadata["Id"],
        "platform_image_id": metadata.get("PlatformId", metadata["Id"]),
        "repo_digests": metadata.get("RepoDigests", []),
        "platform": platform(metadata),
        "size_bytes": metadata["Size"],
        "layers": layers,
        "config_sha256": hashlib.sha256(
            json.dumps(config_identity(metadata["Config"]), sort_keys=True).encode()
        ).hexdigest(),
        "gate_sha256": hashlib.sha256(gate_source).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_revision": (metadata["Config"].get("Labels") or {}).get("org.opencontainers.image.revision"),
        "source_dirty": (metadata["Config"].get("Labels") or {}).get("com.nvidia.artifixer.source_dirty"),
        "gate": gate,
        "saved_payload": saved,
        "gate_stderr": diagnostic,
        "failures": failures,
        "gpu_qualification": "not run",
        "publication_approval": "not assessed",
    }


def destination_absent(destination: str) -> None:
    result = subprocess.run(["docker", "image", "inspect", destination], capture_output=True)
    if result.returncode == 0:
        raise ValueError(f"Destination already exists: {destination}; use a new release tag")
    if b"No such image" not in result.stderr and b"No such object" not in result.stderr:
        raise OperationError("determine destination with docker image inspect", result.returncode, result.stderr)


def flatten(source: str, destination: str, *, expected_platform: str | None = None) -> dict:
    with destination_lock(destination):
        return flatten_locked(source, destination, expected_platform=expected_platform)


def flatten_locked(source: str, destination: str, *, expected_platform: str | None = None) -> dict:
    destination_absent(destination)
    metadata = inspect_image(source, expected_platform=expected_platform)
    assert_exportable(metadata)
    instructions = config_instructions(metadata["Config"])
    before = verify(metadata["Id"], require_flat=False, expected_platform=platform(metadata))
    if before["status"] != "pass":
        return before
    staging = f"artifixer-release-staging:{uuid.uuid4().hex}"
    container = docker(
        "create",
        "--platform",
        platform(metadata),
        "--entrypoint",
        "/bin/true",
        metadata["Id"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        with tempfile.TemporaryFile() as errors:
            exporter = subprocess.Popen(["docker", "export", container], stdout=subprocess.PIPE, stderr=errors)
            try:
                imported = subprocess.run(
                    ["docker", "import", "--platform", platform(metadata), "-", staging],
                    stdin=exporter.stdout,
                    capture_output=True,
                    text=True,
                )
            finally:
                exporter.stdout.close()
            export_status = exporter.wait()
            # A failed importer closes its pipe and can make export fail with
            # SIGPIPE. Preserve the import diagnostic (e.g. storage exhaustion)
            # instead of reporting only that secondary broken pipe.
            if imported.returncode:
                raise OperationError("docker import", imported.returncode, imported.stderr)
            if export_status:
                errors.seek(0)
                raise OperationError("docker export", export_status, errors.read())
        # A metadata-only build preserves SHELL, which `docker import --change`
        # cannot restore. No RUN/COPY instructions and no inherited source layers;
        # Docker may append its canonical empty tar, checked by its content hash.
        with tempfile.TemporaryDirectory(prefix="artifixer-image-config-") as temporary:
            directory = Path(temporary)
            recipe = f"FROM {staging}\n" + "\n".join(instructions) + "\n"
            (directory / "Dockerfile").write_text(recipe)
            docker(
                "build",
                "--network",
                "none",
                "--platform",
                platform(metadata),
                "--iidfile",
                str(directory / "image.id"),
                str(directory),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            candidate_id = (directory / "image.id").read_text().strip()
        candidate = inspect_image(candidate_id, expected_platform=platform(metadata))
        candidate_id = candidate["Id"]
        report = verify(candidate_id, expected_platform=platform(metadata))
        report["source_image_id"] = metadata["Id"]
        report["destination"] = destination
        if config_identity(candidate["Config"]) != config_identity(metadata["Config"]):
            report["status"] = "fail"
            report["failures"].append("Flattened runtime configuration differs from the source")
        if report["status"] == "pass":
            destination_absent(destination)
            docker("tag", candidate_id, destination)
            if inspect_image(destination, expected_platform=platform(metadata))["Id"] != candidate_id:
                raise RuntimeError("Destination changed during tag verification")
        return report
    finally:
        subprocess.run(["docker", "container", "rm", "-f", container], capture_output=True, check=False)
        # Only remove our private staging tag. Preserve failed candidates for diagnosis.
        subprocess.run(["docker", "image", "rm", staging], capture_output=True, check=False)


def retain_inventories(report, base_image, report_path):
    source = (REPO_ROOT / "docker/image_inventory.py").read_bytes()
    inventories = {}
    for role, image in (("base", base_image), ("final", report["image_id"])):
        metadata = inspect_image(image, expected_platform=report.get("platform"))
        output = docker(
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--user",
            "0",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=void",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--platform",
            platform(metadata),
            "--entrypoint",
            "python",
            "-i",
            metadata["Id"],
            "-",
            input=source,
            capture_output=True,
        ).stdout
        inventory = json.loads(output)
        if not inventory.get("pip") or not inventory.get("apt"):
            raise ValueError("Incomplete resolved package inventory")
        inventory["image_id"] = metadata["Id"]
        inventory["platform_image_id"] = metadata.get("PlatformId", metadata["Id"])
        inventory["platform"] = platform(metadata)
        raw = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode()
        path = report_path.with_name(report_path.stem + f".{role}-inventory.json")
        with path.open("xb") as handle:
            handle.write(raw)
        inventories[role] = {
            "path": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "image_id": metadata["Id"],
            "platform_image_id": inventory["platform_image_id"],
            "platform": inventory["platform"],
        }
    report["inventories"] = inventories


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "flatten", "lock-path", "assert-absent"))
    parser.add_argument("image")
    parser.add_argument("destination", nargs="?")
    parser.add_argument("--build-image", action="store_true", help="Check the final filesystem without a layer claim")
    parser.add_argument("--platform", dest="expected_platform")
    parser.add_argument("--base-image", help="Immutable base image used for the resolved package inventory")
    parser.add_argument("--report", type=Path, help="Write a new JSON receipt; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.action == "lock-path":
        print(lock_path(args.image))
        return 0
    if args.action == "assert-absent":
        destination_absent(args.image)
        return 0
    if args.action == "flatten" and (not args.destination or args.build_image):
        parser.error("flatten requires a destination and does not accept --build-image")
    if args.action == "verify" and args.destination:
        parser.error("verify accepts one image")
    if args.report and args.report.exists():
        parser.error("Report already exists; use a new receipt path")
    if args.report and not args.base_image:
        parser.error("--report requires --base-image to retain both resolved package inventories")
    CONTEXT.clear()
    SENSITIVE_VALUES.clear()
    try:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
        base_id = (
            inspect_image(args.base_image, expected_platform=args.expected_platform)["Id"] if args.base_image else None
        )
        report = (
            flatten(args.image, args.destination, expected_platform=args.expected_platform)
            if args.action == "flatten"
            else verify(args.image, require_flat=not args.build_image, expected_platform=args.expected_platform)
        )
        if args.report and report.get("image_id") and report["status"] == "pass":
            retain_inventories(report, base_id, args.report)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile) as exc:
        # Avoid echoing commands containing inherited image ENV/LABEL values.
        report = {
            "status": "fail",
            "error": sanitize(str(exc)) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__,
            **CONTEXT,
        }
        if isinstance(exc, OperationError):
            report.update(exc.details)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    sys.exit(main())
