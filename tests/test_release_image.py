# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release metadata, package replacements, and publish-tag failure boundaries."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "docker" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def release():
    return load_script("release_image")


def test_runtime_metadata_contains_user_ports_shell_and_literal_environment(release):
    lines = release.config_instructions(
        {
            "Env": ["EMPTY=", 'LITERAL=$HOME "quoted" \\path'],
            "User": "1001:1002",
            "WorkingDir": "/path with space",
            "ExposedPorts": {"8888/tcp": {}, "5000/udp": {}},
            "StopSignal": "SIGTERM",
            "Shell": ["/bin/bash", "-o", "pipefail", "-c"],
            "Entrypoint": ["/entrypoint", "one argument"],
            "Cmd": ["/bin/bash"],
        }
    )
    assert 'ENV EMPTY=""' in lines
    assert 'ENV LITERAL="\\$HOME \\"quoted\\" \\\\path"' in lines
    assert 'USER "1001:1002"' in lines
    assert "EXPOSE 8888/tcp" in lines
    assert "EXPOSE 5000/udp" in lines
    assert 'SHELL ["/bin/bash", "-o", "pipefail", "-c"]' in lines
    assert not any(line.startswith(("RUN ", "COPY ")) for line in lines)


@pytest.mark.parametrize(
    "field,value", [("Volumes", {"/data": {}}), ("OnBuild", ["RUN true"]), ("Healthcheck", {"Test": ["NONE"]})]
)
def test_unsupported_settings_are_refused_before_export(release, field, value):
    with pytest.raises(ValueError, match=field):
        release.assert_exportable({"Os": "linux", "Config": {field: value}})


@pytest.mark.parametrize("value", ["a\nb", "a\rb", "a\0b"])
def test_multiline_metadata_is_not_silently_changed(release, value):
    with pytest.raises(ValueError, match="lossy"):
        release.quote(value)


def test_failed_source_gate_does_not_create_a_candidate_or_tag(release, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(release, "destination_absent", lambda destination: None)
    monkeypatch.setattr(release, "lock_path", lambda destination: tmp_path / "tag.lock")
    monkeypatch.setattr(
        release, "inspect_image", lambda image, **kwargs: {"Id": "sha256:source", "Os": "linux", "Config": {}}
    )
    monkeypatch.setattr(release, "verify", lambda *args, **kwargs: {"status": "fail"})
    monkeypatch.setattr(release, "docker", lambda *args, **kwargs: calls.append(args))
    assert release.flatten("source", "destination")["status"] == "fail"
    assert calls == []


def test_existing_destination_is_preserved(release, monkeypatch):
    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(ValueError, match="already exists"):
        release.destination_absent("existing:release")


def test_daemon_failure_is_not_treated_as_an_unused_tag(release, monkeypatch):
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=b"Cannot connect to daemon"),
    )
    with pytest.raises(RuntimeError, match="determine"):
        release.destination_absent("new:release")


@pytest.mark.parametrize(
    "extra_layer,expected", [(None, "pass"), ("empty", "pass"), ("sha256:old-codec-layer", "fail")]
)
def test_clean_filesystem_does_not_qualify_hidden_layer_history(release, monkeypatch, extra_layer, expected):
    layers = ["sha256:rootfs"]
    if extra_layer:
        layers.append(release.EMPTY_LAYER if extra_layer == "empty" else extra_layer)
    metadata = {
        "Id": "sha256:image",
        "Os": "linux",
        "Architecture": "amd64",
        "Config": {},
        "Size": 123,
        "RootFS": {"Layers": layers},
    }
    monkeypatch.setattr(release, "inspect_image", lambda image, **kwargs: metadata)
    monkeypatch.setattr(release, "scan_saved_image", lambda metadata: {"status": "pass"})
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b'{"status":"pass"}', stderr=b""),
    )
    assert release.verify("image")["status"] == expected


def test_package_delta_includes_replaced_versions_and_source_revisions():
    inventory = load_script("container_inventory")
    base = {
        "Torch": {"version": "2.6"},
        "same_pkg": {"version": "1"},
        "git": {"version": "1", "source_revision": "old"},
    }
    final = {
        "torch": {"version": "2.11"},
        "same-pkg": {"version": "1"},
        "git": {"version": "1", "source_revision": "new"},
        "new": {"version": "1"},
    }
    assert set(inventory.changed_packages(base, final)) == {"torch", "git", "new"}


def test_missing_inventory_section_is_not_an_empty_delta(tmp_path):
    inventory = load_script("container_inventory")
    path = tmp_path / "broken.list"
    path.write_text("torch==2.11\n")
    with pytest.raises(ValueError, match="Incomplete"):
        inventory.read_inventory(path)


def test_notices_select_active_distribution_over_shadowed_metadata(monkeypatch, capsys):
    import importlib.metadata
    import json
    import os.path
    import subprocess
    from email.message import Message

    def distribution(name, version):
        metadata = Message()
        metadata["Name"] = name
        metadata["License-Expression"] = "MIT"
        return SimpleNamespace(metadata=metadata, version=version, read_text=lambda _: None)

    active = distribution("typing_extensions", "4.15.0")
    shadowed = distribution("typing-extensions", "4.12.2")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "docker"))
    notices = load_script("generate_container_notices")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [active, shadowed])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: active)
    monkeypatch.setattr(os.path, "isfile", lambda _: False)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="libc6\t1\t\n"))
    exec(notices.INVENTORY, {})
    result = json.loads(capsys.readouterr().out)
    assert list(result["pip"]) == ["typing_extensions"]
    assert result["pip"]["typing_extensions"]["version"] == "4.15.0"


@pytest.mark.parametrize("selected", ["linux/amd64", "linux/arm64"])
def test_notices_use_one_platform_for_both_inventories_and_provenance(monkeypatch, tmp_path, selected):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "docker"))
    notices = load_script("generate_container_notices")
    inspected, runs = [], []

    def inspect(image, *, expected_platform):
        inspected.append((image, expected_platform))
        return {
            "Id": f"sha256:{image}",
            "Os": "linux",
            "Architecture": selected.split("/")[1],
            "Config": {"Labels": {"org.opencontainers.image.revision": selected + "-revision"}},
        }

    def run(command, **kwargs):
        runs.append(command)
        return SimpleNamespace(stdout=json.dumps({"pip": {}, "apt": {}}))

    monkeypatch.setattr(notices, "inspect_image", inspect)
    monkeypatch.setattr(notices.subprocess, "run", run)
    output = tmp_path / "notices.md"
    assert notices.main(["base", "final", "--platform", selected, "--output", str(output)]) == 0
    assert inspected == [("final", selected), ("base", selected)]
    assert len(runs) == 2
    for command, image in zip(runs, ("sha256:base", "sha256:final")):
        assert command[command.index("--platform") + 1] == selected
        assert image in command
    assert selected + "-revision" in output.read_text()


@pytest.mark.parametrize(
    "raw, classifier, expected",
    [
        ("Copyright (c) NumPy developers.\nFull license text follows.", "BSD License", "BSD License"),
        ("BSD-3-Clause", "BSD License", "BSD-3-Clause"),
        ("Copyright (c) Example Corporation", None, ""),
        ("NVIDIA Software License Agreement\nFull terms follow.", None, "NVIDIA Software License Agreement"),
    ],
)
def test_notices_do_not_mislabel_copyright_as_license(monkeypatch, capsys, raw, classifier, expected):
    import importlib.metadata
    import json
    import os.path
    import subprocess
    from email.message import Message

    metadata = Message()
    metadata["Name"] = "example"
    metadata["License"] = raw
    if classifier:
        metadata["Classifier"] = "License :: OSI Approved :: " + classifier
    distribution = SimpleNamespace(metadata=metadata, version="1", read_text=lambda _: None)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "docker"))
    notices = load_script("generate_container_notices")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [distribution])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(os.path, "isfile", lambda _: False)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="libc6\t1\t\n"))
    exec(notices.INVENTORY, {})
    result = json.loads(capsys.readouterr().out)
    assert result["pip"]["example"]["license"] == expected


def test_saved_layer_scans_runtime_masked_paths_and_wheels(release):
    import io
    import runpy
    import tarfile
    import zipfile

    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vendor/libavcodec.so", b"codec")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name, data, mode in [
            ("dev/libavcodec.so", b"\x7fELFcodec", 0o644),
            ("proc/ffmpeg", b"#!/bin/sh", 0o755),
            ("sys/download.whl", wheel.getvalue(), 0o644),
            ("usr/share/doc/ffmpeg", b"documentation", 0o644),
        ]:
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), mode
            archive.addfile(member, io.BytesIO(data))
    payload.seek(0)
    result = release.scan_layer(payload, runpy.run_path(str(release.GATE)))
    assert result["offenders"] == ["dev/libavcodec.so", "proc/ffmpeg", "sys/download.whl!vendor/libavcodec.so"]


def test_inspection_failure_receipt_keeps_useful_diagnostic(release, monkeypatch, capsys):
    import json
    import subprocess

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(
            1, command, stderr=b"Error response from daemon: No such image: missing:tag"
        )

    monkeypatch.setattr(release.subprocess, "run", fail)
    assert release.main(["verify", "missing:tag"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["operation"] == "docker image inspect"
    assert report["exit_status"] == 1
    assert report["image_reference"] == "missing:tag"
    assert "No such image" in report["diagnostic"]


def test_build_failure_redacts_inherited_config_and_transport_credentials(release, monkeypatch):
    import subprocess

    secret = 'private-value-$HOME-"quoted"'
    release.SENSITIVE_VALUES.add(secret)
    release.CONTEXT["image_id"] = "sha256:source"
    diagnostic = "\n".join(
        [
            "prefix " * 1000,
            "Step 3: ENV TOKEN=" + secret,
            "failed to apply metadata " + secret,
            "https://user:password@example.com/data?signature=private",
            "Error response from daemon: no space left on device",
        ]
    )

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(125, command, stderr=diagnostic)

    monkeypatch.setattr(release.subprocess, "run", fail)
    with pytest.raises(release.OperationError) as error:
        release.docker("build", "--secret", secret)
    details = error.value.details
    assert details["operation"] == "docker build"
    assert details["image_id"] == "sha256:source"
    assert details["exit_status"] == 125
    assert len(details["diagnostic"]) <= 4000
    assert "no space left on device" in details["diagnostic"]
    assert secret not in details["diagnostic"]
    assert "user:password" not in details["diagnostic"]
    assert "signature=private" not in details["diagnostic"]


def test_destination_lock_serializes_competing_release_writers(release, monkeypatch, tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    first_inside, second_started, release_first = (ctx.Event() for _ in range(3))
    results = ctx.Queue()
    destination = tmp_path / "published"
    monkeypatch.setattr(release, "lock_path", lambda _: tmp_path / "tag.lock")

    def publish_locked(source, tag, **kwargs):
        if destination.exists():
            raise ValueError("Destination already exists")
        if source == "first":
            first_inside.set()
            assert release_first.wait(10)
        destination.write_text(source)
        return {"status": "pass"}

    monkeypatch.setattr(release, "flatten_locked", publish_locked)

    def writer(source):
        if source == "second":
            second_started.set()
        try:
            results.put((source, release.flatten(source, "shared:tag")["status"]))
        except ValueError:
            results.put((source, "preserved existing tag"))

    processes = [ctx.Process(target=writer, args=(source,)) for source in ("first", "second")]
    try:
        processes[0].start()
        assert first_inside.wait(10)
        # Prove exclusion while the first writer is paused before publication.
        # This fails deterministically without the lock, regardless of how the
        # two writer processes happen to be scheduled after release_first.
        with (tmp_path / "tag.lock").open("a") as probe:
            with pytest.raises(BlockingIOError):
                release.fcntl.flock(probe, release.fcntl.LOCK_EX | release.fcntl.LOCK_NB)
        processes[1].start()
        assert second_started.wait(10)
        release_first.set()
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert sorted(results.get(timeout=1) for _ in processes) == [
            ("first", "pass"),
            ("second", "preserved existing tag"),
        ]
        assert destination.read_text() == "first"
    finally:
        release_first.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


def test_equivalent_docker_tag_spellings_share_lock(release, monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFIXER_RELEASE_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(release, "docker", lambda *args, **kwargs: SimpleNamespace(stdout="daemon-id"))
    assert release.lock_path("example") == release.lock_path("docker.io/library/example:latest")
    assert release.lock_path("index.docker.io/example") == release.lock_path("example:latest")


def test_source_inventory_collects_same_version_new_git_revision(monkeypatch, tmp_path):
    import builtins
    import importlib.metadata
    import io
    import json
    import re
    import subprocess
    import urllib.request

    producer = load_script("image_inventory")
    comparator = load_script("container_inventory")
    revision = "a" * 40
    distribution = SimpleNamespace(
        metadata={"Name": "example"},
        version="1.0",
        read_text=lambda _: json.dumps(
            {"url": "https://github.com/owner/example.git", "vcs_info": {"commit_id": revision}}
        ),
    )
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [distribution])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="libc6\t1\tamd64\tinstalled\n")
    )
    base = tmp_path / "base.json"
    final = tmp_path / "final.json"
    base.write_text(json.dumps(producer.inventory()))
    revision = "b" * 40
    final.write_text(json.dumps(producer.inventory()))
    delta = comparator.changed_packages(comparator.read_inventory(base)["pip"], comparator.read_inventory(final)["pip"])
    assert delta == {"example": {"version": "1.0", "source_revision": revision}}
    added = "".join(f"{name}=={entry['version']}\n" for name, entry in delta.items())
    script = (Path(__file__).resolve().parents[1] / "docker/collect_container_sources.sh").read_text()
    assert '< "$SCRIPT_DIR/image_inventory.py"' in script
    helper = re.search(r"cat > .*fetch_pip.py.*?<<'PYEOF'\n(.*?)\nPYEOF", script, re.S).group(1)
    downloads = []
    real_open = builtins.open

    def open_output(path, *args, **kwargs):
        if str(path) == "/out/pip.added":
            return io.StringIO(added)
        if str(path).startswith("/out/"):
            return io.StringIO()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_output)
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: downloads.append((url, dest)))
    exec(helper, {})
    assert downloads == [
        (
            f"https://codeload.github.com/owner/example/tar.gz/{revision}",
            f"/out/sources/pip/example-1.0-{revision}.tar.gz",
        )
    ]


def test_fa3_without_vcs_metadata_collects_the_pinned_git_source(monkeypatch, tmp_path):
    import builtins
    import importlib.metadata
    import json
    import re
    import subprocess
    import urllib.request

    producer = load_script("image_inventory")
    comparator = load_script("container_inventory")
    installed = {"pip": SimpleNamespace(metadata={"Name": "pip"}, version="25.1", read_text=lambda _: None)}
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: list(installed.values()))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: installed[name])
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="libc6\t1\tamd64\tinstalled\n")
    )
    base = tmp_path / "base.json"
    final = tmp_path / "final.json"
    base.write_text(json.dumps(producer.inventory()))
    installed["flash-attn-3"] = SimpleNamespace(
        metadata={"Name": "flash_attn_3"}, version="3.0.0", read_text=lambda _: None
    )
    final.write_text(json.dumps(producer.inventory()))
    delta = comparator.changed_packages(comparator.read_inventory(base)["pip"], comparator.read_inventory(final)["pip"])
    assert delta == {"flash-attn-3": {"version": "3.0.0", "source_revision": ""}}
    (tmp_path / "pip.added").write_text("".join(f"{name}=={entry['version']}\n" for name, entry in delta.items()))
    root = Path(__file__).resolve().parents[1]
    script = (root / "docker/collect_container_sources.sh").read_text()
    helper = re.search(r"cat > .*fetch_pip.py.*?<<'PYEOF'\n(.*?)\nPYEOF", script, re.S).group(1)
    real_open = builtins.open

    def open_output(path, *args, **kwargs):
        if str(path).startswith("/out/"):
            path = tmp_path / str(path).removeprefix("/out/")
        return real_open(path, *args, **kwargs)

    def refuse_pypi(*args, **kwargs):
        raise AssertionError("FA3 must use its pinned Git source, not PyPI")

    downloads = []
    monkeypatch.setattr(builtins, "open", open_output)
    monkeypatch.setattr(urllib.request, "urlopen", refuse_pypi)
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: downloads.append((url, dest)))
    exec(helper, {})
    revision = "77aacb68d194ba9af1010eda5eac3e7c0df8e6f6"
    for variant in ("cuda12", "cuda13"):
        assert f"git fetch --depth 1 origin {revision}" in (root / f"Dockerfile.{variant}").read_text()
    assert downloads == [
        (
            f"https://codeload.github.com/Dao-AILab/flash-attention/tar.gz/{revision}",
            f"/out/sources/pip/flash-attn-3-3.0.0-{revision}.tar.gz",
        )
    ]
    assert not (tmp_path / "NEEDS-ATTENTION.txt").exists()
    assert revision in (tmp_path / "SOURCE-NOTES.txt").read_text()


def test_check_only_freezes_source_once_for_verification_and_flatten(tmp_path):
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    commands = tmp_path / "commands.jsonl"
    tools = tmp_path / "bin"
    tools.mkdir()
    shim = (
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, pathlib, sys
with open(os.environ['RELEASE_TEST_COMMANDS'], 'a') as handle:
    handle.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + '\n')
if pathlib.Path(sys.argv[0]).name == 'docker':
    if sys.argv[-1] in ('base:fixture', 'sha256:base'):
        print('sha256:base linux/amd64' if '{{.Architecture}}' in sys.argv[-2] else 'sha256:base')
    else:
        print('sha256:frozen-source')
"""
    )
    for name in ("docker", "python3"):
        path = tools / name
        path.write_text(shim)
        path.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(root / "docker/build_release_image.sh"),
            "cuda12",
            "--check-only",
            "--flatten",
            "--tag",
            "mutable:source",
            "--base-image",
            "base:fixture",
            "--report-dir",
            str(tmp_path / "receipts"),
        ],
        env=os.environ | {"PATH": str(tools) + os.pathsep + os.environ["PATH"], "RELEASE_TEST_COMMANDS": str(commands)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in commands.read_text().splitlines()]
    verify = next(call for call in calls if "verify" in call)
    flatten = next(call for call in calls if "flatten" in call)
    assert verify[verify.index("verify") + 1] == "sha256:frozen-source"
    assert flatten[flatten.index("flatten") + 1] == "sha256:frozen-source"
    assert len([call for call in calls if call[0] == "docker" and call[-1] == "mutable:source"]) == 1


def test_receipt_inventory_files_are_bound_to_immutable_ids(release, monkeypatch, tmp_path):
    import hashlib
    import json

    monkeypatch.setattr(
        release, "inspect_image", lambda image, **kwargs: {"Id": image, "Os": "linux", "Architecture": "amd64"}
    )
    calls = []

    def collect(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            stdout=json.dumps(
                {"pip": {"example": {"version": "1", "source_revision": "abc"}}, "apt": {"libc6": {"version": "1"}}}
            ).encode()
        )

    monkeypatch.setattr(release, "docker", collect)
    report = {"image_id": "sha256:final"}
    release.retain_inventories(report, "sha256:base", tmp_path / "release.json")
    assert len(calls) == 2
    for role in ("base", "final"):
        record = report["inventories"][role]
        raw = (tmp_path / record["path"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["image_id"] == json.loads(raw)["image_id"] == "sha256:" + role
        assert "sha256:" + role in calls[0 if role == "base" else 1]


@pytest.mark.parametrize(
    "export_status,import_status,operation,diagnostic",
    [(1, 0, "docker export", "export permission denied"), (-13, 1, "docker import", "no space left on device")],
)
def test_export_import_failures_keep_the_primary_diagnostic(
    release, monkeypatch, export_status, import_status, operation, diagnostic
):
    import io

    monkeypatch.setattr(release, "destination_absent", lambda _: None)
    monkeypatch.setattr(
        release, "inspect_image", lambda _, **kwargs: {"Id": "sha256:source", "Os": "linux", "Config": {}}
    )
    monkeypatch.setattr(release, "verify", lambda *args, **kwargs: {"status": "pass"})
    monkeypatch.setattr(release, "docker", lambda *args, **kwargs: SimpleNamespace(stdout="container-id"))

    def exporter(*args, **kwargs):
        kwargs["stderr"].write(b"export permission denied")
        return SimpleNamespace(stdout=io.BytesIO(), wait=lambda: export_status)

    def run(command, **kwargs):
        if command[1] == "import":
            return SimpleNamespace(returncode=import_status, stderr="no space left on device")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(release.subprocess, "Popen", exporter)
    monkeypatch.setattr(release.subprocess, "run", run)
    with pytest.raises(release.OperationError) as error:
        release.flatten_locked("source", "destination")
    assert error.value.details["operation"] == operation
    assert diagnostic in error.value.details["diagnostic"]
