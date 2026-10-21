# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual build wrapper with controlled Docker cache/platform state."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_wrapper(tmp_path):
    root = tmp_path / "repo"
    (root / "docker").mkdir(parents=True)
    for name in ("build_release_image.sh", "flatten_release_image.sh"):
        shutil.copy2(ROOT / "docker" / name, root / "docker" / name)
    for name in (
        "thirdparty/3DGRUT-ArtiFixer/setup.py",
        "thirdparty/3DGRUT-ArtiFixer/thirdparty/tiny-cuda-nn/include/tiny-cuda-nn/common.h",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    tools = tmp_path / "bin"
    tools.mkdir()
    shim = (
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, pathlib, sys
args = sys.argv[1:]
tool = pathlib.Path(sys.argv[0]).name
with open(os.environ['RELEASE_TEST_COMMANDS'], 'a') as handle:
    handle.write(json.dumps([tool, *args]) + '\n')
state = pathlib.Path(os.environ['RELEASE_TEST_STATE'])
if tool == 'python3':
    if args[1] == 'lock-path':
        print(str(state.parent / 'tag.lock'))
elif args[:2] == ['image', 'inspect']:
    if args[-1] == 'mutable:source':
        print('sha256:frozen-source')
    else:
        assert args[-1] == 'base:fixture' or args[-1].startswith('sha256:base-'), args
        platforms = json.loads(state.read_text())
        requested = args[args.index('--platform') + 1] if '--platform' in args else None
        if not platforms or (requested and requested not in platforms):
            print('No such image: base:fixture for ' + str(requested), file=sys.stderr)
            sys.exit(1)
        platform = requested or ('linux/amd64' if 'linux/amd64' in platforms else platforms[0])
        fields = args[args.index('--format') + 1]
        values = []
        if '.RepoDigests' in fields:
            if os.environ['RELEASE_TEST_NO_DIGEST'] == '1':
                if fields == '{{index .RepoDigests 0}}':
                    print('index out of range', file=sys.stderr)
                    sys.exit(1)
            else:
                values.append('registry.example/base@sha256:' + 'a' * 64)
        if '{{.Id}}' in fields:
            values.append('sha256:base-' + platform.split('/')[1] if requested or len(platforms) == 1 else 'sha256:base-index')
        if '{{.Architecture}}' in fields:
            values.append(platform)
        print(' '.join(values))
elif args[0] == 'pull':
    assert args[1:3] == ['--platform', os.environ['RELEASE_TEST_PLATFORM']], args
    if os.environ['RELEASE_TEST_PULL_FAIL'] == '1':
        print('fixture registry unavailable', file=sys.stderr)
        sys.exit(17)
    platforms = json.loads(state.read_text())
    state.write_text(json.dumps(sorted(set(platforms + [os.environ['RELEASE_TEST_PULLED_PLATFORM']]))))
elif args[0] == 'build':
    pathlib.Path(args[args.index('--iidfile') + 1]).write_text('sha256:built\n')
else:
    raise AssertionError(args)
"""
    )
    for name in ("docker", "python3"):
        path = tools / name
        path.write_text(shim)
        path.chmod(0o755)

    def run(variant, cached_platform, *, check_only=False, pull_fail=False, pulled_platform=None, no_digest=False):
        expected = "linux/arm64" if variant == "cuda13-aarch64" else "linux/amd64"
        (root / f"Dockerfile.{variant}").write_text(
            "ARG ARTIFIXER_BASE_IMAGE=base:fixture\nFROM ${ARTIFIXER_BASE_IMAGE}\n"
        )
        state = tmp_path / "base-platform"
        platforms = (
            cached_platform
            if isinstance(cached_platform, list)
            else [] if cached_platform == "missing" else [cached_platform]
        )
        state.write_text(json.dumps(platforms))
        commands = tmp_path / "commands.jsonl"
        command = [
            "bash",
            str(root / "docker/build_release_image.sh"),
            variant,
            "--tag",
            "mutable:source",
            "--flatten",
            "--report-dir",
            str(tmp_path / "receipts"),
        ]
        if check_only:
            command.append("--check-only")
        result = subprocess.run(
            command,
            env=os.environ
            | {
                "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                "ARTIFIXER_GIT_REVISION": "fixture",
                "RELEASE_TEST_COMMANDS": str(commands),
                "RELEASE_TEST_STATE": str(state),
                "RELEASE_TEST_PLATFORM": expected,
                "RELEASE_TEST_PULLED_PLATFORM": pulled_platform or expected,
                "RELEASE_TEST_PULL_FAIL": str(int(pull_fail)),
                "RELEASE_TEST_NO_DIGEST": str(int(no_digest)),
            },
            capture_output=True,
            text=True,
        )
        return result, [json.loads(line) for line in commands.read_text().splitlines()]

    run.receipts = tmp_path / "receipts"
    return run


@pytest.mark.parametrize("variant", ["cuda12", "cuda13", "cuda13-aarch64"])
@pytest.mark.parametrize("cache", ["missing", "correct", "wrong-platform"])
def test_build_acquires_the_correct_platform_base_before_verification(run_wrapper, variant, cache):
    expected = "linux/arm64" if variant == "cuda13-aarch64" else "linux/amd64"
    wrong = "linux/amd64" if expected == "linux/arm64" else "linux/arm64"
    cached = {"missing": "missing", "correct": expected, "wrong-platform": wrong}[cache]
    result, calls = run_wrapper(variant, cached)
    assert result.returncode == 0, result.stderr
    pulls = [call for call in calls if call[:2] == ["docker", "pull"]]
    assert len(pulls) == (cache != "correct")
    build = next(call for call in calls if call[:2] == ["docker", "build"])
    assert build[build.index("--platform") + 1] == expected
    assert "ARTIFIXER_BASE_IMAGE=registry.example/base@sha256:" + "a" * 64 in build
    if pulls:
        assert pulls[0] == ["docker", "pull", "--platform", expected, "base:fixture"]
        assert calls.index(pulls[0]) < calls.index(build)
    verify = next(call for call in calls if "verify" in call)
    flatten = next(call for call in calls if "flatten" in call)
    for call, operation in ((verify, "verify"), (flatten, "flatten")):
        assert call[call.index(operation) + 1] == "sha256:built"
        root_id = "sha256:base-index" if cache == "wrong-platform" else "sha256:base-" + expected.split("/")[1]
        assert call[call.index("--base-image") + 1] == root_id
        assert call[call.index("--platform") + 1] == expected
        assert calls.index(call) > calls.index(build)


@pytest.mark.parametrize("pull_fail", [True, False])
def test_unavailable_or_wrong_platform_base_stops_before_build(run_wrapper, pull_fail):
    result, calls = run_wrapper("cuda13-aarch64", "missing", pull_fail=pull_fail, pulled_platform="linux/amd64")
    assert result.returncode != 0
    assert "registry unavailable" in result.stderr if pull_fail else "linux/arm64" in result.stderr
    assert not any(call[:2] == ["docker", "build"] or "verify" in call or "flatten" in call for call in calls)


def test_check_only_with_missing_base_does_not_pull_or_build(run_wrapper):
    result, calls = run_wrapper("cuda12", "missing", check_only=True)
    assert result.returncode != 0
    assert "No such image" in result.stderr
    assert not any(call[:2] in (["docker", "pull"], ["docker", "build"]) for call in calls)


@pytest.mark.parametrize("variant", ["cuda12", "cuda13-aarch64"])
def test_base_without_registry_digest_reports_recovery_before_build(run_wrapper, variant):
    platform = "linux/arm64" if variant == "cuda13-aarch64" else "linux/amd64"
    result, calls = run_wrapper(variant, platform, no_digest=True)
    assert result.returncode == 2
    assert "pull the base image first" in result.stderr
    assert not any(call[:2] == ["docker", "build"] or "verify" in call or "flatten" in call for call in calls)


@pytest.mark.parametrize("variant", ["cuda12", "cuda13-aarch64"])
@pytest.mark.parametrize("check_only", [False, True])
def test_coexisting_platforms_select_the_requested_immutable_base(run_wrapper, variant, check_only):
    expected = "linux/arm64" if variant == "cuda13-aarch64" else "linux/amd64"
    result, calls = run_wrapper(variant, ["linux/amd64", "linux/arm64"], check_only=check_only)
    assert result.returncode == 0, result.stderr
    assert not any(call[:2] == ["docker", "pull"] for call in calls)
    assert (run_wrapper.receipts / "base-image.id").read_text().strip() == "sha256:base-" + expected.split("/")[1]
    assert (run_wrapper.receipts / "base-image.root-id").read_text().strip() == "sha256:base-index"
    for operation in ("verify", "flatten"):
        call = next(call for call in calls if operation in call)
        assert call[call.index("--base-image") + 1] == "sha256:base-index"
        assert call[call.index("--platform") + 1] == expected
    if check_only:
        assert not any(call[:2] == ["docker", "build"] for call in calls)
        source = next(call for call in calls if call[:2] == ["docker", "image"] and call[-1] == "mutable:source")
        assert calls.count(source) == 1


def test_check_only_does_not_pull_an_uncached_platform(run_wrapper):
    result, calls = run_wrapper("cuda13-aarch64", ["linux/amd64"], check_only=True)
    assert result.returncode != 0
    assert not any(call[:2] in (["docker", "pull"], ["docker", "build"]) for call in calls)
