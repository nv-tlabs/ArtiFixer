# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "prune_ngc_extras", Path(__file__).resolve().parents[1] / "docker/prune_ngc_extras.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
dependencies = module.dependencies
removal_plan = module.removal_plan


def test_removes_reverse_dependency_closure_only():
    graph = {"numba": {"numpy"}, "cudf": {"numba"}, "consumer": {"cudf"}, "numpy": set(), "torch": {"numpy"}}
    assert removal_plan(graph, {"numba"}, {"torch"}) == ["consumer", "cudf", "numba"]


def test_protects_transitive_project_dependencies():
    graph = {"app": {"intermediate"}, "intermediate": {"numba"}, "numba": set()}
    with pytest.raises(ValueError, match="project dependencies"):
        removal_plan(graph, {"numba"}, {"app"})


def test_optional_features_do_not_force_removal_of_consumers():
    assert dependencies(['Numba>=0.59; extra == "audio"', "NumPy>=2"]) == {"numpy"}


@pytest.mark.parametrize("indirect", [False, True])
def test_selected_extras_protect_required_numba_before_uninstall(monkeypatch, tmp_path, indirect):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example[audio]\n" if not indirect else "example\n")
    installed = {
        "example": SimpleNamespace(requires=["bridge[audio]"] if indirect else ['numba; extra == "audio"']),
        "bridge": SimpleNamespace(requires=['numba; extra == "audio"']),
        "numba": SimpleNamespace(requires=[]),
    }
    monkeypatch.setattr(module, "active_distributions", lambda: installed)
    monkeypatch.setattr(module.sys, "argv", ["prune", str(requirements)])
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="numba"):
        module.main()
    assert not calls


def test_shadowed_metadata_cannot_remove_active_project_dependency(monkeypatch, tmp_path):
    active = SimpleNamespace(metadata={"Name": "Example"}, requires=["numba"], version="1")
    shadowed = SimpleNamespace(metadata={"Name": "example"}, requires=[], version="0")
    numba = SimpleNamespace(metadata={"Name": "numba"}, requires=[], version="1")
    monkeypatch.setattr(module.importlib.metadata, "distributions", lambda: [active, numba, shadowed])
    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: {"example": active, "numba": numba}[name]
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example\n")
    monkeypatch.setattr(module.sys, "argv", ["prune", str(requirements)])
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="numba"):
        module.main()
    assert not calls


def test_later_extra_expands_an_already_visited_dependency():
    installed = {
        "example": SimpleNamespace(requires=["bridge[audio]", "bridge"]),
        "bridge": SimpleNamespace(requires=['numba; extra == "audio"']),
        "numba": SimpleNamespace(requires=[]),
    }
    assert module.protected_closure(installed, ["example"]) == {"example", "bridge", "numba"}


def test_inactive_platform_requirement_does_not_protect_unused_package():
    assert module.protected_closure({}, ['numba; python_version < "2"']) == set()
