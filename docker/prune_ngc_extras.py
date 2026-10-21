# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove unused NGC applications incompatible with the project's Torch/NumPy.

This runs inside an image build, never against the host environment. Refuse to
remove a declared project dependency or anything in its dependency closure.
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

UNUSED_ROOTS = {
    "numba",
    "llvmlite",
    "spacy",
    "thinc",
    "nvidia-modelopt",
    "torch-tensorrt",
    "lightning-thunder",
    # Independent RAPIDS libraries can retain CUDA 12-only Python bindings even
    # after the NumPy-dependent RAPIDS applications have been removed.
    "pylibraft",
    "cuvs",
    "rmm",
    "pylibcudf",
    # NGC's NumPy development helper pins an older Click than the HF CLI.
    "spin",
    # Old NGC nvimgcodec wheels also have malformed wheel tags. Image decoding
    # here uses Pillow/OpenCV, not nvimgcodec.
    "nvidia-nvimgcodec-cu12",
    "nvidia-nvimgcodec-cu13",
}


def selected_requirements(requirements, extras=()):
    result = []
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker is None or any(requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras)):
            result.append(requirement)
    return result


def dependencies(requirements, extras=()):
    return {canonicalize_name(req.name) for req in selected_requirements(requirements, extras)}


def active_distributions():
    installed = {}
    for dist in importlib.metadata.distributions():
        name = canonicalize_name(dist.metadata["Name"])
        if name not in installed:
            installed[name] = importlib.metadata.distribution(name)
    return installed


def protected_closure(installed, requirements):
    """Propagate extras, including extras discovered on transitive edges."""
    pending = selected_requirements(requirements)
    selected = {}
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        extras = set(requirement.extras) | selected.get(name, set())
        if name in selected and extras == selected[name]:
            continue
        selected[name] = extras
        if name in installed:
            pending.extend(selected_requirements(installed[name].requires or (), extras))
    return set(selected)


def removal_plan(graph, roots, protected):
    required = set(protected)
    while True:
        expanded = required | {dep for name in required for dep in graph.get(name, ())}
        if expanded == required:
            break
        required = expanded
    remove = set(roots) & graph.keys()
    while True:
        expanded = remove | {name for name, deps in graph.items() if deps & remove}
        if expanded == remove:
            break
        remove = expanded
    if remove & required:
        raise ValueError(f"Refusing to remove project dependencies: {sorted(remove & required)}")
    return sorted(remove)


def project_requirements(path):
    requirements = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("git+https://github.com/rahul-goel/fused-ssim@"):
            requirements.append("fused-ssim")
        else:
            requirements.append(str(Requirement(line)))
    return requirements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", type=Path, nargs="+")
    args = parser.parse_args()
    requirements = ["torch", "torchvision", "flash-attn-4", "nvidia-cutlass-dsl", "moge", "threedgrut"]
    for path in args.requirements:
        requirements.extend(project_requirements(path))
    installed = active_distributions()
    protected = protected_closure(installed, requirements)
    graph = {name: dependencies(dist.requires or ()) for name, dist in installed.items()}
    remove = removal_plan(graph, UNUSED_ROOTS, protected)
    print(json.dumps({"removed_unused_ngc_packages": {name: installed[name].version for name in remove}}), flush=True)
    if remove:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *remove], check=True)


if __name__ == "__main__":
    main()
