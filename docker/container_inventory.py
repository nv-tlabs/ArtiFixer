# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare added and replaced packages without treating a base version as a match."""

import argparse
import json
import re
from pathlib import Path


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def changed_packages(base: dict, final: dict) -> dict:
    baseline = {normalized_name(name): value for name, value in base.items()}
    return {
        name: value
        for name, value in final.items()
        if any(
            baseline.get(normalized_name(name), {}).get(field) != value.get(field)
            for field in ("version", "source_revision")
        )
    }


def read_inventory(path: Path) -> dict:
    if path.suffix == ".json":
        result = json.loads(path.read_text())
        for kind in ("pip", "apt"):
            if not isinstance(result.get(kind), dict) or not result[kind]:
                raise ValueError(f"Incomplete {kind} inventory: {path}")
            if any(not entry.get("version") for entry in result[kind].values()):
                raise ValueError(f"Malformed {kind} inventory: {path}")
        return result
    pip, separator, apt = path.read_text().partition("---APT---\n")
    if not separator:
        raise ValueError(f"Incomplete inventory: {path}")
    result = {}
    for kind, text, delimiter in (("pip", pip, "=="), ("apt", apt, "=")):
        packages = {}
        for line in text.splitlines():
            if not line:
                continue
            name, separator, version = line.partition(delimiter)
            if not separator or not name or not version:
                raise ValueError(f"Malformed {kind} inventory: {path}")
            packages[name] = {"version": version}
        if not packages:
            raise ValueError(f"Empty {kind} inventory: {path}")
        result[kind] = packages
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("final", type=Path)
    parser.add_argument("kind", choices=("pip", "apt"))
    args = parser.parse_args()
    packages = changed_packages(read_inventory(args.base)[args.kind], read_inventory(args.final)[args.kind])
    delimiter = "==" if args.kind == "pip" else "="
    for name, entry in sorted(packages.items()):
        print(f"{name}{delimiter}{entry['version']}")


if __name__ == "__main__":
    main()
