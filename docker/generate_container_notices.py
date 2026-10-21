# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate THIRD-PARTY-NOTICES-CONTAINER.md from a built release image.

The table lists every pip and apt component the image adds on top of its NGC
base or replaces there, with the version actually installed, the license declared by the
component, and where its source lives. Generating it from the image (instead of
editing rows by hand) keeps the attribution artifact in step with the pins.

Usage:
    python docker/generate_container_notices.py <base-image> <final-image> [--output THIRD-PARTY-NOTICES-CONTAINER.md]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from container_inventory import changed_packages
from release_image import inspect_image, platform, platform_matches

INVENTORY = r"""
import importlib.metadata as md, json, os, re, subprocess
pip = {}
seen = set()
for d in md.distributions():
    name = d.metadata["Name"]
    key = re.sub(r"[-_.]+", "-", name).lower()
    if key in seen:
        continue
    seen.add(key)
    # NGC can expose vendored metadata for an older version. Select the same
    # distribution as importlib.metadata.version(), rather than allowing the
    # last duplicate encountered to overwrite the active version.
    d = md.distribution(name)
    name = d.metadata["Name"]
    lic = d.metadata.get("License-Expression") or ""
    if not lic:
        raw = (d.metadata.get("License") or "").strip()
        # A full license can begin with a copyright attribution (NumPy does).
        # Do not turn that first line into a supposed license identifier.
        first = raw.splitlines()[0] if raw and raw != "UNKNOWN" else ""
        copyright_line = first.lower().startswith(("copyright", "(c)", "©"))
        if first == raw and len(first) <= 120 and not copyright_line:
            lic = first
    if not lic:
        classifiers = [c for c in d.metadata.get_all("Classifier") or [] if c.startswith("License ::")]
        lic = "; ".join(sorted({c.split("::")[-1].strip() for c in classifiers}))
    if not lic and not copyright_line and len(first) <= 120 and re.search(r"\blicen[cs]e\b|\bpublic domain\b", first, re.I):
        lic = first
    url = ""
    for entry in d.metadata.get_all("Project-URL") or []:
        label, _, link = entry.partition(",")
        if label.strip().lower() in ("source", "source code", "repository", "homepage", "home"):
            url = link.strip(); break
    url = url or d.metadata.get("Home-page") or ""
    if not url:
        try:
            direct = json.loads(d.read_text("direct_url.json") or "{}")
            url = direct.get("url", "") if direct.get("url", "").startswith("http") else ""
        except Exception:
            url = ""
    direct = json.loads(d.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id", "")
    pip[name] = {"source_revision": revision, "version": d.version, "license": lic, "url": url or f"https://pypi.org/project/{name}/"}
apt = {}
out = subprocess.run(["dpkg-query", "-W", "-f", "${Package}\t${Version}\t${Homepage}\n"], capture_output=True, text=True, check=True).stdout
for line in out.splitlines():
    pkg, ver, home = (line.split("\t") + ["", ""])[:3]
    lic = ""
    cp = f"/usr/share/doc/{pkg}/copyright"
    if os.path.isfile(cp):
        found = []
        for l in open(cp, errors="replace"):
            if l.startswith("License:"):
                v = l.split(":", 1)[1].strip()
                if v and v not in found: found.append(v)
        lic = "; ".join(found[:8])
    apt[pkg] = {"version": ver, "license": lic, "url": home}
print(json.dumps({"pip": pip, "apt": apt}))
"""

# Rows whose license or provenance needs a statement the package metadata does
# not carry (verified against the upstream repositories).
OVERRIDES = {
    "fused_ssim": {"license": "MIT", "url": "https://github.com/rahul-goel/fused-ssim"},
    "quack-kernels": {"license": "Apache-2.0", "url": "https://github.com/Dao-AILab/quack"},
    "flash_attn_3": {"license": "BSD-3-Clause", "url": "https://github.com/Dao-AILab/flash-attention"},
    "flash-attn-4": {"license": "BSD-3-Clause", "url": "https://github.com/Dao-AILab/flash-attention"},
    "threedgrut": {
        "license": "Apache-2.0 (NVIDIA nv-tlabs/3DGRUT-ArtiFixer fork — NVIDIA first-party)",
        "url": "https://github.com/nv-tlabs/3DGRUT-ArtiFixer",
    },
    "cuda-toolkit": {
        "license": "NVIDIA Proprietary (CUDA EULA) — NVIDIA first-party, not 3rd-party OSS",
        "url": "https://pypi.org/project/cuda-toolkit/",
    },
}
# Components that are not pip or apt packages but ship in the image.
EXTRA_ROWS = [
    (
        "slangc (shader-slang)",
        "see scripts/install_slangc.sh",
        "Apache-2.0 WITH LLVM-exception",
        "https://github.com/shader-slang/slang",
    ),
    (
        "tiny-cuda-nn (vendored source in 3DGRUT submodule)",
        "git submodule",
        "BSD-3-Clause",
        "https://github.com/NVlabs/tiny-cuda-nn",
    ),
    (
        "lpipsPyTorch (vendored source in 3DGRUT submodule)",
        "git snapshot",
        "BSD-2-Clause (LPIPS)",
        "https://github.com/richzhang/PerceptualSimilarity",
    ),
]

HEADER = """<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Third-Party Notices — ArtiFixer Container Images

The pre-built ArtiFixer container images add or replace the components below
relative to the NVIDIA NGC PyTorch base image. Each Dockerfile defines its
default in `ARTIFIXER_BASE_IMAGE`; the build receipt records the selected digest.
Base-image component notices are provided on NGC. Prepare and verify source code with each
container release as `artifixer-container-sources.tar` on the corresponding
GitHub release (collected by `docker/collect_container_sources.sh`).

Vendored source code in this repository is covered by
[THIRD-PARTY-NOTICES.md](https://github.com/nv-tlabs/ArtiFixer/blob/main/THIRD-PARTY-NOTICES.md).

"""


def inventory(image: str, selected_platform: str) -> dict:
    out = subprocess.run(
        ["docker", "run", "--rm", "--platform", selected_platform, "--entrypoint", "python", image, "-c", INVENTORY],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def cell(text: str) -> str:
    return (text or "").replace("|", "\\|").strip() or "—"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base_image")
    parser.add_argument("final_image")
    parser.add_argument(
        "--platform",
        choices=("linux/amd64", "linux/arm64"),
        help="Image platform; defaults to the final image's platform.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "THIRD-PARTY-NOTICES-CONTAINER.md"
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="Earlier version of the table whose hand-verified apt license cells are reused where the image "
        "carries no machine-readable license (default: the current --output file).",
    )
    args = parser.parse_args(argv)
    previous_table = args.previous if args.previous is not None else args.output

    final_metadata = inspect_image(args.final_image, expected_platform=args.platform)
    selected_platform = args.platform or platform(final_metadata)
    base_metadata = inspect_image(args.base_image, expected_platform=selected_platform)
    if not platform_matches(selected_platform, platform(final_metadata)) or not platform_matches(
        selected_platform, platform(base_metadata)
    ):
        raise ValueError(f"Both images must provide {selected_platform}")
    args.base_image, args.final_image = base_metadata["Id"], final_metadata["Id"]
    base, final = inventory(args.base_image, selected_platform), inventory(args.final_image, selected_platform)
    added_pip = changed_packages(base["pip"], final["pip"])
    added_apt = changed_packages(base["apt"], final["apt"])
    for name, override in OVERRIDES.items():
        if name in added_pip:
            added_pip[name].update(override)

    # Many Debian copyright files (gcc, the X11 libraries) are not in the
    # machine-readable DEP-5 format, so no license can be read from the image.
    # Those rows keep the license text of the previous, hand-verified table,
    # but only for the same package version: a bumped package must be
    # re-verified (its row comes out empty and the run fails), and the rows
    # that were carried over are listed under the table so an inherited cell
    # is never mistaken for one read from the image. Reading --previous from
    # the file about to be overwritten is deliberate: the table is the record
    # of that verification, and the footnote keeps the inheritance visible.
    previous_licenses: dict[tuple[str, str], str] = {}
    if previous_table.is_file():
        for line in previous_table.read_text().splitlines():
            if not line.startswith("|"):
                continue
            # Split on unescaped pipes only: cell() writes embedded pipes as \|.
            cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
            if len(cells) == 4 and cells[2] not in ("License", "---", "—"):
                previous_licenses[(cells[0], cells[1])] = cells[2]
    carried_over = []
    for name, entry in added_apt.items():
        if not entry["license"] and (name, entry["version"]) in previous_licenses:
            entry["license"] = previous_licenses[(name, entry["version"])]
            carried_over.append(name)
        if not entry["url"]:
            # Packages without a Homepage field: point at the distribution's package page.
            entry["url"] = f"https://packages.ubuntu.com/noble/{name}"

    def rows(items: dict) -> list[str]:
        ordered = sorted(items.items(), key=lambda kv: (kv[1]["license"].lower(), kv[0].lower()))
        return [f"| {cell(n)} | {cell(v['version'])} | {cell(v['license'])} | {cell(v['url'])} |" for n, v in ordered]

    revision = (final_metadata["Config"].get("Labels") or {}).get("org.opencontainers.image.revision") or "unknown"
    lines = [
        HEADER,
        f"Generated by `docker/generate_container_notices.py` from `{args.final_image}` "
        f"(platform `{selected_platform}`, repository revision `{revision}`) against base `{args.base_image}` on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}. Regenerate after changing any pin.\n",
        "| Component | Version | License | Source |",
        "|---|---|---|---|",
        *rows(added_apt),
        *rows(added_pip),
        *[f"| {cell(n)} | {cell(v)} | {cell(l)} | {cell(u)} |" for n, v, l, u in EXTRA_ROWS],
        "",
    ]
    if carried_over:
        lines.append(
            "License text carried over from the previous revision of this table (same package version; the "
            "Debian copyright file in the image is not machine-readable, so it was verified by hand): "
            + ", ".join(f"`{name}`" for name in sorted(carried_over))
            + ".\n"
        )
    # Refuse before writing: --previous defaults to the file being replaced, so
    # an unverified row must not overwrite the hand-verified text it would
    # otherwise have inherited.
    unresolved = [n for n, v in {**added_apt, **added_pip}.items() if not v["license"]]
    if unresolved:
        print(
            f"NOT WRITTEN ({args.output} left untouched): no license metadata for "
            f"{', '.join(sorted(unresolved))}; verify each against its upstream and add it to OVERRIDES.",
            file=sys.stderr,
        )
        return 1
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}: {len(added_apt)} apt + {len(added_pip)} pip + {len(EXTRA_ROWS)} other components")
    return 0


if __name__ == "__main__":
    sys.exit(main())
