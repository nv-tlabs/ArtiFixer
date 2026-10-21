# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install real distribution metadata for a same-version VCS replacement."""

import json
import site
import sys
from pathlib import Path


def main():
    directory = Path(site.getsitepackages()[0]) / "source_revision_fixture-1.0.dist-info"
    directory.mkdir()
    (directory / "METADATA").write_text("Metadata-Version: 2.1\nName: source-revision-fixture\nVersion: 1.0\n")
    (directory / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "https://github.com/fixture-owner/review-package.git",
                "vcs_info": {"vcs": "git", "commit_id": sys.argv[1]},
            }
        )
    )
    if sys.argv[1] == "b" * 40:
        # FA3 is installed from a local source checkout in both x86 images:
        # the wheel metadata has its version, but no direct_url.json commit.
        fa3 = directory.parent / "flash_attn_3-3.0.0.dist-info"
        fa3.mkdir()
        (fa3 / "METADATA").write_text("Metadata-Version: 2.1\nName: flash_attn_3\nVersion: 3.0.0\n")
    Path("/opt/artifixer-source-collection-fixture").touch()


if __name__ == "__main__":
    main()
