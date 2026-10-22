# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from data_processing.sparse_recon.workflow import copy_reconstruction_outputs, reconstruction_outputs


def test_interrupted_replacement_bundle_is_not_complete(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    outputs = reconstruction_outputs(tmp_path / "output", "dl3dv_1K", "scene", "0", 3)
    for name in ("data.h5", "parsed.yaml", "ckpt_last.pt"):
        (source / name).write_bytes(b"old")
    copy_reconstruction_outputs(source, outputs)
    assert outputs.complete()
    for path in source.iterdir():
        path.write_bytes(b"new")
    replace = Path.replace

    def interrupt(path, target):
        if target == outputs.parsed_yaml:
            raise OSError("publication interrupted")
        return replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt)
        with pytest.raises(OSError, match="publication interrupted"):
            copy_reconstruction_outputs(source, outputs)
    assert outputs.data_h5.read_bytes() == b"new"
    assert outputs.checkpoint.read_bytes() == b"old"
    assert not outputs.complete()
    copy_reconstruction_outputs(source, outputs)
    assert outputs.complete()
    assert all(path.read_bytes() == b"new" for path in outputs.paths())
