# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""dcp_warm_build tests: pin the provenance-file format the launch gates
consume and the refusal paths for unattestable caches."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from model_training import pins
from model_training.model_profiles import COSMOS3_NANO_MODEL_ID, WAN22_TI2V_5B_MODEL_ID
from model_training.utils import dcp_warm_build, train_utils


class _ActionCapableTransformer(torch.nn.Module):
    """Carries all three action-projection key markers."""

    def __init__(self):
        super().__init__()
        self.action_proj_in = torch.nn.Linear(2, 2)
        self.action_proj_out = torch.nn.Linear(2, 2)
        self.action_modality_embed = torch.nn.Parameter(torch.zeros(2))


class _ActionLessTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = torch.nn.Linear(2, 2)


def _install_fake_loader(monkeypatch, transformer_cls):
    class _FakeTransformerCls:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = transformer_cls()
            if kwargs.get("output_loading_info"):
                return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
            return model

    monkeypatch.setattr(train_utils, "_transformer_cls_for_model", lambda model_id: _FakeTransformerCls)
    monkeypatch.setattr(train_utils.torch.cuda, "empty_cache", lambda: None)


def _run(model_id, dcp_cache_dir, code_sha="feedc0de"):
    return dcp_warm_build.main(["--model_id", model_id, "--dcp_cache_dir", str(dcp_cache_dir), "--code_sha", code_sha])


def test_fresh_build_writes_marker_and_the_exact_gated_provenance_lines(monkeypatch, tmp_path, capsys):
    _install_fake_loader(monkeypatch, _ActionCapableTransformer)

    assert _run(COSMOS3_NANO_MODEL_ID, tmp_path, code_sha="feedc0de") == 0

    cache_path = tmp_path / pins.dcp_cache_dir_name(COSMOS3_NANO_MODEL_ID) / "transformer"
    assert (cache_path / ".dcp_complete").is_file()
    provenance_lines = (cache_path / "codex_source_provenance.txt").read_text().splitlines()
    # The templates gate these with grep -Fxq (rq_gate_marker_line): the EXACT
    # lines must be present.
    assert f"model_revision={pins.COSMOS3_NANO_HF_REVISION}" in provenance_lines
    assert "code_sha=feedc0de" in provenance_lines
    assert f"diffusers_commit={pins.DIFFUSERS_COMMIT}" in provenance_lines
    assert "DCP_WARM_BUILD=built" in capsys.readouterr().out
    # The converted payload is a loadable DCP checkpoint with the action keys.
    checkpoint_keys = set(dcp.FileSystemReader(str(cache_path)).read_metadata().state_dict_metadata)
    assert any("action_proj_out" in key for key in checkpoint_keys)


def test_rerun_with_matching_pins_verifies_idempotently(monkeypatch, tmp_path, capsys):
    _install_fake_loader(monkeypatch, _ActionCapableTransformer)
    _run(COSMOS3_NANO_MODEL_ID, tmp_path, code_sha="feedc0de")
    capsys.readouterr()

    assert _run(COSMOS3_NANO_MODEL_ID, tmp_path, code_sha="feedc0de") == 0

    assert "DCP_WARM_BUILD=verified" in capsys.readouterr().out


def test_rerun_at_a_different_code_sha_refuses_to_reattest(monkeypatch, tmp_path):
    """The launch gates require provenance code_sha == launch SHA; silently
    restamping an old conversion would falsify that attestation."""
    _install_fake_loader(monkeypatch, _ActionCapableTransformer)
    _run(COSMOS3_NANO_MODEL_ID, tmp_path, code_sha="feedc0de")

    with pytest.raises(SystemExit, match="delete the cache"):
        _run(COSMOS3_NANO_MODEL_ID, tmp_path, code_sha="0ddba11")

    # The original attestation is untouched.
    cache_path = tmp_path / pins.dcp_cache_dir_name(COSMOS3_NANO_MODEL_ID) / "transformer"
    assert "code_sha=feedc0de" in (cache_path / "codex_source_provenance.txt").read_text().splitlines()


def test_complete_cache_without_provenance_is_refused(monkeypatch, tmp_path):
    """A trainer-side first-start conversion writes only .dcp_complete; its
    source cannot be attested for a gated launch."""
    _install_fake_loader(monkeypatch, _ActionCapableTransformer)
    cache_path = tmp_path / pins.dcp_cache_dir_name(COSMOS3_NANO_MODEL_ID) / "transformer"
    cache_path.mkdir(parents=True)
    dcp.save(
        {"model": _ActionCapableTransformer().state_dict()},
        storage_writer=dcp.FileSystemWriter(str(cache_path)),
        no_dist=True,
    )
    (cache_path / ".dcp_complete").touch()

    with pytest.raises(SystemExit, match="cannot be attested"):
        _run(COSMOS3_NANO_MODEL_ID, tmp_path)


def test_unpinned_families_are_refused(monkeypatch, tmp_path):
    """No launch gate accepts an unattested model_revision, so the warm build
    refuses model ids without a pin-registry record instead of stamping one."""
    _install_fake_loader(monkeypatch, _ActionLessTransformer)

    with pytest.raises(SystemExit, match="pin registry"):
        _run(WAN22_TI2V_5B_MODEL_ID, tmp_path)


def _cache_worker(rank, init_file, cache_root, failure):
    import time

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:

        class TinyLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                if failure:
                    raise OSError("conversion failed")
                return _ActionCapableTransformer(), {"missing_keys": []}

        train_utils._transformer_cls_for_model = lambda _: TinyLoader
        marker = Path(cache_root) / pins.dcp_cache_dir_name(COSMOS3_NANO_MODEL_ID) / "transformer/.dcp_complete"
        if rank == 1 and not failure:
            deadline = time.monotonic() + 15
            while not marker.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("rank zero did not write cache marker")
                time.sleep(0.02)
        if failure:
            with pytest.raises(RuntimeError, match="conversion failed"):
                train_utils._ensure_dcp_cache(COSMOS3_NANO_MODEL_ID, Path(cache_root))
        else:
            cache = train_utils._ensure_dcp_cache(COSMOS3_NANO_MODEL_ID, Path(cache_root))
            assert (cache / ".dcp_complete").is_file()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", [False, True])
def test_distributed_cache_observations_and_build_failure_keep_ranks_aligned(tmp_path, failure):
    import time

    workers = mp.spawn(
        _cache_worker, args=(str(tmp_path / "group"), str(tmp_path / "cache"), failure), nprocs=2, join=False
    )
    deadline = time.monotonic() + 60
    try:
        while not workers.join(timeout=1):
            assert time.monotonic() < deadline, "DCP cache ranks did not finish"
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
