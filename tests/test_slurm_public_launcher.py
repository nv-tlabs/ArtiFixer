# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the generic launcher without private scripts or a Slurm installation."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("payload_status", [0, 17])
def test_public_launcher_preserves_payload_status_and_duplicate_guard(tmp_path, payload_status):
    repo = tmp_path / "public"
    public_slurm = Path(__file__).resolve().parents[1] / "model_training" / "slurm"
    shutil.copytree(public_slurm, repo / "model_training" / "slurm")
    commands = tmp_path / "bin"
    commands.mkdir()
    stubs = {
        "scontrol": '[[ "$1 $2" == "show hostnames" ]] || exit 99\nprintf "localhost\\n"\n',
        "srun": 'while [[ "$1" == --* ]]; do shift; done\nexec "$@"\n',
        "accelerate": (
            'printf "%s\\n" "$WANDB_RUN_ID" "$WANDB_RESUME" "$@" > "$LAUNCH_RECORD"\n' 'exit "$PAYLOAD_STATUS"\n'
        ),
    }
    for name, body in stubs.items():
        command = commands / name
        command.write_text("#!/bin/bash\nset -eu\n" + body)
        command.chmod(0o755)

    project = tmp_path / "run"
    record = tmp_path / "launched.txt"
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "REPO_DIR": str(repo),
        "PROJECT_DIR": str(project),
        "SPLIT_PATH": str(tmp_path / "split.json"),
        "DL3DV_DIR": str(tmp_path / "dataset"),
        "PROMPT_DIR": str(tmp_path / "prompts"),
        "RUN_ID": "abc12345",
        "SLURM_JOB_ID": "9001",
        "SLURM_JOB_NODELIST": "localhost",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "SLURM_RESTART_COUNT": "0",
        "SLURM_NNODES": "1",
        "SLURM_PROCID": "0",
        "GPUS_PER_NODE": "1",
        "RESUME_EXISTING": "0",
        "CONTAINER_IMAGE": "",
        "EXTRA_ENV": "",
        "EXTRA_ACCELERATE_ARGS": "",
        "EXTRA_TRAIN_ARGS": "",
        "LAUNCH_RECORD": str(record),
        "PAYLOAD_STATUS": str(payload_status),
    }
    launcher = repo / "model_training" / "slurm" / "sample-slurm-submit.sh"
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == payload_status, result.stderr
    assert (project / "run_id.txt").read_text() == "abc12345\n"
    launched = record.read_text().splitlines()
    assert launched[:2] == ["abc12345", "never"]
    assert launched[launched.index("--module") + 1] == "model_training.train"
    assert launched[launched.index("--should_checkpoint_flag") + 1] == str(project / "should_checkpoint")

    record.unlink()
    duplicate = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=15)
    assert duplicate.returncode == 3
    assert "Fresh project already exists" in duplicate.stderr
    assert not record.exists()
