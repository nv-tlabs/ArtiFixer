# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from data_processing import run_captioning
from data_processing.captioning import generate_captions, prompt_hdf5
from model_training.prompt_formats import COSMOS3_PROMPT_FORMAT, WAN_T5_PROMPT_FORMAT


@pytest.mark.parametrize("prompt_format", [WAN_T5_PROMPT_FORMAT, COSMOS3_PROMPT_FORMAT])
def test_scene_loader_passes_options_to_every_model_family(tmp_path, monkeypatch, prompt_format):
    calls = []

    def captioner(*args, **kwargs):
        calls.append(kwargs)
        return object(), object()

    def text_encoder(*args, **kwargs):
        calls.append(kwargs)
        return object(), object()

    def tokenizer(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(run_captioning, "get_caption_model_and_processor", captioner)
    monkeypatch.setattr(run_captioning, "get_text_encoder_and_tokenizer", text_encoder)
    monkeypatch.setattr(run_captioning, "get_cosmos3_tokenizer", tokenizer)
    run_captioning.load_models(
        SimpleNamespace(
            captioning_model_id="fixture",
            captioning_attn_implementation=None,
            text_encoder_model_id=None,
            prompt_format=prompt_format,
            hf_cache_dir=tmp_path,
            hf_hub_offline=True,
        )
    )
    assert calls == [{"cache_dir": tmp_path, "local_files_only": True}] * 2


def test_wan_text_encoder_and_tokenizer_receive_download_options(tmp_path, monkeypatch):
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(generate_captions.AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(generate_captions.UMT5EncoderModel, "from_pretrained", load)
    generate_captions.get_text_encoder_and_tokenizer("fixture", cache_dir=tmp_path, local_files_only=True)
    assert [call["subfolder"] for call in calls] == ["tokenizer", "text_encoder"]
    assert all(call["cache_dir"] == tmp_path and call["local_files_only"] for call in calls)


def test_cosmos_offline_cache_miss_does_not_fall_back_to_online(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import LocalEntryNotFoundError

    calls = []

    def miss(*args, **kwargs):
        calls.append(kwargs)
        raise LocalEntryNotFoundError("missing pinned tokenizer")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", miss)
    with pytest.raises(LocalEntryNotFoundError, match="missing pinned tokenizer"):
        prompt_hdf5.get_cosmos3_tokenizer("nvidia/Cosmos3-Nano", cache_dir=tmp_path, local_files_only=True)
    assert len(calls) == 1
    assert calls[0]["cache_dir"] == tmp_path
    assert calls[0]["local_files_only"] is True


def test_cli_options_reach_hub_after_libraries_have_been_imported(tmp_path):
    (tmp_path / "data/scene").mkdir(parents=True)
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from unittest.mock import patch
        import data_processing.run_captioning as captioning
        from huggingface_hub import constants, file_download
        assert not constants.HF_HUB_OFFLINE
        root = Path(sys.argv[1])
        captured = []
        class StopAtHub(Exception):
            pass
        def capture(**kwargs):
            captured.append(kwargs)
            raise StopAtHub
        sys.argv = ['run_captioning', '--dl3dv_dir', str(root / 'data'),
                    '--output_dir', str(root / 'prompts'), '--scene_id', 'scene',
                    '--num_frames', '0', '--hf_cache_dir', str(root / 'cache'),
                    '--hf_hub_offline', '--captioning_model_id', 'audit/nonexistent-fixture']
        with patch.object(file_download, '_hf_hub_download_to_cache_dir', side_effect=capture):
            try:
                captioning.main()
            except StopAtHub:
                pass
        assert captured, 'native model loader did not reach the download boundary'
        assert captured[0]['local_files_only'] is True
        assert Path(captured[0]['cache_dir']) == root / 'cache'
    """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("missing", ["directory", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"])
@pytest.mark.parametrize("offline", [False, True])
def test_cosmos_partial_snapshot_recovers_only_when_online(tmp_path, monkeypatch, missing, offline):
    import huggingface_hub
    import transformers
    from huggingface_hub.utils import LocalEntryNotFoundError

    from model_training.pins import COSMOS3_NANO_PINS

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    tokenizer_dir = snapshot / "text_tokenizer"
    assets = ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja")
    if missing != "directory":
        tokenizer_dir.mkdir()
        for name in assets:
            if name != missing:
                (tokenizer_dir / name).touch()
    calls = []

    def download(model_id, **kwargs):
        assert model_id == "nvidia/Cosmos3-Nano"
        assert kwargs["revision"] == COSMOS3_NANO_PINS.hf_revision
        assert kwargs["cache_dir"] == tmp_path
        assert kwargs["allow_patterns"] == ["text_tokenizer/*"]
        calls.append(kwargs.get("local_files_only", False))
        if not calls[-1]:
            tokenizer_dir.mkdir(exist_ok=True)
            for name in assets:
                (tokenizer_dir / name).touch()
        return str(snapshot)

    loaded = []

    def load(path):
        assert all((path / name).is_file() for name in assets)
        loaded.append(path)
        return "complete tokenizer"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    if offline:
        with pytest.raises(LocalEntryNotFoundError, match="Incomplete pinned tokenizer"):
            prompt_hdf5.get_cosmos3_tokenizer("nvidia/Cosmos3-Nano", cache_dir=tmp_path, local_files_only=True)
        assert calls == [True]
        assert not loaded
    else:
        assert prompt_hdf5.get_cosmos3_tokenizer("nvidia/Cosmos3-Nano", cache_dir=tmp_path) == "complete tokenizer"
        assert calls == [True, False]
        assert loaded == [tokenizer_dir]
