# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from threedgrut.trainer import Trainer3DGRUT

DEFAULT_THREEDGRUT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "thirdparty" / "3DGRUT-ArtiFixer" / "configs"


def register_3dgrut_resolvers() -> None:
    if not OmegaConf.has_resolver("int_list"):
        OmegaConf.register_new_resolver("int_list", lambda values: [int(value) for value in values])


def compose_3dgrut_config(config_name: str, overrides: list[str], config_dir: Path) -> object:
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Missing 3DGRUT config directory: {config_dir}")

    register_3dgrut_resolvers()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def train_3dgrut(config_name: str, overrides: list[str], config_dir: Path) -> None:
    trainer = Trainer3DGRUT(compose_3dgrut_config(config_name, overrides, config_dir))
    trainer.run_training()
