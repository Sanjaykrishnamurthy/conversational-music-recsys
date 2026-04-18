"""Configuration loader using OmegaConf / YAML."""

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML config file into an OmegaConf DictConfig."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return OmegaConf.create(raw)


def save_config(cfg: DictConfig, output_path: str | Path) -> None:
    """Save an OmegaConf config to YAML."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(output_path))
