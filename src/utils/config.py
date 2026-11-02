"""Configuration utilities for YAML-based experiment control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.errors import ConfigurationError


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML file as a dictionary with validation."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Top-level YAML structure must be a mapping: {config_path}")

    return loaded
