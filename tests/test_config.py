from pathlib import Path

from src.utils.config import load_yaml_config


def test_load_yaml_config_returns_dict() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(repo_root / "configs" / "config.yaml")
    assert isinstance(config, dict)
    assert "training" in config
