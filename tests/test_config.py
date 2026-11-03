from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.errors import ConfigurationError


def test_load_yaml_config_returns_dict() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(repo_root / "configs" / "config.yaml")
    assert isinstance(config, dict)
    assert "training" in config


def test_load_yaml_config_rejects_duplicate_keys(tmp_path) -> None:
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text("training:\n  task: image_detection\ntraining:\n  task: video_frame_detection\n")
    try:
        load_yaml_config(config_path)
    except ConfigurationError as exc:
        assert "duplicate key" in str(exc)
    else:
        raise AssertionError("duplicate YAML keys must be rejected")
