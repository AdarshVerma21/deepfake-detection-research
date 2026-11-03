from pathlib import Path

from src.utils.config import load_yaml_config


def test_dataset_config_keeps_image_task_and_frame_path_task() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / "configs" / "dataset.yaml")["dataset"]
    assert config["image_detection"]["image_column"] == "filepath"
    assert config["video_frame_detection"]["frame_column"] == "frame_path"
    assert config["video_frame_detection"]["root"] == "data/processed/faceforensics_c23"


def test_training_task_selects_configured_dataset_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    training = load_yaml_config(root / "configs" / "config.yaml")["training"]
    dataset = load_yaml_config(root / "configs" / "dataset.yaml")["dataset"]
    assert training["task"] == "video_frame_detection"
    assert "image_detection" in dataset
    assert training["task"] in dataset
