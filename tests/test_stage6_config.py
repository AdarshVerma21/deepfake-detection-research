from pathlib import Path

from src.utils.config import load_yaml_config


def test_dataset_config_keeps_image_task_and_frame_path_task() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / "configs" / "dataset.yaml")["dataset"]
    assert config["image_detection"]["image_column"] == "filepath"
    assert config["video_frame_detection"]["frame_column"] == "frame_path"
    assert config["video_frame_detection"]["root"] == "data/processed/faceforensics_c23"
