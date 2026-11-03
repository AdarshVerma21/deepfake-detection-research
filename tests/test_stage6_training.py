import pytest

torch = pytest.importorskip("torch")

from scripts.train import calculate_training_class_weight
from src.data.dataset import ManifestImageDataset
from src.training.engine import TrainingConfig


def test_training_weight_uses_only_training_samples(tmp_path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text("frame_path,label\nreal.jpg,0\nfake.jpg,1\nfake2.jpg,1\n", encoding="utf-8")
    dataset = ManifestImageDataset(manifest, root=tmp_path)
    negative, positive, weight = calculate_training_class_weight(dataset)
    assert (negative, positive) == (1, 2)
    assert weight == 0.5


def test_training_weight_requires_both_classes(tmp_path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text("frame_path,label\nreal.jpg,0\n", encoding="utf-8")
    dataset = ManifestImageDataset(manifest, root=tmp_path)
    with pytest.raises(ValueError, match="both"):
        calculate_training_class_weight(dataset)


def test_training_config_accepts_imbalance_modes() -> None:
    assert TrainingConfig(class_imbalance_mode="none").class_imbalance_mode == "none"
    assert TrainingConfig(class_imbalance_mode="weighted_loss", positive_class_weight=2).positive_class_weight == 2
