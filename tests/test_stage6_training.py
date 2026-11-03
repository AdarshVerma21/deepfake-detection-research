import pytest

torch = pytest.importorskip("torch")

from scripts.train import calculate_training_class_weight
from src.data.dataset import ManifestImageDataset
from src.training.engine import TrainingConfig, build_binary_criterion, train_model


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


def test_none_mode_uses_unweighted_loss_without_both_classes(tmp_path) -> None:
    config = TrainingConfig(class_imbalance_mode="none")
    criterion = build_binary_criterion(config, torch.device("cpu"))
    assert criterion.pos_weight is None

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 1))
    data = torch.utils.data.TensorDataset(torch.zeros(2, 3, 4, 4), torch.zeros(2))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    result = train_model(
        model,
        loader,
        loader,
        "cpu",
        TrainingConfig(
            epochs=1,
            scheduler="none",
            early_stopping_patience=1,
            mixed_precision=False,
            checkpoint_dir=str(tmp_path),
        ),
    )
    assert result["history"]


def test_weighted_loss_mode_uses_positive_class_weight() -> None:
    criterion = build_binary_criterion(
        TrainingConfig(class_imbalance_mode="weighted_loss", positive_class_weight=2.5),
        torch.device("cpu"),
    )
    assert criterion.pos_weight is not None
    assert criterion.pos_weight.item() == 2.5
