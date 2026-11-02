import pytest
from pathlib import Path

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import calculate_metrics
from src.models.classifiers import XceptionNet, create_model
from src.training.engine import TrainingConfig, load_checkpoint, save_checkpoint, train_model


def test_xception_binary_output_and_forward() -> None:
    model = XceptionNet(num_classes=1, width=4)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 1)


def test_efficientnet_binary_classifier_output() -> None:
    model = create_model("efficientnet_b0", num_classes=1)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 1)


def test_efficientnet_training_mode_forward_uses_batch_statistics() -> None:
    model = create_model("efficientnet_b0", num_classes=1)
    model.train()
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 1)


def test_metrics_include_required_outputs() -> None:
    metrics = calculate_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert metrics["accuracy"] == 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert "fake" in metrics["classification_report"]


def test_training_checkpoint_save_load_and_cpu_loop(tmp_path) -> None:
    model = XceptionNet(num_classes=1, width=4)
    data = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0.0, 1.0, 0.0, 1.0]))
    loader = DataLoader(data, batch_size=2)
    config = TrainingConfig(epochs=1, early_stopping_patience=1, mixed_precision=False, checkpoint_dir=str(tmp_path))
    result = train_model(model, loader, loader, "cpu", config)
    checkpoint = tmp_path / "manual.pt"
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(checkpoint, model, optimizer, None, 0, 0.1, [], config)
    loaded = load_checkpoint(checkpoint, model, optimizer, device="cpu")
    assert loaded["epoch"] == 0
    assert Path(result["best_checkpoint"]).exists()
