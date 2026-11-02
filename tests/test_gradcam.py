import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.explainability.gradcam import GradCAM, find_target_layer
from src.models.classifiers import XceptionNet, create_model


def test_gradcam_selects_convolution_for_both_models() -> None:
    models = [create_model("efficientnet_b0"), XceptionNet(width=4)]
    for model in models:
        layer, name = find_target_layer(model)
        assert isinstance(layer, torch.nn.Conv2d)
        assert name.startswith("features")


def test_gradcam_returns_normalized_map_for_xception() -> None:
    model = XceptionNet(width=4)
    layer, _ = find_target_layer(model)
    with GradCAM(model.eval(), layer) as cam:
        heatmap, probability = cam(torch.randn(1, 3, 32, 32), target_class=1)
    assert heatmap.ndim == 2
    assert float(heatmap.min()) >= 0
    assert float(heatmap.max()) <= 1
    assert 0 <= probability <= 1
