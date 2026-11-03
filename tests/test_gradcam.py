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


def test_configured_target_layers_exist_for_both_models() -> None:
    models_and_paths = [
        (create_model("efficientnet_b0"), "features.8.0"),
        (XceptionNet(width=4), "features.9.main.2.block.3"),
    ]
    for model, path in models_and_paths:
        layer, name = find_target_layer(model, path)
        assert isinstance(layer, torch.nn.Conv2d)
        assert name == path


def test_gradcam_returns_normalized_map_for_xception() -> None:
    model = XceptionNet(width=4)
    layer, _ = find_target_layer(model)
    with GradCAM(model.eval(), layer) as cam:
        heatmap, probability = cam(torch.randn(1, 3, 32, 32), target_class=1)
    assert heatmap.ndim == 2
    assert float(heatmap.min()) >= 0
    assert float(heatmap.max()) <= 1
    assert 0 <= probability <= 1


def test_gradcam_rejects_invalid_target_class() -> None:
    model = XceptionNet(width=4)
    layer, _ = find_target_layer(model)
    with GradCAM(model.eval(), layer) as cam:
        with pytest.raises(ValueError, match="target_class must be 0 or 1"):
            cam(torch.randn(1, 3, 32, 32), target_class=2)


def test_invalid_target_layer_reports_fallback() -> None:
    model = XceptionNet(width=4)
    with pytest.warns(RuntimeWarning, match="falling back"):
        layer, name = find_target_layer(model, "features.missing")
    assert isinstance(layer, torch.nn.Conv2d)
    assert name.startswith("features")


def test_hooks_are_removed_after_gradcam_exception() -> None:
    model = XceptionNet(width=4).eval()
    layer, _ = find_target_layer(model)
    cam = GradCAM(model, layer)
    with pytest.raises(ValueError):
        cam(torch.randn(1, 3, 32, 32), target_class=4)
    assert cam._closed
