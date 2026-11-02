import numpy as np
from PIL import Image

from src.evaluation.metrics import calculate_metrics
from src.evaluation.robustness import (
    RobustnessConfig,
    apply_gaussian_noise,
    apply_transformation,
    transformation_specs,
)


def test_transformations_are_deterministic_and_preserve_size() -> None:
    image = Image.fromarray(np.full((16, 20, 3), 128, dtype=np.uint8), mode="RGB")
    config = RobustnessConfig()
    assert len(transformation_specs(config)) == 13
    first = apply_gaussian_noise(image, 0.05, seed=7)
    second = apply_gaussian_noise(image, 0.05, seed=7)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    transformed = apply_transformation(image, "downscale_resize", 0.5, seed=7)
    assert transformed.size == image.size


def test_robustness_config_rejects_invalid_blur_kernel() -> None:
    try:
        RobustnessConfig(gaussian_blur_kernels=(4,))
    except ValueError as exc:
        assert "odd" in str(exc)
    else:
        raise AssertionError("Expected invalid blur kernel to be rejected")


def test_robustness_metrics_include_required_outputs() -> None:
    result = calculate_metrics([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
