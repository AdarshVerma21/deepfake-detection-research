import csv

import numpy as np
from PIL import Image

from src.evaluation.metrics import calculate_metrics
from src.evaluation.robustness import (
    RobustnessConfig,
    apply_gaussian_noise,
    apply_transformation,
    load_manifest_samples,
    transformation_specs,
    validate_samples,
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
    assert result["pr_auc"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_manifest_validation_builds_one_fixed_population(tmp_path) -> None:
    valid_path = tmp_path / "valid.png"
    Image.new("RGB", (4, 4), "white").save(valid_path)
    manifest = tmp_path / "test.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filepath", "label"])
        writer.writerow([valid_path.name, 0])
        writer.writerow(["missing.png", 1])

    samples = load_manifest_samples(manifest, root=tmp_path)
    valid, invalid = validate_samples(samples)
    assert [path for path, _label, _image in valid] == [valid_path]
    assert [record["path"] for record in invalid] == [str(tmp_path / "missing.png")]
    assert len(valid) == 1
    assert len(invalid) == 1


def test_empty_manifest_is_rejected(tmp_path) -> None:
    manifest = tmp_path / "empty.csv"
    manifest.write_text("filepath,label\n", encoding="utf-8")
    try:
        load_manifest_samples(manifest, root=tmp_path)
    except ValueError as exc:
        assert "no samples" in str(exc)
    else:
        raise AssertionError("Expected an empty manifest to be rejected")
