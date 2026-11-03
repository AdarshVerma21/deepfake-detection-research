"""Deterministic corruption transforms and checkpoint-based robustness evaluation."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageFilter

from src.evaluation.metrics import calculate_metrics


@dataclass(frozen=True)
class RobustnessConfig:
    jpeg_qualities: tuple[int, ...] = (95, 75, 50)
    gaussian_noise_stds: tuple[float, ...] = (0.01, 0.03, 0.05)
    gaussian_blur_kernels: tuple[int, ...] = (3, 5, 7)
    downscale_ratios: tuple[float, ...] = (0.75, 0.5, 0.25)
    seed: int = 42

    def __post_init__(self) -> None:
        if any(not 1 <= value <= 100 for value in self.jpeg_qualities):
            raise ValueError("JPEG qualities must be in [1, 100].")
        if any(value < 0 for value in self.gaussian_noise_stds):
            raise ValueError("Gaussian noise standard deviations must be non-negative.")
        if any(value < 1 or value % 2 == 0 for value in self.gaussian_blur_kernels):
            raise ValueError("Gaussian blur kernels must be positive odd integers.")
        if any(not 0 < value <= 1 for value in self.downscale_ratios):
            raise ValueError("Downscale ratios must be in (0, 1].")


def load_manifest_samples(
    manifest_path: str | Path,
    root: str | Path | None = None,
    split: str | None = "test",
    image_column: str = "filepath",
    label_column: str = "label",
) -> list[tuple[Path, int]]:
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Test manifest not found: {manifest}")
    base = Path(root) if root is not None else manifest.parent
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest contains no samples for split={split!r}: {manifest}")
    required = {image_column, label_column}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing required columns {sorted(missing)}: {manifest}")
    if split and rows and "split" in rows[0]:
        rows = [row for row in rows if row.get("split") == split]
    if not rows:
        raise ValueError(f"Manifest contains no samples for split={split!r}: {manifest}")
    samples: list[tuple[Path, int]] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            label = int(row[label_column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid label on manifest row {row_number}: {manifest}") from exc
        if label not in (0, 1):
            raise ValueError(f"Labels must be 0 or 1 on manifest row {row_number}: {manifest}")
        filepath = row.get(image_column, "").strip()
        if not filepath:
            raise ValueError(f"Empty image path on manifest row {row_number}: {manifest}")
        samples.append((base / filepath, label))
    return samples


def validate_samples(
    samples: Iterable[tuple[Path, int]],
) -> tuple[list[tuple[Path, int, Image.Image]], list[dict[str, str | int]]]:
    """Read each sample once and return a fixed valid population plus invalid records."""
    valid: list[tuple[Path, int, Image.Image]] = []
    invalid: list[dict[str, str | int]] = []
    for path, label in samples:
        try:
            with Image.open(path) as opened:
                valid.append((path, label, opened.convert("RGB")))
        except (OSError, ValueError) as exc:
            invalid.append({"path": str(path), "label": label, "error": str(exc)})
    return valid, invalid


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def apply_gaussian_noise(image: Image.Image, standard_deviation: float, seed: int) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = np.random.default_rng(seed).normal(0.0, standard_deviation, array.shape)
    return Image.fromarray(np.uint8(np.clip(array + noise, 0.0, 1.0) * 255), mode="RGB")


def apply_gaussian_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=(kernel_size - 1) / 2))


def apply_downscale_resize(image: Image.Image, ratio: float) -> Image.Image:
    original = image.convert("RGB")
    width = max(1, round(original.width * ratio))
    height = max(1, round(original.height * ratio))
    return original.resize((width, height), Image.Resampling.LANCZOS).resize(original.size, Image.Resampling.LANCZOS)


def transformation_specs(config: RobustnessConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"name": "clean", "parameter": None}]
    specs.extend({"name": "jpeg_compression", "parameter": quality} for quality in config.jpeg_qualities)
    specs.extend({"name": "gaussian_noise", "parameter": std} for std in config.gaussian_noise_stds)
    specs.extend({"name": "gaussian_blur", "parameter": kernel} for kernel in config.gaussian_blur_kernels)
    specs.extend({"name": "downscale_resize", "parameter": ratio} for ratio in config.downscale_ratios)
    return specs


def apply_transformation(
    image: Image.Image,
    name: str,
    parameter: Any,
    seed: int,
) -> Image.Image:
    if name == "clean":
        return image.convert("RGB")
    transforms: dict[str, Callable[..., Image.Image]] = {
        "jpeg_compression": apply_jpeg_compression,
        "gaussian_noise": apply_gaussian_noise,
        "gaussian_blur": apply_gaussian_blur,
        "downscale_resize": apply_downscale_resize,
    }
    if name not in transforms:
        raise ValueError(f"Unsupported robustness transformation: {name}")
    if name == "gaussian_noise":
        return transforms[name](image, parameter, seed)
    return transforms[name](image, parameter)


def evaluate_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    samples: list[tuple[Path, int]] | list[tuple[Path, int, Image.Image]],
    image_size: int,
    config: RobustnessConfig,
    device: Any = "cpu",
    invalid_samples: list[dict[str, str | int]] | None = None,
    requested_sample_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate every corruption with the same ordered sample list."""
    import torch
    from src.data.dataset import build_transforms
    from src.training.engine import load_checkpoint

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint}")
    requested_count = len(samples) if requested_sample_count is None else requested_sample_count
    if requested_count == 0:
        raise ValueError("Robustness evaluation requires at least one requested sample.")
    if samples and len(samples[0]) == 3:
        valid_samples = [
            (path, label, image)
            for path, label, image in samples
        ]
        invalid_records = invalid_samples or []
    else:
        valid_samples, invalid_records = validate_samples(
            [(path, label) for path, label in samples]
        )
    if not valid_samples:
        raise ValueError("Robustness evaluation found no valid images in the manifest.")
    load_checkpoint(checkpoint, model, device=device)
    model.to(device).eval()
    preprocess = build_transforms(image_size)
    results: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "sample_count": len(valid_samples),
        "requested_sample_count": requested_count,
        "valid_sample_count": len(valid_samples),
        "invalid_sample_count": len(invalid_records),
        "invalid_samples": invalid_records,
        "transformations": {},
    }
    for spec in transformation_specs(config):
        labels: list[int] = []
        probabilities: list[float] = []
        for index, (_path, label, image) in enumerate(valid_samples):
            transformed = apply_transformation(image, spec["name"], spec["parameter"], config.seed + index)
            tensor = preprocess(transformed).unsqueeze(0).to(device)
            with torch.inference_mode():
                probabilities.append(float(torch.sigmoid(model(tensor).reshape(-1)[0]).cpu()))
            labels.append(label)
        metric_result = calculate_metrics(labels, probabilities)
        metric_result.update({"parameter": spec["parameter"], "sample_count": len(labels)})
        results["transformations"][spec["name"] if spec["parameter"] is None else f"{spec['name']}_{spec['parameter']}"] = metric_result
    return results
