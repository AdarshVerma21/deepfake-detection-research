"""Grad-CAM utilities for the repository's binary image classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
from PIL import Image


def _module_from_path(model: Any, path: str) -> Any:
    module = model
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def find_target_layer(model: Any, requested: str | None = None) -> tuple[Any, str]:
    """Return a configured layer or the last convolution with spatial output."""
    import torch.nn as nn

    if requested:
        try:
            layer = _module_from_path(model, requested)
            if isinstance(layer, nn.Conv2d):
                return layer, requested
            reason = "it is not a Conv2d layer"
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            reason = f"it does not exist ({exc})"
        warnings.warn(
            f"Invalid Grad-CAM target layer {requested!r}; falling back to the last "
            f"spatial convolution ({reason}).",
            RuntimeWarning,
            stacklevel=2,
        )
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d) and name.startswith("features")
    ]
    if not candidates:
        raise ValueError("Model has no supported convolutional feature layer for Grad-CAM.")
    return candidates[-1][1], candidates[-1][0]


class GradCAM:
    """Compute a Grad-CAM map for a binary-logit model."""

    def __init__(self, model: Any, target_layer: Any) -> None:
        self.model = model
        self.activations: Any = None
        self.gradients: Any = None
        self._closed = False
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.activations = output

    def _save_gradient(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.gradients = output[0]

    def close(self) -> None:
        if not self._closed:
            self._forward_handle.remove()
            self._backward_handle.remove()
            self._closed = True

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __call__(self, image_tensor: Any, target_class: int) -> tuple[Any, float]:
        import torch

        if target_class not in (0, 1):
            raise ValueError(f"target_class must be 0 or 1, got {target_class!r}")
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(image_tensor).reshape(-1)
            if logits.numel() == 0:
                raise RuntimeError("Model produced no logits for Grad-CAM.")
            logit = logits[0]
            score = logit if target_class == 1 else -logit
            score.backward()
            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
            if self.activations.ndim != 4 or self.gradients.ndim != 4:
                raise ValueError("Grad-CAM target layer must produce spatial feature maps.")
            if self.activations.shape[-2:] != self.gradients.shape[-2:]:
                raise RuntimeError("Grad-CAM activation and gradient spatial shapes differ.")
            if not torch.isfinite(self.activations).all() or not torch.isfinite(self.gradients).all():
                raise RuntimeError("Grad-CAM activations or gradients contain non-finite values.")
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(dim=1))[0]
            if not torch.isfinite(cam).all():
                raise RuntimeError("Grad-CAM heatmap contains non-finite values.")
            cam = cam - cam.min()
            maximum = cam.max()
            if not torch.isfinite(maximum):
                raise RuntimeError("Grad-CAM heatmap maximum is non-finite.")
            if float(maximum) > 0:
                cam = cam / maximum
            cam = torch.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            return cam.detach().cpu(), float(torch.sigmoid(logit.detach()))
        finally:
            self.close()


def _overlay(image: Image.Image, cam: Any) -> Image.Image:
    import matplotlib.cm as cm

    heatmap = (cm.get_cmap("jet")(cam.numpy())[..., :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap, mode="RGB").resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), heatmap_image, alpha=0.45)


def generate_gradcam(
    model: Any,
    image_path: str | Path,
    output_path: str | Path,
    image_size: int = 224,
    target_layer: str | None = None,
    device: Any = "cpu",
    target_class: int | None = None,
) -> dict[str, Any]:
    """Generate and save one overlay, returning prediction metadata."""
    import torch
    from src.data.dataset import build_transforms

    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {source}")
    try:
        with Image.open(source) as opened:
            original = opened.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to read input image {source}: {exc}") from exc

    model = model.to(device).eval()
    layer, selected_layer = find_target_layer(model, target_layer)
    tensor = build_transforms(image_size)(original).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(tensor).reshape(-1)[0]).cpu())
    predicted_class = int(probability >= 0.5)
    selected_class = predicted_class if target_class is None else target_class
    if selected_class not in (0, 1):
        raise ValueError(f"target_class must be 0 or 1, got {selected_class!r}")
    with GradCAM(model, layer) as cam:
        with torch.enable_grad():
            heatmap, _ = cam(tensor, selected_class)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _overlay(original, heatmap).save(destination)
    return {
        "input": str(source),
        "output": str(destination),
        "probability_fake": probability,
        "predicted_label": predicted_class,
        "target_class": selected_class,
        "target_layer": selected_layer,
    }
