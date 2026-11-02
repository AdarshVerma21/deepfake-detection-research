"""EfficientNet-B0 and a reproducible Xception classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class XceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reps: int, stride: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(reps):
            layers.append(SeparableConv2d(in_channels if index == 0 else out_channels, out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.pool = nn.MaxPool2d(3, stride=stride, padding=1) if stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.main(x)) + self.skip(x)


class XceptionNet(nn.Module):
    """Xception-style network based on depthwise separable convolutions.

    The implementation follows the published Xception entry, middle, and exit
    flow pattern while keeping widths configurable and CPU-test friendly.
    """

    def __init__(self, num_classes: int = 1, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            XceptionBlock(width * 2, width * 4, 2, 2),
            XceptionBlock(width * 4, width * 8, 2, 2),
            XceptionBlock(width * 8, width * 16, 2, 2),
            XceptionBlock(width * 16, width * 32, 2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).flatten(1))


def _load_efficientnet(weights_path: str | Path | None) -> nn.Module:
    try:
        from torchvision.models import efficientnet_b0
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("EfficientNet-B0 requires a compatible torchvision installation.") from exc
    model = efficientnet_b0(weights=None)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    return model


def create_model(
    name: str,
    num_classes: int = 1,
    pretrained: bool = False,
    weights_path: str | Path | None = None,
    xception_width: int = 32,
) -> nn.Module:
    """Create a binary-logit classifier without downloading weights."""
    normalized = name.lower().replace("-", "_")
    if normalized == "xception":
        if pretrained and not weights_path:
            raise ValueError("Xception pretrained=True requires an explicit local weights_path.")
        model: nn.Module = XceptionNet(num_classes=num_classes, width=xception_width)
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=False)
        return model
    if normalized == "efficientnet_b0":
        if pretrained and not weights_path:
            raise ValueError("EfficientNet pretrained=True requires an explicit local weights_path; downloads are disabled.")
        model = _load_efficientnet(weights_path)
        if not hasattr(model, "classifier") or not isinstance(model.classifier, nn.Sequential):
            raise RuntimeError("Unexpected torchvision EfficientNet classifier structure.")
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {name}")
