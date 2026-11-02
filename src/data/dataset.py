"""Shared manifest-backed image dataset for image and video-frame experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(image_size: int, training: bool = False) -> transforms.Compose:
    """Build the same normalization pipeline for both model architectures."""
    operations: list[object] = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend([transforms.RandomHorizontalFlip()])
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)


class ManifestImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load image paths and binary labels from a CSV manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path | None = None,
        image_column: str = "filepath",
        label_column: str = "label",
        split: str | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.transform = transform
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if split is not None and "split" in (rows[0] if rows else {}):
            rows = [row for row in rows if row.get("split") == split]
        missing = [column for column in (image_column, label_column) if rows and column not in rows[0]]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        if not rows:
            raise ValueError(f"Manifest contains no samples: {self.manifest_path}")
        self.samples = [(self.root / row[image_column], int(row[label_column])) for row in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to load dataset image {path}: {exc}") from exc
        return tensor, torch.tensor(float(label), dtype=torch.float32)


def create_datasets(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    image_size: int,
    root: str | Path | None = None,
    image_column: str = "filepath",
    label_column: str = "label",
) -> tuple[ManifestImageDataset, ManifestImageDataset, ManifestImageDataset]:
    """Create train, validation, and test datasets with shared transforms."""
    return (
        ManifestImageDataset(train_csv, root, image_column, label_column, transform=build_transforms(image_size, True)),
        ManifestImageDataset(val_csv, root, image_column, label_column, transform=build_transforms(image_size)),
        ManifestImageDataset(test_csv, root, image_column, label_column, transform=build_transforms(image_size)),
    )
