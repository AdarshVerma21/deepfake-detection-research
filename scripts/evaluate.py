"""Evaluate a trained checkpoint on the held-out test manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset, build_transforms
from src.evaluation.metrics import calculate_metrics
from src.models.classifiers import create_model
from src.training.engine import load_checkpoint
from src.utils.config import load_yaml_config
from src.utils.device import get_runtime_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/test_metrics.json")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    runtime, training, model_cfg = config["runtime"], config["training"], config["model"]
    dataset_cfg = load_yaml_config("configs/dataset.yaml")["dataset"]
    task_cfg = dataset_cfg[dataset_cfg.get("task", "image_detection")]
    device = get_runtime_device(runtime["device"], runtime["mixed_precision"]).device
    dataset = ManifestImageDataset(
        task_cfg["test_csv"],
        root=task_cfg.get("root"),
        image_column=task_cfg.get("image_column", task_cfg.get("frame_column", "filepath")),
        label_column=task_cfg["label_column"],
        transform=build_transforms(training["image_size"]),
    )
    loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=False, num_workers=runtime["num_workers"])
    model = create_model(model_cfg["name"], num_classes=1, weights_path=model_cfg.get("weights_path"))
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device).eval()
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            probabilities.extend(torch.sigmoid(model(images.to(device)).reshape(-1)).cpu().tolist())
            labels.extend(targets.int().tolist())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calculate_metrics(labels, probabilities), indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
