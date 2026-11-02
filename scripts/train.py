"""Train one configured model on CSV manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset, build_transforms
from src.models.classifiers import create_model
from src.training.engine import TrainingConfig, train_model
from src.utils.config import load_yaml_config
from src.utils.device import get_runtime_device
from src.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    runtime, training, model_cfg, paths = config["runtime"], config["training"], config["model"], config["paths"]
    dataset_root_cfg = load_yaml_config("configs/dataset.yaml")["dataset"]
    task = training.get("task", dataset_root_cfg.get("task", "image_detection"))
    csv_cfg = dataset_root_cfg[task]
    device_cfg = get_runtime_device(runtime["device"], runtime["mixed_precision"])
    output = Path(paths["checkpoints"]) / model_cfg["name"]
    logger = setup_logger(model_cfg["name"], Path(paths["logs"]) / model_cfg["name"])
    common = dict(
        root=csv_cfg.get("root"),
        image_column=csv_cfg.get("image_column", csv_cfg.get("frame_column", "filepath")),
        label_column=csv_cfg["label_column"],
    )
    train_data = ManifestImageDataset(csv_cfg["train_csv"], transform=build_transforms(training["image_size"], True), **common)
    val_data = ManifestImageDataset(csv_cfg["val_csv"], transform=build_transforms(training["image_size"]), **common)
    train_loader = DataLoader(train_data, batch_size=training["batch_size"], shuffle=True, num_workers=runtime["num_workers"])
    val_loader = DataLoader(val_data, batch_size=training["batch_size"], shuffle=False, num_workers=runtime["num_workers"])
    network = create_model(
        model_cfg["name"],
        num_classes=1,
        pretrained=bool(model_cfg.get("pretrained", False)),
        weights_path=model_cfg.get("weights_path"),
    )
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output / "final_config.yaml")
    train_config = {key: value for key, value in training.items() if key in TrainingConfig.__dataclass_fields__}
    train_config["checkpoint_dir"] = str(output)
    train_config["mixed_precision"] = device_cfg.use_mixed_precision
    train_model(
        network,
        train_loader,
        val_loader,
        device_cfg.device,
        TrainingConfig(**train_config),
        logger,
        args.resume_from,
    )


if __name__ == "__main__":
    main()
