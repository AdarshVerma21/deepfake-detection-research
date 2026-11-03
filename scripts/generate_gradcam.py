"""Generate Grad-CAM overlays from a trained checkpoint and manifest images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataset import ManifestImageDataset
from src.explainability.gradcam import generate_gradcam
from src.models.classifiers import create_model
from src.training.engine import load_checkpoint
from src.utils.config import load_yaml_config
from src.utils.device import get_runtime_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", help="CSV manifest; defaults to the configured test manifest.")
    parser.add_argument("--image", action="append", help="Input image path; repeat for multiple images.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-dir", default="results/explainability")
    parser.add_argument("--target-layer")
    parser.add_argument("--target-class", type=int, choices=(0, 1))
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    runtime, training, model_cfg = config["runtime"], config["training"], config["model"]
    target_layers = config.get("explainability", {}).get("grad_cam_target_layers", {})
    dataset_cfg = load_yaml_config("configs/dataset.yaml")["dataset"]
    task_cfg = dataset_cfg[dataset_cfg.get("task", "image_detection")]
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint}")
    device = get_runtime_device(runtime["device"], runtime["mixed_precision"]).device
    model = create_model(model_cfg["name"], num_classes=1)
    load_checkpoint(checkpoint, model, device=device)

    image_paths = [Path(path) for path in args.image or []]
    if not image_paths:
        manifest = args.manifest or task_cfg["test_csv"]
        dataset = ManifestImageDataset(
            manifest,
            root=task_cfg.get("root"),
            image_column=task_cfg.get("image_column", task_cfg.get("frame_column", "filepath")),
            label_column=task_cfg["label_column"],
        )
        image_paths = [path for path, _label in dataset.samples[: args.limit]]
    output_dir = Path(args.output_dir)
    records: list[dict[str, object]] = []
    for image_path in image_paths:
        destination = output_dir / f"{image_path.stem}_gradcam.png"
        try:
            records.append(
                generate_gradcam(
                    model,
                    image_path,
                    destination,
                    image_size=training["image_size"],
                    target_layer=args.target_layer or target_layers.get(model_cfg["name"]),
                    device=device,
                    target_class=args.target_class,
                )
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            records.append({"input": str(image_path), "error": str(exc)})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
