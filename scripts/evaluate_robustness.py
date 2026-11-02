"""Evaluate two trained checkpoints on identical deterministic corruptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.robustness import (
    RobustnessConfig,
    evaluate_checkpoint,
    load_manifest_samples,
    transformation_specs,
)
from src.models.classifiers import create_model
from src.utils.config import load_yaml_config
from src.utils.device import get_runtime_device


def _tuple(config: dict, key: str, cast: type) -> tuple:
    return tuple(cast(value) for value in config.get(key, ()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--checkpoint-efficientnet", required=True)
    parser.add_argument("--checkpoint-xception", required=True)
    parser.add_argument("--output", default="results/robustness/metrics.json")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    dataset_cfg = load_yaml_config(args.dataset_config)["dataset"]
    task_cfg = dataset_cfg[dataset_cfg.get("task", "image_detection")]
    robustness_cfg = config.get("robustness", {})
    corruption = RobustnessConfig(
        jpeg_qualities=_tuple(robustness_cfg, "jpeg_qualities", int),
        gaussian_noise_stds=_tuple(robustness_cfg, "gaussian_noise_stds", float),
        gaussian_blur_kernels=_tuple(robustness_cfg, "gaussian_blur_kernels", int),
        downscale_ratios=_tuple(robustness_cfg, "downscale_ratios", float),
        seed=int(config["runtime"]["seed"]),
    )
    samples = load_manifest_samples(
        task_cfg["test_csv"],
        root=task_cfg.get("root"),
        image_column=task_cfg.get("image_column", task_cfg.get("frame_column", "filepath")),
        label_column=task_cfg["label_column"],
    )
    device = get_runtime_device(config["runtime"]["device"], config["runtime"]["mixed_precision"]).device
    model_results = {}
    for name, checkpoint in (
        ("efficientnet_b0", args.checkpoint_efficientnet),
        ("xception", args.checkpoint_xception),
    ):
        model = create_model(name, num_classes=1)
        model_results[name] = evaluate_checkpoint(
            model,
            checkpoint,
            samples,
            image_size=config["training"]["image_size"],
            config=corruption,
            device=device,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset_manifest": task_cfg["test_csv"],
                "sample_count": len(samples),
                "seed": corruption.seed,
                "transformations": transformation_specs(corruption),
                "models": model_results,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
