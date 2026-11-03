# Explainability and robustness evaluation

Stage 5 evaluates **trained checkpoints only**. The scripts do not download
datasets or pretrained weights, initialize a replacement checkpoint silently,
or run on a training split. They use the configured test manifest and preserve
its labels and sample order. Results therefore depend on the exact checkpoints,
manifest, and dataset files supplied by the experiment.

## Grad-CAM

Generate overlays for the first test samples:

```bash
python scripts/generate_gradcam.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/efficientnet_b0/best.pt \
  --output-dir results/explainability
```

Use `--image path/to/image.jpg` one or more times for selected inputs. The
script uses the architecture-specific target layer in `configs/config.yaml`:
`features.8.0` for EfficientNet-B0 and
`features.9.main.2.block.3` for this repository's XceptionNet.
`--target-layer` (a valid dotted module path) can override selection;
`--target-class 0` or `--target-class 1` can request real or fake attribution
explicitly. Invalid images are recorded in
`results/explainability/metadata.json`; a missing checkpoint is a hard error.
Overlays are saved as PNG files and include the predicted real/fake class and
fake probability in the metadata.

## Robustness

Evaluate both models on exactly the same configured test samples:

```bash
python scripts/evaluate_robustness.py \
  --config configs/config.yaml \
  --checkpoint-efficientnet outputs/checkpoints/efficientnet_b0/best.pt \
  --checkpoint-xception outputs/checkpoints/xception/best.pt \
  --output results/robustness/metrics.json
```

The report includes a clean baseline and deterministic JPEG compression,
Gaussian noise, Gaussian blur, and downscale-then-resize conditions. The test
manifest is validated once before evaluation, so clean and every corruption
use the same ordered valid paths and labels. The report records requested,
valid, and invalid counts plus invalid paths; each condition records its
parameter, sample count, accuracy, precision, recall, F1-score, ROC-AUC when
defined, and a 2x2 confusion matrix. Noise uses `runtime.seed` plus the
ordered sample index. Settings are read from `configs/config.yaml` under
`robustness`; `configs/experiments.yaml` records that source rather than
duplicating the values.

## Kaggle GPU validation

After uploading the repository and prepared data/checkpoints to a Kaggle
notebook with an NVIDIA T4:

```bash
pip install -r requirements.txt
python scripts/run_lightweight_tests.py
python scripts/run_torch_tests.py
python scripts/generate_gradcam.py --config configs/config.yaml \
  --checkpoint outputs/checkpoints/efficientnet_b0/best.pt
python scripts/evaluate_robustness.py --config configs/config.yaml \
  --checkpoint-efficientnet outputs/checkpoints/efficientnet_b0/best.pt \
  --checkpoint-xception outputs/checkpoints/xception/best.pt
```

These are validation commands, not a full-dataset experiment. This repository
does not claim Stage 5 performance until the commands have been run with
actual trained checkpoints and the intended held-out test manifest.
