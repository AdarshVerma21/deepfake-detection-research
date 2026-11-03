# Deepfake Detection Research

Comparative study project for:

**"Comparative Analysis of EfficientNet-B0 and XceptionNet for Explainable and Robust Deepfake Image and Video Detection."**

## Research Objectives

1. Deepfake image detection
2. Frame-based deepfake video detection
3. Comparative evaluation of EfficientNet-B0 and XceptionNet
4. Grad-CAM explainability integration
5. Robustness evaluation under image corruptions
6. External generalization evaluation on Celeb-DF v2
7. Reproducible experimental workflow for IEEE-style reporting

## Research Foundation

The academic foundation, evidence boundaries, research gap, research questions, and planned protocol are documented in:

- [Literature review](docs/literature_review.md)
- [Research gap](docs/research_gap.md)
- [Research questions](docs/research_questions.md)
- [Experimental methodology](docs/experimental_methodology.md)
- [Reference list and verification notes](docs/reference_list.md)
- [Explainability and robustness evaluation](docs/explainability_and_robustness.md)

The literature review contains 12 verified papers and clearly labels sources that could not be independently verified. No experimental results or unsupported novelty claims are made.

## Project Architecture

```text
deepfake-detection-research/
├── configs/                  # YAML configurations
├── src/
│   ├── data/                 # Data loading and preprocessing
│   ├── models/               # Model definitions and wrappers
│   ├── training/             # Training loops and optimization
│   ├── evaluation/           # Metrics and robustness evaluation
│   ├── explainability/       # Grad-CAM and interpretability
│   ├── inference/            # Inference pipelines for image/video frames
│   └── utils/                # Config, logging, device, reproducibility, errors
├── scripts/                  # Execution scripts
├── notebooks/                # Exploratory notebooks
├── tests/                    # Unit tests
├── docs/                     # Technical documentation
├── results/                  # Experiment outputs and reports
├── requirements.txt
├── .gitignore
└── main.py
```

## Models and training

The shared manifest dataset applies the same RGB conversion, resize, ImageNet
normalization, and train-only horizontal flip to both models. Official
FaceForensics++ source-video split assignments remain the only split boundary;
the test split is never used for model selection or threshold tuning.

- **EfficientNet-B0** uses `torchvision.models.efficientnet_b0` with its final
  layer replaced by one binary logit. ImageNet weights are optional and must be
  supplied as a local checkpoint; the code never downloads weights silently.
  The torchvision implementation and distributed weights are documented at
  [pytorch.org/vision](https://pytorch.org/vision/stable/models.html) and are
  distributed under their stated BSD-compatible licenses.
- **XceptionNet** is a reproducible Xception-style entry/middle/exit-flow
  network using depthwise separable convolutions, global average pooling, and a
  one-logit classifier. It is implemented locally from the architecture
  described by Chollet (2017); no external checkpoint is assumed.

Both models use BCE-with-logits, AdamW, a configurable scheduler, gradient
clipping, early stopping, best/last checkpoints, resume support, and optional
CUDA AMP. Each model writes to `outputs/checkpoints/<model>/` and
`outputs/logs/<model>/`.

### Training and evaluation

Create CSV manifests with `filepath` and `label` columns, update
`configs/dataset.yaml`, then run:

```bash
python scripts/train.py --config configs/config.yaml
python scripts/train.py --config configs/config.yaml --resume-from outputs/checkpoints/efficientnet_b0/last.pt
python scripts/evaluate.py --config configs/config.yaml \
  --checkpoint outputs/checkpoints/efficientnet_b0/best.pt \
  --output results/efficientnet_b0/test_metrics.json

# Stage 5: Grad-CAM overlays and paired robustness evaluation
python scripts/generate_gradcam.py --config configs/config.yaml \
  --checkpoint outputs/checkpoints/efficientnet_b0/best.pt
python scripts/evaluate_robustness.py --config configs/config.yaml \
  --checkpoint-efficientnet outputs/checkpoints/efficientnet_b0/best.pt \
  --checkpoint-xception outputs/checkpoints/xception/best.pt
```

Grad-CAM outputs are written to `results/explainability/`; robustness metrics
are written to `results/robustness/metrics.json`. Both commands require
existing trained checkpoints and use the configured held-out test manifest.
See [Explainability and robustness evaluation](docs/explainability_and_robustness.md)
for corruption settings, failure handling, and Kaggle GPU validation commands.

Set `training.task` to `video_frame_detection` for the frame manifest. The
legacy `image_detection` section remains supported. To
compare Xception, set `model.name: xception` and use a separate output
directory or experiment configuration. The reported metrics are frame/image
level; they are not video-level metrics and no temporal aggregation is
implemented here.

### FaceForensics++ preprocessing-to-training integration

The preparation command preserves official train/validation/test assignments,
writes `train.csv`, `val.csv`, and `test.csv`, and validates every
`frame_path` relative to the preprocessing output directory. It rejects
unknown or empty splits, invalid labels, missing images, and video/source
video split leakage. `split_statistics.json` records per-split sample, class,
video, and source-video counts.

```bash
python scripts/prepare_faceforensics.py \
  --dataset-root /kaggle/working/FaceForensics \
  --split-dir /kaggle/working/FaceForensics/splits \
  --output-dir /kaggle/working/data/processed/faceforensics_c23 \
  --frames-per-video 10 \
  --batch-size 100
python scripts/train.py --config configs/config.yaml
python scripts/evaluate_test.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/efficientnet_b0/best.pt \
  --output results/efficientnet_b0/test_metrics.json \
  --predictions results/efficientnet_b0/test_predictions.csv
```

`weighted_loss` computes `negative_count / positive_count` from training rows
only, requires both classes, and passes the result as `pos_weight` to
`BCEWithLogitsLoss`. The default Haar detector is fast and reproducible but
can miss faces or produce false positives under pose, occlusion, lighting, and
compression; warnings and failed detections are recorded rather than hidden.
Evaluation reports frame-level accuracy, precision, recall, F1, ROC-AUC,
PR-AUC, and a confusion matrix for the untouched official test CSV. It does
not silently substitute video-level aggregation, and official source-video
splits do not establish human-identity-disjointness.

### Kaggle resumable preprocessing

In a Kaggle notebook, upload the repository and the extracted FaceForensics++
dataset, then run the preparation command from the repository root. The
default progress file is written beside the manifest as
`/kaggle/working/data/processed/faceforensics_c23/progress.json`.

```bash
python scripts/run_kaggle_preprocessing.py \
  --dataset-root /kaggle/input/faceforensics \
  --split-dir /kaggle/input/faceforensics/splits \
  --output-dir /kaggle/working/data/processed/faceforensics_c23 \
  --frames-per-video 10 \
  --chunk-size 100 \
  --kaggle-dataset adarshverma21/faceforensics-c23-processed-checkpoints \
  --privacy-confirmed
```

`/kaggle/input` is read-only and `/kaggle/working` is temporary. The workflow
uploads a complete private checkpoint after every successfully completed chunk
and verifies the Kaggle CLI status before continuing. A failed or interrupted
upload never advances the workflow.

Kaggle CLI 2.0.2 reports dataset status as plain text and does not expose an
authoritative privacy check. After confirming the dataset is private in Kaggle,
pass `--privacy-confirmed` to allow verified uploads and restores; the workflow
fails safely without that explicit confirmation.

If the Kaggle session stops, rerun the same command with `--resume`. Completed
videos with every manifest crop present are skipped, failed videos are retried,
and missing/corrupt progress is rebuilt from the manifest and crop files:

```bash
python scripts/run_kaggle_preprocessing.py \
  --dataset-root /kaggle/input/faceforensics \
  --split-dir /kaggle/input/faceforensics/splits \
  --output-dir /kaggle/working/data/processed/faceforensics_c23 \
  --kaggle-dataset adarshverma21/faceforensics-c23-processed-checkpoints \
  --frames-per-video 10 \
  --chunk-size 100 \
  --privacy-confirmed \
  --resume
```

For exact setup, recovery, privacy, and credential-safety instructions, see
`docs/faceforensics_preprocessing.md`.

The output and progress locations are independent. For example, keep a
progress file in a restored working directory while writing a new output
directory:

```bash
python scripts/prepare_faceforensics.py \
  --dataset-root /kaggle/input/faceforensics \
  --split-dir /kaggle/input/faceforensics/splits \
  --output-dir /kaggle/working/ff-crops \
  --progress-file /kaggle/working/ff-state/progress.json \
  --frames-per-video 10 \
  --batch-size 100 \
  --resume
```

Do not delete `manifest.csv` or `face_crops/` before resuming: progress alone
does not prove that outputs are complete. If a progress file is missing or
corrupt, `--resume` safely rechecks the manifest and crop files. A configuration
mismatch is rejected; rerun with the original preprocessing settings or start
a new output/progress pair. Once all batches finish, the command writes the
final `train.csv`, `val.csv`, `test.csv`, and `statistics.json`.

### Kaggle NVIDIA T4

Upload the repository and prepared manifests to a Kaggle notebook, select a
GPU accelerator, install `requirements.txt`, set `runtime.device: auto` and
`runtime.mixed_precision: true`, and run the same training command. Use an
explicitly reduced `training.batch_size` or `training.image_size` if the T4
cannot fit the configured experiment. CUDA out-of-memory errors are surfaced
without changing settings automatically.

### Reproducibility

The seed controls Python, NumPy, PyTorch, data ordering, and deterministic
CuDNN settings. Keep the final YAML, `history.json`, `best.pt`, `last.pt`, and
training log together for an experiment. Datasets, checkpoints, and generated
results are ignored by Git. No training has been executed in this repository,
so no performance results are claimed.

## Methodology (Planned)

- Build unified training/evaluation pipelines for two architectures
- Run image and frame-based benchmarks under identical settings
- Produce Grad-CAM visual explanations per model
- Measure clean accuracy, robustness under corruptions, and external transfer performance

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Validation commands

Use targeted commands instead of installing full training dependencies in minimal
environments:

```bash
# Lightweight tests (no torch / torchvision required)
python scripts/run_lightweight_tests.py

# PyTorch-dependent tests
python scripts/run_torch_tests.py

# Model smoke test (synthetic batch only, no dataset downloads)
python scripts/smoke_test_models.py --device cpu
```

The smoke test supports CUDA when available:

```bash
python scripts/smoke_test_models.py --device auto
```

For Kaggle validation cells (using Kaggle's built-in PyTorch runtime), see:
`docs/kaggle_model_validation.md`.

## Configuration

Core settings are managed in `configs/config.yaml`, including:

- Image size
- Batch size
- Learning rate
- Epochs
- Random seed
- Model selection
- Dataset paths
- Device selection (`auto`, `cpu`, `cuda`)
- Mixed precision toggle
- Number of workers
- Robustness transformation parameters and deterministic seed

Additional experiment-specific and dataset-specific settings are in:

- `configs/experiments.yaml`
- `configs/dataset.yaml`

## Execution Workflow

1. Populate dataset metadata and paths in `configs/`.
2. Run bootstrap validation:
   ```bash
   python main.py
   ```
3. Implement/extend model, training, evaluation, and explainability modules in `src/`.
4. Add and run tests before experiment execution.

## FaceForensics++ c23 Dataset Preparation

The preparation pipeline does not download FaceForensics++ or require access
credentials. Place the approved archive under a local, ignored directory with
this structure (the category names and `c23` level are significant):

```text
data/raw/faceforensics/
├── original_sequences/youtube/c23/
├── manipulated_sequences/Deepfakes/c23/
├── manipulated_sequences/Face2Face/c23/
├── manipulated_sequences/FaceSwap/c23/
├── manipulated_sequences/NeuralTextures/c23/
└── splits/                 # optional official train.json, val.json, test.json
```

Configure the root, output directory, sampling, crop size, detector, seed, and
categories in `configs/dataset.yaml`, or override them on the command line.
The default detector is OpenCV's bundled Haar cascade; detector changes should
be recorded in the experiment methodology. The generated manifest contains
source-video IDs and preserves source-level split assignments when official
split files are supplied. Identity disjointness is not inferred or claimed.

Run a small CPU-friendly pilot (at most 10 videos and 5 frames per video):

```bash
python scripts/prepare_faceforensics.py \
  --dataset-root data/raw/faceforensics \
  --split-dir data/raw/faceforensics/splits \
  --output-dir data/processed/faceforensics_c23 \
  --pilot
```

After dataset access is approved and the pilot is inspected, omit `--pilot`
and set `--frames-per-video` as required to generate the full manifest. The
command writes `manifest.csv`, split CSVs, `statistics.json`,
`split_statistics.json`, face crops, and a log under the output directory.
Statistics include class, split, manipulation, warning, and failure counts; no
dataset counts are hard-coded or invented.

Kaggle preprocessing checkpoints record the complete preprocessing
configuration and refuse restores that do not match it. To begin a production
run without reusing or overwriting a pilot checkpoint, use a new local output
directory, a distinct private Kaggle dataset, and `--clean-start`:

```bash
python scripts/run_kaggle_preprocessing.py \
  --dataset-root /kaggle/input/faceforensics \
  --output-dir /kaggle/working/faceforensics_c23_production \
  --kaggle-dataset adarshverma21/faceforensics-c23-production-checkpoints \
  --frames-per-video 10 \
  --clean-start \
  --privacy-confirmed
```

`--clean-start` fails if the output directory already exists; it never deletes
the pilot output or its private Kaggle backup.

## GPU Compatibility (Kaggle NVIDIA T4)

- Runtime device selection uses automatic CPU/CUDA detection.
- Mixed precision is enabled only when CUDA is available.
- The same code path remains executable on CPU-only systems.

## Reproducibility Notes

- Global random seeding is applied to Python, NumPy, and PyTorch.
- Deterministic CuDNN settings are configured for repeatability.

## Current Status

This repository currently contains baseline architecture and engineering utilities only. No datasets are downloaded, no training has been executed, and no research results are claimed.
