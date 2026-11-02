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

## GPU Compatibility (Kaggle NVIDIA T4)

- Runtime device selection uses automatic CPU/CUDA detection.
- Mixed precision is enabled only when CUDA is available.
- The same code path remains executable on CPU-only systems.

## Reproducibility Notes

- Global random seeding is applied to Python, NumPy, and PyTorch.
- Deterministic CuDNN settings are configured for repeatability.

## Current Status

This repository currently contains baseline architecture and engineering utilities only. No datasets are downloaded, no training has been executed, and no research results are claimed.
