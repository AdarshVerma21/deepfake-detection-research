# Kaggle notebook cells for model validation

Use these cells in a Kaggle notebook (GPU optional) to validate both models using
Kaggle's preinstalled PyTorch stack. This workflow does not train models or
download datasets.

```python
# Cell 1: setup
%cd /kaggle/working
!rm -rf deepfake-detection-research
!git clone https://github.com/AdarshVerma21/deepfake-detection-research.git
%cd deepfake-detection-research
```

```python
# Cell 2: lightweight tests (no PyTorch model execution)
!pytest -q tests/test_config.py tests/test_faceforensics.py
```

```python
# Cell 3: PyTorch-dependent unit tests
!pytest -q tests/test_device.py tests/test_reproducibility.py tests/test_models_training.py
```

```python
# Cell 4: synthetic model smoke test (CPU default)
!python scripts/smoke_test_models.py --device cpu
```

```python
# Cell 5: optional CUDA smoke test
!python scripts/smoke_test_models.py --device auto
```

Expected behavior:
- EfficientNet-B0 and XceptionNet instantiate without downloading pretrained
  weights.
- Each model runs on a synthetic batch and returns shape `[batch_size, 1]`.
- Outputs are finite and `BCEWithLogitsLoss` is finite.
