"""Run PyTorch-dependent tests when torch/torchvision are available."""

from __future__ import annotations

import importlib.util
import subprocess
import sys


if __name__ == "__main__":
    if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torchvision"):
        print("[SKIP] torch/torchvision not installed; run this suite in Kaggle or another PyTorch environment.")
        raise SystemExit(0)
    raise SystemExit(
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_device.py",
                "tests/test_reproducibility.py",
                "tests/test_models_training.py",
                "tests/test_gradcam.py",
            ],
            check=False,
        ).returncode
    )
