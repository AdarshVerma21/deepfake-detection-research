"""Run lightweight tests that do not require PyTorch."""

from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_config.py",
                "tests/test_faceforensics.py",
                "tests/test_manifest.py",
                "tests/test_robustness.py",
                "tests/test_stage6_config.py",
                "tests/test_stage6_evaluation.py",
            ],
            check=False,
        ).returncode
    )
