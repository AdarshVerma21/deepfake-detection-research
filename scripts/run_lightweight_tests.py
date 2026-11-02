"""Run lightweight tests that do not require PyTorch."""

from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_config.py", "tests/test_faceforensics.py"],
            check=False,
        ).returncode
    )
