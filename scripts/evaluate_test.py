"""Explicit CLI for untouched official FaceForensics++ test evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from scripts.evaluate import main

    main()
