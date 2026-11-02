"""Main entrypoint for deepfake detection research experiments."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.device import get_runtime_device
from src.utils.error_handling import run_with_error_context
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import set_seed


REPO_ROOT = Path(__file__).resolve().parent


def bootstrap() -> None:
    """Bootstrap baseline runtime configuration."""
    config = load_yaml_config(REPO_ROOT / "configs" / "config.yaml")

    runtime_cfg = config.get("runtime", {})
    paths_cfg = config.get("paths", {})

    logger = setup_logger(
        name="research",
        log_dir=REPO_ROOT / paths_cfg.get("logs", "outputs/logs"),
    )

    set_seed(int(runtime_cfg.get("seed", 42)))
    runtime = get_runtime_device(
        requested=str(runtime_cfg.get("device", "auto")),
        mixed_precision=bool(runtime_cfg.get("mixed_precision", True)),
    )

    logger.info("Project bootstrap completed.")
    logger.info("Resolved device: %s", runtime.device)
    logger.info("Mixed precision enabled: %s", runtime.use_mixed_precision)


if __name__ == "__main__":
    run_with_error_context(bootstrap, "Bootstrap failed")
