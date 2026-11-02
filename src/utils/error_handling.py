"""Execution wrappers for consistent exception handling."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from src.utils.errors import DeepfakeResearchError

T = TypeVar("T")


def run_with_error_context(action: Callable[[], T], context_message: str) -> T:
    """Execute an action and enrich unhandled exceptions with context."""
    try:
        return action()
    except DeepfakeResearchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DeepfakeResearchError(f"{context_message}: {exc}") from exc
